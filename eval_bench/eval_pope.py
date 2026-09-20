"""Unified POPE benchmark runner for LLaVA-1.5 and Qwen2-VL with ONLY intervention.

Designed for Kaggle Notebooks (2x NVIDIA T4 GPUs, BF16 Full Precision) and local execution.
Supports:
  - --model llava | qwen2vl
  - --split all | random | popular | adversarial
  - Auto-detection of COCO val2014 images and POPE annotations
  - True text generation with greedy decoding (max_new_tokens=6)
  - Standard prompt suffix for QwenVL (" Please answer with yes or no.")
  - Device map "auto" across multiple GPUs without CUDA device mismatch
  - Standardized output structure and summary metrics DataFrame
"""
import argparse
import datetime
import json
import os
import sys

# Silence TensorFlow / oneDNN logs
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LogitsProcessorList,
    Qwen2VLForConditionalGeneration,
)

# Ensure workspace roots are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_common import binary_metrics, pope_parse_extended, read_jsonl, append_jsonl, write_json  # noqa: E402
from pope_auto_detect import find_coco_images, find_or_download_pope_annotations  # noqa: E402
from only_utils.only_llava import OnlyLlava, OnlyLlavaLogitsProcessor  # noqa: E402
from only_utils.only_qwen2vl import OnlyQwen2VL, OnlyLogitsProcessor  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_args():
    p = argparse.ArgumentParser(description="POPE Benchmark Runner (ONLY / Baseline)")
    # Model
    p.add_argument("--model", choices=["llava", "qwen2vl"], required=True,
                   help="Model architecture: 'llava' (LLaVA-1.5-7B) or 'qwen2vl' (Qwen2-VL-7B-Instruct)")
    p.add_argument("--model_path", default=None,
                   help="HuggingFace checkpoint name or local path (defaults to official HF repos)")
    p.add_argument("--device_map", default="auto", help="device_map for accelerate (default: 'auto')")
    p.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Benchmark protocol
    p.add_argument("--split", default="all", choices=["all", "random", "popular", "adversarial"],
                   help="POPE split to evaluate ('all' runs random, popular, and adversarial sequentially)")
    p.add_argument("--max_new_tokens", type=int, default=6,
                   help="Standard POPE maximum new tokens (default: 6)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")

    # Data paths (auto-detected if None)
    p.add_argument("--data_path", default=None, help="Path to COCO val2014 images directory")
    p.add_argument("--pope_dir", default=None, help="Directory containing POPE annotations")
    p.add_argument("--out_path", default="./results", help="Base output directory for results")

    # ONLY Intervention
    p.add_argument("--use_only", type=str2bool, default=True, help="Enable ONLY intervention method")
    p.add_argument("--enhance_layer_index", type=int, default=0, help="Layer index for ONLY attention intervention")
    p.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    p.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    p.add_argument("--ritual_beta", type=float, default=0.1)
    p.add_argument("--js_gamma", type=float, default=0.2)

    return p.parse_args()


def resolve_eos_token_id(model, processor):
    """Safely retrieve eos_token_id without triggering 'LlavaConfig has no attribute eos_token_id'."""
    eos = getattr(model, "generation_config", None)
    eos = getattr(eos, "eos_token_id", None) if eos else None
    if eos is None and hasattr(model, "config"):
        eos = getattr(model.config, "eos_token_id", None)
    if eos is None and hasattr(model.config, "text_config"):
        eos = getattr(model.config.text_config, "eos_token_id", None)
    if eos is None and hasattr(processor, "tokenizer"):
        eos = getattr(processor.tokenizer, "eos_token_id", None)
    return eos


class PopeEvaluator:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)

        # Default model checkpoints
        if args.model_path is None:
            if args.model == "llava":
                args.model_path = "llava-hf/llava-1.5-7b-hf"
            else:
                args.model_path = "Qwen/Qwen2-VL-7B-Instruct"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[args.precision]

        print(f"\n=======================================================")
        print(f"Loading {args.model.upper()} from {args.model_path}")
        print(f"Device Map: {args.device_map} | Precision: {args.precision}")
        print(f"ONLY Intervention: {args.use_only}")
        print(f"=======================================================\n")

        self.processor = AutoProcessor.from_pretrained(args.model_path)

        if args.model == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
            ).eval()
            self.only = OnlyLlava(self.model, args.enhance_layer_index) if args.use_only else None
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
                attn_implementation="sdpa",
            ).eval()
            self.only = OnlyQwen2VL(self.model, args.enhance_layer_index) if args.use_only else None

        self.eos_token_id = resolve_eos_token_id(self.model, self.processor)
        print(f"[Model Loaded] Target Device: {self.model.device} | Safe EOS: {self.eos_token_id}")

    @torch.inference_mode()
    def generate(self, image, question):
        args = self.args
        model_device = self.model.device

        if args.model == "qwen2vl":
            # Mandatory prompt suffix for QwenVL
            prompt_text = f"{question} Please answer with yes or no."
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            processors = None
            if self.only is not None:
                self.only.set_image_span(inputs["input_ids"])
                processors = LogitsProcessorList([
                    OnlyLogitsProcessor(
                        self.only,
                        alpha_pos=args.ritual_alpha_pos,
                        alpha_neg=args.ritual_alpha_neg,
                        beta=args.ritual_beta,
                        gamma=args.js_gamma,
                    )
                ])

            out = self.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                logits_processor=processors,
            )
            gen_tokens = out[0, inputs["input_ids"].shape[1]:].tolist()
            text_out = self.processor.batch_decode([gen_tokens], skip_special_tokens=True)[0].strip()

        else:
            # LLaVA 1.5 prompt protocol (verbatim question)
            prompt = (
                f"A chat between a curious human and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the human's questions. "
                f"USER: <image>\n{question} ASSISTANT:"
            )
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            processors = None
            if self.only is not None:
                self.only.set_image_span(inputs["input_ids"])
                processors = LogitsProcessorList([
                    OnlyLlavaLogitsProcessor(
                        self.only,
                        alpha_pos=args.ritual_alpha_pos,
                        alpha_neg=args.ritual_alpha_neg,
                        beta=args.ritual_beta,
                        gamma=args.js_gamma,
                    )
                ])

            out = self.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                logits_processor=processors,
            )
            gen_tokens = out[0, inputs["input_ids"].shape[1]:].tolist()
            text_out = self.processor.batch_decode([gen_tokens], skip_special_tokens=True)[0].strip()

        return text_out


def run_split(evaluator, split, data_path, pope_file, run_dir):
    args = evaluator.args
    split_dir = os.path.join(run_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    raw_file = os.path.join(split_dir, "raw_outputs.jsonl")
    config_file = os.path.join(split_dir, "run_config.json")
    metrics_file = os.path.join(split_dir, "metrics.json")

    # Save run_config.json
    run_cfg = {
        "model": args.model,
        "model_path": args.model_path,
        "split": split,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "seed": args.seed,
        "precision": args.precision,
        "device_map": args.device_map,
        "use_only": args.use_only,
        "hyperparams": {
            "enhance_layer_index": args.enhance_layer_index,
            "ritual_alpha_pos": args.ritual_alpha_pos,
            "ritual_alpha_neg": args.ritual_alpha_neg,
            "ritual_beta": args.ritual_beta,
            "js_gamma": args.js_gamma,
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }
    write_json(config_file, run_cfg)

    # Load annotations
    with open(pope_file, "r") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    if args.limit:
        items = items[:args.limit]

    # Check already finished keys for resumable evaluation
    existing = read_jsonl(raw_file)
    finished_qids = {r["question_id"] for r in existing}

    print(f"\n--- Evaluating Split: {split.upper()} ({len(items)} questions, {len(finished_qids)} already done) ---")

    for i, item in enumerate(tqdm(items, desc=f"POPE {split}")):
        qid = item.get("question_id", i)
        if qid in finished_qids:
            continue

        image_fn = item["image"]
        image_path = os.path.join(data_path, image_fn)
        question = item["text"]
        label = item["label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        output_text = evaluator.generate(image, question)
        pred = pope_parse_extended(output_text)

        record = {
            "question_id": qid,
            "image": image_fn,
            "question": question,
            "label": label,
            "pred": pred,
            "answer": output_text,
            "text": output_text,
        }
        append_jsonl(raw_file, record)

    # Compute metrics
    all_records = read_jsonl(raw_file)
    preds = [r["pred"] for r in all_records]
    labels = [r["label"] for r in all_records]
    metrics = binary_metrics(preds, labels)

    write_json(metrics_file, metrics)
    return metrics


def main():
    args = parse_args()

    # Auto-detect COCO image path
    data_path = find_coco_images(args.data_path)

    # Determine splits to run
    if args.split == "all":
        splits = ["random", "popular", "adversarial"]
    else:
        splits = [args.split]

    # Create run output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "only" if args.use_only else "baseline"
    run_dir = os.path.join(args.out_path, f"{args.model}_{method_name}_pope_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Results Directory] {os.path.abspath(run_dir)}")

    # Initialize evaluator (loads model ONCE)
    evaluator = PopeEvaluator(args)

    summary_rows = []
    for split in splits:
        pope_file = find_or_download_pope_annotations(
            pope_path=os.path.join(args.pope_dir, f"coco_pope_{split}.json") if args.pope_dir else None,
            split=split,
        )
        metrics = run_split(evaluator, split, data_path, pope_file, run_dir)
        summary_rows.append({
            "Split": split,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "YesRatio": metrics["YesRatio"],
            "Unknowns": metrics["Unknowns"],
            "Total": metrics["Total"],
        })

    # Summary DataFrame
    df = pd.DataFrame(summary_rows)
    if len(df) > 1:
        avg_row = {
            "Split": "Average",
            "Accuracy": df["Accuracy"].mean(),
            "Precision": df["Precision"].mean(),
            "Recall": df["Recall"].mean(),
            "F1": df["F1"].mean(),
            "YesRatio": df["YesRatio"].mean(),
            "Unknowns": df["Unknowns"].sum(),
            "Total": df["Total"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    summary_csv = os.path.join(run_dir, "summary_metrics.csv")
    df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print(" " * 26 + "POPE BENCHMARK SUMMARY")
    print("=" * 80)
    pd.set_option("display.precision", 2)
    pd.set_option("display.width", 1000)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Summary metrics saved to: {summary_csv}\n")


if __name__ == "__main__":
    main()
