"""Unified CHAIR benchmark runner for LLaVA-1.5 and Qwen2-VL with ONLY intervention.

Designed for Kaggle Notebooks (2x NVIDIA T4 GPUs, BF16/FP16) and local execution.
Supports:
  - Models: 'llava' (LLaVA-1.5-7B) and 'qwen2vl' (Qwen2-VL-7B-Instruct)
  - Method: ONLY hallucination mitigation (--use_only True/False)
  - Prompt: 'Describe this image.' with max_new_tokens=128 (greedy decoding)
  - Dataset: 500 COCO val2014 images (seed 2027) via selected_chair_val2014_seed2027.json
  - Timing: Microsecond-precision GPU latency measurement with import time (time.perf_counter & torch.cuda.synchronize)
  - Latency testing mode: --max_samples 10 for quick latency benchmarks
  - CHAIR evaluation: Maxlinn/CHAIR-metric-standalone (CHAIRs, CHAIRi, Recall, Caption Length)
"""
import argparse
import datetime
import json
import os
import pickle
import sys
import time
from typing import Optional

# Silence TF logs
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

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

# Setup search paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from only_utils.only_llava import OnlyLlava, OnlyLlavaLogitsProcessor  # noqa: E402
from only_utils.only_qwen2vl import OnlyQwen2VL, OnlyLogitsProcessor  # noqa: E402
from chair import CHAIR, print_metrics, save_hallucinated_words  # noqa: E402


class _CHAIRUnpickler(pickle.Unpickler):
    """Robust Unpickler mapping '__main__.CHAIR' or 'chair.CHAIR' to loaded CHAIR class."""
    def find_class(self, module, name):
        if name == "CHAIR":
            return CHAIR
        return super().find_class(module, name)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_args():
    p = argparse.ArgumentParser(description="CHAIR Benchmark Runner for ONLY (ICCV'25)")
    p.add_argument("--model", choices=["llava", "qwen2vl"], default="llava",
                   help="Model to benchmark: 'llava' (LLaVA-1.5-7B) or 'qwen2vl' (Qwen2-VL-7B-Instruct)")
    p.add_argument("--model_path", default=None,
                   help="HuggingFace checkpoint or local path")
    p.add_argument("--device_map", default="auto",
                   help="Device map strategy (default: 'auto')")
    p.add_argument("--precision", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                   help="Weight precision (default: 'auto')")

    # Benchmark protocol
    p.add_argument("--max_samples", type=int, default=10,
                   help="Number of images to evaluate (default 10 for latency testing; set 500 for full CHAIR)")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Alias for --max_samples")
    p.add_argument("--prompt", type=str, default="Describe this image.",
                   help="Prompt for caption generation (default: 'Describe this image.')")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max new tokens to generate (standard: 128 for CHAIR)")
    p.add_argument("--seed", type=int, default=2027,
                   help="Random seed for sampling (default: 2027)")

    # Data paths
    p.add_argument("--coco_dir", default=None,
                   help="Path to COCO val2014 images directory")
    p.add_argument("--manifest_path", default=None,
                   help="Path to pre-selected CHAIR 500 samples JSON (default: selected_chair_val2014_seed2027.json)")
    p.add_argument("--chair_cache", default=None,
                   help="Path to chair.pkl cache file")
    p.add_argument("--out_dir", default=None,
                   help="Custom output directory to store benchmark results")

    # ONLY Intervention
    p.add_argument("--use_only", type=str2bool, default=True,
                   help="Enable ONLY intervention method (True) or run baseline (False)")
    p.add_argument("--enhance_layer_index", type=int, default=0,
                   help="Layer index for ONLY attention intervention (default: 0)")
    p.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    p.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    p.add_argument("--ritual_beta", type=float, default=0.1)
    p.add_argument("--js_gamma", type=float, default=0.25)

    args = p.parse_args()
    if args.num_samples is not None:
        args.max_samples = args.num_samples
    return args


def auto_detect_coco_dir(override_path: Optional[str] = None) -> str:
    """Auto-detect COCO val2014 images directory across Kaggle, Colab, and local environments."""
    if override_path and os.path.isdir(override_path):
        print(f"[Data Detection] Using user-specified COCO image dir: {override_path}")
        return os.path.abspath(override_path)

    known_paths = [
        "/kaggle/input/datasets/biminhco/val2014/val2014",
        "/kaggle/input/datasets/biminhco/val2014",
        "/kaggle/input/val2014/val2014",
        "/kaggle/input/val2014",
        "/kaggle/input/coco-2014-val/val2014",
        "/kaggle/input/coco-val2014/val2014",
        "/kaggle/input/coco2014/val2014",
        "/content/val2014",
        os.path.join(PROJECT_ROOT, "val2014"),
        os.path.join(PROJECT_ROOT, "dataset", "val2014"),
        os.path.join(PROJECT_ROOT, "..", "val2014"),
        os.path.join(PROJECT_ROOT, "..", "OPERA", "opera_experiments", "dataset", "val2014"),
        os.path.join(PROJECT_ROOT, "..", "VCD", "vcd_experiments", "dataset", "val2014"),
    ]
    for p in known_paths:
        if os.path.isdir(p):
            # Check if there are COCO images
            files = [f for f in os.listdir(p)[:50] if f.startswith("COCO_val2014_") or f.endswith(".jpg")]
            if files:
                print(f"[Data Detection] Found COCO images at: {p}")
                return os.path.abspath(p)

    # Deep scan /kaggle/input/ if present
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        print("[Data Detection] Scanning /kaggle/input for COCO val2014 images...")
        for root, _, files in os.walk(kaggle_input):
            if any(f.startswith("COCO_val2014_") for f in files[:50]):
                print(f"[Data Detection] Auto-detected COCO images at: {root}")
                return os.path.abspath(root)

    raise FileNotFoundError(
        "COCO val2014 image directory not found! Please mount dataset or pass --coco_dir explicitly."
    )


def resolve_chair_cache(cache_path: Optional[str] = None) -> str:
    """Resolve location of chair.pkl cache or download from Maxlinn repo if missing."""
    candidates = []
    if cache_path:
        candidates.append(cache_path)
    candidates.extend([
        os.path.join(SCRIPT_DIR, "chair.pkl"),
        os.path.join(PROJECT_ROOT, "chair.pkl"),
        os.path.join(PROJECT_ROOT, "eval_bench", "chair.pkl"),
        "/kaggle/working/ONLY/chair.pkl",
        "/kaggle/working/ONLY/eval_bench/chair.pkl",
        "/kaggle/input/chair/chair.pkl",
        "/content/ONLY/chair.pkl",
        os.path.join(PROJECT_ROOT, "..", "OPERA", "opera_experiments", "benchmarks", "chair", "chair.pkl"),
        os.path.join(PROJECT_ROOT, "..", "VCD", "vcd_experiments", "benchmarks", "chair", "chair.pkl"),
    ])
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    # Auto-download from Maxlinn's CHAIR-metric-standalone GitHub repository
    target_cache = os.path.join(SCRIPT_DIR, "chair.pkl")
    url = "https://raw.githubusercontent.com/Maxlinn/CHAIR-metric-standalone/main/chair.pkl"
    print(f"[CHAIR Setup] chair.pkl not found locally. Auto-downloading from {url}...")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, target_cache)
        print(f"[CHAIR Setup] Successfully downloaded chair.pkl to {target_cache}")
        return os.path.abspath(target_cache)
    except Exception as e:
        print(f"[CHAIR Warning] Failed to auto-download chair.pkl: {e}")

    return None


def resolve_chair_samples(image_dir: str, num_samples: int = 10, seed: int = 2027, manifest_path: Optional[str] = None) -> list:
    """Load samples for CHAIR benchmark (seed 2027).

    Priority:
      1. Pre-computed manifest (selected_chair_val2014_seed2027.json)
      2. Deterministic sampling with fixed seed 2027
    """
    candidates = []
    if manifest_path and os.path.isfile(manifest_path):
        candidates.append(manifest_path)
    candidates.extend([
        os.path.join(SCRIPT_DIR, f"selected_chair_val2014_seed{seed}.json"),
        os.path.join(PROJECT_ROOT, f"selected_chair_val2014_seed{seed}.json"),
        os.path.join(PROJECT_ROOT, "..", "VCD", "vcd_experiments", "benchmarks", "chair", f"selected_chair_val2014_seed{seed}.json"),
        os.path.join(PROJECT_ROOT, "..", "OPERA", "opera_experiments", "benchmarks", "chair", f"selected_chair_val2014_seed{seed}.json"),
    ])

    for m in candidates:
        if os.path.isfile(m):
            try:
                with open(m, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data.get("samples", data)
                if isinstance(samples, list) and len(samples) > 0:
                    selected = samples if (num_samples <= 0 or num_samples >= len(samples)) else samples[:num_samples]
                    print(f"[Sampling] Using verified manifest ({len(selected)}/{len(samples)} images): {m}")
                    return selected
            except Exception as e:
                print(f"[Sampling Warning] Failed loading manifest {m}: {e}")

    # Fallback: Deterministic sampling from directory
    print(f"[Sampling] Sampling {num_samples} images from {image_dir} with seed {seed}...")
    import random
    all_files = sorted([f for f in os.listdir(image_dir) if f.startswith("COCO_val2014_") and f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not all_files:
        all_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    chosen = random.Random(seed).sample(all_files, min(num_samples if num_samples > 0 else 500, len(all_files)))
    results = []
    for f in chosen:
        base = os.path.basename(f).split(".")[0]
        img_id = int(base.split("_")[-1]) if "_" in base else int(base)
        results.append({"image_id": img_id, "file_name": f})
    return results


def find_image_file(image_dir: str, file_name: str) -> str:
    """Find image file directly or in standard subdirectories."""
    direct = os.path.join(image_dir, file_name)
    if os.path.isfile(direct):
        return direct
    sub = os.path.join(image_dir, "val2014", file_name)
    if os.path.isfile(sub):
        return sub
    return direct


def print_latency_summary(latencies: list, total_time: float, avg_time: float, num_samples: int):
    """Print clean aesthetic summary of timing latency and throughput."""
    sep = "=" * 80
    dash = "-" * 80
    throughput = len(latencies) / total_time if total_time > 0 else 0.0
    est_500 = (avg_time * 500) / 60.0

    print("\n" + sep)
    print("⏱️  GPU INFERENCE LATENCY & TIMING BENCHMARK (import time: time.perf_counter)")
    print(sep)
    print(f"  Samples Tested        : {num_samples}")
    print(f"  Total Inference Time  : {total_time:.4f} s")
    print(f"  Average Time / Sample : {avg_time:.4f} s")
    print(f"  Throughput            : {throughput:.2f} samples/sec ({1.0/avg_time:.2f} s/sample)" if avg_time > 0 else "N/A")
    print(f"  Estimated Full 500 Img: {est_500:.2f} minutes ({avg_time * 500:.1f} s)")
    print(dash)
    if len(latencies) <= 15:
        sample_str = ", ".join([f"{t:.3f}s" for t in latencies])
        print(f"  Per-sample latencies  : [{sample_str}]")
    print(sep + "\n")


def print_chair_summary_table(metrics: dict, model_name: str, mode: str, num_samples: int):
    """Print formatted summary table of CHAIR results."""
    sep = "=" * 88
    dash = "-" * 88

    chairs = f"{metrics.get('CHAIRs', 0.0) * 100:.2f}%" if metrics.get('CHAIRs', 0.0) <= 1.0 else f"{metrics.get('CHAIRs', 0.0):.2f}%"
    chairi = f"{metrics.get('CHAIRi', 0.0) * 100:.2f}%" if metrics.get('CHAIRi', 0.0) <= 1.0 else f"{metrics.get('CHAIRi', 0.0):.2f}%"
    recall = f"{metrics.get('Recall', 0.0) * 100:.2f}%" if metrics.get('Recall', 0.0) <= 1.0 else f"{metrics.get('Recall', 0.0):.2f}%"
    cap_len = f"{metrics.get('Len', metrics.get('Caption_Length', 0.0)):.2f}"
    avg_t = metrics.get("avg_time_per_sample_s", 0.0)
    tot_t = metrics.get("total_inference_time_s", 0.0)

    print("\n" + sep)
    print(f"🎯  CHAIR BENCHMARK RESULTS | Model: {model_name.upper()} | Method: {mode.upper()} | Samples: {num_samples}")
    print(sep)
    header = (
        f"{'Model':<10} | {'Method':<10} | {'CHAIRs':>9} | {'CHAIRi':>9} | {'Recall':>9} | "
        f"{'Cap Len':>8} | {'Avg Latency':>14} | {'Total Time':>12}"
    )
    print(header)
    print(dash)
    row = (
        f"{model_name.upper():<10} | {mode.upper():<10} | {chairs:>9} | {chairi:>9} | {recall:>9} | "
        f"{cap_len:>8} | {f'{avg_t:.4f} s':>14} | {f'{tot_t:.2f} s':>12}"
    )
    print(row)
    print(sep + "\n")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Detect COCO images & CHAIR cache
    coco_image_dir = auto_detect_coco_dir(args.coco_dir)
    chair_cache_file = resolve_chair_cache(args.chair_cache)

    # 2. Load samples (subsampled to --max_samples, default 10 for latency test, 500 for full CHAIR)
    samples = resolve_chair_samples(
        image_dir=coco_image_dir,
        num_samples=args.max_samples,
        seed=args.seed,
        manifest_path=args.manifest_path
    )

    # 3. Model Checkpoint & Precision
    if args.model_path is None:
        if args.model == "llava":
            args.model_path = "llava-hf/llava-1.5-7b-hf"
        else:
            args.model_path = "Qwen/Qwen2-VL-7B-Instruct"

    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            prec_desc = "bfloat16 (Hardware Native)"
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
            prec_desc = "float16 (Tensor Cores)"
        else:
            torch_dtype = torch.float32
            prec_desc = "float32 (CPU)"
    else:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[args.precision]
        prec_desc = args.precision

    print(f"\n=======================================================")
    print(f"  CHAIR Benchmark Runner for ONLY (ICCV'25)")
    print(f"  Model: {args.model.upper()} ({args.model_path})")
    print(f"  Method: {'ONLY' if args.use_only else 'BASELINE'}")
    print(f"  Prompt: '{args.prompt}' | max_new_tokens: {args.max_new_tokens}")
    print(f"  Samples to evaluate: {len(samples)} (seed: {args.seed})")
    print(f"  Device Map: {args.device_map} | Precision: {prec_desc}")
    print(f"  COCO Image Dir: {coco_image_dir}")
    print(f"  CHAIR Evaluator Cache: {chair_cache_file}")
    print(f"=======================================================\n")

    # 4. Output directory setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "only" if args.use_only else "baseline"
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(PROJECT_ROOT, "results", f"{args.model}_chair_{mode}_s{len(samples)}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    raw_outputs_file = os.path.join(output_dir, "raw_outputs.jsonl")
    metrics_file = os.path.join(output_dir, "metrics.json")
    summary_file = os.path.join(output_dir, "summary_metrics.json")

    # 5. Initialize Processor & Model
    print(f"[Model Loading] Loading {args.model.upper()} from HuggingFace Hub...", flush=True)
    if args.model == "qwen2vl":
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=313600,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            attn_implementation="sdpa",
        ).eval()
        only = OnlyQwen2VL(model, args.enhance_layer_index) if args.use_only else None
    else:
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
        ).eval()
        only = OnlyLlava(model, args.enhance_layer_index) if args.use_only else None

    print(f"[Model Loaded] Target Device: {model.device} | Ready for Inference\n")

    # 6. Save Run Config
    run_config = {
        "benchmark": "chair",
        "model": args.model,
        "model_path": args.model_path,
        "use_only": args.use_only,
        "num_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "seed": args.seed,
        "precision": str(torch_dtype),
        "device_map": args.device_map,
        "coco_image_dir": coco_image_dir,
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # 7. Generate Captions with Precise Latency Measurement (import time)
    sample_latencies = []
    results = []

    print(f"---> Starting CHAIR Caption Generation ({len(samples)} samples) | Model: {args.model.upper()} | Method: {mode.upper()}")
    print(f"     Prompt: '{args.prompt}' | max_new_tokens: {args.max_new_tokens} | Greedy Decoding\n")

    with open(raw_outputs_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(samples, desc=f"CHAIR [{args.model.upper()} - {mode.upper()}]")
        for idx, item in enumerate(pbar):
            img_id = item["image_id"]
            file_name = item["file_name"]
            img_path = find_image_file(coco_image_dir, file_name)

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                continue

            # Build multimodal prompt
            if args.model == "qwen2vl":
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                logits_processors = None
                if only is not None:
                    only.set_image_span(inputs["input_ids"])
                    logits_processors = LogitsProcessorList([
                        OnlyLogitsProcessor(
                            only,
                            alpha_pos=args.ritual_alpha_pos,
                            alpha_neg=args.ritual_alpha_neg,
                            beta=args.ritual_beta,
                            gamma=args.js_gamma,
                        )
                    ])
            else:
                prompt_text = (
                    f"A chat between a curious human and an artificial intelligence assistant. "
                    f"The assistant gives helpful, detailed, and polite answers to the human's questions. "
                    f"USER: <image>\n{args.prompt} ASSISTANT:"
                )
                inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                logits_processors = None
                if only is not None:
                    only.set_image_span(inputs["input_ids"])
                    logits_processors = LogitsProcessorList([
                        OnlyLlavaLogitsProcessor(
                            only,
                            alpha_pos=args.ritual_alpha_pos,
                            alpha_neg=args.ritual_alpha_neg,
                            beta=args.ritual_beta,
                            gamma=args.js_gamma,
                        )
                    ])

            # =========================================================================
            # PRECISE GPU INFERENCE TIMING (import time: time.perf_counter)
            # =========================================================================
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    logits_processor=logits_processors,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            latency = t_end - t_start
            sample_latencies.append(latency)

            gen_tokens = out[0, inputs["input_ids"].shape[1]:].tolist()
            caption = processor.batch_decode([gen_tokens], skip_special_tokens=True)[0].strip()

            record = {
                "image_id": img_id,
                "file_name": file_name,
                "prompt": args.prompt,
                "caption": caption,
                "latency_s": round(latency, 4)
            }
            results.append(record)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            # Update progress bar with moving average latency
            avg_so_far = sum(sample_latencies) / len(sample_latencies)
            pbar.set_postfix({"avg_time": f"{avg_so_far:.3f}s", "last": f"{latency:.3f}s"})

    total_time = sum(sample_latencies)
    avg_time = (total_time / len(sample_latencies)) if sample_latencies else 0.0

    # Print inference latency table
    print_latency_summary(sample_latencies, total_time, avg_time, len(sample_latencies))

    # 8. Evaluate CHAIR Metrics via Maxlinn/CHAIR-metric-standalone
    print(f"[CHAIR Evaluation] Computing CHAIR metrics using Maxlinn/CHAIR-metric-standalone...")
    # Pre-download required NLTK tokenizers and wordnet
    import nltk
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    metrics = {}
    if chair_cache_file and os.path.isfile(chair_cache_file):
        try:
            with open(chair_cache_file, "rb") as f:
                evaluator = _CHAIRUnpickler(f).load()

            cap_dict = evaluator.compute_chair(raw_outputs_file, image_id_key="image_id", caption_key="caption")
            metrics = cap_dict.get("overall_metrics", {})
            save_hallucinated_words(os.path.join(output_dir, "chair_details.json"), cap_dict)
            print("[CHAIR Evaluation] Successfully computed metrics from chair.pkl!")
        except Exception as e:
            print(f"[CHAIR Evaluation Notice] Evaluator error: {e}")
            metrics = {"error": str(e)}
    else:
        print("[CHAIR Evaluation Warning] chair.pkl not found. Skipping metric calculation; raw captions are saved.")

    # Attach timing metrics
    metrics["avg_time_per_sample_s"] = round(avg_time, 4)
    metrics["total_inference_time_s"] = round(total_time, 4)
    metrics["num_evaluated"] = len(sample_latencies)
    metrics["samples_tested"] = len(samples)

    # Save final metrics
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    summary_data = {
        "model": args.model,
        "mode": mode,
        "metrics": metrics,
        "timing": {
            "total_samples": len(sample_latencies),
            "total_inference_time_s": round(total_time, 4),
            "avg_time_per_sample_s": round(avg_time, 4),
            "throughput_samples_per_sec": round(len(sample_latencies) / total_time, 2) if total_time > 0 else 0.0
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Print summary table
    print_chair_summary_table(metrics, model_name=args.model, mode=mode, num_samples=len(sample_latencies))
    print(f"All CHAIR evaluations completed! Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
