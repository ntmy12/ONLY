"""POPE / BEAF / CHAIR generation for Qwen2-VL-7B-Instruct (greedy, optionally +ONLY).

Environment: requirements_qwen2vl.txt (transformers 4.56.x) -- NOT the LLaVA env (transformers 4.31 fork).

  python eval_bench/qwen2vl_eval.py --benchmark pope  --pope_path .../coco_pope_random.json --data_path .../val2014 --use_only True
  python eval_bench/qwen2vl_eval.py --benchmark beaf  --qna_path .../beaf_qna.json --image_root .../beaf --beaf_metric .../beaf_metric.py
  python eval_bench/qwen2vl_eval.py --benchmark chair --data_path .../val2014 --chair_seed <SEED>
"""
import argparse
import os
import subprocess
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LogitsProcessorList, Qwen2VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from only_utils.only_qwen2vl import OnlyLogitsProcessor, OnlyQwen2VL  # noqa: E402
from eval_common import (CHAIR_MAX_NEW_TOKENS, CHAIR_NUM_IMAGES, CHAIR_PROMPT, YESNO_MAX_NEW_TOKENS,  # noqa: E402
                         append_jsonl, beaf_image_index, beaf_official_answer, binary_metrics, coco_image_id,
                         done_keys, load_beaf, load_pope, pope_parse, read_jsonl, select_chair_images, write_json)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["pope", "beaf", "chair"], required=True)
    p.add_argument("--model_path", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--out_path", default="./results")
    p.add_argument("--seed", type=int, default=42)
    # data
    p.add_argument("--pope_path", default=None)
    p.add_argument("--type", default=None, help="pope split name (random|popular|adversarial); inferred from path")
    p.add_argument("--data_path", default=None, help="COCO val2014 images (POPE/CHAIR)")
    p.add_argument("--qna_path", default=None)
    p.add_argument("--image_root", default=None)
    p.add_argument("--beaf_metric", default=None)
    p.add_argument("--chair_seed", type=int, default=None)
    p.add_argument("--image_list", default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    # ONLY
    p.add_argument("--use_only", type=str2bool, default=False)
    p.add_argument("--enhance_layer_index", type=int, default=0)
    p.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    p.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    p.add_argument("--ritual_beta", type=float, default=0.1)
    p.add_argument("--js_gamma", type=float, default=None, help="default: 0.2 (POPE/BEAF), 0.25 (CHAIR)")
    return p.parse_args()


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.only = OnlyQwen2VL(self.model, args.enhance_layer_index) if args.use_only else None

    @torch.inference_mode()
    def generate(self, image, text, max_new_tokens):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        processors = None
        if self.only is not None:
            self.only.set_image_span(inputs["input_ids"])
            a = self.args
            processors = LogitsProcessorList([OnlyLogitsProcessor(
                self.only, a.ritual_alpha_pos, a.ritual_alpha_neg, a.ritual_beta, a.js_gamma)])
        # plain greedy; override the checkpoint's generation_config (temperature/top_p/top_k/repetition_penalty)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                  top_p=None, top_k=None, repetition_penalty=1.0, logits_processor=processors)
        gen = out[0, inputs["input_ids"].shape[1]:].tolist()
        num_tokens = len(gen) - (1 if gen and gen[-1] in self.eos_ids else 0)
        text_out = self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
        return text_out, num_tokens

    def run_dir(self, sub):
        a = self.args
        method = "ONLY" if a.use_only else "Regular"
        tag = f"greedy_a{a.ritual_alpha_pos}_{a.ritual_alpha_neg}_b{a.ritual_beta}_g{a.js_gamma}_L{a.enhance_layer_index}_T{a.max_new_tokens}"
        d = os.path.join(a.out_path, a.benchmark, "qwen2-vl-7b-instruct", method, sub)
        os.makedirs(d, exist_ok=True)
        return d, tag


def run_pope(r):
    a = r.args
    split = a.type or next(s for s in ("random", "popular", "adversarial") if s in os.path.basename(a.pope_path))
    d, tag = r.run_dir(f"coco_{split}")
    pred_file, metric_file = f"{d}/{tag}_predictions.jsonl", f"{d}/{tag}_metrics.json"
    items = load_pope(a.pope_path)
    finished = done_keys(pred_file, "question_id")
    for i, q in enumerate(tqdm(items, desc=f"POPE {split}")):
        qid = q.get("question_id", i)
        if qid in finished:
            continue
        image = Image.open(os.path.join(a.data_path, q["image"])).convert("RGB")
        out, _ = r.generate(image, q["text"], a.max_new_tokens)
        append_jsonl(pred_file, {"question_id": qid, "image": q["image"], "question": q["text"],
                                 "label": q["label"], "output": out, "pred": pope_parse(out)})
    recs = read_jsonl(pred_file)
    assert len(recs) == len(items)
    m = binary_metrics([x["pred"] for x in recs], [x["label"] for x in recs])
    m["args"] = vars(a)
    write_json(metric_file, m)
    print(f"[POPE {split}] Acc {m['Accuracy']:.2f} Prec {m['Precision']:.2f} Rec {m['Recall']:.2f} F1 {m['F1']:.2f}")


def run_beaf(r):
    a = r.args
    d, tag = r.run_dir("")
    pred_file, answer_file, metric_file = f"{d}/{tag}_predictions.jsonl", f"{d}/{tag}_answers.json", f"{d}/{tag}_metrics.txt"
    qna = load_beaf(a.qna_path)
    index = beaf_image_index(a.image_root)
    missing = sorted({q["image"] for q in qna} - set(index))
    assert not missing, f"{len(missing)} BEAF images missing, e.g. {missing[:3]}"
    finished = done_keys(pred_file, "id")
    for q in tqdm(qna, desc="BEAF"):
        if q["id"] in finished:
            continue
        out, _ = r.generate(Image.open(index[q["image"]]).convert("RGB"), q["question"], a.max_new_tokens)
        append_jsonl(pred_file, {"id": q["id"], "question": q["question"], "answer": out})
    preds = {x["id"]: x["answer"] for x in read_jsonl(pred_file)}
    assert len(preds) == len(qna)
    answers = [{"id": q["id"], "answer": preds[q["id"]]} for q in qna]
    write_json(answer_file, answers)
    unparsable = sum(beaf_official_answer(x["answer"]) is None for x in answers)
    res = subprocess.run([sys.executable, a.beaf_metric, "--qna-path", a.qna_path, "--model-answers", answer_file],
                         capture_output=True, text=True)
    report = res.stdout + res.stderr + f"\nunparsable_answers: {unparsable}\n"
    print(report)
    with open(metric_file, "w") as f:
        f.write(report)


def run_chair(r):
    a = r.args
    assert a.chair_seed is not None, "--chair_seed is required"
    d, tag = r.run_dir("")
    cap_file = f"{d}/{tag}_seed{a.chair_seed}.jsonl"
    image_list = a.image_list or os.path.join(a.out_path, "chair", f"chair_images_seed{a.chair_seed}.json")
    files = select_chair_images(a.data_path, a.chair_seed, CHAIR_NUM_IMAGES, image_list)
    finished = done_keys(cap_file, "image_id")
    for fn in tqdm(files, desc="CHAIR"):
        img_id = coco_image_id(fn)
        if img_id in finished:
            continue
        out, n = r.generate(Image.open(os.path.join(a.data_path, fn)).convert("RGB"), CHAIR_PROMPT, a.max_new_tokens)
        append_jsonl(cap_file, {"image_id": img_id, "image": fn, "caption": out, "num_tokens": n})
    print(f"captions -> {cap_file}; score with eval_bench/chair.py --cap_file {cap_file}")


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    if a.js_gamma is None:
        a.js_gamma = 0.25 if a.benchmark == "chair" else 0.2
    if a.max_new_tokens is None:
        a.max_new_tokens = CHAIR_MAX_NEW_TOKENS if a.benchmark == "chair" else YESNO_MAX_NEW_TOKENS
    r = Runner(a)
    {"pope": run_pope, "beaf": run_beaf, "chair": run_chair}[a.benchmark](r)


if __name__ == "__main__":
    main()
