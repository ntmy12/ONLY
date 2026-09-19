"""Collect POPE / BEAF / CHAIR results into one CSV laid out like the Baselines-Audit table."""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_common import read_jsonl  # noqa: E402

MODELS = {"llava-1.5-7b": "LLaVA-1.5-7B", "qwen2-vl-7b-instruct": "Qwen2-VL-7B"}
METHODS = {"Regular": "Greedy", "ONLY": "ONLY", "only": "ONLY", "regular": "Greedy"}


def model_of(path):
    for k, v in MODELS.items():
        if f"{os.sep}{k}{os.sep}" in path or f"/{k}/" in path:
            return v
    return "?"


def method_of(path):
    base = os.path.basename(path)
    for part in re.split(r"[\\/]", path) + [base.split("_")[0]]:
        if part in METHODS:
            return METHODS[part]
    return "?"


def pope_rows(root):
    rows = []
    for f in glob.glob(f"{root}/pope/**/*_metrics.json", recursive=True):
        m = json.load(open(f))
        split = next(s for s in ("random", "popular", "adversarial") if f"_{s}" in f)
        rows.append({"benchmark": "POPE-COCO", "split": split, "model": model_of(f), "method": method_of(f),
                     "Acc": m["Accuracy"], "Prec": m["Precision"], "Rec": m["Recall"], "F1": m["F1"], "file": f})
    return rows


BEAF_RE = re.compile(r"([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)(?:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+))?")


def beaf_rows(root):
    rows = []
    for f in glob.glob(f"{root}/beaf/**/*_metrics.txt", recursive=True):
        nums = [tuple(x for x in m.groups() if x) for m in BEAF_RE.finditer(open(f).read())]
        if len(nums) < 2:
            print(f"[warn] could not parse {f}")
            continue
        acc, prec, rec, f1 = map(float, nums[0])
        tu, ig, sbp, sbn, id_, f1tuid = map(float, nums[1])
        rows.append({"benchmark": "BEAF", "split": "", "model": model_of(f), "method": method_of(f),
                     "Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "TU": tu, "IG": ig, "SBp": sbp,
                     "SBn": sbn, "ID": id_, "F1TUID": f1tuid, "file": f})
    return rows


def chair_rows(root):
    rows = []
    for f in glob.glob(f"{root}/chair/**/*_chair.json", recursive=True):
        m = json.load(open(f))["overall_metrics"]
        caps = read_jsonl(f.replace("_chair.json", ".jsonl"))
        len_tok = sum(c["num_tokens"] for c in caps) / len(caps)
        rows.append({"benchmark": "CHAIR", "split": f"n={len(caps)}", "model": model_of(f), "method": method_of(f),
                     "CHAIRs": 100 * m["CHAIRs"], "CHAIRi": 100 * m["CHAIRi"], "Recall": 100 * m["Recall"],
                     "LenTokens": len_tok, "LenWords": 100 * m["Len"], "file": f})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_path", default="./results")
    a = p.parse_args()
    rows = pope_rows(a.out_path) + beaf_rows(a.out_path) + chair_rows(a.out_path)
    cols = ["benchmark", "split", "model", "method", "Acc", "Prec", "Rec", "F1", "TU", "IG", "SBp", "SBn", "ID",
            "F1TUID", "CHAIRs", "CHAIRi", "Recall", "LenTokens", "LenWords", "file"]
    out = os.path.join(a.out_path, "summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["benchmark"], r["split"], r["model"], r["method"])):
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
