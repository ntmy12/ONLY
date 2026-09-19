"""Shared evaluation protocol for POPE / BEAF / CHAIR (LLaVA-1.5 and Qwen2-VL runners).

Kept dependency-free (stdlib only) so both the LLaVA env (transformers 4.31 fork) and the
Qwen2-VL env (transformers 4.56) can import it.
"""
import json
import os
import random

# ----------------------------------------------------------------------------
# Protocol constants (identical for every method / model)
# ----------------------------------------------------------------------------
# POPE / BEAF: the benchmark question is fed verbatim (no answer-format suffix), exactly as in
# the original POPE/BEAF releases and in ONLY's own POPE setting. With greedy decoding the
# first sentence does not depend on max_new_tokens, so 32 only guards against truncation.
YESNO_MAX_NEW_TOKENS = 32
CHAIR_PROMPT = "Describe this image."
CHAIR_MAX_NEW_TOKENS = 128
CHAIR_NUM_IMAGES = 500

POPE_SPLITS = ("random", "popular", "adversarial")


# ----------------------------------------------------------------------------
# POPE
# ----------------------------------------------------------------------------
def pope_parse(text):
    """Official POPE rule (RUCAIBox/POPE evaluate.py): keep the first sentence, drop commas,
    split on spaces; 'No' / 'not' / 'no' -> no, otherwise yes."""
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    return "no" if ("No" in words or "not" in words or "no" in words) else "yes"


def binary_metrics(preds, labels):
    """preds/labels: iterables of 'yes'/'no'. Returns percentages + raw counts."""
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        if p == "yes" and l == "yes":
            tp += 1
        elif p == "yes" and l == "no":
            fp += 1
        elif p == "no" and l == "no":
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "Accuracy": 100 * (tp + tn) / n if n else 0.0,
        "Precision": 100 * prec,
        "Recall": 100 * rec,
        "F1": 100 * f1,
        "YesRatio": 100 * (tp + fp) / n if n else 0.0,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "N": n,
    }


def load_pope(pope_file):
    items = []
    with open(pope_file) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ----------------------------------------------------------------------------
# JSONL helpers (resumable runs)
# ----------------------------------------------------------------------------
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def done_keys(path, key):
    return {r[key] for r in read_jsonl(path)}


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# CHAIR image selection
# ----------------------------------------------------------------------------
def select_chair_images(image_dir, seed, num_images=CHAIR_NUM_IMAGES, cache_file=None):
    """Deterministic: sorted file listing -> random.Random(seed).sample. If `cache_file` exists it
    is reused, so every model/method is evaluated on exactly the same images."""
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        assert cached["seed"] == seed and len(cached["files"]) == num_images, (
            f"{cache_file} was created with a different seed/size; delete it or change --image_list")
        return cached["files"]
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    chosen = random.Random(seed).sample(files, num_images)
    if cache_file:
        write_json(cache_file, {"seed": seed, "image_dir": os.path.abspath(image_dir), "files": chosen})
    return chosen


def coco_image_id(file_name):
    return int(file_name.split(".jpg")[0][-12:])


# ----------------------------------------------------------------------------
# BEAF
# ----------------------------------------------------------------------------
def load_beaf(qna_path):
    with open(qna_path) as f:
        return json.load(f)


def beaf_image_index(image_root):
    """BEAF images are referenced by basename in beaf_qna.json; the Drive archive layout may nest
    them, so index every image under `image_root` by basename."""
    index = {}
    for root, _, files in os.walk(image_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                assert fn not in index, f"duplicate image name {fn} under {image_root}"
                index[fn] = os.path.join(root, fn)
    return index


def beaf_official_answer(text):
    """What beaf_metric.py will read: 'yes' substring first, then 'no'. Returns None when neither
    appears -- beaf_metric.py would then silently reuse the previous answer, so we count these."""
    t = text.lower()
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return None
