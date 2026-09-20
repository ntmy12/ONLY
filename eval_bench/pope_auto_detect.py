"""Smart auto-detection utilities for COCO val2014 images and POPE annotations.

Ensures zero configuration errors whether running on Kaggle Notebooks or locally.
"""
import glob
import os
import urllib.request

POPE_GITHUB_URL_BASE = "https://raw.githubusercontent.com/AoiDragon/POPE/master/output/coco"


def find_coco_images(data_path=None):
    """Automatically detect the COCO val2014 images directory.
    Priority:
      1. Explicit `data_path` parameter if valid.
      2. Kaggle dataset standard path: `/kaggle/input/datasets/biminhco/val2014/val2014`.
      3. Recursive search in `/kaggle/input/` for directory containing `COCO_val2014_*.jpg`.
      4. Common local fallback paths: `./data/coco/val2014`, `../data/coco/val2014`, `./val2014`.
    """
    candidates = []
    if data_path:
        candidates.append(os.path.abspath(data_path))

    # Kaggle default target
    candidates.append("/kaggle/input/datasets/biminhco/val2014/val2014")

    # Local paths
    candidates.extend([
        os.path.abspath("./data/coco/val2014"),
        os.path.abspath("../data/coco/val2014"),
        os.path.abspath("./val2014"),
        os.path.abspath("./data/val2014"),
    ])

    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            # Verify directory has COCO images
            sample = glob.glob(os.path.join(p, "COCO_val2014_*.jpg"))
            if not sample:
                sample = glob.glob(os.path.join(p, "*.jpg"))
            if len(sample) > 0:
                print(f"[AutoDetect] Found COCO images at: {p} ({len(sample)} sample images detected)")
                return p

    # Fallback: recursive scan in /kaggle/input if on Kaggle
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        print("[AutoDetect] Scanning /kaggle/input for COCO_val2014_*.jpg...")
        for root, dirs, files in os.walk(kaggle_input):
            if any(f.startswith("COCO_val2014_") and f.endswith(".jpg") for f in files[:20]):
                print(f"[AutoDetect] Successfully located COCO images at: {root}")
                return root

    raise FileNotFoundError(
        f"Could not locate COCO val2014 images directory.\n"
        f"Checked paths: {candidates}\n"
        f"Please check your Kaggle dataset mount or specify --data_path explicitly."
    )


def find_or_download_pope_annotations(pope_path=None, split="random", cache_dir="./data/pope/coco"):
    """Find or auto-download POPE annotation JSON files.
    Priority:
      1. Explicit `pope_path` if given and exists.
      2. Check candidate directories locally and in /kaggle/input.
      3. Auto-download directly from RUCAIBox/POPE GitHub repository.
    """
    filename = f"coco_pope_{split}.json"

    if pope_path and os.path.exists(pope_path):
        return os.path.abspath(pope_path)

    search_dirs = [
        cache_dir,
        "./eval_bench/pope_data",
        "./pope_annotations",
        "/kaggle/input/pope/coco",
        "/kaggle/input/pope",
    ]

    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            print(f"[AutoDetect] Found POPE annotation: {candidate}")
            return os.path.abspath(candidate)

    # Search in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                found = os.path.join(root, filename)
                print(f"[AutoDetect] Located POPE annotation at: {found}")
                return os.path.abspath(found)

    # Auto-download from GitHub
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.abspath(os.path.join(cache_dir, filename))
    url = f"{POPE_GITHUB_URL_BASE}/{filename}"
    print(f"[AutoDetect] {filename} not found locally. Auto-downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[AutoDetect] Successfully downloaded to {target_path}")
        return target_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download POPE annotations from {url}: {e}\n"
            f"Please ensure internet access is enabled in Kaggle Notebook settings or provide --pope_path."
        )
