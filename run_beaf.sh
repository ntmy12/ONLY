#!/bin/bash
set -euo pipefail

DATA_DIR="../../data"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
QNA_PATH="$DATA_DIR/beaf/beaf_qna.json"
BEAF_METRIC="$DATA_DIR/beaf/beaf_metric.py"
IMAGE_ROOT="$DATA_DIR/beaf"

for f in "$QNA_PATH" "$BEAF_METRIC"; do
  [[ -f "$f" ]] || { echo "THIẾU FILE: $f"; exit 1; }
done
[[ -d "$IMAGE_ROOT" ]] || { echo "THIẾU THƯ MỤC ẢNH: $IMAGE_ROOT"; exit 1; }



python eval_bench/beaf_eval_llava.py \
  --model_path "$MODEL_PATH" \
  --qna_path "$QNA_PATH" \
  --image_root "$IMAGE_ROOT" \
  --beaf_metric "$BEAF_METRIC" \
  --out_path "./results/beaf" \
  --use_only True \
  --enhance_layer_index 0 \
  --ritual_alpha_pos 3.0 \
  --ritual_alpha_neg 1.0 \
  --ritual_beta 0.1 \
  --js_gamma 0.2

echo "Xong. Xem: ./results/beaf/llava-1.5-7b/ONLY/*_metrics.txt"
