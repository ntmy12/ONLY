#!/bin/bash
# Full protocol: POPE-COCO (3 splits) + BEAF + CHAIR, for Regular greedy and ONLY greedy.
#   bash eval_bench/scripts/run_all.sh llava     # env: requirements.txt (+ pip install -e transformers)
#   bash eval_bench/scripts/run_all.sh qwen2vl   # env: requirements_qwen2vl.txt
set -euo pipefail
MODEL=${1:?usage: run_all.sh llava|qwen2vl}

# ---------------- paths (edit) ----------------
LLAVA_PATH="liuhaotian/llava-v1.5-7b"
QWEN_PATH="Qwen/Qwen2-VL-7B-Instruct"
COCO_ROOT="/data/coco"                             # contains val2014/ and annotations/
POPE_DIR="/data/POPE/output/coco"                  # RUCAIBox/POPE output/coco
BEAF_REPO="data/beaf"                             # git clone https://github.com/postech-ami/BEAF (beaf_qna.json, beaf_metric.py)
BEAF_IMAGES="data/beaf"                    # images from the BEAF Google Drive archive
CHAIR_SEED="[TODO]"
OUT="./results"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------------- ONLY hyper-parameters (official scripts of the ONLY repo) ----------------
LAYER=0; ALPHA_POS=3.0; ALPHA_NEG=1.0; BETA=0.1
GAMMA_YESNO=0.2      # eval_bench/scripts/pope_eval.sh (POPE; BEAF uses the same yes/no protocol)
GAMMA_CHAIR=0.25     # eval_bench/scripts/chair_eval.sh

[[ "$CHAIR_SEED" == "[TODO]" ]] && { echo "set CHAIR_SEED first"; exit 1; }
IMAGE_LIST="$OUT/chair/chair_images_seed${CHAIR_SEED}.json"
mkdir -p "$OUT/chair"

for USE_ONLY in False True; do
  COMMON_ONLY="--use_only $USE_ONLY --enhance_layer_index $LAYER --ritual_alpha_pos $ALPHA_POS --ritual_alpha_neg $ALPHA_NEG --ritual_beta $BETA"
  if [[ "$MODEL" == "llava" ]]; then
    for SPLIT in random popular adversarial; do
      python eval_bench/pope_eval_llava.py --model_path $LLAVA_PATH --model_base llava \
        --pope_path $POPE_DIR/coco_pope_${SPLIT}.json --data_path $COCO_ROOT/val2014 --type $SPLIT \
        --log_path $OUT/logs --out_path $OUT/pope --greedy True $COMMON_ONLY --js_gamma $GAMMA_YESNO
    done
    python eval_bench/beaf_eval_llava.py --model_path $LLAVA_PATH --qna_path $BEAF_REPO/beaf_qna.json \
      --image_root $BEAF_IMAGES --beaf_metric $BEAF_REPO/beaf_metric.py --out_path $OUT/beaf \
      --greedy True $COMMON_ONLY --js_gamma $GAMMA_YESNO
    python eval_bench/chair_eval_llava.py --model_path $LLAVA_PATH --model_base llava \
      --data_path $COCO_ROOT/val2014/ --anno_path $COCO_ROOT/annotations/instances_val2014.json \
      --log_path $OUT/logs/chair --out_path $OUT/chair/llava-1.5-7b --method_name only \
      --chair_seed $CHAIR_SEED --image_list $IMAGE_LIST --greedy True $COMMON_ONLY --js_gamma $GAMMA_CHAIR
  else
    for SPLIT in random popular adversarial; do
      python eval_bench/qwen2vl_eval.py --benchmark pope --model_path $QWEN_PATH --out_path $OUT \
        --pope_path $POPE_DIR/coco_pope_${SPLIT}.json --data_path $COCO_ROOT/val2014 --type $SPLIT \
        $COMMON_ONLY --js_gamma $GAMMA_YESNO
    done
    python eval_bench/qwen2vl_eval.py --benchmark beaf --model_path $QWEN_PATH --out_path $OUT \
      --qna_path $BEAF_REPO/beaf_qna.json --image_root $BEAF_IMAGES --beaf_metric $BEAF_REPO/beaf_metric.py \
      $COMMON_ONLY --js_gamma $GAMMA_YESNO
    python eval_bench/qwen2vl_eval.py --benchmark chair --model_path $QWEN_PATH --out_path $OUT \
      --data_path $COCO_ROOT/val2014 --chair_seed $CHAIR_SEED --image_list $IMAGE_LIST \
      $COMMON_ONLY --js_gamma $GAMMA_CHAIR
  fi
done

# ---------------- CHAIR scoring (Maxlinn CHAIR-metric-standalone) ----------------
for CAP in $(find $OUT/chair -name "*seed${CHAIR_SEED}.jsonl"); do
  python eval_bench/chair.py --cap_file $CAP --coco_path $COCO_ROOT/annotations \
    --cache $OUT/chair/chair_evaluator.pkl --save_path ${CAP%.jsonl}_chair.json \
    --image_id_key image_id --caption_key caption
done

python eval_bench/collect_results.py --out_path $OUT
