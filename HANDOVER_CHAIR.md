# Hướng dẫn Vận hành: CHAIR Benchmark cho ONLY trên Kaggle / Colab (2× T4 GPU)

Tài liệu hướng dẫn chi tiết quy trình thực thi benchmark **CHAIR (Caption Hallucination Assessment with Image Relevance)** cho phương pháp **ONLY (ICCV'25)** trên môi trường **Kaggle Notebooks** (2× GPU NVIDIA Tesla T4 16GB, Full Precision BF16 / FP16 Tensor Cores).

---

## 🎯 1. Mục tiêu & Các yêu cầu chuẩn hóa

1. **Đo thời gian chạy chính xác (`import time`)**:
   - Sử dụng `time.perf_counter()` kết hợp `torch.cuda.synchronize()` trước và sau khi sinh caption để đo độ trễ GPU phần cứng ở độ chính xác micro-giây.
   - Chế độ **Latency Benchmark (`max_samples = 10`)** giúp tính thời gian chạy trung bình/mẫu (`avg_time_per_sample_s`), throughput (`samples/s`), và dự toán chính xác tổng thời gian chạy cho 500 ảnh.
2. **Quy chuẩn CHAIR 500 ảnh (seed 2027)**:
   - Tập hợp 500 ảnh COCO val2014 chuẩn hóa cố định (`selected_chair_val2014_seed2027.json`).
   - Prompt chuẩn: `"Describe this image."`.
   - Greedy decoding (`do_sample=False`, `temperature=None`), `max_new_tokens = 128`.
3. **Bộ công cụ đánh giá CHAIR Standalone**:
   - Tích hợp trực tiếp từ [Maxlinn/CHAIR-metric-standalone](https://github.com/Maxlinn/CHAIR-metric-standalone/tree/main).
   - Tự động nạp bộ nhớ đệm `chair.pkl` (hơn 123.000 ảnh COCO + từ điển từ đồng nghĩa) thông qua lớp `_CHAIRUnpickler` chống lỗi import namespace.
   - Tính toán đầy đủ 4 chỉ số:
     - **CHAIRs (%)**: Tỉ lệ câu có ít nhất 1 từ bị ảo giác (Sentence-level Hallucination).
     - **CHAIRi (%)**: Tỉ lệ từ bị ảo giác trên tổng số đối tượng sinh ra (Instance-level Hallucination).
     - **Recall (%)**: Tỉ lệ bao phủ các đối tượng thực tế (Ground-Truth Objects) của ảnh.
     - **Caption Length**: Độ dài trung bình của câu mô tả.
4. **Mô hình hỗ trợ**:
   - **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`) qua `OnlyLlava`.
   - **Qwen2-VL-7B-Instruct** (`Qwen/Qwen2-VL-7B-Instruct`) qua `OnlyQwen2VL`.
   - Hỗ trợ chuyển đổi linh hoạt giữa phương pháp **ONLY** (`--use_only True`) và **Baseline** (`--use_only False`).

---

## 🛠️ 2. Cấu trúc Files & Notebook

| File | Mô tả |
| :--- | :--- |
| [`kaggle_only_chair.ipynb`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/kaggle_only_chair.ipynb) | Notebook Kaggle 9 cell hoàn chỉnh: cài đặt an toàn, nạp model, chạy 10 samples đo độ trễ, chạy 500 samples, tính CHAIR metric, và hiển thị DataFrame. |
| [`eval_bench/eval_chair.py`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/eval_bench/eval_chair.py) | Trình chạy trung tâm CHAIR CLI: tự động tìm COCO val2014, đo micro-latency với `time.perf_counter()`, can thiệp ONLY hooks, và chấm điểm Maxlinn CHAIR. |
| [`eval_bench/chair.py`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/eval_bench/chair.py) | Thuật toán CHAIR Standalone của tác giả Maxlinn (lemmatization NLTK + WordNet, synonyms COCO). |
| [`eval_bench/chair.pkl`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/eval_bench/chair.pkl) | Pre-computed evaluator cache từ Maxlinn repo giúp tính CHAIR tức thì trong 2 giây mà không cần file JSON annotations 1GB. |
| [`eval_bench/selected_chair_val2014_seed2027.json`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/eval_bench/selected_chair_val2014_seed2027.json) | Danh sách 500 ảnh COCO val2014 được lấy mẫu xác định với `seed=2027`. |

---

## 🚀 3. Hướng dẫn Chạy qua Kaggle Notebook

1. **Khởi tạo Kaggle Notebook**:
   - Accelerator: **GPU T4 x 2**
   - Internet: **Always On**
2. **Đính kèm Dataset**:
   - Thêm dataset ảnh COCO val2014 (ví dụ: `datasets/biminhco/val2014` hoặc `coco-2014-val`).
3. **Mở notebook [`kaggle_only_chair.ipynb`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/kaggle_only_chair.ipynb)**:
   - **Cell 1**: Cài đặt môi trường sạch (gỡ `torchaudio`, cài `transformers >= 4.45.0`, `accelerate`, `nltk`).
   - **Cell 6 (Đo độ trễ 10 mẫu)**:
     ```python
     MODEL_CHOICE = "llava"  # hoặc "qwen2vl"
     USE_ONLY = True

     !python eval_bench/eval_chair.py \
         --model {MODEL_CHOICE} \
         --use_only {USE_ONLY} \
         --max_samples 10 \
         --prompt "Describe this image." \
         --max_new_tokens 128 \
         --seed 2027 \
         --precision auto \
         --device_map auto \
         --out_dir ./results/{MODEL_CHOICE}_chair_latency_10samples
     ```
   - **Cell 7 (Đánh giá Full 500 mẫu)**:
     ```python
     !python eval_bench/eval_chair.py \
         --model {MODEL_CHOICE} \
         --use_only {USE_ONLY} \
         --max_samples 500 \
         --prompt "Describe this image." \
         --max_new_tokens 128 \
         --seed 2027 \
         --precision auto \
         --device_map auto \
         --out_dir ./results/{MODEL_CHOICE}_chair_{mode_name}_500samples
     ```
   - **Cell 8**: Tự động tổng hợp và hiển thị bảng DataFrame các chỉ số CHAIR và thời gian chạy.
   - **Cell 9**: Soi chi tiết từng caption sinh ra và các từ bị ảo giác (nếu có).

---

## 💻 4. Chạy trực tiếp qua dòng lệnh (CLI)

```bash
# 1. Chạy đo thời gian latency trên 10 mẫu (LLaVA-1.5, ONLY method)
python eval_bench/eval_chair.py \
    --model llava \
    --use_only True \
    --max_samples 10 \
    --prompt "Describe this image." \
    --max_new_tokens 128 \
    --seed 2027

# 2. Chạy full 500 ảnh CHAIR (LLaVA-1.5, ONLY method)
python eval_bench/eval_chair.py \
    --model llava \
    --use_only True \
    --max_samples 500 \
    --prompt "Describe this image." \
    --max_new_tokens 128 \
    --seed 2027

# 3. Chạy full 500 ảnh CHAIR (Qwen2-VL, ONLY method)
python eval_bench/eval_chair.py \
    --model qwen2vl \
    --use_only True \
    --max_samples 500 \
    --prompt "Describe this image." \
    --max_new_tokens 128 \
    --seed 2027
```

---

## 📊 5. Cấu trúc Kết quả

Mỗi lần chạy sẽ tạo thư mục tại `./results/<model>_chair_<mode>_<samples>_<timestamp>/`:
- `raw_outputs.jsonl`: Lưu từng ảnh (`image_id`, `file_name`), câu caption mô tả đã sinh, và độ trễ chính xác (`latency_s`).
- `run_config.json`: Cấu hình siêu tham số, seed, prompt, mô hình và thiết bị.
- `chair_details.json`: Phân tích chi tiết từng câu: đối tượng ground-truth, đối tượng sinh ra, và danh sách từ bị ảo giác.
- `metrics.json` & `summary_metrics.json`: Báo cáo chỉ số hoàn chỉnh (`CHAIRs`, `CHAIRi`, `Recall`, `avg_time_per_sample_s`, `total_inference_time_s`).
