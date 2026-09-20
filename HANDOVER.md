# Hướng dẫn Vận hành & Bàn giao: POPE Benchmark cho ONLY trên Kaggle (2× T4 GPU, BF16)

Tài liệu hướng dẫn chi tiết quy trình chuẩn hóa và thực thi benchmark **POPE (Random, Popular, Adversarial)** cho phương pháp **ONLY** (ICCV'25) trên môi trường **Kaggle Notebooks** (2× GPU NVIDIA Tesla T4 16GB, BF16 Full Precision).

---

## 🎯 1. Mục tiêu & Các mô hình hỗ trợ
- **POPE Splits**: `random`, `popular`, `adversarial` (hoặc `--split all` chạy tuần tự).
- **Mô hình**:
  - **LLaVA-1.5-7B**: Checkpoint HuggingFace `llava-hf/llava-1.5-7b-hf` via `LlavaForConditionalGeneration`.
  - **Qwen2-VL-7B-Instruct**: Checkpoint HuggingFace `Qwen/Qwen2-VL-7B-Instruct` via `Qwen2VLForConditionalGeneration`.
- **Phương pháp**: Can thiệp giảm thiểu hallucination **ONLY** (`--use_only True`) hoặc baseline (`--use_only False`).

---

## ⚠️ 2. Cảnh báo quan trọng về Dependencies & Môi trường Kaggle
> [!CAUTION]
> **Tuyệt đối KHÔNG chạy `pip install -r requirements.txt` trên Kaggle!**
> File `requirements.txt` cũ chứa các phiên bản cổ điển (`torch==2.0.1`, `torchvision==0.15.2`, `torchaudio`, `transformers==4.31.0`).
> Nếu chạy lệnh này, pip sẽ hạ cấp PyTorch làm hỏng driver CUDA và gây ra lỗi nghiêm trọng:
> `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`

### Cấu hình chuẩn Cell 1 (BẮT BUỘC):
```python
# 1. Gỡ bỏ torchaudio (dự án chỉ dùng Ảnh + Chữ, gỡ bỏ để tránh 100% xung đột CUDA mismatch)
!pip uninstall -y -q torchaudio

# 2. Cài đặt các thư viện cần thiết (KHÔNG cài torch/torchvision để giữ nguyên driver CUDA của Kaggle)
!pip install -q --no-cache-dir \
    "transformers>=4.45.0" \
    "accelerate>=0.26.0" \
    sentencepiece \
    protobuf \
    tiktoken \
    qwen_vl_utils \
    pyyaml \
    tqdm \
    huggingface_hub \
    pandas

# 3. Dòng kiểm tra xác thực dependency:
import torch, transformers, accelerate, qwen_vl_utils, sentencepiece
from transformers import AutoProcessor, AutoTokenizer
print(f"✅ Dependency Verification PASSED! PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()}) | Transformers: {transformers.__version__}")
```

### Phòng ngừa lỗi `AttributeError: 'LlavaConfig' object has no attribute 'eos_token_id'`:
Trên `transformers >= 4.45.0`, `LlavaConfig` không còn attribute `eos_token_id`.
Hàm `resolve_eos_token_id()` trong `eval_bench/eval_pope.py` đã cài đặt cơ chế fallback an toàn:
```python
eos_token_id = getattr(self.model, "generation_config", None)
eos_token_id = getattr(eos_token_id, "eos_token_id", None) if eos_token_id else None
if eos_token_id is None and hasattr(self.model, "config"):
    eos_token_id = getattr(self.model.config, "eos_token_id", None)
if eos_token_id is None and hasattr(self.model.config, "text_config"):
    eos_token_id = getattr(self.model.config.text_config, "eos_token_id", None)
if eos_token_id is None and hasattr(self.processor, "tokenizer"):
    eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
```

---

## 🛠️ 3. Cấu trúc Codebase Mới

| File | Mô tả |
| :--- | :--- |
| `only_utils/only_llava.py` | Implementation sạch của ONLY cho LLaVA-1.5 trên `transformers >= 4.45.0` bằng PyTorch hooks và LogitsProcessor. Hỗ trợ multi-GPU `device_map="auto"`. |
| `only_utils/only_qwen2vl.py` | Implementation của ONLY cho Qwen2-VL, đã bổ sung bảo vệ an toàn phân bổ thiết bị đa GPU và `mrope_section`. |
| `eval_bench/eval_pope.py` | Trình thực thi trung tâm POPE: nạp model 1 lần duy nhất, hỗ trợ `--split all`, greedy `max_new_tokens=6`, tự động thêm prompt suffix cho QwenVL. |
| `eval_bench/pope_auto_detect.py` | Tự động phát hiện ảnh COCO val2014 (`/kaggle/input/datasets/biminhco/val2014/val2014` hoặc quét đệ quy) và POPE annotations (tự tải từ GitHub nếu thiếu). |
| `eval_bench/eval_common.py` | Bổ sung hàm `pope_parse_extended` nhận diện yes/no/unknown và cập nhật `binary_metrics` theo dõi `Unknowns`. |
| `kaggle_only_pope.ipynb` | Notebook Kaggle 8 cell hoàn chỉnh từ setup, login, clone, verify GPU/path, chạy benchmark `--split all`, đến hiển thị DataFrame. |
| `requirements_kaggle.txt` | Danh sách package an toàn cho Kaggle. |
| `tests/test_only_llava.py` | CPU unit test cho `OnlyLlava`. |
| `tests/test_only_qwen2vl.py` | CPU unit test cho `OnlyQwen2VL`. |

---

## 🚀 4. Hướng dẫn Chạy Benchmark

### Chạy qua Kaggle Notebook:
1. Mở Kaggle, tạo một Notebook mới với Accelerator: **GPU T4 x 2**, Internet: **Always On**.
2. Upload hoặc import notebook [`kaggle_only_pope.ipynb`](file:///c:/Project_files/VLM_Hallulu/reference_codebases/ONLY/kaggle_only_pope.ipynb).
3. Đính kèm dataset ảnh COCO val2014 (`datasets/biminhco/val2014`).
4. Thêm Secret `HF_TOKEN` trong mục **Add-ons -> Secrets** (nếu cần tải checkpoint cá nhân).
5. Chọn mô hình tại Cell 6 (`MODEL_CHOICE = "llava"` hoặc `"qwen2vl"`) và nhấn **Run All**.

### Chạy trực tiếp qua dòng lệnh:
```bash
# LLaVA-1.5 7B (ONLY method, cả 3 split)
python eval_bench/eval_pope.py \
    --model llava \
    --split all \
    --use_only True \
    --max_new_tokens 6 \
    --device_map auto \
    --precision bfloat16

# Qwen2-VL 7B (ONLY method, cả 3 split)
python eval_bench/eval_pope.py \
    --model qwen2vl \
    --split all \
    --use_only True \
    --max_new_tokens 6 \
    --device_map auto \
    --precision bfloat16
```

---

## 📊 5. Cấu trúc Kết quả & Bảng Báo cáo
Mỗi lần chạy sẽ sinh thư mục: `results/<model>_<method>_pope_<timestamp>/`:
- `<split>/raw_outputs.jsonl`: Từng câu hỏi, câu trả lời, nhãn gốc, và nhãn dự đoán.
- `<split>/run_config.json`: Cấu hình siêu tham số, seed, thời gian chạy.
- `<split>/metrics.json`: Accuracy, Precision, Recall, F1, YesRatio, Unknowns, Total.
- `summary_metrics.csv`: Bảng tổng hợp DataFrame hiển thị ra console:
```
================================================================================
                          POPE BENCHMARK SUMMARY
================================================================================
      Split  Accuracy  Precision  Recall    F1  YesRatio  Unknowns  Total
     random     88.20      86.50   90.50 88.45     52.30         0   3000
    popular     85.10      83.20   88.10 85.58     53.00         0   3000
adversarial     83.40      81.00   87.20 83.98     53.80         0   3000
    Average     85.57      83.57   88.60 86.00     53.03         0   9000
================================================================================
```
