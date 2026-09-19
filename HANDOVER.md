# Yêu cầu và Bối cảnh: Triển khai can thiệp OPERA cho benchmark POPE

## 🎯 Mục tiêu chính
Chạy đánh giá benchmark POPE cho các mô hình (ví dụ: LLaVA, Qwen-VL) trên cả 3 tập dữ liệu con là **adversarial**, **random**, và **popular**, đồng thời áp dụng phương pháp can thiệp **OPERA** (Over-Trust Penalty and Retrospection-Allocation).

---

## 🔍 Bối cảnh hiện tại (Context)
* **Thư mục làm việc:** Dự án hiện tại (`ONLY`) là một framework dùng để đánh giá các phương pháp can thiệp (intervention) nhằm giảm thiểu hallucination trên các Large Vision-Language Models (LVLMs).
* **Các phương pháp đã có:** Codebase hiện đã hỗ trợ sẵn các thuật toán như RITUAL, VCD, M3ID, và ONLY.
* **Cấu trúc luồng chạy:** 
  * Các script python chạy đánh giá POPE nằm ở thư mục `eval_bench/` (ví dụ: `eval_bench/pope_eval_llava.py`). Các file này parse các cờ (flags) như `--use_ritual`, `--use_vcd`... và truyền chúng vào hàm `model.generate()`.
  * Các shell script để chạy hàng loạt nằm ở `eval_bench/scripts/` (ví dụ: `eval_bench/scripts/pope_eval.sh`).
* **Cách codebase can thiệp vào model:** Dự án sử dụng kỹ thuật "monkey patch" hàm sinh văn bản của thư viện Transformers. File `only_utils/only_sample.py` đang patch hàm `sample` của `transformers.generation.utils.GenerationMixin` để chèn logic của RITUAL, VCD, M3ID và ONLY.
* **Tình trạng OPERA:** Hiện tại, logic thuật toán OPERA **chưa được implement** trong repository này.
* **Decoding method:** Code đang được thiết lập để chạy **greedy decoding** mặc định cho tất cả các framework can thiệp.

---

## 📋 Yêu cầu công việc cho Agent tiếp theo (Next Steps)

**1. Tích hợp thuật toán OPERA (Phần cốt lõi)**
* **Vị trí cần xử lý:** Cần thiết kế logic can thiệp của OPERA (thường liên quan đến việc tính toán attention maps, áp dụng over-trust penalty và retrospection-allocation trong quá trình giải mã).
* **Cách thực hiện:** Bạn có thể tham khảo cách codebase đang làm ở `only_utils/only_sample.py` để patch hàm generate của HuggingFace, hoặc tạo một file utility mới như `only_utils/opera_utils.py` để xử lý việc ghi đè logic decoding riêng cho OPERA (vì OPERA thường dùng dạng beam search sửa đổi thay vì chỉ can thiệp vào logits).

**2. Cập nhật các file Python đánh giá (Evaluation Scripts)**
* **File cần sửa:** `eval_bench/pope_eval_llava.py` (và các mô hình khác nếu cần).
* **Chi tiết:**
  * Thêm cờ `--use_opera` (kiểu boolean) vào `argparse`.
  * Thêm cấu hình các siêu tham số (hyperparameters) đặc thù của OPERA (ví dụ: scale factor, penalty...).
  * Cập nhật logic tạo thư mục log: `elif args.use_opera: method_name = "OPERA"`.
  * Truyền biến `use_opera=args.use_opera` vào `model.generate(...)`.

**3. Tạo hoặc cập nhật Bash Script**
* **File cần sửa:** Có thể sửa `eval_bench/scripts/pope_eval.sh` hoặc tạo một bản sao mới (ví dụ: `pope_eval_opera.sh`).
* **Chi tiết:** 
  * Viết một vòng lặp (for loop) để tự động chạy qua 3 tập dữ liệu: `for type in "random" "popular" "adversarial"; do ... done`.
  * Bật cờ `--use_opera True` và cấu hình các tham số môi trường hoặc tham số thuật toán cần thiết.

**4. Chạy kiểm thử (Testing)**
* Chạy thử script với một batch size nhỏ để đảm bảo hàm generate không bị crash khi có sự can thiệp của OPERA.
* Kiểm tra xem các file `*_predictions.jsonl` và `*_metrics.json` có được sinh ra chính xác tại `results/pope/{model}/OPERA/` hay không.
