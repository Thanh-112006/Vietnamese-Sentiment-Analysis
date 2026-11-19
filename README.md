# Vietnamese Sentiment Analysis

Dự án phân loại phản hồi tiếng Việt thành ba nhóm tiêu cực, trung tính và tích cực. Khung hiện tại chuẩn bị cho TF-IDF + Logistic Regression, BiLSTM và PhoBERT.

## Cấu trúc thư mục chính

```text
Vietnamese-Sentiment-Analysis/
├── README.md
├── pyproject.toml        # Cấu hình package và dependency
├── requirements.txt     # Cài đặt môi trường
├── configs/             # Cấu hình chung, mô hình và ablation
├── data/
│   ├── raw/             # Dữ liệu gốc lưu cục bộ
│   ├── processed/       # Dữ liệu đã chuẩn bị
│   └── noisy/           # Các biến thể của tập test
├── src/
│   └── sentiment/
│       ├── data/         # Nạp, chia và tạo biến thể dữ liệu
│       ├── preprocess/   # Làm sạch, tách từ và chuẩn hóa teencode
│       ├── embeddings/   # Nạp vector fastText
│       ├── models/       # TF-IDF + LR, BiLSTM và PhoBERT
│       ├── utils/        # Cấu hình và seed
│       ├── metrics.py
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── scripts/             # Lệnh chuẩn bị dữ liệu và chạy thí nghiệm
├── experiments/         # Kết quả theo từng lượt chạy
├── notebooks/           # Khám phá dữ liệu và phân tích lỗi
├── reports/             # Bảng, hình và các ca lỗi
├── app/                 # Khung demo Streamlit
└── tests/               # Khung kiểm thử
```

## 1. Cài đặt môi trường

Dự án yêu cầu Python 3.10 trở lên. Các lệnh dưới đây dùng PowerShell trên Windows và đã được kiểm tra với Python 3.13.5 trong môi trường dự án.

Chạy từ thư mục gốc chứa `pyproject.toml`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

Khi `pip check` trả về `No broken requirements found`, pip đã kiểm tra tính nhất quán của các dependency. Với môi trường đã tạo, chỉ cần kích hoạt lại khi mở terminal mới.

**Ghi chú cho TV2.** Nếu dùng mô hình fastText dạng `.bin` cho BiLSTM, hãy cài thêm nhóm dependency `embeddings` trong môi trường đã kích hoạt.

```powershell
python -m pip install -e ".[embeddings]"
```

## 2. Chuẩn bị dữ liệu UIT-VSFC

Tải **UIT-VSFC v1.0** từ mục dữ liệu của UIT NLP Group. Sau khi tải và giải nén nếu cần, đặt ba tập gốc vào `data/raw/UIT-VSFC/`. Dữ liệu gốc được lưu cục bộ và được Git bỏ qua. [Trang dữ liệu UIT NLP Group](https://sites.google.com/uit.edu.vn/uit-nlp/datasets) · [Tải UIT-VSFC](https://drive.google.com/drive/folders/1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X)

```text
data/raw/UIT-VSFC/
├── train/
│   ├── sents.txt
│   └── sentiments.txt
├── dev/
│   ├── sents.txt
│   └── sentiments.txt
└── test/
    ├── sents.txt
    └── sentiments.txt
```

Mỗi dòng trong `sents.txt` ghép với dòng cùng vị trí trong `sentiments.txt`. Giữ thứ tự dòng và các tập `train`, `dev`, `test` đã phát hành. Số cặp câu–nhãn tương ứng là 11.426, 1.583 và 3.166. [Thẻ dữ liệu UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback#data-splits)

[Loader UIT-VSFC](src/sentiment/data/loaders.py) sẽ tạo bản ghi `TextExample(id, text, label_id)`. ID dùng tên tập và số dòng gốc bắt đầu từ 1, chẳng hạn `test:000001`. Cùng một mẫu dùng chung ID trong các biến thể test và `predictions.csv` để có thể ghép dự đoán với nhãn gốc khi chấm điểm.

Mã nhãn trong tệp gốc khớp với thứ tự trong [cấu hình dự án](configs/base.yaml). [Định nghĩa nhãn UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/vietnamese_students_feedback.py)

| `label_id` | Nhãn | Ý nghĩa |
| --- | --- | --- |
| `0` | `negative` | Tiêu cực |
| `1` | `neutral` | Trung tính |
| `2` | `positive` | Tích cực |
