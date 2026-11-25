import csv
from pathlib import Path
from typing import Literal, NamedTuple


class TextExample(NamedTuple):
    """Một mẫu với ID ổn định và nhãn theo thứ tự 0, 1, 2."""

    id: str
    text: str
    label_id: int


def load_uit_vsfc(source: str | Path, *, split: Literal["train", "dev", "test"]) -> list[TextExample]:
    """Đọc một split từ thư mục gốc UIT-VSFC, ghép câu và nhãn theo số dòng.
    Tạo ID như test_000001 từ tên split và số dòng gốc có 6 chữ số, bắt đầu từ 1."""
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Split UIT-VSFC không hợp lệ {split!r}. Chọn train, dev hoặc test.")

    split_dir = Path(source) / split
    sentences_path = split_dir / "sents.txt"
    labels_path = split_dir / "sentiments.txt"
    with (
        sentences_path.open(encoding="utf-8-sig") as sentences_file,
        labels_path.open(encoding="utf-8-sig") as labels_file,
    ):
        sentences = sentences_file.readlines()
        labels = labels_file.readlines()

    if len(sentences) != len(labels):
        raise ValueError(
            f"Split {split!r} có {len(sentences)} dòng trong sents.txt "
            f"và {len(labels)} dòng trong sentiments.txt."
        )
    if not sentences:
        raise ValueError(f"Split {split!r} không có mẫu trong {split_dir}.")

    examples = []
    for line_number, (sentence, label) in enumerate(zip(sentences, labels), start=1):
        text = sentence.removesuffix("\n")
        label = label.strip()
        if not text.strip():
            raise ValueError(f"Câu rỗng tại {sentences_path}:{line_number}.")
        if label not in ("0", "1", "2"):
            raise ValueError(f"Nhãn {label!r} tại {labels_path}:{line_number} phải là 0, 1 hoặc 2.")

        examples.append(TextExample(f"{split}_{line_number:06d}", text, int(label)))

    return examples


def load_examples_csv(source: str | Path, *, split: Literal["train", "dev", "test"]) -> list[TextExample]:
    """Đọc cột text của CSV processed hoặc noisy theo đúng thứ tự trong tệp."""
    path = Path(source)
    examples: list[TextExample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        required = {"id", "split", "text", "label_id"}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{path} cần có các cột id, split, text, label_id.")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Sai số cột tại {path}:{reader.line_num}.")
            sample_id, text, label = row["id"], row["text"], row["label_id"]
            if not sample_id.strip() or sample_id in seen_ids:
                raise ValueError(f"ID trống hoặc trùng tại {path}:{reader.line_num}.")
            if row["split"] != split or label not in ("0", "1", "2") or not text.strip():
                raise ValueError(f"Sai split, nhãn hoặc câu rỗng tại {path}:{reader.line_num}.")
            examples.append(TextExample(sample_id, text, int(label)))
            seen_ids.add(sample_id)
    if not examples:
        raise ValueError(f"Split {split!r} không có mẫu trong {path}.")
    return examples


def load_uit_visfd(source: Path) -> list[TextExample]:
    """Nạp bình luận UIT-ViSFD đã có nhãn cảm xúc toàn văn bản."""
    raise NotImplementedError


def load_vlsp_sentiment(source: Path) -> list[TextExample]:
    """Nạp dữ liệu VLSP Sentiment theo schema chung."""
    raise NotImplementedError
