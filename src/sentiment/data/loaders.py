from pathlib import Path
from typing import Literal, NamedTuple


class TextExample(NamedTuple):
    """Một mẫu với ID ổn định và nhãn theo thứ tự 0, 1, 2."""

    id: str
    text: str
    label_id: int


def load_uit_vsfc(source: Path, *, split: Literal["train", "dev", "test"]) -> list[TextExample]:
    """Đọc một split từ thư mục gốc UIT-VSFC, ghép câu và nhãn theo số dòng.
    Tạo ID như test:000001 từ tên split và số dòng gốc bắt đầu từ 1."""
    raise NotImplementedError


def load_uit_visfd(source: Path) -> list[TextExample]:
    """Nạp bình luận UIT-ViSFD đã có nhãn cảm xúc toàn văn bản."""
    raise NotImplementedError


def load_vlsp_sentiment(source: Path) -> list[TextExample]:
    """Nạp dữ liệu VLSP Sentiment theo schema chung."""
    raise NotImplementedError
