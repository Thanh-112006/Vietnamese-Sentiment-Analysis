from typing import NamedTuple
from unicodedata import normalize

from sentiment.data.loaders import TextExample


class PreparedTextExample(NamedTuple):
    """Một mẫu giữ cả văn bản gốc và bản sạch chưa tách từ."""

    id: str
    text_raw: str
    text: str
    label_id: int


def clean_text(text: str, *, lowercase: bool = False) -> str:
    """Chuẩn hóa Unicode NFC và khoảng trắng, giữ dấu tiếng Việt và dấu câu."""
    if lowercase:
        text = text.lower()
    return " ".join(normalize("NFC", text).split())


def prepare_example(example: TextExample, *, lowercase: bool = False) -> PreparedTextExample:
    """Giữ ID, nhãn và câu gốc khi tạo bản sạch chưa tách từ."""
    return PreparedTextExample(
        id=example.id,
        text_raw=example.text,
        text=clean_text(example.text, lowercase=lowercase),
        label_id=example.label_id,
    )
