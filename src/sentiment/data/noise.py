from collections.abc import Sequence
from typing import Literal

from sentiment.data.loaders import TextExample

TestCondition = Literal["clean", "no_accent", "teencode", "mixed"]


def apply_teencode(text: str, *, seed: int, rate: float = 0.8) -> str:
    """Tạo biến thể teencode của văn bản."""
    raise NotImplementedError


def remove_accents(text: str) -> str:
    """Tạo biến thể không dấu của văn bản tiếng Việt."""
    raise NotImplementedError


def generate_noisy_datasets(
    test_examples: Sequence[TextExample], *, seed: int, teencode_rate: float = 0.8
) -> dict[TestCondition, list[TextExample]]:
    """Trả clean (gốc), no_accent (bỏ dấu), teencode và mixed (teencode rồi bỏ dấu).
    Cả bốn giữ nguyên thứ tự, ID mẫu và nhãn của tập test."""
    raise NotImplementedError
