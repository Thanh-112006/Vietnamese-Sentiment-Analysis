import random
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Literal
from unicodedata import normalize

from sentiment.data.loaders import TextExample
from sentiment.preprocess.teencode import TEENCODE_REPLACEMENTS

TestCondition = Literal["clean", "no_accent", "teencode", "mixed"]

_VIETNAMESE_MARKS = frozenset("\u0300\u0301\u0302\u0303\u0306\u0309\u031b\u0323")
_STROKED_D = str.maketrans("đĐ", "dD")
_TEENCODE_TERMS = "|".join(
    re.escape(term) for term in sorted(TEENCODE_REPLACEMENTS, key=len, reverse=True)
)
_TEENCODE_PATTERN = re.compile(rf"(?<!\w)(?:{_TEENCODE_TERMS})(?!\w)", re.IGNORECASE)


def apply_teencode(text: str, *, seed: int, rate: float = 0.8) -> str:
    """Thay từ hoặc cụm khớp trọn vẹn với xác suất rate cho mỗi lần khớp."""
    if not 0 <= rate <= 1:
        raise ValueError("rate phải nằm trong khoảng [0, 1].")

    rng = random.Random(seed)

    def replace(match: re.Match[str]) -> str:
        original = match.group()
        replacement = TEENCODE_REPLACEMENTS.get(original.casefold())
        if replacement is None or rng.random() >= rate:
            return original
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return _TEENCODE_PATTERN.sub(replace, text)


def remove_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt và đổi đ/Đ thành d/D."""
    decomposed = normalize("NFD", text.translate(_STROKED_D))
    return normalize("NFC", "".join(char for char in decomposed if char not in _VIETNAMESE_MARKS))


def generate_noisy_datasets(
    test_examples: Sequence[TextExample], *, seed: int, teencode_rate: float = 0.8
) -> dict[TestCondition, list[TextExample]]:
    """Trả clean (gốc), no_accent (bỏ dấu), teencode và mixed (teencode rồi bỏ dấu).
    Cả bốn giữ nguyên thứ tự, ID mẫu và nhãn của tập test."""
    if not 0 <= teencode_rate <= 1:
        raise ValueError("teencode_rate phải nằm trong khoảng [0, 1].")

    datasets: dict[TestCondition, list[TextExample]] = {
        "clean": list(test_examples),
        "no_accent": [],
        "teencode": [],
        "mixed": [],
    }

    for example in test_examples:
        digest = sha256(f"{seed}:{example.id}".encode("utf-8")).digest()
        sample_seed = int.from_bytes(digest, "big")
        teencode_text = apply_teencode(
            example.text, seed=sample_seed, rate=teencode_rate
        )

        datasets["no_accent"].append(
            example._replace(text=remove_accents(example.text))
        )
        datasets["teencode"].append(example._replace(text=teencode_text))
        datasets["mixed"].append(
            example._replace(text=remove_accents(teencode_text))
        )

    return datasets
