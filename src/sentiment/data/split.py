from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import TypedDict

from sentiment.data.loaders import TextExample
from sentiment.preprocess.clean import clean_text

_SPLIT_NAMES = ("train", "dev", "test")


class SplitStats(TypedDict):
    num_samples: int
    label_counts: dict[int, int]
    duplicate_rows: int


class SplitValidationReport(TypedDict):
    splits: dict[str, SplitStats]
    cross_split_overlaps: dict[str, int]


def split_dataset(
    examples: Sequence[TextExample], *, seed: int, dev_size: float = 0.1, test_size: float = 0.2
) -> dict[str, list[TextExample]]:
    """Chia dữ liệu theo nhãn thành ba tập train, dev và test với seed cố định."""
    raise NotImplementedError


def validate_splits(
    splits: Mapping[str, Sequence[TextExample]], *, expected_counts: Mapping[str, int] | None = None
) -> SplitValidationReport:
    """Kiểm tra số mẫu, ID, văn bản và nhãn của ba tập.
    Thống kê dòng lặp trong mỗi tập và số câu chuẩn hóa chung theo từng cặp tập."""
    if set(splits) != set(_SPLIT_NAMES):
        raise ValueError(f"Cần đúng ba tập train, dev và test. Nhận được {sorted(splits)}.")
    if expected_counts is not None and set(expected_counts) != set(_SPLIT_NAMES):
        raise ValueError("expected_counts phải có số mẫu cho cả train, dev và test.")

    split_stats: dict[str, SplitStats] = {}
    text_keys: dict[str, set[str]] = {}
    id_to_split: dict[str, str] = {}
    for split in _SPLIT_NAMES:
        examples = splits[split]
        if not examples:
            raise ValueError(f"Tập {split!r} không có mẫu.")
        if expected_counts is not None and len(examples) != expected_counts[split]:
            raise ValueError(f"Tập {split!r} có {len(examples)} mẫu, cần {expected_counts[split]} mẫu.")

        label_counts = {0: 0, 1: 0, 2: 0}
        unique_texts: set[str] = set()
        for example in examples:
            if not isinstance(example.id, str) or not example.id.strip():
                raise ValueError(f"Tập {split!r} có mẫu thiếu ID hợp lệ.")
            if example.id in id_to_split:
                raise ValueError(f"ID {example.id!r} trùng ở tập {split!r} và {id_to_split[example.id]!r}.")
            id_to_split[example.id] = split

            if type(example.label_id) is not int or example.label_id not in label_counts:
                raise ValueError(f"Mẫu {example.id!r} ở tập {split!r} phải có nhãn nguyên 0, 1 hoặc 2.")
            label_counts[example.label_id] += 1

            key = clean_text(example.text).casefold()
            if not key:
                raise ValueError(f"Mẫu {example.id!r} ở tập {split!r} có câu rỗng.")
            unique_texts.add(key)

        text_keys[split] = unique_texts
        split_stats[split] = {
            "num_samples": len(examples),
            "label_counts": label_counts,
            "duplicate_rows": len(examples) - len(unique_texts),
        }

    overlaps = {
        f"{left}/{right}": len(text_keys[left] & text_keys[right])
        for left, right in combinations(_SPLIT_NAMES, 2)
    }
    return {"splits": split_stats, "cross_split_overlaps": overlaps}
