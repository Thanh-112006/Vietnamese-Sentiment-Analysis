from collections.abc import Mapping, Sequence

from sentiment.data.loaders import TextExample


def split_dataset(
    examples: Sequence[TextExample], *, seed: int, dev_size: float = 0.1, test_size: float = 0.2
) -> dict[str, list[TextExample]]:
    """Chia dữ liệu theo nhãn thành ba tập train, dev và test với seed cố định."""
    raise NotImplementedError


def validate_splits(splits: Mapping[str, Sequence[TextExample]]) -> None:
    """Kiểm tra trùng lặp và rò rỉ dữ liệu giữa các tập."""
    raise NotImplementedError
