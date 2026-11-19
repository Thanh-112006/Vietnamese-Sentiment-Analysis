from collections.abc import Sequence

Metrics = dict[str, float | list[list[int]]]


def compute_metrics(expected: Sequence[int], predicted: Sequence[int], *, labels: Sequence[int]) -> Metrics:
    """Tính macro-F1, accuracy và confusion matrix theo thứ tự nhãn."""
    raise NotImplementedError
