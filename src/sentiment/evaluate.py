from collections.abc import Sequence
from typing import TypedDict

from sentiment.data.loaders import TextExample
from sentiment.data.noise import TestCondition
from sentiment.metrics import Metrics


class PredictionRow(TypedDict):
    """Một dòng kết quả trong predictions.csv."""

    id: str
    y_true: int
    y_pred: int
    p_negative: float
    p_neutral: float
    p_positive: float
    condition: TestCondition


def evaluate_predictions(
    expected: Sequence[TextExample],
    predictions: Sequence[PredictionRow],
    *,
    labels: Sequence[int],
    condition: TestCondition,
) -> Metrics:
    """Ghép dự đoán của điều kiện đã chọn với nhãn gốc theo ID, bất kể thứ tự dòng.
    Báo lỗi nếu ID bị thiếu, thừa hoặc trùng trước khi tính điểm."""
    raise NotImplementedError
