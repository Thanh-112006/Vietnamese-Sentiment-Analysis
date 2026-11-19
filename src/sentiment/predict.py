from collections.abc import Sequence
from typing import TypedDict

from sentiment.models.base import BaseSentimentModel


class Prediction(TypedDict):
    label: int
    confidence: float
    probabilities: dict[int, float]


def predict_texts(
    model: BaseSentimentModel, texts: Sequence[str], *, labels: Sequence[int]
) -> list[Prediction]:
    """Trả nhãn, độ tin cậy và phân phối xác suất cho từng văn bản."""
    raise NotImplementedError
