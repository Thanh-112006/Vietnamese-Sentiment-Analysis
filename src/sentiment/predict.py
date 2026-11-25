from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from sentiment.models.base import BaseSentimentModel
from sentiment.preprocess.segment import prepare_model_texts


class Prediction(TypedDict):
    """Nhãn dự đoán và xác suất theo mã nhãn cảm xúc."""

    label: int
    confidence: float
    probabilities: dict[int, float]


def predict_texts(
    model: BaseSentimentModel, texts: Sequence[str], *, labels: Sequence[int], use_segment: bool = False
) -> list[Prediction]:
    """Chuẩn bị văn bản theo use_segment rồi trả xác suất theo thứ tự labels."""
    model_texts = prepare_model_texts(texts, use_segment=use_segment)
    if not model_texts:
        return []
    probabilities = np.asarray(model.predict_proba(model_texts))
    if probabilities.shape != (len(model_texts), len(labels)):
        raise ValueError("Kích thước predict_proba phải khớp số câu và số nhãn.")
    predictions: list[Prediction] = []
    for scores in probabilities:
        best = int(scores.argmax())
        predictions.append(
            Prediction(
                label=labels[best],
                confidence=float(scores[best]),
                probabilities={label: float(score) for label, score in zip(labels, scores, strict=True)},
            )
        )
    return predictions
