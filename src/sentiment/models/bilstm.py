from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sentiment.models.base import BaseSentimentModel


class BiLSTMClassifier(BaseSentimentModel):
    """Mô hình BiLSTM với LSTM tự viết hoặc nn.LSTM."""

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        dev_texts: Sequence[str] | None = None,
        dev_labels: Sequence[int] | None = None,
    ) -> None:
        """Huấn luyện trên tập train với tập dev tùy chọn để đánh giá.
        Truyền dev_texts và dev_labels cùng nhau hoặc để cả hai là None."""
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.floating]:
        """Trả ma trận (n, 3) với các cột theo ID nhãn 0, 1, 2."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Lưu checkpoint của mô hình."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BiLSTMClassifier":
        """Khôi phục mô hình từ checkpoint."""
        raise NotImplementedError
