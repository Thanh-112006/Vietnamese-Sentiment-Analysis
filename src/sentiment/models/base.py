from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseSentimentModel(ABC):
    """fit và predict_proba nhận văn bản runner đã chuẩn bị theo use_segment.

    Mô hình truyền trực tiếp các chuỗi này vào vectorizer hoặc tokenizer của mình.
    """

    @abstractmethod
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

    @abstractmethod
    def predict_proba(self, texts):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path):
        raise NotImplementedError
