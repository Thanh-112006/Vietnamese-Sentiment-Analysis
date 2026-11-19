from collections.abc import Sequence

from sentiment.data.loaders import TextExample
from sentiment.data.noise import TestCondition
from sentiment.evaluate import PredictionRow
from sentiment.models.base import BaseSentimentModel
from sentiment.predict import Prediction
from sentiment.utils.config import Config


def train_model(
    config: Config, train_examples: Sequence[TextExample], *, dev_examples: Sequence[TextExample] | None = None
) -> BaseSentimentModel:
    """Trích text và label_id từ train cùng tập dev tùy chọn để gọi fit()."""
    raise NotImplementedError


def build_prediction_rows(
    examples: Sequence[TextExample], predictions: Sequence[Prediction], *, condition: TestCondition
) -> list[PredictionRow]:
    """Ghép ID và nhãn gốc với dự đoán cùng vị trí để xuất predictions.csv.
    Kiểm tra số dự đoán bằng số mẫu và ghi xác suất theo ID nhãn 0, 1, 2."""
    raise NotImplementedError
