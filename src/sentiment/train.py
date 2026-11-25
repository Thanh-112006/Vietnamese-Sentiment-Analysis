from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from sentiment.data.loaders import TextExample, load_examples_csv
from sentiment.data.noise import TestCondition
from sentiment.evaluate import PredictionRow
from sentiment.models.base import BaseSentimentModel
from sentiment.predict import Prediction, predict_texts
from sentiment.preprocess.segment import prepare_model_texts
from sentiment.utils.config import Config


def _create_model(model_name: str) -> BaseSentimentModel:
    if model_name == "tfidf_lr":
        from sentiment.models.tfidf_lr import TfidfLogisticRegression

        return TfidfLogisticRegression()
    if model_name == "bilstm":
        from sentiment.models.bilstm import BiLSTMClassifier

        return BiLSTMClassifier()
    if model_name == "phobert":
        from sentiment.models.phobert import PhoBERTClassifier

        return PhoBERTClassifier()
    raise ValueError(f"Mô hình chưa được hỗ trợ: {model_name!r}.")


def train_model(
    config: Config, train_examples: Sequence[TextExample], *, dev_examples: Sequence[TextExample] | None = None
) -> BaseSentimentModel:
    """Trích text và label_id từ train cùng tập dev tùy chọn để gọi fit()."""
    use_segment = config["use_segment"]
    model = _create_model(config["model_name"])
    train_texts = prepare_model_texts((row.text for row in train_examples), use_segment=use_segment)
    train_labels = [row.label_id for row in train_examples]
    dev_texts = None
    dev_labels = None
    if dev_examples is not None:
        dev_texts = prepare_model_texts((row.text for row in dev_examples), use_segment=use_segment)
        dev_labels = [row.label_id for row in dev_examples]
    model.fit(train_texts, train_labels, dev_texts=dev_texts, dev_labels=dev_labels)
    return model


def build_prediction_rows(
    examples: Sequence[TextExample], predictions: Sequence[Prediction], *, condition: TestCondition
) -> list[PredictionRow]:
    """Ghép ID và nhãn gốc với dự đoán cùng vị trí để xuất predictions.csv.
    Kiểm tra số dự đoán bằng số mẫu và ghi xác suất theo ID nhãn 0, 1, 2."""
    return [
        PredictionRow(
            id=example.id,
            y_true=example.label_id,
            y_pred=prediction["label"],
            p_negative=prediction["probabilities"][0],
            p_neutral=prediction["probabilities"][1],
            p_positive=prediction["probabilities"][2],
            condition=condition,
        )
        for example, prediction in zip(examples, predictions, strict=True)
    ]


def predict_condition(
    model: BaseSentimentModel, config: Config, *, condition: TestCondition
) -> list[PredictionRow]:
    """Dự đoán trên cột text của đúng điều kiện test bằng cấu hình của mô hình."""
    if condition not in get_args(TestCondition):
        raise ValueError(f"Điều kiện test không hợp lệ: {condition!r}.")
    path = Path(config["paths"]["noisy_data"]) / f"{condition}.csv"
    examples = load_examples_csv(path, split="test")
    predictions = predict_texts(
        model, [row.text for row in examples], labels=(0, 1, 2), use_segment=config["use_segment"]
    )
    return build_prediction_rows(examples, predictions, condition=condition)
