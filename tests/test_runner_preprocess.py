import csv
import sys
from unittest.mock import Mock, call

import pytest
import yaml

from scripts import run_experiment
from sentiment.models.base import BaseSentimentModel
from sentiment.preprocess import segment
from sentiment.train import predict_condition
from sentiment.utils.config import load_config


def _write_examples(path, split, rows, *, processed):
    columns = ["id", "split", "text", "label_id"]
    if processed:
        columns.insert(2, "text_raw")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample_id, text, label in rows:
            row = {"id": sample_id, "split": split, "text": text, "label_id": label}
            if processed:
                row["text_raw"] = f"  {text}  "
            writer.writerow(row)


@pytest.mark.parametrize("use_segment", [False, True])
def test_runner_passes_csv_text_to_fit_and_predict_once_per_sample(tmp_path, monkeypatch, use_segment):
    processed = tmp_path / "processed"
    noisy = tmp_path / "noisy"
    processed.mkdir()
    noisy.mkdir()
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump({
        "model_name": "tfidf_lr",
        "use_segment": use_segment,
        "paths": {"processed_data": str(processed), "noisy_data": str(noisy)},
    }), encoding="utf-8")

    train = [("train_000001", "Giảng viên tốt.", 2), ("train_000002", "Chưa ổn.", 0)]
    dev = [("dev_000001", "Cần cải thiện.", 0)]
    _write_examples(processed / "train.csv", "train", train, processed=True)
    _write_examples(processed / "dev.csv", "dev", dev, processed=True)

    # Đầu ra giả có dấu hiệu riêng để phát hiện cả tách hai lần và bỏ qua kết quả tách.
    tokenizer = Mock(side_effect=lambda text: f"seg({text})")
    monkeypatch.setattr(segment, "segment_text", tokenizer)
    model = Mock(spec=BaseSentimentModel)
    monkeypatch.setattr("sentiment.models.tfidf_lr.TfidfLogisticRegression", lambda: model)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(config_path)])
    run_experiment.main()

    def expected_texts(rows):
        return [f"seg({text})" if use_segment else text for _, text, _ in rows]

    model.fit.assert_called_once_with(
        expected_texts(train), [2, 0], dev_texts=expected_texts(dev), dev_labels=[0]
    )
    assert tokenizer.call_args_list == ([call(text) for _, text, _ in train + dev] if use_segment else [])

    config = load_config(config_path, base_path="configs/base.yaml")
    conditions = {
        "clean": ("Không tốt!", "Được."),
        "no_accent": ("Khong tot!", "Duoc."),
        "teencode": ("Ko tốt!", "Đc."),
        "mixed": ("Ko tot!", "Dc."),
    }
    model.predict_proba.return_value = [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]
    for condition, (first, second) in conditions.items():
        # Thứ tự ID trong tệp cố ý khác thứ tự từ điển.
        rows = [("test_000002", first, 0), ("test_000001", second, 2)]
        _write_examples(noisy / f"{condition}.csv", "test", rows, processed=False)
        tokenizer.reset_mock()
        model.predict_proba.reset_mock()
        predictions = predict_condition(model, config, condition=condition)
        assert tokenizer.call_args_list == ([call(first), call(second)] if use_segment else [])
        model.predict_proba.assert_called_once_with(expected_texts(rows))
        assert [(p["id"], p["y_true"], p["y_pred"], p["condition"]) for p in predictions] == [
            ("test_000002", 0, 0, condition), ("test_000001", 2, 2, condition)
        ]


def test_segment_text_uses_underthesea_without_token_normalization(monkeypatch):
    backend = Mock()
    backend.word_tokenize.return_value = "giảng_viên ko đến ."
    monkeypatch.setitem(sys.modules, "underthesea", backend)
    assert segment.segment_text("giảng viên ko đến.") == "giảng_viên ko đến ."
    backend.word_tokenize.assert_called_once_with(
        "giảng viên ko đến.", format="text", use_token_normalize=False
    )


def test_underthesea_joins_compound_and_preserves_teencode():
    pytest.importorskip("underthesea")
    tokens = segment.segment_text("giảng viên ko đến.").split()
    assert "giảng_viên" in tokens
    assert "ko" in tokens
