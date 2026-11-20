import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.make_noisy_testset import main, make_noisy_testset
from scripts.prepare_data import prepare_data
from sentiment.data.loaders import TextExample
from sentiment.data.noise import apply_teencode, generate_noisy_datasets, remove_accents
from sentiment.preprocess.teencode import TEENCODE_DICTIONARY_VERSION

_CONDITIONS = ("clean", "no_accent", "teencode", "mixed")


def _write_locked_test(directory: Path, examples: list[TextExample]) -> Path:
    directory.mkdir()
    source = directory / "test.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "split", "text_raw", "text", "label_id"))
        writer.writerows(
            (example.id, "test", f"  {example.text}  ", example.text, example.label_id)
            for example in examples
        )
    manifest = {
        "splits": {
            "test": {
                "num_samples": len(examples),
                "processed_file": {"path": source.name, "sha256": sha256(source.read_bytes()).hexdigest()},
            }
        }
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_apply_teencode_preserves_boundaries_case_and_punctuation():
    text = "KHÔNG, được! khôngđược, không_dấu. Bây giờ?"

    assert apply_teencode(text, seed=42, rate=1) == "KO, đc! khôngđược, không_dấu. Bi h?"


def test_remove_accents_preserves_punctuation_and_emoji():
    assert remove_accents("Được, dễ hiểu! ❤️") == "Duoc, de hieu! ❤️"


def test_generate_noisy_datasets_keeps_pairs_and_reuses_teencode():
    examples = [
        TextExample("test_000001", "Không theo kịp được.", 0),
        TextExample("test_000002", "Bình thường.", 1),
        TextExample("test_000003", "Giảng rất rõ.", 2),
    ]
    datasets = generate_noisy_datasets(examples, seed=42, teencode_rate=0.8)

    assert tuple(datasets) == _CONDITIONS
    assert datasets == generate_noisy_datasets(examples, seed=42, teencode_rate=0.8)
    assert datasets["clean"] == examples
    for rows in datasets.values():
        assert [(row.id, row.label_id) for row in rows] == [
            (example.id, example.label_id) for example in examples
        ]
    assert [row.text for row in datasets["mixed"]] == [
        remove_accents(row.text) for row in datasets["teencode"]
    ]

    reversed_datasets = generate_noisy_datasets(examples[::-1], seed=42, teencode_rate=0.8)
    assert all(reversed_datasets[name] == datasets[name][::-1] for name in _CONDITIONS)

    without_teencode = generate_noisy_datasets(examples, seed=42, teencode_rate=0)
    assert without_teencode["teencode"] == without_teencode["clean"]
    assert without_teencode["mixed"] == without_teencode["no_accent"]


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan")])
def test_generate_noisy_datasets_rejects_invalid_rate(rate):
    with pytest.raises(ValueError, match="teencode_rate"):
        generate_noisy_datasets([], seed=42, teencode_rate=rate)


def test_make_noisy_testset_exports_3166_paired_rows_with_reproducible_hashes(tmp_path):
    texts = ("Không theo kịp được.", "Bình thường.", "Giảng rất rõ.")
    examples = [
        TextExample(f"test_{line:06d}", texts[(line - 1) % 3], (line - 1) % 3)
        for line in range(3166, 0, -1)
    ]
    source = _write_locked_test(tmp_path / "processed", examples)
    source_bytes = source.read_bytes()
    output = tmp_path / "noisy"
    output.mkdir()
    (output / ".gitkeep").touch()

    make_noisy_testset(source, output, seed=42, teencode_rate=0.8)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == sha256(source_bytes).hexdigest()
    assert manifest["noise"]["seed"] == 42
    assert manifest["noise"]["teencode_rate"] == 0.8
    assert manifest["noise"]["teencode_dictionary_version"] == TEENCODE_DICTIONARY_VERSION
    assert tuple(manifest["variants"]) == _CONDITIONS

    hashes = {manifest_path.name: sha256(manifest_path.read_bytes()).hexdigest()}
    for condition in _CONDITIONS:
        path = output / f"{condition}.csv"
        metadata = manifest["variants"][condition]
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ["id", "split", "text", "label_id"]
            rows = list(reader)

        assert len(rows) == metadata["num_samples"] == 3166
        assert all(row["split"] == "test" for row in rows)
        assert [(row["id"], int(row["label_id"])) for row in rows] == [
            (example.id, example.label_id) for example in examples
        ]
        changes = sum(row["text"] != example.text for row, example in zip(rows, examples, strict=True))
        assert metadata["changed_rows"] == changes
        assert metadata["changed_rate"] == pytest.approx(changes / 3166)
        if condition == "clean":
            assert changes == 0
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        assert hashes[path.name] == metadata["sha256"]

    make_noisy_testset(source, output, seed=42, teencode_rate=0.8)

    assert {name: sha256((output / name).read_bytes()).hexdigest() for name in hashes} == hashes
    assert source.read_bytes() == source_bytes
    assert (output / ".gitkeep").is_file()


def test_prepare_data_output_can_generate_noisy_testset(tmp_path):
    raw = tmp_path / "raw"
    splits = {
        "train": [("Đã nộp bài.", 1)],
        "dev": [("Tiết học thú vị.", 2)],
        "test": [("  Giảng viên\tKHÔNG  rõ!  ", 0), ("Bình thường.", 1)],
    }
    for split, examples in splits.items():
        directory = raw / split
        directory.mkdir(parents=True)
        (directory / "sents.txt").write_text(
            "".join(f"{text}\n" for text, _ in examples), encoding="utf-8"
        )
        (directory / "sentiments.txt").write_text(
            "".join(f"{label}\n" for _, label in examples), encoding="utf-8"
        )

    processed = tmp_path / "processed"
    expected_counts = {split: len(examples) for split, examples in splits.items()}
    prepare_data(raw, processed, lowercase=False, expected_counts=expected_counts)
    source = processed / "test.csv"
    source_bytes = source.read_bytes()
    clean_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    test_hash = clean_manifest["splits"]["test"]["processed_file"]["sha256"]
    assert test_hash == sha256(source_bytes).hexdigest()

    output = tmp_path / "noisy"
    make_noisy_testset(source, output, seed=42, teencode_rate=1)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {"path": "test.csv", "sha256": test_hash, "num_samples": 2}
    expected_texts = {
        "clean": ["Giảng viên KHÔNG rõ!", "Bình thường."],
        "no_accent": ["Giang vien KHONG ro!", "Binh thuong."],
        "teencode": ["Gv KO rõ!", "Bth."],
        "mixed": ["Gv KO ro!", "Bth."],
    }
    assert set(manifest["variants"]) == set(expected_texts)
    for condition, texts in expected_texts.items():
        path = output / f"{condition}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert [(row["id"], row["split"], row["label_id"]) for row in rows] == [
            ("test_000001", "test", "0"),
            ("test_000002", "test", "1"),
        ]
        assert [row["text"] for row in rows] == texts
        assert manifest["variants"][condition]["sha256"] == sha256(path.read_bytes()).hexdigest()

    assert source.read_bytes() == source_bytes


def test_make_noisy_testset_rejects_source_with_wrong_hash(tmp_path):
    source = _write_locked_test(
        tmp_path / "processed", [TextExample("test_000001", "Không rõ.", 0)]
    )
    with source.open("ab") as handle:
        handle.write(b"\n")
    output = tmp_path / "noisy"

    with pytest.raises(ValueError, match="SHA-256"):
        make_noisy_testset(source, output, seed=42, teencode_rate=0.8)

    assert not output.exists()


def test_cli_uses_seed_and_rate_from_config(tmp_path, monkeypatch):
    source = _write_locked_test(
        tmp_path / "processed", [TextExample("test_000001", "Không rõ.", 0)]
    )
    output = tmp_path / "noisy"
    config = tmp_path / "config.yaml"
    config.write_text("noise:\n  seed: 17\n  teencode_rate: 0\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["make_noisy_testset.py", "--input", str(source), "--output", str(output), "--config", str(config)]
    )

    main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["noise"]["seed"] == 17
    assert manifest["noise"]["teencode_rate"] == 0
    assert (output / "teencode.csv").read_bytes() == (output / "clean.csv").read_bytes()
    assert (output / "mixed.csv").read_bytes() == (output / "no_accent.csv").read_bytes()
