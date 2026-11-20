import argparse
import csv
import json
from collections.abc import Sequence
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from sentiment.data.artifacts import publish_dataset, sha256_file
from sentiment.data.loaders import TextExample
from sentiment.data.noise import generate_noisy_datasets
from sentiment.preprocess.teencode import TEENCODE_DICTIONARY_VERSION, TEENCODE_REPLACEMENTS
from sentiment.utils.config import load_config

_CSV_COLUMNS = ("id", "split", "text", "label_id")


def _read_locked_test(source: Path) -> tuple[list[TextExample], str]:
    manifest_path = source.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        test = manifest["splits"]["test"]
        expected_count = test["num_samples"]
        expected_csv = test["processed_file"]["path"]
        expected_hash = test["processed_file"]["sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Manifest test không hợp lệ trong {manifest_path}.") from exc

    if source.name != expected_csv:
        raise ValueError(f"Tệp {source.name} không khớp tệp test trong {manifest_path}.")
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError(f"Số mẫu test trong {manifest_path} phải là số nguyên dương.")

    content = source.read_bytes()
    source_hash = sha256(content).hexdigest()
    if source_hash != expected_hash:
        raise ValueError(f"SHA-256 của {source} không khớp {manifest_path}.")

    reader = csv.DictReader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    header = reader.fieldnames or []
    if len(header) != len(set(header)) or not set(_CSV_COLUMNS).issubset(header):
        raise ValueError(f"{source} cần có các cột id, split, text và label_id.")

    examples: list[TextExample] = []
    seen_ids: set[str] = set()
    for row in reader:
        if None in row or any(row[column] is None for column in _CSV_COLUMNS):
            raise ValueError(f"Sai số cột tại {source}:{reader.line_num}.")
        sample_id, split, text, label = (row[column] for column in _CSV_COLUMNS)
        if split != "test" or label not in ("0", "1", "2") or not text.strip():
            raise ValueError(f"Sai split, nhãn hoặc văn bản rỗng tại {source}:{reader.line_num}.")
        if not sample_id.strip() or sample_id in seen_ids:
            raise ValueError(f"ID trống hoặc trùng tại {source}:{reader.line_num}.")
        seen_ids.add(sample_id)
        examples.append(TextExample(sample_id, text, int(label)))

    if len(examples) != expected_count:
        raise ValueError(f"{source} có {len(examples)} mẫu, manifest yêu cầu {expected_count}.")
    return examples, source_hash


def _write_variant(path: Path, examples: Sequence[TextExample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_COLUMNS)
        writer.writerows((example.id, "test", example.text, example.label_id) for example in examples)


def make_noisy_testset(
    source: str | Path, output: str | Path, *, seed: int, teencode_rate: float
) -> None:
    """Xuất bốn điều kiện test và manifest từ một tập test đã khóa."""
    source, output = Path(source), Path(output)
    examples, source_hash = _read_locked_test(source)
    datasets = generate_noisy_datasets(examples, seed=seed, teencode_rate=teencode_rate)
    expected_ids_and_labels = [(example.id, example.label_id) for example in examples]

    dictionary_bytes = json.dumps(
        dict(TEENCODE_REPLACEMENTS), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".noisy-", dir=output.parent) as temp_dir:
        staging = Path(temp_dir) / "output"
        staging.mkdir()
        variants = {}
        for condition, rows in datasets.items():
            if [(row.id, row.label_id) for row in rows] != expected_ids_and_labels:
                raise ValueError(f"Điều kiện {condition} bị lệch số mẫu, ID, nhãn hoặc thứ tự.")

            path = staging / f"{condition}.csv"
            _write_variant(path, rows)
            changed_rows = sum(row.text != original.text for row, original in zip(rows, examples, strict=True))
            variants[condition] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "num_samples": len(rows),
                "changed_rows": changed_rows,
                "changed_rate": changed_rows / len(examples),
            }

        manifest = {
            "source": {"path": source.name, "sha256": source_hash, "num_samples": len(examples)},
            "noise": {
                "seed": seed,
                "teencode_rate": teencode_rate,
                "teencode_dictionary_version": TEENCODE_DICTIONARY_VERSION,
                "teencode_dictionary_sha256": sha256(dictionary_bytes).hexdigest(),
            },
            "variants": variants,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        publish_dataset(staging, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh các biến thể teencode và không dấu của tập test.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    try:
        noise = load_config(args.config).get("noise")
        if not isinstance(noise, dict):
            raise ValueError("Cần mục noise trong tệp cấu hình.")
        seed = noise.get("seed")
        rate = noise.get("teencode_rate")
        if type(seed) is not int:
            raise ValueError("noise.seed phải là số nguyên.")
        if type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError("noise.teencode_rate phải là số trong khoảng [0, 1].")
        make_noisy_testset(args.input, args.output, seed=seed, teencode_rate=rate)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Đã xuất bốn tập test và manifest vào {args.output}")


if __name__ == "__main__":
    main()
