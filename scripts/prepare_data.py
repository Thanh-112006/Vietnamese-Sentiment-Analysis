import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

from sentiment.data.artifacts import publish_dataset as _publish
from sentiment.data.artifacts import sha256_file as _sha256
from sentiment.data.loaders import TextExample, load_uit_vsfc
from sentiment.data.split import SplitValidationReport, validate_splits
from sentiment.preprocess.clean import prepare_example
from sentiment.utils.config import load_config

_EXPECTED_COUNTS = MappingProxyType({"train": 11_426, "dev": 1_583, "test": 3_166})
_SOURCE_FILES = ("sents.txt", "sentiments.txt")


def _hash_sources(source: Path) -> dict[str, dict[str, str]]:
    return {
        split: {f"{split}/{name}": _sha256(source / split / name) for name in _SOURCE_FILES}
        for split in _EXPECTED_COUNTS
    }


def _write_split_csv(
    path: Path, split: str, examples: Sequence[TextExample], *, lowercase: bool
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "split", "text_raw", "text", "label_id"))
        for example in examples:
            prepared = prepare_example(example, lowercase=lowercase)
            writer.writerow((prepared.id, split, prepared.text_raw, prepared.text, prepared.label_id))


def prepare_data(
    source: str | Path,
    output: str | Path,
    *,
    lowercase: bool,
    expected_counts: Mapping[str, int] | None = _EXPECTED_COUNTS,
) -> SplitValidationReport:
    """Chuẩn bị ba tập UIT-VSFC và cho phép dùng số mẫu khác khi kiểm thử."""
    source, output = Path(source), Path(output)
    source_hashes = _hash_sources(source)
    splits = {
        "train": load_uit_vsfc(source, split="train"),
        "dev": load_uit_vsfc(source, split="dev"),
        "test": load_uit_vsfc(source, split="test"),
    }
    report = validate_splits(splits, expected_counts=expected_counts)
    if _hash_sources(source) != source_hashes:
        raise ValueError(f"Dữ liệu nguồn trong {source} đã thay đổi trong lúc đọc.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temp_dir:
        staging = Path(temp_dir) / "output"
        staging.mkdir()
        manifest_splits = {}
        for split, examples in splits.items():
            csv_path = staging / f"{split}.csv"
            _write_split_csv(csv_path, split, examples, lowercase=lowercase)
            manifest_splits[split] = {
                **report["splits"][split],
                "source_sha256": source_hashes[split],
                "processed_file": {"path": csv_path.name, "sha256": _sha256(csv_path)},
            }

        manifest = {
            "dataset": "UIT-VSFC",
            "version": "1.0",
            "source": "https://sites.google.com/uit.edu.vn/uit-nlp/datasets",
            "labels": ["negative", "neutral", "positive"],
            "id_format": "{split}_{line_number:06d}",
            "first_line_number": 1,
            "preprocess": {
                "unicode_normalization": "NFC",
                "collapse_whitespace": True,
                "lowercase": lowercase,
                "use_segment": False,
            },
            "duplicate_check": ["NFC", "collapse_whitespace", "casefold"],
            "splits": manifest_splits,
            "cross_split_overlaps": report["cross_split_overlaps"],
        }
        with (staging / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _publish(staging, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị các tập train, dev và test cố định.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    preprocess = load_config(args.config).get("preprocess", {})
    if not isinstance(preprocess, dict) or not isinstance(preprocess.get("lowercase", False), bool):
        parser.error("Cấu hình preprocess.lowercase phải là true hoặc false.")
    report = prepare_data(args.input, args.output, lowercase=preprocess.get("lowercase", False))
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['num_samples']} mẫu")
    print(f"Đã chuẩn bị dữ liệu và manifest tại {args.output}")


if __name__ == "__main__":
    main()
