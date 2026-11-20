import pytest

from sentiment.data.loaders import TextExample, load_uit_vsfc


def _write_split(root, split, sentences, labels):
    split_dir = root / split
    split_dir.mkdir()
    (split_dir / "sents.txt").write_bytes(sentences.encode("utf-8-sig"))
    (split_dir / "sentiments.txt").write_bytes(labels.encode("utf-8-sig"))


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("use_string_path", [False, True])
def test_load_uit_vsfc_pairs_lines_and_preserves_raw_text(tmp_path, split, use_string_path):
    _write_split(tmp_path, split, "  Giảng viên rất tốt.  \r\nDòng\u2028thứ hai", " 2 \r\n1")
    source = str(tmp_path) if use_string_path else tmp_path

    assert load_uit_vsfc(source, split=split) == [
        TextExample(f"{split}_000001", "  Giảng viên rất tốt.  ", 2),
        TextExample(f"{split}_000002", "Dòng\u2028thứ hai", 1),
    ]


@pytest.mark.parametrize("labels", ["0\n", "0\n1\n2\n"])
def test_load_uit_vsfc_rejects_unequal_line_counts(tmp_path, labels):
    _write_split(tmp_path, "test", "Câu một\nCâu hai\n", labels)

    with pytest.raises(ValueError, match=r"test.*sents\.txt.*sentiments\.txt"):
        load_uit_vsfc(tmp_path, split="test")


@pytest.mark.parametrize("label", ["3", "positive", ""])
def test_load_uit_vsfc_rejects_invalid_label_with_line_number(tmp_path, label):
    _write_split(tmp_path, "test", "Câu một\nCâu hai\n", f"0\n{label}\n")

    with pytest.raises(ValueError, match=r"sentiments\.txt:2"):
        load_uit_vsfc(tmp_path, split="test")


def test_load_uit_vsfc_rejects_blank_sentence_with_line_number(tmp_path):
    _write_split(tmp_path, "test", "Câu một\n \t\n", "0\n1\n")

    with pytest.raises(ValueError, match=r"sents\.txt:2"):
        load_uit_vsfc(tmp_path, split="test")


def test_load_uit_vsfc_rejects_empty_split(tmp_path):
    _write_split(tmp_path, "test", "", "")

    with pytest.raises(ValueError, match="không có mẫu"):
        load_uit_vsfc(tmp_path, split="test")


def test_load_uit_vsfc_reports_missing_labels_file(tmp_path):
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    (split_dir / "sents.txt").write_text("Câu một\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        load_uit_vsfc(tmp_path, split="test")

    assert error.value.filename == str(split_dir / "sentiments.txt")


def test_load_uit_vsfc_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        load_uit_vsfc(tmp_path, split="validation")
