from unicodedata import normalize

import pytest

from sentiment.data.loaders import TextExample
from sentiment.preprocess.clean import PreparedTextExample, clean_text, prepare_example


def test_clean_text_normalizes_unicode_and_whitespace():
    raw = normalize("NFD", " \tGiảng viên KHÔNG  đi muộn!\n🙂  ")

    assert clean_text(raw) == "Giảng viên KHÔNG đi muộn! 🙂"


def test_clean_text_lowercases_when_enabled():
    assert clean_text(" \tĐại HỌC. ", lowercase=True) == "đại học."


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_clean_text_returns_empty_string_for_blank_input(text):
    assert clean_text(text) == ""


@pytest.mark.parametrize(
    ("lowercase", "expected"), [(False, "Đại học KHÔNG xa."), (True, "đại học không xa.")]
)
def test_prepare_example_preserves_raw_text_id_and_label(lowercase, expected):
    raw = normalize("NFD", " \tĐại  học KHÔNG xa. ")
    original = TextExample("test_000001", raw, 1)

    assert prepare_example(original, lowercase=lowercase) == PreparedTextExample(
        id="test_000001", text_raw=raw, text=expected, label_id=1
    )
