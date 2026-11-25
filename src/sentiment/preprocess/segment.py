from collections.abc import Iterable


def segment_text(text: str, *, backend: str = "underthesea") -> str:
    """Tách từ tiếng Việt, giữ cách viết của văn bản đầu vào."""
    if backend != "underthesea":
        raise ValueError(f"Bộ tách từ chưa được hỗ trợ: {backend!r}.")
    from underthesea import word_tokenize

    return word_tokenize(text, format="text", use_token_normalize=False)


def prepare_model_texts(texts: Iterable[str], *, use_segment: bool) -> list[str]:
    """Chuẩn bị cột text theo thứ tự đầu vào, ngay trước fit hoặc predict_proba."""
    if not isinstance(use_segment, bool):
        raise ValueError("use_segment phải là true hoặc false.")
    if use_segment:
        return [segment_text(text) for text in texts]
    return list(texts)
