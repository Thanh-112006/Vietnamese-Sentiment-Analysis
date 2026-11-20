from types import MappingProxyType

TEENCODE_DICTIONARY_VERSION = "1"
TEENCODE_REPLACEMENTS = MappingProxyType(
    {
        "bây giờ": "bi h",
        "bình thường": "bth",
        "biết": "bít",
        "được": "đc",
        "không": "ko",
        "rồi": "rùi",
        "giảng viên": "gv",
        "sinh viên": "sv",
        "bài tập": "bt",
        "thời gian": "tg",
        "ví dụ": "vd",
        "với": "vs",
        "cũng": "cx",
        "nhiều": "nhìu",
        "trước": "trc",
        "hiểu": "hỉu",
        "luôn": "lun",
        "thích": "thik",
        "mình": "mk",
        "người": "ng",
        "nhưng": "nhg",
    }
)


def normalize_teencode(text: str) -> str:
    """Chuyển các dạng teencode đã biết về dạng tiếng Việt chuẩn."""
    raise NotImplementedError
