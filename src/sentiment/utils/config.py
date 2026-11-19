from pathlib import Path
from typing import Any

Config = dict[str, Any]


def load_config(path: Path) -> Config:
    """Nạp cấu hình YAML từ đường dẫn chỉ định."""
    raise NotImplementedError
