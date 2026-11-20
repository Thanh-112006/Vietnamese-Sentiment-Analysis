from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

Config = dict[str, Any]


def _validate_config(config: Any, path: Path) -> Config:
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình trong {path} phải là một mapping YAML.")
    if any(not isinstance(key, str) for key in config):
        raise ValueError(f"Khóa cấu hình cấp đầu trong {path} phải là chuỗi.")
    return config


def _read_yaml(path: Path) -> Config:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML không hợp lệ trong {path}: {exc}") from exc
    return _validate_config(config, path)


def _merge_configs(base: Config, override: Config) -> Config:
    merged = base.copy()
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_configs(base_value, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, *, base_path: str | Path | None = None) -> Config:
    """Nạp cấu hình YAML và tùy chọn ghi đè lên cấu hình nền.

    Gộp các mapping lồng nhau. Danh sách và giá trị đơn được thay thế.
    """
    config = _read_yaml(Path(path))
    if base_path is None:
        return config
    return _merge_configs(_read_yaml(Path(base_path)), config)


def save_config(config: Config, path: str | Path) -> None:
    """Lưu cấu hình dưới dạng YAML UTF-8 và tạo thư mục cha khi cần."""
    destination = Path(path)
    _validate_config(config, destination)
    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        temp_path.replace(destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
