import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Tính SHA-256 trên nội dung file."""
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_dataset(staging: Path, output: Path) -> None:
    """Ghi một bộ dữ liệu đã chuẩn bị, với manifest được công bố sau cùng."""
    output.mkdir(parents=True, exist_ok=True)
    staged_files = {path.name: path for path in staging.iterdir()}
    existing = {path.name: path for path in output.iterdir() if path.name != ".gitkeep"}
    if existing:
        if existing.keys() == staged_files.keys() and all(
            existing[name].is_file() and sha256_file(existing[name]) == sha256_file(path)
            for name, path in staged_files.items()
        ):
            return
        raise FileExistsError(f"Thư mục {output} có dữ liệu khác. Hãy chọn --output mới.")

    published: list[Path] = []
    try:
        for name in sorted(staged_files, key=lambda name: (name == "manifest.json", name)):
            destination = output / name
            # Giữ chỗ độc quyền trước khi thay thế để tránh ghi đè khi chạy đồng thời.
            with destination.open("xb"):
                published.append(destination)
            staged_files[name].replace(destination)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
