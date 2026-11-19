import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy một thí nghiệm theo cấu hình.")
    parser.add_argument("--config", type=Path, required=True)
    parser.parse_args()
    pass


if __name__ == "__main__":
    main()
