import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị các tập train, dev và test cố định.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    pass


if __name__ == "__main__":
    main()
