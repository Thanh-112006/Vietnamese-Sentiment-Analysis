import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh các biến thể teencode và không dấu của tập test.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    pass


if __name__ == "__main__":
    main()
