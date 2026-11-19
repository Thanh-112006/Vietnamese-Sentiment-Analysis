import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh tác động của tách từ và nhiễu.")
    parser.add_argument("--config", type=Path, required=True)
    parser.parse_args()
    pass


if __name__ == "__main__":
    main()
