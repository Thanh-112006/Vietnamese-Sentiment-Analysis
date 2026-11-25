import argparse
from pathlib import Path

from sentiment.data.loaders import load_examples_csv
from sentiment.train import train_model
from sentiment.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy một thí nghiệm theo cấu hình.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, base_path=Path("configs/base.yaml"))
    processed = Path(config["paths"]["processed_data"])
    train_examples = load_examples_csv(processed / "train.csv", split="train")
    dev_examples = load_examples_csv(processed / "dev.csv", split="dev")
    train_model(config, train_examples, dev_examples=dev_examples)


if __name__ == "__main__":
    main()
