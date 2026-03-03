"""
make_dataset.py
───────────────
Entry-point for the data acquisition & preprocessing pipeline.

Usage:
    python -m src.data.make_dataset                   # use config.yaml defaults
    python -m src.data.make_dataset --raw path/to.csv # override raw path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Allow running as __main__ from any working directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import DataPreprocessor  # noqa: E402
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline.")
    parser.add_argument(
        "--raw",
        type=str,
        default=None,
        help="Path to raw CSV. Overrides config.yaml value.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output parquet path. Overrides config.yaml value.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "config.yaml"),
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = args.raw or (ROOT / cfg["paths"]["raw_data"])
    out_path = args.out or (ROOT / cfg["paths"]["processed_data"])

    logger.info(f"Starting pipeline | raw={raw_path} | out={out_path}")

    preprocessor = DataPreprocessor(cfg)
    df_raw = preprocessor.load(raw_path)
    df_clean = preprocessor.clean(df_raw)
    df_processed = preprocessor.transform(df_clean)
    preprocessor.save(df_processed, out_path)

    logger.success(
        f"Pipeline complete. Saved {len(df_processed):,} rows × "
        f"{len(df_processed.columns)} cols → {out_path}"
    )


if __name__ == "__main__":
    main()
