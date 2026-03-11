"""
utils.py
────────
Project-wide helper utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config file. Falls back to project root config/config.yaml."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Loaded config from {path}")
    return cfg


def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure loguru based on config.yaml logging section."""
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO")
    fmt = log_cfg.get("format", "{time} | {level} | {name} | {message}")
    log_file = log_cfg.get("file")

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format=fmt,
        colorize=True,
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=fmt, rotation="10 MB")


def get_project_root() -> Path:
    """Return absolute path to project root directory."""
    return Path(__file__).resolve().parents[1]
