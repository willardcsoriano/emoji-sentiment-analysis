"""
Project Configuration Module
----------------------------

Centralized configuration for:

    - Project root discovery
    - Data + artifact paths
    - Environment variable overrides
    - Global constants
    - Directory bootstrapping
    - Logging initialization

This module assumes the repository root is the parent
directory containing:

    data/
    models/
    notebooks/
    reports/
    scripts/
    emoji_sentiment_analysis/

All paths resolve relative to that root.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------
# Project Root Resolution
# ---------------------------------------------------------------------
# config.py lives in:
#   emoji_sentiment_analysis/config.py
#
# So repo root is TWO levels up.
# (package → repo root)

PROJ_ROOT: Path = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------

load_dotenv(dotenv_path=PROJ_ROOT / ".env")

# ---------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------

DATA_DIR: Path = Path(
    os.getenv("DATA_DIR", PROJ_ROOT / "data")
).resolve()

RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
LOGS_DIR: Path = DATA_DIR / "logs"

# ---------------------------------------------------------------------
# Model & Artifact Directories
# ---------------------------------------------------------------------

MODELS_DIR: Path = Path(
    os.getenv("MODELS_DIR", PROJ_ROOT / "models")
).resolve()

MODEL_ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------
# Reporting Directories
# ---------------------------------------------------------------------

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------
# Notebook + Script References (optional convenience)
# ---------------------------------------------------------------------

NOTEBOOKS_DIR: Path = PROJ_ROOT / "notebooks"
SCRIPTS_DIR: Path = PROJ_ROOT / "scripts"

# ---------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------

SEED: int = int(os.getenv("SEED", "42"))

TEXT_COL: str = os.getenv("TEXT_COL", "text")
TARGET_COL: str = os.getenv("TARGET_COL", "label")

# ---------------------------------------------------------------------
# Directory Bootstrapping
# ---------------------------------------------------------------------

def ensure_dirs() -> None:
    """
    Create the standard project directory structure if missing.
    """

    dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        LOGS_DIR,
        MODELS_DIR,
        MODEL_ARTIFACTS_DIR,
        FIGURES_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Logging Initialization
# ---------------------------------------------------------------------

def init_logging(use_tqdm: bool = True) -> None:
    """
    Configure Loguru logging.

    Supports tqdm-safe logging if progress bars are used.
    """

    logger.remove()

    if use_tqdm:
        try:
            from tqdm import tqdm
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        except ModuleNotFoundError:
            logger.add(sys.stderr, colorize=True)
    else:
        logger.add(sys.stderr, colorize=True)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "MODEL_ARTIFACTS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "NOTEBOOKS_DIR",
    "SCRIPTS_DIR",
    "SEED",
    "TEXT_COL",
    "TARGET_COL",
    "ensure_dirs",
    "init_logging",
]
