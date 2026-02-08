# emoji_sentiment_analysis/config.py

from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------
# Project Root Resolution
# ---------------------------------------------------------------------
INTERNAL_PACKAGE_DIR: Path = Path(__file__).resolve().parent
SRC_ROOT: Path = INTERNAL_PACKAGE_DIR.parent
REPO_ROOT: Path = SRC_ROOT.parent

# Use REPO_ROOT for the .env file
load_dotenv(dotenv_path=REPO_ROOT / ".env")

# ---------------------------------------------------------------------
# Data Directories (Now relative to SRC_ROOT)
# ---------------------------------------------------------------------
DATA_DIR: Path = Path(
    os.getenv("DATA_DIR", SRC_ROOT / "data")
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
    os.getenv("MODELS_DIR", SRC_ROOT / "models")
).resolve()

MODEL_ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------
# Reporting, Notebooks, and Scripts (Inside SRC_ROOT)
# ---------------------------------------------------------------------
REPORTS_DIR: Path = SRC_ROOT / "reports" 
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = SRC_ROOT / "notebooks"
SCRIPTS_DIR: Path = SRC_ROOT / "scripts"

# Update PROJ_ROOT to maintain compatibility
PROJ_ROOT = SRC_ROOT

# ---------------------------------------------------------------------
# Global Constants (The missing pieces!)
# ---------------------------------------------------------------------
SEED: int = int(os.getenv("SEED", "42"))
TEXT_COL: str = os.getenv("TEXT_COL", "text")
TARGET_COL: str = os.getenv("TARGET_COL", "label")

# ---------------------------------------------------------------------
# Directory Bootstrapping
# ---------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create the standard project directory structure if missing."""
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
    """Configure Loguru logging."""
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