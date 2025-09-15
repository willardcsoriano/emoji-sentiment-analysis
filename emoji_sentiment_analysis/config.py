# emoji_sentiment_analysis/config.py
# ---------------------------------------------------------------------
# Central place for:
#   - Project paths (data, models, reports)
#   - Environment variables (.env)
#   - Global knobs (SEED, column names)
#   - Helpers to create folders and set up logging
#
# Import this module from any script to avoid hard-coding paths.
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# ---------- Project root ------------------------------------------------
# Find the project’s root folder (the repo top-level).
# We resolve this file (__file__), go up two directories:
#   emoji_sentiment_analysis/config.py -> emoji_sentiment_analysis/ -> <repo root>
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]

# ---------- Environment variables --------------------------------------
# Load variables from "<repo>/.env" (if it exists). Doing this *after*
# PROJ_ROOT is known ensures we point at the correct file even when
# running from notebooks or subfolders.
load_dotenv(dotenv_path=PROJ_ROOT / ".env")

# ---------- Paths (with optional .env overrides) -----------------------
# You can override DATA_DIR or MODELS_DIR in .env, e.g. to store big files
# on another drive:
#   DATA_DIR=D:/nlp_data
#   MODELS_DIR=D:/models_cache
#
# Path(...) ensures we always end up with a Path object.
DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJ_ROOT / "data")).resolve()

# Subfolders under data/ for a standard ML workflow
RAW_DATA_DIR: Path = DATA_DIR / "raw"         # original/unmodified files
INTERIM_DATA_DIR: Path = DATA_DIR / "interim" # temp/cleaned/intermediate
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # final ML-ready tables
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"    # third-party inputs

# Where trained models/artifacts live (checkpoints, vectorizers, etc.)
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", PROJ_ROOT / "models")).resolve()

# Reports/figures for exports, plots, metrics images, etc.
REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------- Global knobs ------------------------------------------------
# Keep project-wide constants here so every script agrees on them.
SEED: int = int(os.getenv("SEED", "42"))                 # reproducibility
TEXT_COL: str = os.getenv("TEXT_COL", "text")            # input text column
TARGET_COL: str = os.getenv("TARGET_COL", "label")       # target label column

# ---------- Helpers -----------------------------------------------------
def ensure_dirs() -> None:
    """
    Create the standard folder structure if it doesn't exist.
    Call this once at the start of entry-point scripts (train/predict).
    """
    for p in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)

def init_logging(use_tqdm: bool = True) -> None:
    """
    Configure Loguru so logs play nicely with progress bars.

    Why not configure logging at import?
    - Config modules get imported everywhere; changing global logging
      on import can cause surprising side effects or duplicate handlers.

    Instead, call init_logging() explicitly in your main scripts.
    """
    logger.remove()  # remove default handler to avoid duplicate logs

    if use_tqdm:
        try:
            # Local import keeps tqdm optional (no hard dependency)
            from tqdm import tqdm

            # Route log lines via tqdm.write so they don't break progress bars
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        except ModuleNotFoundError:
            # Fallback to standard stderr if tqdm isn't installed
            logger.add(sys.stderr, colorize=True)
    else:
        logger.add(sys.stderr, colorize=True)

# ---------- Re-export for clean imports --------------------------------
# __all__ defines the public API of this module. Tools like linters/IDEs
# and "from config import *" will respect this list.
__all__ = [
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "SEED",
    "TEXT_COL",
    "TARGET_COL",
    "ensure_dirs",
    "init_logging",
]
