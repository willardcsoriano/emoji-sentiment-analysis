# src/emoji_sentiment_analysis/config.py

from __future__ import annotations
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------
# Project Root Resolution (Docker-Proofed)
# ---------------------------------------------------------------------
# Check if we are running inside Cloud Run or a Docker container
IS_DOCKER = os.environ.get('K_SERVICE') is not None or os.path.exists('/.dockerenv')

if IS_DOCKER:
    # In Docker, we enforce the explicit absolute paths based on WORKDIR /app
    REPO_ROOT: Path = Path("/app")
    SRC_ROOT: Path = REPO_ROOT / "src"
else:
    # Locally, resolve dynamically based on where this file lives
    INTERNAL_PACKAGE_DIR: Path = Path(__file__).resolve().parent
    SRC_ROOT: Path = INTERNAL_PACKAGE_DIR.parent
    REPO_ROOT: Path = SRC_ROOT.parent

# Load environment variables
load_dotenv(dotenv_path=REPO_ROOT / ".env")

# ---------------------------------------------------------------------
# Data & Model Directories
# ---------------------------------------------------------------------
DATA_DIR: Path = SRC_ROOT / "data"
MODELS_DIR: Path = SRC_ROOT / "models"

RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
LOGS_DIR: Path = DATA_DIR / "logs"

MODEL_ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------
# Reporting, Notebooks, and Scripts
# ---------------------------------------------------------------------
REPORTS_DIR: Path = SRC_ROOT / "reports" 
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = SRC_ROOT / "notebooks"
SCRIPTS_DIR: Path = SRC_ROOT / "scripts"

PROJ_ROOT = REPO_ROOT 

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
    """Create the standard project directory structure if missing."""
    dirs = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR, LOGS_DIR, MODELS_DIR,
        MODEL_ARTIFACTS_DIR, FIGURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Logging Initialization
# ---------------------------------------------------------------------
def init_logging(use_tqdm: bool = True) -> None:
    """Configure Loguru logging."""
    logger.remove()
    logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "PROJ_ROOT", "SRC_ROOT", "DATA_DIR", "RAW_DATA_DIR", 
    "INTERIM_DATA_DIR", "PROCESSED_DATA_DIR", "EXTERNAL_DATA_DIR", 
    "LOGS_DIR", "MODELS_DIR", "MODEL_ARTIFACTS_DIR", "REPORTS_DIR", 
    "FIGURES_DIR", "NOTEBOOKS_DIR", "SCRIPTS_DIR", "SEED", 
    "TEXT_COL", "TARGET_COL", "ensure_dirs", "init_logging",
]