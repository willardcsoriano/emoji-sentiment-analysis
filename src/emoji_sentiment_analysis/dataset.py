# emoji_sentiment_analysis\dataset.py

"""
Dataset Ingestion & Cleaning Pipeline
-------------------------------------

This script operationalizes the deterministic data cleaning
specification defined in Notebooks 1.0 and 1.5.

Responsibilities:

• Load raw datasets
• Standardize schemas
• Remove structural artifacts
• Preserve emoji text exactly
• Validate dataset contracts
• Export cleaned datasets to data/processed/

Outputs:

data/processed/tweets_clean.csv
data/processed/emoji_reference_clean.csv
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from loguru import logger
import typer

from emoji_sentiment_analysis.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TEXT_COL,
    TARGET_COL,
)

app = typer.Typer()


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def validate_tweets(df: pd.DataFrame) -> None:
    """
    Enforces tweet dataset invariants.
    Raises AssertionError if violated.
    """

    assert list(df.columns) == [TARGET_COL, TEXT_COL], \
        f"Tweet schema mismatch: {df.columns}"

    assert df[TARGET_COL].isin([0, 1]).all(), \
        "Labels must be binary (0/1)."

    assert df[TEXT_COL].str.len().gt(0).all(), \
        "Empty text rows detected."


def validate_emoji_reference(df: pd.DataFrame) -> None:
    """
    Enforces emoji reference dataset invariants.
    """

    assert "emoji" in df.columns, \
        "Emoji column missing."

    assert df["emoji"].nunique() == len(df), \
        "Duplicate emojis detected."


# ---------------------------------------------------------------------
# Cleaning Functions
# ---------------------------------------------------------------------

def clean_tweets_dataset(path: Path) -> pd.DataFrame:
    """
    Cleans the tweet sentiment dataset.

    Cleaning operations:

    • Drop redundant index columns
    • Rename schema fields
    • Enforce data types
    • Remove nulls
    • Reset index
    """

    logger.info(f"Loading tweets dataset: {path.name}")

    df = pd.read_csv(path)

    # Drop redundant columns
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Rename schema
    df = df.rename(columns={
        "post": TEXT_COL,
        "tweet": TEXT_COL,
        "text": TEXT_COL,
        "sentiment": TARGET_COL,
        "label": TARGET_COL,
    })

    # Keep only required columns
    df = df[[TARGET_COL, TEXT_COL]]

    # Enforce types
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Remove nulls
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Cleaned tweets shape: {df.shape}")

    # Validate contract
    validate_tweets(df)

    return df


def clean_emoji_reference(path: Path) -> pd.DataFrame:
    """
    Cleans emoji metadata reference dataset.
    """

    logger.info(f"Loading emoji reference: {path.name}")

    df = pd.read_csv(path)

    # Drop redundant index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Rename columns
    df = df.rename(columns={
        "Emoji": "emoji",
        "Unicode codepoint": "unicode_codepoint",
        "Unicode name": "unicode_name",
    })

    # Enforce string types
    for col in df.columns:
        df[col] = df[col].astype(str)

    df = df.reset_index(drop=True)

    logger.info(f"Cleaned emoji reference shape: {df.shape}")

    # Validate contract
    validate_emoji_reference(df)

    return df


# ---------------------------------------------------------------------
# Main Pipeline Entry
# ---------------------------------------------------------------------

@app.command()
def main():
    """
    Executes the full ingestion + cleaning pipeline.
    """

    logger.info("Starting dataset ingestion pipeline...")

    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load + clean tweets dataset
    # -----------------------------------------------------------------

    tweets_path = RAW_DATA_DIR / "1k_data_emoji_tweets_senti_posneg.csv"

    tweets_df = clean_tweets_dataset(tweets_path)

    tweets_out = PROCESSED_DATA_DIR / "tweets_clean.csv"
    tweets_df.to_csv(tweets_out, index=False)

    logger.success(f"Saved cleaned tweets → {tweets_out}")

    # -----------------------------------------------------------------
    # Load + clean emoji reference
    # -----------------------------------------------------------------

    emoji_path = RAW_DATA_DIR / "15_emoticon_data.csv"

    emoji_df = clean_emoji_reference(emoji_path)

    emoji_out = PROCESSED_DATA_DIR / "emoji_reference_clean.csv"
    emoji_df.to_csv(emoji_out, index=False)

    logger.success(f"Saved emoji reference → {emoji_out}")

    # -----------------------------------------------------------------

    logger.success("Dataset ingestion pipeline complete.")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    app()
