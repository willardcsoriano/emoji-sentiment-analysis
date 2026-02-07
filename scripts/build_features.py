# scripts/build_features.py

"""
Feature Engineering Pipeline
----------------------------

This script generates the finalized modeling dataset by applying
emoji polarity feature extraction to the cleaned tweet corpus.

Source Input:
    data/processed/tweets_clean.csv

Output:
    data/processed/features_final.csv

Responsibilities:

• Load cleaned tweet dataset
• Extract emoji polarity features
• Append feature columns
• Validate feature integrity
• Persist modeling-ready dataset

No model training occurs here.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from loguru import logger
import typer

from emoji_sentiment_analysis.config import (
    PROCESSED_DATA_DIR,
    TEXT_COL,
    TARGET_COL,
)

from emoji_sentiment_analysis.features import (
    extract_emoji_polarity_features,
)

app = typer.Typer()


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def validate_feature_dataset(df: pd.DataFrame) -> None:
    """
    Ensures the engineered dataset satisfies modeling contracts.
    """

    required_columns = {
        TEXT_COL,
        TARGET_COL,
        "emoji_pos_count",
        "emoji_neg_count",
    }

    assert required_columns.issubset(df.columns), \
        f"Missing required columns: {required_columns - set(df.columns)}"

    assert df.isna().sum().sum() == 0, \
        "Null values detected in feature dataset."

    assert (df["emoji_pos_count"] >= 0).all()
    assert (df["emoji_neg_count"] >= 0).all()


# ---------------------------------------------------------------------
# Feature Engineering Logic
# ---------------------------------------------------------------------

def build_features(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Executes emoji polarity feature engineering.
    """

    logger.info(f"Loading cleaned dataset → {input_path}")

    df = pd.read_csv(input_path)

    logger.info("Extracting emoji polarity features...")

    features = df[TEXT_COL].apply(extract_emoji_polarity_features)

    df["emoji_pos_count"] = [f[0] for f in features]
    df["emoji_neg_count"] = [f[1] for f in features]

    logger.info("Validating feature dataset...")
    validate_feature_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.success(f"Feature dataset saved → {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")


# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "tweets_clean.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_final.csv",
):
    """
    CLI entry point for feature engineering pipeline.
    """

    logger.info("Starting feature engineering pipeline...")

    build_features(
        input_path=input_path,
        output_path=output_path,
    )

    logger.success("Feature engineering complete.")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    app()
