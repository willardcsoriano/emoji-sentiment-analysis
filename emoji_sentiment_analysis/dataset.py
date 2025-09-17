# emoji_sentiment_analysis/emoji_sentiment_analysis/dataset.py

import pandas as pd
from pathlib import Path

from loguru import logger
# from tqdm import tqdm  # not used here; uncomment if you use it elsewhere
import typer

from emoji_sentiment_analysis.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEXT_COL,
    TARGET_COL,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "1k_data_emoji_tweets_senti_posneg.csv",
    output_path: Path = PROCESSED_DATA_DIR / "1k_data_processed.csv",
):
    """
    Load raw data, align column names to config, clean minimal issues,
    and save a processed dataset.
    """
    # -------------------------------------------------------------
    # Load the dataset
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(
            f"File not found at {input_path}. Please place the file in the data/raw directory."
        )
        return

    # Inspect the initial columns
    logger.info(f"Initial columns: {df.columns.tolist()}")

    # Rename columns for consistency with global knobs.
    # Map common text/label headers -> TEXT_COL/TARGET_COL from config.
    column_mapping = {
        # text aliases
        "post": TEXT_COL,
        "tweet": TEXT_COL,
        "text": TEXT_COL,   # idempotent if already equal
        # target/label aliases
        "sentiment": TARGET_COL,
        "label": TARGET_COL,  # idempotent
        "target": TARGET_COL,
    }
    df = df.rename(columns=column_mapping)
    logger.info(f"Columns renamed to: {df.columns.tolist()}")

    # Basic info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"First 5 rows:\n{df.head()}")

    # Drop any auto-generated index columns like 'Unnamed: 0'
    df = df.drop(
        columns=[c for c in df.columns if str(c).lower().startswith("unnamed")],
        errors="ignore",
    )

    # Sanity checks: required columns must exist after renaming
    missing = [col for col in (TEXT_COL, TARGET_COL) if col not in df.columns]
    if missing:
        logger.error(
            f"Required column(s) missing after renaming: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
        return

    # Ensure text column is string and lightly normalize whitespace
    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Normalize target labels:
    # - If already numeric (e.g., int 0/1), enforce integer dtype.
    # - If strings (e.g., 'positive'/'negative'), map to 1/0 then cast.
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(df[TARGET_COL]):
        df[TARGET_COL] = df[TARGET_COL].astype("Int64")
    else:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype("string")
            .str.lower()
            .str.strip()
            .replace(
                {
                    "pos": 1,
                    "positive": 1,
                    "1": 1,
                    "true": 1,
                    True: 1,
                    "neg": 0,
                    "negative": 0,
                    "0": 0,
                    "false": 0,
                    False: 0,
                }
            )
            .astype("Int64")
        )

    # Optional: quick distribution log
    try:
        logger.info(
            f"Label distribution:\n{df[TARGET_COL].value_counts(dropna=False).to_string()}"
        )
    except Exception:
        pass

    # Save the processed data to the specified output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)

    logger.success("Dataset processing complete.")
    # -------------------------------------------------------------


if __name__ == "__main__":
    app()
