# emoji_sentiment_analysis/emoji_sentiment_analysis/dataset.py

import pandas as pd
from pathlib import Path

from loguru import logger
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
    input_paths: list[Path] = typer.Argument(
        [RAW_DATA_DIR / "1k_data_emoji_tweets_senti_posneg.csv"],
        help="List of paths to raw CSV datasets to be processed.",
    ),
    output_path: Path = PROCESSED_DATA_DIR / "combined_data_processed.csv",
):
    """
    Load raw data from multiple files, align column names to config,
    clean minimal issues, and save a combined, processed dataset.
    """
    # -------------------------------------------------------------
    combined_df = pd.DataFrame()
    
    for input_path in input_paths:
        logger.info(f"Loading data from {input_path}...")
        try:
            df = pd.read_csv(input_path)
            
            # Inspect the initial columns
            logger.info(f"Initial columns: {df.columns.tolist()}")

            # Rename columns for consistency with global knobs.
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

            # Drop any auto-generated index columns like 'Unnamed: 0'
            df = df.drop(
                columns=[c for c in df.columns if str(c).lower().startswith("unnamed")],
                errors="ignore",
            )
            
            # Sanity checks: required columns must exist after renaming
            missing = [col for col in (TEXT_COL, TARGET_COL) if col not in df.columns]
            if missing:
                logger.error(
                    f"Skipping {input_path}. Required column(s) missing after renaming: {missing}. "
                    f"Available columns: {df.columns.tolist()}"
                )
                continue  # Skip to the next file
            
            # Add to combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            logger.info(f"DataFrame shape after adding {input_path.name}: {combined_df.shape}")

        except FileNotFoundError:
            logger.error(f"File not found at {input_path}. Skipping.")
        except Exception as e:
            logger.error(f"An error occurred processing {input_path.name}: {e}")

    if combined_df.empty:
        logger.error("No valid data was loaded. Exiting.")
        return

    # -------------------------------------------------------------
    # Final processing on the combined dataframe
    # Ensure text column is string and lightly normalize whitespace
    combined_df[TEXT_COL] = (
        combined_df[TEXT_COL]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Normalize target labels
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(combined_df[TARGET_COL]):
        combined_df[TARGET_COL] = combined_df[TARGET_COL].astype("Int64")
    else:
        combined_df[TARGET_COL] = (
            combined_df[TARGET_COL]
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
    
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    try:
        logger.info(
            f"Combined label distribution:\n{combined_df[TARGET_COL].value_counts(dropna=False).to_string()}"
        )
    except Exception:
        pass

    # Save the processed data to the specified output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving combined processed data to {output_path}...")
    combined_df.to_csv(output_path, index=False)

    logger.success("Dataset processing complete.")
    # -------------------------------------------------------------


if __name__ == "__main__":
    app()