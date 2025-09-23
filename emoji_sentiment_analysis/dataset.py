import pandas as pd
from pathlib import Path

from loguru import logger
import typer
from pandas.api.types import is_numeric_dtype

# Import the feature engineering function from the features script
from emoji_sentiment_analysis.features import process_text_with_emojis
from emoji_sentiment_analysis.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEXT_COL,
    TARGET_COL,
)

app = typer.Typer()


def load_emoticon_data(path: Path) -> dict[str, str]:
    """
    Loads the emoticon data and creates a mapping from emoji to a
    normalized sentiment word.
    """
    try:
        # The exploration notebook showed this file has an 'Unnamed: 0' column
        df = pd.read_csv(path, index_col=0) 
        # Create a dictionary mapping the Emoji to a normalized word.
        return {emoji: "_EMOJI_" for emoji in df['Emoji'].tolist()}
    except Exception as e:
        logger.error(f"Failed to load emoticon data from {path}: {e}")
        return {}


# A cleaner way to organize the main function
@app.command()
def main(
    input_paths: list[Path] = typer.Option(
        None,
        "--input-paths",
        "-i",
        help="List of paths to raw CSV datasets to be processed.",
    ),
    output_path: Path = PROCESSED_DATA_DIR / "combined_data_processed.csv",
):
    """
    Load raw data from multiple files, align column names to config,
    clean minimal issues, and save a combined, processed dataset.
    """
    # Define default input paths if none are provided
    if not input_paths:
        input_paths = [
            RAW_DATA_DIR / "1k_data_emoji_tweets_senti_posneg.csv",
            RAW_DATA_DIR / "15_emoticon_data.csv"
        ]

    # Load emoticon data first
    emoticon_path = RAW_DATA_DIR / "15_emoticon_data.csv"
    emoji_lookup = load_emoticon_data(emoticon_path)
    if not emoji_lookup:
        logger.warning("No emoticon data found. Emojis will not be processed.")
    
    combined_df = pd.DataFrame()

    # Then, load and process the other datasets
    for input_path in input_paths:
        if input_path.name == "15_emoticon_data.csv":
            continue # Skip the emoticon data file

        logger.info(f"Loading data from {input_path}...")
        try:
            # Your existing loading and processing logic here
            df = pd.read_csv(input_path, index_col=0)
            
            column_mapping = {
                "post": TEXT_COL,
                "tweet": TEXT_COL,
                "text": TEXT_COL,
                "sentiment": TARGET_COL,
                "label": TARGET_COL,
                "target": TARGET_COL,
            }
            df = df.rename(columns=column_mapping)
            
            missing = [col for col in (TEXT_COL, TARGET_COL) if col not in df.columns]
            if missing:
                logger.error(f"Skipping {input_path}. Missing column(s): {missing}.")
                continue
            
            if emoji_lookup:
                df[TEXT_COL] = df[TEXT_COL].astype(str).apply(
                    lambda x: process_text_with_emojis(x, emoji_lookup)
                )
            
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            logger.info(f"DataFrame shape after adding {input_path.name}: {combined_df.shape}")

        except Exception as e:
            logger.error(f"An error occurred processing {input_path.name}: {e}")

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