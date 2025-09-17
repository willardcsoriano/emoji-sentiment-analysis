# emoji_sentiment_analysis\emoji_sentiment_analysis\dataset.py

import pandas as pd
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from emoji_sentiment_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TEXT_COL, TARGET_COL

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "1k_data_emoji_tweets_senti_posneg.csv",
    output_path: Path = PROCESSED_DATA_DIR / "1k_data_processed.csv",
):
    """
    Loads raw data, renames columns, and saves a processed dataset.
    """
    # -------------------------------------------------------------
    # Load the dataset
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}. Please place the file in the data/raw directory.")
        return

    # Inspect the initial columns
    logger.info(f"Initial columns: {df.columns.tolist()}")

    # Rename columns for consistency with global knobs
    # Based on the problem description, 'tweet' seems to be the text and 'sentiment' the target.
    # Adjust these column names if your dataset has different headers.
    column_mapping = {
        'tweet': TEXT_COL,
        'sentiment': TARGET_COL
    }
    df = df.rename(columns=column_mapping)
    
    logger.info(f"Columns renamed to: {df.columns.tolist()}")

    # Display some basic information about the data
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"First 5 rows:\n{df.head()}")

    # Convert sentiment labels to a consistent format (e.g., lowercase)
    df[TARGET_COL] = df[TARGET_COL].str.lower().str.strip()

    # Save the processed data to the specified output path
    logger.info(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    
    logger.success("Dataset processing complete.")
    # -------------------------------------------------------------


if __name__ == "__main__":
    app()