import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from loguru import logger
import typer

from emoji_sentiment_analysis.config import REPORTS_DIR, PROCESSED_DATA_DIR, TEXT_COL, TARGET_COL

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "combined_data_processed.csv",
    output_dir: Path = REPORTS_DIR / "figures",
):
    """
    Generates and saves key exploratory data analysis plots.
    """
    logger.info("Starting EDA plot generation...")
    
    # Load data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Processed data not found at {input_path}. Please run dataset.py first.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    # Plot 1: Sentiment Class Balance
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=TARGET_COL, data=df)
        plt.title('Sentiment Class Balance')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig(output_dir / "class_balance.png")
        plt.close()
        logger.info("Saved sentiment class balance plot.")
    except Exception as e:
        logger.error(f"Failed to generate class balance plot: {e}")

    # Plot 2: Text Length Distribution
    try:
        df['text_length'] = df[TEXT_COL].apply(len)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=TARGET_COL, y='text_length', data=df)
        plt.title('Text Length by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Text Length')
        plt.savefig(output_dir / "text_length_distribution.png")
        plt.close()
        logger.info("Saved text length distribution plot.")
    except Exception as e:
        logger.error(f"Failed to generate text length distribution plot: {e}")
    
    logger.success("All plots generated successfully.")

if __name__ == "__main__":
    app()