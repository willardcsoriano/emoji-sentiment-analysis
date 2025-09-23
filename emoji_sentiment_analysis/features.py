import re
import pandas as pd
from typing import List

from loguru import logger
import typer

from emoji_sentiment_analysis.config import PROCESSED_DATA_DIR, TEXT_COL

# Regex to find emojis and their UTF-8 codes
# This pattern matches common emoji Unicode ranges
# Adjust as needed based on the dataset
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002702-\U000027B0"  # Dingbats
    "]+",
    flags=re.UNICODE
)

def extract_emojis(text: str) -> str:
    """
    Extracts emojis from text and returns them as a single string.
    """
    return " ".join(re.findall(EMOJI_PATTERN, text))

def add_emoji_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature by extracting and repeating emoji strings.
    """
    logger.info("Adding emoji-based feature...")
    # This step is based on the logic you developed in your notebooks
    df['text_with_emojis'] = df[TEXT_COL] + ' ' + df[TEXT_COL].apply(lambda x: extract_emojis(x) * 5)
    return df

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "combined_data_processed.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """
    Applies feature engineering to the processed dataset.
    
    This function demonstrates the intended use of the features.py script,
    though in this project the logic is integrated into the training script
    for simplicity.
    """
    logger.info("Starting feature generation...")
    
    # Load the processed data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Processed data not found at {input_path}. Please run dataset.py first.")
        return
        
    # Apply the feature engineering function
    df_with_features = add_emoji_feature(df)
    
    # Save the new dataframe with the added features
    df_with_features.to_csv(output_path, index=False)
    
    logger.success(f"Features generated and saved to {output_path}.")
    
if __name__ == "__main__":
    app()