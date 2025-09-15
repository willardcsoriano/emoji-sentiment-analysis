import pandas as pd
from pathlib import Path
import re
import joblib

from loguru import logger
from tqdm import tqdm
import typer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from emoji_sentiment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, TEXT_COL, TARGET_COL

app = typer.Typer()

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

def extract_emojis(text):
    """
    Extracts emojis from text and returns them as a single string.
    """
    return " ".join(re.findall(EMOJI_PATTERN, text))

@app.command()
def main(
    processed_data_path: Path = PROCESSED_DATA_DIR / "1k_data_processed.csv",
    model_path: Path = MODELS_DIR / "sentiment_model.pkl",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
):
    """
    Loads processed data, trains a LinearSVC model, and saves the trained model.
    """
    logger.info("Starting model training...")
    
    # Load the processed data
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        logger.error(f"Processed data file not found at {processed_data_path}. Please run 'python -m emoji_sentiment_analysis.dataset' first.")
        return

    # Create a new column to amplify the emoji signal
    df['text_with_emojis'] = df[TEXT_COL] + ' ' + df[TEXT_COL].apply(lambda x: extract_emojis(x) * 5)
    
    X = df['text_with_emojis']
    y = df[TARGET_COL]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Initialize and fit the TF-IDF vectorizer
    logger.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8)
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    # Initialize and train the Linear SVC model
    logger.info("Training LinearSVC model...")
    model = LinearSVC(random_state=SEED, C=0.5)
    model.fit(X_train_transformed, y_train)

    # Make predictions and evaluate the model
    logger.info("Evaluating model performance...")
    y_pred = model.predict(X_test_transformed)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    # Save the trained model and vectorizer
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.success("Model training and saving complete.")


if __name__ == "__main__":
    app()