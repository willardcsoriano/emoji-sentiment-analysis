"""
📑 Script: train.py
-------------------
This script handles training, evaluation, and saving of a sentiment 
analysis model using preprocessed text data.
"""

import pandas as pd
from pathlib import Path
import joblib

from loguru import logger
import typer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Internal imports
from emoji_sentiment_analysis.features import add_emoji_feature
from emoji_sentiment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, TARGET_COL

app = typer.Typer()

@app.command()
def main(
    processed_data_path: Path = PROCESSED_DATA_DIR / "combined_data_processed.csv",
    model_path: Path = MODELS_DIR / "sentiment_model.pkl",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
):
    """
    Loads processed data, trains a Multinomial Naive Bayes model, and saves the trained model.
    """
    logger.info("Starting model training...")

    # 1. Load the processed data
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        logger.error(f"File not found at {processed_data_path}. Run the data processing pipeline first.")
        return

    # 2. Feature Engineering: Amplify Emoji Signals
    df = add_emoji_feature(df)
    
    X = df['text_with_emojis']
    y = df[TARGET_COL]

    # 3. Stratified Train/Val/Test Split
    logger.info("Splitting data into training, validation, and test sets...")
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.5, random_state=SEED, stratify=y_train_temp
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")

    # 4. TF-IDF Vectorization with Noise Filtering
    logger.info("Fitting TF-IDF Vectorizer (Removing Stop Words)...")
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Filters out "is", "the", "this", etc.
        ngram_range=(1, 2),    # Captures "not good" vs "good"
        min_df=5,              # Ignores extremely rare typos
        max_df=0.8             # Ignores words that appear in 80%+ of tweets
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Train Multinomial Naive Bayes
    logger.info("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 6. Evaluation
    y_val_pred = model.predict(X_val_vec)
    accuracy = accuracy_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    # 7. Final Test Evaluation
    y_test_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test Set Accuracy: {test_accuracy:.4f}")

    # 8. Save Artifacts
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.success("Model training and saving complete.")

if __name__ == "__main__":
    app()