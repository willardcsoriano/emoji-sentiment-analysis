# emoji_sentiment_analysis\emoji_sentiment_analysis\modeling\train.py

"""
📑 Script: train.py
-------------------
This script handles **training, evaluation, and saving** of a sentiment 
analysis model using preprocessed text data.

Workflow:
1. Load processed dataset from `combined_data_processed.csv`.
2. Amplify the emoji signal by appending extracted emojis to text.
3. Split the data into train, validation, and test sets (stratified).
4. Fit a TF-IDF vectorizer and train a Multinomial Naive Bayes classifier.
5. Evaluate the model on validation and test sets.
6. Save the trained model (`sentiment_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`) 
   to the `models` directory for future inference.

Usage Example:
    python -m emoji_sentiment_analysis.modeling.train

Expected Output:
    - Validation accuracy and classification report
    - Test accuracy and classification report
    - Confirmation logs that the model and vectorizer have been saved
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

# Import the feature engineering function from the features script
from emoji_sentiment_analysis.features import add_emoji_feature
from emoji_sentiment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, TEXT_COL, TARGET_COL

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

    # Load the processed data
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        logger.error(f"Processed data file not found at {processed_data_path}. Please run the data processing pipeline first.")
        return

    # Call the feature engineering function from features.py
    df = add_emoji_feature(df)
    
    X = df['text_with_emojis']
    y = df[TARGET_COL]

    # Perform a three-way split for train, validation, and test sets
    logger.info("Splitting data into training, validation, and test sets...")
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.5, random_state=SEED, stratify=y_train_temp
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Initialize and fit the TF-IDF vectorizer on the training data
    logger.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize and train the best model (Multinomial Naive Bayes)
    logger.info("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate the model on the validation set
    logger.info("Evaluating model performance on the validation set...")
    y_val_pred = model.predict(X_val_vec)
    accuracy = accuracy_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    # Final evaluation on the test set
    logger.info("Performing final evaluation on the test set...")
    y_test_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    
    logger.info(f"Test Set Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Set Classification Report:\n{test_report}")

    # Save the trained model and vectorizer
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.success("Model training and saving complete.")

if __name__ == "__main__":
    app()