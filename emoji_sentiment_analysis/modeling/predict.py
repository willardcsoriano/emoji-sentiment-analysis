# emoji_sentiment_analysis\emoji_sentiment_analysis\modeling\predict.py

"""
📑 Script: predict.py
---------------------
This script provides a command-line interface (CLI) for **sentiment inference** 
using a previously trained model and TF-IDF vectorizer.

Workflow:
1. Load the saved model (`sentiment_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`).
2. Transform the input text into numerical features.
3. Predict the sentiment (0 = Negative, 1 = Positive).
4. Output the sentiment label in human-readable form.

Usage Example:
    python predict.py "I'm so happy right now! _EMOJI_"

Expected Output:
    Original Text: "I'm so happy right now! _EMOJI_"
    Predicted Sentiment: Positive
"""

from pathlib import Path
import pandas as pd
import joblib

from loguru import logger
import typer

from emoji_sentiment_analysis.config import MODELS_DIR

# Initialize Typer for building a CLI interface
app = typer.Typer()


@app.command()
def main(
    text: str = typer.Argument(..., help="The text to predict the sentiment of."),
    model_path: Path = MODELS_DIR / "sentiment_model.pkl",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
):
    """
    Command-line tool to predict sentiment of input text using a trained model.

    Args:
        text (str): Input text for which sentiment should be predicted.
        model_path (Path): Path to the trained sentiment model file.
        vectorizer_path (Path): Path to the saved TF-IDF vectorizer file.
    """

    # Log start of inference
    logger.info(f"Starting inference for the text: '{text}'")

    # Load the trained model and vectorizer
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully!")

        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully!")

    except FileNotFoundError:
        # Catch missing model/vectorizer files
        logger.error("Model or vectorizer files not found. Please ensure train.py has been run.")
        return

    # Wrap the input text in a Pandas Series (vectorizer expects this format)
    new_text_series = pd.Series([text])

    # Transform the new text into numerical features using the TF-IDF vectorizer
    new_text_vec = vectorizer.transform(new_text_series)

    # Generate sentiment prediction using the trained model
    prediction = model.predict(new_text_vec)

    # Map numeric prediction to human-readable sentiment label
    sentiment_map = {0: "Negative", 1: "Positive"}
    predicted_sentiment = sentiment_map[prediction[0]]

    # Log the final result
    logger.info(f"Original Text: '{text}'")
    logger.success(f"Predicted Sentiment: {predicted_sentiment}")


# Entry point for running as a script
if __name__ == "__main__":
    app()
