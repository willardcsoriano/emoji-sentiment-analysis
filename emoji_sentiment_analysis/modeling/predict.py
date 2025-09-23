from pathlib import Path
import pandas as pd
import joblib

from loguru import logger
import typer

from emoji_sentiment_analysis.config import MODELS_DIR

app = typer.Typer()

@app.command()
def main(
    text: str = typer.Argument(..., help="The text to predict the sentiment of."),
    model_path: Path = MODELS_DIR / "sentiment_model.pkl",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
):
    """
    Loads a trained model and vectorizer to predict the sentiment of a given text.
    """
    logger.info(f"Starting inference for the text: '{text}'")

    # Load the trained model and vectorizer
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully!")
        
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully!")
        
    except FileNotFoundError:
        logger.error("Model or vectorizer files not found. Please ensure train.py has been run.")
        return
    
    # Create a pandas Series from the new text
    new_text_series = pd.Series([text])

    # Transform the new text using the loaded vectorizer
    new_text_vec = vectorizer.transform(new_text_series)

    # Make a prediction using the loaded model
    prediction = model.predict(new_text_vec)

    # Map the prediction to a human-readable label
    sentiment_map = {0: "Negative", 1: "Positive"}
    predicted_sentiment = sentiment_map[prediction[0]]

    # Log the final result
    logger.info(f"Original Text: '{text}'")
    logger.success(f"Predicted Sentiment: {predicted_sentiment}")

if __name__ == "__main__":
    app()