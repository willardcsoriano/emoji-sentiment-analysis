# emoji_sentiment_analysis/modeling/predict.py

"""
Production Inference Pipeline
-----------------------------

This module operationalizes the trained sentiment classification model.

Responsibilities:
    - Load serialized modeling artifacts
    - Reconstruct feature transformation pipeline
    - Generate predictions for text inputs
    - Surface emoji polarity attribution
    - Provide batch and single inference interfaces

Artifacts Consumed:
    data/processed/features_final.csv (optional reference)
    models/sentiment_model.pkl
    models/tfidf_vectorizer.pkl

This script is deployment-ready and notebook-independent.
"""

from pathlib import Path
import numpy as np
import joblib
from scipy.sparse import hstack

from emoji_sentiment_analysis.features import (
    extract_emoji_polarity_features,
)

# -------------------------------------------------------------------
# Artifact Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

# -------------------------------------------------------------------
# Artifact Loading
# -------------------------------------------------------------------

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# -------------------------------------------------------------------
# Core Prediction Function
# -------------------------------------------------------------------


def predict_sentiment(text: str) -> dict:
    """
    Generate sentiment prediction for a single text input.

    Returns:
        dict containing:
            - text
            - predicted_label
            - prediction_proba
            - emoji_pos_count
            - emoji_neg_count
    """

    # Emoji feature extraction
    pos_count, neg_count = extract_emoji_polarity_features(text)

    # Text vectorization
    text_vec = tfidf.transform([text])

    # Feature assembly
    emoji_vec = np.array([[pos_count, neg_count]])
    features = hstack([text_vec, emoji_vec])

    # Prediction
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0, 1]

    return {
        "text": text,
        "predicted_label": int(label),
        "prediction_proba": float(proba),
        "emoji_pos_count": pos_count,
        "emoji_neg_count": neg_count,
    }


# -------------------------------------------------------------------
# Batch Prediction Interface
# -------------------------------------------------------------------


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Generate predictions for multiple text inputs.
    """

    results = [predict_sentiment(text) for text in texts]
    return results


# -------------------------------------------------------------------
# CLI Test Harness (Optional)
# -------------------------------------------------------------------

if __name__ == "__main__":

    sample_texts = [
        "I love this so much 😊",
        "This is terrible 😭",
        "I feel okay about this",
        "Best day ever 😆😍",
        "I hate everything",
    ]

    for result in predict_batch(sample_texts):
        print(result)
