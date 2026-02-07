"""
📑 Script: predict.py
---------------------
Updated with deep decision logging and emoji-aware inference.
"""

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import json
from loguru import logger
import typer

from emoji_sentiment_analysis.config import MODELS_DIR
from emoji_sentiment_analysis.features import extract_emojis

# Initialize Typer
app = typer.Typer()

def generate_inference_log(text, model, vectorizer):
    """
    Core logging logic extracted from Notebook 7.0.
    Identifies decision drivers using model.feature_log_prob_.
    """
    # 1. Pipeline Execution: Emoji Signal Amplification
    emojis = extract_emojis(text)
    engine_input = f"{text} {emojis * 5}"
    vec = vectorizer.transform([engine_input])
    
    # 2. Probability & Prediction (Ref: Notebook 5.0)
    probs = model.predict_proba(vec)[0]
    prediction = int(model.predict(vec)[0])
    
    # 3. Feature Importance Extraction (Ref: Notebook 4.0/5.0 Artifacts)
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = vec.nonzero()[1]
    
    feature_impacts = []
    for idx in nonzero_indices:
        token = feature_names[idx]
        # Weight = LogProb(Pos) - LogProb(Neg)
        weight = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
        feature_impacts.append({
            "token": token,
            "weight": round(weight, 4),
            "sentiment_lean": "Positive" if weight > 0 else "Negative"
        })
    
    # 4. Construct the Deep Log Entry
    log_entry = {
        "raw_text": text,
        "engine_input": engine_input,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "confidence": round(float(np.max(probs)), 4),
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.15 else "Clear Signal",
        "top_drivers": sorted(feature_impacts, key=lambda x: abs(x['weight']), reverse=True)[:3],
        "emoji_detected": len(emojis) > 0
    }
    
    return log_entry

@app.command()
def main(
    text: str = typer.Argument(..., help="The text to predict the sentiment of."),
    model_path: Path = MODELS_DIR / "sentiment_model.pkl",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
):
    """
    Predicts sentiment and outputs a detailed decision log.
    """
    logger.info(f"Starting audit-level inference for: '{text}'")

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        logger.error("Model or vectorizer not found. Please check MODELS_DIR.")
        return

    # Generate the deep log
    log_data = generate_inference_log(text, model, vectorizer)

    # Output to CLI
    logger.success(f"Sentiment: {log_data['prediction']} (Confidence: {log_data['confidence']*100:.2f}%)")
    logger.info(f"Entropy Status: {log_data['entropy_flag']}")
    
    print("\n--- Decision Audit Log ---")
    print(json.dumps(log_data, indent=2))


if __name__ == "__main__":
    app()