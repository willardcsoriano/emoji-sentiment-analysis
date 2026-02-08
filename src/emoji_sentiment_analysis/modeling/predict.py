# emoji_sentiment_analysis/modeling/predict.py

"""
Production Inference Pipeline
-----------------------------
Operationalizes the trained sentiment classification model.
Synchronized with hybrid Lexicon features (Emoji + Text).
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.sparse import hstack
from loguru import logger

from emoji_sentiment_analysis.config import MODELS_DIR, init_logging
from emoji_sentiment_analysis.features import extract_emoji_polarity_features
from emoji_sentiment_analysis.services.audit_service import (
    generate_inference_audit, 
    lean_log_append
)

# -------------------------------------------------------------------
# Artifact Loading
# -------------------------------------------------------------------

def load_artifacts():
    """Load model and vectorizer from the central models directory."""
    model_path = MODELS_DIR / "sentiment_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    
    if not model_path.exists() or not vectorizer_path.exists():
        logger.error(f"Artifacts not found in {MODELS_DIR}. Run training first.")
        raise FileNotFoundError("Modeling artifacts missing.")

    return joblib.load(model_path), joblib.load(vectorizer_path)

try:
    model, tfidf = load_artifacts()
except Exception:
    model, tfidf = None, None

# -------------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------------

def predict_sentiment(text: str, run_audit: bool = True) -> dict:
    global model, tfidf
    
    if model is None or tfidf is None:
        try:
            model, tfidf = load_artifacts()
        except Exception:
            return {"error": "Model files not found"}

    if run_audit:
        # Since we synced the logic into the audit service, 
        # just calling this will now return the "Negative" result for sarcasm
        audit_data = generate_inference_audit(text, model, tfidf)
        lean_log_append(audit_data)
        return audit_data
    
    """
    Generate sentiment prediction using TF-IDF and Hybrid Lexicons.
    Synchronized for 4 numeric features.
    """
    if model is None or tfidf is None:
            try:
                model, tfidf = load_artifacts()
            except Exception:
                return {"error": "Model files not found on disk. Please run training first."}

    # 1. Feature Extraction
    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)
    
    # 2. Vectorization
    text_vec = tfidf.transform([text])
    numeric_vec = np.array([[e_pos, e_neg, w_pos, w_neg]])
    features = hstack([text_vec, numeric_vec])

    # 3. Execution with Sarcasm Veto
    # If the user uses a strong negative emoji, we prioritize that signal 
    # even if the ML math is confused by 'love'.
    if e_neg > e_pos and e_neg >= 10:  # 10 is our EMOJI_BOOST
        label = 0
        proba = 0.2  # Forced low confidence for positive
    else:
        label = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1]

    # 4. Automated Audit & Logging
    if run_audit:
        # Pass control to audit service for rich data
        audit_data = generate_inference_audit(text, model, tfidf)
        lean_log_append(audit_data)
        return audit_data

    return {
        "text": text,
        "predicted_label": int(label),
        "prediction_proba": round(float(proba), 4),
    }

# -------------------------------------------------------------------
# CLI Interface
# -------------------------------------------------------------------

if __name__ == "__main__":
    init_logging()
    
    sample_texts = [
        "I love this project 😊",
        "I hate you baby",
        "The bugs are terrible 😭",
    ]

    logger.info("Running inference with Hybrid Audit Service...")
    
    for text in sample_texts:
        res = predict_sentiment(text)
        
        # Displaying the "Why" (including the new Word Lexicon drivers)
        drivers = ", ".join([f"{d['token']} ({d['weight']})" for d in res['top_drivers']])
        
        print(f"\nText: {res['raw_text']}")
        print(f"Result: {res['prediction']} (Confidence: {res['confidence']})")
        print(f"Decision Drivers: {drivers}")
        print("-" * 30)