# emoji_sentiment_analysis/modeling/predict.py

"""
Production Inference Pipeline
-----------------------------
Operationalizes the trained sentiment classification model.
Synchronized with hybrid Lexicon features (Emoji + Text).
Purely in-memory: No local file logging to prevent repository bloat.
"""

from __future__ import annotations

import joblib
from loguru import logger

from emoji_sentiment_analysis.config import MODELS_DIR, init_logging
from emoji_sentiment_analysis.services.audit_service import generate_inference_audit

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

# Global instances for API performance
try:
    model, tfidf = load_artifacts()
except Exception:
    model, tfidf = None, None

# -------------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------------

def predict_sentiment(text: str, run_audit: bool = False) -> dict:
    """
    Generate sentiment prediction using the Hybrid Audit Service.
    
    Note: run_audit is kept as a legacy argument to prevent breaking existing 
    API calls, but file logging has been decommissioned.
    """
    global model, tfidf
    
    if model is None or tfidf is None:
        try:
            model, tfidf = load_artifacts()
        except Exception:
            return {"error": "Model files not found on disk."}

    # Generate full audit (Math, Sarcasm Veto, and Decision Drivers)
    # This is calculated entirely in-memory.
    return generate_inference_audit(text, model, tfidf)

# -------------------------------------------------------------------
# CLI Interface
# -------------------------------------------------------------------

if __name__ == "__main__":
    init_logging()
    
    sample_texts = [
        "I love this project 😊",
        "i love having bugs 😭", # Sarcasm Test
        "i love hate you baby",   # Lexicon Bias Test
    ]

    logger.info("Running Stateless Inference Test...")
    
    for text in sample_texts:
        res = predict_sentiment(text)
        
        # Format drivers for CLI display
        drivers = ", ".join([f"{d['token']} ({d['weight']})" for d in res['top_drivers']])
        
        print(f"\nText: {res['raw_text']}")
        print(f"Result: {res['prediction']} (Confidence: {res['confidence']})")
        print(f"Decision Drivers: {drivers}")
        print("-" * 30)