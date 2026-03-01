# emoji_sentiment_analysis/modeling/predict.py

"""
Production Inference Pipeline
-----------------------------
Operationalizes the trained sentiment classification model.
Synchronized with hybrid Lexicon features (Emoji + Text).
Purely in-memory: No local file logging to prevent repository bloat.
"""

from __future__ import annotations

from typing import cast

import joblib
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix, hstack

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR, init_logging
from emoji_sentiment_analysis.features import extract_emoji_polarity_features
from emoji_sentiment_analysis.services.audit_service import explain_prediction

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
# Sarcasm Veto
# -------------------------------------------------------------------


def _apply_sarcasm_veto(e_neg: int, e_pos: int, w_pos: int) -> tuple[bool, int, np.ndarray]:
    """
    Refined deterministic override rule with word context guard and
    scaled confidence.

    Guard: suppressed when w_pos >= 2 and e_pos == 0 — strong positive
    word signal with no positive emoji indicates genuine sentiment, not
    sarcasm. Defer to model.

    Confidence: scaled 0.75–0.95 based on signal dominance rather than
    hardcoded. Reflects actual gap between negative and positive emoji
    signal.

    Co-designed with EMOJI_BOOST in features.py — threshold must equal
    EMOJI_BOOST so a single negative emoji exactly meets the threshold.
    Do not change independently.

    Specified in Notebook 3.5 Section 8. Verified in Notebook 4.5
    Section 5 and Notebook 5.0 Section 5.

    Returns: (veto_applied, prediction, probs)
    """
    from emoji_sentiment_analysis.features import EMOJI_BOOST

    # Word context guard
    if w_pos >= 2 and e_pos == 0:
        return False, -1, np.array([])

    # Threshold check
    if e_neg >= EMOJI_BOOST and e_neg >= e_pos:
        signal_gap = (e_neg - e_pos) / (e_neg + e_pos + 1)
        scaled_conf = round(0.75 + (0.20 * signal_gap), 4)
        return True, 0, np.array([1 - scaled_conf, scaled_conf])

    return False, -1, np.array([])


# -------------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------------


def predict_sentiment(text: str) -> dict:
    """
    Full inference pipeline with sarcasm veto and explainability.
    """
    global model, tfidf

    if model is None or tfidf is None:
        try:
            model, tfidf = load_artifacts()
        except Exception:
            return {"error": "Model files not found on disk."}

    # 1. Feature Extraction
    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)

    # 2. Vectorization & Assembly
    text_vec = tfidf.transform([text])
    numeric_vec = np.array([[e_pos, e_neg, w_pos, w_neg]])
    features = cast(csr_matrix, hstack([text_vec, numeric_vec]).tocsr())

    # 3. Sarcasm Veto
    is_veto, veto_prediction, veto_probs = _apply_sarcasm_veto(e_neg, e_pos, w_pos)

    if is_veto:
        prediction = veto_prediction
        probs = veto_probs
    else:
        probs = model.predict_proba(features)[0]
        prediction = int(model.predict(features)[0])

    # --- Determine confidence and ambiguity here! ---
    confidence = float(np.max(probs))
    entropy_flag = "High Ambiguity" if confidence < AMBIGUITY_THRESHOLD else "Clear Signal"

    # 4. Delegate explainability to audit service (pass the new flags in)
    return explain_prediction(
        text=text,
        prediction=prediction,
        probs=probs,
        features=features,
        model=model,
        tfidf=tfidf,
        is_veto=is_veto,
        confidence=confidence,
        entropy_flag=entropy_flag,
    )


# -------------------------------------------------------------------
# CLI Interface
# -------------------------------------------------------------------

if __name__ == "__main__":
    init_logging()

    sample_texts = [
        "I love this project 😊",
        "i love having bugs 😭",  # Sarcasm Test
        "i love hate you baby",  # Lexicon Bias Test
    ]

    logger.info("Running Stateless Inference Test...")

    for text in sample_texts:
        res = predict_sentiment(text)
        drivers = ", ".join([f"{d['token']} ({d['weight']})" for d in res["top_drivers"]])
        print(f"\nText    : {res['raw_text']}")
        print(f"Result  : {res['prediction']} (Confidence: {res['confidence']})")
        print(f"Drivers : {drivers}")
        print("-" * 30)
