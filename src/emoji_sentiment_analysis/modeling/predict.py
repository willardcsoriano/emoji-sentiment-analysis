# emoji_sentiment_analysis/modeling/predict.py

"""
Production Inference Pipeline
-----------------------------
Operationalizes the trained sentiment classification model.
Synchronized with hybrid Lexicon features (Emoji + Text).

Runs on the scikit-learn-free :class:`LiteModel` backend: feature extraction
and the sarcasm veto are pure Python, and the TF-IDF + logistic decision is
pure NumPy, so the serving path imports neither scikit-learn nor scipy.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR, init_logging
from emoji_sentiment_analysis.features import extract_emoji_polarity_features
from emoji_sentiment_analysis.modeling.lite_model import LiteModel
from emoji_sentiment_analysis.services.audit_service import explain_prediction

# -------------------------------------------------------------------
# Artifact Loading
# -------------------------------------------------------------------


def load_lite_model() -> LiteModel:
    """Load the lightweight model artifacts from the central models directory."""
    npz_path = MODELS_DIR / "model_lite.npz"
    json_path = MODELS_DIR / "model_lite.json"

    if not npz_path.exists() or not json_path.exists():
        logger.error(
            f"Lite artifacts not found in {MODELS_DIR}. "
            f"Run scripts/export_lite_model.py against the trained .pkl files."
        )
        raise FileNotFoundError("Lite model artifacts missing.")

    return LiteModel.load(MODELS_DIR)


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


def predict_sentiment(text: str, lite: LiteModel | None = None) -> dict:
    """
    Full inference pipeline with sarcasm veto and explainability.
    Accepts a :class:`LiteModel` injected from app.state (production).
    Falls back to loading from disk for CLI/dev use.
    """
    # Fallback for CLI use
    if lite is None:
        try:
            lite = load_lite_model()
        except Exception:
            return {"error": "Model files not found."}

    # 1. Feature Extraction (pure Python)
    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)

    # 2. Vectorization + logistic decision (pure NumPy).
    #    nonzero_indices feeds the explainer regardless of the veto outcome.
    model_prediction, model_probs, nonzero_indices = lite.predict(text, e_pos, e_neg, w_pos, w_neg)

    # 3. Sarcasm Veto (deterministic override)
    is_veto, veto_prediction, veto_probs = _apply_sarcasm_veto(e_neg, e_pos, w_pos)

    if is_veto:
        prediction = veto_prediction
        probs = veto_probs
    else:
        prediction = model_prediction
        probs = model_probs

    # 4. Confidence & Ambiguity
    confidence = float(np.max(probs))
    entropy_flag = "High Ambiguity" if confidence < AMBIGUITY_THRESHOLD else "Clear Signal"

    # 5. Delegate explainability
    return explain_prediction(
        text=text,
        prediction=prediction,
        probs=probs,
        nonzero_indices=nonzero_indices,
        all_feature_names=lite.all_feature_names,
        coef=lite.coef,
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
        "i love having bugs 😭",
        "i love hate you baby",
    ]

    logger.info("Running Stateless Inference Test...")

    for text in sample_texts:
        res = predict_sentiment(text)
        drivers = ", ".join([f"{d['token']} ({d['weight']})" for d in res["top_drivers"]])
        print(f"\nText    : {res['raw_text']}")
        print(f"Result  : {res['prediction']} (Confidence: {res['confidence']})")
        print(f"Drivers : {drivers}")
        print("-" * 30)
