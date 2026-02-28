# src\emoji_sentiment_analysis\services\audit_service.py

"""
Inference Audit Service
-----------------------
Purely observational. Receives a completed prediction and explains it.
Makes no predictions, applies no overrides, has no side effects on output.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix


def explain_prediction(
    text: str,
    prediction: int,
    probs: np.ndarray,
    features: csr_matrix,
    model,
    tfidf,
    is_veto: bool,
    confidence: float,     # <-- Added to signature
    entropy_flag: str      # <-- Added to signature
) -> dict:
    """
    Formats and explains a completed prediction.
    All decision-making has already occurred in predict.py before this is called.
    """
    # Decision Drivers
    feature_names = tfidf.get_feature_names_out()
    hybrid_feature_names = [
        "emoji_pos_count", "emoji_neg_count",
        "word_pos_count",  "word_neg_count",
    ]
    all_feature_names = np.concatenate([feature_names, hybrid_feature_names])

    nonzero_indices = features.nonzero()[1]
    coefs = model.coef_[0]

    feature_impacts = []
    for idx in nonzero_indices:
        token = all_feature_names[idx]
        weight = coefs[idx]
        feature_impacts.append({"token": token, "weight": round(float(weight), 4)})

    top_drivers = sorted(feature_impacts, key=lambda x: abs(x["weight"]), reverse=True)

    if is_veto:
        veto_feature = next(
            (d for d in feature_impacts if d["token"] == "emoji_neg_count"), None
        )
        if veto_feature and veto_feature in top_drivers:
            top_drivers.insert(0, top_drivers.pop(top_drivers.index(veto_feature)))

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "prediction_int": prediction,
        "confidence": round(confidence, 4),
        "entropy_flag": entropy_flag,
        "veto_applied": is_veto,
        "top_drivers": top_drivers[:6],
    }


def get_global_model_signals(model, vectorizer, top_n: int = 5) -> dict:
    """
    Extracts the strongest global signals (coefficients) from the model.
    Useful for health check reports.
    """
    feature_names = vectorizer.get_feature_names_out()
    hybrid_feature_names = [
        "emoji_pos_count", "emoji_neg_count",
        "word_pos_count", "word_neg_count"
    ]
    all_feature_names = np.concatenate([feature_names, hybrid_feature_names])

    weights = model.coef_[0]
    vocab_weights = list(zip(all_feature_names, weights))

    top_pos = sorted(vocab_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_neg = sorted(vocab_weights, key=lambda x: x[1], reverse=False)[:top_n]

    return {
        "top_positive_signals": [
            {"token": t, "weight": round(float(w), 2)} for t, w in top_pos
        ],
        "top_negative_signals": [
            {"token": t, "weight": round(float(w), 2)} for t, w in top_neg
        ],
    }