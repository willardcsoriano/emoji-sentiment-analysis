# emoji_sentiment_analysis/services/audit_service.py

"""
Inference Audit Service (Lean Version)
--------------------------------------
Calculates probabilities and identifies decision drivers (explainability).
"""

from __future__ import annotations

import numpy as np
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from typing import cast

# Internal imports
from emoji_sentiment_analysis.features import extract_emoji_polarity_features

def generate_inference_audit(text: str, model, tfidf) -> dict:
    """
    Calculates probabilities and identifies decision drivers.
    Includes Veto-Aware logic to ensure manual overrides are documented as drivers.
    """
    # 1. Feature Extraction (Unpack 4)
    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)
    
    # 2. Vectorization & Assembly
    text_vec = tfidf.transform([text])
    numeric_vec = np.array([[e_pos, e_neg, w_pos, w_neg]])
    
    # Combined feature set - Cast to csr_matrix for support
    features = cast(csr_matrix, hstack([text_vec, numeric_vec]).tocsr())
    
    # 3. Probability & Prediction with Veto Sync
    probs = model.predict_proba(features)[0]
    is_veto = False
    
    # Apply Sarcasm Veto: If negative emojis outweigh positive and meet boost threshold
    if e_neg >= 10 and (e_neg >= e_pos):
        prediction = 0 # Force Negative
        probs = np.array([0.85, 0.15]) # High confidence in the Veto
        is_veto = True
    else:
        prediction = int(model.predict(features)[0])
    
    # 4. Decision Drivers (Explainability)
    feature_names = tfidf.get_feature_names_out()
    hybrid_feature_names = [
        "emoji_pos_count", 
        "emoji_neg_count", 
        "word_pos_count", 
        "word_neg_count"
    ]
    all_feature_names = np.concatenate([feature_names, hybrid_feature_names])
    
    nonzero_indices = features.nonzero()[1]
    coefs = model.coef_[0]

    feature_impacts = []
    for idx in nonzero_indices:
        token = all_feature_names[idx]
        weight = coefs[idx]
        feature_impacts.append({"token": token, "weight": round(float(weight), 4)})
    
    # Initial sort by absolute statistical influence
    top_drivers = sorted(feature_impacts, key=lambda x: abs(x['weight']), reverse=True)

    # VETO ADJUSTMENT: If a veto occurred, ensure the veto-ing feature is the primary driver.
    if is_veto:
        veto_feature = next((d for d in feature_impacts if d['token'] == "emoji_neg_count"), None)
        if veto_feature:
            top_drivers.insert(0, top_drivers.pop(top_drivers.index(veto_feature)))

    # Return only the top 6 drivers for UI efficiency
    final_drivers = top_drivers[:6]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "prediction_int": prediction, 
        "confidence": round(float(np.max(probs)), 4),
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.20 else "Clear Signal",
        "top_drivers": final_drivers
    }

def get_global_model_signals(model, vectorizer, top_n: int = 5) -> dict:
    """
    Extracts the strongest global signals (coefficients) from the model.
    Useful for 'Health Check' reports.
    """
    feature_names = vectorizer.get_feature_names_out()
    hybrid_feature_names = ["emoji_pos_count", "emoji_neg_count", "word_pos_count", "word_neg_count"]
    all_feature_names = np.concatenate([feature_names, hybrid_feature_names])
    
    weights = model.coef_[0]
    vocab_weights = list(zip(all_feature_names, weights))
    
    top_pos = sorted(vocab_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_neg = sorted(vocab_weights, key=lambda x: x[1], reverse=False)[:top_n]
    
    return {
        "top_positive_signals": [{"token": t, "weight": round(float(w), 2)} for t, w in top_pos],
        "top_negative_signals": [{"token": t, "weight": round(float(w), 2)} for t, w in top_neg]
    }