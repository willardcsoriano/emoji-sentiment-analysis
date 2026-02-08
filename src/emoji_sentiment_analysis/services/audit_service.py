# emoji_sentiment_analysis/services/audit_service.py

"""
Inference Audit Service
-----------------------
Calculates probabilities, identifies decision drivers (explainability),
and maintains a circular log of model performance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from scipy.sparse import hstack, csr_matrix
from typing import cast

# Internal imports
from emoji_sentiment_analysis.features import extract_emoji_polarity_features
from emoji_sentiment_analysis.config import LOGS_DIR

# Strictly lean configuration
MAX_LOG_ENTRIES = 500

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
    
    # Combined feature set - Cast to csr_matrix for Pylance/nonzero support
    features = cast(csr_matrix, hstack([text_vec, numeric_vec]).tocsr())
    
    # 3. Probability & Prediction with Veto Sync
    probs = model.predict_proba(features)[0]
    is_veto = False
    
    # Apply Sarcasm Veto: If negative emojis outweigh positive and meet boost threshold
    if e_neg > e_pos and e_neg >= 10:
        prediction = 0
        probs = np.array([0.8, 0.2]) # Manual override of probabilities
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

    # VETO ADJUSTMENT: If a veto occurred, the veto-ing feature is the primary driver.
    # We move emoji_neg_count to the front so the Audit/Tests acknowledge the override.
    if is_veto:
        veto_feature = next((d for d in feature_impacts if d['token'] == "emoji_neg_count"), None)
        if veto_feature:
            # Remove from current position and insert at index 0
            top_drivers.insert(0, top_drivers.pop(top_drivers.index(veto_feature)))

    # Return only the top 3 drivers
    final_drivers = top_drivers[:3]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "prediction_int": prediction, 
        "confidence": round(float(np.max(probs)), 4),
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.15 else "Clear Signal",
        "top_drivers": final_drivers
    }

def lean_log_append(log_data: dict, filename: str = "inference_history.csv"):
    """
    Maintains a strictly circular CSV log file in the configured LOGS_DIR.
    """
    log_path = LOGS_DIR / filename
    
    csv_row = log_data.copy()
    csv_row['top_drivers'] = "|".join([f"{d['token']}({d['weight']})" for d in log_data['top_drivers']])
    
    new_entry = pd.DataFrame([csv_row])
    
    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, new_entry], ignore_index=True).tail(MAX_LOG_ENTRIES)
    else:
        df = new_entry
        
    df.to_csv(log_path, index=False)
    logger.debug(f"Audit log updated at {log_path}")

def get_global_model_signals(model, vectorizer, top_n: int = 5) -> dict:
    """
    Extracts the strongest global signals (coefficients) from the model.
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