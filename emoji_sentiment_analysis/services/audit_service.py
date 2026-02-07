# emoji_sentiment_analysis/services/audit_service.py

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from emoji_sentiment_analysis.features import extract_emojis

# Strictly lean configuration
MAX_LOG_ENTRIES = 500

def generate_inference_audit(text, model, vectorizer):
    """
    Calculates probabilities and identifies decision drivers.
    """
    # 1. Pipeline Execution
    emojis = extract_emojis(text)
    engine_input = f"{text} {emojis * 5}"
    vec = vectorizer.transform([engine_input])
    
    # 2. Probability extraction
    probs = model.predict_proba(vec)[0]
    prediction = int(model.predict(vec)[0])
    
    # 3. Decision Drivers (Explainability)
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = vec.nonzero()[1]
    feature_impacts = []
    
    for idx in nonzero_indices:
        token = feature_names[idx]
        # Weight = LogProb(Pos) - LogProb(Neg)
        weight = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
        feature_impacts.append({"token": token, "weight": round(weight, 4)})
    
    # Sort by the tokens with the highest statistical influence
    top_drivers = sorted(feature_impacts, key=lambda x: abs(x['weight']), reverse=True)[:3]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": text,
        "engine_input": engine_input,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "prediction_int": prediction, 
        "confidence": round(float(np.max(probs)), 4),
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.15 else "Clear Signal",
        "top_drivers": top_drivers
    }

def lean_log_append(log_data, log_path: Path):
    """
    Maintains a strictly circular CSV log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten drivers for a clean CSV row
    csv_row = log_data.copy()
    csv_row['top_drivers'] = "|".join([f"{d['token']}" for d in log_data['top_drivers']])
    
    new_entry = pd.DataFrame([csv_row])
    
    if log_path.exists():
        df = pd.read_csv(log_path)
        # Keep only the latest records (Circular Buffer)
        df = pd.concat([df, new_entry], ignore_index=True).tail(MAX_LOG_ENTRIES)
    else:
        df = new_entry
        
    df.to_csv(log_path, index=False)

def get_global_model_signals(model, vectorizer, top_n=5):
    """
    Extracts the strongest global signals the model has learned.
    """
    feature_names = vectorizer.get_feature_names_out()
    # Math: Log probability difference across the entire vocabulary
    weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    
    # Create a sorted list of all words and their weights
    vocab_weights = list(zip(feature_names, weights))
    
    top_pos = sorted(vocab_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_neg = sorted(vocab_weights, key=lambda x: x[1], reverse=False)[:top_n]
    
    return {
        "top_positive_signals": [{"token": t, "weight": round(w, 2)} for t, w in top_pos],
        "top_negative_signals": [{"token": t, "weight": round(w, 2)} for t, w in top_neg]
    }