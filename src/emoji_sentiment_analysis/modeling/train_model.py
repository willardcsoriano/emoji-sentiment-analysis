# emoji_sentiment_analysis/modeling/train_model.py

"""
Final Model Training Script
---------------------------
Trains the production sentiment classifier using the hybrid 
feature dataset (TF-IDF + Emoji Lexicon + Word Lexicon).
"""

from __future__ import annotations

import random
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import hstack, spmatrix
from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Centralized Configuration Sync
from emoji_sentiment_analysis.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SEED,
    TEXT_COL,
    TARGET_COL,
    ensure_dirs,
    init_logging
)

def train_pipeline():
    # 1. Setup
    init_logging()
    ensure_dirs()
    
    random.seed(SEED)
    np.random.seed(SEED)

    # 2. Load Dataset
    data_path = PROCESSED_DATA_DIR / "features_final.csv"
    logger.info(f"Loading feature dataset from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Features file not found. Run build_features.py first.")
        return

    # Validation: Ensure the new Lexicon columns exist
    required_columns = {
        TEXT_COL, 
        TARGET_COL, 
        "emoji_pos_count", 
        "emoji_neg_count",
        "word_pos_count",
        "word_neg_count"
    }
    
    if not required_columns.issubset(df.columns):
        logger.error(f"Dataset missing required columns: {required_columns - set(df.columns)}")
        return

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    # 3. Train / Validation Split
    logger.info("Splitting dataset (80/20)...")
    X_text = df[TEXT_COL]
    
    # Select all four numeric features
    X_numeric = df[[
        "emoji_pos_count", 
        "emoji_neg_count", 
        "word_pos_count", 
        "word_neg_count"
    ]]
    y = df[TARGET_COL]

    (
        X_train_text,
        X_val_text,
        X_train_num,
        X_val_num,
        y_train,
        y_val,
    ) = train_test_split(
        X_text,
        X_numeric,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    # 4. Text Vectorization
    logger.info("Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_text_vec = tfidf.fit_transform(X_train_text)
    X_val_text_vec = tfidf.transform(X_val_text)

    # 5. Feature Assembly
    logger.info("Combining TF-IDF + Hybrid Numeric features...")
    
    # Combine sparse TF-IDF with the 4 numeric columns
    X_train_final = cast(spmatrix, hstack([X_train_text_vec, X_train_num.to_numpy()]).tocsr())
    X_val_final = cast(spmatrix, hstack([X_val_text_vec, X_val_num.to_numpy()]).tocsr())

    # 6. Model Training
    logger.info(f"Training Logistic Regression (Seed: {SEED})...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(X_train_final, y_train)

    # 7. Evaluation
    logger.info("Evaluating model performance...")
    y_pred = model.predict(X_val_final)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print("\n" + "="*30)
    print(" VALIDATION PERFORMANCE ")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_val, y_pred))
    print("="*30 + "\n")

    # 8. Artifact Serialization
    logger.info(f"Saving artifacts to {MODELS_DIR}...")
    joblib.dump(model, MODELS_DIR / "sentiment_model.pkl")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")

    logger.success("Training pipeline complete.")

if __name__ == "__main__":
    train_pipeline()