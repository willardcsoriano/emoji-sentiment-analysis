# emoji_sentiment_analysis/modeling/train_final_model.py

"""
Final Model Training Script
---------------------------

Trains the production sentiment classifier using the frozen
feature dataset and serializes all deployment artifacts.

Inputs:
    data/processed/features_final.csv

Outputs:
    models/sentiment_model.pkl
    models/tfidf_vectorizer.pkl

This script mirrors Notebook 4.0 but executes deterministically
for reproducible training and deployment pipelines.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from scipy.sparse import hstack


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------------------
# Paths (relative to project root)
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/processed/features_final.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "text",
    "label",
    "emoji_pos_count",
    "emoji_neg_count",
}

assert required_columns.issubset(df.columns), "Missing required columns."
assert df.isna().sum().sum() == 0, "Dataset contains null values."

print(f"Dataset shape: {df.shape}")


# -------------------------------------------------------------------
# Train / Validation Split
# -------------------------------------------------------------------

print("Splitting dataset...")

X_text = df["text"]
X_emoji = df[["emoji_pos_count", "emoji_neg_count"]]
y = df["label"]

(
    X_train_text,
    X_val_text,
    X_train_emoji,
    X_val_emoji,
    y_train,
    y_val,
) = train_test_split(
    X_text,
    X_emoji,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED,
)


# -------------------------------------------------------------------
# Text Vectorization
# -------------------------------------------------------------------

print("Vectorizing text...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_train_text_vec = tfidf.fit_transform(X_train_text)
X_val_text_vec = tfidf.transform(X_val_text)


# -------------------------------------------------------------------
# Feature Assembly
# -------------------------------------------------------------------

print("Combining text + emoji features...")

X_train_emoji_np = X_train_emoji.to_numpy()
X_val_emoji_np = X_val_emoji.to_numpy()

X_train_final = hstack([X_train_text_vec, X_train_emoji_np])
X_val_final = hstack([X_val_text_vec, X_val_emoji_np])


# -------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
)

model.fit(X_train_final, y_train)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

print("Evaluating model...")

y_pred = model.predict(X_val_final)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("\nValidation Performance")
print("----------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}\n")

print(classification_report(y_val, y_pred))


# -------------------------------------------------------------------
# Artifact Serialization
# -------------------------------------------------------------------

print("Saving artifacts...")

joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")

print("\nArtifacts saved to /models/")
print("- sentiment_model.pkl")
print("- tfidf_vectorizer.pkl")


# -------------------------------------------------------------------
# Training Closure
# -------------------------------------------------------------------

print("\nTraining pipeline complete.")
