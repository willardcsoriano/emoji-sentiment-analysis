# src/tests/test_lite_parity.py

"""
Lite Model Parity Suite
-----------------------
Guarantees the scikit-learn-free ``LiteModel`` reproduces the scikit-learn
``TfidfVectorizer`` + ``LogisticRegression`` pipeline exactly.

- ``test_synthetic_parity`` fits a model with the *production training config*
  (``ngram_range=(1,2)``, ``min_df=2``, ``max_df=0.95``) on a synthetic corpus
  and asserts identical labels and probabilities (within 1e-9) across an
  adversarial input battery. This runs without the real artifacts.
- ``test_real_artifact_parity`` runs the same gate against the committed/fetched
  ``.pkl`` files when present, and skips otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn only needed for the offline parity gate")

from scipy.sparse import hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from emoji_sentiment_analysis.config import MODELS_DIR  # noqa: E402
from emoji_sentiment_analysis.features import extract_emoji_polarity_features  # noqa: E402
from emoji_sentiment_analysis.modeling.lite_model import LiteModel  # noqa: E402
from scripts.export_lite_model import (  # noqa: E402
    PARITY_TEXTS,
    extract_lite_params,
    parity_check,
    save_lite_artifacts,
)

# Synthetic corpus with enough repetition to survive min_df=2 and to produce
# bigrams. Domain mirrors the real model (sentiment words + emoji + emoticons).
_SYNTH_CORPUS: list[tuple[str, int]] = [
    ("i love this so much 😊", 1),
    ("i love this project", 1),
    ("this is great and wonderful", 1),
    ("great great day today 😍", 1),
    ("what a wonderful amazing time", 1),
    ("happy happy joy joy 🎉", 1),
    ("the best perfect excellent result", 1),
    ("i love good food", 1),
    ("so good so nice", 1),
    ("nice and beautiful day :)", 1),
    ("awesome work team 👍", 1),
    ("love love love this", 1),
    ("feeling great and happy", 1),
    ("perfect score amazing", 1),
    ("i really love it", 1),
    ("good vibes only 😎", 1),
    ("wonderful wonderful news", 1),
    ("this made me so happy", 1),
    ("beautiful beautiful sunset", 1),
    ("great job everyone", 1),
    ("i hate this so much 😭", 0),
    ("this is bad and terrible", 0),
    ("terrible terrible day today", 0),
    ("what an awful horrible time", 0),
    ("sad sad sad 😢", 0),
    ("the worst bad awful result", 0),
    ("i hate bad food", 0),
    ("so bad so sad", 0),
    ("awful and ugly day :(", 0),
    ("horrible work team 👎", 0),
    ("hate hate hate this", 0),
    ("feeling bad and sad", 0),
    ("worst score terrible", 0),
    ("i really hate it", 0),
    ("bad vibes everywhere 💀", 0),
    ("terrible terrible news", 0),
    ("this made me so sad", 0),
    ("ugly ugly mess", 0),
    ("bad job everyone", 0),
    ("i hate this terrible thing", 0),
]


def _fit_reference_pipeline():
    """Fit a TfidfVectorizer + LogisticRegression exactly as train_model.py does."""
    texts = [t for t, _ in _SYNTH_CORPUS]
    labels = np.array([y for _, y in _SYNTH_CORPUS])

    numeric = np.array([extract_emoji_polarity_features(t) for t in texts], dtype=float)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    text_vec = tfidf.fit_transform(texts)
    features = hstack([text_vec, numeric]).tocsr()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features, labels)
    return model, tfidf


def test_synthetic_parity():
    model, tfidf = _fit_reference_pipeline()
    lite = LiteModel.from_params(extract_lite_params(model, tfidf))

    # Battery = adversarial inputs plus the training corpus itself.
    texts = PARITY_TEXTS + [t for t, _ in _SYNTH_CORPUS]
    max_delta = parity_check(model, tfidf, lite, texts)
    assert max_delta <= 1e-9, f"probability drift too large: {max_delta:.2e}"


def test_serialized_roundtrip_parity(tmp_path):
    """The on-disk artifacts (npz + json) must load back to an identical model."""
    model, tfidf = _fit_reference_pipeline()
    params = extract_lite_params(model, tfidf)
    save_lite_artifacts(params, tmp_path)

    lite = LiteModel.load(tmp_path)
    max_delta = parity_check(model, tfidf, lite, PARITY_TEXTS)
    assert max_delta <= 1e-9


def test_real_artifact_parity():
    """Parity against the actual trained model, when the .pkl files are present."""
    import joblib

    model_path = MODELS_DIR / "sentiment_model.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not (model_path.exists() and vec_path.exists()):
        pytest.skip("real .pkl artifacts not present (run export in an env with the model)")

    model = joblib.load(model_path)
    tfidf = joblib.load(vec_path)
    lite = LiteModel.from_params(extract_lite_params(model, tfidf))
    max_delta = parity_check(model, tfidf, lite, PARITY_TEXTS)
    assert max_delta <= 1e-9
