# src/emoji_sentiment_analysis/modeling/lite_model.py

"""
Pure-NumPy Inference Backend
----------------------------
Reproduces the trained ``TfidfVectorizer`` + ``LogisticRegression`` pipeline
using only NumPy and the standard library, loading lightweight exported weights
instead of pickled scikit-learn objects.

Why this exists
~~~~~~~~~~~~~~~~
Inference for this model is mathematically a sparse TF-IDF projection followed
by a logistic decision. Importing scikit-learn + scipy at runtime just to do
that costs hundreds of MB of image size and several seconds of cold-start
import time. This module removes both from the serving path; scikit-learn is
needed only offline to *train* and *export* (see ``scripts/export_lite_model.py``).

Parity contract
~~~~~~~~~~~~~~~~
The output must be identical to the scikit-learn pipeline. ``export_lite_model``
enforces this with a parity gate before any artifact is shipped. The replication
mirrors scikit-learn's defaults exactly: word analyzer, ``lowercase`` then
``token_pattern`` tokenization, ``_word_ngrams`` joining with a single space,
raw term-frequency (``sublinear_tf=False``) times the fitted ``idf_``, ``l2``
row normalization, and a binary logistic decision (``decision = x·coef + b``,
``predict = 1 if decision > 0``).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Numeric hybrid features appended after the TF-IDF block, in this exact order.
# Mirrors the hstack order in modeling/train_model.py.
HYBRID_FEATURE_NAMES: list[str] = [
    "emoji_pos_count",
    "emoji_neg_count",
    "word_pos_count",
    "word_neg_count",
]

ARTIFACT_NPZ = "model_lite.npz"
ARTIFACT_JSON = "model_lite.json"


class LiteModel:
    """In-memory, scikit-learn-free sentiment model."""

    def __init__(
        self,
        feature_names: list[str],
        idf: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        classes: np.ndarray,
        ngram_range: tuple[int, int],
        lowercase: bool,
        token_pattern: str,
    ) -> None:
        self.feature_names = list(feature_names)
        self.n_text = len(self.feature_names)
        # term -> column index, for the sparse TF-IDF lookup.
        self._vocab = {term: i for i, term in enumerate(self.feature_names)}
        self.idf = np.asarray(idf, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.classes = np.asarray(classes)
        self.ngram_min, self.ngram_max = int(ngram_range[0]), int(ngram_range[1])
        self.lowercase = bool(lowercase)
        self._token_re = re.compile(token_pattern)

        # Full feature-name axis (TF-IDF terms ‖ hybrid numeric), aligned to coef.
        self.all_feature_names = self.feature_names + HYBRID_FEATURE_NAMES

        if self.idf.shape[0] != self.n_text:
            raise ValueError(f"idf length {self.idf.shape[0]} != vocab size {self.n_text}")
        if self.coef.shape[0] != self.n_text + len(HYBRID_FEATURE_NAMES):
            raise ValueError(
                f"coef length {self.coef.shape[0]} != "
                f"{self.n_text + len(HYBRID_FEATURE_NAMES)} (text + hybrid)"
            )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_params(cls, params: dict) -> "LiteModel":
        return cls(
            feature_names=params["feature_names"],
            idf=params["idf"],
            coef=params["coef"],
            intercept=params["intercept"],
            classes=params["classes"],
            ngram_range=params["ngram_range"],
            lowercase=params["lowercase"],
            token_pattern=params["token_pattern"],
        )

    @classmethod
    def load(cls, models_dir: Path) -> "LiteModel":
        meta = json.loads((Path(models_dir) / ARTIFACT_JSON).read_text(encoding="utf-8"))
        arrays = np.load(Path(models_dir) / ARTIFACT_NPZ)
        return cls(
            feature_names=meta["feature_names"],
            idf=arrays["idf"],
            coef=arrays["coef"],
            intercept=float(np.asarray(arrays["intercept"]).reshape(-1)[0]),
            classes=arrays["classes"],
            ngram_range=tuple(meta["ngram_range"]),
            lowercase=meta["lowercase"],
            token_pattern=meta["token_pattern"],
        )

    # ------------------------------------------------------------------ #
    # TF-IDF replication
    # ------------------------------------------------------------------ #
    def _analyze(self, text: str) -> list[str]:
        """Replicate scikit-learn's word analyzer: lowercase, regex tokenize,
        then ``_word_ngrams`` (unigrams plus consecutive n-grams joined by a
        single space)."""
        if self.lowercase:
            text = text.lower()
        tokens = self._token_re.findall(text)

        if self.ngram_max == 1:
            return tokens

        # Mirrors sklearn CountVectorizer._word_ngrams exactly.
        out: list[str] = list(tokens) if self.ngram_min == 1 else []
        n_tokens = len(tokens)
        min_n = max(self.ngram_min, 2)
        for n in range(min_n, min(self.ngram_max + 1, n_tokens + 1)):
            for i in range(n_tokens - n + 1):
                out.append(" ".join(tokens[i : i + n]))
        return out

    def _transform(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, values)`` of the L2-normalized TF-IDF row over the
        fitted vocabulary. Out-of-vocabulary terms are dropped, matching
        ``TfidfVectorizer.transform``."""
        counts = Counter(g for g in self._analyze(text) if g in self._vocab)
        if not counts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        indices = np.fromiter((self._vocab[g] for g in counts), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        values = tf * self.idf[indices]

        norm = float(np.sqrt(np.dot(values, values)))
        if norm > 0.0:
            values = values / norm
        return indices, values

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(x: float) -> float:
        # Stable logistic, equivalent to scipy.special.expit within float eps.
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        z = np.exp(x)
        return z / (1.0 + z)

    def predict(
        self, text: str, e_pos: int, e_neg: int, w_pos: int, w_neg: int
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run the logistic decision over the hybrid feature row.

        Returns ``(prediction, probs, nonzero_indices)`` where ``probs`` is
        ``[P(class0), P(class1)]`` and ``nonzero_indices`` is the ascending set
        of active feature columns (TF-IDF terms present ‖ nonzero numerics),
        matching ``scipy.sparse`` ``nonzero()`` ordering for the explainer.
        """
        text_idx, text_vals = self._transform(text)

        decision = float(np.dot(text_vals, self.coef[text_idx])) if text_idx.size else 0.0
        numeric = np.array([e_pos, e_neg, w_pos, w_neg], dtype=np.float64)
        decision += float(np.dot(numeric, self.coef[self.n_text : self.n_text + 4]))
        decision += self.intercept

        p1 = self._sigmoid(decision)
        probs = np.array([1.0 - p1, p1], dtype=np.float64)
        prediction = int(self.classes[1] if decision > 0.0 else self.classes[0])

        numeric_nonzero = [self.n_text + k for k, v in enumerate(numeric) if v != 0.0]
        nonzero_indices = np.concatenate([text_idx, np.array(numeric_nonzero, dtype=np.int64)])
        nonzero_indices.sort()
        return prediction, probs, nonzero_indices
