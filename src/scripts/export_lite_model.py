# src/scripts/export_lite_model.py

"""
Lite Model Exporter
-------------------
Converts the trained scikit-learn artifacts (``sentiment_model.pkl`` +
``tfidf_vectorizer.pkl``) into lightweight, scikit-learn-free weights
(``model_lite.npz`` + ``model_lite.json``) consumed by
``modeling/lite_model.LiteModel`` at serving time.

This is an *offline / build-time* step: it is the only place that still imports
scikit-learn for inference purposes. It runs a hard **parity gate** comparing
the NumPy pipeline against scikit-learn across a broad battery of inputs and
aborts (non-zero exit) on any divergence, so a mismatched model can never be
deployed.

Usage:
    python src/scripts/export_lite_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow execution as a bare script (cloud build) without installing the package.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Resolve the models directory from this file's location so it is correct
# regardless of cwd or the IS_DOCKER heuristic in config.py — the export runs in
# dev shells and Cloud Build steps, not in the serving container.
MODELS_DIR = _SRC / "models"

from emoji_sentiment_analysis.features import extract_emoji_polarity_features  # noqa: E402
from emoji_sentiment_analysis.modeling.lite_model import (  # noqa: E402
    ARTIFACT_JSON,
    ARTIFACT_NPZ,
    HYBRID_FEATURE_NAMES,
    LiteModel,
)

PROB_TOLERANCE = 1e-9

# Broad, adversarial battery: golden cases, UI presets, and edge inputs that
# stress tokenization (bigrams, repeats, punctuation, unicode, casing, empties).
PARITY_TEXTS: list[str] = [
    "i love you baby",
    "i hate you baby",
    "today was okay 😊",
    "today was okay 😭",
    "i love having bugs 😭",
    "great, another meeting 😧",
    "wow amazing day 😭😭",
    "love hate love 😊",
    "😊😊😊😊",
    "😭😧😔",
    "feeling great today :(",
    "This engineering is top tier! 😍🚀",
    "The service was extremely slow today... 😒",
    "The movie was just okay, I guess.",
    "Everything is great. 💀",
    "I'm having a blast at this party! XD <3",
    "I love waiting in long lines. :-) 😭",
    "The Q4 results were excellent and perfect. 📈",
    "I hate how much I love this! 😂💔",
    "It is what it is.",
    "the table is brown and the chair is wooden",
    "i absolutely hate this terrible experience 😭😭",
    "!!!!!!!!!!!!!!!",
    "",
    "   ",
    "GREAT GREAT great Great",
    "good good good bad bad",
    "a",
    "ab",
    "the quick brown fox jumps over the lazy dog",
    "naïve café résumé coördinate",
    "wonderful amazing excellent perfect beautiful awesome",
    "terrible awful bad sad unhappy hate",
    "MixedCASE Tokens With Punctuation, semicolons; and-dashes",
    "repeat repeat repeat repeat repeat",
]


def extract_lite_params(model, tfidf) -> dict:
    """Pull the minimal weights from fitted sklearn objects, asserting that the
    vectorizer/model configuration matches what ``LiteModel`` replicates."""
    # Guard against silent drift: LiteModel only reproduces these settings.
    assert tfidf.analyzer == "word", f"unsupported analyzer: {tfidf.analyzer!r}"
    assert tfidf.stop_words is None, "stop_words not supported by LiteModel"
    assert tfidf.strip_accents is None, "strip_accents not supported by LiteModel"
    assert tfidf.binary is False, "binary tf not supported by LiteModel"
    assert tfidf.sublinear_tf is False, "sublinear_tf not supported by LiteModel"
    assert tfidf.norm == "l2", f"unsupported norm: {tfidf.norm!r}"
    assert tfidf.use_idf is True, "use_idf must be True"
    assert callable(getattr(tfidf, "build_analyzer", None)), "not a fitted vectorizer"

    coef = np.asarray(model.coef_, dtype=np.float64)
    assert coef.shape[0] == 1, f"expected binary model (coef rows=1), got {coef.shape[0]}"
    classes = np.asarray(model.classes_)
    assert classes.shape[0] == 2, f"expected binary classes, got {classes.tolist()}"

    feature_names = [str(t) for t in tfidf.get_feature_names_out()]
    n_text = len(feature_names)
    coef_flat = coef[0]
    expected = n_text + len(HYBRID_FEATURE_NAMES)
    assert coef_flat.shape[0] == expected, (
        f"coef width {coef_flat.shape[0]} != vocab({n_text}) + hybrid({len(HYBRID_FEATURE_NAMES)})"
    )

    return {
        "feature_names": feature_names,
        "idf": np.asarray(tfidf.idf_, dtype=np.float64),
        "coef": coef_flat,
        "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
        "classes": classes,
        "ngram_range": tuple(int(x) for x in tfidf.ngram_range),
        "lowercase": bool(tfidf.lowercase),
        "token_pattern": str(tfidf.token_pattern),
    }


def save_lite_artifacts(params: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / ARTIFACT_NPZ
    json_path = out_dir / ARTIFACT_JSON

    np.savez(
        npz_path,
        idf=params["idf"],
        coef=params["coef"],
        intercept=np.array([params["intercept"]], dtype=np.float64),
        classes=np.asarray(params["classes"]),
    )
    json_path.write_text(
        json.dumps(
            {
                "feature_names": params["feature_names"],
                "n_text_features": len(params["feature_names"]),
                "hybrid_feature_names": HYBRID_FEATURE_NAMES,
                "ngram_range": list(params["ngram_range"]),
                "lowercase": params["lowercase"],
                "token_pattern": params["token_pattern"],
            }
        ),
        encoding="utf-8",
    )
    return npz_path, json_path


def _sklearn_probs(model, tfidf, text: str) -> tuple[np.ndarray, int]:
    """Reference scikit-learn inference for the (non-veto) model path."""
    from scipy.sparse import csr_matrix, hstack

    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)
    text_vec = tfidf.transform([text])
    numeric_vec = np.array([[e_pos, e_neg, w_pos, w_neg]])
    features = hstack([text_vec, numeric_vec]).tocsr()
    assert isinstance(features, csr_matrix)
    probs = model.predict_proba(features)[0]
    prediction = int(model.predict(features)[0])
    return probs, prediction


def parity_check(model, tfidf, lite: LiteModel, texts: list[str]) -> float:
    """Compare NumPy vs scikit-learn across the battery. Returns the max
    absolute probability delta. Raises AssertionError on any mismatch."""
    max_delta = 0.0
    failures: list[str] = []
    for text in texts:
        e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)
        sk_probs, sk_pred = _sklearn_probs(model, tfidf, text)
        lt_pred, lt_probs, _ = lite.predict(text, e_pos, e_neg, w_pos, w_neg)

        delta = float(np.max(np.abs(sk_probs - lt_probs)))
        max_delta = max(max_delta, delta)
        if sk_pred != lt_pred:
            failures.append(f"label mismatch on {text!r}: sklearn={sk_pred} lite={lt_pred}")
        if delta > PROB_TOLERANCE:
            failures.append(f"prob drift {delta:.2e} on {text!r}: {sk_probs} vs {lt_probs}")

    if failures:
        raise AssertionError("Lite model parity FAILED:\n  " + "\n  ".join(failures))
    return max_delta


def main() -> None:
    import joblib

    model_path = MODELS_DIR / "sentiment_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        print(f"ERROR: artifacts not found in {MODELS_DIR}. Train or fetch them first.")
        sys.exit(1)

    print(f"Loading scikit-learn artifacts from {MODELS_DIR} ...")
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    params = extract_lite_params(model, tfidf)
    n_text = len(params["feature_names"])
    print(f"Vocabulary: {n_text} TF-IDF terms (+{len(HYBRID_FEATURE_NAMES)} hybrid numeric)")

    npz_path, json_path = save_lite_artifacts(params, MODELS_DIR)

    # Re-load from disk so the gate validates the *serialized* artifacts.
    lite = LiteModel.load(MODELS_DIR)
    max_delta = parity_check(model, tfidf, lite, PARITY_TEXTS)

    npz_kb = npz_path.stat().st_size / 1024
    json_kb = json_path.stat().st_size / 1024
    print(f"Parity OK across {len(PARITY_TEXTS)} inputs — max prob delta {max_delta:.2e}")
    print(f"Wrote {npz_path.name} ({npz_kb:.0f} KB) + {json_path.name} ({json_kb:.0f} KB)")


if __name__ == "__main__":
    main()
