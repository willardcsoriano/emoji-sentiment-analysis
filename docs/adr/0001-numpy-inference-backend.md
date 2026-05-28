# 1. Serve inference from exported NumPy weights

## Overview

The deployed service had a ~14.5s cold start on Cloud Run because every scale-up
rebuilt a fresh container and imported scikit-learn + scipy purely to unpickle
the model and run a TF-IDF projection plus a logistic decision — work that is
mathematically a sparse matrix multiply and a sigmoid. With a no-recurring-cost
constraint (so warm `min-instances` was off the table), we removed scikit-learn
and scipy from the serving path entirely: a pure-NumPy backend reproduces the
`TfidfVectorizer` + `LogisticRegression` pipeline, loading lightweight exported
weights instead of pickled estimators. scikit-learn is now used only offline to
train and to export, and an automated parity gate guarantees the NumPy path
matches scikit-learn bit-for-bit before anything ships. This record explains the
decision, the parity contract, the alternatives weighed, and the maintenance
obligations it creates.

## Table of Contents

- [Overview](#overview)
- [Status](#status)
- [Context](#context)
- [Decision](#decision)
- [Consequences](#consequences)
- [Alternatives considered](#alternatives-considered)
- [Maintenance](#maintenance)

## Status

Accepted — 2026-05-28.

## Context

- The Cloud Run service runs with `--min-instances 0` (scale-to-zero), so an
  idle service pays a full cold start on the next request: ~14.5s measured.
- The cold start is dominated by **container scheduling + image pull** and the
  **scikit-learn/scipy import** on a fresh interpreter — *not* by model storage.
  The `.pkl` artifacts are already baked into the image and load from local disk;
  relocating them changes nothing.
- A latent issue compounded the image size: the `Dockerfile` ran `pip install .`,
  whose `pyproject.toml` dynamic dependencies resolve to the *dev* `requirements.txt`
  (scikit-learn, scipy, pandas, …), re-bloating the runtime image even though
  `requirements-prod.txt` was meant to be lean.
- Hard constraint: **no recurring cost.** Keeping an instance warm
  (`--min-instances 1`, ~$10–12/mo) was explicitly rejected.
- Inference is a hybrid model: an L2-normalized TF-IDF vector (`ngram_range=(1,2)`,
  `min_df=2`, `max_df=0.95`) concatenated with four deterministic numeric features
  (`emoji_pos/neg_count`, `word_pos/neg_count`), fed to a binary
  `LogisticRegression`, with a deterministic sarcasm-veto layer on top. The
  feature extraction and veto were already pure Python; only the TF-IDF transform
  and the logistic decision required scikit-learn/scipy.

## Decision

Serve inference from a scikit-learn-free NumPy backend:

- `modeling/lite_model.py` (`LiteModel`) reproduces the pipeline exactly —
  the word analyzer (lowercase → `token_pattern` → `_word_ngrams`), TF-IDF
  (`tf × idf`, L2 normalization), and the binary logistic decision
  (`predict = 1 if x·coef + b > 0`) — using only NumPy and the standard library.
- `scripts/export_lite_model.py` converts the trained `.pkl` artifacts into
  lightweight weights (`model_lite.npz` + `model_lite.json`) and runs a **parity
  gate**: labels must be identical and probabilities within `1e-9` across an
  adversarial battery, or it aborts. It also asserts the vectorizer config it
  relies on (`analyzer='word'`, `norm='l2'`, `sublinear_tf=False`, no stop words /
  accent stripping), so a future retrain with incompatible settings fails loudly
  rather than shipping a wrong model.
- The serving graph (`app.py` → `predict.py` → `audit_service.py`) imports
  neither scikit-learn nor scipy. `requirements-prod.txt` drops scikit-learn,
  scipy, joblib, and threadpoolctl; the `Dockerfile` installs the package with
  `--no-deps`; `.dockerignore` excludes the `.pkl` files from the runtime image.
- The export runs at build time (`cloudbuild.yaml` step 1, in an offline
  sklearn-equipped step) and is also wired into `make train`. scikit-learn/scipy
  remain in `requirements.txt` for training and export only.

## Consequences

**Positive**

- No scikit-learn/scipy import on the serving path → a major slice of the cold
  start removed, and a much smaller runtime image (faster pull).
- The `--no-deps` fix stops the dev dependencies leaking into the runtime image.
- The parity gate makes correctness a build-time guarantee, not a hope.

**Obligations / trade-offs**

- The lite weights must be re-exported whenever the model is retrained. This is
  automated: `make train` runs the export, and Cloud Build re-exports from the
  GCS `.pkl`s on every deploy.
- `LiteModel` reproduces the *current* `TfidfVectorizer` configuration. Changing
  it (e.g. enabling `sublinear_tf`, adding stop words, a custom analyzer) requires
  updating `LiteModel` too; the export-time assertions and parity gate prevent a
  silent mismatch.
- One extra build step (the export) and a build-time scikit-learn install.

**Parity contract**

Identical labels and probabilities within `1e-9` of scikit-learn. Enforced in CI
by `tests/test_lite_parity.py` (synthetic + real artifacts) and at deploy time by
the Cloud Build export gate.

## Alternatives considered

- **`--min-instances 1` (keep an instance warm).** Eliminates the cold start with
  one flag, but costs ~$10–12/mo. Rejected by the no-recurring-cost constraint.
- **Keep-warm pinger** (a free scheduler hitting the endpoint every few minutes).
  Effectively free and near-eliminates the cold start, but is a workaround that
  doesn't address the underlying runtime weight; not chosen.
- **Dedicated / always-on host (VPS, Fly.io, Render).** Eliminates cold start but
  adds ops burden and/or cost; rejected.
- **Do nothing.** A ~14.5s first-request latency was deemed unacceptable.

## Maintenance

- Retrain + export locally: `make train` (trains, then exports + parity-gates).
- Export only, from existing `.pkl`s: `make export`.
- Deploy: `make deploy` → Cloud Build downloads the `.pkl`s, re-exports with the
  parity gate, builds the slim image, and deploys.
