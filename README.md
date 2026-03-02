# Hybrid Sentiment Engine: Emoji-Augmented NLP Pipeline

A lexicon-augmented NLP pipeline for sentiment inference on emoji-enriched social media text.

**Live Deployment:**
[https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app](https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app)

> ⚠️ **Heads up:** This service is deployed serverlessly on Google Cloud Run.
> If the endpoint hasn't been called recently, the first request may take
> **5–15 seconds** to respond due to a cold start. Subsequent requests will be fast.

---

## Overview

This project implements an end-to-end machine learning system designed to infer sentiment polarity from short-form social media text containing emojis.

Unlike conventional sentiment pipelines that discard symbolic tokens, this system preserves and interprets emoji signals through lexicon-based sentiment referencing. The result is a hybrid feature space combining linguistic representations with emoji-derived polarity features.

The system is fully reproducible — spanning data ingestion, feature engineering, model training, interpretability analysis, and production inference deployment.

---

## Key Capabilities

- Emoji-aware sentiment modeling
- Lexicon-augmented feature engineering
- TF-IDF linguistic vectorization
- Logistic Regression classification
- Reproducible training + inference parity
- API-driven prediction service
- Deterministic sarcasm veto layer
- Model health reporting

---

## Live Demo

Try the deployed inference interface:

**[https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app](https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app)**

Features:

- Real-time sentiment prediction
- Emoji signal interpretation
- Confidence scoring
- Top 3 decision driver breakdown

---

## Machine Learning Pipeline

```
Data Ingestion
      ↓
Data Cleaning & Validation
      ↓
Exploratory Analysis
      ↓
Emoji Sentiment Referencing
      ↓
Feature Engineering
      ↓
Model Training & Evaluation
      ↓
Inference Pipeline Deployment
```

---

## Modeling Approach

### Text Representation

- Tokenization
- N-gram extraction
- TF-IDF weighting

### Emoji Signal Processing

- Emoji detection within tweets
- Polarity referencing via curated lexicon
- Derived sentiment feature signals (emoji_pos_count, emoji_neg_count)

### Word Lexicon Layer

- Deterministic word polarity counts (word_pos_count, word_neg_count)
- Anchors probabilistic model with stable priors

### Classifier

- Logistic Regression
- Linear decision boundary
- Interpretable feature coefficients

### Sarcasm Veto

- Deterministic override layer
- Fires when emoji_neg signal conflicts with model's positive prediction
- Suppressed by word context guard when positive text is strong enough

---

## Project Structure

```
emoji_sentiment_analysis/
│
├── app.py                  # FastAPI web entrypoint
├── Dockerfile              # Container spec for Cloud Run
├── cloudbuild.yaml         # CI/CD pipeline configuration
├── requirements.txt        # Production dependencies
├── Makefile                # Task automation (build, train, deploy)
│
├── templates/              # HTMX/Frontend UI templates
│
└── src/
    ├── main.py             # Runtime orchestration
    ├── .dockerignore       # Docker build optimization
    │
    ├── emoji_sentiment_analysis/   # Core Package
    │   ├── dataset.py      # Data loading logic
    │   ├── features.py     # Feature engineering (Emoji-to-Sentiment)
    │   ├── config.py       # Project-wide constants
    │   ├── services/       # Inference audit service
    │   └── modeling/       # Training & inference pipelines
    │
    ├── data/               # Versioned data layers
    │   ├── raw/            # Original datasets
    │   ├── processed/      # Cleaned tweets and features
    │   └── logs/           # Application runtime logs
    │
    ├── models/             # Local artifact cache (gitignored)
    │   ├── sentiment_model.pkl     # ← served from GCS in production
    │   └── tfidf_vectorizer.pkl    # ← served from GCS in production
    │
    ├── notebooks/          # Complete ML lifecycle (Exploration to Validation)
    ├── scripts/            # Model health & feature building utilities
    └── tests/              # Pytest suite for model behavior & features
```

> **Note on model artifacts:** `.pkl` files are excluded from version control. In production, artifacts are stored in Google Cloud Storage (`gs://hybrid-sentiment-models/models/`) and downloaded by the container at startup. Locally, run `make train` to regenerate them.

---

## Reproducibility

### Local Development

```bash
# Clone repository
git clone https://github.com/<your-username>/emoji_sentiment_analysis.git
cd emoji_sentiment_analysis

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
make requirements

# Train model (generates local .pkl artifacts)
make train

# Run application
make dev
```

### Production Deployment

Model artifacts are stored in GCS and loaded at container startup — no `.pkl` files are committed to the repository.

```bash
# Deploy code-only changes
make deploy

# Retrain model + upload to GCS + deploy
make deploy-full
```

---

## Docker Deployment

```bash
docker build -t emoji-sentiment .
docker run -p 8000:8080 -e PORT=8080 emoji-sentiment
```

---

## Testing

```bash
pytest src/tests
```

Covers:

- Feature consistency
- Model behavior
- Prediction stability

---

## Notebooks

Lifecycle notebooks document the full experimentation process:

1. Data Ingestion & Exploration
2. Cleaning & Validation
3. Representation & Feature Design
4. Modeling & Evaluation
5. Final Training
6. Interpretability & Performance Deep Dive
7. End-to-End System Validation

---

## Tech Stack

- **Language:** Python 3.12
- **ML Library:** Scikit-learn (TF-IDF + Logistic Regression)
- **Web Framework:** FastAPI + HTMX + TailwindCSS
- **Containerization:** Docker
- **Cloud Infrastructure:** Google Cloud Run (Serverless)
- **CI/CD Pipeline:** Google Cloud Build
- **Artifact Registry:** Google Artifact Registry
- **Model Storage:** Google Cloud Storage (GCS)

---

## Author

Built by **Willard C. Soriano**

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
