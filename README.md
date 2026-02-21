# Hybrid Sentiment Engine: Emoji-Augmented NLP Pipeline

A lexicon-augmented NLP pipeline for sentiment inference on emoji-enriched social media text.

**Live Deployment:**
[https://hybrid-sentiment-service-232900046311.us-central1.run.app/](https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app)
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

* Emoji-aware sentiment modeling
* Lexicon-augmented feature engineering
* TF-IDF linguistic vectorization
* Logistic Regression classification
* Reproducible training + inference parity
* API-driven prediction service
* Inference audit logging
* Model health reporting

---

## Live Demo

Try the deployed inference interface:

**[https://hybrid-sentiment-service-232900046311.us-central1.run.app/](https://emoji-sentiment-app-5e62vyvgya-uc.a.run.app)**

Features:

* Real-time sentiment prediction
* Emoji signal interpretation
* Confidence scoring
* Prediction logging

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

* Tokenization
* N-gram extraction
* TF-IDF weighting

### Emoji Signal Processing

* Emoji detection within tweets
* Polarity referencing via lexicon dataset
* Derived sentiment feature signals

### Classifier

* Logistic Regression
* Linear decision boundary
* Interpretable feature coefficients

---

## Project Structure

```
emoji_sentiment_analysis/
│
├── app.py                  # FastAPI web entrypoint
├── Dockerfile              # Container spec for Cloud Run
├── cloudbuild.yaml         # CI/CD pipeline configuration
├── requirements.txt        # Production dependencies
├── Makefile                # Task automation (build, test, etc.)
│
├── templates/              # HTMX/Frontend UI templates
│
└── src/
    ├── main.py             # Runtime orchestration & API logic
    ├── .dockerignore       # Docker build optimization
    │
    ├── emoji_sentiment_analysis/   # Core Package
    │   ├── dataset.py      # Data loading logic
    │   ├── features.py     # Feature engineering (Emoji-to-Sentiment)
    │   ├── config.py       # Project-wide constants
    │   ├── services/       # Business logic (Audit/Logging)
    │   └── modeling/       # Training & Inference pipelines
    │
    ├── data/               # Versioned data layers
    │   ├── raw/            # Original Kaggle/Scraped datasets
    │   ├── processed/      # Cleaned tweets and features
    │   └── logs/           # Application runtime logs
    │
    ├── models/             # Serialized artifacts
    │   ├── sentiment_model.pkl
    │   └── tfidf_vectorizer.pkl
    │
    ├── notebooks/          # Complete ML lifecycle (Exploration to Validation)
    ├── scripts/            # Model health & feature building utilities
    └── tests/              # Pytest suite for model behavior & features
```

---

## Data Assets

### Tweet Corpus

* Emoji-bearing tweets
* Binary sentiment labels
* Short-form informal text

### Emoji Reference Lexicon

* Emoji / emoticon symbols
* Polarity annotations
* Used for sentiment referencing (not training rows)

---

## Reproducibility

To run locally:

```bash
# Clone repository
git clone https://github.com/<your-username>/emoji_sentiment_analysis.git
cd emoji_sentiment_analysis

# Create environment
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate (Windows)

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

---

## Docker Deployment

```bash
docker build -t emoji-sentiment .
docker run -p 8080:8080 emoji-sentiment
```

---

## Testing

```bash
pytest src/tests
```

Covers:

* Feature consistency
* Model behavior
* Prediction stability

---

## Monitoring & Reporting

Generated artifacts include:

* `inference_history.csv` — prediction logs
* `model_health_report.json` — performance diagnostics

---

## Notebooks

Lifecycle notebooks document the full experimentation process:

1. Data Ingestion & Exploration
2. Cleaning & Validation
3. Representation & Feature Design
4. Modeling & Evaluation
5. Final Training
6. Interpretability Analysis
7. Inference Reconstruction

---

## Tech Stack

* **Language:** Python 3.11+
* **ML Library:** Scikit-learn (Sentiment Analysis)
* **Web Framework:** FastAPI + HTMX + TailwindCSS
* **Containerization:** Docker
* **Cloud Infrastructure:** Google Cloud Run (Serverless Hosting)
* **CI/CD Pipeline:** Google Cloud Build (Automated Triggers on Push)
* **Artifact Management:** Google Artifact Registry

---

## Author

Built by **Willard C. Soriano**

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---
