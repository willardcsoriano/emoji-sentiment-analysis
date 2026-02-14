# Hybrid Sentiment Engine: Emoji-Augmented NLP Pipeline

A lexicon-augmented NLP pipeline for sentiment inference on emoji-enriched social media text.

**Live Deployment:**
https://hybrid-sentiment-service-232900046311.us-central1.run.app/

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

**https://hybrid-sentiment-service-232900046311.us-central1.run.app/**

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
├── app.py                      # Application entrypoint
├── Dockerfile                 # Container build spec
├── requirements.txt           # Python dependencies
│
├── templates/                 # Web UI templates
│
├── src/
│   ├── main.py                # Runtime orchestration
│   │
│   ├── data/
│   │   ├── raw/               # Original datasets
│   │   ├── processed/         # Cleaned + engineered data
│   │   └── logs/              # Inference history
│   │
│   ├── emoji_sentiment_analysis/
│   │   ├── dataset.py         # Data handling
│   │   ├── features.py        # Feature engineering
│   │   ├── config.py          # Configuration
│   │   │
│   │   ├── modeling/
│   │   │   ├── train_model.py
│   │   │   └── predict.py
│   │   │
│   │   └── services/
│   │       └── audit_service.py
│   │
│   ├── models/
│   │   ├── sentiment_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   │
│   ├── notebooks/             # Full ML lifecycle notebooks
│   ├── scripts/               # Feature + health utilities
│   └── tests/                 # Pipeline validation tests
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

* Python
* Scikit-learn
* FastAPI
* HTMX
* TailwindCSS
* Docker
* Google Cloud Run  
* Google Cloud Build (CI/CD)  
* Artifact Registry

---

## Author

Built by **Willard C. Soriano**

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---
