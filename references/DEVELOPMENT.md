# Development Commands

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `make requirements` |
| Format code | `make format` |
| Lint code | `make lint` |
| Train model | `make train` |
| Run locally | `make dev` |
| Deploy (code only) | `make deploy` |
| Train + deploy | `make deploy-full` |

---

## Workflows

### First-time setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
make requirements
```

### Daily development
```bash
make format
make lint
make dev
```

### After installing a new package
```bash
pip install <package>
make freeze
```

### Deploy code-only changes
```bash
make deploy
```

### Retrain model + deploy
```bash
make deploy-full
```

---

## Dependency Management

```bash
make requirements       # Install all deps + editable install
make freeze             # Regenerate & scrub requirements.txt for Linux
```

> `make freeze` strips Windows-only packages (pywinpty, win32, etc.) that break Linux Docker builds.

---

## Code Quality

```bash
make format             # Auto-fix formatting (ruff)
make lint               # Check for issues (ruff)
make clean              # Remove .pyc and __pycache__
```

---

## Testing

```bash
pytest                          # Run all tests
pytest -v                       # Verbose
pytest --cov=emoji_sentiment_analysis  # With coverage
pytest src/tests/test_features.py      # Single file
```

---

## Local Development

```bash
make dev                # Hot reload, port 8000
make serve              # Production mode, port 8080
make dev-mobile         # Accessible on local network (same Wi-Fi)
```

---

## Model Training

```bash
make train
```

Trains the classifier, saves artifacts to `src/models/`, and uploads to GCS:
```
gs://hybrid-sentiment-models/models/sentiment_model.pkl
gs://hybrid-sentiment-models/models/tfidf_vectorizer.pkl
```

---

## Cloud Deployment

```bash
make deploy             # Deploy current code (models loaded from GCS)
make deploy-full        # Retrain → upload to GCS → deploy
make logs               # View recent Cloud Run logs
make stream-logs        # Stream live production logs
```

### Useful GCP commands

```bash
# List services
gcloud run services list

# Describe service
gcloud run services describe emoji-sentiment-app --region us-central1

# Stream live logs (run in separate terminal while testing)
gcloud beta run services logs tail emoji-sentiment-app --region=us-central1 --project=hybrid-sentiment-analysis

# View recent revisions
gcloud run revisions list --service=emoji-sentiment-app --limit=5
```

---

## Virtual Environment

```bash
# Create
python -m venv .venv

# Activate — Windows
.\.venv\Scripts\Activate.ps1

# Activate — Linux/Mac
source .venv/bin/activate

# Deactivate
deactivate
```

---

## Git

```bash
git status
git add .
git commit -m "message"
git push origin main
git checkout -b feature/branch-name
git pull origin main
```
