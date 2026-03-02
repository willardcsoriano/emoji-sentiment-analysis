# app.py - Main FastAPI application for Hybrid Sentiment Engine

# app.py - Main FastAPI application for Hybrid Sentiment Engine

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD
from emoji_sentiment_analysis.modeling.predict import predict_sentiment

GCS_BUCKET = "hybrid-sentiment-models"
GCS_MODEL_PREFIX = "models"


def download_artifacts() -> Path:
    """Download model artifacts from GCS to a temp directory."""
    tmp_dir = Path(tempfile.mkdtemp())
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for filename in ["sentiment_model.pkl", "tfidf_vectorizer.pkl"]:
        blob = bucket.blob(f"{GCS_MODEL_PREFIX}/{filename}")
        dest = tmp_dir / filename
        blob.download_to_filename(dest)
        print(f"✅ Downloaded {filename} from GCS")

    return tmp_dir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Downloads artifacts from GCS on startup."""
    try:
        artifacts_dir = download_artifacts()
        app.state.model = joblib.load(artifacts_dir / "sentiment_model.pkl")
        app.state.vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.pkl")
        print("✅ Production artifacts loaded from GCS")
        yield
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        raise
    finally:
        if hasattr(app.state, "model"):
            del app.state.model
        if hasattr(app.state, "vectorizer"):
            del app.state.vectorizer
        print("🛑 System shutdown complete.")


app = FastAPI(
    title="Hybrid Sentiment Engine",
    description=(
        "A production-aligned pipeline synthesizing text vectors and deterministic emoji signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")
templates.env.filters["uppercase"] = lambda s: str(s).upper()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    start_time = time.perf_counter()

    result = predict_sentiment(
        text,
        model=request.app.state.model,
        tfidf=request.app.state.vectorizer,
    )

    execution_time_ms = (time.perf_counter() - start_time) * 1000
    result["latency"] = round(execution_time_ms, 2)
    result["prediction_int"] = 1 if result["prediction"] == "Positive" else 0

    if "entropy_flag" not in result:
        result["entropy_flag"] = (
            "High Ambiguity" if result["confidence"] < AMBIGUITY_THRESHOLD else "Clear Signal"
        )

    return templates.TemplateResponse(
        "components/header/prediction_result.html",
        {"request": request, "result": result},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
