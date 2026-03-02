# app.py - Main FastAPI application for Hybrid Sentiment Engine

import os
import time
from contextlib import asynccontextmanager

import joblib
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Modular Imports
from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR
from emoji_sentiment_analysis.modeling.predict import predict_sentiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads artifacts on startup and cleans up on shutdown."""
    model_path = MODELS_DIR / "sentiment_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    try:
        app.state.model = joblib.load(model_path)
        app.state.vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Production artifacts loaded: {MODELS_DIR}")
        yield
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        yield
    finally:
        # Graceful cleanup
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
    """
    Unified Prediction Route:
    Captures high-precision execution latency to verify performance claims.
    """
    # 1. Start high-precision timer
    start_time = time.perf_counter()

    # 2. Execute the hybrid inference pipeline
    result = predict_sentiment(text)

    # 3. Calculate execution latency in milliseconds
    execution_time_ms = (time.perf_counter() - start_time) * 1000

    # 4. Inject metadata for the UI
    result["latency"] = round(execution_time_ms, 2)
    result["prediction_int"] = 1 if result["prediction"] == "Positive" else 0

    # 5. Handle Ambiguity Flagging
    if "entropy_flag" not in result:
        result["entropy_flag"] = (
            "High Ambiguity" if result["confidence"] < AMBIGUITY_THRESHOLD else "Clear Signal"
        )

    # 6. Render the result component
    return templates.TemplateResponse(
        "components/header/prediction_result.html",
        {"request": request, "result": result},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
