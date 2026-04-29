# app.py - Main FastAPI application for Hybrid Sentiment Engine

import asyncio
import os
import time
from contextlib import asynccontextmanager

import joblib
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR
from emoji_sentiment_analysis.modeling.predict import predict_sentiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model artifacts from the baked-in image path on startup."""
    app.state.model = await asyncio.to_thread(joblib.load, MODELS_DIR / "sentiment_model.pkl")
    app.state.vectorizer = await asyncio.to_thread(
        joblib.load, MODELS_DIR / "tfidf_vectorizer.pkl"
    )
    print("✅ Artifacts loaded from baked-in path")
    yield
    del app.state.model
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
