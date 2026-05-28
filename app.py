# app.py - Main FastAPI application for Hybrid Sentiment Engine

import asyncio
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR
from emoji_sentiment_analysis.modeling.lite_model import LiteModel
from emoji_sentiment_analysis.modeling.predict import predict_sentiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the lightweight (scikit-learn-free) model artifacts on startup."""
    app.state.lite = await asyncio.to_thread(LiteModel.load, MODELS_DIR)
    print("✅ Lite model loaded (scikit-learn-free)")
    yield
    del app.state.lite
    print("🛑 System shutdown complete.")


app = FastAPI(
    title="Hybrid Sentiment Engine",
    description=(
        "A production-aligned pipeline synthesizing text vectors and deterministic emoji signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
templates.env.filters["uppercase"] = lambda s: str(s).upper()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    start_time = time.perf_counter()

    result = predict_sentiment(text, lite=request.app.state.lite)

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
