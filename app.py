# app.py - Main FastAPI application for Emoji-Aware Sentiment Analysis Engine

import joblib
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Modular Imports
from emoji_sentiment_analysis.config import MODELS_DIR
from emoji_sentiment_analysis.services.audit_service import generate_inference_audit

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model ONCE when the server starts."""
    try:
        app.state.model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
        app.state.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        print("✅ Production artifacts loaded.")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
    yield

app = FastAPI(title="Emoji-Aware Sentiment Analysis Engine", lifespan=lifespan)

# Templates look in the root /templates folder
templates = Jinja2Templates(directory="templates")
templates.env.filters["uppercase"] = lambda s: str(s).upper()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    # Pull the pre-loaded model from app state (Fast & Stable)
    model = request.app.state.model
    vectorizer = request.app.state.vectorizer

    # Run the math in-memory (No logging to disk)
    result = generate_inference_audit(text, model, vectorizer)

    return templates.TemplateResponse(
        "components/prediction_result.html", 
        {"request": request, "result": result}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)