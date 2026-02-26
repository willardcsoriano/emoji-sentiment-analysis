# app.py - Main FastAPI application for Hybrid Sentiment Engine

import os
import joblib
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Modular Imports
from emoji_sentiment_analysis.config import MODELS_DIR
from emoji_sentiment_analysis.modeling.predict import predict_sentiment

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model ONCE when the server starts."""
    model_path = MODELS_DIR / "sentiment_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    
    try:
        # Explicitly check for paths so Cloud Run logs the EXACT missing path
        if not model_path.exists():
            print(f"❌ CRITICAL ERROR: Model missing at path -> {model_path}")
        if not vectorizer_path.exists():
            print(f"❌ CRITICAL ERROR: Vectorizer missing at path -> {vectorizer_path}")
            
        # We load them into state, though predict_sentiment likely handles its own loading,
        # keeping this here ensures artifacts are present and readable on boot.
        app.state.model = joblib.load(model_path)
        app.state.vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Production artifacts verified and loaded from {MODELS_DIR}")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
    yield

app = FastAPI(
    title="Hybrid Sentiment Engine",
    description="A production-aligned pipeline synthesizing text vectors and deterministic emoji signals.",
    version="1.0.0",
    lifespan=lifespan
)

# Templates look in the root /templates folder
templates = Jinja2Templates(directory="templates")
templates.env.filters["uppercase"] = lambda s: str(s).upper()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    """
    Unified Prediction Route:
    Uses the master logic from predict_sentiment to ensure parity with main.py
    """
    # 1. Execute Master Logic
    result = predict_sentiment(text)

    # 2. Sync UI-specific keys
    # Ensure 'prediction_int' exists for your HTML's CSS classes
    result["prediction_int"] = 1 if result["prediction"] == "Positive" else 0
    
    # Ensure 'entropy_flag' exists (Safety check in case config/audit fails)
    if "entropy_flag" not in result:
        from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD
        result["entropy_flag"] = "High Ambiguity" if result["confidence"] < AMBIGUITY_THRESHOLD else "Clear Signal"

    # 3. Final Return (Only one return at the end!)
    return templates.TemplateResponse(
        "components/header/prediction_result.html", 
        {"request": request, "result": result}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Using 0.0.0.0 for Cloud Run compatibility
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)