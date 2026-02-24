# app.py - Main FastAPI application for Hybrid Sentiment Engine

import os
import joblib
import uvicorn
import numpy as np
import scipy.sparse as sp
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Modular Imports
from emoji_sentiment_analysis.config import MODELS_DIR
# Corrected import to match your features.py
from emoji_sentiment_analysis.features import extract_emoji_polarity_features

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
            
        app.state.model = joblib.load(model_path)
        app.state.vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Production artifacts loaded successfully from {MODELS_DIR}")
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
    model = request.app.state.model
    vectorizer = request.app.state.vectorizer

    # 1. Standard TF-IDF Transform (gets the 1265 text features)
    text_features = vectorizer.transform([text])

    # 2. Calculate the 4 custom Hybrid Features using YOUR function
    # Returns: (total_pos_signal, total_neg_signal, pos_word, neg_word)
    e_pos, e_neg, w_pos, w_neg = extract_emoji_polarity_features(text)
    
    hybrid_signals = sp.csr_matrix([[e_pos, e_neg, w_pos, w_neg]])

    # 3. Combine them: 1265 + 4 = 1269 total features
    final_features = sp.hstack([text_features, hybrid_signals])

    # 4. Prediction Logic
    prediction_int = int(model.predict(final_features)[0])
    probs = model.predict_proba(final_features)[0]
    confidence = float(max(probs))

    # 5. Logic Breakdown (Top 6 Drivers) - This powers your UI!
    text_feature_names = list(vectorizer.get_feature_names_out())
    hybrid_feature_names = ["emoji_pos_count", "emoji_neg_count", "word_pos_count", "word_neg_count"]
    all_feature_names = text_feature_names + hybrid_feature_names
    
    coefs = model.coef_[0]
    
    # Map nonzero features in this specific text to their weights
    nonzero_indices = final_features.nonzero()[1]
    feature_impacts = []
    for idx in nonzero_indices:
        feature_impacts.append({
            "token": all_feature_names[idx],
            "weight": float(coefs[idx])
        })

    # Sort by absolute impact (strongest signals first)
    top_drivers = sorted(feature_impacts, key=lambda x: abs(x["weight"]), reverse=True)[:6]

    # 6. Construct the Result object the HTML expects
    result = {
        "raw_text": text,
        "prediction": "Positive" if prediction_int == 1 else "Negative",
        "confidence": confidence,
        "prediction_int": prediction_int,
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.20 else "Clear Signal",
        "top_drivers": top_drivers
    }

    return templates.TemplateResponse(
        "components/prediction_result.html", 
        {"request": request, "result": result}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)