# src/api.py

import os
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from emoji_sentiment_analysis.modeling.predict import predict_sentiment
import uvicorn

app = FastAPI(title="VibeCheck AI")

# --- PATH LOGIC ---
# This ensures it finds the templates folder regardless of where you run the script from
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Custom filter so your 'uppercase' calls in HTML don't crash
templates.env.filters["uppercase"] = lambda s: str(s).upper()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main landing page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    """
    Endpoint for HTMX to call. 
    Processes sentiment and returns the HTML fragment for the results area.
    """
    # 1. Execute the Hybrid ML Logic WITHOUT saving to CSV
    # We set run_audit=False here
    result = predict_sentiment(text, run_audit=False) 
    
    # 2. Return the partial component
    return templates.TemplateResponse(
        "components/prediction_result.html", 
        {"request": request, "result": result}
    )

if __name__ == "__main__":
    # Using reload=True during development is a life-saver
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)