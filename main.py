# main.py

import joblib
import uvicorn
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

# Modular Imports
from emoji_sentiment_analysis.config import MODELS_DIR
from emoji_sentiment_analysis.services.audit_service import (
    generate_inference_audit,
    lean_log_append,
    get_global_model_signals
)

# ---------------------------------------------------------------------
# Lifespan (startup + shutdown) — canonical FastAPI pattern
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Owns application startup and shutdown.
    All long-lived resources must be created here.
    """

    try:
        app.state.model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
        app.state.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        print("✅ ML Artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ Critical startup error: {e}")
        raise e  # fail fast — correct behavior

    yield  # Application runs while paused here

    # Optional but correct: explicit teardown
    print("🧹 Application shutdown complete.")


# ---------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------

app = FastAPI(
    title="VibeCheck AI",
    description="Production-grade emoji-aware sentiment analysis pipeline.",
    lifespan=lifespan
)

# Templates and Paths
templates = Jinja2Templates(directory="templates")
DATA_DIR = Path("data")
LOG_FILE = DATA_DIR / "logs" / "inference_history.csv"


# ---------------------------------------------------------------------
# UI ROUTES
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    model = request.app.state.model
    vectorizer = request.app.state.vectorizer

    # 1. Generate deep audit logic
    audit = generate_inference_audit(text, model, vectorizer)

    # 2. Extract the model's "internal memory"
    global_signals = get_global_model_signals(model, vectorizer)

    # 3. Save to circular log
    lean_log_append(audit, LOG_FILE)

    # 4. Prepare UI context
    is_pos = audit["prediction_int"] == 1
    ui_context = {
        "request": request,
        "audit": audit,
        "global_signals": global_signals,
        "icon": "✓" if is_pos else "✕",
        "color": (
            "text-emerald-600 border-emerald-200 bg-emerald-50"
            if is_pos
            else "text-rose-600 border-rose-200 bg-rose-50"
        )
    }

    return templates.TemplateResponse(
        "components/result.html",
        ui_context
    )


# ---------------------------------------------------------------------
# DATA ACCESS ROUTES
# ---------------------------------------------------------------------

@app.get("/download/{filename}")
async def download_data(filename: str):
    if "emoticon_data" in filename or "emoji_tweets" in filename:
        file_path = DATA_DIR / "raw" / filename
    elif "combined_data" in filename:
        file_path = DATA_DIR / "processed" / filename
    elif filename == "inference_history.csv":
        file_path = DATA_DIR / "logs" / filename
    else:
        file_path = DATA_DIR / filename

    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv"
        )

    return {"error": f"File {filename} not found."}


@app.get("/data-preview", response_class=HTMLResponse)
async def get_data_preview(request: Request):
    try:
        df = pd.read_csv(
            DATA_DIR / "raw" / "1k_data_emoji_tweets_senti_posneg.csv"
        )
        return templates.TemplateResponse(
            "components/table_rows.html",
            {"request": request, "data": df.head(5)}
        )
    except Exception as e:
        return (
            "<tr><td colspan='3' "
            "class='text-center py-4 text-red-400'>"
            f"Error: {e}</td></tr>"
        )


# ---------------------------------------------------------------------
# LOGS / DASHBOARD
# ---------------------------------------------------------------------

@app.get("/audit-logs", response_class=HTMLResponse)
async def get_audit_logs(request: Request):
    try:
        if LOG_FILE.exists():
            df = pd.read_csv(LOG_FILE)
            latest_logs = df.tail(10).iloc[::-1].to_dict("records")
            return templates.TemplateResponse(
                "components/audit_table.html",
                {"request": request, "logs": latest_logs}
            )

        return "<p class='text-slate-500 text-center py-4'>No logs recorded yet.</p>"

    except Exception as e:
        return f"<tr><td class='text-rose-400'>Log Error: {e}</td></tr>"


# ---------------------------------------------------------------------
# SYSTEM ROUTES
# ---------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "Multinomial-NB",
        "version": "1.3.0 (lifespan)"
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
