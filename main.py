import joblib
import uvicorn
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

# Internal imports
from emoji_sentiment_analysis.features import extract_emojis
from emoji_sentiment_analysis.config import MODELS_DIR

# 1. Initialize FastAPI
app = FastAPI(
    title="VibeCheck AI",
    description="Production-grade emoji-aware sentiment analysis pipeline with Deep Audit Logging."
)

templates = Jinja2Templates(directory="templates")
DATA_DIR = Path("data") 
LOG_FILE = DATA_DIR / "logs" / "inference_history.csv"
MAX_LOG_ENTRIES = 500

# Global Model State
model = None
vectorizer = None

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer
    try:
        model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
        vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        print("✅ ML Artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")

# --- LEAN LOGGING UTILITY ---

def lean_log_append(log_data):
    """Appends to a single CSV, keeping only the latest N records to stay lean."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten the 'top_drivers' list for CSV storage
    csv_ready_data = log_data.copy()
    csv_ready_data['top_drivers'] = "|".join([f"{d['token']}({d['weight']})" for d in log_data['top_drivers']])
    
    new_entry = pd.DataFrame([csv_ready_data])
    
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, new_entry], ignore_index=True).tail(MAX_LOG_ENTRIES)
    else:
        df = new_entry
        
    df.to_csv(LOG_FILE, index=False)

# --- UI ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    # 1. Pipeline Execution & Decision Audit Extraction
    emojis = extract_emojis(text)
    engine_input = f"{text} {emojis * 5}"
    vec = vectorizer.transform([engine_input])
    
    # Math: Probabilities and Prediction
    probs = model.predict_proba(vec)[0]
    prediction = int(model.predict(vec)[0])
    
    # Decision Drivers (Explainability)
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = vec.nonzero()[1]
    feature_impacts = []
    for idx in nonzero_indices:
        token = feature_names[idx]
        weight = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
        feature_impacts.append({"token": token, "weight": round(weight, 4)})
    
    top_drivers = sorted(feature_impacts, key=lambda x: abs(x['weight']), reverse=True)[:3]

    # 2. Build the Log Data
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "confidence": round(float(np.max(probs)), 4),
        "entropy_flag": "High Ambiguity" if abs(probs[0] - probs[1]) < 0.15 else "Clear Signal",
        "top_drivers": top_drivers
    }

    # 3. Trigger Lean Circular Logging
    lean_log_append(log_entry)

    # 4. Prepare UI components
    sentiment = log_entry["prediction"]
    color = "text-emerald-600 border-emerald-200 bg-emerald-50" if prediction == 1 else "text-rose-600 border-rose-200 bg-rose-50"
    icon = "✓" if prediction == 1 else "✕"
    
    # Format drivers for display
    drivers_html = "".join([f"<span class='bg-white/50 px-2 py-0.5 rounded border border-current/10 mx-0.5'>{d['token']}</span>" for d in top_drivers])

    return f"""
    <div class="{color} w-full p-8 rounded-3xl border-2 animate-in fade-in zoom-in duration-500">
        <div class="flex flex-col items-center gap-3">
            <span class="text-[10px] font-black uppercase tracking-[0.3em] opacity-50">{log_entry['entropy_flag']}</span>
            <span class="text-5xl font-black tracking-tighter">{icon} {sentiment}</span>
            <p class="text-xs font-bold opacity-70">Confidence: {log_entry['confidence']*100:.1f}%</p>
            
            <div class="mt-4 px-4 py-2 bg-white/30 rounded-full border border-current/5 text-[9px] font-mono italic opacity-80 text-center">
                Primary Drivers: {drivers_html if drivers_html else 'None (Neutral)'}
            </div>
        </div>
    </div>
    """

# --- DATA ACCESS ROUTES ---

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
        return FileResponse(path=file_path, filename=filename, media_type='text/csv')
    return {"error": f"File {filename} not found."}

@app.get("/data-preview", response_class=HTMLResponse)
async def get_data_preview():
    try:
        df = pd.read_csv(DATA_DIR / "raw" / "1k_data_emoji_tweets_senti_posneg.csv")
        top_5 = df.head(5)
        rows_html = ""
        for _, row in top_5.iterrows():
            text = row.get('post', 'N/A')
            label = row.get('sentiment', '0')
            color = "text-green-600" if str(label) == "1" else "text-red-600"
            rows_html += f"<tr><td class='px-4 py-3 truncate max-w-[300px] text-slate-600 italic'>\"{text}\"</td><td class='px-4 py-3 text-right {color} font-bold'>{label}</td></tr>"
        return rows_html
    except Exception as e:
        return f"<tr><td colspan='2' class='text-center py-4 text-red-400'>Error loading data: {e}</td></tr>"

# --- SYSTEM ROUTES ---

@app.get("/health")
def health_check():
    return {"status": "online", "engine": "Multinomial-NB", "version": "1.1.0 (Audit Enabled)"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)