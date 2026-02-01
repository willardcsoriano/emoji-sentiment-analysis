import joblib
import uvicorn
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Internal imports
from emoji_sentiment_analysis.features import extract_emojis
from emoji_sentiment_analysis.config import MODELS_DIR

# 1. Initialize FastAPI
app = FastAPI(
    title="VibeCheck AI",
    description="Production-grade emoji-aware sentiment analysis pipeline."
)

templates = Jinja2Templates(directory="templates")
DATA_DIR = Path("data") 

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

class SentimentRequest(BaseModel):
    text: str

# --- UI ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    emojis = extract_emojis(text)
    processed_text = f"{text} {emojis * 5}"
    vec = vectorizer.transform([processed_text])
    prediction = model.predict(vec)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    color = "text-emerald-600 border-emerald-200 bg-emerald-50" if prediction == 1 else "text-rose-600 border-rose-200 bg-rose-50"
    icon = "✓" if prediction == 1 else "✕"

    return f"""
    <div class="{color} w-full p-8 rounded-3xl border-2 animate-in fade-in zoom-in duration-500">
        <div class="flex flex-col items-center gap-3">
            <span class="text-[10px] font-black uppercase tracking-[0.3em] opacity-50">Inference Complete</span>
            <span class="text-5xl font-black tracking-tighter">{icon} {sentiment}</span>
            <div class="mt-4 px-4 py-2 bg-white/50 rounded-full border border-current/10 text-[10px] font-mono italic opacity-80">
                Engine Input: <span class="font-bold">{processed_text}</span>
            </div>
        </div>
    </div>
    """

# --- DATA ACCESS ROUTES ---

@app.get("/download/{filename}")
async def download_data(filename: str):
    """Serves the raw CSV files for download based on their subfolder location."""
    if "emoticon_data" in filename or "emoji_tweets" in filename:
        file_path = DATA_DIR / "raw" / filename
    elif "combined_data" in filename:
        file_path = DATA_DIR / "processed" / filename
    else:
        file_path = DATA_DIR / filename

    if file_path.exists():
        return FileResponse(path=file_path, filename=filename, media_type='text/csv')
    
    return {"error": f"File {filename} not found."}

@app.get("/data-preview", response_class=HTMLResponse)
async def get_data_preview():
    """Returns real HTML rows from the CSV for the HTMX table preview."""
    try:
        df = pd.read_csv(DATA_DIR / "raw" / "1k_data_emoji_tweets_senti_posneg.csv")
        top_5 = df.head(5)
        
        rows_html = ""
        for _, row in top_5.iterrows():
            text = row.get('text', 'N/A')
            label = row.get('sentiment', '0')
            color = "text-green-600" if str(label) == "1" else "text-red-600"
            rows_html += f"""
            <tr>
                <td class="px-4 py-3 truncate max-w-[300px] text-slate-600 italic">"{text}"</td>
                <td class="px-4 py-3 text-right {color} font-bold">{label}</td>
            </tr>
            """
        return rows_html
    except Exception as e:
        return f"<tr><td colspan='2' class='text-center py-4 text-red-400'>Error loading data: {e}</td></tr>"

# --- SYSTEM ROUTES ---

@app.get("/health")
def health_check():
    return {"status": "online", "engine": "Multinomial-NB", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)