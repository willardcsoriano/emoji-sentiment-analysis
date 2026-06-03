# app.py - Main FastAPI application for Hybrid Sentiment Engine

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from emoji_sentiment_analysis.config import AMBIGUITY_THRESHOLD, MODELS_DIR
from emoji_sentiment_analysis.modeling.lite_model import LiteModel
from emoji_sentiment_analysis.modeling.predict import predict_sentiment

# Configure Loguru for the serving process: replace the default handler with one
# whose diagnose=False, so user-submitted text can never be rendered into an
# exception traceback's variable dump (PII leak into logs).
logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)

# ---------------------------------------------------------------------
# Hardening limits
# ---------------------------------------------------------------------
# Cap the analyzed text. The n-gram analyzer in LiteModel grows super-linearly
# with token count, so an unbounded field is a CPU-amplification (DoS) vector.
MAX_TEXT_LEN = 2000
# Reject oversized request bodies before they are buffered/parsed. Comfortably
# fits one MAX_TEXT_LEN field plus multipart overhead.
MAX_BODY_BYTES = 16 * 1024

# Response headers applied to every response. CSP is the durable defense against
# the supply-chain class of bug (e.g. a CDN going rogue): scripts may only load
# from this origin and the explicitly trusted CDNs actually in use.
#
# NOTE: HSTS is intentionally omitted here. This service is reverse-proxied under
# willardcsoriano.dev/projects, so an HSTS header would bind the apex domain's
# TLS policy from a sub-path app — that belongs at the edge/proxy, not here.
CSP = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'self'; "
    "img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' https://cdn.jsdelivr.net data:; "
    "connect-src 'self'; "
    # 'unsafe-inline' covers the inline event handler + config blocks; 'unsafe-eval'
    # is required by the Tailwind Play CDN's in-browser JIT. Both can be dropped
    # once Tailwind is moved to a built, self-hosted stylesheet.
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://unpkg.com https://cdn.jsdelivr.net https://cdn.tailwindcss.com"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


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


@app.middleware("http")
async def harden_response(request: Request, call_next):
    """Reject oversized bodies up front and attach security headers to every
    response."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return PlainTextResponse("Payload too large.", status_code=413)
        except ValueError:
            return PlainTextResponse("Invalid Content-Length.", status_code=400)

    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
templates.env.filters["uppercase"] = lambda s: str(s).upper()


def _error_fragment(request: Request, message: str) -> HTMLResponse:
    """Render the inline error fragment. Returned with 200 so htmx swaps it into
    the result panel (htmx ignores non-2xx responses by default)."""
    return templates.TemplateResponse(
        "components/header/error.html",
        {"request": request, "message": message},
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    if len(text) > MAX_TEXT_LEN:
        return _error_fragment(
            request, f"Input exceeds the {MAX_TEXT_LEN}-character limit. Please shorten it."
        )

    start_time = time.perf_counter()

    try:
        result = predict_sentiment(text, lite=request.app.state.lite)
    except Exception:
        # Sink is configured diagnose=False, so the user's text never appears in
        # the logged traceback (PII).
        logger.exception("Inference failed for /predict-ui")
        return _error_fragment(
            request, "Something went wrong analyzing that input. Please try again."
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
