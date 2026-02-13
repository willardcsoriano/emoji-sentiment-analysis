FROM python:3.12-slim
WORKDIR /app
# Cloud Run requirement
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy everything (app.py, templates, src, models)
COPY . .
# Explicitly add /app/src to PYTHONPATH so app.py can see your modules
ENV PYTHONPATH=/app/src
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
