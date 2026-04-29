FROM python:3.12-slim

WORKDIR /app

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY requirements-prod.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

COPY . .

RUN pip install --no-cache-dir .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level warning"]