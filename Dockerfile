# Use a slim image to keep cold starts fast
FROM python:3.12-slim

WORKDIR /app

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 1. Copy dependencies first
COPY requirements.txt pyproject.toml ./

# 2. Install external libraries ONLY (this layer stays cached)
RUN pip install --no-cache-dir -r requirements.txt

# 3. NOW copy the rest of the application code (including src/ and README.md)
COPY . .

# 4. Install your own package now that the files exist
RUN pip install --no-cache-dir .

# 5. Execute the application
# Note: Ensure this matches where your 'app' object is defined (e.g., src.main:app)
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}