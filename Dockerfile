# Use a slim image to keep cold starts fast
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# 1. Copy dependency files first
COPY requirements.txt pyproject.toml ./

# 2. Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project
COPY . .

# 4. Install the local project
# Installing the current directory ensures all imports in src/ work
RUN pip install --no-cache-dir .

# 5. Execute the application
# We point to app:app because app.py is in your root directory
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]