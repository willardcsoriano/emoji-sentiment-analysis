# Use a slim image to keep cold starts fast
FROM python:3.12-slim

# Build-time argument for the port (defaults to 8080)
ARG APP_PORT=8080

# Set the working directory
WORKDIR /app

# Environment variables
ENV PORT=${APP_PORT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 1. Copy dependency files first to leverage Docker layer caching
COPY requirements.txt pyproject.toml ./

# 2. Install dependencies (Cached unless requirements change)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project into the container
# This includes src/, README.md, and any config files
COPY . .

# 4. Install the local project as an editable package
# This resolves all internal pathing issues automatically
RUN pip install --no-cache-dir .

# 5. Execute the application using 'exec' form for better signal handling
# We use the variable ${PORT} so it respects the Cloud Run environment
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]