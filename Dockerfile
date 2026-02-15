# Use a slim image to keep cold starts fast
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Cloud Run requirements
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 1. Install system dependencies (if any were needed, otherwise skip)
# 2. Copy requirements and pyproject.toml first to leverage Docker cache
COPY requirements.txt pyproject.toml ./

# 3. Install dependencies 
# We also install '.' to treat your src/emoji_sentiment_analysis as a package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

# 4. Copy the rest of the application code
COPY . .

# 5. No more manual PYTHONPATH hacks needed! 
# The 'pip install .' handles the package discovery.

# Execute the application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}