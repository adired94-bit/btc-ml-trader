# Single-container deployment: FastAPI on :8000 (internal) + Streamlit on $PORT.
# Works on Render, Railway, Fly.io, Hugging Face Spaces (Docker) or any VPS.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8501
ENV API_HOST=127.0.0.1 API_PORT=8000 API_URL=http://127.0.0.1:8000 DASHBOARD_PORT=8501
CMD ["./docker-entrypoint.sh"]
