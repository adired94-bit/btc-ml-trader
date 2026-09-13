#!/usr/bin/env sh
set -e
PORT="${PORT:-${DASHBOARD_PORT:-8501}}"
if [ ! -f models/direction_ensemble.joblib ]; then
  echo "[entrypoint] no model artifacts found - training"
  python -m src.models.train
fi
python -m uvicorn src.api.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" &
sleep 3
exec python -m streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
