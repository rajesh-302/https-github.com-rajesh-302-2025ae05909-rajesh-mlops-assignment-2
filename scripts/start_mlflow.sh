#!/usr/bin/env bash
# Start the MLflow tracking server (SQLite backend + local artifact store) in the
# background. Safe to re-run. UI: http://127.0.0.1:5000
cd "$(dirname "$0")/.." || exit 1
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/health | grep -q 200; then
  echo "MLflow server already running at http://127.0.0.1:5000"; exit 0
fi
mkdir -p mlartifacts
nohup mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000 \
  > /tmp/mlflow_a2.log 2>&1 &
for _ in $(seq 1 30); do
  sleep 2
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/health | grep -q 200; then
    echo "MLflow server running at http://127.0.0.1:5000  (log: /tmp/mlflow_a2.log)"
    exit 0
  fi
done
echo "MLflow server did not start - see: tail -30 /tmp/mlflow_a2.log"; exit 1
