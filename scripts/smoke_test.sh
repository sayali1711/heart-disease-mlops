#!/usr/bin/env bash
# Build + run the container and hit it with a sample request. Fails on any error.
set -euo pipefail
IMAGE="${IMAGE:-heart-disease-api:latest}"
NAME="heart-api-smoke"
PORT="${PORT:-8000}"

docker build -t "$IMAGE" .
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE"
trap 'docker logs "$NAME" || true; docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then break; fi
  sleep 1
  [ "$i" -eq 30 ] && { echo "API did not become healthy"; exit 1; }
done

echo "--- /health";  curl -sf "http://localhost:$PORT/health"; echo
echo "--- /predict"
curl -sf -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" -d @sample_input.json; echo
echo "--- /metrics (head)"; curl -sf "http://localhost:$PORT/metrics" | head -n 8
echo "SMOKE TEST PASSED"
