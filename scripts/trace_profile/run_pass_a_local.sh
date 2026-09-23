#!/usr/bin/env bash
# Pass A local first test: zero-latency mock + OTEL/WEKA stall fixture.
# Usage (from repo root):
#   bash scripts/trace_profile/run_pass_a_local.sh
#   bash scripts/trace_profile/run_pass_a_local.sh weka
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/guidellm" ]]; then
  PY="$ROOT/.venv/bin/python"
  GL="$ROOT/.venv/bin/guidellm"
else
  echo "Expected .venv at $ROOT/.venv — set it up and retry." >&2
  exit 1
fi

FORMAT="${1:-otel}"
OUT_DIR="$ROOT/scripts/trace_profile/out"
mkdir -p "$OUT_DIR"

case "$FORMAT" in
  otel)
    TRACE="$ROOT/otel_stall.jsonl"
    DATA_KIND="otel"
    ;;
  weka)
    TRACE="$ROOT/weka_stall.jsonl"
    DATA_KIND="weka"
    ;;
  *)
    echo "Usage: $0 [otel|weka]" >&2
    exit 2
    ;;
esac

if [[ ! -f "$TRACE" ]]; then
  echo "Missing trace file: $TRACE" >&2
  exit 1
fi

PORT="${GUIDELLM_MOCK_PORT:-8000}"
REPORT="$OUT_DIR/${FORMAT}_stall_mock.json"
COMPARE="$OUT_DIR/${FORMAT}_stall_mock_compare.csv"

echo "Starting mock-server on 127.0.0.1:${PORT} (ttft=0 itl=0 latency=0)"
"$GL" mock-server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ttft-ms 0 \
  --itl-ms 0 \
  --output-tokens 1 \
  --request-latency 0 >/dev/null 2>"$OUT_DIR/mock-server.log" &
MOCK_PID=$!
trap 'kill "$MOCK_PID" 2>/dev/null || true' EXIT

echo "Waiting for mock-server..."
ready=0
for _ in $(seq 1 60); do
  if "$PY" - <<PY
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/v1/models", timeout=1)
PY
  then
    ready=1
    break
  fi
  sleep 0.25
done
if [[ "$ready" -ne 1 ]]; then
  echo "mock-server did not become ready; log:" >&2
  cat "$OUT_DIR/mock-server.log" >&2 || true
  exit 1
fi

echo "Running replay against $TRACE"
"$GL" run \
  --backend "kind=openai_http,target=http://127.0.0.1:${PORT}" \
  --profile kind=replay,time_scale=1.0 \
  --data "kind=${DATA_KIND},source.kind=json_file,source.path=${TRACE},time_scale=1.0" \
  --disable-progress \
  --output "kind=json,path=${REPORT}"

echo "Comparing timings"
"$PY" "$ROOT/scripts/trace_profile/compare_replay_timings.py" \
  --benchmarks "$REPORT" \
  --trace "$TRACE" \
  --format "$FORMAT" \
  --backend mock \
  --output "$COMPARE"
