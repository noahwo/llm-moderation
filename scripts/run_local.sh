#!/usr/bin/env bash
set -euo pipefail

export LG4_MODEL_ID="${LG4_MODEL_ID:-meta-llama/Llama-Guard-4-12B}"
export LG4_TORCH_DTYPE="${LG4_TORCH_DTYPE:-bfloat16}"
export LG4_DEVICE_MAP="${LG4_DEVICE_MAP:-auto}"
export LG4_MAX_CONCURRENT="${LG4_MAX_CONCURRENT:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18082}"

python -m uvicorn server.app:app --host "$HOST" --port "$PORT"