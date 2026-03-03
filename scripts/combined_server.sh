#!/usr/bin/env bash
set -euo pipefail

PORT="18084"
HOST="0.0.0.0"
NODE_IP="$(hostname -I | awk '{print $1}')"

# Cache locations
export HF_HOME="${HF_HOME:-/wrk-vakka/users/$USER/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Load secrets from .env
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." ; pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
    echo "Loaded .env from $REPO_ROOT/.env"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# --- LlamaGuard-4 knobs ---
export LG4_TORCH_DTYPE="${LG4_TORCH_DTYPE:-bfloat16}"
export LG4_DEVICE_MAP="${LG4_DEVICE_MAP:-auto}"
export LG4_MAX_CONCURRENT="${LG4_MAX_CONCURRENT:-1}"

# --- ToxicChat-T5 knobs ---
export T5_MODEL_ID="${T5_MODEL_ID:-lmsys/toxicchat-t5-large-v1.0}"
export T5_TORCH_DTYPE="${T5_TORCH_DTYPE:-float32}"
export T5_DEVICE_MAP="${T5_DEVICE_MAP:-auto}"
export T5_MAX_CONCURRENT="${T5_MAX_CONCURRENT:-2}"

mkdir -p logs

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HF_HOME=$HF_HOME"
echo "================"

source ~/.bashrc.bak
conda activate moderation

python -c "import torch; print('torch:', torch.__version__, '  cuda:', torch.version.cuda, '  available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"

echo "Starting combined moderation server on ${NODE_IP}:${PORT}"
echo "  GET  http://${NODE_IP}:${PORT}/healthz"
echo "  POST http://${NODE_IP}:${PORT}/lg4/moderate"
echo "  POST http://${NODE_IP}:${PORT}/t5/moderate"
echo "TIP: SSH tunnel:"
echo "  ssh -N -L ${PORT}:${NODE_IP}:${PORT} <login-host>"

exec python -m uvicorn server.combined_app:app --host "$HOST" --port "$PORT" 2>&1 | tee >(grep -v "^Loading weights:" > "logs/combined_${SLURM_JOB_ID:-manual}.log")
