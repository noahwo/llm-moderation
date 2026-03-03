#!/usr/bin/env bash
# serve.sh — launch the LLM moderation service (LlamaGuard-4 + ToxicChat-T5)
# Usage: ./scripts/serve.sh
# All variables below can be overridden from the outside environment.
set -euo pipefail

# --- HPC / HuggingFace cache paths ---
export HF_HOME="${HF_HOME:-/wrk-vakka/users/$USER/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# --- Load secrets from .env (HF_TOKEN, etc.) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." ; pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# --- LlamaGuard-4 ---
export LG4_TORCH_DTYPE="${LG4_TORCH_DTYPE:-bfloat16}"
export LG4_DEVICE_MAP="${LG4_DEVICE_MAP:-auto}"
export LG4_MAX_CONCURRENT="${LG4_MAX_CONCURRENT:-1}"

# --- ToxicChat-T5 ---
export T5_MODEL_ID="${T5_MODEL_ID:-lmsys/toxicchat-t5-large-v1.0}"
export T5_TORCH_DTYPE="${T5_TORCH_DTYPE:-float32}"
export T5_DEVICE_MAP="${T5_DEVICE_MAP:-0}"
export T5_MAX_CONCURRENT="${T5_MAX_CONCURRENT:-2}"

# --- Server ---
export PORT="${PORT:-18084}"
export HOST="${HOST:-0.0.0.0}"

mkdir -p "$REPO_ROOT/logs"

# --- Activate environment ---
# Temporarily disable nounset (-u) so that .bashrc.bak can reference variables
# like LD_LIBRARY_PATH that may not be set in the current environment.
set +u
source ~/.bashrc.bak
set -u
conda activate moderation

# --- Startup diagnostics ---
export NODE_IP="$(hostname -I | awk '{print $1}')"
echo "=== env ==="
echo "Node : $(hostname)"
echo "GPU  : ${CUDA_VISIBLE_DEVICES:-all}"
echo "HF   : $HF_HOME"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
echo "==========="
echo "Service  : http://${NODE_IP}:${PORT}"
echo "API docs : http://${NODE_IP}:${PORT}/docs"
echo "SSH tunnel: ssh -N -L ${PORT}:${NODE_IP}:${PORT} <login-host>"

# --- Run (filter weight-loading progress lines from the log file only) ---
cd "$REPO_ROOT"
exec python -m server 2>&1 \
    | tee >(grep -v "^Loading weights" > "logs/service_${SLURM_JOB_ID:-manual}.log")
