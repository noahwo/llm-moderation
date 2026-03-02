#!/usr/bin/env bash
set -euo pipefail

# --- Customize these for your environment ---
# CONDA_ENV_PATH="/wrk-vakka/users/wuguangh/conda-envs/llm-mod"
PORT="18083"
HOST="0.0.0.0"
NODE_IP="$(hostname -I | awk '{print $1}')"

# Cache locations (recommended on HPC)
export HF_HOME="${HF_HOME:-/wrk-vakka/users/$USER/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Load HF_TOKEN (and other secrets) from .env if present
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."; pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
    echo "Loaded .env from $REPO_ROOT/.env"
fi
# Transformers checks both names; export both to be safe
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# Service knobs
export T5_MODEL_ID="${T5_MODEL_ID:-lmsys/toxicchat-t5-large-v1.0}"
export T5_TORCH_DTYPE="${T5_TORCH_DTYPE:-float32}"
export T5_DEVICE_MAP="${T5_DEVICE_MAP:-auto}"
export T5_MAX_CONCURRENT="${T5_MAX_CONCURRENT:-2}"

mkdir -p logs

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "================"

# Activate conda env
# source "$(dirname "$CONDA_ENV_PATH")/bin/activate" "$CONDA_ENV_PATH"
source ~/.bashrc.bak
conda activate moderation

python -c "import torch; print('torch:', torch.__version__, '  cuda:', torch.version.cuda, '  available:', torch.cuda.is_available())"

python -c "import transformers; print('transformers:', transformers.__version__)"

# If your sbatch is submitted from elsewhere, set repo path:
# cd /path/to/moderation_service

echo "Starting ToxicChat-T5 server on ${NODE_IP}:${PORT}"
echo "TIP: SSH tunnel from your laptop:"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} <login-host>"
echo "Then call http://${NODE_IP}:${PORT}/moderate"

exec python -m uvicorn server.t5_app:app --host "$HOST" --port "$PORT" 2>&1 | tee "logs/t5_${SLURM_JOB_ID:-manual}.log"