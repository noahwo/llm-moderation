#!/usr/bin/env bash
#SBATCH --job-name=lg4-serve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lg4-%j.out
#SBATCH --error=logs/lg4-%j.err

set -euo pipefail

# --- Customize these for your environment ---
CONDA_ENV_PATH="/wrk-vakka/users/wuguangh/conda-envs/llm-mod"
PORT="18082"
HOST="0.0.0.0"

# Optional: put HF caches somewhere fast/persistent for the cluster
export HF_HOME="${HF_HOME:-/wrk-vakka/users/$USER/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Optional knobs for the service
export LG4_MODEL_ID="meta-llama/Llama-Guard-4-12B"
export LG4_TORCH_DTYPE="bfloat16"
export LG4_DEVICE_MAP="auto"
export LG4_MAX_CONCURRENT="1"

mkdir -p logs

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "================"

# Activate conda env
source "$(dirname "$CONDA_ENV_PATH")/bin/activate" "$CONDA_ENV_PATH"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

# Run from your repo root (adjust if needed)
# Expecting:
#   server/app.py
#   server/model_runner.py
#
# If your sbatch is run from elsewhere, uncomment and set:
# cd /path/to/moderation_service

echo "Starting LG4 server on ${HOST}:${PORT}"
echo "TIP: If calling from login node, you can SSH tunnel:"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} <login-host>"
echo "Then call http://127.0.0.1:${PORT}/moderate"

# Start server
python -m uvicorn server.app:app --host "$HOST" --port "$PORT"