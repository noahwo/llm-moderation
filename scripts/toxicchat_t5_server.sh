#!/usr/bin/env bash
#SBATCH --job-name=t5-serve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/t5-%j.out
#SBATCH --error=logs/t5-%j.err

set -euo pipefail

# --- Customize these for your environment ---
CONDA_ENV_PATH="/wrk-vakka/users/wuguangh/conda-envs/llm-mod"
PORT="18083"
HOST="0.0.0.0"

# Cache locations (recommended on HPC)
export HF_HOME="${HF_HOME:-/wrk-vakka/users/$USER/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Service knobs
# Change this to your exact ToxicChat-T5 model id if different
export T5_MODEL_ID="${T5_MODEL_ID:-lmsys/toxic-chat-t5-large}"
export T5_TORCH_DTYPE="${T5_TORCH_DTYPE:-float16}"
export T5_DEVICE_MAP="${T5_DEVICE_MAP:-auto}"
export T5_MAX_CONCURRENT="${T5_MAX_CONCURRENT:-2}"

mkdir -p logs

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "T5_MODEL_ID=$T5_MODEL_ID"
echo "================"

# Activate conda env
source "$(dirname "$CONDA_ENV_PATH")/bin/activate" "$CONDA_ENV_PATH"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

# If your sbatch is submitted from elsewhere, set repo path:
# cd /path/to/moderation_service

echo "Starting ToxicChat-T5 server on ${HOST}:${PORT}"
echo "TIP: SSH tunnel from your laptop:"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} <login-host>"
echo "Then call http://127.0.0.1:${PORT}/moderate"

# IMPORTANT: Update the import path below to your actual T5 FastAPI app.
# Example options (pick one):
#   python -m uvicorn server.t5_app:app --host "$HOST" --port "$PORT"
#   python -m uvicorn server_t5.app:app --host "$HOST" --port "$PORT"
python -m uvicorn server.t5_app:app --host "$HOST" --port "$PORT"