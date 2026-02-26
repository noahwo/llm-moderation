#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
sbatch scripts/slurm_llamaguard4_server.sbatch