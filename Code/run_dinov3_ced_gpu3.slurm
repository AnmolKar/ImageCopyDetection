#!/bin/bash
#SBATCH --job-name=dinov3_ced_gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --partition=week-long-gpu
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:A30:3
#SBATCH --output=/home/jowatson/slurm_logs/dinov3_ced_gpu3_%j.out
#SBATCH --error=/home/jowatson/slurm_logs/dinov3_ced_gpu3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowatson@clarku.edu

set -euo pipefail

LOG_DIR="/home/jowatson/slurm_logs"
PROJECT_ROOT="/home/jowatson/Deep Learning/Code"
SCRIPT_PATH="${PROJECT_ROOT}/DINOv3_CED_SLURM.py"

mkdir -p "$LOG_DIR"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "GPUs requested: ${SLURM_GPUS_ON_NODE:-1}"
echo "Starting run at $(date)"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gpu_training
else
    echo "[FATAL] Conda not found in PATH" >&2
    exit 1
fi

nvidia-smi

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1

echo "Launching training script at $(date)"
python3 -u "$SCRIPT_PATH"

echo "Job finished at $(date)"
