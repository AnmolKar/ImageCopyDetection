#!/bin/bash
#SBATCH --job-name=dinov3_ced_gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --partition=week-long-gpu
#SBATCH --nodelist=gpu4
#SBATCH --gres=gpu:A30:2
#SBATCH --output="/home/jowatson/Deep Learning/slurm_logs/dinov3_ced_gpu4_%j.out"
#SBATCH --error="/home/jowatson/Deep Learning/slurm_logs/dinov3_ced_gpu4_%j.err"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowatson@clarku.edu

set -euo pipefail

LOG_DIR="/home/jowatson/Deep Learning/slurm_logs"
PROJECT_ROOT="/home/jowatson/Deep Learning/Code"
SCRIPT_PATH="${PROJECT_ROOT}/DINOv3_CED_DDP.py"

DEFAULT_NPROC=2                                                  # matches --gres request
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
    GPU_SOURCE="SLURM_GPUS_ON_NODE"
else
    NPROC_PER_NODE=$DEFAULT_NPROC
    GPU_SOURCE="fallback to SBATCH --gres"
fi

if [ "$NPROC_PER_NODE" -lt 1 ]; then
    echo "[FATAL] NPROC_PER_NODE resolved to $NPROC_PER_NODE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "GPUs requested (per node): $NPROC_PER_NODE [$GPU_SOURCE]"
echo "Torch processes per node: $NPROC_PER_NODE"

# ----------------- Performance tuning for gpu4 (2 GPUs, 257GB RAM) -----------------
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
THREADS_PER_PROC=$(( CPUS_PER_TASK / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then
    THREADS_PER_PROC=1
fi

export OMP_NUM_THREADS=$THREADS_PER_PROC
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS (CPUS_PER_TASK=$CPUS_PER_TASK, NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "CUDA optimizations enabled for 2x A30 GPUs"

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

echo "Launching torchrun at $(date)"
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  "$SCRIPT_PATH"

echo "Job finished at $(date)"
