#!/bin/bash
# =============================================================================
# SLURM Batch Script for vLLM Inference
# =============================================================================
# This script runs vLLM inference jobs on a SLURM cluster.
#
# Usage:
#   sbatch --export=ALL,FILE_NAME="...",FOLDER_PATH="...",MODEL_NAME="...",ENABLE_THINKING="true" \
#          vllm_inference.sbatch
#
# Required Environment Variables (passed via --export):
#   FILE_NAME       - Input JSON file name (without .json extension)
#   FOLDER_PATH     - Path to data folder (e.g., "legal/multi_lexsum/summarization")
#   MODEL_NAME      - HuggingFace model identifier (e.g., "Qwen/Qwen3-14B")
#   ENABLE_THINKING - "true" or "false" to enable thinking/reasoning mode
#
# Optional Environment Variables:
#   GPU_TYPE        - GPU type to request (default: a40)
#   GPU_COUNT       - Number of GPUs (default: 4)
#   SLURM_PARTITION - SLURM partition (default: gpu)
#   SLURM_QOS       - SLURM QOS (default: normal)
#   ENV_FILE        - Path to environment file (optional)
#
# Example:
#   sbatch --export=ALL,FILE_NAME="test_data",\
#          FOLDER_PATH="legal/multi_lexsum/summarization",\
#          ENABLE_THINKING="true",\
#          MODEL_NAME="Qwen/Qwen3-14B" \
#          vllm_inference.sbatch
# =============================================================================

#SBATCH --job-name=vllm_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a40:4
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --output=vllm_inference_logs/%x-%j.out

echo "=============================================="
echo "vLLM Inference Job"
echo "=============================================="
echo "START TIME: $(date)"
echo "HOSTNAME:   $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "=============================================="

set -euo pipefail
set -x

# -----------------------------------------------------------------------------
# Environment setup (optional)
# -----------------------------------------------------------------------------
if [[ -n "${ENV_FILE:-}" && -f "$ENV_FILE" ]]; then
    echo "Loading environment from: $ENV_FILE"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# -----------------------------------------------------------------------------
# Inference parameters (from environment or defaults)
# -----------------------------------------------------------------------------
file_name="${FILE_NAME:-test_data}"
folder_path="${FOLDER_PATH:-legal/multi_lexsum/summarization}"
enable_thinking="${ENABLE_THINKING:-false}"
model_name="${MODEL_NAME:-Qwen/Qwen3-14B}"

echo "Configuration:"
echo "  FILE_NAME:       $file_name"
echo "  FOLDER_PATH:     $folder_path"
echo "  MODEL_NAME:      $model_name"
echo "  ENABLE_THINKING: $enable_thinking"

# -----------------------------------------------------------------------------
# Build log file path
# -----------------------------------------------------------------------------
model_save_name=$(basename "$model_name")
log_path="vllm_inference_logs/${folder_path}/${model_save_name}/${file_name}_thinking_${enable_thinking}.txt"
mkdir -p "$(dirname "$log_path")"

echo "Log path: $log_path"

# -----------------------------------------------------------------------------
# Launch vLLM inference
# -----------------------------------------------------------------------------
srun python vllm_inference.py \
    --file_name "$file_name" \
    --folder_path "$folder_path" \
    --model_name "$model_name" \
    $( [[ "$enable_thinking" == "true" ]] && echo "--enable_thinking" ) \
    2>&1 | tee "$log_path"

echo "=============================================="
echo "END TIME: $(date)"
echo "=============================================="
