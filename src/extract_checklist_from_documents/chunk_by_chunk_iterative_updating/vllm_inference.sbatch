#!/bin/bash
#
#SBATCH --job-name=chunk_vllm_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # 1 task per node (vLLM handles GPU parallelism)
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a40:4
#SBATCH --qos=short
#SBATCH --partition=overcap
#SBATCH --output=vllm_inference_logs/%x-%j.out
#SBATCH --exclude=gundam,dendrite,puma,xaea-12,spot,hk47,consu,nestor,dave

# Example launch:
# sbatch --export=ALL,FILE_NAME="2025_example_cases",\
#        CHECKLIST_ITEM="Cause_of_Action",\
#        ENABLE_THINKING="true",\
#        MODEL_NAME="Qwen/Qwen3-14B" \
#        vllm_inference.sbatch

echo "START TIME: $(date)"
echo "RUNNING ON: $(hostname)   |  SLURM_NODELIST=${SLURM_JOB_NODELIST}"

set -euo pipefail
set -x

# ----------------------------------------------------------------------
# Environment setup (if you keep a shared `.env` for CUDA paths, etc.)
# ----------------------------------------------------------------------
set -o allexport
source /srv/nlprx-lab/share6/douy/common.env
set +o allexport

# ----------------------------------------------------------------------
# Inference parameters (from environment or sensible defaults)
# ----------------------------------------------------------------------
file_name=${FILE_NAME:-"2025_example_cases"}
checklist_item=${CHECKLIST_ITEM:-""}              # empty means process all items
enable_thinking=${ENABLE_THINKING:-"false"}        # "true" or "false"
model_name=${MODEL_NAME:-"Qwen/Qwen3-14B"}

# ----------------------------------------------------------------------
# Build the log-file path for the *python* run (separate from Slurm log)
# ----------------------------------------------------------------------
model_save_name=$(basename "$model_name")

# Create item suffix for log file name
if [[ -n "$checklist_item" ]]; then
    item_suffix="_${checklist_item}"
else
    item_suffix="_all_items"
fi

log_path="vllm_inference_logs/${model_save_name}/${file_name}${item_suffix}_thinking_${enable_thinking}.txt"
mkdir -p "$(dirname "$log_path")"

# ----------------------------------------------------------------------
# Launch!  vLLM will auto-detect the 8 GPUs and shard across them.
# ----------------------------------------------------------------------
srun python vllm_inference.py \
      --file_name "$file_name" \
      --model_name "$model_name" \
      $( [[ "$enable_thinking" == "true" ]] && echo "--enable_thinking" ) \
      $( [[ -n "$checklist_item" ]] && echo "--checklist_item \"$checklist_item\"" ) \
      2>&1 | tee "$log_path"

echo "END TIME: $(date)"