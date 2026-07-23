#!/bin/bash
#
# Generic SLURM launcher for one chunk-by-chunk inference run.
# Adjust the resource headers below for your cluster (GPU type/count, CPUs;
# add --partition/--qos/--account lines if your site requires them).
#
#SBATCH --job-name=chunk_vllm_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # 1 task per node (vLLM handles GPU parallelism)
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=vllm_inference_logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue

# Example launch:
# sbatch --export=ALL,DOMAIN="legal",FILE_NAME="2025_example_cases",\
#        CHECKLIST_ITEM="Cause_of_Action",\
#        ENABLE_THINKING="true",\
#        MODEL_NAME="Qwen/Qwen3-14B" \
#        vllm_inference.sbatch

echo "START TIME: $(date)"
echo "RUNNING ON: $(hostname)   |  SLURM_NODELIST=${SLURM_JOB_NODELIST}"

set -euo pipefail

# ----------------------------------------------------------------------
# Optional environment setup BEFORE enabling xtrace, so secrets in the env
# file are never echoed into the log files. Point COMMON_ENV at your own env
# file (CUDA paths, HF cache, etc.) if you need one; skipped otherwise.
# ----------------------------------------------------------------------
COMMON_ENV=${COMMON_ENV:-}
if [[ -n "$COMMON_ENV" && -f "$COMMON_ENV" ]]; then
    set -o allexport
    source "$COMMON_ENV"
    set +o allexport
fi

set -x

# ----------------------------------------------------------------------
# Inference parameters (from environment or sensible defaults)
# ----------------------------------------------------------------------
domain=${DOMAIN:-"legal"}                          # "legal" or "medical"
file_name=${FILE_NAME:-"2025_example_cases"}       # without the domain prefix
checklist_item=${CHECKLIST_ITEM:-""}               # empty means process all items
enable_thinking=${ENABLE_THINKING:-"true"}         # "true" or "false"
model_name=${MODEL_NAME:-"Qwen/Qwen3-14B"}

# ----------------------------------------------------------------------
# Build the log-file path for the *python* run (separate from Slurm log)
# ----------------------------------------------------------------------
model_save_name=$(basename "$model_name")

# File naming (matches the python script): legal keeps the historical
# unprefixed names; other domains get a {domain}_ prefix (tolerating an
# already-prefixed FILE_NAME).
if [[ "$domain" == "legal" || "$file_name" == ${domain}_* ]]; then
    prefixed_name="$file_name"
else
    prefixed_name="${domain}_${file_name}"
fi

# Create item suffix for log file name
if [[ -n "$checklist_item" ]]; then
    item_suffix="_${checklist_item}"
else
    item_suffix="_all_items"
fi

log_path="vllm_inference_logs/${model_save_name}/${prefixed_name}${item_suffix}_thinking_${enable_thinking}.txt"
mkdir -p "$(dirname "$log_path")"

# ----------------------------------------------------------------------
# Launch!  vLLM will auto-detect the GPUs and shard across them.
# ----------------------------------------------------------------------
# Build optional args as an array so values are passed without mangling
extra_args=()
if [[ "$enable_thinking" == "true" ]]; then
    extra_args+=("--enable_thinking")
fi
if [[ -n "$checklist_item" ]]; then
    extra_args+=("--checklist_item" "$checklist_item")
fi

# Append to the python log (a requeued/resumed run must not wipe earlier output)
echo "==== New run at $(date) | SLURM job ${SLURM_JOB_ID:-unknown} ====" >> "$log_path"

srun python -u vllm_inference.py \
      --domain "$domain" \
      --file_name "$file_name" \
      --model_name "$model_name" \
      ${extra_args[@]+"${extra_args[@]}"} \
      2>&1 | tee -a "$log_path"

echo "END TIME: $(date)"
