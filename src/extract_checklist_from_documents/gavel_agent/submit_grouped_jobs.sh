#!/usr/bin/env bash
set -euo pipefail

# Script to submit legal agent jobs for all grouped checklist configs
# Usage: ./submit_grouped_jobs.sh

# ----------------------------------------------------------------------
# Define the case IDs to process
# ----------------------------------------------------------------------
declare -a CASE_IDS=(
  # Previous case IDs (commented out, except 46210 and 46094 which are in the new list)
  # "45696"
  # "46234"
  # "46349"
  
  # 20 human evaluation cases
  # Common 5
  "46210"
  "46094"
  "43840"
  "43417"
  "15426"
  # Rest 15
  "46083"
  "43972"
  "17762"
  "46239"
  "45157"
  "45858"
  "12429"
  "17701"
  "45737"
  "46114"
  "43966"
  "46226"
  "46071"
  "17268"
  "46310"

  # # common 5 for 20_human_eval_cases_2
  # "46507"
  # "46329"
  # "46758"
  # "46666"
  # "46746"

  # # rest 15 for 20_human_eval_cases_2
  # "46678"
  # "46390"
  # "46348"
  # "46340"
  # "46482"

  # "46755"
  # "46341"
  # "46342"
  # "46651"
  # "46351"

  # "46620"
  # "46625"
  # "46499"
  # "46602"
  # "46805"
)

# ----------------------------------------------------------------------
# Define the checklist configs - ALL 9 GROUPED CONFIGS
# ----------------------------------------------------------------------
declare -a CHECKLIST_CONFIGS=(
  "config/checklist_configs/grouped/01_basic_case_info.yaml"
  "config/checklist_configs/grouped/02_legal_foundation.yaml"
  "config/checklist_configs/grouped/03_judge_info.yaml"
  "config/checklist_configs/grouped/04_related_cases.yaml"
  "config/checklist_configs/grouped/05_filings_proceedings.yaml"
  "config/checklist_configs/grouped/06_decrees.yaml"
  "config/checklist_configs/grouped/07_settlements.yaml"
  "config/checklist_configs/grouped/08_monitoring.yaml"
  "config/checklist_configs/grouped/09_context.yaml"
)

# ----------------------------------------------------------------------
# Define the models to test
# ----------------------------------------------------------------------
declare -a MODELS=(
  # "Qwen/Qwen3-8B"
  # "Qwen/Qwen3-14B"
  # "Qwen/Qwen3-32B"
  # "Qwen/Qwen3-30B-A3B-Thinking-2507"
  # "unsloth/gpt-oss-120b-BF16"
  "unsloth/gpt-oss-20b-BF16"
  # Add more models as needed
)

# ----------------------------------------------------------------------
# Define the max steps configurations to test
# ----------------------------------------------------------------------
declare -a MAX_STEPS=(
  "200"
)

# ----------------------------------------------------------------------
# Default settings
# ----------------------------------------------------------------------
DEFAULT_RESUME="false"     # Set to "true" to resume from existing state
DEFAULT_DEBUG="true"       # Set to "true" for debug mode with full prompts

# ----------------------------------------------------------------------
# Output and data directory settings
# ----------------------------------------------------------------------
OUTPUT_BASE_DIR="output_new_definitions"   # Base directory for output files
DATA_DIR="20_human_eval_cases"             # Data directory containing case documents

# ----------------------------------------------------------------------
# Optional: Define specific configurations for certain models
# ----------------------------------------------------------------------
get_model_config() {
  local model="$1"
  local resume="$DEFAULT_RESUME"
  local debug="$DEFAULT_DEBUG"
  
  echo "$resume $debug"
}

# ----------------------------------------------------------------------
# Function to submit a single job
# ----------------------------------------------------------------------
submit_job() {
  local case_id="$1"
  local checklist_config="$2"
  local model="$3"
  local max_steps="$4"
  local resume="$5"
  local debug="$6"
  
  # Extract model suffix for job name
  local model_suffix=$(echo "$model" | awk -F'/' '{print $NF}')
  # Extract config suffix for job name
  local config_suffix=$(basename "$checklist_config" .yaml)
  local job_name="agent_${case_id}_${config_suffix}_${model_suffix}_s${max_steps}"
  
  if [[ "$resume" == "true" ]]; then
    job_name="${job_name}_resume"
  fi
  
  echo "----------------------------------------"
  echo "Submitting job: ${job_name}"
  echo "  Case ID: ${case_id}"
  echo "  Config: ${config_suffix}"
  echo "  Model: ${model}"
  echo "  Max Steps: ${max_steps}"
  echo "  Resume: ${resume}"
  echo "  Debug: ${debug}"
  
  sbatch --export=ALL,CASE_ID="$case_id",CHECKLIST_CONFIG="$checklist_config",MODEL_NAME="$model",MAX_STEPS="$max_steps",RESUME="$resume",DEBUG="$debug",OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR",DATA_DIR="$DATA_DIR" \
         --job-name="${job_name}" \
         run_agent.sbatch
}

# ----------------------------------------------------------------------
# Main submission loop
# ----------------------------------------------------------------------
echo "========================================="
echo "Legal Agent Job Submission - GROUPED CONFIGS"
echo "========================================="
echo "Cases: ${#CASE_IDS[@]}"
echo "Checklist Configs: ${#CHECKLIST_CONFIGS[@]} (all 9 grouped configs)"
echo "Models: ${#MODELS[@]}"
echo "Max Steps configs: ${#MAX_STEPS[@]}"
echo "Total jobs: $((${#CASE_IDS[@]} * ${#CHECKLIST_CONFIGS[@]} * ${#MODELS[@]} * ${#MAX_STEPS[@]}))"
echo "========================================="
echo ""

job_count=0

# Loop order: for each case, submit all config variations first
for case_id in "${CASE_IDS[@]}"; do
  echo ""
  echo "Processing Case: ${case_id}"
  echo "========================================="
  
  for checklist_config in "${CHECKLIST_CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
      # Get model-specific configuration
      read -r resume debug <<< "$(get_model_config "$model")"
      
      for max_steps in "${MAX_STEPS[@]}"; do
        submit_job "$case_id" "$checklist_config" "$model" "$max_steps" "$resume" "$debug"
        job_count=$((job_count + 1))
      done
    done
  done
done

echo ""
echo "========================================="
echo "All $job_count jobs submitted!"
echo "========================================="
echo ""
echo "Monitor job status with:"
echo "  squeue -u $USER"
echo ""
echo "View logs in:"
echo "  agent_logs/{model_name}/{case_id}/"
echo ""
echo "Results will be saved in:"
echo "  output_new_definitions/{model_name}/{case_id}/grouped/{config_name}/"
echo "  e.g., output_new_definitions/gpt-oss-20b-BF16/45696/grouped/01_basic_case_info/"
echo "        output_new_definitions/gpt-oss-20b-BF16/45696/grouped/02_legal_foundation/"
echo "========================================="