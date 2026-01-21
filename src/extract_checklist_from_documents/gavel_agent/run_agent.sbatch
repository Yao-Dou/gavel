#!/bin/bash
#
#SBATCH --job-name=legal_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a40:4       # Adjust GPU requirements as needed
#SBATCH --qos=short
#SBATCH --partition=overcap
#SBATCH --output=agent_logs/%x-%j.out
#SBATCH --exclude=gundam,dendrite,puma,xaea-12,spot,hk47,consu,nestor,dave

# Example launch:
# sbatch --export=ALL,CASE_ID="45696",\
#        MODEL_NAME="Qwen/Qwen3-8B",\
#        CHECKLIST_CONFIG="config/checklist_configs/all/all_26_items.yaml",\
#        MAX_STEPS="100",\
#        RESUME="false",\
#        DEBUG="false",\
#        OUTPUT_BASE_DIR="output",\
#        DATA_DIR="20_human_eval_cases" \
#        run_agent.sbatch

echo "========================================="
echo "START TIME: $(date)"
echo "RUNNING ON: $(hostname)"
echo "SLURM_NODELIST: ${SLURM_JOB_NODELIST}"
echo "========================================="

set -euo pipefail
set -x

# ----------------------------------------------------------------------
# Environment setup (if you keep a shared `.env` for CUDA paths, etc.)
# ----------------------------------------------------------------------
set -o allexport
source /srv/nlprx-lab/share6/douy/common.env
set +o allexport

# ----------------------------------------------------------------------
# Agent parameters (from environment or sensible defaults)
# ----------------------------------------------------------------------
case_id=${CASE_ID:-"45696"}
model_name=${MODEL_NAME:-"Qwen/Qwen3-8B"}
checklist_config=${CHECKLIST_CONFIG:-"config/checklist_configs/all/all_26_items.yaml"}
max_steps=${MAX_STEPS:-"1000"}
resume=${RESUME:-"false"}              # "true" or "false"
debug=${DEBUG:-"false"}                # "true" or "false"
output_base_dir=${OUTPUT_BASE_DIR:-"output"}
data_dir=${DATA_DIR:-"20_human_eval_cases"}

# Extract model suffix for output directory
model_suffix=$(echo "$model_name" | awk -F'/' '{print $NF}')

# Extract config suffix and category for output directory
config_suffix=$(basename "$checklist_config" .yaml)

# Determine config category (all, grouped, or individual) from path
if [[ "$checklist_config" == *"/all/"* ]]; then
    config_category="all"
elif [[ "$checklist_config" == *"/grouped/"* ]]; then
    config_category="grouped"
elif [[ "$checklist_config" == *"/individual/"* ]]; then
    config_category="individual"
else
    # Default to "custom" for configs outside standard structure
    config_category="custom"
fi

# ----------------------------------------------------------------------
# Base directories
# ----------------------------------------------------------------------
BASE_DIR="/coc/pskynet6/douy/summarization-rl/legal_agents"
CORPUS_PATH="${BASE_DIR}/data/${data_dir}/${case_id}"
OUTPUT_DIR="${BASE_DIR}/${output_base_dir}/${model_suffix}/${case_id}/${config_category}/${config_suffix}"

# ----------------------------------------------------------------------
# Create necessary directories
# ----------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
mkdir -p "${BASE_DIR}/agent_logs"

# ----------------------------------------------------------------------
# Build the log file path for the python run (separate from Slurm log)
# ----------------------------------------------------------------------
if [[ "$debug" == "true" ]]; then
    # In debug mode, save log to output directory (consistent with debug_case.sh)
    log_path="${OUTPUT_DIR}/debug.log"
    echo "Debug mode: Log will be saved to output directory"
else
    # In normal mode, save log to agent_logs directory
    log_name="${case_id}_${config_category}_${config_suffix}_steps${max_steps}"
    if [[ "$resume" == "true" ]]; then
        log_name="${log_name}_resume"
    fi
    log_path="${BASE_DIR}/agent_logs/${model_suffix}/${case_id}/${log_name}.log"
    mkdir -p "$(dirname "$log_path")"
fi

# ----------------------------------------------------------------------
# Print configuration
# ----------------------------------------------------------------------
echo ""
echo "========================================="
echo "Legal Agent Configuration"
echo "========================================="
echo "Case ID: ${case_id}"
echo "Model: ${model_name}"
echo "Checklist Config: ${checklist_config}"
echo "Config Category: ${config_category}"
echo "Config Name: ${config_suffix}"
echo "Max Steps: ${max_steps}"
echo "Resume: ${resume}"
echo "Debug: ${debug}"
echo "Output Base Dir: ${output_base_dir}"
echo "Data Dir: ${data_dir}"
echo "Corpus Path: ${CORPUS_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Log Path: ${log_path}"
echo "========================================="
echo ""

# ----------------------------------------------------------------------
# Build command arguments
# ----------------------------------------------------------------------
cmd_args=""
if [[ "$resume" == "true" ]]; then
    cmd_args="${cmd_args} --resume"
fi
if [[ "$debug" == "true" ]]; then
    cmd_args="${cmd_args} --debug"
fi

# ----------------------------------------------------------------------
# Change to the legal_agents directory
# ----------------------------------------------------------------------
cd "$BASE_DIR"

# ----------------------------------------------------------------------
# Launch the agent
# ----------------------------------------------------------------------
srun python run_agent.py "${CORPUS_PATH}" \
    --model "${model_name}" \
    --checklist-config "${checklist_config}" \
    --max-steps ${max_steps} \
    --store-path "${OUTPUT_DIR}/checklist.json" \
    --ledger-path "${OUTPUT_DIR}/ledger.jsonl" \
    ${cmd_args} \
    2>&1 | tee "$log_path"

# ----------------------------------------------------------------------
# Check exit status and print summary
# ----------------------------------------------------------------------
exit_status=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [[ $exit_status -eq 0 ]]; then
    echo "Agent completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    
    # Show stats summary if available
    if [[ -f "${OUTPUT_DIR}/stats.json" ]]; then
        echo ""
        echo "Token Usage Summary:"
        python -c "
import json
with open('${OUTPUT_DIR}/stats.json', 'r') as f:
    stats = json.load(f)
    print(f\"  Model: {stats.get('model', 'N/A')}\")
    print(f\"  Total Steps: {stats.get('steps', 0)}\")
    print(f\"  Total System Tokens: {stats.get('total_system_prompt_tokens', 0):,}\")
    print(f\"  Total User Tokens: {stats.get('total_user_prompt_tokens', 0):,}\")
    print(f\"  Total Completion Tokens: {stats.get('total_completion_tokens', 0):,}\")
    total = stats.get('total_system_prompt_tokens', 0) + stats.get('total_user_prompt_tokens', 0) + stats.get('total_completion_tokens', 0)
    print(f\"  Total All Tokens: {total:,}\")
" 2>/dev/null || echo "  Could not read stats file"
    fi
else
    echo "Agent terminated with errors (exit code: $exit_status)"
    echo "Check the log file: ${log_path}"
fi
echo "========================================="
echo "END TIME: $(date)"
echo "========================================="

exit $exit_status