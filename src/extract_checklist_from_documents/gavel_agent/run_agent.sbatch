#!/bin/bash
#
# Generic SLURM launcher for a single GAVEL-Agent run.
# Used by run_agent_jobs.sh --use-slurm; you normally don't invoke this directly.
#
# Adjust the resource headers below for your cluster (GPU type/count, CPUs,
# partition/qos if your site requires them). Each run loads the model
# in-process with vLLM, so request enough GPU memory for the chosen model.
#
#SBATCH --job-name=gavel_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=agent_logs/%x-%j.out

# Example manual launch:
# sbatch --export=ALL,CASE_ID="46210",\
#        MODEL_NAME="unsloth/gpt-oss-20b-BF16",\
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
echo "SLURM_NODELIST: ${SLURM_JOB_NODELIST:-<none>}"
echo "========================================="

set -euo pipefail

# ----------------------------------------------------------------------
# Agent parameters (from environment or sensible defaults)
# ----------------------------------------------------------------------
case_id=${CASE_ID:?CASE_ID must be set (e.g. 46210 or PMC11706636)}
model_name=${MODEL_NAME:-"unsloth/gpt-oss-20b-BF16"}
checklist_config=${CHECKLIST_CONFIG:-"config/checklist_configs/all/all_26_items.yaml"}
max_steps=${MAX_STEPS:-"100"}
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
    config_category="custom"
fi

# ----------------------------------------------------------------------
# Base directories (everything is relative to this script's folder)
# ----------------------------------------------------------------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_PATH="${BASE_DIR}/data/${data_dir}/${case_id}"
OUTPUT_DIR="${BASE_DIR}/${output_base_dir}/${model_suffix}/${case_id}/${config_category}/${config_suffix}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${BASE_DIR}/agent_logs"

# ----------------------------------------------------------------------
# Log file for the python run (separate from the SLURM log)
# ----------------------------------------------------------------------
if [[ "$debug" == "true" ]]; then
    log_path="${OUTPUT_DIR}/debug.log"
else
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
echo "GAVEL-Agent Configuration"
echo "========================================="
echo "Case ID: ${case_id}"
echo "Model: ${model_name}"
echo "Checklist Config: ${checklist_config}"
echo "Config Category: ${config_category}"
echo "Config Name: ${config_suffix}"
echo "Max Steps: ${max_steps}"
echo "Resume: ${resume}"
echo "Debug: ${debug}"
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
