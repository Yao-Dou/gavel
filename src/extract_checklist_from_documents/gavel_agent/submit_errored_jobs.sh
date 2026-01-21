#!/bin/bash

# Auto-generated script to resubmit errored jobs
# Generated for 1 errored jobs
# Jobs missing ledger.jsonl will be resubmitted

set -e

# Job: Qwen3-30B-A3B-Thinking-2507/46390/individual/06_statutory_basis
sbatch --export=ALL,\
CASE_ID="46390",\
CHECKLIST_CONFIG="config/checklist_configs/individual/06_statutory_basis.yaml",\
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507",\
MAX_STEPS="200",\
RESUME="false",\
DEBUG="true",\
OUTPUT_BASE_DIR="output_new_definitions",\
DATA_DIR="20_human_eval_cases_2" \
--job-name="agent_46390_06_statutory_basis_Qwen3-30B-A3B-Thinking-2507_s200" \
run_agent.sbatch
