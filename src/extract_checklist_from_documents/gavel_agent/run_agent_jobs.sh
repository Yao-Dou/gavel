#!/usr/bin/env bash
# =============================================================================
# run_agent_jobs.sh — run GAVEL-Agent checklist extraction jobs
# =============================================================================
#
# Runs the agent (run_agent.py) over every combination of
#     case  x  checklist config  x  model
# either locally (default, sequentially — each run loads the model on your
# GPUs via vLLM) or by submitting one SLURM job per combination (--use-slurm).
#
# Before running, prepare the per-case document directories once:
#     Legal:   python data_processing.py ../../../data/full_case_data/20_human_eval_cases.json
#     Medical: python data_processing.py ../../../data/full_case_data/medical/10_human_eval_cases.json --domain medical
# which creates data/20_human_eval_cases/ and data/medical_10_human_eval_cases/.
#
# -----------------------------------------------------------------------------
# ARGUMENTS (all optional; every one has a sensible default)
# -----------------------------------------------------------------------------
#   --domain legal|medical
#       Which domain to run. Selects the default checklist-config folder and
#       data directory:
#         legal   -> config/checklist_configs         + data/20_human_eval_cases
#         medical -> config/medical_checklist_configs + data/medical_10_human_eval_cases
#       Default: legal
#
#   --category all|grouped|individual
#       Which extraction granularity to run. Expands to the YAML files in the
#       corresponding subfolder of the domain's config directory:
#         all        -> one config covering every item   (legal: 26, medical: 29)
#         grouped    -> one config per thematic group    (legal: 9,  medical: 6)
#         individual -> one config per single item       (legal: 26, medical: 29)
#       Default: all
#
#   --checklist-configs "<path1> <path2> ..."
#       Advanced: run exactly these config YAMLs instead of the --category
#       expansion (paths relative to this folder).
#       Example: --checklist-configs "config/medical_checklist_configs/individual/07_primary_benefit_outcome.yaml"
#
#   --models "<model1> <model2> ..."
#       HuggingFace model identifiers, space-separated. Each run loads the
#       model locally with vLLM (no server needed). Models used in the paper:
#         Qwen/Qwen3-30B-A3B-Thinking-2507
#         unsloth/gpt-oss-20b-BF16
#       Default: unsloth/gpt-oss-20b-BF16
#
#   --case-ids "<id1> <id2> ..."
#       Which cases to process (legal: numeric ids like "46210"; medical: PMC
#       ids like "PMC11706636"). Default: every case directory found in
#       data/<data-dir>/.
#
#   --data-dir NAME
#       Case-directory folder under data/ (created by data_processing.py).
#       Default: 20_human_eval_cases (legal) / medical_10_human_eval_cases (medical)
#
#   --output-base DIR
#       Base output directory. Results are written to
#       <output-base>/<model>/<case_id>/<all|grouped|individual>/<config_name>/
#       containing checklist.json, ledger.jsonl, raw_responses.jsonl, stats.json.
#       Default: output
#
#   --max-steps N
#       Maximum agent steps per run. The paper used 200 for legal and 100 for
#       medical. Default: 100
#
#   --resume
#       Resume runs from an existing checklist store instead of starting fresh.
#
#   --debug
#       Verbose agent logging (full prompts/responses); the run log is written
#       to the run's output directory as debug.log.
#
#   --use-slurm
#       Submit each combination as a separate SLURM job via run_agent.sbatch
#       (adjust the generic #SBATCH headers in that file for your cluster)
#       instead of running sequentially on this machine.
#
#   --dry-run
#       Print every command that would be executed, without running anything.
#
#   -h | --help
#       Show this header.
#
# -----------------------------------------------------------------------------
# EXAMPLES
# -----------------------------------------------------------------------------
#   # Legal, all 26 items in one config, every prepared case, default model:
#   ./run_agent_jobs.sh
#
#   # Medical, all 29 items, both paper models:
#   ./run_agent_jobs.sh --domain medical \
#       --models "Qwen/Qwen3-30B-A3B-Thinking-2507 unsloth/gpt-oss-20b-BF16"
#
#   # Medical, one job per checklist item, two specific cases:
#   ./run_agent_jobs.sh --domain medical --category individual \
#       --case-ids "PMC11706636 PMC11770842"
#
#   # Legal grouped extraction with more steps, via SLURM:
#   ./run_agent_jobs.sh --category grouped --max-steps 200 --use-slurm
#
#   # See what would run without running it:
#   ./run_agent_jobs.sh --domain medical --category grouped --dry-run
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
DOMAIN="legal"
CATEGORY="all"
CHECKLIST_CONFIGS=""          # explicit override of the category expansion
MODELS="unsloth/gpt-oss-20b-BF16"
CASE_IDS=""                   # empty = auto-discover from data dir
DATA_DIR=""                   # empty = domain default
OUTPUT_BASE="output"
MAX_STEPS=100
RESUME="false"
DEBUG="false"
USE_SLURM="false"
DRY_RUN="false"

print_help() {
  # Print the header comment block (lines starting with '#' up to the first
  # non-comment line).
  sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)             DOMAIN="$2"; shift 2 ;;
    --category)           CATEGORY="$2"; shift 2 ;;
    --checklist-configs)  CHECKLIST_CONFIGS="$2"; shift 2 ;;
    --models)             MODELS="$2"; shift 2 ;;
    --case-ids)           CASE_IDS="$2"; shift 2 ;;
    --data-dir)           DATA_DIR="$2"; shift 2 ;;
    --output-base)        OUTPUT_BASE="$2"; shift 2 ;;
    --max-steps)          MAX_STEPS="$2"; shift 2 ;;
    --resume)             RESUME="true"; shift ;;
    --debug)              DEBUG="true"; shift ;;
    --use-slurm)          USE_SLURM="true"; shift ;;
    --dry-run)            DRY_RUN="true"; shift ;;
    -h|--help)            print_help; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)"; exit 1 ;;
  esac
done

# ----------------------------------------------------------------------
# Domain defaults
# ----------------------------------------------------------------------
case "$DOMAIN" in
  legal)
    CONFIG_ROOT="config/checklist_configs"
    [[ -z "$DATA_DIR" ]] && DATA_DIR="20_human_eval_cases"
    ;;
  medical)
    CONFIG_ROOT="config/medical_checklist_configs"
    [[ -z "$DATA_DIR" ]] && DATA_DIR="medical_10_human_eval_cases"
    ;;
  *) echo "Invalid --domain '$DOMAIN' (expected: legal or medical)"; exit 1 ;;
esac

# ----------------------------------------------------------------------
# Expand checklist configs
# ----------------------------------------------------------------------
declare -a CONFIGS=()
if [[ -n "$CHECKLIST_CONFIGS" ]]; then
  read -r -a CONFIGS <<< "$CHECKLIST_CONFIGS"
else
  case "$CATEGORY" in
    all|grouped|individual)
      for f in "$CONFIG_ROOT/$CATEGORY"/*.yaml; do
        [[ -e "$f" ]] && CONFIGS+=("$f")
      done
      ;;
    *) echo "Invalid --category '$CATEGORY' (expected: all, grouped, or individual)"; exit 1 ;;
  esac
fi
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No checklist configs found for domain='$DOMAIN' category='$CATEGORY'"; exit 1
fi
for f in "${CONFIGS[@]}"; do
  [[ -f "$f" ]] || { echo "Checklist config not found: $f"; exit 1; }
done

# ----------------------------------------------------------------------
# Resolve cases
# ----------------------------------------------------------------------
CORPUS_ROOT="data/$DATA_DIR"
declare -a CASES=()
if [[ -n "$CASE_IDS" ]]; then
  read -r -a CASES <<< "$CASE_IDS"
else
  if [[ ! -d "$CORPUS_ROOT" ]]; then
    echo "Data directory '$CORPUS_ROOT' not found."
    echo "Prepare it first with data_processing.py, e.g.:"
    if [[ "$DOMAIN" == "medical" ]]; then
      echo "  python data_processing.py ../../../data/full_case_data/medical/10_human_eval_cases.json --domain medical"
    else
      echo "  python data_processing.py ../../../data/full_case_data/20_human_eval_cases.json"
    fi
    exit 1
  fi
  for d in "$CORPUS_ROOT"/*/; do
    [[ -d "$d" ]] && CASES+=("$(basename "$d")")
  done
fi
if [[ ${#CASES[@]} -eq 0 ]]; then
  echo "No cases found in $CORPUS_ROOT (and no --case-ids given)"; exit 1
fi

read -r -a MODELS_ARR <<< "$MODELS"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
total=$(( ${#CASES[@]} * ${#CONFIGS[@]} * ${#MODELS_ARR[@]} ))
echo "========================================="
echo "GAVEL-Agent Job Runner"
echo "========================================="
echo "Domain:            $DOMAIN"
echo "Category:          $CATEGORY (${#CONFIGS[@]} config(s))"
echo "Models:            ${MODELS_ARR[*]}"
echo "Cases:             ${#CASES[@]}"
echo "Max steps:         $MAX_STEPS"
echo "Resume / Debug:    $RESUME / $DEBUG"
echo "Data dir:          $CORPUS_ROOT"
echo "Output base:       $OUTPUT_BASE"
echo "Mode:              $([[ "$USE_SLURM" == "true" ]] && echo "SLURM" || echo "local")$([[ "$DRY_RUN" == "true" ]] && echo " (dry run)")"
echo "Total runs:        $total"
echo "========================================="

# ----------------------------------------------------------------------
# Run / submit each combination
# ----------------------------------------------------------------------
# SLURM writes its own job logs to agent_logs/ (see run_agent.sbatch), and the
# directory must exist at submission time.
if [[ "$USE_SLURM" == "true" && "$DRY_RUN" != "true" ]]; then
  mkdir -p agent_logs
fi

run_count=0
declare -a FAILED=()

for case_id in "${CASES[@]}"; do
  for checklist_config in "${CONFIGS[@]}"; do
    for model in "${MODELS_ARR[@]}"; do
      model_suffix="${model##*/}"
      config_suffix="$(basename "$checklist_config" .yaml)"

      # Category of this particular config (used in the output path)
      if [[ "$checklist_config" == *"/all/"* ]]; then
        config_category="all"
      elif [[ "$checklist_config" == *"/grouped/"* ]]; then
        config_category="grouped"
      elif [[ "$checklist_config" == *"/individual/"* ]]; then
        config_category="individual"
      else
        config_category="custom"
      fi

      output_dir="$OUTPUT_BASE/$model_suffix/$case_id/$config_category/$config_suffix"
      corpus_path="$CORPUS_ROOT/$case_id"
      run_count=$((run_count + 1))
      label="[$run_count/$total] $case_id | $config_suffix | $model_suffix"

      if [[ "$USE_SLURM" == "true" ]]; then
        job_name="agent_${case_id}_${config_suffix}_${model_suffix}_s${MAX_STEPS}"
        [[ "$RESUME" == "true" ]] && job_name="${job_name}_resume"
        cmd=(sbatch
             --export=ALL,CASE_ID="$case_id",CHECKLIST_CONFIG="$checklist_config",MODEL_NAME="$model",MAX_STEPS="$MAX_STEPS",RESUME="$RESUME",DEBUG="$DEBUG",OUTPUT_BASE_DIR="$OUTPUT_BASE",DATA_DIR="$DATA_DIR"
             --job-name="$job_name"
             run_agent.sbatch)
        echo "$label -> submit"
        if [[ "$DRY_RUN" == "true" ]]; then
          echo "  DRY RUN: ${cmd[*]}"
        else
          "${cmd[@]}" || FAILED+=("$label")
        fi
      else
        cmd=(python run_agent.py "$corpus_path"
             --model "$model"
             --checklist-config "$checklist_config"
             --max-steps "$MAX_STEPS"
             --store-path "$output_dir/checklist.json"
             --ledger-path "$output_dir/ledger.jsonl")
        [[ "$RESUME" == "true" ]] && cmd+=(--resume)
        [[ "$DEBUG" == "true" ]] && cmd+=(--debug)

        if [[ "$DEBUG" == "true" ]]; then
          log_path="$output_dir/debug.log"
        else
          log_name="${case_id}_${config_category}_${config_suffix}_steps${MAX_STEPS}"
          [[ "$RESUME" == "true" ]] && log_name="${log_name}_resume"
          log_path="agent_logs/$model_suffix/$case_id/${log_name}.log"
        fi

        echo "$label -> run"
        if [[ "$DRY_RUN" == "true" ]]; then
          echo "  DRY RUN: ${cmd[*]}"
          echo "           (log: $log_path)"
        else
          mkdir -p "$output_dir" "$(dirname "$log_path")"
          if ! "${cmd[@]}" 2>&1 | tee "$log_path"; then
            FAILED+=("$label")
          fi
        fi
      fi
    done
  done
done

# ----------------------------------------------------------------------
# Wrap-up
# ----------------------------------------------------------------------
echo ""
echo "========================================="
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete: $total run(s) previewed."
elif [[ "$USE_SLURM" == "true" ]]; then
  echo "All $total job(s) submitted. Monitor with: squeue -u \$USER"
else
  echo "Finished $total run(s)."
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "Results tree: $OUTPUT_BASE/<model>/<case_id>/<all|grouped|individual>/<config_name>/"
echo "========================================="
