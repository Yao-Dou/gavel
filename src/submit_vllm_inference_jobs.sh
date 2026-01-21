#!/usr/bin/env bash
# =============================================================================
# vLLM Inference Job Submission Script
# =============================================================================
# Submits SLURM batch jobs for running vLLM inference across multiple
# file/folder/model combinations.
#
# Usage:
#   ./submit_vllm_inference_jobs.sh [OPTIONS]
#
# Options:
#   --help      Show this help message
#   --dry-run   Print commands without executing them
#
# Configuration:
#   Edit the arrays below to specify which combinations to run:
#   - FILES:   Input data file names (without .json extension)
#   - FOLDERS: Data folder paths relative to base directory
#   - MODELS:  HuggingFace model identifiers
#
# Environment Variables:
#   DEFAULT_ENABLE_THINKING: Set to "true" or "false" (default: "true")
#
# Auto-Detection:
#   - Models with "Thinking-2507" in name: enable_thinking=true
#   - Models with "Instruct-2507" in name: enable_thinking=false
#   - Other models: uses DEFAULT_ENABLE_THINKING
#
# Examples:
#   # Run with default settings
#   ./submit_vllm_inference_jobs.sh
#
#   # Preview jobs without submitting
#   ./submit_vllm_inference_jobs.sh --dry-run
#
#   # Set default thinking mode
#   DEFAULT_ENABLE_THINKING=false ./submit_vllm_inference_jobs.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --help|-h)
      head -35 "$0" | tail -33
      exit 0
      ;;
    --dry-run)
      DRY_RUN=true
      echo "DRY RUN MODE: Commands will be printed but not executed"
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# =============================================================================
# CONFIGURATION - Edit these arrays before running
# =============================================================================

# Input file names (without .json extension)
declare -a FILES=(
  "your_input_file"
  # Add more files here, e.g.:
  # "50_cases_for_benchmarking"
  # "model_name/input_file_name"
)

# Folder paths containing input JSON files
declare -a FOLDERS=(
  "legal/multi_lexsum/summarization"
  # Add more folders here, e.g.:
  # "legal/multi_lexsum/extract_checklist_evidence"
  # "legal/multi_lexsum/evaluate_checklist"
)

# HuggingFace model identifiers
declare -a MODELS=(
  "Qwen/Qwen3-14B"
  # Add more models here, e.g.:
  # "Qwen/Qwen3-32B"
  # "Qwen/Qwen3-30B-A3B-Thinking-2507"
  # "google/gemma-3-27b-it"
  # "unsloth/gpt-oss-20b-BF16"
)

# Default enable_thinking setting for models without auto-detection
DEFAULT_ENABLE_THINKING="${DEFAULT_ENABLE_THINKING:-true}"

# =============================================================================
# Job Submission Loop
# =============================================================================

job_count=0

for file in "${FILES[@]}"; do
  for folder in "${FOLDERS[@]}"; do
    for model in "${MODELS[@]}"; do
      # Auto-detect enable_thinking based on model name
      if [[ "$model" == *"Thinking-2507"* ]]; then
        ENABLE_THINKING="true"
      elif [[ "$model" == *"Instruct-2507"* ]]; then
        ENABLE_THINKING="false"
      else
        ENABLE_THINKING="$DEFAULT_ENABLE_THINKING"
      fi

      echo "----------------------------------------"
      echo "Job $((++job_count)):"
      echo "  File:     $file"
      echo "  Folder:   $folder"
      echo "  Model:    $model"
      echo "  Thinking: $ENABLE_THINKING"

      if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would submit: sbatch vllm_inference.sbatch"
      else
        sbatch --export=ALL,FILE_NAME="$file",FOLDER_PATH="$folder",\
ENABLE_THINKING="$ENABLE_THINKING",MODEL_NAME="$model" \
               vllm_inference.sbatch
      fi
    done
  done
done

echo "========================================"
echo "Total jobs submitted: $job_count"
