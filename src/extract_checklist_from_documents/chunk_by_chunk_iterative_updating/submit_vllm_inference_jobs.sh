#!/usr/bin/env bash
set -euo pipefail

# Define the settings you want to sweep over
declare -a FILES=(
  # "2025_example_cases"
  "20_human_eval_cases"
  # "20_human_eval_cases_2"
  # Add more file names here as needed
)

# Define which checklist items to process
# Empty string means process all items
# You can also specify individual items like "Cause_of_Action", "Court_Rulings", etc.
declare -a CHECKLIST_ITEMS=(
  ""  # Process all items
  # "Cause_of_Action"
  # "Court_Rulings"
  # "Who_are_the_Parties"
  # "Factual_Basis_of_Case"
  # "Remedy_Sought"
  # "Filing_Date"
  # "First_and_Last_name_of_Judge"
  # "Class_Action_or_Individual_Plaintiffs_"
  # "Statutory_or_Constitutional_Basis_for_the_Case"
  # "Note_Important_Filings"
  # "Appeal"
  # "Trials"
  # "Date_of_Settlement"
  # "Significant_Terms_of_Settlement"
  # "How_Long_Settlement_will_Last"
  # "Whether_the_Settlement_is_Court-enforced_or_Not"
  # "Disputes_Over_Settlement_Enforcement"
  # "Dates_of_All_Decrees"
  # "Significant_Terms_of_Decrees"
  # "How_Long_Decrees_will_Last"
  # "Name_of_the_Monitor."
  # "Monitor_Reports"
  # "Type_of_Counsel"
  # "Consolidated_Cases_Noted"
  # "Related_Cases_Listed_by_Their_Case_Code_Number"
  # "All_Reported_Opinions_Cited_with_Shortened_Bluebook_Citation"
)

declare -a MODELS=(
  "unsloth/gpt-oss-20b-BF16"
  # "unsloth/gpt-oss-120b-BF16"
  # "Qwen/Qwen3-32B"
  # "Qwen/Qwen3-8B"
  # "Qwen/Qwen3-14B"
  # "Qwen/Qwen3-30B-A3B-Thinking-2507"
  # "Qwen/Qwen3-30B-A3B-Instruct-2507"
  # "Qwen/Qwen3-4B-Thinking-2507"
  # "Qwen/Qwen3-4B-Instruct-2507"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"
)

# Default enable_thinking setting for non-2507 models
# Set this to "true" or "false" as needed
DEFAULT_ENABLE_THINKING="true"

# Submit jobs for each combination
for file in "${FILES[@]}"; do
  for model in "${MODELS[@]}"; do
    # Automatically determine enable_thinking based on model name for 2507 versions
    if [[ "$model" == *"Thinking-2507"* ]]; then
      ENABLE_THINKING="true"
      echo "Auto-setting enable_thinking=true for Thinking model: $model"
    elif [[ "$model" == *"Instruct-2507"* ]]; then
      ENABLE_THINKING="false"
      echo "Auto-setting enable_thinking=false for Instruct model: $model"
    else
      # For non-2507 models, use the default setting
      ENABLE_THINKING="$DEFAULT_ENABLE_THINKING"
    fi
    
    for item in "${CHECKLIST_ITEMS[@]}"; do
      # Build job name suffix based on whether we're processing a specific item
      if [[ -n "$item" ]]; then
        job_suffix="_${item}"
        echo "Submitting job for file=$file, model=$model, item=$item, thinking=$ENABLE_THINKING"
      else
        job_suffix="_all"
        echo "Submitting job for file=$file, model=$model, all items, thinking=$ENABLE_THINKING"
      fi
      
      sbatch --export=ALL,FILE_NAME="$file",\
CHECKLIST_ITEM="$item",\
ENABLE_THINKING="$ENABLE_THINKING",\
MODEL_NAME="$model" \
             --job-name="chunk_vllm${job_suffix}" \
             vllm_inference.sbatch
    done
  done
done

echo "All jobs submitted!"