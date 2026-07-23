#!/usr/bin/env bash
set -euo pipefail

# Domain to run: "legal" or "medical". Selects the prompt template and
# checklist definitions. Data-file naming: legal files are unprefixed
# (data/{file}.json); other domains are prefixed (data/{domain}_{file}.json).
DOMAIN="legal"

# Define the data files to process, WITHOUT the domain prefix
# (legal "20_human_eval_cases" loads data/20_human_eval_cases.json;
#  medical "10_human_eval_cases" loads data/medical_10_human_eval_cases.json)
declare -a FILES=(
  # "2025_example_cases"
  # "20_human_eval_cases"
  # "20_human_eval_cases_2"
  # "50_cases_for_benchmarking"
  # "50_cases_for_benchmarking_2"
  # -- medical files (set DOMAIN="medical") --
  # "10_human_eval_cases"
  # Add more file names here as needed
)

# Define which checklist items to process
# Empty string means process all items
declare -a CHECKLIST_ITEMS=(
  ""  # Process all items
  # -- legal items --
  # "All_Reported_Opinions_Cited_with_Shortened_Bluebook_Citation"
  # "Appeal"
  # "Cause_of_Action"
  # "Class_Action_or_Individual_Plaintiffs"
  # "Consolidated_Cases_Noted"
  # "Court_Rulings"
  # "Date_of_Settlement"
  # "Dates_of_All_Decrees"
  # "Disputes_Over_Settlement_Enforcement"
  # "Factual_Basis_of_Case"
  # "Filing_Date"
  # "First_and_Last_name_of_Judge"
  # "How_Long_Decrees_will_Last"
  # "How_Long_Settlement_will_Last"
  # "Monitor_Reports"
  # "Name_of_the_Monitor"
  # "Note_Important_Filings"
  # "Related_Cases_Listed_by_Their_Case_Code_Number"
  # "Remedy_Sought"
  # "Significant_Terms_of_Decrees"
  # "Significant_Terms_of_Settlement"
  # "Statutory_or_Constitutional_Basis_for_the_Case"
  # "Trials"
  # "Type_of_Counsel"
  # "Whether_the_Settlement_is_Court-enforced_or_Not"
  # "Who_are_the_Parties"
  # -- medical items --
  # "Condition_Defined"
  # "Population_Defined"
  # "Intervention_Defined"
  # "Comparator_Defined"
  # "Timing_Intervention"
  # "Purpose_Aim"
  # "Primary_Benefit_Outcome"
  # "Secondary_Benefit_Outcome"
  # "Harm_Safety_Outcome"
  # "Outcome_Timeframe"
  # "Outcome_Definition"
  # "Benefit_Direction"
  # "Harm_Direction"
  # "Magnitude_Descriptor"
  # "Quantitative_Data"
  # "Evidence_Absence"
  # "Certainty_Level"
  # "Reason_Downgrading"
  # "Study_Design"
  # "Number_Studies"
  # "Total_Sample_Size"
  # "Search_Strategy"
  # "Inclusion_Criteria"
  # "Evidence_Gaps"
  # "Condition_Background"
  # "Intervention_Rationale"
  # "Definition_Technical_Terms"
  # "Applicability"
  # "Evidence_Currency"
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

# SLURM writes job logs into vllm_inference_logs/ (see the sbatch --output);
# the directory must exist at submission time.
mkdir -p vllm_inference_logs

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
        echo "Submitting job for domain=$DOMAIN, file=$file, model=$model, item=$item, thinking=$ENABLE_THINKING"
      else
        job_suffix="_all"
        echo "Submitting job for domain=$DOMAIN, file=$file, model=$model, all items, thinking=$ENABLE_THINKING"
      fi

      sbatch --export=ALL,DOMAIN="$DOMAIN",\
FILE_NAME="$file",\
CHECKLIST_ITEM="$item",\
ENABLE_THINKING="$ENABLE_THINKING",\
MODEL_NAME="$model" \
             --job-name="chunk_vllm_${DOMAIN}${job_suffix}" \
             vllm_inference.sbatch
    done
  done
done

echo "All jobs submitted!"
