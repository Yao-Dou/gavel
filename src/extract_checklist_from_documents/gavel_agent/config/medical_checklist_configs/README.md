# Medical Checklist Configurations

This directory contains modular checklist configurations for the agent system applied to the **medical-review plain-language summarization** domain, allowing targeted extraction of specific subsets of the 29 medical checklist items.

## Directory Structure

```
medical_checklist_configs/
├── all/
│   └── all_29_items.yaml              # Complete 29-item checklist
├── grouped/                           # 6 thematic groups
│   ├── 01_clinical_question_structure.yaml  (6 items)
│   ├── 02_outcomes_specification.yaml       (5 items)
│   ├── 03_results_reporting.yaml            (5 items)
│   ├── 04_certainty_evidence_quality.yaml   (8 items)
│   ├── 05_contextual_background.yaml        (3 items)
│   └── 06_applicability_currency.yaml       (2 items)
└── individual/                        # 29 single-item configs
    ├── 01_condition_defined.yaml
    ├── 02_population_defined.yaml
    ├── ... (26 more files)
    └── 29_evidence_currency.yaml
```

## Usage

### Running with specific configs

Medical runs must pass `--checklist-config` explicitly — the default in `run_agent.py` still points to the legal `all_26_items.yaml`.

```bash
# Use all 29 medical items
python run_agent.py data/<medical_case_dir> --checklist-config config/medical_checklist_configs/all/all_29_items.yaml

# Use a grouped config
python run_agent.py data/<medical_case_dir> --checklist-config config/medical_checklist_configs/grouped/01_clinical_question_structure.yaml

# Use an individual item config
python run_agent.py data/<medical_case_dir> --checklist-config config/medical_checklist_configs/individual/07_primary_benefit_outcome.yaml
```

### Batch runs

Use the consolidated runner (from the `gavel_agent/` folder) with
`--domain medical` — it selects this config directory and the medical data
directory automatically:
```bash
./run_agent_jobs.sh --domain medical                          # all 29 items
./run_agent_jobs.sh --domain medical --category grouped       # 6 groups
./run_agent_jobs.sh --domain medical --category individual    # 29 single items
```
See `./run_agent_jobs.sh --help` for every option.

## Group Breakdown

### 1. Clinical Question Structure (6 items)
- Condition_Defined
- Population_Defined
- Intervention_Defined
- Comparator_Defined
- Timing_Intervention
- Purpose_Aim

### 2. Outcomes Specification (5 items)
- Primary_Benefit_Outcome
- Secondary_Benefit_Outcome
- Harm_Safety_Outcome
- Outcome_Timeframe
- Outcome_Definition

### 3. Results Reporting (5 items)
- Benefit_Direction
- Harm_Direction
- Magnitude_Descriptor
- Quantitative_Data
- Evidence_Absence

### 4. Certainty and Evidence Quality (8 items)
- Certainty_Level
- Reason_Downgrading
- Study_Design
- Number_Studies
- Total_Sample_Size
- Search_Strategy
- Inclusion_Criteria
- Evidence_Gaps

### 5. Contextual / Background Information (3 items)
- Condition_Background
- Intervention_Rationale
- Definition_Technical_Terms

### 6. Applicability & Currency (2 items)
- Applicability
- Evidence_Currency

## Output Structure

When using different configs, outputs are organized by category:
```
output/
└── {model_name}/
    └── {case_id}/
        ├── all/
        │   └── {config_name}/
        │       ├── checklist.json
        │       ├── ledger.jsonl
        │       └── stats.json
        ├── grouped/
        │   └── {config_name}/
        │       └── ...
        └── individual/
            └── {config_name}/
                └── ...
```

For example:
- `output/gpt-oss-20b-BF16/<case_id>/all/all_29_items/`
- `output/gpt-oss-20b-BF16/<case_id>/grouped/01_clinical_question_structure/`
- `output/gpt-oss-20b-BF16/<case_id>/individual/07_primary_benefit_outcome/`

## Benefits

- **Parallel Processing**: Run multiple agents on different item groups simultaneously
- **Targeted Extraction**: Focus computational resources on specific items
- **Modular Testing**: Test and debug extraction for specific items independently
- **Scalable**: Easy to add new groupings or modify existing ones
- **Efficient**: Smaller configs may require fewer tokens and steps

## Notes on Domain Generality

The agent runtime (prompts, tools, state, driver) is domain-general. The only medical-specific content lives in this directory. To extend to another domain, mirror this structure with a new `<domain>_checklist_configs/` folder.
