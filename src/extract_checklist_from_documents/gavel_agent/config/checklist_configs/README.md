# Checklist Configurations

This directory contains modular checklist configurations for the legal agent system, allowing targeted extraction of specific subsets of the 26 legal checklist items.

## Directory Structure

```
checklist_configs/
├── all/
│   └── all_26_items.yaml       # Complete 26-item checklist
├── grouped/                    # 9 thematic groups
│   ├── 01_basic_case_info.yaml      (4 items)
│   ├── 02_legal_foundation.yaml     (3 items)
│   ├── 03_judge_info.yaml           (1 item)
│   ├── 04_related_cases.yaml        (2 items)
│   ├── 05_filings_proceedings.yaml  (5 items)
│   ├── 06_decrees.yaml              (3 items)
│   ├── 07_settlements.yaml          (5 items)
│   ├── 08_monitoring.yaml           (2 items)
│   └── 09_context.yaml              (1 item)
└── individual/                  # 26 single-item configs
    ├── 01_filing_date.yaml
    ├── 02_parties.yaml
    ├── ... (24 more files)
    └── 26_factual_basis.yaml
```

## Usage

### Running with specific configs

```bash
# Use all 26 items (default)
python run_agent.py data/case_123

# Use a grouped config
python run_agent.py data/case_123 --checklist-config config/checklist_configs/grouped/01_basic_case_info.yaml

# Use an individual item config
python run_agent.py data/case_123 --checklist-config config/checklist_configs/individual/08_judge_name.yaml
```

### Batch submission with SLURM

#### Submit jobs for specific configs
Edit `submit_agent_jobs.sh` and uncomment the desired configs in the `CHECKLIST_CONFIGS` array, then run:
```bash
./submit_agent_jobs.sh
```

#### Submit all grouped configs
Use the specialized script for all 9 grouped configs:
```bash
./submit_grouped_jobs.sh
```

## Group Breakdown

### 1. Basic Case Information (4 items)
- Filing_Date
- Who_are_the_Parties
- Class_Action_or_Individual_Plaintiffs
- Type_of_Counsel

### 2. Legal Foundation (3 items)
- Cause_of_Action
- Statutory_or_Constitutional_Basis_for_the_Case
- Remedy_Sought

### 3. Judge Information (1 item)
- First_and_Last_name_of_Judge

### 4. Related Cases (2 items)
- Consolidated_Cases_Noted
- Related_Cases_Listed_by_Their_Case_Code_Number

### 5. Filings and Proceedings (5 items)
- Note_Important_Filings
- Court_Rulings
- All_Reported_Opinions_Cited_with_Shortened_Bluebook_Citation
- Trials
- Appeal

### 6. Decrees (3 items)
- Significant_Terms_of_Decrees
- Dates_of_All_Decrees
- How_Long_Decrees_will_Last

### 7. Settlements (5 items)
- Significant_Terms_of_Settlement
- Date_of_Settlement
- How_Long_Settlement_will_Last
- Whether_the_Settlement_is_Court-enforced_or_Not
- Disputes_Over_Settlement_Enforcement

### 8. Monitoring (2 items)
- Name_of_the_Monitor
- Monitor_Reports

### 9. Context (1 item)
- Factual_Basis_of_Case

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
- `output/gpt-oss-20b-BF16/45696/all/all_26_items/`
- `output/gpt-oss-20b-BF16/45696/grouped/01_basic_case_info/`
- `output/gpt-oss-20b-BF16/45696/grouped/02_legal_foundation/`
- `output/gpt-oss-20b-BF16/45696/individual/08_judge_name/`

Logs are similarly organized:
- `agent_logs/{model_name}/{case_id}/`

## Benefits

- **Parallel Processing**: Run multiple agents on different item groups simultaneously
- **Targeted Extraction**: Focus computational resources on specific items
- **Modular Testing**: Test and debug extraction for specific items independently
- **Scalable**: Easy to add new groupings or modify existing ones
- **Efficient**: Smaller configs may require fewer tokens and steps