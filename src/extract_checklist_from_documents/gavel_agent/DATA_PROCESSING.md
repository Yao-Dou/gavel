# Data Processing for the Gavel Agent

This document describes how to process case documents for use with the agent scaffold. Two input schemas are supported, selected with `--domain`:

- `--domain legal` (default): multi_lexsum court-case format
- `--domain medical`: Cochrane systematic-review format

## Data Format

### Input Format (legal, default)
The input data is located in `data/full_case_data/` at the repo root (JSON files containing a list of cases). Each case has:
- `case_id`: Unique identifier
- `case_documents_text`: List of document texts
- `case_documents_title`: List of document titles
- `case_documents_doc_type`: List of document types
- `case_documents_token_num`: Token counts (will be recalculated)
- Additional metadata (filing_date, case_url, case_type)

### Input Format (medical)
Medical input lives in `data/full_case_data/medical/` (e.g. `10_human_eval_cases.json`). Each record is one Cochrane review whose sections (Background, Objectives, Methods, Results, Discussion, References) act as the "documents". Differences from the legal schema:
- `case_id` is a PMC identifier (e.g. `PMC11706636`)
- There is **no** `case_documents_doc_type` (or per-document dates) — the section titles are used as document types
- Case-level metadata fields are `publication_date`, `case_url`, `doi`, `pmid`, `journal`, `authors`, `year`, `title`, `subjects`
- Output is prefixed with the domain: `data/medical_<input_stem>/`

### Output Format
The processed data is organized as:
```
data/
  {dataset_name}/
    {case_id}/
      complaint_001.txt
      motion_002.txt
      order_003.txt
      ...
      metadata.json
```

Each case directory contains:
- Individual document files (`.txt`)
- `metadata.json` with document information and token counts

## Usage

### Basic Processing
Process all cases from an input file:
```bash
# Legal (default domain) -> data/20_human_eval_cases/
python data_processing.py ../../../data/full_case_data/20_human_eval_cases.json

# Medical -> data/medical_10_human_eval_cases/
python data_processing.py ../../../data/full_case_data/medical/10_human_eval_cases.json --domain medical
```

### Process Specific Cases
Process only specific case IDs:
```bash
python data_processing.py input.json --case-ids 46210 46094
```

### Dry Run
Test processing without writing files:
```bash
python data_processing.py input.json --dry-run
```

### Validate Output
Validate the processed data structure:
```bash
python data_processing.py input.json --validate
```

### Custom Output Directory
Specify a different output directory:
```bash
python data_processing.py input.json --output-dir custom_data
```

### Use Different Tokenizer
Process with a different model's tokenizer:
```bash
python data_processing.py input.json --model Qwen/Qwen3-14B
```

## Running the Agent on Processed Data

After processing, run the agent on a specific case:
```bash
# Legal
python run_agent.py data/20_human_eval_cases/46210 --model Qwen/Qwen3-8B

# Medical (must pass the medical checklist config)
python run_agent.py data/medical_10_human_eval_cases/PMC11706636 \
    --checklist-config config/medical_checklist_configs/all/all_29_items.yaml
```

Or sweep every prepared case with the batch runner (see `./run_agent_jobs.sh --help`):
```bash
./run_agent_jobs.sh                    # legal
./run_agent_jobs.sh --domain medical   # medical
```

## Token Counts

The processing script recalculates token counts using the specified model's tokenizer (default: Qwen/Qwen3-8B). This ensures accurate token counts for:
- Budget management during agent execution
- Document chunking decisions
- Token limit compliance

## Document Naming Convention

Documents are named using the pattern:
```
{doc_type}_{index:03d}_{sanitized_title}.txt
```

Where:
- `doc_type`: Sanitized document type (legal: e.g. "complaint", "motion"; medical: the section title, e.g. "background", "methods")
- `index`: 3-digit document index
- `sanitized_title`: Optional sanitized title (first 20 chars; omitted when identical to the doc type, as in medical sections)

## Metadata Structure

Domain-specific case-level fields are propagated into `metadata.json` when present — legal: `filing_date`, `case_url`, `case_type`; medical: `publication_date`, `case_url`, `doi`, `pmid`, `journal`, `authors`, `year`, `title`, `subjects`.

Each case's `metadata.json` contains:
```json
{
  "case_id": "46210",
  "document_count": 14,
  "total_tokens": 672575,
  "filing_date": "2023-02-28",
  "case_url": "...",
  "documents": [
    {
      "filename": "complaint_001.txt",
      "title": "Complaint",
      "doc_type": "Complaint",
      "token_count": 9952,
      "doc_id": "157685",
      "date": "2023-02-28"
    },
    ...
  ]
}
```

## Large Documents

Some legal documents can be very large (>100K tokens). The agent handles this by:
- Reading documents in chunks
- Using search to jump to relevant sections
- Tracking coverage to avoid re-reading

## Example Workflow

1. Process the data:
```bash
python data_processing.py ../../../data/full_case_data/20_human_eval_cases.json
```

2. Run the agent on a case:
```bash
python run_agent.py data/20_human_eval_cases/46210
```

3. Check results:
```bash
cat output/run_*.json | jq '.completion_stats'
```

## Statistics

The processing script provides statistics including:
- Total cases processed
- Total documents
- Total tokens (with new tokenizer)
- Average documents per case
- Average tokens per case

## Error Handling

The script handles:
- Missing or inconsistent document lists
- Invalid filenames
- Large documents exceeding model limits
- Encoding issues

Errors are logged and reported in the summary.