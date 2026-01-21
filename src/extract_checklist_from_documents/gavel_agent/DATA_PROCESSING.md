# Data Processing for Legal Agent

This document describes how to process legal case documents for use with the legal agent scaffold.

## Data Format

### Input Format
The input data is located in `data/full_case_data/` (JSON files containing a list of cases). Each case has:
- `case_id`: Unique identifier
- `case_documents_text`: List of document texts
- `case_documents_title`: List of document titles
- `case_documents_doc_type`: List of document types
- `case_documents_token_num`: Token counts (will be recalculated)
- Additional metadata (filing_date, case_url, etc.)

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
python data_processing.py ../../../data/full_case_data/20_human_eval_cases.json
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
python run_agent.py data/20_human_eval_cases/46210 --model Qwen/Qwen3-8B
```

Or process multiple cases in batch:
```bash
python run_agent.py data/20_human_eval_cases --batch
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
- `doc_type`: Sanitized document type (e.g., "complaint", "motion")
- `index`: 3-digit document index
- `sanitized_title`: Optional sanitized title (first 20 chars)

## Metadata Structure

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