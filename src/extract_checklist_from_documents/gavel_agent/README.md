# Gavel Agent - Legal Document Checklist Extraction

An LLM-powered agent for extracting 26 structured checklist items from legal case documents with evidence-based citations.

## Features

- **Evidence-based extraction**: Every extracted value includes source citations with document name, location, and exact quotes
- **Multi-model support**: Works with Qwen3 (native tool calling) and GPT-OSS (Harmony format)
- **Configurable extraction**: Extract all 26 items, thematic groups (9 configs), or individual items (26 configs)
- **Two-stage stopping**: Agent reviews its work before finalizing extraction
- **Batch processing**: Process multiple cases in parallel with SLURM integration
- **Resume capability**: Continue interrupted runs from saved state

## Quick Start

### Prerequisites

- Python 3.8+
- vLLM or compatible inference server
- tiktoken (for tokenization)

### Installation

```bash
pip install pydantic pyyaml pytz tiktoken
```

### Basic Usage

```bash
# First, prepare documents (see DATA_PROCESSING.md)
python data_processing.py path/to/input.json

# Run agent on a single case
python run_agent.py data/case_folder --model Qwen/Qwen3-8B

# Run with debug output
python run_agent.py data/case_folder --model Qwen/Qwen3-8B --debug

# Batch processing multiple cases
python run_agent.py data/cases --batch --model Qwen/Qwen3-8B --output-dir results
```

For data preparation details, see [DATA_PROCESSING.md](DATA_PROCESSING.md).

## Architecture

### Directory Structure

```
gavel_agent/
├── run_agent.py               # CLI entry point
├── data_processing.py         # Data preparation script
├── agent/
│   ├── driver.py              # Main execution loop
│   ├── orchestrator.py        # LLM decision engine
│   ├── snapshot_builder.py    # State snapshot construction
│   ├── snapshot_formatter.py  # Markdown snapshot formatting
│   ├── llm_client.py          # vLLM/OpenAI-compatible client
│   ├── document_manager.py    # Document access layer
│   ├── tokenizer.py           # Token counting wrapper
│   ├── validator.py           # Stop condition validation
│   ├── logger.py              # Action/performance logging
│   ├── stats_tracker.py       # Token usage statistics
│   └── tools/
│       ├── base.py            # BaseTool interface
│       ├── list_documents.py
│       ├── read_document.py
│       ├── search_document_regex.py
│       ├── get_checklist.py
│       ├── update_checklist.py
│       └── append_checklist.py
├── state/
│   ├── store.py               # ChecklistStore + Ledger
│   └── schemas.py             # Pydantic models
├── config/
│   ├── model_config.yaml      # Model-specific sampling parameters
│   ├── prompts_qwen.yaml      # Qwen3 system prompt + tool definitions
│   ├── prompts_gpt_oss.yaml   # GPT-OSS Harmony format prompts
│   └── checklist_configs/     # Modular checklist definitions
│       ├── all/
│       ├── grouped/
│       └── individual/
├── README.md                  # This file
└── DATA_PROCESSING.md         # Data preparation guide
```

### Component Interaction

```
Input Documents
       ↓
DocumentManager (tokenization, metadata)
       ↓
SnapshotBuilder (compact state representation)
       ↓
Orchestrator (LLM decision-maker)
   ↙   ↘   ↖   ↗
Tools (read, search, update)
   ↙   ↘   ↖   ↗
ChecklistStore + Ledger (state persistence)
       ↓
Final Checklist JSON
```

## Agent Loop

The driver executes steps until completion or max_steps:

```
1. Build Snapshot
   - Run header (run_id, step, timestamp)
   - Task definition with checklist items to extract
   - Document catalog with coverage stats
   - Current checklist state (filled/empty keys)
   - Recent action history
   - Last tool result (ephemeral - only for current turn)

2. Call Orchestrator
   - Format snapshot as markdown
   - Send to LLM with system prompt
   - Parse JSON response (tool call or stop decision)
   - Handle parse errors with retry (max 2 retries)

3. Execute Tool
   - Validate arguments
   - Execute tool logic
   - Record to ledger
   - Update last_tool_result

4. Check Stop Conditions
   - Agent-initiated stop (two-stage mechanism)
   - Max steps reached
   - Continue to next step
```

### Two-Stage Stop Mechanism

When the agent decides to stop:

**First Stop Attempt:**
1. Agent issues `{"decision": "stop", "reason": "..."}`
2. Driver auto-injects `get_checklist()` call
3. Agent reviews full checklist state
4. Agent can continue extracting or confirm stop

**Second Stop Attempt:**
1. Agent confirms stop with second stop decision
2. Driver terminates execution
3. Results are finalized and saved

**Safety Limit:** Force stop after 3 stop attempts to prevent infinite loops.

This ensures the agent reviews its work before finalizing.

## Tool Reference

### list_documents

Discover available documents in the corpus.

```json
{"tool": "list_documents", "args": {}}
```

**Returns:**
```json
{
  "documents": [
    {"name": "complaint.txt", "type": "Complaint", "token_count": 9952, "visited": false}
  ]
}
```

### read_document

Read a specific token range from a document.

```json
{
  "tool": "read_document",
  "args": {
    "doc_name": "complaint.txt",
    "start_token": 0,
    "end_token": 5000
  }
}
```

**Parameters:**
- `doc_name`: Document name (supports fuzzy matching)
- `start_token`: Starting position (inclusive)
- `end_token`: Ending position (exclusive, max 10,000 token range)

### search_document_regex

Search documents using regular expressions.

```json
{
  "tool": "search_document_regex",
  "args": {
    "doc_name": "all",
    "pattern": "Case No\\.\\s+\\d+",
    "flags": ["IGNORECASE"],
    "top_k": 5,
    "context_tokens": 200
  }
}
```

**Document Selection:**
- `doc_name: "all"` - Search all documents
- `doc_name: "<name>"` - Search single document
- `doc_names: ["doc1", "doc2"]` - Search specific documents

### get_checklist

Retrieve current extraction state.

```json
{"tool": "get_checklist", "args": {"item": "all"}}
{"tool": "get_checklist", "args": {"item": "Filing_Date"}}
{"tool": "get_checklist", "args": {"items": ["Filing_Date", "Who_are_the_Parties"]}}
```

### update_checklist

Replace entire extracted list for checklist keys. Use for complete/authoritative sets or corrections.

```json
{
  "tool": "update_checklist",
  "args": {
    "patch": [{
      "key": "Filing_Date",
      "extracted": [{
        "value": "2023-02-28",
        "evidence": [{
          "text": "Filed February 28, 2023",
          "source_document": "complaint.txt",
          "location": "Page 1"
        }]
      }]
    }]
  }
}
```

### append_checklist

Add new entries to existing extracted lists. Use for incremental discovery.

```json
{
  "tool": "append_checklist",
  "args": {
    "patch": [{
      "key": "Court_Rulings",
      "extracted": [{
        "value": "Motion to dismiss denied on June 15, 2023",
        "evidence": [{
          "text": "The Court DENIES defendant's motion to dismiss",
          "source_document": "order_005.txt",
          "location": "Page 3"
        }]
      }]
    }]
  }
}
```

**Difference:** `update_checklist` replaces; `append_checklist` adds. Both support batching multiple keys.

## Output Format

### Checklist Entry Structure

Each checklist item follows this evidence-based format:

```json
{
  "Filing_Date": {
    "extracted": [
      {
        "value": "2023-02-28",
        "evidence": [
          {
            "text": "Filed February 28, 2023",
            "source_document": "complaint.txt",
            "location": "Page 1, Header"
          }
        ]
      }
    ],
    "last_updated": "2025-01-12T10:30:00-05:00"
  }
}
```

### Not Applicable Encoding

Items that don't apply to a case are encoded as:

```json
{
  "Settlement_Terms": {
    "extracted": [{
      "value": "Not Applicable",
      "evidence": [{
        "text": "Case dismissed with prejudice before settlement",
        "source_document": "order_final.txt",
        "location": "Page 2"
      }]
    }]
  }
}
```

### Output Files

After execution, results are saved to:
- `checklist.json` - Final checklist state with all extracted values
- `ledger.jsonl` - Action audit trail (all tool calls)
- `run_<id>.json` - Run summary with token statistics
- `raw_responses.jsonl` - Raw LLM outputs for debugging

## Configuration

### Model Configuration

Model-specific sampling parameters in `config/model_config.yaml`:

```yaml
models:
  qwen3:
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    max_tokens: 32000

  gpt-oss:
    temperature: 0.7
    top_p: 1.0
    top_k: -1
    max_tokens: 32000
```

### Checklist Configurations

Three levels of granularity:

**All Items:** `config/checklist_configs/all/all_26_items.yaml`
- Extracts all 26 checklist items in one run

**Grouped (9 configs):** `config/checklist_configs/grouped/`
| Config | Items |
|--------|-------|
| `01_basic_case_info.yaml` | Filing_Date, Who_are_the_Parties, Class_Action_or_Individual_Plaintiffs, Type_of_Counsel |
| `02_legal_foundation.yaml` | Cause_of_Action, Statutory_or_Constitutional_Basis_for_the_Case, Remedy_Sought |
| `03_judge_info.yaml` | First_and_Last_name_of_Judge |
| `04_related_cases.yaml` | Consolidated_Cases_Noted, Related_Cases_Listed_by_Their_Case_Code_Number |
| `05_filings_proceedings.yaml` | Note_Important_Filings, Court_Rulings, All_Reported_Opinions_Cited, Trials, Appeal |
| `06_decrees.yaml` | Significant_Terms_of_Decrees, Dates_of_All_Decrees, How_Long_Decrees_will_Last |
| `07_settlements.yaml` | Settlement terms, dates, duration, enforcement, disputes |
| `08_monitoring.yaml` | Name_of_the_Monitor, Monitor_Reports |
| `09_context.yaml` | Factual_Basis_of_Case |

**Individual (26 configs):** `config/checklist_configs/individual/`
- One config per checklist item for targeted extraction

### Using Custom Configurations

```bash
# Use grouped config
python run_agent.py data/case --checklist-config config/checklist_configs/grouped/01_basic_case_info.yaml

# Use individual config
python run_agent.py data/case --checklist-config config/checklist_configs/individual/08_judge_name.yaml
```

See [checklist_configs/README.md](config/checklist_configs/README.md) for full details.

## Model Support

### Qwen3 Models

Uses native tool calling format. Prompt file: `config/prompts_qwen.yaml`

```bash
python run_agent.py data/case --model Qwen/Qwen3-8B
python run_agent.py data/case --model Qwen/Qwen3-14B
python run_agent.py data/case --model Qwen/Qwen3-32B
```

### GPT-OSS Models

Uses Harmony format with TypeScript-style tool definitions. Prompt file: `config/prompts_gpt_oss.yaml`

```bash
python run_agent.py data/case --model unsloth/gpt-oss-20b-BF16
```

### Model Detection

The orchestrator auto-detects model type from the model name:
- Contains "qwen3" (case-insensitive) → Qwen3 format
- Contains "gpt-oss" or "gptoss" → Harmony format
- Other models → Default format

## CLI Reference

```bash
python run_agent.py <corpus_path> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `corpus_path` | Path to document corpus (directory with legal documents) |

### Optional Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-8B` | Model for orchestration |
| `--max-steps` | `100` | Maximum steps before stopping |
| `--store-path` | `checklist_store.json` | Checklist persistence path |
| `--ledger-path` | `ledger.jsonl` | Ledger persistence path |
| `--config-dir` | `config` | Configuration directory |
| `--checklist-config` | `all_26_items.yaml` | Checklist config file |
| `--resume` | `false` | Resume from existing state |
| `--batch` | `false` | Batch mode for multiple cases |
| `--case-ids` | - | Specific case IDs for batch |
| `--output-dir` | `output` | Output directory for batch |
| `--quiet` | `false` | Reduce output verbosity |
| `--debug` | `false` | Show full prompts/responses |
| `--recent-actions` | `5` | Recent actions shown in snapshot |

### Examples

```bash
# Basic extraction
python run_agent.py data/case_46210

# Debug mode with custom model
python run_agent.py data/case_46210 --model Qwen/Qwen3-14B --debug --max-steps 50

# Extract only basic case info
python run_agent.py data/case --checklist-config config/checklist_configs/grouped/01_basic_case_info.yaml

# Batch processing specific cases
python run_agent.py data/cases --batch --case-ids 46210 46094 --output-dir results

# Resume an interrupted run
python run_agent.py data/case_46210 --resume
```

## Debugging and Performance

### Verbosity Levels

| Mode | Flag | Output |
|------|------|--------|
| Normal | (default) | Progress summaries, step counts |
| Debug | `--debug` | Full prompts, LLM responses, action details |
| Quiet | `--quiet` | Minimal output |

### Log Files

- `agent_logs/<model>/<case_id>/` - Action logs per run
- `output/<model>/<case_id>/raw_responses.jsonl` - Raw LLM outputs

### Token Usage Tracking

The stats_tracker reports:
- Total prompt tokens
- Total completion tokens
- System prompt tokens (cached after first call)
- Per-step token usage

### Common Issues

**Parse Errors:**
The orchestrator retries up to 2 times on JSON parse failures. Check `raw_responses.jsonl` for malformed outputs.

**Infinite Loops:**
The two-stage stop mechanism and 3-attempt safety limit prevent most loops. If stuck:
- Check `--debug` output for repetitive actions
- Verify checklist config has extractable items
- Ensure documents contain relevant content

**Document Not Found:**
The tools support fuzzy matching (e.g., "Answer..." matches "Answer, Affirmative Defenses, and Counterclaims"). Check exact names in `list_documents` output.

**Token Budget:**
For large documents (>100K tokens), the agent reads in chunks. Coverage tracking in the snapshot prevents re-reading.

## Related Documentation

- [DATA_PROCESSING.md](DATA_PROCESSING.md) - Document preparation and corpus setup
- [config/checklist_configs/README.md](config/checklist_configs/README.md) - Checklist configuration details
- [config/prompts_qwen.yaml](config/prompts_qwen.yaml) - Qwen3 prompt configuration
- [config/prompts_gpt_oss.yaml](config/prompts_gpt_oss.yaml) - GPT-OSS Harmony format prompts
