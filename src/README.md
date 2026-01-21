# Source Code for LLM Inference

This directory contains scripts for running LLM inference through closed-source cloud APIs and open-source models on local GPU clusters.

## Overview

### Closed-Source API Inference (Batch Processing)

For processing large numbers of prompts through proprietary cloud APIs (Claude, GPT, Gemini) with automatic batching, rate limiting, and cost optimization:

| File | Purpose |
|------|---------|
| `create_batch_claude.ipynb` | Create batch jobs for Claude API |
| `create_batch_gemini.ipynb` | Create batch jobs for Gemini API |
| `create_batch_gpt.ipynb` | Create batch jobs for OpenAI/Azure API |
| `retrieve_batch_results_claude.ipynb` | Retrieve Claude batch results |
| `retrieve_batch_results_gemini.ipynb` | Retrieve Gemini batch results |
| `retrieve_batch_results_gpt.ipynb` | Retrieve OpenAI batch results |

### Open-Source Model Inference (vLLM)

For running open-source models (Qwen, Gemma, GPT-oss) on local GPU clusters with SLURM:

| File | Purpose |
|------|---------|
| `submit_vllm_inference_jobs.sh` | Orchestrate SLURM job submission |
| `vllm_inference.sbatch` | SLURM batch configuration |
| `vllm_inference.py` | Core inference engine with YaRN |

### Checklist Extraction from Documents

For extracting 26 structured checklist items from legal document corpora:

| Approach | Location | Description |
|----------|----------|-------------|
| Chunk-by-Chunk | `extract_checklist_from_documents/chunk_by_chunk_iterative_updating/` | Simple batch processing through document chunks |
| Gavel Agent | `extract_checklist_from_documents/gavel_agent/` | Multi-tool agentic orchestration |

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for cloud providers (see `.env.example`)
- For vLLM: CUDA-capable GPUs and SLURM cluster

### Installation

```bash
pip install anthropic openai google-genai  # For batch API
pip install vllm torch transformers ray    # For vLLM inference
```

### Environment Setup

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export directly:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

## Data Format

Both closed-source API and open-source vLLM inference use the same input/output JSON format.

For prompt templates, see the [prompts/](../prompts/) folder and [prompts/README.md](../prompts/README.md).

### Input Format

Input JSON with `keys` and `contexts` arrays (must have matching lengths):

**For summary generation** - keys are case IDs (strings):
```json
{
  "keys": [
    "46567",
    "46582",
    "46496",
    "46432"
  ],
  "contexts": [
    [{"role": "user", "content": "Generate summary for case 46567..."}],
    [{"role": "user", "content": "Generate summary for case 46582..."}],
    [{"role": "user", "content": "Generate summary for case 46496..."}],
    [{"role": "user", "content": "Generate summary for case 46432..."}]
  ]
}
```

**For checklist extraction/evaluation** - keys are `[case_id, checklist_item]` pairs:
```json
{
  "keys": [
    ["46210", "Filing Date"],
    ["46210", "Cause of Action"],
    ["46210", "Who are the Parties"],
    ["46094", "Filing Date"]
  ],
  "contexts": [
    [{"role": "user", "content": "Extract Filing Date from case 46210..."}],
    [{"role": "user", "content": "Extract Cause of Action from case 46210..."}],
    [{"role": "user", "content": "Extract Parties from case 46210..."}],
    [{"role": "user", "content": "Extract Filing Date from case 46094..."}]
  ]
}
```

### Output Format

Results are keyed the same way as the input:

**For summary generation** (string keys):
```json
{
  "meta_data": {
    "file_name": "input_file",
    "inference_model": "claude-sonnet-4-20250514"
  },
  "token_stats": {
    "total_input_tokens": 50000,
    "total_output_tokens": 25000,
    "num_prompts": 100,
    "avg_input_tokens": 500.0,
    "avg_output_tokens": 250.0
  },
  "results": {
    "46567": "Summary text for case 46567...",
    "46582": "Summary text for case 46582...",
    "46496": "Summary text for case 46496..."
  }
}
```

**For checklist extraction/evaluation** (nested dict: `results[case_id][checklist_item]`):
```json
{
  "meta_data": {
    "file_name": "input_file",
    "inference_model": "claude-sonnet-4-20250514"
  },
  "token_stats": {
    "total_input_tokens": 80000,
    "total_output_tokens": 40000,
    "num_prompts": 200,
    "avg_input_tokens": 400.0,
    "avg_output_tokens": 200.0
  },
  "results": {
    "46210": {
      "Filing Date": "{\"extracted\": [{\"value\": \"2023-02-28\", ...}]}",
      "Cause of Action": "{\"extracted\": [{\"value\": \"Clean Air Act...\", ...}]}"
    },
    "46094": {
      "Filing Date": "{\"extracted\": [{\"value\": \"2024-01-15\", ...}]}"
    }
  }
}
```

---

## Closed-Source API Workflow

### 1. Create Batch Job

1. Open the appropriate notebook for your provider
2. Edit the configuration cell at the top:
   - `BASE_DIR`: Base directory for data files
   - `INPUT_DIR`: Directory containing input JSON
   - `INPUT_FILE`: Input file name (without `.json`)
   - `MODEL_NAME`: Model to use

3. Run all cells to submit the batch job

### 2. Retrieve Results

1. Open the corresponding retrieve notebook
2. Use the same configuration as the create notebook
3. Run all cells to download and parse results

---

## Open-Source Model Workflow (vLLM)

### 1. Configure Jobs

Edit `submit_vllm_inference_jobs.sh`:

```bash
# Input file names (without .json extension)
declare -a FILES=(
  "my_input_data"
)

# Folder paths containing input JSON files
declare -a FOLDERS=(
  "data/prompts"
)

# HuggingFace model identifiers
declare -a MODELS=(
  "Qwen/Qwen3-14B"
)
```

### 2. Submit Jobs

```bash
# Preview jobs without submitting
./submit_vllm_inference_jobs.sh --dry-run

# Submit jobs
./submit_vllm_inference_jobs.sh
```

### 3. Monitor Progress

```bash
squeue -u $USER
tail -f vllm_inference_logs/*.out
```

## Checklist Extraction from Documents

Extract 26 structured checklist items from legal case document corpora. Two approaches are available depending on your use case.

### Approaches

#### Chunk-by-Chunk Iterative Updating

Simple batch processing that iterates through document chunks sequentially, extracting all 26 checklist items per chunk and merging results.

**Best for:**
- Straightforward extraction tasks
- When document structure is predictable
- Batch processing multiple cases in parallel
- Lower computational overhead

**Key files:**
- `vllm_inference.py` - Main processing pipeline with YaRN context scaling
- `submit_vllm_inference_jobs.sh` - SLURM job submission

#### Gavel Agent (Agentic Approach)

Multi-tool orchestration loop where an LLM agent decides which documents to read, search, and when to update the checklist. Supports targeted extraction of specific items or groups.

**Best for:**
- Complex documents requiring intelligent navigation
- When specific items need targeted extraction
- Documents where relevant information is sparse or scattered
- Research and experimentation with extraction strategies

**Key components:**
- Orchestrator with tool-calling interface (read, search, update)
- Configurable checklist items (all 26, 9 grouped, or individual)
- Evidence-backed extraction with source citations
- Two-stage stopping mechanism for quality assurance

**Documentation:**
- [Agent Architecture](extract_checklist_from_documents/gavel_agent/README.md)
- [Data Processing Guide](extract_checklist_from_documents/gavel_agent/DATA_PROCESSING.md)
- [Checklist Configurations](extract_checklist_from_documents/gavel_agent/config/checklist_configs/README.md)

## Supported Models

### Closed-Source Models (Cloud APIs)

| Provider | Models | Thinking Support |
|----------|--------|------------------|
| Claude | claude-sonnet-4, claude-opus-4 | Extended thinking mode |
| Gemini | gemini-2.5-pro, gemini-2.5-flash, gemini-3 | thinkingBudget (2.5) / thinkingLevel (3) |
| OpenAI | gpt-4.1, o3, gpt-5 | reasoning_effort (low/medium/high) |

### Open-Source Models (vLLM)

| Model Family | Thinking Support | Notes |
|--------------|------------------|-------|
| Qwen3 | `<think>` tags | 8B, 14B, 32B, 2507 variants |
| GPT-OSS | Channel-based | 20B-BF16, 120B |
| Gemma | No | 4B, 12B, 27B |

## Features

### Closed-Source API Features

- **Automatic batching**: Handles rate limits and batching automatically
- **Cost optimization**: 50% discount on batch API pricing
- **Resume capability**: Merge new results with existing ones
- **Token tracking**: Detailed usage statistics
- **Error handling**: Reports failed prompts for resubmission

### Open-Source Model Features (vLLM)

- **YaRN context extension**: Handle prompts up to 262K tokens
- **Automatic bucketing**: Optimal memory usage by prompt length
- **Thinking parsers**: Extract reasoning from Qwen3 and GPT-OSS outputs
- **Multi-GPU support**: Automatic tensor parallelism
- **Resume capability**: Skip already-processed prompts

## Directory Structure

```
src/
├── README.md                          # This file
├── .env.example                       # Environment variable template
├── create_batch_claude.ipynb          # Claude batch creation
├── create_batch_gemini.ipynb          # Gemini batch creation
├── create_batch_gpt.ipynb             # OpenAI batch creation
├── retrieve_batch_results_claude.ipynb # Claude result retrieval
├── retrieve_batch_results_gemini.ipynb # Gemini result retrieval
├── retrieve_batch_results_gpt.ipynb   # OpenAI result retrieval
├── submit_vllm_inference_jobs.sh      # SLURM job submission
├── vllm_inference.sbatch              # SLURM configuration
├── vllm_inference.py                  # vLLM inference engine
└── extract_checklist_from_documents/  # Checklist extraction pipelines
    ├── chunk_by_chunk_iterative_updating/
    │   ├── vllm_inference.py          # Chunk-by-chunk extraction
    │   ├── submit_vllm_inference_jobs.sh
    │   └── data/                      # Input data files
    └── gavel_agent/                   # Agentic extraction approach
        ├── README.md                  # Agent architecture docs
        ├── DATA_PROCESSING.md         # Data preparation guide
        ├── run_agent.py               # CLI entry point
        ├── agent/                     # Core agent modules
        │   ├── driver.py              # Main execution loop
        │   ├── orchestrator.py        # LLM orchestration
        │   └── tools/                 # Agent tools (6 tools)
        └── config/                    # Configuration files
            └── checklist_configs/     # Modular checklist definitions
```

## Troubleshooting

### Common Issues

**API rate limits**
- Batch APIs handle rate limits automatically
- Results are typically ready within 24 hours

**GPU out of memory**
- Reduce the number of GPUs in `vllm_inference.sbatch`
- Use a smaller model variant
- The bucketing system handles most cases automatically

**Missing results**
- Check the log files for errors
- Resubmit failed prompts using a filtered input file
- Results merge automatically with existing outputs

**Thinking mode not working**
- Ensure model supports thinking (Qwen3, GPT-OSS, Claude, Gemini, o3/GPT-5)
- Check that `enable_thinking` or equivalent is set correctly
- For GPT-OSS, verify `skip_special_tokens=False` is set

### Getting Help

For issues specific to this codebase, check the log files first:
- Closed-source APIs: Check the notebook output cells
- Open-source models (vLLM): Check `vllm_inference_logs/` directory
