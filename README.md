# GAVEL: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yao-dou.github.io/gavel/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.04424-b31b1b)](https://www.arxiv.org/abs/2601.04424)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research framework for evaluating large language models on long-context legal document summarization using a structured 26-item checklist approach with evidence-based citations.

**Key Statistics:**
- 100 legal cases for benchmarking (32K to 512K tokens)
- 26 structured checklist items for evaluation
- 12 frontier models evaluated (6 proprietary, 6 open-source)
- 83% of cases from 2025 to minimize data contamination

---

## Features

- **GAVEL-Ref Evaluation**: Reference-free evaluation framework with three complementary metrics—Checklist, Residual Facts, and Writing Style
- **Multi-Value Extraction**: Each checklist item yields a list of (value, supporting_text) pairs, enabling partial credit for overlapping information
- **Multi-Model Support**: Evaluate GPT-5, Claude Opus 4.1, Gemini 2.5 Pro, Qwen3, and more
- **Document-Level Extraction**: Three approaches (end-to-end, chunk-by-chunk, agentic) to extract checklists directly from case documents
- **Flexible Infrastructure**: Cloud batch APIs (50% cost savings) or local vLLM inference with YaRN context extension

---

## Project Structure

```
gavel/
├── README.md                           # This file
├── data/                               # Datasets and evaluation results
│   ├── summaries/                      # Human and model-generated summaries
│   │   ├── 50_cases_for_benchmarking.json
│   │   ├── 50_cases_for_benchmarking_2.json
│   │   └── <model_name>/               # Model outputs (12 models)
│   ├── summary_checklists/             # Checklists extracted from summaries
│   ├── evaluation/                     # Summary checklist evaluation results
│   ├── document_checklists/            # Checklists extracted from documents
│   │   ├── human.json                  # Human reference
│   │   ├── end_to_end/                 # GPT-4.1 extraction
│   │   ├── chunk_by_chunk/             # Iterative extraction
│   │   └── gavel_agent/                # Agentic extraction
│   └── evaluation_documents_checklist/ # Document checklist evaluation
├── prompts/                            # Prompt templates
│   ├── extract_checklist_item/         # Extract from summaries
│   ├── extract_checklist_item_from_docs/ # Extract from documents
│   ├── evaluate_checklist/             # Evaluation prompts
│   ├── generate_summary.txt            # Summary generation
│   └── evaluate_writing_style.txt      # Style comparison
└── src/                                # Source code
    ├── create_batch_*.ipynb            # Batch API job creation (3 notebooks)
    ├── retrieve_batch_*.ipynb          # Result retrieval (3 notebooks)
    ├── vllm_inference.py               # Local GPU inference engine
    ├── submit_vllm_inference_jobs.sh   # SLURM job submission
    └── extract_checklist_from_documents/
        ├── chunk_by_chunk_iterative_updating/  # Iterative extraction
        └── gavel_agent/                        # Agentic extraction
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for cloud providers OR local GPU cluster with SLURM

### Installation

```bash
git clone https://github.com/Yao-Dou/gavel.git
cd gavel

# For cloud batch API processing
pip install anthropic openai google-genai

# For local vLLM inference
pip install vllm torch transformers ray tiktoken

# Configure environment
cp src/.env.example src/.env
# Edit src/.env with your API keys
```

### Running LLM Inference

The `src/` folder contains **general-purpose prompt runners** for batch inference. Users create prompts using templates from `prompts/` and submit them via:

**Cloud Batch APIs:**
```bash
# Notebooks for Claude, GPT, Gemini
jupyter notebook src/create_batch_claude.ipynb
jupyter notebook src/create_batch_gpt.ipynb
jupyter notebook src/create_batch_gemini.ipynb
```

**Local vLLM:**
```bash
./src/submit_vllm_inference_jobs.sh
```

See [src/README.md](src/README.md) for input/output formats and detailed usage.

---

## GAVEL-Ref Evaluation Framework

GAVEL-Ref is a **reference-free evaluation framework** that compares model-generated summaries against human-written reference summaries using three complementary metrics:

```
GAVEL-Ref Evaluation
├── Checklist Evaluation (S_checklist)
│   ├── Multi-value extraction with supporting text
│   └── Score aggregation over applicable items only
├── Residual Facts Evaluation (S_residual)
│   ├── Extract atomic facts from non-checklist text spans
│   └── List-wise F1 comparison
└── Writing Style Evaluation (S_style)
    ├── 5 dimensions: readability, narrative order, sentence structure, formatting, citation
    └── 1-5 Likert scale averaged and scaled to 0-100
```

### Checklist Evaluation (S_checklist)

Extracts 26 structured checklist items from both model and reference summaries, then compares them.

**Key improvements over prior work:**
1. **Multi-value extraction**: Each item yields a list of (value, supporting_text) pairs, enabling partial credit for overlapping lists (e.g., three matching remedies out of five)
2. **Applicable-only scoring**: Only items present in at least one summary count toward the score, avoiding inflation from non-applicable items

**Scoring:**
- **Single-value items**: 1 (equal), 0.5 (containment), 0 (different)
- **Multi-value items**: F1 measure over matched elements

### Residual Facts Evaluation (S_residual)

Evaluates information beyond the 26 checklist items:
1. Identify text spans not covered by checklist extraction (two-stage matching)
2. Extract atomic facts from these residual spans
3. Compare using list-wise F1 (scaled to 0-100)

### Writing Style Evaluation (S_style)

Measures stylistic **similarity** (not quality) across 5 dimensions:
- Readability & jargon level
- Narrative order
- Sentence structure & voice
- Formatting & layout
- Citation & reference style

Each rated 1-5 (identical to completely different), averaged, and scaled to 0-100.

---

## The 26 Checklist Items

Legal case summaries are evaluated against 26 structured items commonly found in legal case documentation:

| # | Category | Checklist Items |
|---|----------|-----------------|
| 1-4 | **Basic Case Info** | Filing Date, Who are the Parties, Class Action or Individual Plaintiffs, Type of Counsel |
| 5-7 | **Legal Foundation** | Cause of Action, Statutory or Constitutional Basis, Remedy Sought |
| 8 | **Judge Info** | First and Last Name of Judge |
| 9-10 | **Related Cases** | Consolidated Cases Noted, Related Cases Listed |
| 11-15 | **Proceedings** | Important Filings, Court Rulings, Reported Opinions Cited, Trials, Appeals |
| 16-18 | **Decrees** | Significant Terms of Decrees, Dates of All Decrees, How Long Decrees Last |
| 19-23 | **Settlements** | Settlement Terms, Date of Settlement, Settlement Duration, Court-Enforced or Not, Enforcement Disputes |
| 24-25 | **Monitoring** | Name of the Monitor, Monitor Reports |
| 26 | **Context** | Factual Basis of Case |

See [data/README.md](data/README.md) for complete item definitions.

---

## Supported Models

### Proprietary Models

| Model | ID |
|-------|-----|
| GPT-5 | `gpt-5-2025-08-07` |
| GPT-4.1 | `gpt-4.1-2025-04-14` |
| Claude Opus 4.1 | `claude-opus-4-1-20250805-thinking` |
| Claude Sonnet 4 | `claude-sonnet-4-20250514-thinking` |
| Gemini 2.5 Pro | `gemini-2.5-pro` |
| Gemini 2.5 Flash | `gemini-2.5-flash` |

### Open-Source Models

| Model | ID |
|-------|-----|
| GPT-oss 20B | `gpt-oss-20b-BF16` |
| Qwen3 32B | `Qwen3-32B` |
| Qwen3 14B | `Qwen3-14B` |
| Qwen3 30B Thinking | `Qwen3-30B-A3B-Thinking-2507` |
| Gemma 3 27B | `gemma-3-27b-it` |
| Gemma 3 12B | `gemma-3-12b-it` |

---

## Document-Level Checklist Extraction

Three approaches for extracting 26 checklist items directly from case documents (bypassing the summary stage):

| Approach | Description | Location |
|----------|-------------|----------|
| **End-to-End** | Concat all documents, extract each item one by one | `prompts/extract_checklist_item_from_docs/end_to_end_template.txt` |
| **Chunk-by-Chunk** | Process 16K-token chunks iteratively | [chunk_by_chunk/README.md](src/extract_checklist_from_documents/chunk_by_chunk_iterative_updating/README.md) |
| **GAVEL-Agent** | Multi-tool agentic orchestration | [gavel_agent/README.md](src/extract_checklist_from_documents/gavel_agent/README.md) |

### End-to-End Extraction
Concatenate all case documents and feed to a long-context LLM. Each of the 26 items is extracted one by one using prompts. **No dedicated folder**—users run prompts through the general inference infrastructure.

### Chunk-by-Chunk Iterative Updating
Process documents in 16K-token chunks, iteratively building up extraction state. The vLLM pipeline processes all cases and items in parallel across chunks.

### GAVEL-Agent (Agentic Approach)
Multi-tool orchestration where an LLM agent autonomously decides which documents to read, what to search for, and when to update the checklist. Features a two-stage stopping mechanism and supports configurable item subsets (all 26, 9 grouped, or individual items).

---

## Prompt Templates

The `prompts/` folder contains templates for all pipeline stages:

| Template | Purpose |
|----------|---------|
| `generate_summary.txt` | Generate summaries with 26-item checklist guidance |
| `extract_checklist_item/` | Extract checklist items from summaries |
| `extract_checklist_item_from_docs/` | Extract checklist items from documents (end-to-end + chunk-by-chunk) |
| `evaluate_checklist/` | Compare extracted checklists (string-wise + list-wise) |
| `extract_facts_from_residual_spans.txt` | Extract atomic facts from non-checklist text |
| `evaluate_writing_style.txt` | Compare writing style across 5 dimensions |

See [prompts/README.md](prompts/README.md) for detailed template documentation.

---

## Documentation

| Component | Documentation |
|-----------|---------------|
| Data Structure | [data/README.md](data/README.md) |
| Prompt Templates | [prompts/README.md](prompts/README.md) |
| Source Code | [src/README.md](src/README.md) |
| GAVEL-Agent | [gavel_agent/README.md](src/extract_checklist_from_documents/gavel_agent/README.md) |

---

## Citation

```bibtex
@article{dou2026gavel,
  title={Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization},
  author={Dou, Yao and Xu, Wei},
  journal={arXiv preprint arXiv:2601.04424},
  year={2026}
}
```
