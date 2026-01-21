# GAVEL Data

This folder contains the data for the GAVEL (Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization) project.

## Overview

- **100 cases** are used for evaluating LLM-generated summaries (50 cases in two parts: `50_cases_for_benchmarking.json` and `50_cases_for_benchmarking_2.json`)
- **40 cases** are used for meta-evaluating document-extracted checklists (20 cases in two parts: `20_human_eval_cases.json` and `20_human_eval_cases_2.json`)
- Evaluation uses **26 checklist items** commonly found in legal case summaries
- Case lengths range from **32K to 512K tokens**, with 83% of cases from 2025 to reduce data contamination

**Note:** Some JSON files may be empty due to server errors during inference. This is a known issue documented in the project blog.

---

## Folder Structure

### 1. `full_case_data/`

**Purpose:** Contains full case data including case documents, metadata, and human-written summaries. This is the source data used for summary generation and document-level checklist extraction.

#### Files
- `50_cases_for_benchmarking.json` - 50 cases for benchmarking (part 1)
- `50_cases_for_benchmarking_2.json` - 50 cases for benchmarking (part 2)
- `20_human_eval_cases.json` - 20 cases for meta-evaluation (part 1)
- `20_human_eval_cases_2.json` - 20 cases for meta-evaluation (part 2)

#### JSON Schema
```json
[
  {
    "case_id": "46210",
    "case_type": "Environmental Justice",
    "filing_date": "2023-02-28",
    "case_url": "https://clearinghouse.net/case/46210",
    "summary/long": "Full narrative summary of the legal case...",
    "total_token_num": 620071,
    "total_length_bin": "240K+",
    "case_documents": ["doc1 text...", "doc2 text...", ...],
    "case_documents_text": ["doc1 text...", "doc2 text...", ...],
    "case_documents_title": ["Complaint", "Motion to Dismiss", ...],
    "case_documents_doc_type": ["Complaint", "Motion", ...],
    "case_documents_date": ["2023-02-28", "2023-04-15", ...],
    "case_documents_id": [157685, 157686, ...],
    "case_documents_token_num": [9952, 5432, ...]
  }
]
```

Each case object contains:
- `case_id`: Unique case identifier
- `case_type`: Category of the case (e.g., Environmental Justice, Immigration)
- `filing_date`: Date the case was filed
- `case_url`: URL to the case on clearinghouse.net
- `summary/long`: Human-written reference summary
- `total_token_num`: Total tokens across all documents
- `total_length_bin`: Token count bin (32K, 64K, 120K, 240K+)
- `case_documents` / `case_documents_text`: List of document text content
- `case_documents_title`: List of document titles
- `case_documents_doc_type`: List of document types
- `case_documents_date`: List of document dates
- `case_documents_id`: List of document IDs
- `case_documents_token_num`: List of token counts per document

---

### 2. `summaries/`

**Purpose:** Contains human-written reference summaries and model-generated summaries for legal cases.

#### Top-Level Files (Human Reference Summaries)
- `50_cases_for_benchmarking.json` - 50 human-written case summaries (part 1)
- `50_cases_for_benchmarking_2.json` - 50 human-written case summaries (part 2)
- `20_human_eval_cases.json` - 20 cases for meta-evaluation (part 1)
- `20_human_eval_cases_2.json` - 20 cases for meta-evaluation (part 2)

#### Model Subfolders
Each subfolder contains model-generated summaries:
- `claude-sonnet-4-20250514-thinking/`
- `claude-opus-4-1-20250805-thinking/`
- `gpt-4.1-2025-04-14/`
- `gpt-5-2025-08-07/`
- `gpt-oss-20b-BF16/`
- `gemini-2.5-flash/`
- `gemini-2.5-pro/`
- `gemma-3-12b-it/`
- `gemma-3-27b-it/`
- `Qwen3-14B/`
- `Qwen3-32B/`
- `Qwen3-30B-A3B-Thinking-2507/`

#### JSON Schema
```json
{
  "<case_id>": {
    "summary/long": "Full narrative summary of the legal case...",
    "filing_date": "YYYY-MM-DD",
    "case_type": "Immigration and/or the Border | Environmental Justice | etc.",
    "total_token_num": 28031,
    "total_length_bin": "32K | 64K | 120K | 240K+"
  }
}
```

---

### 3. `summary_checklists/`

**Purpose:** Checklists extracted from summaries (both human and model-generated) using **GPT-oss 20B** as the extraction model.

#### Top-Level Files (Human Reference Checklists)
- `50_cases_for_benchmarking_thinking_True.json` - Checklists from human summaries (part 1)
- `50_cases_for_benchmarking_2_thinking_True.json` - Checklists from human summaries (part 2)

#### Model Subfolders
Each subfolder contains checklists extracted from the corresponding model's summaries:
- `claude-sonnet-4-20250514-thinking_summarization/`
- `claude-opus-4-1-20250805-thinking_summarization/`
- `gpt-4.1-2025-04-14_summarization/`
- `gpt-5-2025-08-07_summarization/`
- `gpt-oss-20b-BF16_summarization/`
- `gemini-2.5-flash_summarization/`
- `gemini-2.5-pro_summarization/`
- `gemma-3-12b-it_summarization/`
- `gemma-3-27b-it_summarization/`
- `Qwen3-14B_summarization/`
- `Qwen3-32B_summarization/`
- `Qwen3-30B-A3B-Thinking-2507_summarization/`

#### JSON Schema
```json
{
  "results": {
    "<case_id>": {
      "Filing Date": [
        {
          "evidences_indices": [[618, 750]],
          "evidences": ["On February 28, 2023, the United States government filed a lawsuit..."],
          "value": "February 28, 2023"
        }
      ],
      "Cause of Action": [
        {
          "evidences_indices": [[1001, 1322]],
          "evidences": ["Plaintiff sued Defendant Denka under Section 303 of the Clean Air Act..."],
          "value": "Clean Air Act Section 303 (42 U.S.C. Section 7603)..."
        }
      ]
    }
  }
}
```

Each checklist item contains:
- `evidences_indices`: Character positions in the source summary
- `evidences`: Extracted text snippets supporting the value
- `value`: The extracted checklist item value

---

### 4. `evaluation/`

**Purpose:** Checklist evaluation results comparing model-extracted checklists vs. human-extracted checklists. Evaluation is performed using **Gemma3 27B** as the evaluator model.

#### Model Subfolders
Each subfolder contains evaluation results for the corresponding model:
- `claude-sonnet-4-20250514-thinking_summarization/`
- `claude-opus-4-1-20250805-thinking_summarization/`
- `gpt-4.1-2025-04-14_summarization/`
- `gpt-5-2025-08-07_summarization/`
- `gpt-oss-20b-BF16_summarization/`
- `gemini-2.5-flash_summarization/`
- `gemini-2.5-pro_summarization/`
- `gemma-3-12b-it_summarization/`
- `gemma-3-27b-it_summarization/`
- `Qwen3-14B_summarization/`
- `Qwen3-32B_summarization/`
- `Qwen3-30B-A3B-Thinking-2507_summarization/`

#### JSON Schema
```json
{
  "meta_data": {
    "file_name": "model_name/...",
    "inference_model": "google/gemma-3-27b-it",
    "enable_thinking": false
  },
  "token_stats": {
    "total_input_tokens": 369792,
    "total_output_tokens": 234163,
    "num_prompts": 568,
    "avg_input_tokens": 651.04,
    "avg_output_tokens": 412.26
  },
  "results": {
    "<case_id>": {
      "Cause of Action": {
        "common": [{"A_index": 2, "B_index": 1}],
        "only_in_A": [1],
        "only_in_B": []
      },
      "First and Last name of Judge": "A equals B"
    }
  }
}
```

Evaluation results contain:
- `common`: Matched items between model (A) and human reference (B) with their indices
- `only_in_A`: Items found only in model output
- `only_in_B`: Items found only in human reference
- String-wise comparison returns `"A equals B"` when values match exactly

---

### 5. `document_checklists/`

**Purpose:** Checklists extracted directly from case documents using three different methods. This enables evaluation without relying on human-written summaries.

#### Special File
- `human.json` - Human-extracted checklists **from summaries** (not from documents). This serves as the reference for evaluating document extraction methods.

#### Subfolders

##### `end_to_end/`
End-to-end extraction feeding all documents to a long-context LLM (GPT-4.1).
- Files: `20_human_eval_cases_order_by_date_True_batch_*.json`, `20_human_eval_cases_2_order_by_date_True_batch_*.json`

##### `chunk_by_chunk/`
Iterative chunk processing (16K-token chunks). Models: Qwen3-32B, Qwen3-30B-A3B-Thinking-2507, GPT-oss 20B.
- Subfolders: `Qwen3-32B/`, `Qwen3-30B-A3B-Thinking-2507/`, `gpt-oss-20b-BF16/`

##### `gavel_agent/`
Autonomous agent approach with six specialized tools. Models: Qwen3-30B-A3B-Thinking-2507, GPT-oss 20B.
- Structure: `<model>/<case_id>/<extraction_mode>/`
- Extraction modes:
  - `all/all_26_items/` - All 26 items extracted together
  - `individual/<item_name>/` - Each item extracted separately
  - `grouped/<group_name>/` - Items grouped by category

Each extraction contains:
- `checklist.json` - Extracted checklist values
- `stats.json` - Token usage statistics
- `ledger.jsonl` - Processing log
- `raw_responses.jsonl` - Model outputs (may be empty due to server errors)

#### JSON Schema (human.json)
```json
{
  "metadata": {
    "name": "total_new_definitions",
    "description": "Reviewed annotations for all 40 cases from thresh_format/review"
  },
  "results": {
    "<case_id>": {
      "Filing Date": [
        {
          "evidences_indices": [[618, 750]],
          "evidences": ["On February 28, 2023, the United States government filed a lawsuit..."],
          "value": "February 28, 2023"
        }
      ]
    }
  }
}
```

---

### 6. `evaluation_documents_checklist/`

**Purpose:** Evaluation of document-extracted checklists vs. human-extracted checklists from summaries. Evaluation is performed using **Gemma3 27B**.

#### Subfolders
- `end_to_end/` - Evaluation for end-to-end extraction
- `chunk_by_chunk/` - Evaluation for chunk-by-chunk extraction
- `agent/` - Evaluation for GAVEL-Agent extraction

#### JSON Schema
```json
{
  "meta_data": {
    "file_name": "end_to_end/gpt-4.1-2025-04-14",
    "inference_model": "google/gemma-3-27b-it",
    "enable_thinking": false
  },
  "token_stats": {
    "total_input_tokens": 442744,
    "total_output_tokens": 235612,
    "num_prompts": 524
  },
  "results": {
    "<case_id>": {
      "Cause of Action": {
        "common": [{"model_index": 1, "reference_index": 1}],
        "only_in_model": [1],
        "only_in_reference": [2, 3, 4, 5]
      }
    }
  }
}
```

---

## 26 Checklist Items

The evaluation framework uses 26 key items commonly found in legal case summaries:

| # | Item Name | Description |
|---|-----------|-------------|
| 1 | Filing Date | Date when the case was filed |
| 2 | Who are the Parties | Plaintiffs and defendants in the case |
| 3 | Cause of Action | Legal basis for the lawsuit |
| 4 | Type of Counsel | Type of legal representation |
| 5 | Statutory or Constitutional Basis | Laws or constitutional provisions cited |
| 6 | Remedy Sought | Relief requested by plaintiffs |
| 7 | First and Last name of Judge | Presiding judge(s) |
| 8 | Consolidated Cases | Cases consolidated with this one |
| 9 | Related Cases | Related legal proceedings |
| 10 | Important Filings | Key documents filed in the case |
| 11 | Court Rulings | Decisions made by the court |
| 12 | Reported Opinions | Published judicial opinions |
| 13 | Trials | Trial proceedings |
| 14 | Appeals | Appellate proceedings |
| 15 | Decree Terms | Terms of any court orders/decrees |
| 16 | Dates of All Decrees | Dates of court orders/decrees |
| 17 | How Long Decrees will Last | Duration of court orders |
| 18 | Settlement Terms | Terms of any settlements |
| 19 | Settlement Date | Date of settlement |
| 20 | Settlement Duration | How long the settlement lasts |
| 21 | Court Enforced | Whether court enforcement occurred |
| 22 | Enforcement Disputes | Disputes over enforcement |
| 23 | Court-Appointed Monitor | Any appointed monitors |
| 24 | Monitor Reports | Reports from monitors |
| 25 | Factual Basis of Case | Key facts underlying the case |
| 26 | Context | Additional contextual information |

---

## Models Evaluated

### Proprietary Models
- **GPT-5** (`gpt-5-2025-08-07`)
- **GPT-4.1** (`gpt-4.1-2025-04-14`)
- **Gemini 2.5 Pro** (`gemini-2.5-pro`)
- **Gemini 2.5 Flash** (`gemini-2.5-flash`)
- **Claude Sonnet 4** (`claude-sonnet-4-20250514-thinking`)
- **Claude Opus 4.1** (`claude-opus-4-1-20250805-thinking`)

### Open-Source Models
- **GPT-oss 20B** (`gpt-oss-20b-BF16`)
- **Qwen3 32B** (`Qwen3-32B`)
- **Qwen3 14B** (`Qwen3-14B`)
- **Qwen3 30B-A3B Thinking** (`Qwen3-30B-A3B-Thinking-2507`)
- **Gemma 3 27B** (`gemma-3-27b-it`)
- **Gemma 3 12B** (`gemma-3-12b-it`)

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
