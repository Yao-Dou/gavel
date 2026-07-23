# Chunk-by-Chunk Hierarchical Merging Checklist Extraction

This pipeline extracts checklist items from case documents by processing them chunk-by-chunk and then **hierarchically merging** the per-chunk results — an alternative to the sibling [iterative-updating](../chunk_by_chunk_iterative_updating/README.md) method, which threads a single evolving state through the chunks sequentially.

Two domains are supported via `--domain`:
- **legal** (default): 26 checklist items from legal case documents
- **medical**: 29 checklist items from Cochrane systematic reviews

## Method

A three-stage pipeline, run as one job per data file with a model swap mid-job:

1. **EXTRACT** (default: `unsloth/gpt-oss-20b-BF16`)
   Every (checklist item × case chunk) prompt runs independently and extracts **high-recall** candidates from a single chunk. One vLLM batch per item. After this stage the extraction model is unloaded and the merge model is loaded.

2. **MERGE** (default: `Qwen/Qwen3-30B-A3B-Thinking-2507`)
   Per (item, case), a binary-tree merge of the chunk checklists in chunk order: pairs (1,2), (3,4), …; an odd tail carries up unchanged; repeat until one checklist remains. Empty chunk checklists are dropped up front, so only pairs where both sides are non-empty cost an LLM call. Each global merge level (across all items and cases) is one vLLM batch. Oversize prompts and parse failures fall back to a lossless in-code concatenation, so no extraction is ever silently lost.

3. **PRUNE** (same Qwen model)
   One strict verification/cleanup prompt per (item, case) on the final merged checklist, removing duplicates and unsupported entries. Skipped for empty checklists.

Final results follow the same entry schema as the iterative method (`{"reasoning", "extracted": [{"value", "evidence": [{"text", "source_document", "location"}]}]}`), with provenance tracked through the merge tree.

## Data Preparation (shared with the iterative method)

This pipeline reads the **same chunked data files** as the iterative-updating method, from `../chunk_by_chunk_iterative_updating/data/`:

- Legal: `20_human_eval_cases.json`, `20_human_eval_cases_2.json` (ship with the repo; regenerate with `../chunk_by_chunk_iterative_updating/create_data_for_chunk_by_chunk_pipeline.ipynb`)
- Medical: `medical_10_human_eval_cases.json` (generate with `../chunk_by_chunk_iterative_updating/medical_create_data_for_chunk_by_chunk_pipeline.ipynb`)

File naming: legal files are unprefixed; other domains carry a `{domain}_` prefix.

## Prompt Templates

Three templates per domain, loaded from `prompts/extract_checklist_item_from_docs/` (legal) and `prompts/extract_checklist_item_from_docs/medical/` (medical) at the repo root:

| Template | Stage | Placeholders |
|----------|-------|--------------|
| `high_recall_extraction_template.txt` | Extract | `item_description`, `document_name`, `chunk_id`, `total_chunks`, `document_chunk` |
| `merge_two_checklists_template.txt` | Merge | `item_name`, `item_description`, `checklist_a`, `checklist_a_provenance`, `checklist_b`, `checklist_b_provenance` |
| `prune_checklist_template.txt` | Prune | `item_name`, `item_description`, `checklist` |

Checklist item definitions come from `item_specific_info.json` in the same folders (legal: 26 items; medical: 29 items).

## Usage

### Direct run

```bash
# Legal (reads ../chunk_by_chunk_iterative_updating/data/20_human_eval_cases.json)
python vllm_inference.py \
    --domain legal \
    --file_name 20_human_eval_cases \
    --extract_model_name unsloth/gpt-oss-20b-BF16 \
    --merge_model_name Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --enable_thinking

# Medical (reads ../chunk_by_chunk_iterative_updating/data/medical_10_human_eval_cases.json)
python vllm_inference.py \
    --domain medical \
    --file_name 10_human_eval_cases \
    --enable_thinking
```

### Command line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--domain` | `legal` or `medical` (selects prompts/checklist + file naming) | Required |
| `--file_name` | Dataset name (without .json and without domain prefix) | Required |
| `--extract_model_name` | HF model for the high-recall extraction stage | `unsloth/gpt-oss-20b-BF16` |
| `--merge_model_name` | HF model for the merge and prune stages | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `--enable_thinking` | Thinking mode (GPT-OSS `reasoning_effort=high`; Qwen thinking sampling) | off |
| `--checklist_item` | Extract a specific item only | All items |

### SLURM batch submission

Edit `submit_vllm_inference_jobs.sh` (`DOMAIN`, `FILES`, `EXTRACT_MODELS`, `MERGE_MODEL`, `CHECKLIST_ITEMS`), adjust the generic `#SBATCH` headers in `vllm_inference.sbatch` for your cluster (optionally set `COMMON_ENV` to an env file to source), then:

```bash
./submit_vllm_inference_jobs.sh
```

## Outputs

- Results: `results/<extract_model>__<merge_model>/<file>_thinking_<bool>.json` — with `meta_data` (`"method": "chunk_by_chunk_hierarchical_merging"`, both model names), `token_stats` (per stage), and `results[case_id][item_name]`.
- Checkpoints: `states/<extract_model>__<merge_model>/…_{extract|merge|prune}_<item>.json`, written atomically (tmp + fsync + rename) and deleted only after the final results file lands. Rerunning the same command resumes from the last completed stage/level.
- Logs: `vllm_inference_logs/`.

## Files

| File | Description |
|------|-------------|
| `vllm_inference.py` | Main three-stage pipeline (extract → merge → prune) |
| `submit_vllm_inference_jobs.sh` | SLURM job submission orchestrator |
| `vllm_inference.sbatch` | Generic SLURM batch configuration (adjust for your cluster) |
| `utils.py` | Shared nested-dict merge helper |
