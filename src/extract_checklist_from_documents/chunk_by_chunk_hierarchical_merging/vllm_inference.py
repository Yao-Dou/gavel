#!/usr/bin/env python3
"""Chunk-by-chunk HIERARCHICAL MERGING extraction utility for long documents
(legal cases or medical systematic reviews, selected via --domain).

Three-stage pipeline (one SLURM job per data file, model swap mid-job):

1. EXTRACT (default: unsloth/gpt-oss-20b-BF16)
   Every (checklist item x case chunk) prompt runs independently and extracts
   HIGH-RECALL candidates from a single chunk. One vLLM batch per item.

2. MERGE (default: Qwen/Qwen3-30B-A3B-Thinking-2507)
   Per (item, case), binary-tree merge of the chunk checklists in chunk order:
   pairs (1,2),(3,4),...; an odd tail carries up; repeat until one remains.
   Trivial merges are skipped in code: empty chunk checklists are dropped after
   extraction, and only pairs where BOTH sides are non-empty get an LLM prompt.
   Each global merge level across all items+cases is one vLLM batch.

3. PRUNE (same Qwen model)
   One strict verification/cleanup prompt per (item, case) on the final merged
   checklist. Skipped when the final checklist is empty.

Checkpoint/resume design (all writes atomic: tmp + fsync + os.replace):
- extract_{item}.json  - written ONCE per item, only after the item's full
  extraction batch is applied. Absence => redo the item. Validated structurally
  against the data file (case-key set + per-case chunk counts).
- merge_{item}.json    - written after every merge level the item participated
  in. Resume is CONTENT-based: the checkpoint stores the current surviving
  checklist lists; the merge loop just continues pairing until every list has
  length <= 1 ("level" is informational only).
- prune_{item}.json    - written per item after the prune batch completes.
Checkpoints are deleted only AFTER the final results file is safely on disk.

The domain selects the prompt templates and checklist definitions (see
DOMAIN_CONFIGS) and the data-file prefix: --domain legal --file_name X loads
../chunk_by_chunk_iterative_updating_release/data/legal_X.json, and all
outputs (results, states, logs) carry the same domain-prefixed name.

Command-line flags:
    --domain              (str)  required; one of DOMAIN_CONFIGS (legal, medical)
    --file_name           (str)  name of the JSON data file, without extension
                                 and without the domain prefix; loaded from
                                 ../chunk_by_chunk_iterative_updating_release/data/
    --enable_thinking     (flag) thinking mode (GPT-OSS reasoning_effort=high; Qwen thinking)
    --extract_model_name  (str)  HF model for stage 1 (default: unsloth/gpt-oss-20b-BF16)
    --merge_model_name    (str)  HF model for stages 2+3 (default: Qwen/Qwen3-30B-A3B-Thinking-2507)
    --checklist_item      (str)  specific checklist item to process (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc

from utils import merge_nested_dicts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# All static paths are resolved from this script's location, so runs are
# CWD-independent. Chunked data files are shared with the sibling
# chunk_by_chunk_iterative_updating folder (prepared by its notebooks); legal
# files keep their historical unprefixed names (e.g. 20_human_eval_cases.json),
# other domains use a {domain}_ prefix (e.g. medical_10_human_eval_cases.json).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "chunk_by_chunk_iterative_updating" / "data"
PROMPTS_DIR = SCRIPT_DIR.parents[2] / "prompts" / "extract_checklist_item_from_docs"

# Each domain supplies its prompt template directory (holding the three
# hierarchical templates) and checklist definitions. Adding a new domain only
# requires a new entry here plus the template/checklist files.
DOMAIN_CONFIGS: Dict[str, Dict[str, Path]] = {
    "legal": {
        "template_dir": PROMPTS_DIR,
        "checklist_path": PROMPTS_DIR / "item_specific_info.json",
    },
    "medical": {
        "template_dir": PROMPTS_DIR / "medical",
        "checklist_path": PROMPTS_DIR / "medical" / "item_specific_info.json",
    },
}

# Set by resolve_domain_paths() in main() once --domain is known.
TEMPLATE_DIR: Optional[Path] = None
CHECKLIST_PATH: Optional[Path] = None
EXTRACT_TEMPLATE: Optional[Path] = None
MERGE_TEMPLATE: Optional[Path] = None
PRUNE_TEMPLATE: Optional[Path] = None


def resolve_domain_paths(domain: str) -> None:
    """Resolve the module-level template/checklist paths for the domain."""
    global TEMPLATE_DIR, CHECKLIST_PATH, EXTRACT_TEMPLATE, MERGE_TEMPLATE, PRUNE_TEMPLATE
    cfg = DOMAIN_CONFIGS[domain]
    TEMPLATE_DIR = Path(cfg["template_dir"])
    CHECKLIST_PATH = Path(cfg["checklist_path"])
    EXTRACT_TEMPLATE = TEMPLATE_DIR / "high_recall_extraction_template.txt"
    MERGE_TEMPLATE = TEMPLATE_DIR / "merge_two_checklists_template.txt"
    PRUNE_TEMPLATE = TEMPLATE_DIR / "prune_checklist_template.txt"

# Qwen merge/prune generation budget. Thinking + a large merged JSON can be
# long; 48K leaves a ~213K prompt budget under the 262K context.
MERGE_MAX_TOKENS = 48_000
PRUNE_MAX_TOKENS = 48_000
QWEN_MAX_MODEL_LEN = 262_144
PROMPT_GUARD_MARGIN = 1_024


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk-by-chunk hierarchical merging extraction for long documents (domain-configurable).")
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DOMAIN_CONFIGS.keys()),
        help="Domain to run: selects the prompt templates, checklist definitions, and the data-file prefix",
    )
    parser.add_argument("--file_name", required=True, help="Base name of the JSON data file (without .json and without the domain prefix)")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (GPT-OSS reasoning_effort=high; Qwen thinking sampling)",
    )
    parser.add_argument(
        "--extract_model_name",
        default="unsloth/gpt-oss-20b-BF16",
        help="HF model for the high-recall per-chunk extraction stage",
    )
    parser.add_argument(
        "--merge_model_name",
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="HF model for the binary-tree merge and final prune stages",
    )
    parser.add_argument(
        "--checklist_item",
        default=None,
        help="Specific checklist item to process (if not specified, all items will be processed)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Thinking-parser helpers (verbatim from the iterative script)
# ---------------------------------------------------------------------------

class Qwen3ThinkingParser:
    """Extract <think></think> content and final answer string from model output.

    Handles two cases:
    1. Full <think>...</think> tags in output
    2. Only </think> in output (when <think> was added by the chat template)
    """

    def __init__(self, think_end_token_id: int = 151668):
        self.think_end_token_id = think_end_token_id

    def parse_from_text(self, text: str) -> Dict[str, str]:
        # Case 1: Check for full <think>...</think> pattern
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Case 2: Check for only </think> (meaning <think> was in the prompt)
        elif "</think>" in text:
            parts = text.split("</think>", 1)
            thinking = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No thinking tags found
            thinking, answer = "", text.strip()
        return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}


class GPTOSSThinkingParser:
    """Extract channel-based thinking and final answer from GPT-OSS model output.

    GPT-OSS output format:
    <|channel|>analysis<|message|>...<|end|>
    <|start|>assistant<|channel|>final<|message|>...<|return|>

    Note: vLLM stops at <|return|> or <|call|> without including them in output.
    """

    def parse_from_text(self, text: str) -> Dict[str, str]:
        thinking = ""
        answer = text.strip()

        # Check if special tokens are present
        has_channel_tokens = '<|channel|>' in text
        has_message_tokens = '<|message|>' in text

        # Detect if special tokens were stripped (common patterns without tokens)
        looks_stripped = ('analysisWe' in text or 'assistantfinal' in text or
                         'assistantcommentary' in text or 'commentaryanalysis' in text or
                         'analysisThe' in text or 'finalThe' in text)

        if not has_channel_tokens and not has_message_tokens and looks_stripped:
            # Special tokens appear to be missing - return text as-is with warning
            print("Warning: GPT-OSS special tokens appear to be missing from output. "
                  "Ensure skip_special_tokens=False in sampling params.")
            return {"thinking": "", "answer": text.strip(), "has_thinking": False}

        # Extract content from analysis channel (thinking/reasoning)
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)'
        analysis_match = re.search(analysis_pattern, text, re.DOTALL)
        if analysis_match:
            thinking = analysis_match.group(1).strip()

        # Extract content from final channel (the actual answer)
        # Note: vLLM stops at <|return|> without including it, so we look for content
        # after <|channel|>final<|message|> until end of string or <|end|>
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
        final_match = re.search(final_pattern, text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
        elif analysis_match and not final_match:
            # If only analysis channel exists (model might have put answer there)
            # Check if the analysis content looks like JSON
            if thinking and (thinking.startswith('{') or thinking.startswith('[')):
                answer = thinking
                thinking = ""  # Clear thinking since it was actually the answer

        return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}


# ---------------------------------------------------------------------------
# LLM Cache for reusing instances across batches (verbatim from iterative)
# ---------------------------------------------------------------------------

# Global cache for LLM instances. The config hash includes the model name, so
# requesting the merge model after the extraction stage automatically cleans
# up the extraction model and loads the new one (the mid-job model swap).
_LLM_CACHE = {
    "instance": None,
    "config_hash": None,
    "model_name": None,
    "rope_overrides": None,
    "tp_size": None
}

def _get_config_hash(model_name: str, rope_overrides: dict | None, tp_size: int) -> str:
    """Generate a hash to identify unique LLM configurations."""
    import hashlib
    config_str = f"{model_name}_{str(rope_overrides)}_{tp_size}"
    return hashlib.md5(config_str.encode()).hexdigest()

def get_cached_llm(model_name: str, cfg: dict, tp_size: int, rope_overrides: dict | None):
    """Get a cached LLM instance if configuration matches, otherwise build new."""
    global _LLM_CACHE

    config_hash = _get_config_hash(model_name, rope_overrides, tp_size)

    # Check if we can reuse the cached instance
    if (_LLM_CACHE["instance"] is not None and
        _LLM_CACHE["config_hash"] == config_hash and
        _LLM_CACHE["model_name"] == model_name):
        print(f"    → Reusing cached LLM instance (config hash: {config_hash[:8]}...)")
        return _LLM_CACHE["instance"], True  # Return LLM and flag indicating it's cached

    # Need to build a new LLM
    if _LLM_CACHE["instance"] is not None:
        print(f"    → Config changed, cleaning up old LLM and building new one")
        cleanup_llm(_LLM_CACHE["instance"])
        _LLM_CACHE["instance"] = None

    print(f"    → Building new LLM instance (config hash: {config_hash[:8]}...)")
    llm = build_llm(model_name, cfg, tp_size, rope_overrides)

    # Cache the new instance
    _LLM_CACHE["instance"] = llm
    _LLM_CACHE["config_hash"] = config_hash
    _LLM_CACHE["model_name"] = model_name
    _LLM_CACHE["rope_overrides"] = rope_overrides
    _LLM_CACHE["tp_size"] = tp_size

    return llm, False  # Return LLM and flag indicating it's newly built

def clear_llm_cache():
    """Clear the global LLM cache and cleanup resources."""
    global _LLM_CACHE
    if _LLM_CACHE["instance"] is not None:
        cleanup_llm(_LLM_CACHE["instance"])
        _LLM_CACHE["instance"] = None
        _LLM_CACHE["config_hash"] = None
        _LLM_CACHE["model_name"] = None
        _LLM_CACHE["rope_overrides"] = None
        _LLM_CACHE["tp_size"] = None
        print("    → Cleared LLM cache")

# ---------------------------------------------------------------------------
# YaRN bucket configuration (verbatim from iterative)
# ---------------------------------------------------------------------------

BUCKETS_QWEN: Dict[str, dict] = {
    "short":  {"max_prompt": 22_000,  "max_model_len": 32_768,  "hf_overrides": None},
    "medium": {"max_prompt": 56_000,  "max_model_len": 65_536,  "hf_overrides": {
            "rope_scaling": {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 32_768},
            "max_model_len": 65_536,
        }},
    "long":   {"max_prompt": 124_000, "max_model_len": 131_072, "hf_overrides": {
            "rope_scaling": {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 32_768},
            "max_model_len": 131_072,
        }},
}

BUCKETS_QWEN_2507: Dict[str, dict] = {
    "base": {"max_prompt": 248_000, "max_model_len": 262_144, "hf_overrides": None},
}

BUCKETS_GENERIC: Dict[str, dict] = {
    "base": {"max_prompt": 125_000, "max_model_len": 131_072, "hf_overrides": None},
    "long": {"max_prompt": 248_000, "max_model_len": 262_144, "hf_overrides": {
            "rope_scaling": {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 131_072},
            "max_model_len": 262_144,
        }},
}


# ---------------------------------------------------------------------------
# vLLM helpers (verbatim from iterative, except sampling_params/generate_batch
# gain an optional max_tokens override and per-prompt finish_reason tracking)
# ---------------------------------------------------------------------------

def tokenizer_for(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def bucketize(prompts: List[str], tokenizer: AutoTokenizer, table: Dict[str, dict]) -> Dict[str, List[Tuple[int, str]]]:
    buckets: Dict[str, List[Tuple[int, str]]] = {k: [] for k in table}
    for idx, prompt in enumerate(prompts):
        n_tok = len(tokenizer.encode(prompt))
        for bucket_name, cfg in table.items():
            if n_tok <= cfg["max_prompt"]:
                buckets[bucket_name].append((idx, prompt))
                break
        else:
            raise ValueError(f"Prompt at index {idx} has {n_tok} tokens (exceeds maximum supported length).")
    return buckets


def build_llm(model_name: str, cfg: dict, tp_size: int, rope_overrides: dict | None):
    is_gpt_oss = "gpt-oss" in model_name.lower()
    is_qwen = "Qwen" in model_name

    hf_overrides = rope_overrides or {}

    # Only override quantization for GPT-OSS models that are not already BF16
    if is_gpt_oss and "bf16" not in model_name.lower():
        # GPT-OSS specific overrides to disable quantization for non-BF16 models
        hf_overrides = {**hf_overrides, "quantization_config": None}

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tp_size,
        "download_dir": os.environ.get("HF_HOME"),
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "hf_overrides": hf_overrides if hf_overrides else None,
        "trust_remote_code": is_gpt_oss or is_qwen,  # Both GPT-OSS and Qwen need trust_remote_code
    }

    # Only set quantization=None for GPT-OSS models that are not already BF16
    if is_gpt_oss and "bf16" not in model_name.lower():
        llm_kwargs["quantization"] = None

    return LLM(**llm_kwargs)


def sampling_params(model_name: str, enable_thinking: bool, max_tokens: int | None = None) -> SamplingParams:
    is_gpt_oss = "gpt-oss" in model_name.lower()

    if is_gpt_oss:
        # GPT-OSS specific parameters
        # Need to preserve special tokens and add stop tokens
        return SamplingParams(
            temperature=0.7,  # Lower temperature for more consistent extraction
            top_p=1.0,
            max_tokens=max_tokens or 64_000,
            skip_special_tokens=False,  # CRITICAL: Keep special tokens for parsing
            stop_token_ids=[200002, 200012],  # Stop on <|return|> or <|call|>
        )
    elif "Qwen3" in model_name:
        return SamplingParams(
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            max_tokens=max_tokens or 16_000,
        )
    return SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_tokens or 16_000)


def get_gpu_memory_info():
    """Get current GPU memory usage using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 4:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'used_mb': int(parts[2]),
                    'total_mb': int(parts[3]),
                    'used_gb': int(parts[2]) / 1024,
                    'total_gb': int(parts[3]) / 1024
                })
        return gpu_info
    except Exception as e:
        print(f"Failed to get GPU info: {e}")
        return []

def cleanup_llm(llm, *, check_vram: bool = True):
    """Clean up a vLLM engine and free GPU memory.

    Uses the proven cleanup sequence from vLLM examples that properly
    frees GPU memory by destroying model parallel state and distributed environment.

    Args:
        llm: vLLM LLM instance.
        check_vram: If True, report GPU memory usage before and after cleanup.
    """
    import contextlib
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    import ray

    # Get memory usage before cleanup (optional)
    gpu_info_before = []
    if check_vram:
        gpu_info_before = get_gpu_memory_info()
        if gpu_info_before:
            print("GPU memory before cleanup:")
            for gpu in gpu_info_before:
                print(f"  GPU {gpu['index']} ({gpu['name']}): {gpu['used_gb']:.2f}/{gpu['total_gb']:.2f} GB")

    # Destroy the model parallel state and distributed environment first
    # This is critical for properly freeing GPU memory with tensor parallel models
    try:
        destroy_model_parallel()
    except Exception as e:
        print(f"Warning: destroy_model_parallel failed: {e}")

    try:
        destroy_distributed_environment()
    except Exception as e:
        print(f"Warning: destroy_distributed_environment failed: {e}")

    # For vLLM v1, use engine_core.shutdown() instead of deleting model_executor
    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
            llm.llm_engine.engine_core.shutdown()
        elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            # Fallback for older versions
            del llm.llm_engine.model_executor
    except Exception as e:
        print(f"Warning: Could not shutdown engine_core or delete model_executor: {e}")

    # Delete the LLM object
    del llm

    # Destroy the distributed process group with error suppression
    with contextlib.suppress(AssertionError):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:
        print(f"Warning: Ray shutdown failed: {e}")

    print("Successfully deleted the llm pipeline and freed GPU memory.")

    # Check memory usage after cleanup
    if check_vram:
        gpu_info_after = get_gpu_memory_info()
        if gpu_info_after:
            print("GPU memory after cleanup:")
            total_freed_mb = 0
            for i, gpu in enumerate(gpu_info_after):
                print(f"  GPU {gpu['index']} ({gpu['name']}): {gpu['used_gb']:.2f}/{gpu['total_gb']:.2f} GB", end="")
                if i < len(gpu_info_before):
                    freed_mb = gpu_info_before[i]['used_mb'] - gpu['used_mb']
                    total_freed_mb += freed_mb
                    print(f" (freed: {freed_mb/1024:.2f} GB)")
                else:
                    print()
            if total_freed_mb > 0:
                print(f"Total memory freed: {total_freed_mb/1024:.2f} GB")


def generate_batch(prompts: List[str], model_name: str, enable_thinking: bool,
                   max_tokens: int | None = None) -> Tuple[List[str], Dict[str, Any]]:
    """Generate responses for a batch of prompts using vLLM.

    Returns:
        Tuple of (responses, token_stats) where token_stats contains:
        - total_input_tokens / total_output_tokens / num_prompts / avg_*
        - per_prompt_stats: [{input_tokens, output_tokens, finish_reason}, ...]
    """
    # Check if this is a Qwen3-2507 model (native long context support)
    if "Qwen3" in model_name and "2507" in model_name:
        table = BUCKETS_QWEN_2507
    elif "Qwen3" in model_name:
        table = BUCKETS_QWEN
    else:
        table = BUCKETS_GENERIC

    tok = tokenizer_for(model_name)
    buckets = bucketize(prompts, tok, table)
    sparams = sampling_params(model_name, enable_thinking, max_tokens=max_tokens)

    indexed_out: List[Tuple[int, Any, Dict[str, Any]]] = []
    tp_size = torch.cuda.device_count() or 1

    per_prompt_stats = []  # Will store stats for each prompt in order
    total_input_tokens = 0
    total_output_tokens = 0

    # Check if all prompts belong to base or short bucket (no cleanup needed)
    active_buckets = [name for name in table if buckets[name]]
    skip_cleanup = len(active_buckets) == 1 and active_buckets[0] in ["base", "short"]

    if skip_cleanup:
        print(f"  All prompts in '{active_buckets[0]}' bucket - skipping LLM cleanup for efficiency")

    for bucket_name in table:
        pairs = buckets[bucket_name]
        if not pairs:
            continue
        idxs, bucket_prompts = zip(*pairs)
        cfg = table[bucket_name]
        # Use hf_overrides from config (handles Qwen3, Qwen3-2507, and generic models)
        rope_overrides = cfg.get("hf_overrides")
        print(f"[{model_name}] Bucket '{bucket_name}': {len(bucket_prompts)} prompt(s), max_len={cfg['max_model_len']}")

        # Use cached LLM if possible
        llm, is_cached = get_cached_llm(model_name, cfg, tp_size, rope_overrides)
        outs = llm.generate(list(bucket_prompts), sparams)

        # Collect per-prompt token statistics from outputs
        bucket_stats = []
        for out in outs:
            prompt_input_tokens = 0
            prompt_output_tokens = 0
            finish_reason = None

            if hasattr(out, 'outputs') and len(out.outputs) > 0:
                finish_reason = getattr(out.outputs[0], 'finish_reason', None)

            if hasattr(out, 'metrics') and out.metrics:
                # vLLM provides metrics including token counts
                if hasattr(out.metrics, 'prompt_tokens'):
                    prompt_input_tokens = out.metrics.prompt_tokens
                    total_input_tokens += prompt_input_tokens
                if hasattr(out.metrics, 'completion_tokens'):
                    prompt_output_tokens = out.metrics.completion_tokens
                    total_output_tokens += prompt_output_tokens
            # Alternative: count tokens manually if metrics not available
            elif hasattr(out, 'prompt_token_ids'):
                prompt_input_tokens = len(out.prompt_token_ids)
                total_input_tokens += prompt_input_tokens
                if hasattr(out, 'outputs') and len(out.outputs) > 0 and hasattr(out.outputs[0], 'token_ids'):
                    prompt_output_tokens = len(out.outputs[0].token_ids)
                    total_output_tokens += prompt_output_tokens

            bucket_stats.append({
                "input_tokens": prompt_input_tokens,
                "output_tokens": prompt_output_tokens,
                "finish_reason": finish_reason
            })

        # Store outputs with their stats
        indexed_out.extend(zip(idxs, outs, bucket_stats))

        # Don't cleanup cached instances when skip_cleanup is True (base/short buckets)
        # The cache will be reused for the next batch if it has the same configuration
        if not skip_cleanup and not is_cached:
            # Only cleanup non-cached instances or when switching configurations
            cleanup_llm(llm)
            # Clear the cache entry since we're cleaning up
            _LLM_CACHE["instance"] = None
            _LLM_CACHE["config_hash"] = None

    # Sort by original index and extract text with stats
    sorted_data = sorted(indexed_out, key=lambda t: t[0])
    outputs = [o for _i, o, _s in sorted_data]
    per_prompt_stats = [s for _i, _o, s in sorted_data]

    # Parse thinking if applicable
    if "Qwen3" in model_name:
        parser = Qwen3ThinkingParser()
    elif "gpt-oss" in model_name.lower():
        parser = GPTOSSThinkingParser()
    else:
        parser = None

    results = []
    for out in outputs:
        text = out.outputs[0].text if hasattr(out, "outputs") else str(out)
        if parser:
            parsed = parser.parse_from_text(text)
            results.append(parsed["answer"])
        else:
            results.append(text)

    num_prompts = len(prompts)
    token_stats = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_prompts": num_prompts,
        "avg_input_tokens": total_input_tokens / num_prompts if num_prompts > 0 else 0,
        "avg_output_tokens": total_output_tokens / num_prompts if num_prompts > 0 else 0,
        "per_prompt_stats": per_prompt_stats
    }

    return results, token_stats


def extract_json_from_response(response: str):
    """Extract JSON from model response.

    Returns a dict (preferred), a list (bare top-level array fallback), or None.
    Candidates are tried in order, falling through on parse failure - a bare
    top-level array contains {...} objects, so the raw-object pattern alone
    would otherwise hijack it with an unparseable fragment.
    """
    candidates: List[str] = []
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        candidates.append(match.group(1))
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
    if match:
        candidates.append(match.group(1))
    # Raw candidates: prefer the longest span - an object response encloses its
    # inner lists, and a bare-list response encloses its inner objects.
    raw_candidates: List[str] = []
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        raw_candidates.append(match.group(0))
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        raw_candidates.append(match.group(0))
    raw_candidates.sort(key=len, reverse=True)
    candidates.extend(raw_candidates)

    for json_str in candidates:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to clean up common issues
            cleaned = json_str.replace('\n', ' ').replace('\r', '')
            try:
                return json.loads(cleaned)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def normalize_state(obj: Any) -> Optional[Dict[str, Any]]:
    """Coerce a parsed model response into the canonical state shape.

    Returns {"reasoning": str, "extracted": list} or None if unusable.
    """
    if isinstance(obj, dict) and isinstance(obj.get("extracted"), list):
        reasoning = obj.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        return {"reasoning": reasoning, "extracted": obj["extracted"]}
    if isinstance(obj, list):
        # Bare top-level list: treat it as the extracted list
        return {"reasoning": "", "extracted": obj}
    return None


def canonical_empty_state() -> Dict[str, Any]:
    return {
        "reasoning": "No relevant information was found for this checklist item in this case.",
        "extracted": []
    }


def concat_states(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> Dict[str, Any]:
    """Lossless fallback merge: concatenate the two extracted lists."""
    return {
        "reasoning": "[concatenation fallback] Entries from two checklists were combined without LLM merging.",
        "extracted": list(state_a.get("extracted", [])) + list(state_b.get("extracted", []))
    }


def sanitize_extracted_entries(extracted: List[Any]) -> List[Dict[str, Any]]:
    """Coerce extracted entries into the exact iterative-format schema:
    [{"evidence": [{"text": str, "source_document": str, "location": str}], "value": str}]

    Lenient about model deviations (bare-string entries, single evidence dict,
    string evidence, 'evidences' key, non-string values) but the OUTPUT shape
    is always exact. Unusable entries are dropped.
    """
    clean: List[Dict[str, Any]] = []
    for entry in extracted:
        if isinstance(entry, str):
            if entry.strip():
                clean.append({"evidence": [], "value": entry})
            continue
        if not isinstance(entry, dict):
            continue
        value = entry.get("value", "")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        evidence_in = entry.get("evidence", entry.get("evidences", []))
        if isinstance(evidence_in, (dict, str)):
            evidence_in = [evidence_in]
        evidence: List[Dict[str, str]] = []
        if isinstance(evidence_in, list):
            for ev in evidence_in:
                if isinstance(ev, str):
                    evidence.append({"text": ev, "source_document": "", "location": ""})
                elif isinstance(ev, dict):
                    evidence.append({
                        "text": ev.get("text", "") if isinstance(ev.get("text", ""), str) else str(ev.get("text")),
                        "source_document": ev.get("source_document", "") if isinstance(ev.get("source_document", ""), str) else str(ev.get("source_document")),
                        "location": ev.get("location", "") if isinstance(ev.get("location", ""), str) else str(ev.get("location")),
                    })
        if not value.strip() and not evidence:
            continue
        clean.append({"evidence": evidence, "value": value})
    return clean


def sanitize_state(state: Any) -> Dict[str, Any]:
    """Final-boundary guarantee: the exact result schema the evaluation
    notebooks consume, identical to the iterative method's format."""
    norm = normalize_state(state)
    if norm is None:
        return canonical_empty_state()
    return {"reasoning": norm["reasoning"], "extracted": sanitize_extracted_entries(norm["extracted"])}


# ---------------------------------------------------------------------------
# Data organisation (flattening loop verbatim from iterative)
# ---------------------------------------------------------------------------

def build_case_data(keys: List[List[str]], chunks: List[List[str]]) -> Dict[str, Dict[str, Any]]:
    """Reorganize per-document chunk lists into per-case flattened chunk lists."""
    case_data: Dict[str, Dict[str, Any]] = {}
    for doc_idx, (key, doc_chunks) in enumerate(zip(keys, chunks)):
        case_id, doc_name = key
        if case_id not in case_data:
            case_data[case_id] = {
                "flattened_chunks": [],
                "chunk_sources": [],  # (doc_name, doc_chunk_idx, total_doc_chunks)
                "total_chunks": 0
            }
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            case_data[case_id]["flattened_chunks"].append(chunk_text)
            case_data[case_id]["chunk_sources"].append((doc_name, chunk_idx, len(doc_chunks)))
        case_data[case_id]["total_chunks"] = len(case_data[case_id]["flattened_chunks"])
    return case_data


# ---------------------------------------------------------------------------
# Chat template application (factored from the iterative inline block)
# ---------------------------------------------------------------------------

def apply_chat_template_for(prompt: str, model_name: str, tokenizer: AutoTokenizer, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    is_gpt_oss = "gpt-oss" in model_name.lower()

    if is_gpt_oss:
        # GPT-OSS specific template parameters
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": "high" if enable_thinking else "medium"
            # Explicitly NOT setting 'tools' to avoid the tools message in the template
        }
    else:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking and "Qwen3" in model_name:
            kwargs["enable_thinking"] = True

    formatted = tokenizer.apply_chat_template(messages, **kwargs)

    # For GPT-OSS, remove the unwanted tools line that the template adds
    if is_gpt_oss:
        tools_line = "\nCalls to these tools must go to the commentary channel: 'functions'."
        if tools_line in formatted:
            formatted = formatted.replace(tools_line, "")

    return formatted


# ---------------------------------------------------------------------------
# Atomic JSON IO + checkpoint paths/validators
# ---------------------------------------------------------------------------

def save_json_atomic(path: Path, obj: Any, indent: int = 2):
    """Write atomically: temp file + fsync + rename, so a kill/preemption
    mid-write can never leave a truncated or corrupted file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_json_tolerant(path: Path) -> Optional[Any]:
    """Load JSON, returning None on a missing or unreadable file (never raises)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError) as e:
        print(f"  WARNING: {path} is unreadable ({e}); treating it as absent.")
        return None


def combined_model_dir(extract_model: str, merge_model: str) -> str:
    return f"{Path(extract_model).name}__{Path(merge_model).name}"


def get_ckpt_path(kind: str, item_name: str, extract_model: str, merge_model: str,
                  file_name: str, enable_thinking: bool) -> Path:
    """kind in {'extract', 'merge', 'prune'}."""
    states_dir = Path("states") / combined_model_dir(extract_model, merge_model)
    return states_dir / f"{file_name}_thinking_{enable_thinking}_{kind}_{item_name}.json"


def build_saving_path(extract_model: str, merge_model: str, file_name: str, enable_thinking: bool) -> Path:
    return Path("results") / combined_model_dir(extract_model, merge_model) / f"{file_name}_thinking_{enable_thinking}.json"


def nyc_timestamps() -> Dict[str, Any]:
    nyc_time = datetime.now(ZoneInfo("America/New_York"))
    return {
        "timestamp": nyc_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "timestamp_epoch": time.time()
    }


def validate_extract_ckpt(ckpt: Any, case_data: Dict[str, Dict]) -> bool:
    if not isinstance(ckpt, dict) or ckpt.get("complete") is not True:
        return False
    cases = ckpt.get("cases")
    if not isinstance(cases, dict) or set(cases.keys()) != set(case_data.keys()):
        return False
    for case_id, chunk_states in cases.items():
        if not isinstance(chunk_states, list) or len(chunk_states) != case_data[case_id]["total_chunks"]:
            return False
    return True


def validate_merge_ckpt(ckpt: Any, case_data: Dict[str, Dict]) -> bool:
    if not isinstance(ckpt, dict):
        return False
    cases = ckpt.get("cases")
    if not isinstance(cases, dict) or set(cases.keys()) != set(case_data.keys()):
        return False
    for case_id, entries in cases.items():
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if not isinstance(entry, dict) or "state" not in entry or "chunk_range" not in entry:
                return False
            if normalize_state(entry["state"]) is None:
                return False
    return True


def validate_prune_ckpt(ckpt: Any, case_data: Dict[str, Dict]) -> bool:
    if not isinstance(ckpt, dict) or ckpt.get("complete") is not True:
        return False
    final_states = ckpt.get("final_states")
    if not isinstance(final_states, dict) or set(final_states.keys()) != set(case_data.keys()):
        return False
    for state in final_states.values():
        if normalize_state(state) is None:
            return False
    return True


def resolve_item_status(item_name: str, case_data: Dict[str, Dict], extract_model: str,
                        merge_model: str, file_name: str, enable_thinking: bool) -> Tuple[str, Optional[Dict]]:
    """Decide where an item resumes, from its validated checkpoints.

    Priority: done (prune ckpt) > merging (merge ckpt) > extracted (extract
    ckpt) > needs_extract. Absence or any validation failure of a checkpoint
    simply demotes the item - an item can never silently skip work.
    """
    prune_ckpt = load_json_tolerant(get_ckpt_path("prune", item_name, extract_model, merge_model, file_name, enable_thinking))
    if prune_ckpt is not None:
        if validate_prune_ckpt(prune_ckpt, case_data):
            return "done", prune_ckpt
        print(f"  WARNING: prune checkpoint for {item_name} failed validation; ignoring it.")

    merge_ckpt = load_json_tolerant(get_ckpt_path("merge", item_name, extract_model, merge_model, file_name, enable_thinking))
    if merge_ckpt is not None:
        if validate_merge_ckpt(merge_ckpt, case_data):
            return "merging", merge_ckpt
        print(f"  WARNING: merge checkpoint for {item_name} failed validation; ignoring it.")

    extract_ckpt = load_json_tolerant(get_ckpt_path("extract", item_name, extract_model, merge_model, file_name, enable_thinking))
    if extract_ckpt is not None:
        if validate_extract_ckpt(extract_ckpt, case_data):
            return "extracted", extract_ckpt
        print(f"  WARNING: extract checkpoint for {item_name} failed validation; ignoring it.")

    return "needs_extract", None


# ---------------------------------------------------------------------------
# Stage 1: high-recall per-chunk extraction (GPT-OSS)
# ---------------------------------------------------------------------------

def run_extraction_stage(
    items_needing: List[str],
    items_to_process: Dict[str, str],
    case_data: Dict[str, Dict],
    extract_model: str,
    enable_thinking: bool,
    extract_tokenizer: AutoTokenizer,
    file_name: str,
    merge_model: str,
) -> Dict[str, Dict]:
    """Run one extraction batch per item; checkpoint each item on completion.

    Returns {item_name: extract_ckpt_payload}.
    """
    with open(EXTRACT_TEMPLATE, "r") as f:
        template = f.read()

    payloads = {}
    for item_idx, item_name in enumerate(items_needing):
        item_description = items_to_process[item_name]
        print(f"\n{'='*60}")
        print(f"[EXTRACT {item_idx + 1}/{len(items_needing)}] {item_name}")
        print(f"{'='*60}")

        batch_prompts: List[str] = []
        batch_keys: List[Tuple[str, int]] = []  # (case_id, flattened_chunk_idx)
        for case_id, data in case_data.items():
            for flat_idx, chunk_text in enumerate(data["flattened_chunks"]):
                doc_name, doc_chunk_idx, total_doc_chunks = data["chunk_sources"][flat_idx]
                prompt = template.format(
                    item_description=item_description,
                    document_name=doc_name,
                    chunk_id=doc_chunk_idx + 1,
                    total_chunks=total_doc_chunks,
                    document_chunk=chunk_text
                )
                batch_prompts.append(apply_chat_template_for(prompt, extract_model, extract_tokenizer, enable_thinking))
                batch_keys.append((case_id, flat_idx))

        print(f"  Generating {len(batch_prompts)} chunk-extraction prompts ({len(case_data)} cases)...")
        responses, token_stats = generate_batch(batch_prompts, extract_model, enable_thinking)
        per_prompt = token_stats.get("per_prompt_stats", [])

        # Apply responses
        cases = {case_id: [None] * data["total_chunks"] for case_id, data in case_data.items()}
        item_case_stats = {
            case_id: {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_prompts": 0,
                "chunks_in_case": data["total_chunks"],
                "n_nonempty": 0,
                "n_parse_failures": 0,
            }
            for case_id, data in case_data.items()
        }
        parse_failures = 0
        for idx, (response, (case_id, flat_idx)) in enumerate(zip(responses, batch_keys)):
            stats = item_case_stats[case_id]
            if idx < len(per_prompt):
                stats["total_input_tokens"] += per_prompt[idx]["input_tokens"]
                stats["total_output_tokens"] += per_prompt[idx]["output_tokens"]
            stats["total_prompts"] += 1

            state = normalize_state(extract_json_from_response(response))
            if state is None:
                parse_failures += 1
                stats["n_parse_failures"] += 1
                print(f"    Warning: failed to parse extraction JSON for case {case_id}, chunk {flat_idx + 1} (treated as empty)")
            else:
                cases[case_id][flat_idx] = state
                if state["extracted"]:
                    stats["n_nonempty"] += 1

        if parse_failures > 0.5 * len(batch_prompts):
            print(f"\n  !!!!!! WARNING: {parse_failures}/{len(batch_prompts)} extraction prompts failed to parse "
                  f"for {item_name} - check the template/model output format !!!!!!\n")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "item_name": item_name,
            "extract_model": extract_model,
            "n_chunks_per_case": {case_id: data["total_chunks"] for case_id, data in case_data.items()},
            "cases": cases,
            "item_case_stats": item_case_stats,
            "parse_failures": parse_failures,
            **nyc_timestamps(),
        }
        ckpt_path = get_ckpt_path("extract", item_name, extract_model, merge_model, file_name, enable_thinking)
        save_json_atomic(ckpt_path, payload)
        n_nonempty = sum(s["n_nonempty"] for s in item_case_stats.values())
        print(f"  Extraction checkpoint saved for {item_name} "
              f"({n_nonempty}/{len(batch_prompts)} non-empty chunk checklists, {parse_failures} parse failures)")
        payloads[item_name] = payload

    return payloads


# ---------------------------------------------------------------------------
# Stage 2: binary-tree merge (Qwen)
# ---------------------------------------------------------------------------

def init_merge_lists(extract_cases: Dict[str, List]) -> Dict[str, List[Dict]]:
    """Build initial merge lists: non-empty chunk states in chunk order.

    null (parse failure) and empty-extracted chunk states are dropped here -
    this is the 'skip trivial merges in code' rule.
    """
    lists: Dict[str, List[Dict]] = {}
    for case_id, chunk_states in extract_cases.items():
        entries = []
        for flat_idx, state in enumerate(chunk_states):
            norm = normalize_state(state) if state is not None else None
            if norm is not None and norm["extracted"]:
                entries.append({"state": norm, "chunk_range": [flat_idx, flat_idx]})
        lists[case_id] = entries
    return lists


def fresh_merge_stats(case_data: Dict[str, Dict]) -> Dict[str, Dict]:
    return {
        case_id: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_prompts": 0,
            "n_llm_merges": 0,
            "n_concat_fallbacks": 0,
            "n_oversize_skips": 0,
            "n_length_truncations": 0,
        }
        for case_id in case_data
    }


def ensure_merge_stats_keys(stats: Dict[str, Dict], case_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Fill any missing cases/keys in stats loaded from a checkpoint."""
    fresh = fresh_merge_stats(case_data)
    for case_id, defaults in fresh.items():
        if case_id not in stats or not isinstance(stats[case_id], dict):
            stats[case_id] = defaults
        else:
            for key, value in defaults.items():
                stats[case_id].setdefault(key, value)
    return stats


def build_provenance(label: str, entry: Dict, case_data: Dict[str, Dict], case_id: str) -> str:
    start, end = entry["chunk_range"]
    total = case_data[case_id]["total_chunks"]
    doc_names: List[str] = []
    for i in range(start, end + 1):
        doc_name = case_data[case_id]["chunk_sources"][i][0]
        if doc_name not in doc_names:
            doc_names.append(doc_name)
    if len(doc_names) > 3:
        docs_shown = ", ".join(doc_names[:3]) + f", and {len(doc_names) - 3} more documents"
    else:
        docs_shown = ", ".join(doc_names)
    return (f"(Checklist {label} was extracted from chunks {start + 1}-{end + 1} "
            f"of the {total} total document chunks; source documents: {docs_shown})")


def checklist_json_for_prompt(state: Dict[str, Any]) -> str:
    """Only the extracted list goes into prompts (stale reasoning is dropped)."""
    return json.dumps({"extracted": state.get("extracted", [])}, indent=2)


def display_item_name(item_name: str) -> str:
    """Human-readable checklist item name for prompts (underscores -> spaces)."""
    return item_name.replace("_", " ").strip()


def run_merge_stage(
    items_state: Dict[str, Dict],
    case_data: Dict[str, Dict],
    merge_model: str,
    enable_thinking: bool,
    qwen_tokenizer: AutoTokenizer,
    file_name: str,
    extract_model: str,
    item_descriptions: Dict[str, str],
):
    """Run global merge levels until every (item, case) list has length <= 1.

    items_state: {item_name: {"cases": {case_id: [entry...]}, "stats": {...}, "level": int}}
    Mutated in place; per-item checkpoints written after every level.
    """
    with open(MERGE_TEMPLATE, "r") as f:
        template = f.read()

    prompt_budget = QWEN_MAX_MODEL_LEN - MERGE_MAX_TOKENS - PROMPT_GUARD_MARGIN

    round_idx = 0
    while True:
        round_idx += 1
        batch_prompts: List[str] = []
        batch_meta: List[Tuple[str, str]] = []  # (item_name, case_id)
        plans: Dict[Tuple[str, str], List[Tuple]] = {}
        items_with_pairs = set()
        n_oversize_this_round = 0

        for item_name, payload in items_state.items():
            for case_id, lst in payload["cases"].items():
                if len(lst) < 2:
                    continue
                items_with_pairs.add(item_name)
                ops: List[Tuple] = []
                i = 0
                while i < len(lst):
                    if i + 1 < len(lst):
                        a, b = lst[i], lst[i + 1]
                        prompt = template.format(
                            item_name=display_item_name(item_name),
                            item_description=item_descriptions[item_name],
                            checklist_a=checklist_json_for_prompt(a["state"]),
                            checklist_a_provenance=build_provenance("A", a, case_data, case_id),
                            checklist_b=checklist_json_for_prompt(b["state"]),
                            checklist_b_provenance=build_provenance("B", b, case_data, case_id),
                        )
                        formatted = apply_chat_template_for(prompt, merge_model, qwen_tokenizer, enable_thinking)
                        n_tok = len(qwen_tokenizer.encode(formatted))
                        if n_tok > prompt_budget:
                            # Oversize guard: vLLM would silently clamp generation
                            # and truncate the JSON - concatenate in code instead.
                            ops.append(("concat", a, b))
                            n_oversize_this_round += 1
                        else:
                            ops.append(("llm", a, b, len(batch_prompts)))
                            batch_prompts.append(formatted)
                            batch_meta.append((item_name, case_id))
                        i += 2
                    else:
                        # Odd tail carries up to the next level unchanged
                        ops.append(("keep", lst[i]))
                        i += 1
                plans[(item_name, case_id)] = ops

        if not plans:
            break  # every list is down to <= 1 entry

        print(f"\n{'='*60}")
        print(f"[MERGE level {round_idx}] {len(batch_prompts)} LLM merges, "
              f"{n_oversize_this_round} oversize concats, across {len(plans)} (item, case) pairs")
        print(f"{'='*60}")

        responses: List[str] = []
        per_prompt: List[Dict] = []
        if batch_prompts:
            responses, token_stats = generate_batch(batch_prompts, merge_model, enable_thinking,
                                                    max_tokens=MERGE_MAX_TOKENS)
            per_prompt = token_stats.get("per_prompt_stats", [])

        # Attribute token usage
        for idx, (item_name, case_id) in enumerate(batch_meta):
            stats = items_state[item_name]["stats"][case_id]
            if idx < len(per_prompt):
                stats["total_input_tokens"] += per_prompt[idx]["input_tokens"]
                stats["total_output_tokens"] += per_prompt[idx]["output_tokens"]
                if per_prompt[idx].get("finish_reason") == "length":
                    stats["n_length_truncations"] += 1
                    print(f"    Warning: merge output hit the length limit for case {case_id}, item {item_name}")
            stats["total_prompts"] += 1

        # Apply plans
        for (item_name, case_id), ops in plans.items():
            stats = items_state[item_name]["stats"][case_id]
            new_lst: List[Dict] = []
            for op in ops:
                if op[0] == "keep":
                    new_lst.append(op[1])
                    continue
                a, b = op[1], op[2]
                chunk_range = [min(a["chunk_range"][0], b["chunk_range"][0]),
                               max(a["chunk_range"][1], b["chunk_range"][1])]
                if op[0] == "concat":
                    merged = concat_states(a["state"], b["state"])
                    stats["n_oversize_skips"] += 1
                else:
                    parsed = normalize_state(extract_json_from_response(responses[op[3]]))
                    if parsed is None or not parsed["extracted"]:
                        # Parse failure, or the model dropped everything from two
                        # non-empty inputs - both fall back to lossless concat.
                        merged = concat_states(a["state"], b["state"])
                        stats["n_concat_fallbacks"] += 1
                        print(f"    Warning: merge fallback (concat) for case {case_id}, item {item_name}")
                    else:
                        merged = parsed
                        stats["n_llm_merges"] += 1
                new_lst.append({"state": merged, "chunk_range": chunk_range})
            items_state[item_name]["cases"][case_id] = new_lst

        # Checkpoint every item that participated in this level
        print(f"  Saving merge checkpoints for {len(items_with_pairs)} items...")
        for item_name in items_with_pairs:
            payload = items_state[item_name]
            payload["level"] = payload.get("level", 0) + 1
            ckpt = {
                "schema_version": SCHEMA_VERSION,
                "item_name": item_name,
                "merge_model": merge_model,
                "level": payload["level"],
                "cases": payload["cases"],
                "item_case_stats": payload["stats"],
                **nyc_timestamps(),
            }
            save_json_atomic(get_ckpt_path("merge", item_name, extract_model, merge_model, file_name, enable_thinking), ckpt)


# ---------------------------------------------------------------------------
# Stage 3: final verification / pruning (Qwen)
# ---------------------------------------------------------------------------

def run_prune_stage(
    items_state: Dict[str, Dict],
    case_data: Dict[str, Dict],
    merge_model: str,
    enable_thinking: bool,
    qwen_tokenizer: AutoTokenizer,
    file_name: str,
    extract_model: str,
    item_descriptions: Dict[str, str],
) -> Dict[str, Dict]:
    """One strict cleanup prompt per non-empty (item, case); per-item checkpoints.

    Returns {item_name: prune_ckpt_payload}.
    """
    with open(PRUNE_TEMPLATE, "r") as f:
        template = f.read()

    prompt_budget = QWEN_MAX_MODEL_LEN - PRUNE_MAX_TOKENS - PROMPT_GUARD_MARGIN

    batch_prompts: List[str] = []
    batch_meta: List[Tuple[str, str]] = []  # (item_name, case_id)
    final_states: Dict[str, Dict[str, Dict]] = {}  # item -> case -> state
    prune_stats: Dict[str, Dict[str, Dict]] = {}   # item -> case -> stats

    for item_name, payload in items_state.items():
        final_states[item_name] = {}
        prune_stats[item_name] = {}
        for case_id in case_data:
            lst = payload["cases"].get(case_id, [])
            stats = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_prompts": 0,
                "skipped_empty": 0,
                "oversize_skip": 0,
                "parse_failure": 0,
                "length_truncation": 0,
                "pruned": 0,
            }
            prune_stats[item_name][case_id] = stats

            if not lst:
                final_states[item_name][case_id] = canonical_empty_state()
                stats["skipped_empty"] = 1
                continue
            assert len(lst) == 1, f"merge stage left {len(lst)} entries for case {case_id}, item {item_name}"
            merged_state = lst[0]["state"]

            prompt = template.format(
                item_name=display_item_name(item_name),
                item_description=item_descriptions[item_name],
                checklist=checklist_json_for_prompt(merged_state),
            )
            formatted = apply_chat_template_for(prompt, merge_model, qwen_tokenizer, enable_thinking)
            if len(qwen_tokenizer.encode(formatted)) > prompt_budget:
                final_states[item_name][case_id] = {
                    "reasoning": "[prune skipped: prompt exceeded the context budget] " + merged_state.get("reasoning", ""),
                    "extracted": merged_state.get("extracted", []),
                }
                stats["oversize_skip"] = 1
                continue
            # Placeholder until the batch response is applied; pre-prune state
            # is the fallback if the prune output cannot be parsed.
            final_states[item_name][case_id] = merged_state
            batch_meta.append((item_name, case_id))
            batch_prompts.append(formatted)

    print(f"\n{'='*60}")
    print(f"[PRUNE] {len(batch_prompts)} prompts "
          f"({sum(s['skipped_empty'] for it in prune_stats.values() for s in it.values())} empty cases skipped)")
    print(f"{'='*60}")

    if batch_prompts:
        responses, token_stats = generate_batch(batch_prompts, merge_model, enable_thinking,
                                                max_tokens=PRUNE_MAX_TOKENS)
        per_prompt = token_stats.get("per_prompt_stats", [])
        for idx, (response, (item_name, case_id)) in enumerate(zip(responses, batch_meta)):
            stats = prune_stats[item_name][case_id]
            if idx < len(per_prompt):
                stats["total_input_tokens"] += per_prompt[idx]["input_tokens"]
                stats["total_output_tokens"] += per_prompt[idx]["output_tokens"]
                if per_prompt[idx].get("finish_reason") == "length":
                    stats["length_truncation"] = 1
                    print(f"    Warning: prune output hit the length limit for case {case_id}, item {item_name}")
            stats["total_prompts"] += 1

            parsed = normalize_state(extract_json_from_response(response))
            if parsed is None:
                # Keep the pre-prune (merged) state - lossless fallback
                stats["parse_failure"] = 1
                print(f"    Warning: prune parse failure for case {case_id}, item {item_name} (keeping merged checklist)")
            else:
                # An empty extracted list IS legitimate here (strict pruning)
                final_states[item_name][case_id] = parsed
                stats["pruned"] = 1

    # Final-boundary sanitization: every state stored in results must match
    # the iterative format exactly (entries: {"evidence": [{"text",
    # "source_document", "location"}], "value"}), regardless of how the
    # merge/prune models deviated.
    for item_name in items_state:
        for case_id in case_data:
            raw = final_states[item_name][case_id]
            cleaned = sanitize_state(raw)
            if cleaned != raw:
                print(f"    Note: sanitized final state for case {case_id}, item {item_name}")
            final_states[item_name][case_id] = cleaned

    # Per-item prune checkpoints
    payloads = {}
    print(f"  Saving prune checkpoints for {len(items_state)} items...")
    for item_name in items_state:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "item_name": item_name,
            "merge_model": merge_model,
            "final_states": final_states[item_name],
            "item_case_stats": prune_stats[item_name],
            **nyc_timestamps(),
        }
        save_json_atomic(get_ckpt_path("prune", item_name, extract_model, merge_model, file_name, enable_thinking), payload)
        payloads[item_name] = payload

    return payloads


# ---------------------------------------------------------------------------
# Token-stats assembly
# ---------------------------------------------------------------------------

CORE_STAT_KEYS = ("total_input_tokens", "total_output_tokens", "total_prompts")


def derive_stats_views(by_stage_item_case: Dict[str, Dict[str, Dict[str, Dict]]],
                       extract_model: str, merge_model: str) -> Dict[str, Any]:
    """Derive all aggregate views from the by_stage_item_case source of truth."""

    def _zeros():
        return {k: 0 for k in CORE_STAT_KEYS}

    by_stage: Dict[str, Dict] = {}
    by_item: Dict[str, Dict] = {}
    by_case: Dict[str, Dict] = {}
    by_item_case: Dict[str, Dict[str, Dict]] = {}

    for stage, items in by_stage_item_case.items():
        stage_agg = by_stage.setdefault(stage, {
            **_zeros(),
            "model_name": extract_model if stage == "extract" else merge_model
        })
        for item_name, cases in items.items():
            item_agg = by_item.setdefault(item_name, _zeros())
            for case_id, stats in cases.items():
                case_agg = by_case.setdefault(case_id, _zeros())
                ic_agg = by_item_case.setdefault(item_name, {}).setdefault(case_id, _zeros())
                for key in CORE_STAT_KEYS:
                    value = stats.get(key, 0)
                    stage_agg[key] += value
                    item_agg[key] += value
                    case_agg[key] += value
                    ic_agg[key] += value

    def _add_avgs(d: Dict):
        if d.get("total_prompts", 0) > 0:
            d["avg_input_tokens_per_prompt"] = d["total_input_tokens"] / d["total_prompts"]
            d["avg_output_tokens_per_prompt"] = d["total_output_tokens"] / d["total_prompts"]

    for agg in list(by_stage.values()) + list(by_item.values()) + list(by_case.values()):
        _add_avgs(agg)
    for cases in by_item_case.values():
        for agg in cases.values():
            _add_avgs(agg)

    def _sum_extra(stage: str, key: str) -> int:
        total = 0
        for cases in by_stage_item_case.get(stage, {}).values():
            for stats in cases.values():
                total += stats.get(key, 0)
        return total

    overall = {
        "total_input_tokens": sum(s["total_input_tokens"] for s in by_stage.values()),
        "total_output_tokens": sum(s["total_output_tokens"] for s in by_stage.values()),
        "total_prompts": sum(s["total_prompts"] for s in by_stage.values()),
        "items_processed": len(by_item),
        "cases_processed": len(by_case),
        "parse_failures": {
            "extract": _sum_extra("extract", "n_parse_failures"),
            "merge": _sum_extra("merge", "n_concat_fallbacks"),
            "prune": _sum_extra("prune", "parse_failure"),
        },
        "n_llm_merges": _sum_extra("merge", "n_llm_merges"),
        "concat_fallbacks": _sum_extra("merge", "n_concat_fallbacks"),
        "oversize_merge_skips": _sum_extra("merge", "n_oversize_skips"),
        "oversize_prune_skips": _sum_extra("prune", "oversize_skip"),
        "length_truncations": _sum_extra("merge", "n_length_truncations") + _sum_extra("prune", "length_truncation"),
    }
    _add_avgs(overall)

    return {
        "overall": overall,
        "by_stage": by_stage,
        "by_item": by_item,
        "by_case": by_case,
        "by_item_case": by_item_case,
        "by_stage_item_case": by_stage_item_case,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    file_name: str = args.file_name
    enable_thinking: bool = args.enable_thinking
    extract_model: str = args.extract_model_name
    merge_model: str = args.merge_model_name
    selected_item: str = args.checklist_item
    domain: str = args.domain

    # Resolve the domain's template/checklist paths. Legal keeps the
    # historical unprefixed file naming; other domains are prefixed so
    # data/checkpoints/results are domain-tagged. The guard tolerates an
    # already-prefixed --file_name.
    resolve_domain_paths(domain)
    if domain != "legal" and not file_name.startswith(f"{domain}_"):
        file_name = f"{domain}_{file_name}"
    print(f"Domain: {domain} | data/results name: {file_name}")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    chunk_data_path = DATA_DIR / f"{file_name}.json"
    print(f"Loading chunk data from {chunk_data_path}...")
    with open(chunk_data_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    keys = chunk_data["keys"]
    chunks = chunk_data["chunks"]
    print(f"Loaded {len(keys)} documents with chunks")

    with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
        checklist_items = json.load(f)
    print(f"Loaded {len(checklist_items)} checklist items")

    case_data = build_case_data(keys, chunks)
    print(f"Organized {len(case_data)} cases:")
    for case_id, data in case_data.items():
        print(f"  Case {case_id}: {data['total_chunks']} total chunks")

    items_to_process = (
        {selected_item: checklist_items[selected_item]}
        if selected_item and selected_item in checklist_items
        else checklist_items
    )

    # -------------------------------------------------------------------
    # Check existing results: skip items that are already fully present
    # (merge_nested_dicts keeps existing values, so reprocessing them would
    # be wasted compute that gets discarded - delete the results file to redo)
    # -------------------------------------------------------------------
    saving_path = build_saving_path(extract_model, merge_model, file_name, enable_thinking)
    existing_data: Dict[str, Any] = {}
    existing_results: Dict[str, Any] = {}
    if saving_path.exists():
        loaded = load_json_tolerant(saving_path)
        if loaded is not None:
            existing_data = loaded
            existing_results = existing_data.get("results", {})
            print(f"Found existing results at {saving_path}, will merge new results (existing values win)")

    skipped_items = [
        item_name for item_name in items_to_process
        if existing_results and all(
            case_id in existing_results and item_name in existing_results[case_id]
            for case_id in case_data
        )
    ]
    remaining_items = {k: v for k, v in items_to_process.items() if k not in skipped_items}
    if skipped_items:
        print(f"\nSkipping {len(skipped_items)} items already complete in the existing results file "
              f"(delete {saving_path} to redo them):")
        for item_name in skipped_items:
            print(f"  - {item_name}")

    # -------------------------------------------------------------------
    # Resolve per-item resume status from checkpoints
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Resolving resume status for {len(remaining_items)} items...")
    print(f"{'='*60}")
    statuses: Dict[str, Tuple[str, Optional[Dict]]] = {}
    for item_name in remaining_items:
        status, payload = resolve_item_status(item_name, case_data, extract_model, merge_model, file_name, enable_thinking)
        statuses[item_name] = (status, payload)
        print(f"  {item_name}: {status}")

    needs_extract = [i for i, (s, _) in statuses.items() if s == "needs_extract"]
    extract_payloads = {i: p for i, (s, p) in statuses.items() if s == "extracted"}
    merging_payloads = {i: p for i, (s, p) in statuses.items() if s == "merging"}
    done_payloads = {i: p for i, (s, p) in statuses.items() if s == "done"}

    # -------------------------------------------------------------------
    # Stage 1: extraction (GPT-OSS) - loaded only if some item needs it
    # -------------------------------------------------------------------
    if needs_extract:
        print(f"\n{'#'*60}")
        print(f"# STAGE 1: EXTRACT - {len(needs_extract)} items with {extract_model}")
        print(f"{'#'*60}")
        extract_tokenizer = tokenizer_for(extract_model)
        new_payloads = run_extraction_stage(
            needs_extract, remaining_items, case_data, extract_model,
            enable_thinking, extract_tokenizer, file_name, merge_model,
        )
        extract_payloads.update(new_payloads)
        # Model swap boundary: free the extraction model before loading the
        # merge model. Skip when they are the SAME model (e.g. the
        # Qwen3-30B-extract baseline) - the merge stage then reuses the already
        # loaded instance instead of paying for an unload + identical reload.
        if extract_model != merge_model:
            print("\nExtraction stage complete - releasing the extraction model before loading the merge model...")
            clear_llm_cache()
        else:
            print("\nExtraction stage complete - extract and merge models are identical; keeping it loaded.")
    else:
        print(f"\nStage 1 (extract) fully checkpointed - the extraction model will not be loaded.")

    # -------------------------------------------------------------------
    # Stage 2: binary-tree merge (Qwen)
    # -------------------------------------------------------------------
    items_state: Dict[str, Dict] = {}
    for item_name, payload in merging_payloads.items():
        items_state[item_name] = {
            "cases": payload["cases"],
            "stats": ensure_merge_stats_keys(payload.get("item_case_stats", {}), case_data),
            "level": payload.get("level", 0),
        }
    for item_name, payload in extract_payloads.items():
        items_state[item_name] = {
            "cases": init_merge_lists(payload["cases"]),
            "stats": fresh_merge_stats(case_data),
            "level": 0,
        }

    qwen_tokenizer = None
    if items_state:
        print(f"\n{'#'*60}")
        print(f"# STAGE 2: MERGE - {len(items_state)} items with {merge_model}")
        print(f"{'#'*60}")
        qwen_tokenizer = tokenizer_for(merge_model)
        run_merge_stage(items_state, case_data, merge_model, enable_thinking,
                        qwen_tokenizer, file_name, extract_model, remaining_items)

    # -------------------------------------------------------------------
    # Stage 3: final verification / pruning (Qwen)
    # -------------------------------------------------------------------
    prune_payloads: Dict[str, Dict] = dict(done_payloads)
    if items_state:
        print(f"\n{'#'*60}")
        print(f"# STAGE 3: PRUNE - {len(items_state)} items with {merge_model}")
        print(f"{'#'*60}")
        new_prune = run_prune_stage(items_state, case_data, merge_model, enable_thinking,
                                    qwen_tokenizer, file_name, extract_model, remaining_items)
        prune_payloads.update(new_prune)

    # -------------------------------------------------------------------
    # Assemble results + token stats
    # -------------------------------------------------------------------
    results: Dict[str, Dict[str, Dict]] = {}
    for case_id in case_data:
        results[case_id] = {}
        for item_name in remaining_items:
            results[case_id][item_name] = prune_payloads[item_name]["final_states"][case_id]

    by_stage_item_case: Dict[str, Dict[str, Dict[str, Dict]]] = {"extract": {}, "merge": {}, "prune": {}}
    for item_name in remaining_items:
        # Extract stats: from this run or from the (still-on-disk) checkpoint
        if item_name in extract_payloads:
            by_stage_item_case["extract"][item_name] = extract_payloads[item_name]["item_case_stats"]
        else:
            ckpt = load_json_tolerant(get_ckpt_path("extract", item_name, extract_model, merge_model, file_name, enable_thinking))
            if isinstance(ckpt, dict) and isinstance(ckpt.get("item_case_stats"), dict):
                by_stage_item_case["extract"][item_name] = ckpt["item_case_stats"]
            else:
                print(f"  Note: no extract stats available for {item_name} (reported as zeros)")
                by_stage_item_case["extract"][item_name] = {}
        # Merge stats
        if item_name in items_state:
            by_stage_item_case["merge"][item_name] = items_state[item_name]["stats"]
        else:
            ckpt = load_json_tolerant(get_ckpt_path("merge", item_name, extract_model, merge_model, file_name, enable_thinking))
            if isinstance(ckpt, dict) and isinstance(ckpt.get("item_case_stats"), dict):
                by_stage_item_case["merge"][item_name] = ckpt["item_case_stats"]
            else:
                by_stage_item_case["merge"][item_name] = {}
        # Prune stats
        by_stage_item_case["prune"][item_name] = prune_payloads[item_name].get("item_case_stats", {})

    # Preserve existing stats for items not reprocessed in this run
    existing_bsic = existing_data.get("token_stats", {}).get("by_stage_item_case", {})
    for stage, items in existing_bsic.items():
        if stage not in by_stage_item_case:
            by_stage_item_case[stage] = {}
        for item_name, cases in items.items():
            if item_name not in by_stage_item_case[stage] or not by_stage_item_case[stage][item_name]:
                by_stage_item_case[stage][item_name] = cases

    token_stats = derive_stats_views(by_stage_item_case, extract_model, merge_model)

    # -------------------------------------------------------------------
    # Merge with existing results and save (atomic), then delete checkpoints
    # -------------------------------------------------------------------
    if existing_results:
        results = merge_nested_dicts(existing_results, results)
        print("Merged with existing results")

    final_output = {
        "meta_data": {
            "file_name": file_name,
            "method": "chunk_by_chunk_hierarchical_merging",
            "inference_model": combined_model_dir(extract_model, merge_model),
            "extract_model_name": extract_model,
            "merge_model_name": merge_model,
            "checklist_item": selected_item if selected_item else "all",
            "chunk_by_chunk": True,
            "hierarchical_merging": True,
            "enable_thinking": enable_thinking,
            "merge_max_tokens": MERGE_MAX_TOKENS,
            "prune_max_tokens": PRUNE_MAX_TOKENS,
            **nyc_timestamps(),
        },
        "token_stats": token_stats,
        "results": results,
    }

    save_json_atomic(saving_path, final_output, indent=4)
    print(f"\nSaved results → {saving_path}")

    # Delete checkpoints only now that the results file is safely on disk; if
    # the job dies any earlier, the next run still resumes from the checkpoints.
    for item_name in list(remaining_items) + skipped_items:
        for kind in ("extract", "merge", "prune"):
            ckpt_path = get_ckpt_path(kind, item_name, extract_model, merge_model, file_name, enable_thinking)
            if ckpt_path.exists():
                ckpt_path.unlink()
    print("Cleaned up checkpoints")

    # Print token usage summary
    overall = token_stats["overall"]
    print(f"\n{'='*60}")
    print("Token Usage Summary:")
    print(f"{'='*60}")
    print(f"Total input tokens:   {overall['total_input_tokens']:,}")
    print(f"Total output tokens:  {overall['total_output_tokens']:,}")
    print(f"Total prompts:        {overall['total_prompts']:,}")
    for stage in ("extract", "merge", "prune"):
        if stage in token_stats["by_stage"]:
            s = token_stats["by_stage"][stage]
            print(f"  {stage:>8}: {s['total_prompts']:,} prompts, "
                  f"in={s['total_input_tokens']:,}, out={s['total_output_tokens']:,}")
    print(f"LLM merges:           {overall['n_llm_merges']:,}")
    print(f"Concat fallbacks:     {overall['concat_fallbacks']:,}")
    print(f"Oversize skips:       {overall['oversize_merge_skips'] + overall['oversize_prune_skips']:,}")
    print(f"Length truncations:   {overall['length_truncations']:,}")
    print(f"Parse failures:       {overall['parse_failures']}")
    print(f"{'='*60}")

    # Final cleanup to ensure all resources are freed
    clear_llm_cache()


if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    main()
