#!/usr/bin/env python3
"""Chunk-by-chunk iterative extraction utility for legal documents.

This script processes legal documents chunk by chunk, iteratively building up
extracted information for each checklist item. It uses vLLM for inference on
open-source models with YaRN scaling for long contexts.

Command-line flags:
    --file_name          (str)  name of the JSON data file (without extension)
    --enable_thinking    (flag) switch on thinking mode (Qwen3 or GPT-OSS)
    --model_name         (str)  HF model name (default: "Qwen/Qwen3-14B")
    --checklist_item     (str)  specific checklist item to extract (optional)
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
from typing import Any, Dict, List, Tuple
import copy

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc

# Note: ray is imported by vLLM internally but we don't need to manage it directly

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk-by-chunk iterative extraction for legal documents.")
    parser.add_argument("--file_name", required=True, help="Base name of the JSON data file (without .json)")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (supported by Qwen-3 and GPT-OSS models)",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="HF model identifier (e.g. 'google/gemma-3-12b-it', 'Qwen/Qwen3-8B', 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--checklist_item",
        default=None,
        help="Specific checklist item to extract (if not specified, all items will be processed)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Thinking-parser helpers (from original vllm_inference.py)
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
# LLM Cache for reusing instances across chunks
# ---------------------------------------------------------------------------

# Global cache for LLM instances to avoid reinitializing when processing
# multiple chunks with the same configuration
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
# YaRN bucket configuration
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
# Helper functions
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


def sampling_params(model_name: str, enable_thinking: bool) -> SamplingParams:
    is_gpt_oss = "gpt-oss" in model_name.lower()
    
    if is_gpt_oss:
        # GPT-OSS specific parameters
        # Need to preserve special tokens and add stop tokens
        return SamplingParams(
            temperature=0.7,  # Lower temperature for more consistent extraction
            top_p=1.0,
            max_tokens=64_000,
            skip_special_tokens=False,  # CRITICAL: Keep special tokens for parsing
            stop_token_ids=[200002, 200012],  # Stop on <|return|> or <|call|>
        )
    elif "Qwen3" in model_name:
        return SamplingParams(
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            max_tokens=16_000,
        )
    return SamplingParams(temperature=0.7, top_p=1.0, max_tokens=16_000)


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


# ---------------------------------------------------------------------------
# Chunk-by-chunk generation pipeline
# ---------------------------------------------------------------------------

def generate_batch(prompts: List[str], model_name: str, enable_thinking: bool) -> Tuple[List[str], Dict[str, Any]]:
    """Generate responses for a batch of prompts using vLLM.
    
    Returns:
        Tuple of (responses, token_stats) where token_stats contains:
        - total_input_tokens: Total input tokens processed
        - total_output_tokens: Total output tokens generated
        - num_prompts: Number of prompts processed
        - avg_input_tokens: Average input tokens per prompt
        - avg_output_tokens: Average output tokens per prompt
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
    sparams = sampling_params(model_name, enable_thinking)

    indexed_out: List[Tuple[int, Any]] = []
    tp_size = torch.cuda.device_count() or 1
    
    # Initialize per-prompt token statistics
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
                "output_tokens": prompt_output_tokens
            })
        
        # Store outputs with their stats
        indexed_out.extend(zip(idxs, outs, bucket_stats))
        
        # Don't cleanup cached instances when skip_cleanup is True (base/short buckets)
        # The cache will be reused for the next chunk if it has the same configuration
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
    
    # Return both aggregated and per-prompt statistics
    num_prompts = len(prompts)
    token_stats = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_prompts": num_prompts,
        "avg_input_tokens": total_input_tokens / num_prompts if num_prompts > 0 else 0,
        "avg_output_tokens": total_output_tokens / num_prompts if num_prompts > 0 else 0,
        "per_prompt_stats": per_prompt_stats  # New: individual stats for each prompt
    }
    
    return results, token_stats


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from model response."""
    # Try to find JSON block within markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # Try to find raw JSON object
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to clean up common issues
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        try:
            return json.loads(json_str)
        except:
            return None




# ---------------------------------------------------------------------------
# Token statistics aggregation functions
# ---------------------------------------------------------------------------

def calculate_item_stats(item_case_token_stats: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Calculate aggregated statistics per checklist item from item-case stats.
    
    Args:
        item_case_token_stats: Dict of {item_name: {case_id: token_stats}}
    
    Returns:
        Dict of {item_name: aggregated_stats}
    """
    item_stats = {}
    
    for item_name, case_stats in item_case_token_stats.items():
        total_input = sum(stats.get("total_input_tokens", 0) for stats in case_stats.values())
        total_output = sum(stats.get("total_output_tokens", 0) for stats in case_stats.values())
        total_prompts = sum(stats.get("total_prompts", 0) for stats in case_stats.values())
        
        # Total chunks processed is the SUM across all cases for this item
        # Each item processes chunks for all cases, so we sum them
        total_chunks_processed = sum(
            stats.get("chunks_processed", 0) for stats in case_stats.values()
        )
        
        item_stats[item_name] = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_prompts": total_prompts,
            "total_chunks_processed": total_chunks_processed,  # Changed from actual_chunks_processed
            "avg_input_tokens_per_prompt": total_input / total_prompts if total_prompts > 0 else 0,
            "avg_output_tokens_per_prompt": total_output / total_prompts if total_prompts > 0 else 0,
        }
    
    return item_stats


def calculate_case_stats(item_case_token_stats: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Calculate aggregated statistics per case from item-case stats.
    
    Args:
        item_case_token_stats: Dict of {item_name: {case_id: token_stats}}
    
    Returns:
        Dict of {case_id: aggregated_stats}
    """
    case_stats = {}
    
    # Reorganize data by case
    for item_name, item_cases in item_case_token_stats.items():
        for case_id, stats in item_cases.items():
            if case_id not in case_stats:
                case_stats[case_id] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_prompts": 0,
                    "items_processed": 0,
                    "chunks_in_case": stats.get("chunks_in_case", 0)
                }
            
            case_stats[case_id]["total_input_tokens"] += stats.get("total_input_tokens", 0)
            case_stats[case_id]["total_output_tokens"] += stats.get("total_output_tokens", 0)
            case_stats[case_id]["total_prompts"] += stats.get("total_prompts", 0)
            case_stats[case_id]["items_processed"] += 1
    
    # Calculate averages
    for case_id, stats in case_stats.items():
        if stats["total_prompts"] > 0:
            stats["avg_input_tokens_per_prompt"] = stats["total_input_tokens"] / stats["total_prompts"]
            stats["avg_output_tokens_per_prompt"] = stats["total_output_tokens"] / stats["total_prompts"]
    
    return case_stats


# ---------------------------------------------------------------------------
# Checkpoint saving/loading functions
# ---------------------------------------------------------------------------

def get_checkpoint_path(model_name: str, file_name: str, enable_thinking: bool, item_name: str) -> Path:
    """Get the checkpoint file path for a specific checklist item."""
    model_save_name = Path(model_name).name
    states_dir = Path("states") / model_save_name
    checkpoint_name = f"{file_name}_thinking_{enable_thinking}_{item_name}.json"
    return states_dir / checkpoint_name


def save_checkpoint(case_states: Dict, chunk_idx: int, item_name: str, model_name: str, file_name: str, enable_thinking: bool, item_case_stats: Dict = None):
    """Save the current state as a checkpoint including item-specific case token stats."""
    checkpoint_path = get_checkpoint_path(model_name, file_name, enable_thinking, item_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get current time in NYC timezone
    nyc_time = datetime.now(ZoneInfo("America/New_York"))
    readable_timestamp = nyc_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    checkpoint_data = {
        "last_completed_chunk": chunk_idx,
        "case_states": case_states,
        "item_name": item_name,
        "timestamp": readable_timestamp,
        "timestamp_epoch": time.time(),
        "item_case_stats": item_case_stats or {}  # Save item-specific case stats
    }
    
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"    Checkpoint saved for {item_name} after chunk {chunk_idx + 1} at {readable_timestamp}")


def load_checkpoint(item_name: str, model_name: str, file_name: str, enable_thinking: bool) -> Tuple[Dict, int, Dict]:
    """Load checkpoint if it exists. Returns (case_states, last_completed_chunk_idx, item_case_stats)."""
    checkpoint_path = get_checkpoint_path(model_name, file_name, enable_thinking, item_name)
    
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        last_chunk = checkpoint_data.get("last_completed_chunk", -1)
        case_states = checkpoint_data.get("case_states", {})
        item_case_stats = checkpoint_data.get("item_case_stats", {})
        print(f"  Loaded checkpoint for {item_name}: completed up to chunk {last_chunk + 1}")
        return case_states, last_chunk, item_case_stats
    
    return {}, -1, {}  # Return empty item_case_stats as 3rd element


def cleanup_checkpoint(item_name: str, model_name: str, file_name: str, enable_thinking: bool):
    """Remove checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(model_name, file_name, enable_thinking, item_name)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Cleaned up checkpoint for {item_name}")


# ---------------------------------------------------------------------------
# Main chunk-by-chunk processing pipeline
# ---------------------------------------------------------------------------

def process_chunks_iteratively(
    keys: List[List[str]],
    chunks: List[List[str]],
    chunks_tokens: List[List[int]],
    checklist_items: Dict[str, str],
    model_name: str,
    enable_thinking: bool,
    tokenizer: AutoTokenizer,
    selected_item: str = None,
    file_name: str = None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict]]]:
    """Process all chunks iteratively for all checklist items in parallel.
    
    Returns:
        Tuple of (results, item_case_token_stats)
    """
    
    # Load prompt template
    template_path = Path("../../../../prompts/extract_checklist_item_from_docs/chunk_by_chunk_template.txt")
    with open(template_path, "r") as f:
        prompt_template = f.read()
    
    # Initialize results dictionary
    results = {}
    
    # Filter checklist items if a specific one is selected
    items_to_process = {selected_item: checklist_items[selected_item]} if selected_item and selected_item in checklist_items else checklist_items
    
    # Step 1: Reorganize data by case and flatten chunks
    print("Reorganizing data by case...")
    case_data = {}
    for doc_idx, (key, doc_chunks) in enumerate(zip(keys, chunks)):
        case_id, doc_name = key
        if case_id not in case_data:
            case_data[case_id] = {
                "flattened_chunks": [],
                "chunk_sources": [],  # Track which doc and chunk index
                "total_chunks": 0
            }
        # Add this document's chunks to the case's flattened list
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            case_data[case_id]["flattened_chunks"].append(chunk_text)
            case_data[case_id]["chunk_sources"].append((doc_name, chunk_idx, len(doc_chunks)))
        case_data[case_id]["total_chunks"] = len(case_data[case_id]["flattened_chunks"])
    
    print(f"Organized {len(case_data)} cases:")
    for case_id, data in case_data.items():
        print(f"  Case {case_id}: {data['total_chunks']} total chunks")
    
    # Find maximum chunks across all cases
    max_chunks = max(data["total_chunks"] for data in case_data.values())
    print(f"Maximum chunks to process: {max_chunks}")
    
    # Initialize or load states for all case-item combinations
    all_states = {}  # {case_id: {item_name: state}}
    item_case_token_stats = {}  # {item_name: {case_id: token_stats}} - Single source of truth for all stats
    
    # Load checkpoints for all items if they exist
    last_completed_chunk = -1
    all_items_have_same_checkpoint = True
    checkpoint_chunks = []
    
    for item_name in items_to_process:
        item_states, item_last_chunk, checkpoint_item_case_stats = load_checkpoint(item_name, model_name, file_name, enable_thinking)
        checkpoint_chunks.append(item_last_chunk)
        
        # Load item-specific case stats if available
        if checkpoint_item_case_stats:
            item_case_token_stats[item_name] = checkpoint_item_case_stats
            print(f"  Loaded case stats for {item_name}")
        
        if item_states:
            # Reorganize loaded states by case
            for state_key, state in item_states.items():
                case_id = state_key.split('_')[0]
                if case_id not in all_states:
                    all_states[case_id] = {}
                all_states[case_id][item_name] = state
            
            # Initialize item-case stats if not loaded from checkpoint
            if item_name not in item_case_token_stats:
                item_case_token_stats[item_name] = {}
        else:
            # Initialize empty item-case stats
            item_case_token_stats[item_name] = {}
    
    # Initialize item-case stats for all items and cases
    for item_name in items_to_process:
        if item_name not in item_case_token_stats:
            item_case_token_stats[item_name] = {}
        
        # Initialize stats for each case if not already present
        for case_id in case_data:
            if case_id not in item_case_token_stats[item_name]:
                item_case_token_stats[item_name][case_id] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_prompts": 0,
                    "chunks_in_case": case_data[case_id]["total_chunks"],
                    "chunks_processed": 0  # Track chunks processed per case
                }
    
    # Check if all items have the same checkpoint
    if checkpoint_chunks and len(set(checkpoint_chunks)) == 1:
        last_completed_chunk = checkpoint_chunks[0]
    else:
        # If items have different checkpoints, we need to start from the earliest
        valid_chunks = [c for c in checkpoint_chunks if c >= 0]
        if valid_chunks:
            last_completed_chunk = min(valid_chunks)
            if len(set(valid_chunks)) > 1:
                print(f"\n{'='*60}")
                print(f"WARNING: Items have different checkpoint positions:")
                for item_name, chunk_pos in zip(items_to_process.keys(), checkpoint_chunks):
                    if chunk_pos >= 0:
                        print(f"  {item_name}: completed up to chunk {chunk_pos + 1}")
                print(f"Starting from earliest position: chunk {last_completed_chunk + 1}")
                print(f"Some items will reprocess chunks (safe - states are replaced)")
                print(f"{'='*60}\n")
        else:
            last_completed_chunk = -1
    
    # Initialize states for any missing case-item combinations
    for case_id in case_data:
        if case_id not in all_states:
            all_states[case_id] = {}
        for item_name in items_to_process:
            if item_name not in all_states[case_id]:
                all_states[case_id][item_name] = {
                    "reasoning": "No information extracted yet.",
                    "extracted": []
                }
    
    # Check if already completed
    if last_completed_chunk >= max_chunks - 1:
        print(f"\n{'='*60}")
        print(f"Already completed all {max_chunks} chunks for all items")
        print(f"{'='*60}")
        # Store results and return
        for case_id in case_data:
            results[case_id] = {}
            for item_name in items_to_process:
                results[case_id][item_name] = all_states[case_id][item_name]
        return results, item_case_token_stats
    
    print(f"\n{'='*60}")
    print(f"Processing all {len(items_to_process)} checklist items in parallel")
    print(f"Starting from chunk {last_completed_chunk + 1}")
    print(f"{'='*60}")
        
    # Process chunk by chunk (global index across all cases)
    start_chunk = last_completed_chunk + 1
    for global_chunk_idx in range(start_chunk, max_chunks):
        print(f"\n{'='*50}")
        print(f"Processing global chunk {global_chunk_idx + 1}/{max_chunks}")
        print(f"{'='*50}")
        
        # Prepare prompts for ALL items × ALL cases that have this chunk
        batch_prompts = []
        batch_keys = []  # (case_id, item_name, doc_name, doc_chunk_idx, total_doc_chunks)
        
        # Count how many cases still have chunks to process
        active_cases = 0
        for case_id, data in case_data.items():
            if global_chunk_idx < data["total_chunks"]:
                active_cases += 1
        
        print(f"Active cases at this chunk: {active_cases}")
        
        # Generate prompts for all item-case combinations
        for item_name, item_description in items_to_process.items():
            for case_id, data in case_data.items():
                # Skip if this case has no more chunks
                if global_chunk_idx >= data["total_chunks"]:
                    continue
                
                # Get the chunk and its source info
                chunk_text = data["flattened_chunks"][global_chunk_idx]
                doc_name, doc_chunk_idx, total_doc_chunks = data["chunk_sources"][global_chunk_idx]
                
                # Get current state for this case-item combination
                current_state = all_states[case_id][item_name]
                
                # Only pass the extracted items to the prompt, not the reasoning
                # The reasoning should be fresh for each chunk
                current_state_for_prompt = {
                    "extracted": current_state.get("extracted", [])
                }
                
                # Format the prompt
                prompt = prompt_template.format(
                    item_description=item_description,
                    current_state=json.dumps(current_state_for_prompt, indent=2),
                    document_name=doc_name,
                    chunk_id=doc_chunk_idx + 1,
                    total_chunks=total_doc_chunks,
                    document_chunk=chunk_text
                )
                
                # Create chat format - all models just use user message
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                # Apply chat template with model-specific parameters
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
                    # Standard parameters for other models
                    kwargs = dict(tokenize=False, add_generation_prompt=True)
                    
                    if enable_thinking and "Qwen3" in model_name:
                        # Qwen3 thinking mode
                        kwargs["enable_thinking"] = True
                
                formatted_prompt = tokenizer.apply_chat_template(messages, **kwargs)
                
                # For GPT-OSS, remove the unwanted tools line that the template adds
                if is_gpt_oss:
                    # Remove the line about tools going to commentary channel
                    tools_line = "\nCalls to these tools must go to the commentary channel: 'functions'."
                    if tools_line in formatted_prompt:
                        formatted_prompt = formatted_prompt.replace(tools_line, "")
                        # print(f"  Removed unwanted tools line from GPT-OSS prompt")

                # print(f"Formatted prompt:")
                # print(formatted_prompt)
                # print("#############################################")
                # exit()
                
                batch_prompts.append(formatted_prompt)
                batch_keys.append((case_id, item_name, doc_name, doc_chunk_idx, total_doc_chunks))
            
        if not batch_prompts:
            print("  No more active cases at this chunk")
            break
        
        # Generate responses for this batch
        print(f"  Generating responses for {len(batch_prompts)} prompts ({active_cases} cases × {len(items_to_process)} items)...")
        responses, chunk_token_stats = generate_batch(batch_prompts, model_name, enable_thinking)
        
        # Get per-prompt stats from the chunk_token_stats
        per_prompt_stats = chunk_token_stats.get("per_prompt_stats", [])
        
        # Process each response with its actual token stats
        for idx, (response, (case_id, item_name, _, _, _)) in enumerate(zip(responses, batch_keys)):
            # Get actual token stats for this specific prompt
            if idx < len(per_prompt_stats):
                prompt_stats = per_prompt_stats[idx]
                input_tokens = prompt_stats["input_tokens"]
                output_tokens = prompt_stats["output_tokens"]
            else:
                # Fallback to averages if per-prompt stats not available
                input_tokens = chunk_token_stats["avg_input_tokens"]
                output_tokens = chunk_token_stats["avg_output_tokens"]
            
            # Update item-case specific stats (single source of truth)
            item_case_token_stats[item_name][case_id]["total_input_tokens"] += input_tokens
            item_case_token_stats[item_name][case_id]["total_output_tokens"] += output_tokens
            item_case_token_stats[item_name][case_id]["total_prompts"] += 1
        
        # Update chunks_processed count for each case that processed a chunk
        cases_in_batch = set((case_id, item_name) for _, (case_id, item_name, _, _, _) in zip(responses, batch_keys))
        for case_id, item_name in cases_in_batch:
            item_case_token_stats[item_name][case_id]["chunks_processed"] += 1
        
        # Calculate total cumulative stats from item_case_token_stats
        total_input = sum(
            stats["total_input_tokens"]
            for item_stats in item_case_token_stats.values()
            for stats in item_stats.values()
        )
        total_output = sum(
            stats["total_output_tokens"]
            for item_stats in item_case_token_stats.values()
            for stats in item_stats.values()
        )
        
        print(f"    Chunk token usage - Input: {chunk_token_stats['total_input_tokens']:,}, Output: {chunk_token_stats['total_output_tokens']:,}")
        print(f"    Cumulative totals - Input: {total_input:,.0f}, Output: {total_output:,.0f}")
        
        # Process responses and update states
        for response, (case_id, item_name, doc_name, doc_chunk_idx, total_doc_chunks) in zip(responses, batch_keys):
            # Extract JSON from response
            extracted_json = extract_json_from_response(response)
            
            if extracted_json:
                # Update state with model's output (model already handles merging)
                all_states[case_id][item_name] = extracted_json
            else:
                print(f"    Warning: Failed to extract JSON for case {case_id}, item {item_name}, doc {doc_name}, chunk {doc_chunk_idx + 1}")
                print(f"    Response: {response}")
        
        # Save checkpoints for all items after processing each global chunk
        print(f"  Saving checkpoints for all {len(items_to_process)} items...")
        for item_name in items_to_process:
            # Collect states for this item across all cases
            item_states = {}
            for case_id in case_data:
                state_key = f"{case_id}_{item_name}"
                item_states[state_key] = all_states[case_id][item_name]
            
            # Save checkpoint for this item with item-specific case stats
            save_checkpoint(item_states, global_chunk_idx, item_name, model_name, file_name, enable_thinking, 
                          item_case_token_stats[item_name])  # Pass item-specific case stats
    
    # After all chunks processed, clean up checkpoints and prepare results
    print(f"\n{'='*60}")
    print("All chunks processed. Cleaning up checkpoints and LLM cache...")
    print(f"{'='*60}")
    
    # Clear the LLM cache to free GPU memory
    clear_llm_cache()
    
    for item_name in items_to_process:
        cleanup_checkpoint(item_name, model_name, file_name, enable_thinking)
    
    # Store final states in results
    for case_id in case_data:
        results[case_id] = {}
        for item_name in items_to_process:
            results[case_id][item_name] = all_states[case_id][item_name]
    
    # Calculate averages for item-case stats
    for item_name in item_case_token_stats:
        for case_id in item_case_token_stats[item_name]:
            if item_case_token_stats[item_name][case_id]["total_prompts"] > 0:
                item_case_token_stats[item_name][case_id]["avg_input_tokens_per_prompt"] = (
                    item_case_token_stats[item_name][case_id]["total_input_tokens"] / 
                    item_case_token_stats[item_name][case_id]["total_prompts"]
                )
                item_case_token_stats[item_name][case_id]["avg_output_tokens_per_prompt"] = (
                    item_case_token_stats[item_name][case_id]["total_output_tokens"] / 
                    item_case_token_stats[item_name][case_id]["total_prompts"]
                )
    
    # Return results with item-case stats (single source of truth)
    return results, item_case_token_stats


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def build_saving_path(model_name: str, file_name: str, enable_thinking: bool) -> Path:
    model_save_name = Path(model_name).name
    base_dir = Path("results")
    return base_dir / model_save_name / f"{file_name}_thinking_{enable_thinking}.json"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test_llm_cleanup(model_name: str = "Qwen/Qwen3-14B"):
    """Test LLM initialization and cleanup to verify memory is properly freed.
    
    This function creates an LLM instance and immediately cleans it up to test
    if the cleanup function properly frees GPU memory.
    """
    print("\n" + "="*60)
    print("TESTING LLM CLEANUP")
    print("="*60)
    print(f"Model: {model_name}")
    print("This test will initialize an LLM and immediately clean it up.")
    print("Watch the GPU memory usage to verify cleanup works properly.\n")
    
    # Get initial GPU state
    initial_gpu_info = get_gpu_memory_info()
    if initial_gpu_info:
        print("Initial GPU memory state:")
        for gpu in initial_gpu_info:
            print(f"  GPU {gpu['index']}: {gpu['used_gb']:.2f}/{gpu['total_gb']:.2f} GB")
        print()
    
    # Build a minimal LLM instance
    print("Initializing LLM for cleanup test...")
    tp_size = torch.cuda.device_count() or 1
    
    # Use base configuration (no rope scaling) for fastest init
    test_llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        download_dir=os.environ.get("HF_HOME"),
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=1024,  # Use small context for quick test
    )
    
    print("LLM initialized. Now testing cleanup...")
    print()
    
    # Test the cleanup function
    cleanup_llm(test_llm, check_vram=True)
    
    print("\n" + "="*60)
    print("CLEANUP TEST COMPLETE")
    print("If GPU memory was properly freed, you should see most memory released.")
    print("You can now comment out this test if cleanup is working correctly.")
    print("="*60 + "\n")

def main() -> None:
    args = parse_args()
    file_name: str = args.file_name
    enable_thinking: bool = args.enable_thinking
    model_name: str = args.model_name
    selected_item: str = args.checklist_item

    # Only Qwen3 and GPT-OSS models support thinking mode
    if "Qwen3" not in model_name and "gpt-oss" not in model_name.lower():
        enable_thinking = False
    
    # UNCOMMENT TO TEST CLEANUP (then comment out after verifying it works)
    # test_llm_cleanup(model_name)
    # return

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print(f"Loading chunk data from {file_name}.json...")
    chunk_data_path = Path("data") / f"{file_name}.json"
    with open(chunk_data_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    
    keys = chunk_data["keys"]
    chunks = chunk_data["chunks"]
    chunks_tokens = chunk_data["chunks_tokens"]
    
    print(f"Loaded {len(keys)} documents with chunks")
    
    # Load checklist items
    checklist_path = Path("../../../../prompts/extract_checklist_item_from_docs/item_specific_info.json")
    with open(checklist_path, "r", encoding="utf-8") as f:
        checklist_items = json.load(f)
    
    print(f"Loaded {len(checklist_items)} checklist items")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # -------------------------------------------------------------------
    # Check for existing results
    # -------------------------------------------------------------------
    saving_path = build_saving_path(model_name, file_name, enable_thinking)
    existing_results = {}
    if saving_path.exists():
        with open(saving_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_results = existing_data.get("results", {})
        print(f"Found existing results, will merge new results")
    
    # -------------------------------------------------------------------
    # Process chunks iteratively
    # -------------------------------------------------------------------
    results, item_case_token_stats = process_chunks_iteratively(
        keys=keys,
        chunks=chunks,
        chunks_tokens=chunks_tokens,
        checklist_items=checklist_items,
        model_name=model_name,
        enable_thinking=enable_thinking,
        tokenizer=tokenizer,
        selected_item=selected_item,
        file_name=file_name
    )
    
    # -------------------------------------------------------------------
    # Merge with existing results if any
    # -------------------------------------------------------------------
    if existing_results:
        from utils import merge_nested_dicts
        results = merge_nested_dicts(existing_results, results)
        # Also merge item_case stats if they exist
        if "token_stats" in existing_data:
            # Merge by_item_case stats
            if "by_item_case" in existing_data["token_stats"]:
                existing_item_case_stats = existing_data["token_stats"]["by_item_case"]
                for item_name, case_stats in existing_item_case_stats.items():
                    if item_name not in item_case_token_stats:
                        item_case_token_stats[item_name] = case_stats
        print("Merged with existing results")
    
    # Calculate aggregated statistics from item_case_token_stats
    item_stats = calculate_item_stats(item_case_token_stats)
    case_stats = calculate_case_stats(item_case_token_stats)
    
    # Calculate overall token statistics
    total_input = sum(
        stats["total_input_tokens"]
        for item_cases in item_case_token_stats.values()
        for stats in item_cases.values()
    )
    total_output = sum(
        stats["total_output_tokens"]
        for item_cases in item_case_token_stats.values()
        for stats in item_cases.values()
    )
    total_prompts = sum(
        stats["total_prompts"]
        for item_cases in item_case_token_stats.values()
        for stats in item_cases.values()
    )
    
    # Calculate the number of global iterations (max chunks in any single case)
    # We process chunk-by-chunk globally until all cases are done
    global_iterations = 0
    for item_cases in item_case_token_stats.values():
        for case_id, stats in item_cases.items():
            # The global iterations equals the maximum chunks_in_case value
            global_iterations = max(global_iterations, stats.get("chunks_in_case", 0))
        break  # All items have the same cases, so we only need to check one item
    
    overall_stats = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_prompts": total_prompts,
        "global_iterations": global_iterations,  # Number of chunk-by-chunk iterations (max chunks in any case)
        "total_chunks_processed": sum(stats["total_chunks_processed"] for stats in item_stats.values()),  # Sum of all chunks across all items
        "items_processed": len(item_case_token_stats),
        "cases_processed": len(case_stats)
    }
    if overall_stats["total_prompts"] > 0:
        overall_stats["avg_input_tokens_per_prompt"] = overall_stats["total_input_tokens"] / overall_stats["total_prompts"]
        overall_stats["avg_output_tokens_per_prompt"] = overall_stats["total_output_tokens"] / overall_stats["total_prompts"]
    
    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    # Get current time in NYC timezone
    nyc_time = datetime.now(ZoneInfo("America/New_York"))
    readable_timestamp = nyc_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    final_output = {
        "meta_data": {
            "file_name": file_name,
            "inference_model": model_name,
            "checklist_item": selected_item if selected_item else "all",
            "chunk_by_chunk": True,
            "enable_thinking": enable_thinking,
            "timestamp": readable_timestamp,
            "timestamp_epoch": time.time()
        },
        "token_stats": {
            "overall": overall_stats,
            "by_item": item_stats,  # Calculated from item_case_token_stats
            "by_case": case_stats,  # Calculated from item_case_token_stats
            "by_item_case": item_case_token_stats  # The single source of truth
        },
        "results": results
    }
    
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(saving_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    print(f"\nSaved results → {saving_path}")
    
    # Print token usage summary
    print(f"\n{'='*60}")
    print("Token Usage Summary:")
    print(f"{'='*60}")
    print(f"Total input tokens:  {overall_stats['total_input_tokens']:,}")
    print(f"Total output tokens: {overall_stats['total_output_tokens']:,}")
    print(f"Total prompts:       {overall_stats['total_prompts']:,}")
    print(f"Global iterations:   {overall_stats['global_iterations']:,}")
    print(f"Total chunks:        {overall_stats.get('total_chunks_processed', 0):,}")
    print(f"Items processed:     {overall_stats['items_processed']}")
    if overall_stats.get('avg_input_tokens_per_prompt'):
        print(f"Avg input/prompt:    {overall_stats['avg_input_tokens_per_prompt']:.1f}")
        print(f"Avg output/prompt:   {overall_stats['avg_output_tokens_per_prompt']:.1f}")
    print(f"{'='*60}")
    
    # Final cleanup to ensure all resources are freed
    clear_llm_cache()


if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    main()