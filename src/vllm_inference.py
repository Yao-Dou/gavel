#!/usr/bin/env python3
"""
vLLM Batch Inference Script
============================

Runs batch inference on long-context prompts using vLLM with YaRN context extension.

Features:
- Supports Qwen3, GPT-OSS, and Gemma model families
- YaRN (Yet another RoPE extension) for long context handling
- Automatic bucketing by prompt length for optimal memory usage
- Thinking/reasoning mode support for Qwen3 and GPT-OSS models
- Resume capability for interrupted jobs
- Token usage statistics tracking

Usage:
    python vllm_inference.py \\
        --file_name "input_data" \\
        --folder_path "data/prompts" \\
        --model_name "Qwen/Qwen3-14B" \\
        --enable_thinking

Arguments:
    --file_name       Base name of input JSON file (without .json)
    --folder_path     Directory containing input JSON file
    --output_dir      Output directory for results (default: output)
    --model_name      HuggingFace model identifier
    --enable_thinking Enable thinking/reasoning mode (Qwen3, GPT-OSS only)

Input Format:
    {
        "keys": ["case_001", "case_002", ...],
        "contexts": [
            [{"role": "user", "content": "..."}],
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
            ...
        ]
    }

Output Format:
    {
        "meta_data": {"file_name": "...", "inference_model": "...", "enable_thinking": ...},
        "token_stats": {"total_input_tokens": ..., "total_output_tokens": ...},
        "results": {
            "case_001": {"thinking": "...", "answer": "...", "has_thinking": true},
            ...
        }
    }

Supported Models:
    - Qwen3: Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B, Qwen3-2507 variants
    - GPT-OSS: unsloth/gpt-oss-20b-BF16, openai/gpt-oss-120b
    - Gemma: google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference for long-context tasks using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--file_name",
        required=True,
        help="Base name of the JSON data file (without .json extension)"
    )
    parser.add_argument(
        "--folder_path",
        required=True,
        help="Directory containing the input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (supported by Qwen3 and GPT-OSS models)"
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="HuggingFace model identifier (default: Qwen/Qwen3-14B)"
    )
    return parser.parse_args()


# =============================================================================
# Thinking Parsers
# =============================================================================

class Qwen3ThinkingParser:
    """
    Parse Qwen3 model output to extract thinking and answer components.

    Qwen3 uses <think>...</think> tags to wrap reasoning content.

    Parsing Rules:
    - Complete <think>...</think> blocks: Extract all blocks as thinking,
      remaining text as answer
    - No <think> but </think> present: Text before </think> is thinking
    - <think> but no </think>: Text after <think> is thinking
    - No tags: Entire text is the answer

    Example:
        >>> parser = Qwen3ThinkingParser()
        >>> result = parser.parse_from_text("<think>Let me consider...</think>The answer is 42.")
        >>> result["thinking"]
        'Let me consider...'
        >>> result["answer"]
        'The answer is 42.'
    """

    def __init__(self, think_end_token_id: int = 151668):
        self.think_end_token_id = think_end_token_id

    def parse_from_text(self, text: str) -> Dict[str, str]:
        """Parse text to extract thinking and answer components."""
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        # Case 1: Complete <think>...</think> blocks
        blocks = list(re.finditer(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE))
        if blocks:
            thinking_parts = [(m.group(1) or "").strip() for m in blocks]
            thinking = "\n\n".join([p for p in thinking_parts if p])
            answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}

        # Case 2: No <think>, but </think> present
        close_pos = text.lower().find("</think>")
        if close_pos != -1:
            thinking = text[:close_pos].strip()
            answer = text[close_pos + len("</think>"):].strip()
            return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}

        # Case 3: <think> but no </think>
        open_pos = text.lower().find("<think>")
        if open_pos != -1:
            answer = text[:open_pos].strip()
            thinking = text[open_pos + len("<think>"):].strip()
            return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}

        # Case 4: No tags
        return {"thinking": "", "answer": text.strip(), "has_thinking": False}


class GPTOSSThinkingParser:
    """
    Parse GPT-OSS model output to extract thinking and answer components.

    GPT-OSS uses channel-based formatting:
        <|channel|>analysis<|message|>...<|end|>
        <|start|>assistant<|channel|>final<|message|>...<|return|>

    The "analysis" channel contains reasoning, "final" channel contains the answer.

    Note:
        vLLM stops at <|return|> or <|call|> without including them in output.
        Set skip_special_tokens=False in SamplingParams to preserve channel tokens.
    """

    def parse_from_text(self, text: str) -> Dict[str, str]:
        """Parse text to extract thinking and answer components."""
        thinking = ""
        answer = text.strip()

        has_channel_tokens = '<|channel|>' in text
        has_message_tokens = '<|message|>' in text

        # Detect if special tokens were stripped
        looks_stripped = any(pattern in text for pattern in [
            'analysisWe', 'assistantfinal', 'assistantcommentary',
            'commentaryanalysis', 'analysisThe', 'finalThe'
        ])

        if not has_channel_tokens and not has_message_tokens and looks_stripped:
            print("Warning: GPT-OSS special tokens appear to be missing. "
                  "Ensure skip_special_tokens=False in sampling params.")
            return {"thinking": "", "answer": text.strip(), "has_thinking": False}

        # Extract analysis channel (thinking/reasoning)
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)'
        analysis_match = re.search(analysis_pattern, text, re.DOTALL)
        if analysis_match:
            thinking = analysis_match.group(1).strip()

        # Extract final channel (answer)
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
        final_match = re.search(final_pattern, text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
        elif analysis_match and not final_match:
            # If only analysis exists and looks like JSON, treat as answer
            if thinking and (thinking.startswith('{') or thinking.startswith('[')):
                answer = thinking
                thinking = ""

        return {"thinking": thinking, "answer": answer, "has_thinking": bool(thinking)}


# =============================================================================
# YaRN Bucket Configuration
# =============================================================================
# YaRN (Yet another RoPE extension) enables models to handle longer contexts
# by scaling the rotary position embeddings. Different bucket sizes allow
# optimal memory utilization for varying prompt lengths.

BUCKETS_QWEN: Dict[str, dict] = {
    "short": {
        "max_prompt": 22_000,
        "max_model_len": 32_768,
        "hf_overrides": None
    },
    "medium": {
        "max_prompt": 56_000,
        "max_model_len": 65_536,
        "hf_overrides": {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2,
                "original_max_position_embeddings": 32_768
            },
            "max_model_len": 65_536,
        }
    },
    "long": {
        "max_prompt": 124_000,
        "max_model_len": 131_072,
        "hf_overrides": {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4,
                "original_max_position_embeddings": 32_768
            },
            "max_model_len": 131_072,
        }
    },
}

BUCKETS_QWEN_2507: Dict[str, dict] = {
    "base": {
        "max_prompt": 248_000,
        "max_model_len": 262_144,
        "hf_overrides": None
    },
}

BUCKETS_GENERIC: Dict[str, dict] = {
    "base": {
        "max_prompt": 125_000,
        "max_model_len": 131_072,
        "hf_overrides": None
    },
    "long": {
        "max_prompt": 248_000,
        "max_model_len": 262_144,
        "hf_overrides": {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2,
                "original_max_position_embeddings": 131_072
            },
            "max_model_len": 262_144,
        }
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def tokenizer_for(model_name: str) -> AutoTokenizer:
    """Load tokenizer for the specified model."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def bucketize(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    table: Dict[str, dict]
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Assign prompts to YaRN buckets based on token count.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer for counting tokens
        table: Bucket configuration dictionary

    Returns:
        Dictionary mapping bucket names to lists of (index, prompt) tuples

    Raises:
        ValueError: If a prompt exceeds the maximum supported length
    """
    buckets: Dict[str, List[Tuple[int, str]]] = {k: [] for k in table}
    for idx, prompt in enumerate(prompts):
        n_tok = len(tokenizer.encode(prompt))
        for bucket_name, cfg in table.items():
            if n_tok <= cfg["max_prompt"]:
                buckets[bucket_name].append((idx, prompt))
                break
        else:
            raise ValueError(
                f"Prompt at index {idx} has {n_tok} tokens "
                f"(exceeds maximum supported length)."
            )
    return buckets


def build_llm(
    model_name: str,
    cfg: dict,
    tp_size: int,
    rope_overrides: dict | None
) -> LLM:
    """
    Build vLLM LLM instance with appropriate configuration.

    Args:
        model_name: HuggingFace model identifier
        cfg: Bucket configuration
        tp_size: Tensor parallel size (number of GPUs)
        rope_overrides: RoPE scaling configuration for YaRN

    Returns:
        Configured LLM instance
    """
    is_gpt_oss = "gpt-oss" in model_name.lower()
    is_qwen = "Qwen" in model_name

    hf_overrides = rope_overrides or {}

    # Disable quantization for non-BF16 GPT-OSS models
    if is_gpt_oss and "bf16" not in model_name.lower():
        hf_overrides = {**hf_overrides, "quantization_config": None}

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tp_size,
        "download_dir": os.environ.get("HF_HOME"),
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "hf_overrides": hf_overrides if hf_overrides else None,
        "trust_remote_code": is_gpt_oss or is_qwen,
    }

    if is_gpt_oss and "bf16" not in model_name.lower():
        llm_kwargs["quantization"] = None

    return LLM(**llm_kwargs)


def sampling_params(model_name: str, enable_thinking: bool) -> SamplingParams:
    """
    Get sampling parameters appropriate for the model type.

    Args:
        model_name: HuggingFace model identifier
        enable_thinking: Whether thinking mode is enabled

    Returns:
        Configured SamplingParams instance
    """
    is_gpt_oss = "gpt-oss" in model_name.lower()

    if is_gpt_oss:
        return SamplingParams(
            temperature=0.7,
            top_p=1.0,
            max_tokens=50_000,
            skip_special_tokens=False,  # Preserve channel tokens for parsing
            stop_token_ids=[200002, 200012],  # <|return|> and <|call|>
        )
    elif "Qwen3" in model_name:
        return SamplingParams(
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            max_tokens=24_000,
        )
    return SamplingParams(temperature=0.7, top_p=1.0, max_tokens=8_000)


def cleanup_llm(llm: LLM) -> None:
    """Clean up LLM resources and free GPU memory."""
    destroy_model_parallel()
    destroy_distributed_environment()

    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
            llm.llm_engine.engine_core.shutdown()
        elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            del llm.llm_engine.model_executor
    except Exception as e:
        print(f"Warning: Could not shutdown engine: {e}")

    del llm

    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully freed GPU memory.")


# =============================================================================
# Generation Pipeline
# =============================================================================

def generate_all(
    prompts: List[str],
    model_name: str,
    enable_thinking: bool
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Generate responses for all prompts using appropriate YaRN buckets.

    Args:
        prompts: List of prompt strings
        model_name: HuggingFace model identifier
        enable_thinking: Whether thinking mode is enabled

    Returns:
        Tuple of (outputs list, token_stats dictionary)
    """
    # Select bucket configuration based on model
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

    total_input_tokens = 0
    total_output_tokens = 0

    for bucket_name in table:
        pairs = buckets[bucket_name]
        if not pairs:
            continue

        idxs, bucket_prompts = zip(*pairs)
        cfg = table[bucket_name]
        rope_overrides = cfg.get("hf_overrides")

        print(f"[{model_name}] Bucket '{bucket_name}': "
              f"{len(bucket_prompts)} prompt(s), max_len={cfg['max_model_len']}")

        llm = build_llm(model_name, cfg, tp_size, rope_overrides)
        outs = llm.generate(list(bucket_prompts), sparams)

        # Collect token statistics
        for out in outs:
            if hasattr(out, 'prompt_token_ids'):
                total_input_tokens += len(out.prompt_token_ids)
            if hasattr(out.outputs[0], 'token_ids'):
                total_output_tokens += len(out.outputs[0].token_ids)

        indexed_out.extend(zip(idxs, outs))
        cleanup_llm(llm)

    outputs = [o for _i, o in sorted(indexed_out, key=lambda t: t[0])]

    num_prompts = len(prompts)
    token_stats = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_prompts": num_prompts,
        "avg_input_tokens": total_input_tokens / num_prompts if num_prompts > 0 else 0,
        "avg_output_tokens": total_output_tokens / num_prompts if num_prompts > 0 else 0
    }

    return outputs, token_stats


# =============================================================================
# Result Merging
# =============================================================================

def merge_nested_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries, with dict1 values taking precedence.

    Recursively merges nested dictionaries.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key not in result:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge_nested_dicts(result[key], value)
    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for batch inference."""
    args = parse_args()
    file_name: str = args.file_name
    folder_path: str = args.folder_path
    output_dir: str = args.output_dir
    enable_thinking: bool = args.enable_thinking
    model_name: str = args.model_name

    print(f"Python executable: {sys.executable}")
    print(f"Configuration:")
    print(f"  file_name: {file_name}")
    print(f"  folder_path: {folder_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  model_name: {model_name}")
    print(f"  enable_thinking: {enable_thinking}")

    # Only Qwen3 and GPT-OSS models support thinking mode
    if "Qwen3" not in model_name and "gpt-oss" not in model_name.lower():
        if enable_thinking:
            print(f"Note: Thinking mode not supported for {model_name}, disabling.")
        enable_thinking = False

    # Load input data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_path = Path(folder_path) / f"{file_name}.json"

    print(f"Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_contexts: List[Any] = data["contexts"]
    keys: List[Any] = data["keys"]
    print(f"Loaded {len(keys)} cases")

    # Build output path
    model_save_name = Path(model_name).name
    output_path = Path(output_dir) / folder_path / model_save_name / \
                  f"{file_name}_thinking_{enable_thinking}.json"

    print(f"Output path: {output_path}")

    # Load existing results for resume capability
    existing_dict: Dict[str, Any] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_dict = json.load(f)
        print(f"Found existing results: {len(existing_dict.get('results', {}))} cases")

    # Filter out already processed prompts
    filtered_contexts: List[Any] = []
    filtered_keys: List[Any] = []

    if existing_dict:
        for i, key in enumerate(keys):
            if isinstance(key, str):
                if key not in existing_dict.get("results", {}):
                    filtered_contexts.append(full_contexts[i])
                    filtered_keys.append(key)
            elif isinstance(key, list) and len(key) == 2:
                res0 = existing_dict.get("results", {}).get(key[0], {})
                if key[1] not in res0:
                    filtered_contexts.append(full_contexts[i])
                    filtered_keys.append(key)
            else:
                raise ValueError(f"Unsupported key structure: {key}")
    else:
        filtered_contexts, filtered_keys = full_contexts, keys

    # Build chat prompts
    prompts: List[str] = []
    is_gpt_oss = "gpt-oss" in model_name.lower()

    for ctx in filtered_contexts:
        kwargs = {"tokenize": False, "add_generation_prompt": True}

        if is_gpt_oss:
            kwargs["reasoning_effort"] = "high" if enable_thinking else "medium"
        elif "Qwen3" in model_name and enable_thinking:
            kwargs["enable_thinking"] = True

        prompt = tokenizer.apply_chat_template(ctx, **kwargs)

        # Clean up GPT-OSS template artifacts
        if is_gpt_oss:
            tools_line = "\nCalls to these tools must go to the commentary channel: 'functions'."
            prompt = prompt.replace(tools_line, "")

        prompts.append(prompt)

    print(f"Prompts to process: {len(prompts)}")

    if not prompts:
        print("Nothing to do. All keys already processed.")
        sys.exit(0)

    # Generate responses
    outputs, token_stats = generate_all(prompts, model_name, enable_thinking)

    print(f"\n{'='*60}")
    print("Token Usage Statistics:")
    print(f"{'='*60}")
    print(f"Total input tokens:  {token_stats['total_input_tokens']:,}")
    print(f"Total output tokens: {token_stats['total_output_tokens']:,}")
    print(f"Number of prompts:   {token_stats['num_prompts']:,}")
    print(f"Avg input/prompt:    {token_stats['avg_input_tokens']:.1f}")
    print(f"Avg output/prompt:   {token_stats['avg_output_tokens']:.1f}")
    print(f"{'='*60}\n")

    # Parse outputs
    result_dict: Dict[str, Any] = {}

    if "Qwen3" in model_name:
        parser = Qwen3ThinkingParser()
    elif is_gpt_oss:
        parser = GPTOSSThinkingParser()
    else:
        parser = None

    for i, out in enumerate(outputs):
        key = filtered_keys[i]
        text = out.outputs[0].text if hasattr(out, "outputs") else str(out)
        if not text:
            continue

        parsed = parser.parse_from_text(text) if parser else text

        if isinstance(key, str):
            result_dict[key] = parsed
        elif isinstance(key, list) and len(key) == 2:
            result_dict.setdefault(key[0], {})[key[1]] = parsed
        else:
            raise ValueError(f"Unsupported key structure: {key}")

    # Merge with existing results
    if existing_dict:
        result_dict = merge_nested_dicts(existing_dict.get("results", {}), result_dict)
        print("Merged with existing results")

        if "token_stats" in existing_dict:
            old_stats = existing_dict["token_stats"]
            token_stats["total_input_tokens"] += old_stats.get("total_input_tokens", 0)
            token_stats["total_output_tokens"] += old_stats.get("total_output_tokens", 0)
            token_stats["num_prompts"] += old_stats.get("num_prompts", 0)
            if token_stats["num_prompts"] > 0:
                token_stats["avg_input_tokens"] = \
                    token_stats["total_input_tokens"] / token_stats["num_prompts"]
                token_stats["avg_output_tokens"] = \
                    token_stats["total_output_tokens"] / token_stats["num_prompts"]

    # Save results
    final_dict = {
        "meta_data": {
            "file_name": file_name,
            "inference_model": model_name,
            "enable_thinking": enable_thinking
        },
        "token_stats": token_stats,
        "results": result_dict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4)

    print(f"Saved {len(result_dict)} results to: {output_path}")


if __name__ == "__main__":
    main()
