"""
LLM Client for local VLLM inference.
Based on the vllm_inference.py pattern for non-server mode operation.

GPT-OSS Harmony Format Special Token IDs (o200k_harmony encoding):
- <|start|>: 200006 - Beginning of a message
- <|end|>: 200007 - End of a message  
- <|message|>: 200008 - Transition from header to content
- <|channel|>: 200005 - Channel information
- <|constrain|>: 200003 - Data type definition in tool call
- <|return|>: 200002 - Model done sampling (stop token)
- <|call|>: 200012 - Model wants to call a tool (stop token)
"""

import json
import gc
import contextlib
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ray
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)


class ResponseParser(ABC):
    """Abstract base class for response parsers."""
    
    @abstractmethod
    def parse(self, response: str, **kwargs) -> Any:
        """Parse the response into the desired format."""
        pass
    
    @abstractmethod
    def get_system_prompt_addon(self) -> str:
        """Get additional text to add to system prompt for this parser."""
        pass


class TextParser(ResponseParser):
    """Default parser that returns text as-is."""
    
    def parse(self, response: str, **kwargs) -> str:
        """Return the response text as-is."""
        return response
    
    def get_system_prompt_addon(self) -> str:
        """No additional system prompt needed for text."""
        return ""


class JSONParser(ResponseParser):
    """Parser for JSON responses with robust error handling. Always returns a dict."""
    
    def __init__(self):
        """Initialize JSON parser."""
        pass
    
    def parse(self, response: str, **kwargs) -> Dict[str, Any]:
        """
        Parse JSON from response with multiple fallback strategies.
        Always returns a dict - either parsed JSON or fallback with error.
        
        Args:
            response: Raw response text
            **kwargs: Additional options (unused, kept for compatibility)
            
        Returns:
            Dict with parsed JSON or fallback with parse_error
        """
        # Step 1: Clean up the response
        cleaned = response.strip()
        
        # Remove thinking tags if present (Qwen3 format)
        if "<think>" in cleaned and "</think>" in cleaned:
            think_end = cleaned.find("</think>")
            if think_end != -1:
                cleaned = cleaned[think_end + 8:].strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Step 2: Try to parse the entire cleaned text as JSON first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Step 3: Try to extract JSON using brace counting (handles nested structures)
        json_str = self._extract_json_by_braces(cleaned)
        if json_str:
            try:
                parsed = json.loads(json_str)
                # Verify it's a valid action
                if "tool" in parsed or "decision" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Step 4: Fallback - try to find JSON in the original response
        json_str = self._extract_json_by_braces(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if "tool" in parsed or "decision" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Step 5: Last resort - try simple regex patterns
        # This is kept as a fallback for simple cases
        simple_patterns = [
            r'\{"tool"[^}]*"args"[^}]*\}',  # Simple tool call
            r'\{"decision"[^}]*"reason"[^}]*\}',  # Simple stop decision
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    return parsed
                except:
                    continue
        
        # If all else fails, return a fallback dictionary with the raw text
        return {
            "text": response,
            "parse_error": "Could not extract valid JSON",
            "_fallback": True
        }
    
    def _extract_json_by_braces(self, text: str) -> Optional[str]:
        """
        Extract JSON by counting braces to handle nested structures.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON string or None
        """
        # Find the first opening brace
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            # Handle string literals
            if not escape_next:
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif char == '\\' and in_string:
                    escape_next = True
                    continue
            else:
                escape_next = False
                continue
            
            # Count braces only outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        return text[start:i+1]
        
        return None
    
    def get_system_prompt_addon(self) -> str:
        """Add JSON instruction to system prompt."""
        return "\n\nYou must respond with valid JSON only. No additional text."


class GPTOSSJSONParser(JSONParser):
    """
    Parser for GPT-OSS model JSON responses with channel-based output format.
    Extends JSONParser to handle GPT-OSS specific output structure.
    
    Note: The model stops generation on <|return|> or <|call|> tokens.
    These are handled as stop tokens in SamplingParams.
    """
    
    def parse(self, response: str, **kwargs) -> Dict[str, Any]:
        """
        Parse JSON from GPT-OSS channel-based response format.
        
        GPT-OSS output formats:
        1. Function call: <|channel|>commentary to=functions.func_name <|constrain|>json<|message|>{args}
        2. Regular response: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>{json}
        
        Args:
            response: Raw response text with GPT-OSS formatting
            **kwargs: Additional options
            
        Returns:
            Parsed JSON dict (either function call or regular response)
        """
        # Check if special tokens are present
        has_channel_tokens = '<|channel|>' in response
        has_message_tokens = '<|message|>' in response
        
        # Detect if special tokens were stripped (common patterns without tokens)
        looks_stripped = ('analysisWe' in response or 'assistantfinal' in response or 
                         'assistantcommentary' in response or 'commentaryanalysis' in response)
        
        if not has_channel_tokens and not has_message_tokens and looks_stripped:
            raise ValueError(
                "GPT-OSS special tokens appear to be missing from the model output. "
                "This likely means skip_special_tokens=True was used during generation. "
                f"Raw output starts with: {response[:100]}..."
            )
        
        # Check if this is a function call (commentary channel with to=functions.)
        if '<|channel|>commentary to=functions.' in response:
            return self._extract_function_call(response)
        
        # Otherwise, try to extract regular response from final channel
        cleaned = self._extract_final_channel(response)
        
        # If no final channel found, try analysis channel
        if not cleaned:
            cleaned = self._extract_analysis_channel(response)
        
        # If still no content, use the full response
        if not cleaned:
            cleaned = response.strip()
        
        # Now use the parent class's parsing logic on the extracted content
        # We need to temporarily set the response to the cleaned version
        # and call the parent's parse method
        return super().parse(cleaned, **kwargs)
    
    def _extract_final_channel(self, text: str) -> str:
        """
        Extract content from the final channel.
        
        Note: vLLM stops at <|return|> or <|call|> without including them,
        so we just extract everything after <|channel|>final<|message|>
        
        Args:
            text: Full GPT-OSS response
            
        Returns:
            Content from final channel or empty string
        """
        # Find the final channel marker and extract everything after it
        final_marker = '<|channel|>final<|message|>'
        final_index = text.rfind(final_marker)  # Use rfind to get the last occurrence
        if final_index != -1:
            # Extract everything after the marker
            content = text[final_index + len(final_marker):].strip()
            # Remove <|end|> if present (from earlier messages in multi-turn)
            if content.endswith('<|end|>'):
                content = content[:-7].strip()
            return content
        return ""
    
    def _extract_analysis_channel(self, text: str) -> str:
        """
        Extract content from the analysis channel (fallback if no final channel).
        
        Note: vLLM stops at <|return|> or <|call|> without including them.
        This is a fallback for cases where model puts JSON directly in analysis channel.
        
        Args:
            text: Full GPT-OSS response
            
        Returns:
            Content from analysis channel or empty string
        """
        # Find the last analysis channel and check if it contains JSON
        analysis_marker = '<|channel|>analysis<|message|>'
        analysis_index = text.rfind(analysis_marker)
        if analysis_index != -1:
            # Extract content after the marker
            content = text[analysis_index + len(analysis_marker):].strip()
            
            # Remove <|end|> if present
            if '<|end|>' in content:
                content = content[:content.index('<|end|>')].strip()
            
            # Only return if it looks like JSON (starts with { or [)
            if content and (content[0] in '{['):
                return content
        return ""
    
    def _extract_function_call(self, text: str) -> Dict[str, Any]:
        """
        Extract function call from commentary channel.
        
        Format: <|channel|>commentary to=functions.func_name <|constrain|>json<|message|>{args}
        
        Args:
            text: Full GPT-OSS response containing function call
            
        Returns:
            Dict with tool and parameters keys, or fallback dict on parse failure
        """
        # Pattern to extract function name and arguments
        # Note: vLLM stops at <|call|> without including it
        # Captures: function name (including underscores) and JSON arguments
        func_pattern = r'<\|channel\|>commentary to=functions\.([\w_]+)\s*<\|constrain\|>json<\|message\|>(.*?)$'
        func_match = re.search(func_pattern, text, re.DOTALL)
        
        if func_match:
            function_name = func_match.group(1)
            json_args = func_match.group(2).strip()
            
            try:
                # Parse the JSON arguments
                parameters = json.loads(json_args)
            except json.JSONDecodeError as e:
                # Try to extract JSON using brace counting as fallback
                try:
                    json_str = self._extract_json_by_braces(json_args)
                    if json_str:
                        parameters = json.loads(json_str)
                    else:
                        parameters = {}
                except (json.JSONDecodeError, Exception) as fallback_error:
                    # If all parsing attempts fail, return a fallback dict
                    # This allows the orchestrator to detect the failure and retry
                    return {
                        "text": text,
                        "parse_error": f"Failed to parse function arguments: {str(e)}",
                        "_fallback": True,
                        "_attempted_function": function_name,
                        "_raw_args": json_args[:500]  # Include partial args for debugging
                    }
            
            # Return in the format expected by orchestrator
            return {
                "tool": function_name,
                "parameters": parameters
            }
        
        # If we can't extract the function call pattern, return fallback
        # (instead of raising an error which would bypass retry mechanism)
        return {
            "text": text,
            "parse_error": "Could not extract function call pattern from GPT-OSS response",
            "_fallback": True
        }


class Qwen3JSONParser(JSONParser):
    """
    Parser for Qwen3 model JSON responses with tool calling format.
    Extends JSONParser to handle Qwen3 specific output structure.
    
    Qwen3 uses XML-like tags for tool calls:
    - <think>...</think> for reasoning (should be stripped)
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call> for function calls
    """
    
    def parse(self, response: str, **kwargs) -> Dict[str, Any]:
        """
        Parse JSON from Qwen3 response format.
        
        Qwen3 output formats:
        1. Tool call with tags: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
        2. Tool call without tags: {"name": "function_name", "arguments": {...}}
        3. Regular JSON response: {"decision": "stop", "reason": "..."}
        
        Args:
            response: Raw response text with Qwen3 formatting
            **kwargs: Additional options
            
        Returns:
            Parsed JSON dict (either tool call or regular response)
        """
        # Step 1: Remove thinking tags if present
        # Note: <think> is added to the user prompt, so model only generates content + </think>
        cleaned = response.strip()
        if "</think>" in cleaned:
            # Extract content after </think> tag
            import re
            # Find the closing think tag and get everything after it
            think_end_index = cleaned.find("</think>")
            if think_end_index != -1:
                cleaned = cleaned[think_end_index + len("</think>"):].strip()
            else:
                # Fallback: try regex approach
                cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        
        # Step 2: Check if this is a tool call
        # First check for explicit <tool_call> tag
        if "<tool_call>" in cleaned:
            return self._extract_tool_call(cleaned)
        
        # Step 3: Try to detect tool call without tags by checking JSON structure
        # Tool calls have "name" and "arguments", stop decisions have "decision"
        try:
            # Try to parse the JSON to check its structure
            test_parse = json.loads(cleaned)
            if isinstance(test_parse, dict) and "name" in test_parse:
                # This looks like a tool call without tags
                return self._extract_tool_call(cleaned)
        except (json.JSONDecodeError, Exception):
            # If we can't parse it, let the parent class handle it
            pass
        
        # Step 4: Otherwise, use parent class for stop decisions and other JSON
        return super().parse(cleaned, **kwargs)
    
    def _extract_tool_call(self, text: str) -> Dict[str, Any]:
        """
        Extract tool call from Qwen3 format.
        
        Handles two formats:
        1. With tags: <tool_call>{"name": "func_name", "arguments": {...}}
        2. Without tags: {"name": "func_name", "arguments": {...}}
        
        Note: vLLM stops before outputting closing tags like </tool_call>
        
        Args:
            text: Response containing tool call
            
        Returns:
            Dict with tool and args keys, or fallback dict on parse failure
        """
        import re
        
        # Step 1: Check if there's a <tool_call> tag and extract content
        json_content = text.strip()
        
        if "<tool_call>" in text:
            # Extract content after <tool_call> tag
            # Note: vLLM stops at </tool_call> without including it
            tool_pattern = r'<tool_call>(.*?)$'  # Match everything after <tool_call> to end of string
            tool_match = re.search(tool_pattern, text, re.DOTALL)
            
            if tool_match:
                json_content = tool_match.group(1).strip()
            else:
                # Fallback: try to get everything after <tool_call>
                split_text = text.split("<tool_call>", 1)
                if len(split_text) > 1:
                    json_content = split_text[1].strip()
        
        # Step 2: Try to parse the tool call JSON
        try:
            # First attempt: direct JSON parsing
            tool_data = json.loads(json_content)
            
            # Verify it's a tool call structure
            if "name" in tool_data:
                function_name = tool_data.get("name")
                arguments = tool_data.get("arguments", {})
                
                if not function_name:
                    # Fallback if name is empty
                    return {
                        "text": text,
                        "parse_error": "Tool call has empty 'name' field",
                        "_fallback": True,
                        "_raw_tool_data": tool_data
                    }
                
                # Return in the format expected by orchestrator
                return {
                    "tool": function_name,
                    "args": arguments
                }
            else:
                # Not a tool call structure
                return {
                    "text": text,
                    "parse_error": "JSON does not have tool call structure (missing 'name' field)",
                    "_fallback": True,
                    "_raw_data": tool_data
                }
            
        except json.JSONDecodeError as e:
            # Try to extract JSON using brace counting as fallback
            try:
                json_str = self._extract_json_by_braces(json_content)
                if json_str:
                    tool_data = json.loads(json_str)
                    
                    # Check for tool call structure
                    if "name" in tool_data:
                        function_name = tool_data.get("name")
                        arguments = tool_data.get("arguments", {})
                        
                        if function_name:
                            return {
                                "tool": function_name,
                                "args": arguments
                            }
                    
            except (json.JSONDecodeError, Exception):
                pass
            
            # If all parsing attempts fail, return a fallback dict
            return {
                "text": text,
                "parse_error": f"Failed to parse tool call JSON: {str(e)}",
                "_fallback": True,
                "_raw_content": json_content[:500] if len(json_content) > 500 else json_content
            }


class VLLMClient:
    """
    Client for local VLLM inference without running a server.
    Creates and destroys LLM instances per call for efficient GPU memory management.
    """
    
    # Parser registry for easy lookup
    PARSERS: Dict[str, Type[ResponseParser]] = {
        "text": TextParser,
        "json": JSONParser,
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 1024,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: Optional[int] = None,
        persistent: bool = True,
    ):
        """
        Initialize the VLLM client.
        
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
            temperature: Sampling temperature for generation (default: 0.6)
            top_p: Top-p sampling parameter (default: 1.0)
            top_k: Top-k sampling parameter, -1 or 0 means no filtering (default: -1)
            max_tokens: Maximum tokens to generate
            dtype: Data type for model weights
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs for tensor parallelism (auto-detect if None)
            persistent: Keep model loaded in memory across calls (default: True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size or (torch.cuda.device_count() or 1)
        self.persistent = persistent
        
        # Initialize tokenizer once (lightweight, can keep in memory)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left' if 'Qwen' in model_name else 'right'
        )
        
        # Check if model supports thinking mode
        self.supports_thinking = 'qwen3' in model_name.lower() or 'gpt-oss' in model_name.lower()
        
        # LLM instance (created lazily)
        self._llm = None
        
        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        }
    
    @classmethod
    def register_parser(cls, name: str, parser_class: Type[ResponseParser]):
        """
        Register a custom parser.
        
        Args:
            name: Name to register the parser under
            parser_class: Parser class (must inherit from ResponseParser)
        """
        cls.PARSERS[name] = parser_class
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "text",
        tools: Optional[List[Dict[str, Any]]] = None,
        **format_kwargs
    ) -> Union[str, Dict[str, Any], Any]:
        """
        Generate text using VLLM with local inference.
        
        Args:
            prompt: User prompt text
            system: System prompt for standard models, or developer prompt for GPT-OSS models
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            response_format: Output format - "text" (default) or "json"
            tools: Optional list of tool definitions for Qwen3 models
            **format_kwargs: Additional arguments for the parser
                For JSON format:
                    - return_raw: Include raw response in result (default: False)
            
        Returns:
            Parsed response based on the format specified
            
        Examples:
            # Generate plain text
            text = client.generate("What is 2+2?")
            
            # Generate and parse JSON
            result = client.generate(
                "List 3 colors as JSON array",
                response_format="json"
            )
            
            # Generate JSON with tools (Qwen3)
            result = client.generate(
                prompt="Return a tool call",
                response_format="json",
                tools=[{...}]
            )
        """
        # Check model type
        is_gpt_oss = "gpt-oss" in self.model_name.lower()
        is_qwen3 = "qwen3" in self.model_name.lower()
        
        # Get the parser class based on response format and model type
        if response_format == "json":
            if is_gpt_oss:
                parser_class = GPTOSSJSONParser
            elif is_qwen3:
                parser_class = Qwen3JSONParser
            else:
                parser_class = JSONParser
            parser = parser_class()  # All parsers now always return dict, no strict mode
        else:
            parser_class = self.PARSERS.get(response_format)
            if not parser_class:
                raise ValueError(f"Unknown response format: {response_format}. Available: {list(self.PARSERS.keys())}")
            parser = parser_class()
        
        # Update system prompt if parser needs it (for non-GPT-OSS and non-Qwen3 models)
        if not is_gpt_oss and not is_qwen3:
            parser_addon = parser.get_system_prompt_addon()
            if system and parser_addon:
                if parser_addon not in system:
                    system = f"{system}{parser_addon}"
            elif not system and parser_addon:
                system = parser_addon.strip()
        
        # Build messages for chat template
        messages = []
        
        if is_gpt_oss:
            # For GPT-OSS models using Harmony format
            # The template will automatically generate the system message
            # The 'system' parameter contains the developer prompt content
            if system:
                messages.append({"role": "developer", "content": system})
        else:
            # For standard models (including Qwen3), use system message normally
            if system:
                messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template with appropriate parameters
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }
        
        if is_gpt_oss:
            # For GPT-OSS models, pass additional parameters for system message generation
            # The template will build the system message with these parameters
            template_kwargs["reasoning_effort"] = "medium"  # High reasoning for extraction tasks
            # Pass tools as empty list - this makes 'if tools is defined' true (triggers channel message)
            # but 'if tools' false (doesn't try to render tool definitions)
            template_kwargs["tools"] = []  # Empty list triggers the tools channel message
        elif is_qwen3 and tools:
            # For Qwen3 models, pass the tools list to enable tool calling mode
            template_kwargs["tools"] = tools
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            **template_kwargs
        )

        # print("#############################################")
        # print(f"[VLLMClient] Formatted prompt: {formatted_prompt}")
        # print(f"[VLLMClient] Model name: {self.model_name}")
        # print("#############################################")
        
        # Create sampling parameters
        # Build sampling parameters
        sampling_kwargs = {
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": max_tokens or self.max_tokens,
            "skip_special_tokens": False,  # Keep special tokens for GPT-OSS and Qwen3
        }
        
        # Add stop token IDs based on model type
        if is_gpt_oss:
            # Stop on <|return|> (token ID 200002) or <|call|> (token ID 200012)
            sampling_kwargs["stop_token_ids"] = [200002, 200012]  # <|return|> and <|call|>
        elif is_qwen3:
            # Stop on </tool_call> (151658), <|im_end|> (151645), <|endoftext|> (151643)
            sampling_kwargs["stop_token_ids"] = [151658, 151645, 151643]
        
        sampling_params = SamplingParams(**sampling_kwargs)
        
        # Get or create LLM instance
        if self.persistent:
            if self._llm is None:
                self._llm = self._create_llm()
                print(f"[VLLMClient] Model loaded: {self.model_name}")
            llm = self._llm
        else:
            # Non-persistent mode: create and cleanup each time
            llm = self._create_llm()

        try:
            # Generate
            outputs = llm.generate([formatted_prompt], sampling_params)
            
            # Extract result
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # Update statistics
            self.stats["total_calls"] += 1
            if hasattr(output, 'prompt_token_ids'):
                prompt_tokens = len(output.prompt_token_ids)
                self.stats["total_prompt_tokens"] += prompt_tokens
            else:
                prompt_tokens = 0
                
            if hasattr(output.outputs[0], 'token_ids'):
                completion_tokens = len(output.outputs[0].token_ids)
                self.stats["total_completion_tokens"] += completion_tokens
            else:
                completion_tokens = 0
            
            # Parse the response using the selected parser
            parsed_result = parser.parse(generated_text, **format_kwargs)
            
            # Add raw response if requested (for JSON format)
            if response_format == "json" and format_kwargs.get('return_raw', False):
                if isinstance(parsed_result, dict):
                    parsed_result['__raw_response__'] = generated_text
            
            # Add stats if requested
            if format_kwargs.get('include_stats', False) and isinstance(parsed_result, dict):
                parsed_result['__stats__'] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": self.model_name
                }
            
            return parsed_result
                
        finally:
            # Only cleanup if not persistent
            if not self.persistent:
                self._cleanup_llm(llm)
    
    # generate_batch has been removed as it was unused
    # All generation now goes through the generate() method above
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about LLM usage.
        
        Returns:
            Dictionary with usage statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        }
    
    def _create_llm(self) -> LLM:
        """
        Create LLM instance following vllm_inference.py pattern.
        
        Returns:
            Configured LLM instance
        """
        # Check for special model configurations
        is_qwen = "Qwen" in self.model_name
        is_gpt_oss = "gpt-oss" in self.model_name.lower()

        max_model_len = 131072 if is_gpt_oss else 250000
        
        # Build kwargs
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": is_qwen or is_gpt_oss,
            "max_model_len": max_model_len,  # Set max model length to 128K to fit in available GPU memory
        }
        
        # Add download directory if set
        import os
        if "HF_HOME" in os.environ:
            llm_kwargs["download_dir"] = os.environ["HF_HOME"]
        
        # Handle special cases
        if is_gpt_oss and "bf16" not in self.model_name.lower():
            # GPT-OSS specific settings from vllm_inference.py
            llm_kwargs["quantization"] = None
            llm_kwargs["hf_overrides"] = {"quantization_config": None}
        
        return LLM(**llm_kwargs)
    
    def cleanup(self) -> None:
        """
        Manually cleanup the persistent model.
        Call this when done with the client.
        """
        if self._llm is not None:
            print(f"[VLLMClient] Cleaning up model: {self.model_name}")
            self._cleanup_llm(self._llm)
            self._llm = None
    
    def __del__(self):
        """
        Cleanup on garbage collection.
        """
        if self.persistent and self._llm is not None:
            try:
                self.cleanup()
            except:
                pass
    
    def _cleanup_llm(self, llm: LLM) -> None:
        """
        Clean up GPU memory after inference.
        Follows the exact pattern from vllm_inference.py.
        
        Args:
            llm: LLM instance to clean up
        """
        # Destroy model parallel
        destroy_model_parallel()
        destroy_distributed_environment()
        
        # Try to shutdown engine_core (vLLM v1)
        try:
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
                llm.llm_engine.engine_core.shutdown()
            elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
                # Fallback for older versions
                del llm.llm_engine.model_executor
        except Exception as e:
            # Ignore cleanup errors
            pass
        
        # Delete the LLM object
        del llm
        
        # Destroy process group
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        
        # Garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Shutdown Ray
        ray.shutdown()

