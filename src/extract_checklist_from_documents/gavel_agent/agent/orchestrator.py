"""
Orchestrator for the legal agent system.
Makes decisions about which tool to call next based on the current state.
"""

import json
import re
import yaml
import pytz
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from state.schemas import Snapshot, OrchestratorAction
from agent.llm_client import VLLMClient
from agent.tools.base import BaseTool
from agent.snapshot_formatter import SnapshotFormatter
from agent.tokenizer import TokenizerWrapper


class Orchestrator:
    """
    The decision-making component that analyzes snapshots and chooses actions.
    Uses an LLM to select which tool to call next or when to stop.
    """
    
    def __init__(
        self,
        llm_client: Optional[VLLMClient] = None,
        tools: Optional[List[BaseTool]] = None,
        config_dir: str = "config",
        verbose: bool = True,
        max_retries: int = 2,
        tokenizer: Optional[TokenizerWrapper] = None,
        raw_responses_path: Optional[str] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: VLLM client instance (creates default if None)
            tools: List of available tools
            config_dir: Directory containing config files
            verbose: Whether to print prompts and responses
            max_retries: Maximum retries when parsing fails
            tokenizer: TokenizerWrapper instance for counting tokens
            raw_responses_path: Optional path to JSONL file for recording raw LLM responses
        """
        self.config_dir = Path(config_dir)
        self.verbose = verbose if isinstance(verbose, int) else (1 if verbose else 0)
        self.max_retries = max_retries
        self.tokenizer = tokenizer
        
        # Determine model type - will be updated after we get the actual model name
        self.use_harmony_format = False
        self.use_qwen3_format = False
        self.model_name = None
        
        # Load configurations (will detect model type if llm_client provided)
        if llm_client and hasattr(llm_client, 'model_name'):
            self.model_name = llm_client.model_name
            self.use_harmony_format = self._is_gpt_oss_model(self.model_name)
            self.use_qwen3_format = self._is_qwen3_model(self.model_name)
        
        self._load_configs()
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            # Create client with orchestrator settings
            # Note: This path is rarely used since Driver now always passes an llm_client
            model_config = self.model_config.get("orchestrator", {})
            # Model name must be provided via command line; default to Qwen3-8B if somehow missing
            model_name = model_config.get("model_name", "Qwen/Qwen3-8B")
            
            # Update model name and format flags
            self.model_name = model_name
            self.use_harmony_format = self._is_gpt_oss_model(model_name)
            self.use_qwen3_format = self._is_qwen3_model(model_name)
            
            # Reload configs if we just determined the model type
            self._load_configs()
            
            self.llm_client = VLLMClient(
                model_name=model_name,
                temperature=model_config.get("temperature", 0.3),
                top_p=model_config.get("top_p", 1.0),
                top_k=model_config.get("top_k", -1),
                max_tokens=model_config.get("max_tokens", 1024)
            )
            
            # Create tokenizer if not provided
            if not self.tokenizer:
                self.tokenizer = TokenizerWrapper(model_name)
        
        # If we have an LLM client but no tokenizer, create one from the client's model
        if llm_client and not self.tokenizer:
            if hasattr(llm_client, 'model_name'):
                self.tokenizer = TokenizerWrapper(llm_client.model_name)
        
        # Store tools
        self.tools = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool
        
        # Track if system prompt has been printed (for verbose modes)
        self._system_prompt_printed = False

        # Initialize system prompt token count (will be calculated once)
        self.system_prompt_tokens = None

        # Store raw responses path for logging LLM outputs
        self.raw_responses_path = raw_responses_path
    
    def reset_for_new_run(self):
        """Reset internal state for a new run."""
        self._system_prompt_printed = False
        self.system_prompt_tokens = None  # Reset to recalculate for new run
    
    def _prepare_harmony_prompts(self):
        """
        Prepare the developer prompt for Harmony format (without special tokens).
        The llm_client will handle the actual formatting with the chat template.
        
        Returns:
            The complete developer prompt with tool descriptions included
        """
        # Replace the {{TOOL_DESCRIPTIONS}} placeholder with actual tool descriptions
        developer_content = self.developer_prompt
        if "{{TOOL_DESCRIPTIONS}}" in developer_content and self.tool_descriptions:
            developer_content = developer_content.replace("{{TOOL_DESCRIPTIONS}}", self.tool_descriptions)
        elif "{{TOOL_DESCRIPTIONS}}" in developer_content:
            # Remove placeholder if no tool descriptions
            developer_content = developer_content.replace("{{TOOL_DESCRIPTIONS}}", "")
        
        return developer_content
    
    def _is_gpt_oss_model(self, model_name: str) -> bool:
        """Check if the model is a GPT-OSS model that uses Harmony format."""
        if not model_name:
            return False
        model_lower = model_name.lower()
        return 'gpt-oss' in model_lower or 'gptoss' in model_lower
    
    def _is_qwen3_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen3 model that uses native tool calling."""
        if not model_name:
            return False
        # Case-insensitive check for Qwen3
        return 'qwen3' in model_name.lower()
    
    def _load_configs(self):
        """Load configuration files based on model type."""
        # Determine which prompts file to load
        if self.use_harmony_format:
            prompts_file = self.config_dir / "prompts_gpt_oss.yaml"
            if self.verbose >= 1:
                print(f"[ORCHESTRATOR] Loading GPT-OSS Harmony format prompts from {prompts_file}")
        elif self.use_qwen3_format:
            prompts_file = self.config_dir / "prompts_qwen.yaml"
            if self.verbose >= 1:
                print(f"[ORCHESTRATOR] Loading Qwen3 format prompts from {prompts_file}")
        else:
            prompts_file = self.config_dir / "prompts.yaml"
        
        # Load prompts based on model type
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts = yaml.safe_load(f)
                
                if self.use_harmony_format:
                    # For GPT-OSS models using Harmony format
                    # System message is auto-generated by the chat template
                    self.developer_prompt = prompts.get("developer_prompt", "")
                    self.tool_descriptions = prompts.get("tool_descriptions", "")
                    # For compatibility, set system_prompt to developer_prompt for token counting
                    self.system_prompt = self.developer_prompt
                    self.qwen3_tools = None
                elif self.use_qwen3_format:
                    # For Qwen3 models using native tool calling
                    self.system_prompt = prompts.get("system_prompt", "")
                    self.tool_descriptions = ""  # Not used for Qwen3
                    self.developer_prompt = None
                    # Load tool definitions for Qwen3
                    self.qwen3_tools = prompts.get("tool_definitions", [])
                else:
                    # Standard format for other models
                    self.system_prompt = prompts.get("system_prompt", "")
                    self.tool_descriptions = prompts.get("tool_descriptions", "")
                    self.developer_prompt = None
                    self.qwen3_tools = None
        else:
            # Default fallback
            self.system_prompt = "You are an orchestrator for extracting checklist items."
            self.tool_descriptions = ""
            self.developer_prompt = None
            self.qwen3_tools = None
        
        # Load model config
        model_config_file = self.config_dir / "model_config.yaml"
        if model_config_file.exists():
            with open(model_config_file, 'r') as f:
                model_data = yaml.safe_load(f)
                
                # Select model-specific config if we know the model type
                if self.model_name:
                    models_config = model_data.get('models', {})
                    
                    if self.use_qwen3_format:
                        specific_config = models_config.get('qwen3', models_config.get('default', {}))
                    elif self.use_harmony_format:
                        specific_config = models_config.get('gpt-oss', models_config.get('default', {}))
                    else:
                        specific_config = models_config.get('default', {})
                    
                    # Store as orchestrator config for compatibility
                    self.model_config = {"orchestrator": specific_config}
                else:
                    # No model name yet, use the whole config
                    self.model_config = model_data
        else:
            self.model_config = {"orchestrator": {}}
    
    def choose_action(
        self,
        snapshot: Snapshot,
        retry_count: int = 0
    ) -> OrchestratorAction:
        """
        Choose the next action based on the current snapshot.
        
        Args:
            snapshot: Current state snapshot
            retry_count: Current retry attempt
            
        Returns:
            OrchestratorAction with tool/args or stop decision
        """
        # Use unified prompting for all turns - no special first turn
        # The snapshot formatter will include all necessary information
        user_prompt = SnapshotFormatter.format_as_markdown(snapshot)
        
        # Prepare prompts based on format type
        if self.use_harmony_format:
            # For GPT-OSS models using Harmony format
            # Prepare developer prompt with tools included
            system_prompt = self._prepare_harmony_prompts()
            # The auto-generated system message is approximately 200 tokens
            # We'll use developer prompt + estimated system tokens for counting
            token_count_text = system_prompt + "\n[System message ~200 tokens]"
        elif self.use_qwen3_format:
            # For Qwen3 models - system prompt without tool descriptions
            # (tools are passed separately via the tools parameter)
            system_prompt = self.system_prompt
            token_count_text = system_prompt
        else:
            # Standard format - build the full system prompt from YAML components
            system_prompt = self.system_prompt
            if self.tool_descriptions:
                system_prompt += "\n\n" + self.tool_descriptions
            token_count_text = system_prompt
        
        # Calculate system prompt tokens once (first time it's built)
        if self.system_prompt_tokens is None and self.tokenizer:
            self.system_prompt_tokens = self.tokenizer.count_tokens(token_count_text)
            if self.verbose >= 1:
                print(f"[ORCHESTRATOR] System prompt tokens (calculated once): {self.system_prompt_tokens}")
        
        # Get current step number from snapshot
        current_step = snapshot.run_header.step if snapshot.run_header else 0
        
        # Log the prompt based on verbose level and step number
        if self.verbose >= 2:  # Debug mode - show full prompts always
            print("\n" + "="*80)
            # Only print system/developer prompt on first call
            if not self._system_prompt_printed:
                prompt_type = "Developer" if self.use_harmony_format else "System"
                print(f"[ORCHESTRATOR DEBUG] Full {prompt_type} Prompt (shown once):")
                print("-"*80)
                print(system_prompt)
                self._system_prompt_printed = True
            else:
                print("[ORCHESTRATOR DEBUG] System/Developer prompt unchanged (not shown)")
            print("\n[ORCHESTRATOR DEBUG] Full User Prompt (snapshot):")
            print("-"*80)
            print(user_prompt)
        elif self.verbose >= 1:  # Normal mode with special step-based printing
            # Special printing for step 1 and every 10 steps
            if current_step == 1 or (current_step > 0 and current_step % 10 == 0):
                print("\n" + "="*80)
                
                # Print system/developer prompt only at step 1
                if current_step == 1 and not self._system_prompt_printed:
                    prompt_type = "Developer" if self.use_harmony_format else "System"
                    print(f"[ORCHESTRATOR] Step {current_step} - Full {prompt_type} Prompt (shown once):")
                    print("-"*80)
                    print(system_prompt)
                    print("\n" + "="*80)
                    self._system_prompt_printed = True
                
                # Print full user prompt at step 1 and every 10 steps
                print(f"[ORCHESTRATOR] Step {current_step} - Full User Prompt:")
                print("-"*80)
                print(user_prompt)
                print("="*80)
            else:
                # For other steps, print concise summary
                snapshot_dict = snapshot.dict()
                checklist = snapshot_dict.get("checklist", [])
                extracted_count = sum(1 for item in checklist if item.get("extracted"))
                
                # Print a one-line summary for normal steps
                print(f"[ORCHESTRATOR] Step {current_step} - Progress: {extracted_count}/{len(checklist)} keys extracted")
        
        # Generate response
        # Initialize stats variable (will be set if available)
        stats = None

        # Always include raw response if we're recording to file, regardless of verbose level
        # Otherwise, include based on verbose settings for console logging
        should_include_raw = (
            self.raw_responses_path is not None or  # Always get raw if recording
            self.verbose >= 2 or
            (self.verbose >= 1 and (current_step == 1 or (current_step > 0 and current_step % 10 == 0)))
        )
        
        # Build generation kwargs
        generate_kwargs = {
            "prompt": user_prompt,
            "response_format": "json",
            "return_raw": should_include_raw,  # Return raw for verbose modes at appropriate steps
            "include_stats": True  # Request token usage statistics
        }
        
        # Pass system prompt (which is developer prompt for GPT-OSS)
        generate_kwargs["system"] = system_prompt
        
        # For Qwen3 models, pass the tool definitions
        if self.use_qwen3_format and self.qwen3_tools:
            generate_kwargs["tools"] = self.qwen3_tools
        
        # Generate response - with strict=False, this should always return a dict
        # Either a parsed JSON dict or a fallback dict with parse_error
        response = self.llm_client.generate(**generate_kwargs)
        
        # Extract stats if present
        if isinstance(response, dict) and '__stats__' in response:
            stats = response.pop('__stats__')
        
        # Log the raw response based on verbose level
        if isinstance(response, dict) and '__raw_response__' in response:
            raw = response.get('__raw_response__', '')

            # verbose=2: Always show raw output
            if self.verbose >= 2:
                print("\n[ORCHESTRATOR DEBUG] Raw Model Output:")
                print("="*80)
                print(raw)  # Print full output without truncation
                print("="*80)
            # verbose=1: Show raw output at step 1 and every 10 steps
            elif self.verbose >= 1 and (current_step == 1 or (current_step > 0 and current_step % 10 == 0)):
                print(f"\n[ORCHESTRATOR] Step {current_step} - Model Response:")
                print("-"*80)
                print(raw)  # Print full output without truncation
                print("="*80)

            # Remove raw from response for processing
            response = {k: v for k, v in response.items() if k != '__raw_response__'}

            # Save raw response for recording (will be used after action is parsed)
            raw_to_record = raw if self.raw_responses_path else None
        else:
            raw_to_record = None
        
        # Log the parsed response if verbose
        if self.verbose >= 2:
            print("\n[ORCHESTRATOR] Parsed Response:")
            print("-"*80)
            if isinstance(response, dict):
                print(json.dumps(response, indent=2, default=str))
            else:
                print(str(response)[:1000] + "..." if len(str(response)) > 1000 else str(response))
        
        # Parse the response - it should always be a dict from llm_client
        if isinstance(response, dict):
            # Check if it's a parse error or fallback
            if "parse_error" in response or "_fallback" in response:
                if self.verbose >= 1:
                    print("\n[ORCHESTRATOR] Parse error detected:")
                    if "_attempted_function" in response:
                        print(f"  - Failed to parse arguments for function: {response.get('_attempted_function')}")
                    if "_raw_args" in response:
                        print(f"  - Raw arguments (truncated): {response.get('_raw_args', '')[:200]}")
                    if "parse_error" in response:
                        print(f"  - Error: {response.get('parse_error')}")
                
                # Don't try to parse - just set action to None to trigger retry
                action = None
            else:
                # Valid response, parse it
                action = self._parse_json_response(response)
        else:
            # This should not happen - log error and set action to None
            if self.verbose >= 1:
                print(f"\n[ORCHESTRATOR] Unexpected non-dict response: {type(response)}")
            action = None
        
        # Check if parsing failed and we should retry
        if action is None or (isinstance(response, dict) and 
                             ("_json_decode_error" in response or "_fallback" in response)):
            if retry_count < self.max_retries:
                if self.verbose >= 1:
                    print(f"\n[ORCHESTRATOR] Parse failure detected. Retrying ({retry_count + 1}/{self.max_retries})...\n")
                
                # Retry with same prompt
                return self.choose_action(snapshot, retry_count + 1)
            else:
                # Max retries reached, return parse error action for transparency
                error_msg = "Failed to parse model response after multiple attempts"
                if isinstance(response, dict):
                    if "_attempted_function" in response:
                        error_msg = f"Failed to parse arguments for function: {response.get('_attempted_function')}"
                    elif "parse_error" in response:
                        error_msg = response.get("parse_error", error_msg)
                
                if self.verbose >= 1:
                    print(f"\n[ORCHESTRATOR] Max retries reached. Recording parse error: {error_msg}\n")
                
                # Return a parse_error action that will be recorded in history
                # This lets the model see the error and decide what to do next
                return OrchestratorAction(
                    tool="parse_error",
                    args={"error": error_msg, "retry_count": retry_count},
                    parse_failed=True  # Flag to indicate this is a parse failure
                )
        
        # Validate action if we got one
        if action:
            action = self._validate_action(action)
            
        # Handle validation failures
        if action is None:
            # Validation failed, return error action
            return OrchestratorAction(
                tool="validation_error", 
                args={"error": "Tool validation failed"},
                parse_failed=True
            )
        
        # Log the parsed action if verbose
        if self.verbose >= 2:
            print("\n[ORCHESTRATOR] Parsed Action:")
            print("-"*80)
            if action.decision == "stop":
                print(f"Decision: STOP")
                print(f"Reason: {action.reason}")
            else:
                print(f"Tool: {action.tool}")
                print(f"Args: {json.dumps(action.args, indent=2, default=str)}")
            print("="*80 + "\n")

        # Record raw response to file if configured
        if self.raw_responses_path and raw_to_record:
            run_id = snapshot.run_header.run_id if snapshot.run_header else "unknown"
            self._record_raw_response(
                step=current_step,
                run_id=run_id,
                raw_response=raw_to_record,
                action=action,
                retry_count=retry_count,
                stats=stats  # Will be None if not yet attached to action
            )

        # Attach stats to action if available
        if stats:
            action.stats = stats
            # Add system prompt tokens if calculated
            if self.system_prompt_tokens is not None:
                action.stats['system_prompt_tokens'] = self.system_prompt_tokens

        return action
    
    
    def _parse_json_response(self, response: Dict[str, Any]) -> OrchestratorAction:
        """
        Parse JSON response into OrchestratorAction.
        
        Args:
            response: JSON response dict
            
        Returns:
            Parsed OrchestratorAction
        """
        # Check for stop decision
        if response.get("decision") == "stop":
            return OrchestratorAction(
                decision="stop",
                reason=response.get("reason", "Task completed"),
                remaining_empty_keys=response.get("remaining_empty_keys", response.get("remaining_unresolved", []))
            )
        
        # Parse tool call
        tool_name = response.get("tool")
        if not tool_name:
            # Try alternative keys
            tool_name = response.get("action") or response.get("function")
        
        if tool_name:
            args = response.get("args", {})
            # Also check alternative keys
            if not args:
                args = response.get("arguments", {}) or response.get("parameters", {})
            
            return OrchestratorAction(
                tool=tool_name,
                args=args
            )
        
        # Couldn't parse, return None to trigger retry
        return None
    
    def _validate_action(self, action: OrchestratorAction) -> OrchestratorAction:
        """
        Validate and fix action if needed.
        
        Args:
            action: Action to validate
            
        Returns:
            Valid OrchestratorAction
        """
        # If it's a stop decision, it's valid
        if action.decision == "stop":
            return action
        
        # Check if tool exists
        if action.tool and action.tool in self.tools:
            # Validate args match tool schema
            # (simplified - in production, validate against tool schema)
            return action
        
        # Tool doesn't exist or is invalid
        # Return None to indicate validation failure
        if self.verbose >= 1:
            print(f"\n[ORCHESTRATOR] Tool '{action.tool}' not recognized or invalid.\n")
        return None
    
    def _record_raw_response(
        self,
        step: int,
        run_id: str,
        raw_response: str,
        action: OrchestratorAction,
        retry_count: int = 0,
        stats: Optional[Dict] = None
    ):
        """
        Record raw model response to JSONL file for analysis and debugging.

        This creates an append-only log of all LLM outputs, including:
        - The raw text generated by the model
        - The parsed action that resulted
        - Whether parsing succeeded
        - Token usage statistics
        - Retry information for failed parse attempts

        Args:
            step: Current step number
            run_id: Current run ID
            raw_response: Raw text response from LLM
            action: Parsed OrchestratorAction object
            retry_count: Number of retries for this response (0 for first attempt)
            stats: Optional token usage statistics dict
        """
        if not self.raw_responses_path:
            return  # No-op if path not configured

        try:
            # Build record with all relevant information
            record = {
                "step": step,
                "run_id": run_id,
                "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
                "raw_response": raw_response,
                "parsed_action": action.dict() if hasattr(action, 'dict') else str(action),
                "parse_success": not (hasattr(action, 'parse_failed') and action.parse_failed),
                "retry_count": retry_count,
                "stats": stats or {}
            }

            # Append to JSONL file (one JSON object per line)
            with open(self.raw_responses_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + '\n')

        except Exception as e:
            # Graceful degradation - log error but don't crash the agent
            if self.verbose >= 1:
                print(f"[ORCHESTRATOR] Warning: Failed to record raw response to {self.raw_responses_path}: {e}")

    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from dictionary recursively.

        Args:
            d: Dictionary to clean

        Returns:
            Cleaned dictionary
        """
        if not isinstance(d, dict):
            return d

        cleaned = {}
        for key, value in d.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned[key] = self._clean_dict(value)
                elif isinstance(value, list):
                    cleaned[key] = [
                        self._clean_dict(item) if isinstance(item, dict) else item
                        for item in value if item is not None
                    ]
                else:
                    cleaned[key] = value

        return cleaned