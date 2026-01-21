"""
Driver loop for the legal agent system.
Coordinates the orchestrator, tools, and state management.
"""

import json
import uuid
import yaml
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pytz
from pathlib import Path
import traceback

from state.store import ChecklistStore, Ledger
from state.schemas import OrchestratorAction
from agent.document_manager import DocumentManager
from agent.tokenizer import TokenizerWrapper
from agent.snapshot_builder import SnapshotBuilder
from agent.orchestrator import Orchestrator
from agent.llm_client import VLLMClient
from agent.logger import ActionLogger, PerformanceTracker
from agent.validator import StopValidator
from agent.stats_tracker import StatsTracker

# Import all tools
from agent.tools import (
    GetChecklistTool,
    UpdateChecklistTool,
    AppendChecklistTool,
    ListDocumentsTool,
    ReadDocumentTool,
    SearchDocumentRegexTool
)


class Driver:
    """
    Main execution loop that coordinates all components.
    Runs the agent from start to completion or max steps.
    """
    
    def __init__(
        self,
        corpus_path: str,
        store_path: str = "checklist_store.json",
        ledger_path: str = "ledger.jsonl",
        config_dir: str = "config",
        checklist_config_path: str = None,
        model_name: str = "Qwen/Qwen3-8B",
        max_steps: int = 100,
        verbose: bool = True,
        log_dir: str = "logs",
        recent_actions: int = 5
    ):
        """
        Initialize the driver with all components.
        
        Args:
            corpus_path: Path to document corpus
            store_path: Path for checklist store persistence
            ledger_path: Path for ledger persistence
            config_dir: Directory containing config files
            checklist_config_path: Path to specific checklist config file
            model_name: Model to use for orchestration
            max_steps: Maximum steps before stopping
            verbose: Whether to print progress
            log_dir: Directory for logging
            recent_actions: Number of recent actions to show with detailed results
        """
        self.corpus_path = Path(corpus_path)
        self.config_dir = Path(config_dir)
        self.checklist_config_path = checklist_config_path
        self.max_steps = max_steps
        self.verbose = verbose
        self.model_name = model_name  # Store model name for later use
        self.recent_actions = recent_actions  # Store for passing to SnapshotBuilder

        # Derive output directory early (before creating orchestrator)
        # This ensures consistent placement for all output files
        store_path_obj = Path(store_path)
        if store_path_obj.parent.name != ".":
            # Use the parent directory of store_path as the output directory
            self.output_dir = store_path_obj.parent
        else:
            # Fallback to old behavior if store_path is just a filename
            case_id = self.corpus_path.name if self.corpus_path.is_dir() else None
            model_suffix = model_name.split('/')[-1] if '/' in model_name else model_name
            self.output_dir = Path(f"output/{model_suffix}")
            if case_id:
                self.output_dir = self.output_dir / case_id

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Derive raw responses path for orchestrator
        raw_responses_path = self.output_dir / "raw_responses.jsonl"

        # Load configuration FIRST before creating store
        self._load_config()
        
        # Initialize state management with checklist configuration
        self.store = ChecklistStore(
            storage_path=store_path,
            checklist_config=self.checklist_config
        )
        self.ledger = Ledger(ledger_path)
        
        # Initialize document manager with tokenizer
        tokenizer = TokenizerWrapper(model_name)
        self.document_manager = DocumentManager(self.corpus_path, tokenizer)
        
        # Initialize snapshot builder with checklist configuration and user instruction
        self.snapshot_builder = SnapshotBuilder(
            self.store,
            self.ledger,
            self.document_manager,
            checklist_config=self.checklist_config,
            user_instruction=self.user_instruction,
            task_constraints=self.task_constraints,
            recent_actions_detail=self.recent_actions
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create LLM client with the specified model and orchestrator settings
        llm_client = VLLMClient(
            model_name=model_name,
            temperature=self.orchestrator_config.get('temperature', 0.3),
            top_p=self.orchestrator_config.get('top_p', 1.0),
            top_k=self.orchestrator_config.get('top_k', -1),
            max_tokens=self.orchestrator_config.get('max_tokens', 1024)
        )
        
        # Initialize orchestrator with tokenizer and LLM client
        self.orchestrator = Orchestrator(
            llm_client=llm_client,  # Pass the LLM client with correct model
            tools=list(self.tools.values()),
            config_dir=config_dir,
            verbose=verbose,
            tokenizer=tokenizer,  # Pass tokenizer for accurate system prompt token counting
            raw_responses_path=str(raw_responses_path)  # Pass path for recording raw LLM outputs
        )
        
        # Run tracking
        self.current_run_id = None
        self.current_step = 0
        self.last_tool_result = None
        self.last_tool_name = None  # Track the tool that produced last_tool_result
        self.action_history = []
        self.stop_count = 0  # Track stop attempts for two-stage stop mechanism
        self.first_stop_step = None  # Step number of first stop attempt
        
        # Initialize logger and validator
        self.logger = ActionLogger(log_dir=log_dir)

        # Initialize stats tracker with the output directory
        # (output_dir was already derived earlier in __init__)
        self.stats_tracker = StatsTracker(output_dir=str(self.output_dir), case_id=None)
        
        # Set a lower threshold to give the model more control over stopping
        # The model can decide to stop when it thinks it has extracted enough
        total_keys = len(self.checklist_config) if self.checklist_config else 0
        # Lower threshold: ~50% to allow model more flexibility
        default_min_filled = int(total_keys * 0.5) if total_keys > 0 else 0
        self.validator = StopValidator(
            min_filled_keys=default_min_filled,
            plateau_steps=20  # Increased to be less restrictive
        )
        self.performance = PerformanceTracker()
    
    def _load_config(self):
        """Load configuration files."""
        # Load checklist configuration
        if self.checklist_config_path:
            # Use specified checklist config path
            checklist_file = Path(self.checklist_config_path)
        else:
            # Default to all_26_items.yaml in new structure
            checklist_file = self.config_dir / "checklist_configs" / "all" / "all_26_items.yaml"
            # Fall back to old location if new doesn't exist
            if not checklist_file.exists():
                checklist_file = self.config_dir / "checklist_config.yaml"
        
        if checklist_file.exists():
            with open(checklist_file, 'r') as f:
                checklist_data = yaml.safe_load(f)
                self.checklist_config = checklist_data.get('checklist_items', {})
                self.user_instruction = checklist_data.get('user_instruction', '')
                self.task_constraints = checklist_data.get('constraints', [])
        else:
            # Default to empty checklist if no config
            self.checklist_config = {}
            self.user_instruction = ""
            self.task_constraints = []
            if self.verbose:
                print(f"Warning: Checklist config not found at {checklist_file}")
        
        # Load model configuration for orchestrator settings
        model_config_file = self.config_dir / "model_config.yaml"
        if model_config_file.exists():
            with open(model_config_file, 'r') as f:
                model_data = yaml.safe_load(f)
                
                # Select model-specific config based on model name
                models_config = model_data.get('models', {})
                
                # Detect model type
                if 'qwen3' in self.model_name.lower():
                    self.orchestrator_config = models_config.get('qwen3', models_config.get('default', {}))
                elif 'gpt-oss' in self.model_name.lower():
                    self.orchestrator_config = models_config.get('gpt-oss', models_config.get('default', {}))
                else:
                    # Use default config for other models
                    self.orchestrator_config = models_config.get('default', {})
        else:
            # Default orchestrator settings
            self.orchestrator_config = {
                'temperature': 0.3,
                'top_p': 1.0,
                'top_k': -1,
                'max_tokens': 1024
            }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all tools with proper dependencies.
        
        Returns:
            Dictionary mapping tool names to tool instances
        """
        tools = {}
        
        # Initialize each tool
        tools["get_checklist"] = GetChecklistTool(self.store)
        tools["update_checklist"] = UpdateChecklistTool(self.store, self.ledger)
        tools["append_checklist"] = AppendChecklistTool(self.store, self.ledger)
        tools["list_documents"] = ListDocumentsTool(self.document_manager, self.ledger)
        tools["read_document"] = ReadDocumentTool(self.document_manager, self.ledger)
        tools["search_document_regex"] = SearchDocumentRegexTool(self.document_manager, self.ledger)
        
        return tools
    
    def run(
        self,
        run_id: Optional[str] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Run the agent to completion.
        
        Args:
            run_id: Optional run ID (generates if not provided)
            resume: Whether to resume from existing state
            
        Returns:
            Final results dictionary
        """
        # Initialize run
        self.current_run_id = run_id or str(uuid.uuid4())
        self.current_step = 0
        self.last_tool_result = None
        self.last_tool_name = None
        self.action_history = []
        self.stop_count = 0
        self.first_stop_step = None
        
        # Re-initialize logger with run_id
        self.logger = ActionLogger(log_dir="logs", run_id=self.current_run_id)
        self.performance = PerformanceTracker()
        
        # Reset orchestrator for new run (clears system prompt printed flag)
        self.orchestrator.reset_for_new_run()
        
        if not resume:
            # Clear state for fresh run
            self.store.reset()
            # Reset ledger for fresh run to avoid showing old actions
            self.ledger.reset()
            # Clear raw responses file for fresh run
            raw_responses_file = self.output_dir / "raw_responses.jsonl"
            if raw_responses_file.exists():
                raw_responses_file.unlink()
        else:
            # Load existing stats if resuming
            self.stats_tracker.load_existing_stats()
        
        if self.verbose:
            print(f"{'='*60}")
            print(f"Starting Legal Agent Run")
            print(f"Run ID: {self.current_run_id}")
            print(f"Corpus: {self.corpus_path}")
            print(f"Max Steps: {self.max_steps}")
            print(f"{'='*60}\n")
        
        # Main execution loop
        while self.current_step < self.max_steps:
            try:
                # Execute one step
                action, should_stop, stop_reason = self._execute_step()
                
                # Check if we should stop
                if should_stop:
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"Stopping: {stop_reason}")
                        print(f"{'='*60}")
                    break
                    
            except KeyboardInterrupt:
                if self.verbose:
                    print("\n\nInterrupted by user")
                break
                
            except Exception as e:
                if self.verbose:
                    print(f"\nError at step {self.current_step}: {e}")
                    if self.verbose > 1:
                        traceback.print_exc()
                
                # Try to recover or stop
                if self.current_step > self.max_steps - 10:
                    break
        
        # Finalize and return results
        return self._finalize_run()
    
    def _execute_step(self) -> Tuple[OrchestratorAction, bool, str]:
        """
        Execute a single step of the agent.
        
        Returns:
            Tuple of (action taken, should_stop, stop_reason)
        """
        self.current_step += 1
        
        if self.verbose:
            print(f"\n--- Step {self.current_step}/{self.max_steps} ---")
        
        # Build snapshot
        self.performance.start_timer("snapshot_build")
        snapshot = self.snapshot_builder.build_snapshot(
            self.current_run_id,
            self.current_step,
            self.last_tool_result,
            self.last_tool_name,
            include_last_result=(self.current_step == 1 or self.last_tool_result is not None),
            action_history=self.action_history,  # Pass action history for recent actions
            stop_count=self.stop_count,
            first_stop_step=self.first_stop_step
        )
        snapshot_time = self.performance.end_timer("snapshot_build")
        
        # Log snapshot info
        snapshot_dict = snapshot.dict() if hasattr(snapshot, 'dict') else snapshot
        snapshot_size = len(json.dumps(snapshot_dict, default=str))  # Approximate size
        self.logger.log_snapshot(self.current_step, snapshot_dict, snapshot_size)
        
        # Get orchestrator decision
        self.performance.start_timer("llm_inference")
        action = self.orchestrator.choose_action(snapshot)
        llm_time = self.performance.end_timer("llm_inference")
        
        # Update stats if available
        if action.stats:
            # Determine if system prompt is cached (after first step in persistent mode)
            is_system_cached = (
                self.current_step > 1 and 
                hasattr(self.orchestrator, 'llm_client') and 
                self.orchestrator.llm_client.persistent
            )
            
            self.stats_tracker.update_stats(
                step=self.current_step,
                prompt_tokens=action.stats.get("prompt_tokens", 0),
                completion_tokens=action.stats.get("completion_tokens", 0),
                model=action.stats.get("model", "unknown"),
                is_system_cached=is_system_cached,
                system_tokens=action.stats.get("system_prompt_tokens")  # Pass actual system tokens if available
            )
        
        # Check if this is a parse/validation error action
        is_error_action = hasattr(action, 'parse_failed') and action.parse_failed
        
        if self.verbose:
            if action.decision == "stop":
                print(f"Decision: STOP - {action.reason}")
            elif is_error_action:
                print(f"Error Action: {action.tool}")
                if action.args and "error" in action.args:
                    print(f"  Error: {action.args['error']}")
            else:
                print(f"Action: {action.tool}")
                if self.verbose > 1 and action.args:
                    print(f"  Args: {json.dumps(action.args, indent=2)}")
        
        # Handle parse/validation errors first
        if is_error_action:
            # Record the error action in history without executing
            self.action_history.append({
                "step": self.current_step,
                "action": action.dict(),
                "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
                "success": False,
                "error": action.args.get("error", "Parse or validation error"),
                "tool_result": None,
                "auto_generated": False
            })
            
            # Also record parse errors in ledger (these only happen after max retries, consuming a step)
            # parse_error happens after retries exhausted, validation_error shouldn't happen here
            if action.tool == "parse_error":
                self.ledger.record_tool(
                    tool_name="parse_error",
                    args=action.args,
                    result={"error": action.args.get("error", "Parse failure after retries")},
                    step=self.current_step,
                    run_id=self.current_run_id,
                    success=False
                )
            
            # Clear last tool result since this wasn't a real tool execution
            self.last_tool_result = None
            self.last_tool_name = action.tool
            
            # Continue to next step - let model see the error and decide
            return action, False, "Parse error recorded"
        
        # Handle stop decision with two-stage mechanism
        if action.decision == "stop":
            self.stop_count += 1
            
            # Record stop action in action history
            self.action_history.append({
                "step": self.current_step,
                "action": action.dict(),
                "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
                "success": True,  # Stop actions don't fail
                "error": None,
                "auto_generated": False
            })
            
            # Also record stop decision in ledger for audit trail
            self.ledger.record_tool(
                tool_name="stop",
                args={"reason": action.reason},
                result={"stop_count": self.stop_count, "decision": "stop"},
                step=self.current_step,
                run_id=self.current_run_id,
                success=True
            )
            
            # Safeguard: Force stop after 3 attempts to prevent infinite loops
            if self.stop_count >= 3:
                if self.verbose:
                    print(f"  Maximum stop attempts (3) reached - terminating agent")
                
                # Log the forced stop
                self.logger.log_decision(
                    self.current_step, "forced_stop", f"Maximum stop attempts reached: {action.reason}",
                    {"stop_count": self.stop_count, "first_stop_step": self.first_stop_step}
                )
                
                return action, True, f"Forced stop after {self.stop_count} attempts: {action.reason}"
            
            if self.stop_count == 1:
                # First stop - record it and auto-call get_checklist
                self.first_stop_step = self.current_step
                
                if self.verbose:
                    print(f"  First stop attempt - automatically running get_checklist('all')")
                
                # Log the first stop
                self.logger.log_decision(
                    self.current_step, "first_stop", action.reason, 
                    {"stop_count": self.stop_count}
                )
                
                # Increment step for the auto-generated get_checklist
                self.current_step += 1
                
                # Create and execute get_checklist action
                auto_action = OrchestratorAction(
                    tool="get_checklist",
                    args={}  # Empty args means get all
                )
                
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Step {self.current_step}")
                    print(f"Action: get_checklist [AUTO-GENERATED]")
                    print(f"{'='*60}")
                
                # Execute the auto-generated get_checklist
                tool_result = self._execute_tool(auto_action)
                
                # Record the auto-generated action
                self.action_history.append({
                    "step": self.current_step,
                    "action": auto_action.dict(),
                    "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
                    "success": tool_result.get("error") is None if tool_result else False,
                    "error": tool_result.get("error") if tool_result else None,
                    "validation_errors": tool_result.get("validation_errors", []) if tool_result else [],
                    "tool_result": tool_result,
                    "auto_generated": True  # Mark as auto-generated
                })
                
                # Update last tool result for next snapshot
                self.last_tool_result = tool_result
                self.last_tool_name = "get_checklist"
                
                # Log the auto-generated action
                self.logger.log_action(
                    self.current_step, auto_action.dict(), tool_result,
                    {"auto_generated": True}
                )
                
                # Continue execution - don't stop yet
                return action, False, "First stop - reviewing checklist"
            else:
                # Second or later stop - actually stop
                if self.verbose:
                    print(f"  Second stop attempt - terminating agent")
                
                # Log the final stop
                self.logger.log_decision(
                    self.current_step, "final_stop", action.reason,
                    {"stop_count": self.stop_count, "first_stop_step": self.first_stop_step}
                )
                
                return action, True, action.reason
        
        # Execute tool
        tool_result = self._execute_tool(action)
        
        # Record action with result status and tool result
        action_record = {
            "step": self.current_step,
            "action": action.dict(),
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "success": tool_result.get("error") is None if tool_result else False,
            "error": tool_result.get("error") if tool_result else None,
            "validation_errors": tool_result.get("validation_errors", []) if tool_result else [],
            "tool_result": tool_result,  # Store the full result for formatting
            "auto_generated": False  # Regular actions are not auto-generated
        }
        self.action_history.append(action_record)
        
        # Update last tool result and tool name (cleared after first use)
        if self.current_step == 1:
            self.last_tool_result = tool_result
            self.last_tool_name = action.tool if tool_result else None
        else:
            # Only keep for one turn
            self.last_tool_result = tool_result if tool_result else None
            self.last_tool_name = action.tool if tool_result else None
        
        # Check if we've reached max steps
        should_stop, stop_reason = self._check_max_steps()
        
        return action, should_stop, stop_reason
    
    def _execute_tool(self, action: OrchestratorAction) -> Optional[Dict[str, Any]]:
        """
        Execute a tool with error handling.
        
        Args:
            action: Action to execute
            
        Returns:
            Tool result or error dict
        """
        tool_name = action.tool
        args = action.args or {}
        
        # Check if tool exists
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            if self.verbose:
                print(f"  ERROR: {error_msg}")
            return {"error": error_msg}
        
        tool = self.tools[tool_name]
        
        # Set context for tools that need it
        if hasattr(tool, 'set_context'):
            tool.set_context(self.current_run_id, self.current_step)
        
        # Track execution time
        self.performance.start_timer(f"tool_{tool_name}")
        
        try:
            # Execute tool
            result = tool.call(args)
            execution_time = self.performance.end_timer(f"tool_{tool_name}")
            
            # Check if this is a validation error (tool executed but returned validation errors)
            has_validation_error = result.get('validation_errors') and not result.get('success', True)
            
            # Record tool calls to ledger for tools that don't self-record
            # Tools like read_document, search_document_regex, update_checklist, append_checklist
            # already record themselves internally, but list_documents and get_checklist don't
            # ALSO record if there's a validation error (since the tool won't record failed validations)
            if tool_name in ['list_documents', 'get_checklist'] or has_validation_error:
                self.ledger.record_tool(
                    tool_name=tool_name,
                    args=args,
                    result={'documents': len(result.get('documents', [])) if tool_name == 'list_documents' else None,
                            'items_requested': args.get('items') or args.get('item', 'all') if tool_name == 'get_checklist' else None,
                            'validation_errors': result.get('validation_errors') if has_validation_error else None},
                    step=self.current_step,
                    run_id=self.current_run_id,
                    success=not has_validation_error
                )
            
            # Mark documents as discovered if list_documents was called
            if tool_name == "list_documents":
                self.snapshot_builder.mark_documents_discovered()
            
            # Extract changed keys if available
            changed_keys = []
            if tool_name == "update_checklist" and "updated_keys" in result:
                changed_keys = result["updated_keys"]
            
            # Log successful execution
            self.logger.log_action(
                self.current_step, tool_name, args, execution_time, changed_keys
            )
            
            # Log tool result summary
            result_size = len(json.dumps(result, default=str)) if result else 0
            self.logger.log_tool_result(
                self.current_step, tool_name, result_size, True
            )
            
            # Print summary based on tool type
            if self.verbose:
                self._print_tool_result(tool_name, result)
            
            return result
            
        except Exception as e:
            execution_time = self.performance.end_timer(f"tool_{tool_name}")
            error_msg = f"Tool execution error: {str(e)}"
            
            # Log error
            self.logger.log_action(
                self.current_step, tool_name, args, execution_time, [], error_msg
            )
            self.logger.log_tool_result(
                self.current_step, tool_name, 0, False, error_msg
            )
            
            # Record failed tool execution in ledger
            # This captures validation errors that happen before the tool can record itself
            self.ledger.record_tool(
                tool_name=tool_name,
                args=args,
                result={"error": error_msg},
                step=self.current_step,
                run_id=self.current_run_id,
                success=False
            )
            
            if self.verbose:
                print(f"  ERROR: {error_msg}")
                if self.verbose > 1:
                    traceback.print_exc()
            
            return {"error": error_msg}
    
    def _print_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Print a summary of tool result."""
        if not self.verbose:
            return
        
        if tool_name == "list_documents":
            docs = result.get("documents", [])
            print(f"  Found {len(docs)} documents")
            
        elif tool_name == "read_document":
            tokens_read = result.get("end_token", 0) - result.get("start_token", 0)
            print(f"  Read {tokens_read} tokens from {result.get('doc_name', 'unknown')}")
            
        elif tool_name == "search_document_regex":
            # Handle new multi-document search result format
            if "results" in result:  # New format
                total_matches = result.get("total_matches", 0)
                docs_searched = result.get("documents_searched", [])
                docs_with_matches = [r for r in result.get("results", []) if r.get("matches", [])]
                
                if len(docs_searched) == 1:
                    # Single document search - always show the document name
                    doc_name = docs_searched[0] if docs_searched else 'unknown'
                    print(f"  Found {total_matches} matches in {doc_name}")
                elif len(docs_searched) > 1:
                    # Multiple documents searched
                    if total_matches == 0:
                        # No matches in any document
                        if len(docs_searched) <= 3:
                            print(f"  Found 0 matches in [{', '.join(docs_searched)}]")
                        else:
                            print(f"  Found 0 matches in {len(docs_searched)} documents")
                    elif len(docs_with_matches) == 1:
                        # Matches in only one document
                        doc_result = docs_with_matches[0]
                        doc_name = doc_result.get("doc_name", "unknown")
                        match_count = len(doc_result.get("matches", []))
                        print(f"  Found {match_count} matches in {doc_name} (searched {len(docs_searched)} documents)")
                    else:
                        # Matches in multiple documents
                        match_summary = []
                        for doc_result in docs_with_matches[:3]:  # Show first 3
                            doc_name = doc_result.get("doc_name", "unknown")
                            match_count = len(doc_result.get("matches", []))
                            match_summary.append(f"{match_count} in {doc_name}")
                        
                        summary = f"  Found {total_matches} matches: {', '.join(match_summary)}"
                        if len(docs_with_matches) > 3:
                            summary += f" (+{len(docs_with_matches)-3} more docs)"
                        print(summary)
                else:
                    # No documents searched or empty list
                    print(f"  Found 0 matches (no documents searched)")
            else:  # Old format fallback
                matches = result.get("matches", [])
                print(f"  Found {len(matches)} matches in {result.get('doc_name', 'unknown')}")
            
        elif tool_name == "get_checklist":
            stats = result.get("completion_stats", {})
            total_keys = stats.get('total', stats.get('filled', 0) + stats.get('empty', 0))
            print(f"  Checklist: {stats.get('filled', 0)}/{total_keys} filled, "
                  f"{stats.get('empty', 0)} empty")
            
        elif tool_name == "update_checklist":
            updated = result.get("updated_keys", [])
            if updated:
                print(f"  Updated: {', '.join(updated)}")
            else:
                print(f"  No updates made")
                if result.get("validation_errors"):
                    print(f"  Errors: {result['validation_errors']}")
    
    def _check_max_steps(self) -> Tuple[bool, str]:
        """
        Check if we've reached the maximum step limit.
        
        This is the ONLY automatic stop condition. The model decides when to stop
        based on extraction completeness.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Only stop if we've reached the max steps limit
        # Let the model decide when extraction is complete
        if self.current_step >= self.max_steps:
            return True, "Reached maximum steps limit"
        
        return False, ""
    
    def _finalize_run(self) -> Dict[str, Any]:
        """
        Finalize the run and prepare results.
        
        Returns:
            Final results dictionary
        """
        # Get final statistics
        final_stats = self.store.get_completion_stats()
        final_output = self.store.get_final_output()
        empty_keys = self.store.get_empty_keys()
        
        # Determine stop reason
        if self.action_history:
            last_action = self.action_history[-1].get("action", {})
            stop_reason = last_action.get("reason", "Unknown")
        else:
            stop_reason = "No actions taken"
        
        # Log final summary
        # Using a lower threshold now since model has control over stopping
        total_keys = len(self.store.checklist_keys)
        completion_threshold = int(total_keys * 0.5) if total_keys > 0 else 0  # 50% threshold
        
        self.logger.log_run_summary(
            total_steps=self.current_step,
            filled_keys=final_stats['filled'],
            total_keys=total_keys,
            stop_reason=stop_reason,
            final_status="completed" if final_stats['filled'] >= completion_threshold else "partial"
        )
        
        # Get token usage stats
        token_stats = self.stats_tracker.get_summary()
        
        # Build results
        results = {
            "run_id": self.current_run_id,
            "total_steps": self.current_step,
            "completion_stats": final_stats,
            "empty_keys": empty_keys,
            "checklist": final_output,
            "action_history": self.action_history[:10],  # First 10 actions
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "performance_metrics": self.performance.get_metrics(),
            "token_usage": token_stats
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Run Complete")
            print(f"{'='*60}")
            print(f"Total Steps: {self.current_step}")
            print(f"Keys Filled: {final_stats['filled']}/{len(self.store.checklist_keys)}")
            print(f"Empty Keys: {final_stats['empty']}")
            print(f"\nLogs saved to: {self.logger.log_dir}")
            
            if empty_keys:
                print(f"\nEmpty Keys: {', '.join(empty_keys[:5])}")
                if len(empty_keys) > 5:
                    print(f"  ... and {len(empty_keys)-5} more")
        
        # Print token usage statistics
        self.stats_tracker.print_summary()
        
        # Save results to file
        self._save_results(results)
        
        # Cleanup LLM if using persistent mode
        if hasattr(self.orchestrator, 'llm_client') and hasattr(self.orchestrator.llm_client, 'cleanup'):
            self.orchestrator.llm_client.cleanup()
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        # Use the same output directory as stats tracker (derived from store_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.output_dir / f"run_{self.current_run_id[:8]}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\nResults saved to: {output_file}")


class BatchDriver:
    """
    Driver for running multiple cases in batch.
    """
    
    def __init__(
        self,
        corpus_base_path: str,
        output_base_path: str = "output",
        **driver_kwargs
    ):
        """
        Initialize batch driver.
        
        Args:
            corpus_base_path: Base directory containing case folders
            output_base_path: Base directory for outputs
            **driver_kwargs: Arguments passed to individual drivers
        """
        self.corpus_base = Path(corpus_base_path)
        # Extract model name and create model-specific output directory
        model_name = driver_kwargs.get('model_name', 'Qwen/Qwen3-8B')
        model_suffix = model_name.split('/')[-1] if '/' in model_name else model_name
        self.output_base = Path(output_base_path) / model_suffix
        self.driver_kwargs = driver_kwargs
    
    def run_batch(self, case_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run agent on multiple cases.
        
        Args:
            case_ids: List of case IDs to process (all if None)
            
        Returns:
            Summary of batch results
        """
        # Find cases to process
        if case_ids:
            cases = [self.corpus_base / case_id for case_id in case_ids]
        else:
            # Find all case directories
            cases = [d for d in self.corpus_base.iterdir() if d.is_dir()]
        
        results = {}
        
        for case_path in cases:
            case_id = case_path.name
            print(f"\n{'='*60}")
            print(f"Processing Case: {case_id}")
            print(f"{'='*60}")
            
            # Create driver for this case
            store_path = self.output_base / case_id / "checklist.json"
            ledger_path = self.output_base / case_id / "ledger.jsonl"
            
            store_path.parent.mkdir(parents=True, exist_ok=True)
            
            driver = Driver(
                corpus_path=str(case_path),
                store_path=str(store_path),
                ledger_path=str(ledger_path),
                **self.driver_kwargs
            )
            
            # Run the agent
            try:
                case_results = driver.run()
                results[case_id] = {
                    "success": True,
                    "stats": case_results["completion_stats"],
                    "steps": case_results["total_steps"]
                }
            except Exception as e:
                results[case_id] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"Error processing {case_id}: {e}")
        
        # Save batch summary
        summary_file = self.output_base / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Batch Complete: {len(results)} cases processed")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
        
        return results