"""
Snapshot Builder for creating compact state representations.
Builds the snapshot that the orchestrator uses to decide next actions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pytz
from pathlib import Path

from state.store import ChecklistStore, Ledger
from state.schemas import (
    Snapshot, RunHeader, Task, DocumentInfo,
    ActionRecord, Evidence, ChecklistItem
)
from agent.document_manager import DocumentManager


class SnapshotBuilder:
    """
    Builds compact snapshots of the current state for the orchestrator.
    The snapshot contains only the essential information needed for decision-making.
    """
    
    def __init__(
        self,
        store: ChecklistStore,
        ledger: Ledger,
        document_manager: DocumentManager,
        max_action_tail: int = 100,  # Increased from 10 to show more history
        max_evidence_headers: int = 5,
        checklist_config: Optional[Dict[str, Any]] = None,
        user_instruction: Optional[str] = None,
        task_constraints: Optional[List[str]] = None,
        recent_actions_detail: int = 5
    ):
        """
        Initialize the snapshot builder.
        
        Args:
            store: Checklist store for current state
            ledger: Ledger for action history
            document_manager: Document manager for corpus info
            max_action_tail: Maximum recent actions to include
            max_evidence_headers: Maximum recent evidence to show
            checklist_config: Configuration for checklist items
            user_instruction: User's task instruction
            task_constraints: Optional task-specific constraints
            recent_actions_detail: Number of recent actions to show with detailed results
        """
        self.store = store
        self.ledger = ledger
        self.document_manager = document_manager
        self.max_action_tail = max_action_tail
        self.max_evidence_headers = max_evidence_headers
        self.checklist_config = checklist_config or {}
        self.user_instruction = user_instruction or ""
        self.task_constraints = task_constraints or []
        self.recent_actions_detail = recent_actions_detail
        self.documents_discovered = False  # Track whether list_documents has been called
    
    def build_snapshot(
        self,
        run_id: str,
        step: int,
        last_tool_result: Optional[Dict[str, Any]] = None,
        last_tool_name: Optional[str] = None,
        include_last_result: bool = True,
        action_history: Optional[List[Dict[str, Any]]] = None,
        stop_count: int = 0,
        first_stop_step: Optional[int] = None
    ) -> Snapshot:
        """
        Build a complete snapshot of the current state.
        
        Args:
            run_id: Current run identifier
            step: Current step number
            last_tool_result: Result from the last tool execution
            last_tool_name: Name of the tool that produced last_tool_result
            include_last_result: Whether to include last_tool_result (ephemeral)
            action_history: Full action history from the driver
            
        Returns:
            Snapshot object with all components
        """
        # Build each component
        run_header = self._build_run_header(run_id, step)
        task = self._build_task()
        documents = self._build_document_list()
        checklist = self.store.get_checklist()  # Get full checklist
        action_tail = self._build_action_tail(action_history)
        recent_evidence = self._build_recent_evidence()
        
        # Include last tool result only if requested (ephemeral for current turn)
        if include_last_result and last_tool_result:
            final_result = last_tool_result
            final_tool_name = last_tool_name
        else:
            final_result = None
            final_tool_name = None
        
        return Snapshot(
            run_header=run_header,
            task=task,
            documents=documents,
            checklist=checklist,
            action_tail=action_tail,
            recent_evidence_headers=recent_evidence,
            last_tool_result=final_result,
            last_tool_name=final_tool_name,
            stop_count=stop_count,
            first_stop_step=first_stop_step,
            recent_actions_detail=self.recent_actions_detail
        )
    
    def _build_run_header(self, run_id: str, step: int) -> RunHeader:
        """Build run header with metadata."""
        return RunHeader(
            run_id=run_id,
            step=step,
            timestamp=datetime.now(pytz.timezone('America/New_York'))
        )
    
    def _build_task(self) -> Task:
        """Build task specification with checklist definitions and user instruction."""
        # Build checklist definitions for the task
        checklist_definitions = {}
        for key, config in self.checklist_config.items():
            checklist_definitions[key] = config.get('description', '')
        
        return Task(
            user_instruction=self.user_instruction or "Extract all checklist items from the case corpus.",
            constraints=self.task_constraints,  # Use task-specific constraints from config
            checklist_definitions=checklist_definitions
        )
    
    def mark_documents_discovered(self):
        """Mark that documents have been discovered (list_documents was called)."""
        self.documents_discovered = True
    
    def _build_document_list(self) -> List[DocumentInfo]:
        """Build list of documents with coverage info."""
        # Only return documents if they've been discovered
        if not self.documents_discovered:
            return []
        
        documents = []
        for doc_name in self.document_manager.list_documents():
            doc_info = self.document_manager.get_document_info(doc_name, self.ledger)
            documents.append(doc_info)
        
        # Sort by importance (visited first, then by type)
        documents.sort(key=lambda d: (not d.visited, d.type))
        
        return documents
    
    def _build_action_tail(self, action_history: Optional[List[Dict[str, Any]]] = None) -> List[ActionRecord]:
        """Build list of recent actions from driver's action history."""
        if not action_history:
            return []
        
        # Get the last N actions (limit by max_action_tail)
        recent_actions = action_history[-self.max_action_tail:] if len(action_history) > self.max_action_tail else action_history
        
        # Convert to ActionRecord objects
        action_records = []
        for action_data in recent_actions:
            action = action_data.get("action", {})
            tool_result = action_data.get("tool_result", {})
            
            # Build target based on tool type and args
            target = {}
            if action.get("tool") == "read_document" and action.get("args"):
                target = {
                    "doc_name": action["args"].get("doc_name", ""),
                    "start_token": action["args"].get("start_token", 0),
                    "end_token": action["args"].get("end_token", 0)
                }
            elif action.get("tool") == "search_document_regex" and action.get("args"):
                target = {
                    "pattern": action["args"].get("pattern", "")
                }
                # Include document specification (either doc_names array or doc_name)
                if "doc_names" in action["args"]:
                    target["doc_names"] = action["args"]["doc_names"]
                if "doc_name" in action["args"]:
                    target["doc_name"] = action["args"]["doc_name"]
            elif action.get("args"):
                target = action["args"]
            
            # Extract changed keys for update/append operations
            changed_keys = []
            if action.get("tool") in ["update_checklist", "append_checklist"]:
                if tool_result and "updated_keys" in tool_result:
                    changed_keys = tool_result["updated_keys"]
                elif action.get("args") and "patch" in action["args"]:
                    # Extract keys from patch
                    changed_keys = [p.get("key") for p in action["args"]["patch"] if "key" in p]
            
            # Handle special error actions (parse_error, validation_error) and stop decisions
            tool_name = action.get("tool", "unknown")
            
            # Check if this is a stop decision
            if action.get("decision") == "stop":
                tool_name = "stop"
                target = {
                    "reason": action.get("reason", "No reason provided"),
                    "remaining_empty_keys": action.get("remaining_empty_keys", [])
                }
            elif tool_name in ["parse_error", "validation_error"]:
                # For error actions, store all args (error message, retry_count, etc.)
                target = action.get("args", {"error": "Unknown error"})
                
            record = ActionRecord(
                tool=tool_name,
                target=target,
                step=action_data.get("step", 0),  # Preserve the original step number
                purpose=action.get("purpose"),
                changed_keys=changed_keys,
                timestamp=datetime.fromisoformat(action_data["timestamp"]) if action_data.get("timestamp") else datetime.now(pytz.timezone('America/New_York')),
                success=action_data.get("success", True),
                error=action_data.get("error"),
                validation_errors=action_data.get("validation_errors", []),
                result_summary=tool_result,  # Store full result for formatting
                auto_generated=action_data.get("auto_generated", False)  # Track auto-generated actions
            )
            action_records.append(record)
        
        return action_records
    
    def _build_recent_evidence(self) -> List[Evidence]:
        """
        Build list of recent evidence headers.
        Shows recently found evidence without full text.
        """
        recent_evidence = []
        
        # Get recent updates from the checklist
        checklist = self.store.get_checklist()
        
        # Sort by last_updated to get most recent
        sorted_items = sorted(checklist, key=lambda x: x.last_updated, reverse=True)
        
        # Collect evidence from recently updated items
        evidence_count = 0
        for item in sorted_items[:5]:  # Check last 5 updated items
            for extracted in item.extracted:
                for evidence in extracted.evidence:
                    if evidence_count >= self.max_evidence_headers:
                        break
                    
                    # Create evidence header (truncate text)
                    header = Evidence(
                        text=evidence.text[:100] + "..." if len(evidence.text) > 100 else evidence.text,
                        source_document=evidence.source_document,
                        location=evidence.location
                    )
                    recent_evidence.append(header)
                    evidence_count += 1
                
                if evidence_count >= self.max_evidence_headers:
                    break
            
            if evidence_count >= self.max_evidence_headers:
                break
        
        return recent_evidence
    
    def build_compact_snapshot(
        self,
        run_id: str,
        step: int,
        last_tool_result: Optional[Dict[str, Any]] = None,
        last_tool_name: Optional[str] = None,
        action_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Build a more compact snapshot as a dictionary.
        Useful for token efficiency.
        
        Args:
            run_id: Current run identifier
            step: Current step number
            last_tool_result: Result from last tool
            last_tool_name: Name of the tool that produced last_tool_result
            action_history: Full action history from the driver
            
        Returns:
            Compact dictionary representation
        """
        snapshot = self.build_snapshot(run_id, step, last_tool_result, last_tool_name, action_history=action_history)
        
        # Convert to dict and compact it
        data = snapshot.dict()
        
        # Remove empty fields
        if not data.get("recent_evidence_headers"):
            del data["recent_evidence_headers"]
        if not data.get("last_tool_result"):
            del data["last_tool_result"]
        
        # Shorten document info if many documents
        if len(data.get("documents", [])) > 10:
            # Only show first 5 and last 5
            docs = data["documents"]
            data["documents"] = docs[:5] + [{"summary": f"... {len(docs)-10} more documents ..."}] + docs[-5:]
        
        # Compact the action tail
        if len(data.get("action_tail", [])) > 5:
            data["action_tail"] = data["action_tail"][-5:]  # Only last 5
        
        return data