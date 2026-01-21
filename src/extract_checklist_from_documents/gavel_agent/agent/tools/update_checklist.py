"""
Update checklist tool - applies patches to update checklist items.
"""

from typing import Dict, Any, List
from pydantic import ValidationError
from .base import BaseTool
from state.store import ChecklistStore, Ledger
from state.schemas import UpdateChecklistInput, UpdateChecklistOutput, ChecklistPatch, UpdateEvent


class UpdateChecklistTool(BaseTool):
    """
    Tool for updating checklist items with extracted information.
    Validates that all extracted items have supporting evidence.
    """
    
    def __init__(self, store: ChecklistStore, ledger: Ledger = None):
        """
        Initialize the update_checklist tool.
        
        Args:
            store: ChecklistStore instance to update
            ledger: Optional Ledger instance for recording updates
        """
        super().__init__(
            name="update_checklist",
            description="Update checklist items with extracted information and evidence"
        )
        self.store = store
        self.ledger = ledger
        self._current_step = 0
        self._run_id = "default"
    
    def set_context(self, run_id: str, step: int):
        """
        Set the current run context for ledger recording.
        
        Args:
            run_id: Current run ID
            step: Current step number
        """
        self._run_id = run_id
        self._current_step = step
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema for update_checklist.
        
        Returns:
            Schema for UpdateChecklistInput
        """
        return {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "extracted": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "evidence": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "text": {"type": "string"},
                                                    "source_document": {"type": "string"},
                                                    "location": {"type": "string"}
                                                },
                                                "required": ["text", "source_document", "location"]
                                            }
                                        },
                                        "value": {"type": "string"}
                                    },
                                    "required": ["evidence", "value"]
                                }
                            },
                            "add_extracted": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "evidence": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "text": {"type": "string"},
                                                    "source_document": {"type": "string"},
                                                    "location": {"type": "string"}
                                                },
                                                "required": ["text", "source_document", "location"]
                                            }
                                        },
                                        "value": {"type": "string"}
                                    },
                                    "required": ["evidence", "value"]
                                }
                            },
                            "add_candidates": {
                                "type": "array",
                                "items": {"type": "object"}
                            }
                        },
                        "required": ["key"]
                    }
                }
            },
            "required": ["patch"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for UpdateChecklistOutput
        """
        return {
            "type": "object",
            "properties": {
                "updated_keys": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "validation_errors": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "success": {"type": "boolean"}
            },
            "required": ["updated_keys", "validation_errors", "success"]
        }
    
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to update the checklist.
        
        Args:
            args: Dictionary containing 'patch' list
            
        Returns:
            Dictionary with updated keys, validation errors, and success status
        """
        try:
            # Validate input
            input_data = self.validate_input(args, UpdateChecklistInput)
            
            # Convert to ChecklistPatch objects if needed
            patches = []
            for patch_item in input_data.patch:
                try:
                    # Check if already a ChecklistPatch object or a dict
                    if isinstance(patch_item, ChecklistPatch):
                        patches.append(patch_item)
                    else:
                        # Convert dict to ChecklistPatch
                        patch = ChecklistPatch(**patch_item)
                        patches.append(patch)
                except ValidationError as e:
                    # Return validation error
                    return self.format_output(UpdateChecklistOutput(
                        updated_keys=[],
                        validation_errors=[f"Invalid patch: {e}"],
                        success=False
                    ))
            
            # Apply patches to the store
            updated_keys, validation_errors = self.store.update_items(patches)
            
            # Record in ledger if available
            if self.ledger and updated_keys:
                update_event = UpdateEvent(
                    keys_updated=updated_keys,
                    patch=patches,
                    step=self._current_step,
                    success=len(validation_errors) == 0
                )
                self.ledger.record_update(update_event, self._run_id)
            
            # Create output
            output = UpdateChecklistOutput(
                updated_keys=updated_keys,
                validation_errors=validation_errors,
                success=len(validation_errors) == 0
            )
            
            return self.format_output(output)
            
        except ValidationError as e:
            # Input validation failed
            return self.format_output(UpdateChecklistOutput(
                updated_keys=[],
                validation_errors=[str(e)],
                success=False
            ))
        except Exception as e:
            # Unexpected error
            return self.format_output(UpdateChecklistOutput(
                updated_keys=[],
                validation_errors=[f"Unexpected error: {str(e)}"],
                success=False
            ))