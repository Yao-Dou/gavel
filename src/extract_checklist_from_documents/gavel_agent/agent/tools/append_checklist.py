"""
Append checklist tool - adds new extracted items to existing checklist entries.
"""

from typing import Dict, Any, List
from pydantic import ValidationError
from .base import BaseTool
from state.store import ChecklistStore, Ledger
from state.schemas import UpdateChecklistInput, AppendChecklistOutput, ChecklistPatch, UpdateEvent, ExtractedItem


class AppendChecklistTool(BaseTool):
    """
    Tool for appending new extracted items to existing checklist entries.
    Unlike update_checklist which replaces, this adds to the existing list.
    Useful for incremental discovery when finding values one by one.
    Any checklist item can have multiple values in its extracted list.
    """
    
    def __init__(self, store: ChecklistStore, ledger: Ledger = None):
        """
        Initialize the append_checklist tool.
        
        Args:
            store: ChecklistStore instance to update
            ledger: Optional Ledger instance for recording updates
        """
        super().__init__(
            name="append_checklist",
            description="Append new extracted items to existing checklist entries (adds to list, doesn't replace)"
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
        Get the input schema for append_checklist.
        
        Returns:
            Schema for append operations
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
                                "description": "New items to append to existing list",
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
                            }
                        },
                        "required": ["key", "extracted"]
                    }
                }
            },
            "required": ["patch"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for append operation output
        """
        return {
            "type": "object",
            "properties": {
                "appended_keys": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "items_added": {
                    "type": "object",
                    "description": "Number of items added per key"
                },
                "duplicates_skipped": {
                    "type": "object",
                    "description": "Number of duplicate values skipped per key"
                },
                "validation_errors": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "success": {"type": "boolean"}
            },
            "required": ["appended_keys", "items_added", "duplicates_skipped", "validation_errors", "success"]
        }
    
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to append to the checklist.
        
        Args:
            args: Dictionary containing 'patch' list with items to append
            
        Returns:
            Dictionary with appended keys, items added, duplicates skipped, and success status
        """
        try:
            # Validate that patch exists
            if "patch" not in args:
                return {
                    "appended_keys": [],
                    "items_added": {},
                    "duplicates_skipped": {},
                    "validation_errors": ["Missing required field: patch"],
                    "success": False
                }
            
            # Convert patches to use add_extracted instead of extracted
            # This ensures we append rather than replace
            converted_patches = []
            items_added = {}
            duplicates_skipped = {}
            
            for patch_dict in args["patch"]:
                if "key" not in patch_dict:
                    return {
                        "appended_keys": [],
                        "items_added": {},
                        "duplicates_skipped": {},
                        "validation_errors": ["Patch missing required field: key"],
                        "success": False
                    }
                
                key = patch_dict["key"]
                
                # Get current values for duplicate checking
                current_item = self.store.get_item(key)
                existing_values = set()
                if current_item and current_item.extracted:
                    existing_values = {e.value for e in current_item.extracted}
                
                # Convert extracted field to add_extracted for appending
                if "extracted" in patch_dict:
                    new_items = []
                    skipped = 0
                    
                    for item in patch_dict["extracted"]:
                        # Check if this value already exists
                        if item.get("value") in existing_values:
                            skipped += 1
                        else:
                            new_items.append(ExtractedItem(**item))
                            existing_values.add(item.get("value"))
                    
                    if new_items:
                        # Create patch with add_extracted instead of extracted
                        converted_patch = ChecklistPatch(
                            key=key,
                            add_extracted=new_items
                        )
                        converted_patches.append(converted_patch)
                        items_added[key] = len(new_items)
                        duplicates_skipped[key] = skipped
                    elif skipped > 0:
                        # All items were duplicates
                        duplicates_skipped[key] = skipped
            
            # Apply converted patches to the store
            updated_keys, validation_errors = self.store.update_items(converted_patches)
            
            # Record in ledger if available
            if self.ledger and updated_keys:
                update_event = UpdateEvent(
                    keys_updated=updated_keys,
                    patch=converted_patches,
                    step=self._current_step,
                    success=len(validation_errors) == 0
                )
                self.ledger.record_update(update_event, self._run_id)
            
            # Create output
            output = AppendChecklistOutput(
                appended_keys=updated_keys,
                items_added=items_added,
                duplicates_skipped=duplicates_skipped,
                validation_errors=validation_errors,
                success=len(validation_errors) == 0
            )
            return self.format_output(output)
            
        except ValidationError as e:
            # Input validation failed
            return self.format_output(AppendChecklistOutput(
                appended_keys=[],
                items_added={},
                duplicates_skipped={},
                validation_errors=[str(e)],
                success=False
            ))
        except Exception as e:
            # Unexpected error
            return self.format_output(AppendChecklistOutput(
                appended_keys=[],
                items_added={},
                duplicates_skipped={},
                validation_errors=[f"Unexpected error: {str(e)}"],
                success=False
            ))