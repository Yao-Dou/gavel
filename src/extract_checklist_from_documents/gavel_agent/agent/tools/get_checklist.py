"""
Get checklist tool - returns the current state of checklist items.
Can return all items, specific items, or a single item.
"""

from typing import Dict, Any, List, Optional
from .base import BaseTool
from state.store import ChecklistStore
from state.schemas import GetChecklistOutput


class GetChecklistTool(BaseTool):
    """
    Tool for retrieving the current state of the checklist.
    Can return:
    - Multiple specific items if 'items' parameter is provided (non-empty array)
    - A single specific item if 'item' parameter is provided
    - All items if 'item' is 'all' or neither parameter is provided
    """
    
    def __init__(self, store: ChecklistStore):
        """
        Initialize the get_checklist tool.
        
        Args:
            store: ChecklistStore instance to read from
        """
        super().__init__(
            name="get_checklist",
            description="Retrieve the current state of checklist items (all or specific)"
        )
        self.store = store
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema - accepts optional 'items' array or 'item' parameter.
        
        Returns:
            Schema with optional 'items' array and 'item' parameter
        """
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of specific checklist item names to retrieve"
                },
                "item": {
                    "type": "string",
                    "description": "Specific checklist item name to retrieve, or 'all' for all items (default: 'all')",
                    "default": "all"
                }
            },
            "required": []
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for GetChecklistOutput
        """
        return {
            "type": "object",
            "properties": {
                "checklist": {
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
                                                }
                                            }
                                        },
                                        "value": {"type": "string"}
                                    }
                                }
                            },
                            "last_updated": {"type": "string", "format": "date-time"}
                        },
                        "required": ["key", "extracted", "last_updated"]
                    }
                },
                "completion_stats": {
                    "type": "object",
                    "properties": {
                        "filled": {"type": "integer"},
                        "empty": {"type": "integer"},
                        "total": {"type": "integer"}
                    }
                }
            },
            "required": ["checklist", "completion_stats"]
        }
    
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to get the checklist.
        
        Args:
            args: Dictionary with optional 'items' array or 'item' parameter
            
        Returns:
            Dictionary containing checklist and completion statistics
        """
        # Priority 1: Check if 'items' is a non-empty array
        items_list = args.get('items', [])
        if items_list and isinstance(items_list, list) and len(items_list) > 0:
            # Get multiple specific items in the requested order
            checklist = []
            not_found = []
            
            for item_name in items_list:
                specific_item = self.store.get_item(item_name)
                if specific_item is None:
                    not_found.append(item_name)
                else:
                    checklist.append(specific_item)
            
            # If some items were not found, return error
            if not_found:
                return {
                    "error": f"Checklist items not found: {', '.join(not_found)}",
                    "requested_items": items_list,
                    "found_items": [item.key for item in checklist],
                    "available_keys": self.store.checklist_keys,
                    "success": False
                }
        
        # Priority 2: Check 'item' parameter
        else:
            item_name = args.get('item', 'all')
            
            # Handle different cases for 'item'
            if item_name == 'all' or item_name == '' or item_name is None:
                # Get the full checklist
                checklist = self.store.get_checklist()
            else:
                # Get specific single item
                specific_item = self.store.get_item(item_name)
                if specific_item is None:
                    # Item not found
                    return {
                        "error": f"Checklist item '{item_name}' not found",
                        "available_keys": self.store.checklist_keys,
                        "success": False
                    }
                checklist = [specific_item]
        
        # Calculate statistics for the returned items
        filled = sum(1 for item in checklist if item.extracted)
        empty = sum(1 for item in checklist if not item.extracted)
        total = len(checklist)
        
        stats = {
            "filled": filled,
            "empty": empty,
            "total": total
        }
        
        # Create output
        output = GetChecklistOutput(
            checklist=checklist,
            completion_stats=stats
        )
        
        result = self.format_output(output)
        
        # Add note about partial results based on what was requested
        if items_list and len(items_list) > 0:
            result["items_requested"] = items_list
            result["note"] = f"Showing {len(checklist)} requested items (statistics are for these items only)"
        elif item_name != 'all' and item_name != '' and item_name is not None:
            result["item_requested"] = item_name
            result["note"] = f"Showing only '{item_name}' (statistics are for this item only)"
        
        return result