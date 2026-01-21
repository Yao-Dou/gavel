"""Utility functions for chunk-by-chunk processing."""

import json
from typing import Dict, Any


def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges two nested dictionaries with unique ending keys.
    Args:
        dict1: First dictionary
        dict2: Second dictionary
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    def recursive_merge(current_dict, other_dict):
        for key, value in other_dict.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                recursive_merge(current_dict[key], value)
            else:
                # If we reach here, we're at a leaf node or there's a conflict
                # Since we guarantee unique ending keys, we keep the existing value
                print(f"Conflict at key: {key}. Keeping existing value.")
                continue
    
    recursive_merge(result, dict2)
    return result