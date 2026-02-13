"""
JSON-based storage for checklist state and action ledger.
Provides ChecklistStore and Ledger classes for persistent state management.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime
import pytz
from threading import Lock

from .schemas import (
    ChecklistItem, ChecklistPatch, Evidence, ExtractedItem,
    DocumentInfo, DocumentCoverage, DocumentReadInfo,
    ReadEvent, SearchEvent, UpdateEvent, LedgerEntry, ToolEvent
)


class ChecklistStore:
    """
    Manages the checklist state with JSON persistence.
    Thread-safe for concurrent access.
    Supports dynamic checklist keys.
    """
    
    def __init__(self, storage_path: str = "checklist_store.json", checklist_keys: List[str] = None, checklist_config: Dict[str, Any] = None):
        """
        Initialize the checklist store.
        
        Args:
            storage_path: Path to JSON file for persistence
            checklist_keys: List of checklist keys to track (if None, uses empty checklist)
            checklist_config: Full configuration with keys and descriptions
        """
        self.storage_path = Path(storage_path)
        self.lock = Lock()
        self._checklist: Dict[str, ChecklistItem] = {}
        
        # Store checklist configuration
        if checklist_config:
            self.checklist_keys = list(checklist_config.keys())
            self.checklist_config = checklist_config
        elif checklist_keys:
            self.checklist_keys = checklist_keys
            self.checklist_config = {key: {} for key in checklist_keys}
        else:
            # Default empty if no keys provided
            self.checklist_keys = []
            self.checklist_config = {}
        
        self._initialize_checklist()
        self._load()
    
    def _initialize_checklist(self):
        """Initialize empty checklist with all configured keys."""
        for key in self.checklist_keys:
            self._checklist[key] = ChecklistItem(key=key)
    
    def _load(self):
        """Load checklist from JSON file if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for key, item_data in data.items():
                        if key in self.checklist_keys:
                            # Convert datetime strings back to datetime objects
                            if 'last_updated' in item_data:
                                # Parse the datetime string and ensure it has NYC timezone
                                dt = datetime.fromisoformat(item_data['last_updated'])
                                # If timezone-naive, assume it's NYC time
                                if dt.tzinfo is None:
                                    nyc_tz = pytz.timezone('America/New_York')
                                    dt = nyc_tz.localize(dt)
                                item_data['last_updated'] = dt
                            self._checklist[key] = ChecklistItem(**item_data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load checklist from {self.storage_path}: {e}")
                print("Starting with empty checklist.")
    
    def _save(self):
        """Save checklist to JSON file. Should be called from within a lock context."""
        # Don't acquire lock here - this method is always called from within a locked context
        data = {}
        for key, item in self._checklist.items():
            item_dict = item.dict()
            # Convert datetime to ISO format string with timezone info
            if isinstance(item_dict['last_updated'], datetime):
                # Ensure datetime has NYC timezone
                dt = item_dict['last_updated']
                if dt.tzinfo is None:
                    nyc_tz = pytz.timezone('America/New_York')
                    dt = nyc_tz.localize(dt)
                item_dict['last_updated'] = dt.isoformat()
            data[key] = item_dict
        
        # Write with pretty formatting for readability
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    
    def get_checklist(self) -> List[ChecklistItem]:
        """
        Get the full checklist state.
        
        Returns:
            List of all checklist items
        """
        with self.lock:
            return list(self._checklist.values())
    
    def get_item(self, key: str) -> Optional[ChecklistItem]:
        """
        Get a specific checklist item.
        
        Args:
            key: The checklist key to retrieve
            
        Returns:
            The checklist item or None if key not found
        """
        with self.lock:
            return self._checklist.get(key)
    
    def update_items(self, patches: List[ChecklistPatch]) -> Tuple[List[str], List[str]]:
        """
        Apply patches to update checklist items.
        
        Args:
            patches: List of patches to apply
            
        Returns:
            Tuple of (updated_keys, validation_errors)
        """
        updated_keys = []
        validation_errors = []
        
        with self.lock:
            for patch in patches:
                if patch.key not in self.checklist_keys:
                    validation_errors.append(f"Unknown key: {patch.key}")
                    continue
                
                item = self._checklist[patch.key]
                
                # Apply the patch
                # Replace entire extracted list if provided
                if patch.extracted is not None:
                    # Validate each extracted item has evidence
                    has_error = False
                    for ext_item in patch.extracted:
                        if not ext_item.evidence:
                            validation_errors.append(
                                f"ExtractedItem for {patch.key} must have evidence"
                            )
                            has_error = True
                            break
                    if not has_error:
                        item.extracted = patch.extracted
                
                # Add to extracted list incrementally
                if patch.add_extracted is not None:
                    for ext_item in patch.add_extracted:
                        if not ext_item.evidence:
                            validation_errors.append(
                                f"ExtractedItem for {patch.key} must have evidence"
                            )
                            continue  # Skip this item but continue with others
                        # Check for duplicates based on value
                        existing_values = {e.value for e in item.extracted}
                        if ext_item.value not in existing_values:
                            item.extracted.append(ext_item)
                
                item.last_updated = datetime.now(pytz.timezone('America/New_York'))
                updated_keys.append(patch.key)
            
            # Save after all updates
            if updated_keys:
                self._save()
        
        return updated_keys, validation_errors
    
    def get_empty_keys(self) -> List[str]:
        """
        Get list of keys that have no extracted values.
        
        Returns:
            List of key names that are empty
        """
        empty_keys = []
        
        with self.lock:
            for key, item in self._checklist.items():
                if not item.extracted:
                    empty_keys.append(key)
        
        return empty_keys
    
    def get_completion_stats(self) -> Dict[str, int]:
        """
        Get completion statistics for the checklist.
        
        Returns:
            Dictionary with counts of filled, empty, and total
        """
        stats = {
            "filled": 0,  # Has extracted items
            "empty": 0,   # No extracted items
            "total": len(self.checklist_keys)
        }
        
        with self.lock:
            for item in self._checklist.values():
                if item.extracted:
                    stats["filled"] += 1
                else:
                    stats["empty"] += 1
        
        return stats
    
    def get_final_output(self) -> Dict[str, Dict[str, List]]:
        """
        Get the checklist in the final output format.
        
        Returns:
            Dictionary with each key mapped to {"extracted": [...]}
        """
        output = {}
        
        with self.lock:
            for key, item in self._checklist.items():
                if item.extracted:  # Only include keys with extracted values
                    output[key] = {
                        "extracted": [
                            {
                                "evidence": [
                                    {
                                        "text": ev.text,
                                        "source_document": ev.source_document,
                                        "location": ev.location
                                    }
                                    for ev in ext_item.evidence
                                ],
                                "value": ext_item.value
                            }
                            for ext_item in item.extracted
                        ]
                    }
        
        return output
    
    def reset(self):
        """Reset the checklist to initial empty state."""
        with self.lock:
            self._initialize_checklist()
            self._save()


class Ledger:
    """
    Append-only ledger for tracking all reads, searches, and updates.
    Provides coverage statistics and audit trail.
    """
    
    def __init__(self, storage_path: str = "ledger.jsonl"):
        """
        Initialize the ledger.
        
        Args:
            storage_path: Path to JSONL file for append-only storage
        """
        self.storage_path = Path(storage_path)
        self.lock = Lock()
        self._document_coverage: Dict[str, DocumentCoverage] = {}
        self._last_reads: Dict[str, DocumentReadInfo] = {}
        self._load_coverage()
    
    def _add_token_range(self, doc_name: str, start_token: int, end_token: int):
        """
        Add a token range to document coverage, merging overlapping ranges.
        
        Args:
            doc_name: Document name
            start_token: Start of range
            end_token: End of range
        """
        if doc_name not in self._document_coverage:
            self._document_coverage[doc_name] = DocumentCoverage()
        
        coverage = self._document_coverage[doc_name]
        new_range = (start_token, end_token)
        
        # Add new range and merge overlapping ones
        ranges = coverage.token_ranges + [new_range]
        coverage.token_ranges = self._merge_ranges(ranges)
    
    def _merge_ranges(self, ranges: List[tuple]) -> List[tuple]:
        """
        Merge overlapping token ranges.
        
        Args:
            ranges: List of (start, end) tuples
            
        Returns:
            Merged list of non-overlapping ranges
        """
        if not ranges:
            return []
        
        # Sort ranges by start position
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # Check if ranges overlap or are adjacent
            if current_start <= last_end + 1:
                # Merge ranges
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Add as separate range
                merged.append((current_start, current_end))
        
        return merged
    
    def _load_coverage(self):
        """Load and compute coverage from existing ledger entries."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    entry_data = json.loads(line)
                    # Only use event_name (actual tool names)
                    event_name = entry_data.get('event_name', '')
                    
                    if event_name == 'read_document':
                        event = entry_data['event']
                        doc_name = event['doc_name']
                        
                        if doc_name not in self._document_coverage:
                            self._document_coverage[doc_name] = DocumentCoverage()
                        
                        # Update coverage
                        self._document_coverage[doc_name].windows_read += 1
                        tokens_read = event['end_token'] - event['start_token']
                        self._document_coverage[doc_name].approx_tokens_read += tokens_read
                        
                        # Add token range
                        self._add_token_range(doc_name, event['start_token'], event['end_token'])
                        
                        # Update last read
                        self._last_reads[doc_name] = DocumentReadInfo(
                            start_token=event['start_token'],
                            end_token=event['end_token']
                        )
                    
                    elif event_name == 'search_document_regex':
                        event = entry_data['event']
                        doc_name = event['doc_name']
                        
                        # Mark document as visited and add token ranges from matches
                        if 'match_ranges' in event and event['match_ranges']:
                            if doc_name not in self._document_coverage:
                                self._document_coverage[doc_name] = DocumentCoverage()
                            
                            for range_data in event['match_ranges']:
                                if isinstance(range_data, (list, tuple)) and len(range_data) == 2:
                                    self._add_token_range(doc_name, range_data[0], range_data[1])
        except Exception as e:
            print(f"Warning: Could not load ledger from {self.storage_path}: {e}")
    
    def _append_entry(self, entry: LedgerEntry):
        """Append an entry to the JSONL ledger file."""
        entry_dict = entry.dict()
        # Convert datetime to ISO format
        if 'event' in entry_dict:
            if 'timestamp' in entry_dict['event']:
                entry_dict['event']['timestamp'] = entry_dict['event']['timestamp'].isoformat()
        
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(entry_dict) + '\n')
    
    def record_read(self, read_event: ReadEvent, run_id: str):
        """
        Record a document read event.
        
        Args:
            read_event: The read event to record
            run_id: Current run ID
        """
        with self.lock:
            # Update coverage
            doc_name = read_event.doc_name
            if doc_name not in self._document_coverage:
                self._document_coverage[doc_name] = DocumentCoverage()
            
            self._document_coverage[doc_name].windows_read += 1
            tokens_read = read_event.end_token - read_event.start_token
            self._document_coverage[doc_name].approx_tokens_read += tokens_read
            
            # Add token range to coverage
            self._add_token_range(doc_name, read_event.start_token, read_event.end_token)
            
            # Update last read
            self._last_reads[doc_name] = DocumentReadInfo(
                start_token=read_event.start_token,
                end_token=read_event.end_token
            )
            
            # Append to ledger with tool name
            entry = LedgerEntry(
                event_name="read_document",  # Use actual tool name
                event=read_event,
                run_id=run_id,
                step=read_event.step
            )
            self._append_entry(entry)
    
    def record_search(self, search_event: SearchEvent, run_id: str):
        """
        Record a regex search event (single or multi-document).
        
        Args:
            search_event: The search event to record
            run_id: Current run ID
        """
        with self.lock:
            # Update coverage for each document with matches
            for doc_name, match_ranges in search_event.document_matches.items():
                if match_ranges:  # Has matches
                    # Initialize coverage if needed
                    if doc_name not in self._document_coverage:
                        self._document_coverage[doc_name] = DocumentCoverage()
                    
                    # Add token ranges from all matches
                    for start_token, end_token in match_ranges:
                        self._add_token_range(doc_name, start_token, end_token)
            
            # Always append single entry to ledger (even if no matches)
            entry = LedgerEntry(
                event_name="search_document_regex",  # Use actual tool name
                event=search_event,
                run_id=run_id,
                step=search_event.step
            )
            self._append_entry(entry)
    
    def record_update(self, update_event: UpdateEvent, run_id: str):
        """
        Record a checklist update event.
        
        Args:
            update_event: The update event to record
            run_id: Current run ID
        """
        with self.lock:
            # Determine tool name based on the update type
            # If patch has add_extracted, it's append_checklist, otherwise update_checklist
            tool_name = "update_checklist"  # Default
            if update_event.patch and len(update_event.patch) > 0:
                first_patch = update_event.patch[0]
                if hasattr(first_patch, 'add_extracted') and first_patch.add_extracted:
                    tool_name = "append_checklist"
            
            entry = LedgerEntry(
                event_name=tool_name,  # Use actual tool name
                event=update_event,
                run_id=run_id,
                step=update_event.step
            )
            self._append_entry(entry)
    
    def record_tool(self, tool_name: str, args: Dict[str, Any], result: Optional[Dict[str, Any]], 
                    step: int, run_id: str, success: bool = True):
        """
        Record a generic tool execution event.
        
        Args:
            tool_name: Name of the tool executed
            args: Arguments passed to the tool
            result: Result returned by the tool
            step: Current step number
            run_id: Current run ID
            success: Whether the tool execution was successful
        """
        with self.lock:
            tool_event = ToolEvent(
                tool_name=tool_name,
                args=args,
                result=result,
                step=step,
                success=success
            )
            
            entry = LedgerEntry(
                event_name=tool_name,
                event=tool_event,
                run_id=run_id,
                step=step
            )
            self._append_entry(entry)
    
    def get_document_coverage(self, doc_name: str) -> Optional[DocumentCoverage]:
        """
        Get coverage statistics for a document.
        
        Args:
            doc_name: Name of the document
            
        Returns:
            Coverage statistics or None if document not visited
        """
        with self.lock:
            return self._document_coverage.get(doc_name)
    
    def get_last_read(self, doc_name: str) -> Optional[DocumentReadInfo]:
        """
        Get the last read position for a document.
        
        Args:
            doc_name: Name of the document
            
        Returns:
            Last read info or None if document not read
        """
        with self.lock:
            return self._last_reads.get(doc_name)
    
    def get_visited_documents(self) -> Set[str]:
        """
        Get the set of all visited document names.
        
        Returns:
            Set of document names that have been read
        """
        with self.lock:
            return set(self._document_coverage.keys())
    
    def get_all_events(self) -> List[Dict]:
        """
        Get all events from the ledger.
        
        Returns:
            List of all event records
        """
        if not self.storage_path.exists():
            return []
        
        events = []
        with open(self.storage_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Return simplified event
                    event = {
                        "tool": entry.get('event_type', ''),
                        "step": entry.get('step', 0),
                        "timestamp": entry.get('timestamp', '')
                    }
                    if 'event' in entry and isinstance(entry['event'], dict):
                        event.update(entry['event'])
                    events.append(event)
                except:
                    continue
        
        return events
    
    def get_recent_actions(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent actions from the ledger.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of recent action records
        """
        if not self.storage_path.exists():
            return []
        
        actions = []
        with open(self.storage_path, 'r') as f:
            # Read all lines (could optimize with deque for large files)
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    entry = json.loads(line)
                    # Simplified action record
                    # Only use event_name (actual tool names)
                    event_name = entry.get('event_name', '')
                    
                    action = {
                        "tool": event_name,  # Use the actual tool name
                        "step": entry['step'],
                        "timestamp": entry['event'].get('timestamp', '')
                    }
                    
                    # Handle different event types based on actual tool names
                    if event_name == 'read_document':
                        action['target'] = {
                            "doc_name": entry['event']['doc_name'],
                            "start_token": entry['event']['start_token'],
                            "end_token": entry['event']['end_token']
                        }
                        if 'purpose' in entry['event']:
                            action['purpose'] = entry['event']['purpose']
                    elif event_name == 'search_document_regex':
                        action['target'] = {
                            "doc_name": entry['event']['doc_name'],
                            "pattern": entry['event']['pattern']
                        }
                        action['hits'] = entry['event']['matches_found']
                    elif event_name in ['update_checklist', 'append_checklist']:
                        action['changed_keys'] = entry['event']['keys_updated']
                    elif event_name == 'list_documents':
                        # For list_documents, just record it was called
                        if 'result' in entry['event'] and 'documents' in entry['event']['result']:
                            action['documents_found'] = len(entry['event']['result']['documents'])
                    elif event_name == 'get_checklist':
                        # For get_checklist, record what was requested
                        if 'args' in entry['event']:
                            action['items_requested'] = entry['event']['args'].get('items') or entry['event']['args'].get('item', 'all')
                    
                    actions.append(action)
                except:
                    continue
        
        return actions
    
    def reset(self):
        """Reset the ledger (creates backup if file exists)."""
        with self.lock:
            if self.storage_path.exists():
                # Create backup
                backup_path = self.storage_path.with_suffix(
                    f".backup_{datetime.now(pytz.timezone('America/New_York')).isoformat()}.jsonl"
                )
                os.rename(self.storage_path, backup_path)
            
            self._document_coverage = {}
            self._last_reads = {}