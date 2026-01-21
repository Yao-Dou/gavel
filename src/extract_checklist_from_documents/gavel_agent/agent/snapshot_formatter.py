"""
Markdown formatter for snapshots - creates human-readable prompts for the LLM.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from state.schemas import Snapshot, DocumentInfo, ActionRecord, Evidence


class SnapshotFormatter:
    """
    Formats snapshots as readable markdown for better LLM comprehension.
    """
    
    @staticmethod
    def format_as_markdown(snapshot: Snapshot) -> str:
        """
        Format snapshot as markdown for the user prompt.
        
        Args:
            snapshot: The snapshot to format
            
        Returns:
            Markdown-formatted string
        """
        sections = []
        
        # Header with context
        sections.append(SnapshotFormatter._format_header(snapshot))
        
        # Full action history (if any actions beyond the recent actions limit)
        if len(snapshot.action_tail) > snapshot.recent_actions_detail:
            sections.append(SnapshotFormatter._format_full_action_history(snapshot))
        
        # Recent actions (if any) - show history first before current state
        if snapshot.action_tail:
            sections.append(SnapshotFormatter._format_recent_actions(snapshot))
        
        # Current status
        sections.append(SnapshotFormatter._format_status(snapshot))
        
        # Documents section
        sections.append(SnapshotFormatter._format_documents(snapshot))
        
        # Progress section
        sections.append(SnapshotFormatter._format_progress(snapshot))
        
        # Stop status section (only on the step immediately after auto get_checklist)
        # This appears only when current_step == first_stop_step + 2
        # (first_stop_step + 1 is the auto get_checklist, +2 is when model sees results)
        if (snapshot.stop_count > 0 and 
            snapshot.first_stop_step is not None and 
            snapshot.run_header.step == snapshot.first_stop_step + 2):
            sections.append(SnapshotFormatter._format_stop_status(snapshot))
        
        # Decision prompt
        sections.append(SnapshotFormatter._format_decision_prompt())
        
        return "\n\n".join(sections)
    
    @staticmethod
    def _format_header(snapshot: Snapshot) -> str:
        """Format the header section."""
        header = f"""# Legal Checklist Extraction
**Step {snapshot.run_header.step}**

## Your Task
{snapshot.task.user_instruction}"""

        # Only show additional constraints if there are task-specific ones
        if snapshot.task.constraints:
            header += f"\n## Requirements"
            for constraint in snapshot.task.constraints:
                header += f"\n- {constraint}"
            header += "\n"

        # Add checklist definitions if available
        if snapshot.task.checklist_definitions:
            header += "\n## Checklist Items to Extract"
            for key, description in snapshot.task.checklist_definitions.items():
                header += f"\n- **{key}**: {description}"
        
        return header
    
    @staticmethod
    def _format_status(snapshot: Snapshot) -> str:
        """Format the current status section."""
        # Calculate summary from full checklist
        checklist = snapshot.checklist
        extracted_count = sum(1 for item in checklist if item.extracted)
        empty_count = sum(1 for item in checklist if not item.extracted)
        total_keys = len(checklist)
        total_values = sum(len(item.extracted) for item in checklist if item.extracted)
        
        # Count "Not Applicable" items
        not_applicable_count = sum(
            1 for item in checklist 
            if len(item.extracted) == 1 and item.extracted[0].value == "Not Applicable"
        )
        
        status = f"""## Current Status
- **Keys with Values**: {extracted_count}/{total_keys}
- **Empty Keys**: {empty_count}/{total_keys}
- **Not Applicable**: {not_applicable_count}/{total_keys}
- **Total Values Extracted**: {total_values}
- **Documents in Corpus**: {len(snapshot.documents)}"""
        
        return status
    
    @staticmethod
    def _calculate_coverage_tokens(token_ranges: list) -> int:
        """Calculate total unique tokens covered by the ranges."""
        if not token_ranges:
            return 0
        
        # Merge overlapping ranges first
        sorted_ranges = sorted(token_ranges, key=lambda x: x[0])
        merged = []
        
        for start, end in sorted_ranges:
            if merged and start <= merged[-1][1]:
                # Overlapping or adjacent - extend the last range
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # Non-overlapping - add new range
                merged.append((start, end))
        
        # Calculate total covered tokens
        total = sum(end - start for start, end in merged)
        return total
    
    @staticmethod
    def _format_documents(snapshot: Snapshot) -> str:
        """Format the documents section with detailed visit status."""
        if not snapshot.documents:
            return """## Available Documents
No documents discovered yet."""
        
        lines = ["## Available Documents"]
        for doc in snapshot.documents:
            # Determine visit status based on coverage
            if doc.visited and doc.coverage and hasattr(doc.coverage, 'token_ranges') and doc.coverage.token_ranges:
                covered_tokens = SnapshotFormatter._calculate_coverage_tokens(doc.coverage.token_ranges)
                coverage_percentage = (covered_tokens / doc.token_count * 100) if doc.token_count > 0 else 0
                
                # Classify as fully or partially visited
                if covered_tokens >= doc.token_count:
                    status = "✓ Fully Visited"
                else:
                    status = "◐ Partially Visited"
            else:
                status = "○ Unvisited"
                coverage_percentage = 0
            
            # Format the main document line
            lines.append(f"- **{doc.name}** [{doc.type}] - {doc.token_count:,} tokens - {status}")
            
            # Show token ranges if document has been visited
            if doc.visited and doc.coverage and hasattr(doc.coverage, 'token_ranges') and doc.coverage.token_ranges:
                # Show all token ranges without truncation
                ranges = doc.coverage.token_ranges
                range_str = ", ".join([f"{start}-{end}" for start, end in ranges])
                lines.append(f"  Viewed tokens: {range_str} ({coverage_percentage:.0f}% coverage)")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_progress(snapshot: Snapshot) -> str:
        """Format the progress section with document-level breakdown."""
        checklist = snapshot.checklist
        
        lines = ["## Extraction Progress"]
        
        # Categorize items
        extracted_items = []
        not_applicable_items = []
        empty_items = []
        
        for item in checklist:
            if not item.extracted:
                empty_items.append(item)
            elif len(item.extracted) == 1 and item.extracted[0].value == "Not Applicable":
                not_applicable_items.append(item)
            else:
                extracted_items.append(item)
        
        # Show keys with extracted values with document breakdown
        if extracted_items:
            lines.append(f"\n**Keys with Extracted Values** ({len(extracted_items)}):")
            for item in extracted_items:
                # Count how many extracted values reference each document
                doc_values = {}
                for extracted in item.extracted:
                    # Get unique documents for this extracted value
                    unique_docs_for_value = set()
                    for evidence in extracted.evidence:
                        unique_docs_for_value.add(evidence.source_document)
                    
                    # Count this value for each unique document it references
                    for doc in unique_docs_for_value:
                        if doc not in doc_values:
                            doc_values[doc] = 0
                        doc_values[doc] += 1
                
                # Format the item with document breakdown
                total_values = len(item.extracted)
                value_text = "value" if total_values == 1 else "values"
                doc_info = ", ".join([f"{count} from {doc}" for doc, count in doc_values.items()])
                lines.append(f"- **{item.key}**: {total_values} {value_text} ({doc_info})")
        
        # Show Not Applicable items
        if not_applicable_items:
            lines.append(f"\n**Keys Marked as Not Applicable** ({len(not_applicable_items)}):")
            for item in not_applicable_items:
                # Get the evidence document for Not Applicable
                evidence_doc = item.extracted[0].evidence[0].source_document if item.extracted[0].evidence else "unknown"
                lines.append(f"- **{item.key}**: Not Applicable (evidence from {evidence_doc})")
        
        # Show empty keys
        if empty_items:
            lines.append(f"\n**Keys Not Yet Explored** ({len(empty_items)}):")
            for item in empty_items:
                lines.append(f"- **{item.key}**: Empty")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_recent_actions(snapshot: Snapshot) -> str:
        """Format recent actions section with full tool names, arguments, and detailed results."""
        # Get the last N actions (or all if less than N)
        recent_actions = snapshot.action_tail[-snapshot.recent_actions_detail:] if len(snapshot.action_tail) > snapshot.recent_actions_detail else snapshot.action_tail
        
        # Adjust the title based on whether we have a full history section and show step range
        if len(snapshot.action_tail) > snapshot.recent_actions_detail and recent_actions:
            first_step = recent_actions[0].step
            last_step = recent_actions[-1].step
            lines = [f"## Recent Actions (Steps {first_step}-{last_step} with Detailed Results)"]
        elif recent_actions:
            first_step = recent_actions[0].step
            last_step = recent_actions[-1].step
            lines = [f"## Recent Actions (Steps {first_step}-{last_step})"]
        else:
            lines = ["## Recent Actions"]
        
        # Show detailed results for each action
        for action in recent_actions:
            # Use the preserved step number from the action
            step_number = action.step
            # Format the action line with full tool name and arguments
            action_line = SnapshotFormatter._format_action_line(action, step_number)
            lines.append(action_line)
            
            # Add detailed result for each action
            if action.result_summary:
                result_lines = SnapshotFormatter._format_action_result(
                    action.result_summary, 
                    action.tool,
                    indent="   ",
                    action=action
                )
                lines.extend(result_lines)
            elif action.error:
                # Show error details
                lines.append(f"   **❌ ERROR**: {action.error}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_action_line(action: ActionRecord, step_number: int) -> str:
        """Format a single action line with step number, tool name and arguments."""
        # Start with step number and full tool name
        line_parts = [f"Step {step_number}. `{action.tool}`"]
        
        # Add arguments based on tool type
        if action.tool == "list_documents":
            # No arguments
            pass
        
        elif action.tool == "search_document_regex":
            if action.target:
                # Handle multi-document searches
                doc_names = action.target.get("doc_names", [])
                doc_name = action.target.get("doc_name", "")
                pattern = action.target.get("pattern", "")
                # Never truncate pattern - show full pattern
                
                # Format document specification
                if doc_names and len(doc_names) > 0:
                    if len(doc_names) == 1:
                        line_parts.append(f"on {doc_names[0]} (pattern: \"{pattern}\")")
                    else:
                        # For multiple documents, show all names
                        docs_str = ", ".join(doc_names)
                        line_parts.append(f"on [{docs_str}] (pattern: \"{pattern}\")")
                elif doc_name == "all":
                    line_parts.append(f"on all documents (pattern: \"{pattern}\")")
                elif doc_name:
                    line_parts.append(f"on {doc_name} (pattern: \"{pattern}\")")
        
        elif action.tool == "read_document":
            if action.target:
                doc_name = action.target.get("doc_name", "")
                start = action.target.get("start_token", 0)
                end = action.target.get("end_token", 0)
                line_parts.append(f"on {doc_name} (tokens {start}-{end})")
        
        elif action.tool in ["update_checklist", "append_checklist"]:
            if action.changed_keys:
                keys_str = ", ".join(action.changed_keys[:3])  # Show first 3 keys
                if len(action.changed_keys) > 3:
                    keys_str += f", +{len(action.changed_keys)-3} more"
                line_parts.append(f"({keys_str})")
            elif action.target and "patch" in action.target:
                # Extract keys from patch if changed_keys not available
                patch = action.target.get("patch", [])
                keys = [p.get("key") for p in patch if isinstance(p, dict) and "key" in p]
                if keys:
                    keys_str = ", ".join(keys[:3])
                    if len(keys) > 3:
                        keys_str += f", +{len(keys)-3} more"
                    line_parts.append(f"({keys_str})")
        
        elif action.tool == "get_checklist":
            if action.target:
                item = action.target.get("item", "all")
                if item != "all":
                    line_parts.append(f"(item: {item})")
        
        elif action.tool in ["parse_error", "validation_error"]:
            # For error actions, show the error message inline
            if action.target and isinstance(action.target, dict):
                error_msg = action.target.get("error", "Unknown error")
                # Truncate if too long
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                line_parts.append(f"- {error_msg}")
        
        elif action.tool == "stop":
            # For stop actions, show the reason
            if action.target and isinstance(action.target, dict):
                reason = action.target.get("reason", "No reason provided")
                line_parts.append(f"- {reason}")
        
        # Add auto-generated indicator if applicable
        if action.auto_generated:
            line_parts.append("[AUTO-GENERATED]")
        
        # Add error indicator if failed
        if action.error:
            line_parts.append("**❌ ERROR**")
        elif action.validation_errors:
            line_parts.append(f"**⚠️ {len(action.validation_errors)} validation error(s)**")
        
        return " ".join(line_parts)
    
    @staticmethod
    def _format_result_snippet(action: ActionRecord) -> str:
        """Format a brief snippet of the action result."""
        # Handle special actions (stop, parse_error, validation_error)
        if action.tool == "stop":
            # Stop actions don't have results to show as a snippet
            reason = ""
            if action.target and isinstance(action.target, dict):
                reason = action.target.get("reason", "No reason provided")
            if len(reason) > 60:
                reason = reason[:60] + "..."
            return f"Stop attempt: {reason}"
        
        elif action.tool in ["parse_error", "validation_error"]:
            error_msg = ""
            if action.target and isinstance(action.target, dict):
                error_msg = action.target.get("error", "Unknown error")
            elif action.error:
                error_msg = action.error
            else:
                error_msg = "Parse/validation failure"
            
            if len(error_msg) > 60:
                error_msg = error_msg[:60] + "..."
            return f"**ERROR**: {error_msg}"
        
        if not action.result_summary or action.error:
            if action.error:
                # Truncate error message if too long
                error_msg = str(action.error)
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                return f"Error: {error_msg}"
            return ""
        
        result = action.result_summary
        
        if action.tool == "list_documents":
            docs = result.get("documents", [])
            return f"Found {len(docs)} documents"
        
        elif action.tool == "search_document_regex":
            # Handle multi-document search results
            if "results" in result:  # New multi-document format
                doc_results = result.get("results", [])
                total = result.get("total_matches", 0)
                docs_with_matches = [d for d in doc_results if d.get("matches", [])]
                
                if total == 0:
                    return "No matches found"
                elif len(docs_with_matches) == 1:
                    # Single document with matches
                    doc_result = docs_with_matches[0]
                    matches = doc_result.get("matches", [])
                    return f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} in {doc_result.get('doc_name', 'unknown')}"
                else:
                    # Multiple documents with matches
                    # Show which documents had matches
                    doc_summary = []
                    for doc in docs_with_matches[:3]:  # Show first 3 docs with matches
                        doc_name = doc.get('doc_name', 'unknown')
                        match_count = len(doc.get('matches', []))
                        doc_summary.append(f"{match_count} in {doc_name}")
                    
                    summary = f"Found {total} matches: {', '.join(doc_summary)}"
                    if len(docs_with_matches) > 3:
                        summary += f" (+{len(docs_with_matches)-3} more docs)"
                    return summary
            else:  # Old single-document format (fallback)
                matches = result.get("matches", [])
                if not matches:
                    return "No matches found"
                return f"Found {len(matches)} match{'es' if len(matches) != 1 else ''}"
        
        elif action.tool == "read_document":
            tokens_read = result.get("end_token", 0) - result.get("start_token", 0)
            doc_name = result.get("doc_name", "")
            return f"Read {tokens_read} tokens"
        
        elif action.tool in ["update_checklist", "append_checklist"]:
            updated = result.get("updated_keys", result.get("appended_keys", []))
            if updated:
                # Count total values from the patch in action.target
                total_values = 0
                if action.target and "patch" in action.target:
                    patch = action.target.get("patch", [])
                    for patch_item in patch:
                        if isinstance(patch_item, dict) and "extracted" in patch_item:
                            total_values += len(patch_item.get("extracted", []))
                
                # Format message based on tool type
                operation = "updated" if action.tool == "update_checklist" else "appended"
                key_word = "key" if len(updated) == 1 else "keys"
                value_word = "value" if total_values == 1 else "values"
                
                if total_values > 0:
                    return f"Successfully {operation} {total_values} {value_word} for {len(updated)} {key_word}: {', '.join(updated)}"
                else:
                    # Fallback if we can't count values
                    return f"Successfully {operation} {len(updated)} {key_word}: {', '.join(updated)}"
            elif result.get("validation_errors"):
                return f"Validation failed: {result['validation_errors'][0][:40]}"
            return "No changes made"
        
        elif action.tool == "get_checklist":
            stats = result.get("completion_stats", {})
            filled = stats.get("filled", 0)
            # Calculate total from filled + empty if not provided
            total = stats.get("total", stats.get("filled", 0) + stats.get("empty", 0))
            return f"{filled}/{total} keys have values"
        
        return ""
    
    @staticmethod
    def _format_last_result(result: Dict[str, Any], tool_name: Optional[str] = None) -> str:
        """Format the last tool result - show full results for specific tools, brief for others."""
        # Tools that need detailed output
        TOOLS_WITH_FULL_RESULTS = {"read_document", "search_document_regex", "get_checklist"}
        
        lines = ["## Last Tool Result"]
        
        # ALWAYS check for errors first, regardless of tool
        if "error" in result:
            lines.append(f"**❌ ERROR**: {result['error']}")
            if tool_name:
                lines.append(f"Tool: `{tool_name}`")
            # Show any additional fields
            other_fields = {k: v for k, v in result.items() if k != "error"}
            if other_fields:
                lines.append("Additional info:")
                lines.append(f"```json\n{json.dumps(other_fields, indent=2, default=str)}\n```")
            return "\n".join(lines)
        
        # Handle results based on tool type
        if tool_name in TOOLS_WITH_FULL_RESULTS:
            # Full results for specific tools
            if tool_name == "read_document":
                # read_document result - show the FULL text that was read
                # The model requested this specific range, so it needs to see all of it
                full_text = result.get('text', '')
                lines.append(f"Read from **{result.get('doc_name', 'unknown')}** (tokens {result.get('start_token', 0)}-{result.get('end_token', 0)}):")
                lines.append(f"```\n{full_text}\n```")
                
            elif tool_name == "search_document_regex":
                # search_document_regex result - show up to 20 total matches
                # Handle both multi-document and single-document formats
                if "results" in result:  # New multi-document format
                    doc_results = result.get('results', [])
                    total_matches = result.get('total_matches', 0)
                    docs_searched = result.get('documents_searched', [])
                    
                    lines.append(f"Search in {len(docs_searched)} document{'s' if len(docs_searched) != 1 else ''} found {total_matches} total match{'es' if total_matches != 1 else ''}:")
                    
                    # Show up to 20 matches total across all documents
                    matches_shown = 0
                    MAX_TOTAL_MATCHES = 20
                    
                    for doc_result in doc_results:
                        if matches_shown >= MAX_TOTAL_MATCHES:
                            break
                            
                        doc_name = doc_result.get('doc_name', 'unknown')
                        matches = doc_result.get('matches', [])
                        
                        # Calculate how many matches to show from this document
                        matches_to_show = min(len(matches), MAX_TOTAL_MATCHES - matches_shown)
                        
                        if matches_to_show > 0:
                            lines.append(f"\n**{doc_name}** ({len(matches)} match{'es' if len(matches) != 1 else ''}):")
                            
                            for i, match in enumerate(matches[:matches_to_show], 1):
                                snippet = match.get('snippet', '')
                                lines.append(f"\n  Match {i} (tokens {match.get('start_token', 0)}-{match.get('end_token', 0)}):")
                                lines.append(f"```\n{snippet}\n```")
                                matches_shown += 1
                    
                    # If there are more matches, provide a summary
                    if total_matches > MAX_TOTAL_MATCHES:
                        lines.append(f"\n**Showing first {MAX_TOTAL_MATCHES} of {total_matches} total matches.**")
                        # Calculate remaining matches per document
                        remaining_summary = []
                        for doc_result in doc_results:
                            doc_name = doc_result.get('doc_name', 'unknown')
                            doc_matches = len(doc_result.get('matches', []))
                            shown_from_doc = min(doc_matches, MAX_TOTAL_MATCHES) if matches_shown <= MAX_TOTAL_MATCHES else 0
                            remaining = doc_matches - shown_from_doc
                            if remaining > 0:
                                remaining_summary.append(f"{remaining} in {doc_name}")
                        
                        if remaining_summary:
                            lines.append(f"Remaining matches: {', '.join(remaining_summary[:5])}")
                            if len(remaining_summary) > 5:
                                lines.append(f"... and {len(remaining_summary) - 5} more documents with matches")
                            lines.append("Search individual documents for complete results.")
                        
                else:  # Old single-document format (fallback)
                    matches = result.get('matches', [])
                    lines.append(f"Search in **{result.get('doc_name', 'unknown')}** found {len(matches)} matches:")
                    # Show up to 20 matches for single document
                    for i, match in enumerate(matches[:20], 1):
                        snippet = match.get('snippet', '')
                        lines.append(f"\n**Match {i}** (tokens {match.get('start_token', 0)}-{match.get('end_token', 0)}):")
                        lines.append(f"```\n{snippet}\n```")
                    
                    if len(matches) > 20:
                        lines.append(f"\n**Showing first 20 of {len(matches)} matches. Search again with more specific pattern for other matches.**")
                
            elif tool_name == "get_checklist":
                # get_checklist result - show full checklist status
                stats = result.get('completion_stats', {})
                total_keys = stats.get('total', stats.get('filled', 0) + stats.get('empty', 0))
                lines.append(f"Checklist Status: {stats.get('filled', 0)}/{total_keys} filled, {stats.get('empty', 0)} empty")
                
                # Show some details about what's filled
                checklist = result.get('checklist', [])
                filled_items = [item for item in checklist if item.get('extracted')]
                if filled_items:
                    lines.append("\nFilled keys:")
                    for item in filled_items[:10]:  # Show first 10
                        key = item.get('key', 'unknown')
                        value_count = len(item.get('extracted', []))
                        lines.append(f"- {key}: {value_count} value{'s' if value_count != 1 else ''}")
        else:
            # Brief results for other tools (similar to what would appear in Recent Actions)
            if tool_name == "list_documents":
                docs = result.get('documents', [])
                lines.append(f"Found {len(docs)} documents")
                
            elif tool_name in ["update_checklist", "append_checklist"]:
                updated = result.get('updated_keys', result.get('appended_keys', []))
                if updated:
                    # Show ALL updated keys - the model needs to see the full list
                    lines.append(f"Successfully updated {len(updated)} key{'s' if len(updated) != 1 else ''}: {', '.join(updated)}")
                elif result.get('validation_errors'):
                    # Show full validation errors, not truncated
                    for error in result.get('validation_errors', []):
                        lines.append(f"Validation failed: {error}")
                else:
                    lines.append("No changes made")
            
            else:
                # Generic result display for unknown tools
                lines.append(f"Tool: `{tool_name}`")
                # Show a brief summary of the result
                if isinstance(result, dict):
                    # Show key metrics if available
                    key_count = len(result)
                    lines.append(f"Returned {key_count} field{'s' if key_count != 1 else ''}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_action_result(result: Dict[str, Any], tool_name: str, indent: str = "", action: Optional[ActionRecord] = None) -> List[str]:
        """Format detailed result for an action."""
        # Tools that need detailed output - now including append/update checklist
        TOOLS_WITH_FULL_RESULTS = {"read_document", "search_document_regex", "get_checklist", "append_checklist", "update_checklist"}
        
        lines = []
        
        # Check for errors first
        if "error" in result:
            lines.append(f"{indent}**❌ ERROR**: {result['error']}")
            # Show any additional fields
            other_fields = {k: v for k, v in result.items() if k != "error"}
            if other_fields:
                lines.append(f"{indent}Additional info:")
                json_str = json.dumps(other_fields, indent=2, default=str)
                indented_json = "\n".join(f"{indent}{line}" for line in json_str.split("\n"))
                lines.append(f"{indent}```json\n{indented_json}\n{indent}```")
            return lines
        
        # Handle results based on tool type
        if tool_name in TOOLS_WITH_FULL_RESULTS:
            # Full results for specific tools
            if tool_name == "read_document":
                # read_document result - show the FULL text that was read
                full_text = result.get('text', '')
                lines.append(f"{indent}Read from **{result.get('doc_name', 'unknown')}** (tokens {result.get('start_token', 0)}-{result.get('end_token', 0)}):")
                lines.append(f"{indent}```")
                # Indent the text content
                for text_line in full_text.split('\n'):
                    lines.append(f"{indent}{text_line}")
                lines.append(f"{indent}```")
                
            elif tool_name == "search_document_regex":
                # search_document_regex result - show up to 20 total matches
                # Handle both multi-document and single-document formats
                if "results" in result:  # New multi-document format
                    doc_results = result.get('results', [])
                    total_matches = result.get('total_matches', 0)
                    docs_searched = result.get('documents_searched', [])
                    
                    lines.append(f"{indent}Search in {len(docs_searched)} document{'s' if len(docs_searched) != 1 else ''} found {total_matches} total match{'es' if total_matches != 1 else ''}:")
                    
                    # Show up to 20 matches total across all documents
                    matches_shown = 0
                    MAX_TOTAL_MATCHES = 20
                    
                    for doc_result in doc_results:
                        if matches_shown >= MAX_TOTAL_MATCHES:
                            break
                            
                        doc_name = doc_result.get('doc_name', 'unknown')
                        matches = doc_result.get('matches', [])
                        
                        # Calculate how many matches to show from this document
                        matches_to_show = min(len(matches), MAX_TOTAL_MATCHES - matches_shown)
                        
                        if matches_to_show > 0:
                            lines.append(f"{indent}**{doc_name}** ({len(matches)} match{'es' if len(matches) != 1 else ''}):")
                            
                            for i, match in enumerate(matches[:matches_to_show], 1):
                                snippet = match.get('snippet', '')
                                lines.append(f"{indent}  Match {i} (tokens {match.get('start_token', 0)}-{match.get('end_token', 0)}):")
                                lines.append(f"{indent}  ```")
                                # Indent the snippet content
                                for snippet_line in snippet.split('\n'):
                                    lines.append(f"{indent}  {snippet_line}")
                                lines.append(f"{indent}  ```")
                                matches_shown += 1
                    
                    # If there are more matches, provide a summary
                    if total_matches > MAX_TOTAL_MATCHES:
                        lines.append(f"{indent}**Showing first {MAX_TOTAL_MATCHES} of {total_matches} total matches.**")
                        # Calculate remaining matches per document
                        remaining_summary = []
                        for doc_result in doc_results:
                            doc_name = doc_result.get('doc_name', 'unknown')
                            doc_matches = len(doc_result.get('matches', []))
                            # Fix calculation - count how many were actually shown from this doc
                            shown_count = 0
                            for shown_doc in doc_results:
                                if shown_doc.get('doc_name') == doc_name:
                                    shown_count = min(doc_matches, MAX_TOTAL_MATCHES - sum(len(d.get('matches', [])) for d in doc_results[:doc_results.index(shown_doc)]))
                                    shown_count = max(0, shown_count)
                                    break
                            remaining = doc_matches - shown_count
                            if remaining > 0:
                                remaining_summary.append(f"{remaining} in {doc_name}")
                        
                        if remaining_summary:
                            lines.append(f"{indent}Remaining matches: {', '.join(remaining_summary[:5])}")
                            if len(remaining_summary) > 5:
                                lines.append(f"{indent}... and {len(remaining_summary) - 5} more documents with matches")
                            lines.append(f"{indent}Search individual documents for complete results.")
                        
                else:  # Old single-document format (fallback)
                    matches = result.get('matches', [])
                    lines.append(f"{indent}Search in **{result.get('doc_name', 'unknown')}** found {len(matches)} matches:")
                    # Show up to 20 matches for single document
                    for i, match in enumerate(matches[:20], 1):
                        snippet = match.get('snippet', '')
                        lines.append(f"{indent}**Match {i}** (tokens {match.get('start_token', 0)}-{match.get('end_token', 0)}):")
                        lines.append(f"{indent}```")
                        # Indent the snippet content
                        for snippet_line in snippet.split('\n'):
                            lines.append(f"{indent}{snippet_line}")
                        lines.append(f"{indent}```")
                    
                    if len(matches) > 20:
                        lines.append(f"{indent}**Showing first 20 of {len(matches)} matches. Search again with more specific pattern for other matches.**")
                
            elif tool_name == "get_checklist":
                # get_checklist result - show full checklist status
                stats = result.get('completion_stats', {})
                total_keys = stats.get('total', stats.get('filled', 0) + stats.get('empty', 0))
                lines.append(f"{indent}Checklist Status: {stats.get('filled', 0)}/{total_keys} filled, {stats.get('empty', 0)} empty")
                
                # Show some details about what's filled
                checklist = result.get('checklist', [])
                filled_items = [item for item in checklist if item.get('extracted')]
                if filled_items:
                    lines.append(f"{indent}Filled keys:")
                    for item in filled_items[:10]:  # Show first 10
                        key = item.get('key', 'unknown')
                        value_count = len(item.get('extracted', []))
                        lines.append(f"{indent}- {key}: {value_count} value{'s' if value_count != 1 else ''}")
            
            elif tool_name in ["append_checklist", "update_checklist"]:
                # Show the actual extracted values that were appended/updated
                updated_keys = result.get('updated_keys', result.get('appended_keys', []))
                operation = "updated" if tool_name == "update_checklist" else "appended"
                
                if updated_keys:
                    lines.append(f"{indent}→ Successfully {operation} {len(updated_keys)} key{'s' if len(updated_keys) != 1 else ''}: {', '.join(updated_keys)}")
                    
                    # Get the patch data from the action's target to show what was actually added
                    if action and action.target and 'patch' in action.target:
                        patch = action.target.get('patch', [])
                        lines.append(f"{indent}")
                        lines.append(f"{indent}**Extracted Values:**")
                        
                        for patch_item in patch:
                            key = patch_item.get('key', 'unknown')
                            extracted_items = patch_item.get('extracted', [])
                            
                            if extracted_items:
                                lines.append(f"{indent}• **{key}**:")
                                
                                for idx, extracted in enumerate(extracted_items, 1):
                                    value = extracted.get('value', '')
                                    evidence_list = extracted.get('evidence', [])
                                    
                                    # Check if this is a "Not Applicable" value
                                    if value == "Not Applicable":
                                        lines.append(f"{indent}  → Value: **Not Applicable**")
                                    else:
                                        lines.append(f"{indent}  → Value {idx}: \"{value}\"")
                                    
                                    # Show evidence for this value
                                    if evidence_list:
                                        for ev_idx, evidence in enumerate(evidence_list, 1):
                                            source_doc = evidence.get('source_document', 'unknown')
                                            location = evidence.get('location', 'unknown')
                                            text = evidence.get('text', '')
                                            
                                            # Truncate very long evidence text
                                            if len(text) > 200:
                                                text = text[:200] + "..."
                                            
                                            lines.append(f"{indent}    Evidence {ev_idx} from {source_doc} ({location}):")
                                            lines.append(f"{indent}    \"{text}\"")
                
                elif result.get('validation_errors'):
                    # Show validation errors
                    for error in result.get('validation_errors', []):
                        lines.append(f"{indent}→ Validation failed: {error}")
                else:
                    lines.append(f"{indent}→ No changes made")
            
            elif tool_name == "stop":
                # For stop actions, show the stop attempt details
                if action and action.target and isinstance(action.target, dict):
                    reason = action.target.get("reason", "No reason provided")
                    remaining_keys = action.target.get("remaining_empty_keys", [])
                    
                    lines.append(f"{indent}→ **Stop Attempt**")
                    lines.append(f"{indent}  Reason: {reason}")
                    
                    if remaining_keys:
                        lines.append(f"{indent}  Remaining empty keys: {len(remaining_keys)}")
                        for key in remaining_keys[:5]:  # Show first 5
                            lines.append(f"{indent}    - {key}")
                        if len(remaining_keys) > 5:
                            lines.append(f"{indent}    ... and {len(remaining_keys) - 5} more")
            
            elif tool_name in ["parse_error", "validation_error"]:
                # For error actions, show the detailed error information
                if action and action.target and isinstance(action.target, dict):
                    error_msg = action.target.get("error", "Unknown error")
                    retry_count = action.target.get("retry_count", 0)
                    
                    lines.append(f"{indent}→ **{('Parse' if tool_name == 'parse_error' else 'Validation')} Error**:")
                    lines.append(f"{indent}  {error_msg}")
                    
                    if retry_count > 0:
                        lines.append(f"{indent}  (After {retry_count} retry attempt{'s' if retry_count != 1 else ''})")
                    
                    # If there's additional debug info
                    if "_attempted_function" in action.target:
                        lines.append(f"{indent}  Attempted function: {action.target['_attempted_function']}")
                    if "_raw_args" in action.target:
                        raw_args = action.target["_raw_args"]
                        if len(raw_args) > 100:
                            raw_args = raw_args[:100] + "..."
                        lines.append(f"{indent}  Raw arguments: {raw_args}")
        else:
            # Brief results for other tools
            if tool_name == "list_documents":
                docs = result.get('documents', [])
                lines.append(f"{indent}→ Found {len(docs)} documents")
            
            else:
                # Generic result display for unknown tools
                if isinstance(result, dict):
                    # Show key metrics if available
                    key_count = len(result)
                    lines.append(f"{indent}→ Returned {key_count} field{'s' if key_count != 1 else ''}")
        
        return lines
    
    @staticmethod
    def _format_full_action_history(snapshot: Snapshot) -> str:
        """Format the complete action history for all actions beyond the recent actions limit."""
        # Get all actions except the last N (which will be shown in Recent Actions)
        all_actions = snapshot.action_tail
        if len(all_actions) <= snapshot.recent_actions_detail:
            return ""  # No need for this section if we have N or fewer actions
        
        older_actions = all_actions[:-snapshot.recent_actions_detail]  # All except last N
        
        # Get the step range for the headline
        if older_actions:
            first_step = older_actions[0].step
            last_step = older_actions[-1].step
            lines = [f"## Prior Actions (Steps {first_step}-{last_step})"]
        else:
            lines = ["## Prior Actions"]
        
        # Show older actions with brief results
        for action in older_actions:
            # Use the preserved step number from the action
            step_number = action.step
            
            # Format the action line with full details (no truncation)
            action_line = SnapshotFormatter._format_action_line(action, step_number)
            lines.append(action_line)
            
            # Add brief result summary
            if action.result_summary and not action.error:
                result_snippet = SnapshotFormatter._format_result_snippet(action)
                if result_snippet:
                    lines.append(f"   → {result_snippet}")
            elif action.error:
                error_msg = str(action.error)
                if len(error_msg) > 80:
                    error_msg = error_msg[:80] + "..."
                lines.append(f"   → Error: {error_msg}")
            elif action.validation_errors:
                lines.append(f"   → Validation failed: {len(action.validation_errors)} error(s)")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_stop_status(snapshot: Snapshot) -> str:
        """Format the stop status section when after first stop."""
        return f"""## 🔄 Automatic Checklist Review After First Stop

**What happened**: At step {snapshot.first_stop_step}, you called `stop` for the first time. The system has automatically 
executed `get_checklist("all")` to provide you with a complete view of the extraction progress.

**Your decision now**: 
- Call `stop` again if you're satisfied with the current extraction state
- Continue with more tool calls if you identify gaps or need to extract more items
- This is stop attempt {snapshot.stop_count}/2 (system will accept second stop as final)"""
    
    @staticmethod
    def _format_decision_prompt() -> str:
        """Format the decision prompt section."""
        return """## Your Next Action

Choose **ONE** action based on the current state:

1. **`list_documents`** — if the document catalog hasn't been discovered.
2. **`read_document`** — to read a targeted text window (≤10,000 tokens) in a document.
3. **`search_document_regex`** — to search one, many, or all documents for a regex pattern and return top matches with surrounding context.
4. **`get_checklist`** — retrieve the extracted values for either all keys or a specified key.
5. **`append_checklist`** — to add newly extracted values (you can batch multiple keys/entries in one call) based on the text just read/searched.
6. **`update_checklist`** — to replace the entire extracted list for one or more keys or to set "Not Applicable" based on the text just read/searched (batching allowed as well).
7. **`stop`** — when every key is complete or Not Applicable, or no further progress is possible.
```"""