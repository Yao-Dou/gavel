"""
Search document regex tool - searches documents using regular expressions.
"""

from typing import Dict, Any, List
from pydantic import ValidationError
from .base import BaseTool
from state.store import Ledger
from state.schemas import (
    SearchDocumentRegexInput, SearchDocumentRegexOutput, 
    RegexMatch, SearchEvent, DocumentSearchResult
)
from agent.document_manager import DocumentManager


class SearchDocumentRegexTool(BaseTool):
    """
    Tool for searching documents using regular expressions.
    Supports searching single documents, multiple documents, or all documents.
    Returns matches with context windows and records searches in the ledger.
    """
    
    def __init__(self, document_manager: DocumentManager, ledger: Ledger = None):
        """
        Initialize the search_document_regex tool.
        
        Args:
            document_manager: DocumentManager instance for accessing documents
            ledger: Optional Ledger instance for recording searches
        """
        super().__init__(
            name="search_document_regex",
            description="Search one or more documents using regular expression patterns"
        )
        self.document_manager = document_manager
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
        Get the input schema for search_document_regex.
        Supports three patterns:
        1. doc_names array - search specific documents
        2. doc_name="all" - search all documents
        3. doc_name="<name>" - search single document
        
        Returns:
            Schema for SearchDocumentRegexInput
        """
        return {
            "type": "object",
            "properties": {
                "doc_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of document names to search"
                },
                "doc_name": {
                    "type": "string",
                    "description": "Single document name or 'all' for all documents"
                },
                "pattern": {"type": "string"},
                "flags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["IGNORECASE", "MULTILINE", "DOTALL"]
                    },
                    "default": []
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "context_tokens": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 1000,
                    "default": 200
                }
            },
            "required": ["pattern"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for SearchDocumentRegexOutput with multi-document support
        """
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "documents_searched": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of documents that were searched"
                },
                "results": {
                    "type": "array",
                    "description": "Per-document search results",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_name": {"type": "string"},
                            "match_count": {"type": "integer"},
                            "matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_token": {"type": "integer"},
                                        "end_token": {"type": "integer"},
                                        "start_char": {"type": "integer"},
                                        "end_char": {"type": "integer"},
                                        "snippet": {"type": "string"},
                                        "groups": {"type": "object"},
                                        "pattern": {"type": "string"},
                                        "flags": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "total_matches": {"type": "integer"}
            },
            "required": ["pattern", "documents_searched", "results", "total_matches"]
        }
    
    def _find_best_matching_document(self, query_name: str, available_docs: list) -> str:
        """
        Find the best matching document name using fuzzy matching.
        Handles cases where the model uses abbreviated names like "Answer..." or "...Counterclaims".
        
        Args:
            query_name: The document name provided by the model (may be abbreviated)
            available_docs: List of actual document names available
            
        Returns:
            The best matching document name
            
        Raises:
            ValueError: If no reasonable match is found
        """
        # Clean the query name
        query_clean = query_name.strip()
        
        # 1. Check for exact match first
        if query_clean in available_docs:
            return query_clean
        
        # 2. Check for case-insensitive exact match
        for doc in available_docs:
            if doc.lower() == query_clean.lower():
                return doc
        
        # 3. Handle ellipsis patterns ("...", "…")
        # Split by ellipsis to get prefix and suffix parts
        ellipsis_patterns = ['...', '…', '. . .', '..', '….', '....']
        query_parts = []
        for pattern in ellipsis_patterns:
            if pattern in query_clean:
                parts = query_clean.split(pattern)
                # Filter out empty parts
                query_parts = [p.strip() for p in parts if p.strip()]
                break
        
        if not query_parts:
            # No ellipsis, treat the whole query as a single part
            query_parts = [query_clean]
        
        # 4. Find best match based on prefix/suffix overlap
        best_match = None
        best_score = 0
        
        for doc in available_docs:
            doc_lower = doc.lower()
            score = 0
            
            # If we have multiple parts (separated by ellipsis), check if all parts exist in order
            if len(query_parts) > 1:
                # Check if all parts appear in the document name in order
                last_pos = 0
                all_found = True
                total_length = 0
                
                for part in query_parts:
                    part_lower = part.lower()
                    pos = doc_lower.find(part_lower, last_pos)
                    if pos == -1:
                        all_found = False
                        break
                    total_length += len(part)
                    last_pos = pos + len(part)
                
                if all_found:
                    # Score based on total matched length
                    score = total_length
                    # Bonus if first part is a prefix
                    if doc_lower.startswith(query_parts[0].lower()):
                        score += 10
                    # Bonus if last part is a suffix
                    if doc_lower.endswith(query_parts[-1].lower()):
                        score += 10
            else:
                # Single part - check for prefix or suffix match
                single_part = query_parts[0].lower()
                
                # Strong preference for prefix match
                if doc_lower.startswith(single_part):
                    score = len(single_part) + 20  # Bonus for prefix match
                # Check for suffix match
                elif doc_lower.endswith(single_part):
                    score = len(single_part) + 10  # Smaller bonus for suffix
                # Check if it's contained anywhere
                elif single_part in doc_lower:
                    score = len(single_part)
                
                # Also check for word-boundary matches (e.g., "Answer" matches start of "Answer, ...")
                words = single_part.split()
                doc_words = doc_lower.split()
                if words and doc_words:
                    # Check if query words match beginning of doc words
                    matching_words = 0
                    for i, word in enumerate(words):
                        if i < len(doc_words) and doc_words[i].startswith(word):
                            matching_words += 1
                        else:
                            break
                    if matching_words > 0:
                        word_score = matching_words * 5 + len(' '.join(words[:matching_words]))
                        score = max(score, word_score)
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = doc
        
        # 5. If still no match found with reasonable confidence, raise error with suggestions
        if best_match is None or best_score < 3:
            # Find documents that contain any significant word from the query
            suggestions = []
            significant_words = [w for w in query_clean.split() if len(w) > 3 and w.lower() not in ['the', 'and', 'for', 'with']]
            
            for doc in available_docs:
                doc_lower = doc.lower()
                for word in significant_words:
                    if word.lower() in doc_lower:
                        suggestions.append(doc)
                        break
            
            if suggestions:
                raise ValueError(f"Document not found: '{query_name}'. Did you mean one of: {', '.join(suggestions[:3])}?")
            else:
                raise ValueError(f"Document not found: '{query_name}'. Available documents: {', '.join(available_docs[:5])}")
        
        return best_match
    
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to search documents with regex.
        Supports:
        - Single document: doc_name="document.txt"
        - All documents: doc_name="all"
        - Multiple specific documents: doc_names=["doc1.txt", "doc2.txt"]
        
        Args:
            args: Dictionary containing pattern and document specification
            
        Returns:
            Dictionary with search results across all specified documents
        """
        try:
            # Handle common field name variations from LLM
            if 'document' in args and 'doc_name' not in args:
                args['doc_name'] = args.pop('document')
            if 'documents' in args and 'doc_names' not in args:
                args['doc_names'] = args.pop('documents')
            if 'regex' in args and 'pattern' not in args:
                args['pattern'] = args.pop('regex')
            
            # Remove any extra fields that the LLM might include
            if 'keys' in args:
                args.pop('keys')  # This is not a valid field for search
            
            # Validate input
            input_data = self.validate_input(args, SearchDocumentRegexInput)
            
            # Get all available documents
            all_doc_names = self.document_manager.list_documents()
            
            # Check if corpus is empty
            if not all_doc_names:
                raise ValueError("No documents available in corpus. Please ensure documents are loaded before searching.")
            
            # Determine which documents to search
            docs_to_search = []
            
            # Priority 1: Check if 'doc_names' is a non-empty array
            if input_data.doc_names and isinstance(input_data.doc_names, list) and len(input_data.doc_names) > 0:
                # Fuzzy match all requested documents
                docs_to_search = []
                fuzzy_matched = []
                for doc_name in input_data.doc_names:
                    try:
                        matched_doc = self._find_best_matching_document(doc_name, all_doc_names)
                        docs_to_search.append(matched_doc)
                        if matched_doc != doc_name:
                            fuzzy_matched.append(f"'{doc_name}' -> '{matched_doc}'")
                    except ValueError as e:
                        raise ValueError(f"Document not found in list: {e}")
                
                # Log fuzzy matches if any
                if fuzzy_matched:
                    print(f"[SEARCH_DOCUMENT_REGEX] Fuzzy matched: {'; '.join(fuzzy_matched)}")
            
            # Priority 2: Check 'doc_name' parameter
            elif input_data.doc_name:
                if input_data.doc_name == 'all':
                    # Search all documents
                    docs_to_search = all_doc_names
                else:
                    # Search specific single document with fuzzy matching
                    matched_doc = self._find_best_matching_document(input_data.doc_name, all_doc_names)
                    if matched_doc != input_data.doc_name:
                        print(f"[SEARCH_DOCUMENT_REGEX] Fuzzy matched '{input_data.doc_name}' to '{matched_doc}'")
                    docs_to_search = [matched_doc]
            else:
                # Default to searching all documents if nothing specified
                docs_to_search = all_doc_names
            
            # Collect ALL results first (including no-match documents)
            document_matches = {}  # doc_name -> list of (start, end) ranges
            document_results = []  # For output formatting
            total_matches = 0
            
            for doc_name in docs_to_search:
                # Perform the search on this document
                matches = self.document_manager.search_document(
                    doc_name,
                    input_data.pattern,
                    input_data.flags,
                    input_data.top_k,
                    input_data.context_tokens
                )
                
                # Convert to RegexMatch objects
                regex_matches = []
                for match in matches:
                    regex_match = RegexMatch(
                        start_token=match["start_token"],
                        end_token=match["end_token"],
                        start_char=match["start_char"],
                        end_char=match["end_char"],
                        snippet=match["snippet"],
                        groups=match["groups"],
                        pattern=match["pattern"],
                        flags=match["flags"]
                    )
                    regex_matches.append(regex_match)
                
                # Store match ranges for this document (even if empty)
                match_ranges = [(m.start_token, m.end_token) for m in regex_matches]
                document_matches[doc_name] = match_ranges
                
                # Add to results only if there are matches (for display)
                if regex_matches:
                    doc_result = DocumentSearchResult(
                        doc_name=doc_name,
                        matches=regex_matches,
                        match_count=len(regex_matches)
                    )
                    document_results.append(doc_result)
                    total_matches += len(regex_matches)
            
            # Create ONE SearchEvent for the entire operation
            if self.ledger:  # Always record, even with no matches
                search_event = SearchEvent(
                    doc_name=input_data.doc_name if input_data.doc_name else None,
                    doc_names=input_data.doc_names if input_data.doc_names else None,
                    pattern=input_data.pattern,
                    flags=input_data.flags,
                    matches_found=total_matches,
                    step=self._current_step,
                    document_matches=document_matches  # All searched docs
                )
                self.ledger.record_search(search_event, self._run_id)
            
            # Create output
            output = SearchDocumentRegexOutput(
                pattern=input_data.pattern,
                documents_searched=docs_to_search,
                results=document_results,
                total_matches=total_matches
            )
            
            result = self.format_output(output)
            
            # Add note about what was searched
            if input_data.doc_names and len(input_data.doc_names) > 0:
                result["note"] = f"Searched {len(docs_to_search)} specified documents"
            elif input_data.doc_name == 'all':
                result["note"] = f"Searched all {len(docs_to_search)} documents"
            elif input_data.doc_name:
                result["note"] = f"Searched single document: {input_data.doc_name}"
            
            return result
            
        except ValidationError as e:
            # Re-raise with clearer message
            raise ValueError(f"Invalid input for search_document_regex: {e}")
        except ValueError as e:
            # Pattern error or document not found
            raise ValueError(str(e))
        except Exception as e:
            raise Exception(f"Error searching document: {e}")