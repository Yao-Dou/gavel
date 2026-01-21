"""
Read document tool - reads a specific token range from a document.
"""

from typing import Dict, Any
from pydantic import ValidationError
from .base import BaseTool
from state.store import Ledger
from state.schemas import ReadDocumentInput, ReadDocumentOutput, ReadEvent
from agent.document_manager import DocumentManager


class ReadDocumentTool(BaseTool):
    """
    Tool for reading specific token ranges from documents.
    Records read events in the ledger for coverage tracking.
    """
    
    def __init__(self, document_manager: DocumentManager, ledger: Ledger = None):
        """
        Initialize the read_document tool.
        
        Args:
            document_manager: DocumentManager instance for accessing documents
            ledger: Optional Ledger instance for recording reads
        """
        super().__init__(
            name="read_document",
            description="Read a specific token range from a document"
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
        Get the input schema for read_document.
        
        Returns:
            Schema for ReadDocumentInput
        """
        return {
            "type": "object",
            "properties": {
                "doc_name": {"type": "string"},
                "start_token": {"type": "integer", "minimum": 0},
                "end_token": {"type": "integer", "minimum": 1},
                "purpose": {
                    "type": "string",
                    "enum": ["scan", "confirm"],
                    "default": "scan"
                }
            },
            "required": ["doc_name", "start_token", "end_token"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for ReadDocumentOutput
        """
        return {
            "type": "object",
            "properties": {
                "doc_name": {"type": "string"},
                "start_token": {"type": "integer"},
                "end_token": {"type": "integer"},
                "text": {"type": "string"}
            },
            "required": ["doc_name", "start_token", "end_token", "text"]
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
        Execute the tool to read a document range.
        
        Args:
            args: Dictionary containing doc_name, start_token, end_token
            
        Returns:
            Dictionary with the read text and metadata
        """
        try:
            # Handle common field name variations from LLM
            if 'document' in args and 'doc_name' not in args:
                args['doc_name'] = args.pop('document')
            
            # Validate input
            input_data = self.validate_input(args, ReadDocumentInput)
            
            # Get available documents
            doc_names = self.document_manager.list_documents()
            
            # Find the best matching document name (handles fuzzy matching)
            actual_doc_name = self._find_best_matching_document(input_data.doc_name, doc_names)
            
            # Log if we had to do fuzzy matching
            if actual_doc_name != input_data.doc_name:
                print(f"[READ_DOCUMENT] Fuzzy matched '{input_data.doc_name}' to '{actual_doc_name}'")
            
            # Read the token range using the actual document name
            text, actual_start, actual_end = self.document_manager.read_token_range(
                actual_doc_name,  # Use the fuzzy-matched name
                input_data.start_token,
                input_data.end_token
            )
            
            # Record in ledger if available
            if self.ledger:
                read_event = ReadEvent(
                    doc_name=actual_doc_name,  # Record the actual document read
                    start_token=actual_start,
                    end_token=actual_end,
                    tokens_read=actual_end - actual_start,
                    step=self._current_step
                )
                self.ledger.record_read(read_event, self._run_id)
            
            # Create output with actual document name
            output = ReadDocumentOutput(
                doc_name=actual_doc_name,  # Return the actual document name
                start_token=actual_start,
                end_token=actual_end,
                text=text
            )
            
            return self.format_output(output)
            
        except ValidationError as e:
            # Re-raise with clearer message
            raise ValueError(f"Invalid input for read_document: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Document not found: {e}")
        except Exception as e:
            raise Exception(f"Error reading document: {e}")