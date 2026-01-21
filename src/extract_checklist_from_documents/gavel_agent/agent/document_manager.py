"""
Document manager for loading and managing legal case documents.
Handles document loading, caching, and token/character position mapping.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from state.schemas import DocumentInfo, DocumentCoverage, DocumentReadInfo
from .tokenizer import TokenizerWrapper


class DocumentManager:
    """
    Manages the corpus of legal documents.
    Provides efficient access to documents and their metadata.
    """
    
    def __init__(self, corpus_path: str, tokenizer: Optional[TokenizerWrapper] = None, 
                 tokenizer_model: Optional[str] = None):
        """
        Initialize the document manager.
        
        Args:
            corpus_path: Path to the directory containing documents
            tokenizer: Tokenizer wrapper instance (creates default if None)
            tokenizer_model: Model name for tokenizer if creating new one
                           (e.g., "gpt-3.5-turbo", "Qwen/Qwen3-8B")
        """
        self.corpus_path = Path(corpus_path)
        if tokenizer:
            self.tokenizer = tokenizer
        elif tokenizer_model:
            self.tokenizer = TokenizerWrapper(tokenizer_model)
        else:
            self.tokenizer = TokenizerWrapper()
        self._document_cache: Dict[str, str] = {}  # Keyed by display name
        self._token_counts: Dict[str, int] = {}  # Keyed by display name
        self._document_types: Dict[str, str] = {}  # Keyed by display name
        self._metadata_by_filename: Dict[str, Dict] = {}
        self._filename_by_name: Dict[str, str] = {}
        self._name_by_filename: Dict[str, str] = {}  # Reverse mapping
        
        # Ensure corpus directory exists
        if not self.corpus_path.exists():
            self.corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Load document metadata if available
        self._load_metadata()
    
    def _load_metadata(self):
        """Load document metadata from a metadata file if it exists."""
        metadata_file = self.corpus_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Create mappings: filename -> metadata and name -> filename
                    self._metadata_by_filename = {}
                    self._filename_by_name = {}
                    self._name_by_filename = {}
                    
                    for doc in metadata.get("documents", []):
                        filename = doc.get("filename")
                        name = doc.get("name", filename)
                        if filename:
                            self._metadata_by_filename[filename] = doc
                            self._filename_by_name[name] = filename
                            self._name_by_filename[filename] = name
                            # Store document type using display name as key
                            doc_type_str = doc.get("doc_type", "Other")
                            self._document_types[name] = doc_type_str
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
    
    def list_documents(self) -> List[str]:
        """
        List all documents in the corpus.
        
        Returns:
            List of document display names (human-readable)
        """
        documents = []
        
        # Always use display names from metadata
        if self._filename_by_name:
            # Return display names from metadata
            documents = list(self._filename_by_name.keys())
        else:
            # This should not happen - metadata.json should always exist
            raise ValueError("No metadata.json found in corpus directory. Cannot list documents.")
        
        return sorted(documents)
    
    def load_document(self, doc_name: str, cache: bool = True) -> str:
        """
        Load a document's content.
        
        Args:
            doc_name: Name of the document (display name preferred)
            cache: Whether to cache the document in memory
            
        Returns:
            Document content as string
            
        Raises:
            FileNotFoundError: If document doesn't exist
        """
        # Get the filename for file system operations
        filename = self._filename_by_name.get(doc_name, doc_name)
        
        # Get the display name for cache keys (prefer metadata name over filename)
        display_name = self._name_by_filename.get(filename, doc_name)
        
        # Check cache first using display name
        if display_name in self._document_cache:
            return self._document_cache[display_name]
        
        doc_path = self.corpus_path / filename
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_name}")
        
        # Read document content
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            with open(doc_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Cache if requested using display name as key
        if cache:
            self._document_cache[display_name] = content
            # Also cache token count using display name
            self._token_counts[display_name] = self.tokenizer.count_tokens(content)
        
        return content
    
    def get_document_type(self, doc_name: str) -> str:
        """
        Get the type of a document.
        
        Args:
            doc_name: Name of the document (display name preferred)
            
        Returns:
            Document type as string
        """
        # Try direct lookup first (if doc_name is already a display name)
        if doc_name in self._document_types:
            return self._document_types[doc_name]
        
        # Resolve to filename then back to display name
        filename = self._filename_by_name.get(doc_name, doc_name)
        display_name = self._name_by_filename.get(filename, doc_name)
        
        if display_name in self._document_types:
            return self._document_types[display_name]
        
        # Get from metadata if available
        metadata = self._metadata_by_filename.get(filename, {})
        if "doc_type" in metadata:
            return metadata["doc_type"]
        
        # If no metadata, return Unknown
        return "Unknown"
    
    def get_token_count(self, doc_name: str) -> int:
        """
        Get the token count for a document.
        
        Args:
            doc_name: Name of the document (display name preferred)
            
        Returns:
            Number of tokens in the document
        """
        # Get the filename for resolution
        filename = self._filename_by_name.get(doc_name, doc_name)
        # Get the display name for cache key
        display_name = self._name_by_filename.get(filename, doc_name)
        
        if display_name not in self._token_counts:
            content = self.load_document(display_name, cache=True)
            self._token_counts[display_name] = self.tokenizer.count_tokens(content)
        
        return self._token_counts[display_name]
    
    def get_document_info(self, doc_name: str, ledger=None) -> DocumentInfo:
        """
        Get complete information about a document.
        
        Args:
            doc_name: Name of the document (can be filename or display name)
            ledger: Optional Ledger instance for coverage info
            
        Returns:
            DocumentInfo object with all metadata
        """
        # Resolve name to filename if it's a display name
        filename = self._filename_by_name.get(doc_name, doc_name)
        
        # Get metadata if available
        metadata = self._metadata_by_filename.get(filename, {})
        display_name = metadata.get("name", doc_name)
        
        # Get basic info
        doc_type = self.get_document_type(filename)
        token_count = self.get_token_count(filename)
        
        # Get coverage info from ledger if available
        visited = False
        coverage = DocumentCoverage()
        last_read = None
        
        if ledger:
            # Use display_name for ledger lookups since tools record using doc_name (display name)
            visited = display_name in ledger.get_visited_documents()
            coverage = ledger.get_document_coverage(display_name) or DocumentCoverage()
            last_read = ledger.get_last_read(display_name)
        
        return DocumentInfo(
            name=display_name,  # Use human-readable name for display
            type=doc_type,
            token_count=token_count,
            visited=visited,
            coverage=coverage,
            last_read=last_read
        )
    
    def read_token_range(self, doc_name: str, start_token: int, end_token: int) -> Tuple[str, int, int]:
        """
        Read a specific token range from a document.
        
        Args:
            doc_name: Name of the document (display name preferred)
            start_token: Starting token index (inclusive)
            end_token: Ending token index (exclusive)
            
        Returns:
            Tuple of (text, actual_start_token, actual_end_token)
        """
        # Use doc_name directly - load_document will handle resolution
        content = self.load_document(doc_name)
        return self.tokenizer.get_text_for_token_range(content, start_token, end_token)
    
    def search_document(self, doc_name: str, pattern: str, flags: List[str] = None, 
                       top_k: int = 5, context_tokens: int = 200) -> List[Dict]:
        """
        Search a document using regex pattern.
        
        Args:
            doc_name: Name of the document (display name preferred)
            pattern: Regex pattern to search for
            flags: List of regex flags (IGNORECASE, MULTILINE, DOTALL)
            top_k: Maximum number of matches to return
            context_tokens: Number of context tokens around each match
            
        Returns:
            List of match dictionaries with position and context
        """
        import re
        
        # Use doc_name directly - load_document will handle resolution
        content = self.load_document(doc_name)
        
        # Build regex flags
        regex_flags = 0
        if flags:
            for flag in flags:
                if flag == "IGNORECASE":
                    regex_flags |= re.IGNORECASE
                elif flag == "MULTILINE":
                    regex_flags |= re.MULTILINE
                elif flag == "DOTALL":
                    regex_flags |= re.DOTALL
        
        # Find matches
        matches = []
        try:
            regex = re.compile(pattern, regex_flags)
            for match in regex.finditer(content):
                matches.append({
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "text": match.group(0),
                    "groups": match.groupdict()
                })
                
                if len(matches) >= top_k:
                    break
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        # Convert character positions to token positions and add context
        result_matches = []
        tokens = self.tokenizer.encode(content)
        
        for match in matches:
            # Approximate token position from character position
            # This is a simple approximation - could be improved
            char_ratio = match["start_char"] / len(content)
            approx_token = int(char_ratio * len(tokens))
            
            # Get context window around the match
            context_start = max(0, approx_token - context_tokens // 2)
            context_end = min(len(tokens), approx_token + context_tokens // 2)
            context_text, _, _ = self.tokenizer.get_text_for_token_range(
                content, context_start, context_end
            )
            
            result_matches.append({
                "start_token": context_start,  # Context window start - where to read from
                "end_token": context_end,      # Context window end - where to read to
                "start_char": match["start_char"],
                "end_char": match["end_char"],
                "snippet": context_text,
                "groups": match["groups"],
                "pattern": pattern,
                "flags": flags or []
            })
        
        return result_matches
    
    def clear_cache(self):
        """Clear the document cache to free memory."""
        self._document_cache.clear()
        self._token_counts.clear()