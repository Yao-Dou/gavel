"""
List documents tool - returns information about all documents in the corpus.
"""

from typing import Dict, Any, List
from .base import BaseTool
from state.store import Ledger
from state.schemas import ListDocumentsOutput, DocumentInfo
from agent.document_manager import DocumentManager


class ListDocumentsTool(BaseTool):
    """
    Tool for listing all documents in the corpus with their metadata.
    Includes token counts, visit status, and coverage information.
    """
    
    def __init__(self, document_manager: DocumentManager, ledger: Ledger = None):
        """
        Initialize the list_documents tool.
        
        Args:
            document_manager: DocumentManager instance for accessing documents
            ledger: Optional Ledger instance for coverage information
        """
        super().__init__(
            name="list_documents",
            description="List all documents in the corpus with metadata"
        )
        self.document_manager = document_manager
        self.ledger = ledger
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema - this tool takes no inputs.
        
        Returns:
            Empty schema (no required parameters)
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema.
        
        Returns:
            Schema for ListDocumentsOutput
        """
        return {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [
                                "complaint", "docket", "motion", "order", 
                                "exhibit", "judgment", "other"
                            ]},
                            "token_count": {"type": "integer"},
                            "visited": {"type": "boolean"},
                            "coverage": {
                                "type": "object",
                                "properties": {
                                    "windows_read": {"type": "integer"},
                                    "approx_tokens_read": {"type": "integer"}
                                }
                            },
                            "last_read": {
                                "type": "object",
                                "properties": {
                                    "start_token": {"type": "integer"},
                                    "end_token": {"type": "integer"}
                                }
                            }
                        },
                        "required": ["name", "type", "token_count", "visited"]
                    }
                }
            },
            "required": ["documents"]
        }
    
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to list all documents.
        
        Args:
            args: Empty dictionary (no inputs required)
            
        Returns:
            Dictionary containing list of documents with metadata
        """
        # Get list of document names
        doc_names = self.document_manager.list_documents()
        
        # Build document info for each
        documents = []
        for doc_name in doc_names:
            doc_info = self.document_manager.get_document_info(doc_name, self.ledger)
            documents.append(doc_info)
        
        # Create output
        output = ListDocumentsOutput(documents=documents)
        
        return self.format_output(output)