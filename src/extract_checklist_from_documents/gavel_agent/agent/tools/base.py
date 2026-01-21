"""
Base tool interface for the legal agent system.
All tools must inherit from BaseTool and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from pydantic import BaseModel, ValidationError


class BaseTool(ABC):
    """
    Abstract base class for all tools in the legal agent system.
    
    Each tool must:
    1. Have a unique name
    2. Provide a JSON schema description of its input/output
    3. Implement the call method to execute the tool
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool's input parameters.
        
        Returns:
            JSON schema dictionary describing the input format
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool's output.
        
        Returns:
            JSON schema dictionary describing the output format
        """
        pass
    
    def describe(self) -> Dict[str, Any]:
        """
        Get a complete description of the tool including schemas.
        
        Returns:
            Dictionary with tool name, description, and input/output schemas
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema()
        }
    
    @abstractmethod
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.
        
        Args:
            args: Dictionary of input arguments matching the input schema
            
        Returns:
            Dictionary of outputs matching the output schema
            
        Raises:
            ValidationError: If input/output doesn't match schemas
            Exception: For tool-specific errors
        """
        pass
    
    def validate_input(self, args: Dict[str, Any], input_model: Optional[BaseModel] = None) -> Any:
        """
        Validate input arguments against a Pydantic model.
        
        Args:
            args: Input arguments to validate
            input_model: Pydantic model class for validation
            
        Returns:
            Validated model instance
            
        Raises:
            ValidationError: If validation fails
        """
        if input_model:
            try:
                return input_model(**args)
            except ValidationError as e:
                # Re-raise with more context
                raise ValueError(f"Invalid input for {self.name}: {e}")
        return args
    
    def format_output(self, output: Any) -> Dict[str, Any]:
        """
        Format output to match the expected schema.
        
        Args:
            output: Tool output (may be a Pydantic model)
            
        Returns:
            Dictionary representation of the output
        """
        if isinstance(output, BaseModel):
            # Use model_dump for Pydantic v2, fall back to dict() for v1
            if hasattr(output, 'model_dump'):
                return output.model_dump()
            else:
                return output.dict()
        return output