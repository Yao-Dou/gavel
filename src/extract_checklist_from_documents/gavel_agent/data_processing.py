#!/usr/bin/env python3
"""
Data processing script for converting legal case documents to agent-compatible format.
Processes cases from the multi_lexsum format into individual case directories.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agent.tokenizer import TokenizerWrapper


class LegalDataProcessor:
    """
    Processes legal case data from multi_lexsum format to agent-compatible format.
    """
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = "data",
        model_name: str = "Qwen/Qwen3-8B",
        verbose: bool = True
    ):
        """
        Initialize the data processor.
        
        Args:
            input_file: Path to input JSON file containing cases
            output_dir: Base output directory for processed data
            model_name: Model name for tokenizer (default: Qwen/Qwen3-8B)
            verbose: Whether to print progress
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Extract dataset name from input file
        self.dataset_name = self.input_file.stem
        
        # Initialize tokenizer
        self.tokenizer = TokenizerWrapper(model_name)
        if self.verbose:
            print(f"Initialized tokenizer for {model_name}")
            print(f"Tokenizer backend: {self.tokenizer._backend}")
        
        # Statistics
        self.stats = {
            "total_cases": 0,
            "processed_cases": 0,
            "total_documents": 0,
            "total_tokens": 0,
            "errors": []
        }
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the input JSON data.
        
        Returns:
            List of case dictionaries
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Expected list of cases in input file")
        
        self.stats["total_cases"] = len(data)
        
        if self.verbose:
            print(f"Loaded {len(data)} cases from {self.input_file}")
        
        return data
    
    def sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use as filename.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length of filename
            
        Returns:
            Sanitized filename
        """
        # Remove special characters and replace spaces
        sanitized = re.sub(r'[^\w\s-]', '', text)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        
        # Convert to lowercase and truncate
        sanitized = sanitized.lower()[:max_length]
        
        # Remove trailing underscores
        sanitized = sanitized.rstrip('_')
        
        return sanitized or "document"
    
    def create_document_filename(self, doc_type: str, index: int, title: str = "") -> str:
        """
        Create a filename for a document.
        
        Args:
            doc_type: Document type
            index: Document index
            title: Optional document title
            
        Returns:
            Filename for the document
        """
        # Sanitize document type
        doc_type_clean = self.sanitize_filename(doc_type, 20)
        
        # Create base filename
        filename = f"{doc_type_clean}_{index:03d}"
        
        # Add sanitized title if available and short
        if title:
            title_clean = self.sanitize_filename(title, 20)
            if title_clean and title_clean != doc_type_clean:
                filename = f"{filename}_{title_clean}"
        
        return f"{filename}.txt"
    
    def process_case(self, case: Dict[str, Any], output_base: Path) -> bool:
        """
        Process a single case.
        
        Args:
            case: Case dictionary
            output_base: Base output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            case_id = str(case.get("case_id", "unknown"))
            
            # Create case directory
            case_dir = output_base / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Get document data
            texts = case.get("case_documents_text", [])
            titles = case.get("case_documents_title", [])
            doc_types = case.get("case_documents_doc_type", [])
            dates = case.get("case_documents_date", [])
            doc_ids = case.get("case_documents_id", [])
            
            # Validate data consistency
            num_docs = len(texts)
            if not all(len(lst) == num_docs for lst in [titles, doc_types]):
                raise ValueError(f"Inconsistent document list lengths for case {case_id}")
            
            # Ensure all lists have same length (pad if necessary)
            if len(dates) < num_docs:
                dates.extend([None] * (num_docs - len(dates)))
            if len(doc_ids) < num_docs:
                doc_ids.extend([f"doc_{i:03d}" for i in range(len(doc_ids), num_docs)])
            
            # Track title occurrences to handle duplicates
            title_counts = {}
            unique_names = []
            
            # First pass: count occurrences of each title
            for title in titles:
                if title not in title_counts:
                    title_counts[title] = 0
                title_counts[title] += 1
            
            # Second pass: create unique names
            title_counters = {}
            for title in titles:
                if title_counts[title] > 1:
                    # This title appears multiple times, add counter
                    if title not in title_counters:
                        title_counters[title] = 1
                    else:
                        title_counters[title] += 1
                    unique_name = f"{title} ({title_counters[title]:03d})"
                else:
                    # Title is unique, use as-is
                    unique_name = title
                if not title:
                    unique_name = "Untitled Document"
                unique_names.append(unique_name)
            
            # Process each document
            documents_metadata = []
            total_tokens = 0
            
            for i, text in enumerate(texts):
                # Get document info
                title = titles[i] if i < len(titles) else f"Document {i+1}"
                unique_name = unique_names[i]  # Use the unique name we created

                doc_type = doc_types[i] if i < len(doc_types) else "Unknown"
                doc_type = doc_type or "Unknown" # fix for None values

                date = dates[i] if i < len(dates) else None
                doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{i:03d}"
                
                # Create filename
                if title:
                    filename = self.create_document_filename(doc_type, i+1, title)
                else:
                    filename = self.create_document_filename(doc_type, i+1, "Untitled Document")
                
                # Save document text
                doc_path = case_dir / filename
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Calculate token count with our tokenizer
                token_count = self.tokenizer.count_tokens(text)
                total_tokens += token_count
                
                # Add to metadata
                doc_metadata = {
                    "filename": filename,
                    "name": unique_name,  # Use unique name for DocumentManager compatibility
                    "doc_type": doc_type,
                    "token_count": token_count,
                    "doc_id": doc_id
                }
                
                if date:
                    doc_metadata["date"] = date
                
                documents_metadata.append(doc_metadata)
                
                self.stats["total_documents"] += 1
            
            # Create case metadata
            case_metadata = {
                "case_id": case_id,
                "document_count": num_docs,
                "total_tokens": total_tokens,
                "documents": documents_metadata
            }
            
            # Add optional fields if present
            if "filing_date" in case:
                case_metadata["filing_date"] = case["filing_date"]
            if "case_url" in case:
                case_metadata["case_url"] = case["case_url"]
            if "case_type" in case:
                case_metadata["case_type"] = case["case_type"]
            
            # Save metadata
            metadata_path = case_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(case_metadata, f, indent=2)
            
            self.stats["total_tokens"] += total_tokens
            self.stats["processed_cases"] += 1
            
            if self.verbose:
                print(f"✓ Processed case {case_id}: {num_docs} documents, {total_tokens:,} tokens")
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing case {case.get('case_id', 'unknown')}: {str(e)}"
            self.stats["errors"].append(error_msg)
            if self.verbose:
                print(f"✗ {error_msg}")
            return False
    
    def process_all(self, case_ids: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all cases or specific case IDs.
        
        Args:
            case_ids: Optional list of case IDs to process (None for all)
            dry_run: If True, don't actually write files
            
        Returns:
            Processing statistics
        """
        # Load data
        cases = self.load_data()
        
        # Filter cases if specific IDs provided
        if case_ids:
            cases = [c for c in cases if str(c.get("case_id")) in case_ids]
            if self.verbose:
                print(f"Processing {len(cases)} specific cases")
        
        # Create output directory
        output_base = self.output_dir / self.dataset_name
        
        if not dry_run:
            output_base.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Output directory: {output_base}")
            if dry_run:
                print("DRY RUN - No files will be written")
            print("-" * 60)
        
        # Process each case
        for i, case in enumerate(cases, 1):
            if self.verbose and len(cases) > 1:
                print(f"\n[{i}/{len(cases)}] Processing case {case.get('case_id')}...")
            
            if not dry_run:
                self.process_case(case, output_base)
            else:
                # Dry run - just count tokens
                texts = case.get("case_documents_text", [])
                total_tokens = sum(self.tokenizer.count_tokens(text) for text in texts)
                self.stats["total_tokens"] += total_tokens
                self.stats["total_documents"] += len(texts)
                self.stats["processed_cases"] += 1
                
                if self.verbose:
                    print(f"  Would process: {len(texts)} documents, {total_tokens:,} tokens")
        
        # Print summary
        if self.verbose:
            self.print_summary()
        
        return self.stats
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"Total cases: {self.stats['total_cases']}")
        print(f"Processed cases: {self.stats['processed_cases']}")
        print(f"Total documents: {self.stats['total_documents']}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        
        if self.stats['processed_cases'] > 0:
            avg_docs = self.stats['total_documents'] / self.stats['processed_cases']
            avg_tokens = self.stats['total_tokens'] / self.stats['processed_cases']
            print(f"Average documents per case: {avg_docs:.1f}")
            print(f"Average tokens per case: {avg_tokens:,.0f}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")
    
    def validate_output(self, output_base: Path) -> bool:
        """
        Validate the processed output.
        
        Args:
            output_base: Base output directory
            
        Returns:
            True if valid, False otherwise
        """
        if not output_base.exists():
            print(f"Output directory does not exist: {output_base}")
            return False
        
        # Check each case directory
        case_dirs = [d for d in output_base.iterdir() if d.is_dir()]
        
        for case_dir in case_dirs:
            # Check for metadata
            metadata_file = case_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"Missing metadata.json in {case_dir}")
                return False
            
            # Load and validate metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check required fields
                required = ["case_id", "documents", "document_count", "total_tokens"]
                for field in required:
                    if field not in metadata:
                        print(f"Missing field '{field}' in {metadata_file}")
                        return False
                
                # Check document files exist
                for doc in metadata["documents"]:
                    doc_file = case_dir / doc["filename"]
                    if not doc_file.exists():
                        print(f"Missing document file: {doc_file}")
                        return False
                        
            except Exception as e:
                print(f"Error validating {metadata_file}: {e}")
                return False
        
        print(f"✓ Validation successful: {len(case_dirs)} cases validated")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process legal case documents for agent scaffold"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input JSON file containing cases"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory (default: data)"
    )
    
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model for tokenizer (default: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="Specific case IDs to process"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without writing files"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output after processing"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = LegalDataProcessor(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model,
        verbose=not args.quiet
    )
    
    # Process cases
    stats = processor.process_all(
        case_ids=args.case_ids,
        dry_run=args.dry_run
    )
    
    # Validate if requested
    if args.validate and not args.dry_run:
        output_base = Path(args.output_dir) / processor.dataset_name
        processor.validate_output(output_base)
    
    # Return exit code based on errors
    sys.exit(0 if not stats["errors"] else 1)


if __name__ == "__main__":
    main()