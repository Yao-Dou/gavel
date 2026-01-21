#!/usr/bin/env python3
"""
Main entry point for running the legal agent.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.driver import Driver, BatchDriver


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the legal agent to extract checklist items from case documents"
    )
    
    # Required arguments
    parser.add_argument(
        "corpus_path",
        help="Path to the document corpus (directory containing legal documents)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model to use for orchestration (default: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps before stopping (default: 100)"
    )
    
    parser.add_argument(
        "--store-path",
        default="checklist_store.json",
        help="Path for checklist persistence (default: checklist_store.json)"
    )
    
    parser.add_argument(
        "--ledger-path",
        default="ledger.jsonl",
        help="Path for ledger persistence (default: ledger.jsonl)"
    )
    
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing configuration files (default: config)"
    )
    
    parser.add_argument(
        "--checklist-config",
        default="config/checklist_configs/all/all_26_items.yaml",
        help="Path to specific checklist config file (default: config/checklist_configs/all/all_26_items.yaml)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing state instead of starting fresh"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode on multiple cases"
    )
    
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="Specific case IDs to process in batch mode"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for batch mode (default: output)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: show full prompts and LLM responses"
    )
    
    parser.add_argument(
        "--recent-actions",
        type=int,
        default=5,
        help="Number of recent actions to show with detailed results (default: 5)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbose = False
    elif args.debug:
        verbose = 2  # Debug mode with full prompts/responses
    else:
        verbose = True  # Normal mode (verbose=1)
    
    print(f"Legal Agent System")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus_path}")
    print(f"{'='*60}\n")
    
    if args.batch:
        # Run in batch mode
        driver = BatchDriver(
            corpus_base_path=args.corpus_path,
            output_base_path=args.output_dir,
            model_name=args.model,
            max_steps=args.max_steps,
            config_dir=args.config_dir,
            checklist_config_path=args.checklist_config,
            verbose=verbose,
            recent_actions=args.recent_actions
        )
        
        results = driver.run_batch(args.case_ids)
        
        # Print summary
        successful = sum(1 for r in results.values() if r["success"])
        print(f"\nBatch Summary: {successful}/{len(results)} cases successful")
        
    else:
        # Run single case
        driver = Driver(
            corpus_path=args.corpus_path,
            store_path=args.store_path,
            ledger_path=args.ledger_path,
            config_dir=args.config_dir,
            checklist_config_path=args.checklist_config,
            model_name=args.model,
            max_steps=args.max_steps,
            verbose=verbose,
            recent_actions=args.recent_actions
        )
        
        results = driver.run(resume=args.resume)
        
        # Print final summary
        stats = results["completion_stats"]
        print(f"\nFinal Summary:")
        total_keys = stats.get('total', stats['filled'] + stats['empty'])
        print(f"  Filled: {stats['filled']}/{total_keys}")
        print(f"  Empty: {stats['empty']}")
        print(f"  Total Steps: {results['total_steps']}")


if __name__ == "__main__":
    main()