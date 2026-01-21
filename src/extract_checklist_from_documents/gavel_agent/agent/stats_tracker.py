"""
Token usage statistics tracker for the legal agent system.
Tracks and persists LLM token usage across steps with resume support.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock


class StatsTracker:
    """
    Tracks token usage statistics for LLM calls.
    Supports persistence and resume functionality.
    """
    
    def __init__(self, output_dir: str, case_id: Optional[str] = None):
        """
        Initialize the stats tracker.
        
        Args:
            output_dir: Directory to save stats file
            case_id: Optional case ID for organizing outputs
        """
        self.output_dir = Path(output_dir)
        if case_id:
            self.output_dir = self.output_dir / case_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_file = self.output_dir / "stats.json"
        self.lock = Lock()
        
        # Initialize stats structure
        self.stats = {
            "model": None,
            "steps": 0,
            "total_system_prompt_tokens": 0,
            "total_user_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "per_step_details": []
        }
        
        # Track current system prompt tokens (cached between steps)
        self.current_system_tokens = 0
        
    def load_existing_stats(self) -> bool:
        """
        Load existing stats from file if available.
        
        Returns:
            True if stats were loaded, False otherwise
        """
        if self.stats_file.exists():
            try:
                with self.lock:
                    with open(self.stats_file, 'r') as f:
                        self.stats = json.load(f)
                print(f"[StatsTracker] Loaded existing stats from {self.stats_file}")
                print(f"  Resuming from step {self.stats['steps']} with {self.stats['total_completion_tokens']} total tokens used")
                return True
            except Exception as e:
                print(f"[StatsTracker] Warning: Could not load stats from {self.stats_file}: {e}")
                return False
        return False
    
    def update_stats(
        self,
        step: int,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        is_system_cached: bool = False,
        system_tokens: Optional[int] = None
    ):
        """
        Update statistics with new LLM call data.
        
        Args:
            step: Current step number
            prompt_tokens: Total prompt tokens (system + user)
            completion_tokens: Completion/response tokens
            model: Model name/identifier
            is_system_cached: Whether system prompt was cached (reused from previous call)
            system_tokens: Explicit system prompt tokens (if known)
        """
        with self.lock:
            # Update model name if not set
            if not self.stats["model"]:
                self.stats["model"] = model
            
            # Calculate system vs user tokens
            if system_tokens is not None:
                # Explicit system tokens provided - use actual count
                self.current_system_tokens = system_tokens
                user_tokens = prompt_tokens - system_tokens
            elif is_system_cached and self.current_system_tokens > 0:
                # System is cached, use stored value
                user_tokens = prompt_tokens - self.current_system_tokens
            else:
                # Fallback: estimate if no actual count available
                # This should rarely happen with the new implementation
                if step == 1:
                    # First step, estimate system tokens as fallback
                    estimated_system = int(prompt_tokens * 0.35)
                    self.current_system_tokens = estimated_system
                    user_tokens = prompt_tokens - estimated_system
                    print(f"[StatsTracker] Warning: Using estimated system tokens ({estimated_system})")
                else:
                    # Use stored system tokens
                    user_tokens = prompt_tokens - self.current_system_tokens
            
            # Update cumulative totals
            # Always count system tokens - vLLM sends them with every request
            self.stats["total_system_prompt_tokens"] += self.current_system_tokens
            
            self.stats["total_user_prompt_tokens"] += user_tokens
            self.stats["total_completion_tokens"] += completion_tokens
            self.stats["steps"] = step
            
            # Add per-step details
            step_detail = {
                "step": step,
                "system_tokens": self.current_system_tokens,  # Always show actual system tokens used
                "user_tokens": user_tokens,
                "completion_tokens": completion_tokens,
                "total_prompt_tokens": prompt_tokens,
                "is_system_cached": is_system_cached
            }
            
            # Check if we're updating an existing step (in case of retries)
            existing_step = next((s for s in self.stats["per_step_details"] if s["step"] == step), None)
            if existing_step:
                # Update existing step
                idx = self.stats["per_step_details"].index(existing_step)
                self.stats["per_step_details"][idx] = step_detail
            else:
                # Add new step
                self.stats["per_step_details"].append(step_detail)
            
            # Save immediately for resilience
            self.save()
    
    def save(self):
        """
        Save current stats to file.
        Should be called within lock context.
        """
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"[StatsTracker] Warning: Could not save stats to {self.stats_file}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current statistics.
        
        Returns:
            Dictionary with stats summary
        """
        with self.lock:
            total_tokens = (
                self.stats["total_system_prompt_tokens"] +
                self.stats["total_user_prompt_tokens"] +
                self.stats["total_completion_tokens"]
            )
            
            return {
                "model": self.stats["model"],
                "total_steps": self.stats["steps"],
                "total_tokens": total_tokens,
                "total_system_prompt_tokens": self.stats["total_system_prompt_tokens"],
                "total_user_prompt_tokens": self.stats["total_user_prompt_tokens"],
                "total_completion_tokens": self.stats["total_completion_tokens"],
                "average_tokens_per_step": total_tokens / self.stats["steps"] if self.stats["steps"] > 0 else 0,
                "stats_file": str(self.stats_file)
            }
    
    def print_summary(self):
        """Print a formatted summary of statistics."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print("Token Usage Statistics")
        print(f"{'='*60}")
        print(f"Model: {summary['model']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - System Prompt: {summary['total_system_prompt_tokens']:,}")
        print(f"  - User Prompts: {summary['total_user_prompt_tokens']:,}")
        print(f"  - Completions: {summary['total_completion_tokens']:,}")
        print(f"Average per Step: {summary['average_tokens_per_step']:.0f}")
        print(f"Stats saved to: {summary['stats_file']}")
        print(f"{'='*60}")