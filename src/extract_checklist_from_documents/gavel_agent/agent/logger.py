"""
JSON logging system for the legal agent.
Provides structured logging for actions, decisions, and performance metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz
from threading import Lock


class ActionLogger:
    """
    Structured JSON logger for agent actions and decisions.
    Logs to JSONL format for streaming and analysis.
    """
    
    def __init__(self, log_dir: str = "logs", run_id: Optional[str] = None):
        """
        Initialize the action logger.
        
        Args:
            log_dir: Directory for log files
            run_id: Unique run identifier
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run_id if not provided
        if run_id is None:
            nyc_tz = pytz.timezone('America/New_York')
            timestamp = datetime.now(nyc_tz).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"
        self.run_id = run_id
        
        # Log files
        self.action_log_file = self.log_dir / f"{run_id}_actions.jsonl"
        self.summary_log_file = self.log_dir / f"{run_id}_summary.json"
        
        # Thread safety
        self.lock = Lock()
        
        # Performance tracking
        self.start_time = time.time()
        self.action_count = 0
        self.tool_latencies: Dict[str, List[float]] = {}
        self.token_usage: List[int] = []
        
    def log_action(
        self,
        step: int,
        tool: str,
        args: Dict[str, Any],
        latency_ms: float,
        changed_keys: List[str] = None,
        error: Optional[str] = None
    ):
        """
        Log an action taken by the agent.
        
        Args:
            step: Current step number
            tool: Tool name
            args: Tool arguments
            latency_ms: Execution time in milliseconds
            changed_keys: Keys modified by this action
            error: Error message if action failed
        """
        log_entry = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "tool": tool,
            "args": args,
            "latency_ms": latency_ms,
            "changed_keys": changed_keys or [],
            "success": error is None,
            "error": error
        }
        
        with self.lock:
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
            
            # Track latency
            if tool not in self.tool_latencies:
                self.tool_latencies[tool] = []
            self.tool_latencies[tool].append(latency_ms)
            self.action_count += 1
    
    def log_tool_result(
        self,
        step: int,
        tool: str,
        result_size: int,
        success: bool,
        result_summary: Optional[str] = None
    ):
        """
        Log the result of a tool execution.
        
        Args:
            step: Current step number
            tool: Tool name
            result_size: Size of result in bytes/tokens
            success: Whether tool succeeded
            result_summary: Brief summary of result
        """
        log_entry = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "type": "tool_result",
            "tool": tool,
            "result_size": result_size,
            "success": success,
            "summary": result_summary
        }
        
        with self.lock:
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
    
    def log_decision(
        self,
        step: int,
        decision_type: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a decision made by the agent.
        
        Args:
            step: Current step number
            decision_type: Type of decision (stop, continue, etc.)
            reason: Reason for the decision
            metadata: Additional decision metadata
        """
        log_entry = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "type": "decision",
            "decision_type": decision_type,
            "reason": reason,
            "metadata": metadata or {}
        }
        
        with self.lock:
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
    
    def log_snapshot(
        self,
        step: int,
        snapshot: Dict[str, Any],
        snapshot_size: int
    ):
        """
        Log snapshot information.
        
        Args:
            step: Current step number
            snapshot: Snapshot data (will log summary only)
            snapshot_size: Size of snapshot in tokens
        """
        # Extract summary info
        summary = {
            "documents_count": len(snapshot.get("documents", [])),
            "unresolved_keys_count": len(snapshot.get("unresolved_keys", [])),
            "action_tail_length": len(snapshot.get("action_tail", []))
        }
        
        log_entry = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "type": "snapshot",
            "snapshot_size": snapshot_size,
            "summary": summary
        }
        
        with self.lock:
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
            self.token_usage.append(snapshot_size)
    
    def log_validation(
        self,
        step: int,
        validation_type: str,
        is_valid: bool,
        details: Dict[str, Any]
    ):
        """
        Log validation results.
        
        Args:
            step: Current step number
            validation_type: Type of validation performed
            is_valid: Whether validation passed
            details: Validation details
        """
        log_entry = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "type": "validation",
            "validation_type": validation_type,
            "is_valid": is_valid,
            "details": details
        }
        
        with self.lock:
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
    
    def log_run_summary(
        self,
        total_steps: int,
        filled_keys: int,
        total_keys: int,
        stop_reason: str,
        final_status: str = "completed"
    ):
        """
        Log final run summary.
        
        Args:
            total_steps: Total number of steps taken
            filled_keys: Number of keys filled
            total_keys: Total number of keys
            stop_reason: Reason for stopping
            final_status: Final status of run
        """
        duration_seconds = time.time() - self.start_time
        
        # Calculate statistics
        avg_latencies = {}
        for tool, latencies in self.tool_latencies.items():
            if latencies:
                avg_latencies[tool] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "count": len(latencies)
                }
        
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "duration_seconds": duration_seconds,
            "total_steps": total_steps,
            "total_actions": self.action_count,
            "filled_keys": filled_keys,
            "total_keys": total_keys,
            "completion_rate": filled_keys / total_keys if total_keys > 0 else 0,
            "stop_reason": stop_reason,
            "final_status": final_status,
            "performance": {
                "tool_latencies": avg_latencies,
                "total_tokens": sum(self.token_usage) if self.token_usage else 0,
                "avg_tokens_per_step": sum(self.token_usage) / len(self.token_usage) if self.token_usage else 0
            }
        }
        
        with self.lock:
            with open(self.summary_log_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Also append to action log for completeness
            with open(self.action_log_file, 'a') as f:
                f.write(json.dumps({
                    "type": "run_complete",
                    **summary
                }, default=str) + '\n')
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get current run summary.
        
        Returns:
            Dictionary with current statistics
        """
        with self.lock:
            duration = time.time() - self.start_time
            return {
                "run_id": self.run_id,
                "duration_seconds": duration,
                "action_count": self.action_count,
                "tool_calls": list(self.tool_latencies.keys()),
                "total_tokens": sum(self.token_usage) if self.token_usage else 0
            }


class PerformanceTracker:
    """
    Track performance metrics for the agent.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.timers: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a named timer and return elapsed milliseconds.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in milliseconds
        """
        if name not in self.timers:
            return 0.0
        
        elapsed = (time.time() - self.timers[name]) * 1000
        del self.timers[name]
        return elapsed
    
    def record_metric(self, name: str, value: Any):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()