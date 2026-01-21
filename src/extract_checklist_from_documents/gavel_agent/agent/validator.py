"""
Stop validation logic for the legal agent.
Ensures the agent only stops when objectives are met.
"""

from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta

from state.store import ChecklistStore, Ledger


class StopValidator:
    """
    Validates stop decisions to ensure objectives are met.
    """
    
    def __init__(
        self,
        min_filled_keys: int = 20,
        plateau_steps: int = 10,
        plateau_evidence_steps: int = 5,
        require_final_checklist: bool = True
    ):
        """
        Initialize the stop validator.
        
        Args:
            min_filled_keys: Minimum number of keys that must be filled
            plateau_steps: Steps without new keys to detect plateau
            plateau_evidence_steps: Steps without new evidence to detect plateau
            require_final_checklist: Whether to require get_checklist before stop
        """
        self.min_filled_keys = min_filled_keys
        self.plateau_steps = plateau_steps
        self.plateau_evidence_steps = plateau_evidence_steps
        self.require_final_checklist = require_final_checklist
        
        # Track progress for plateau detection
        self.keys_filled_history: List[int] = []
        self.evidence_count_history: List[int] = []
        self.last_checklist_step: Optional[int] = None
    
    def validate_stop_decision(
        self,
        checklist_store: ChecklistStore,
        ledger: Ledger,
        current_step: int,
        stop_reason: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate whether the agent should stop.
        
        Args:
            checklist_store: Current checklist state
            ledger: Action ledger
            current_step: Current step number
            stop_reason: Agent's reason for stopping
            
        Returns:
            Tuple of (is_valid, validation_message, details)
        """
        details = {
            "step": current_step,
            "stop_reason": stop_reason,
            "checks": {}
        }
        
        # Check 1: Check if enough keys have values
        empty_keys = checklist_store.get_empty_keys()
        all_filled = len(empty_keys) == 0
        details["checks"]["all_keys_filled"] = {
            "passed": all_filled,
            "empty_count": len(empty_keys),
            "empty_keys": empty_keys[:5]  # First 5 for brevity
        }
        
        # Check 2: Minimum keys filled
        filled_count = self.count_filled_keys(checklist_store)
        min_filled_met = filled_count >= self.min_filled_keys
        details["checks"]["minimum_keys_filled"] = {
            "passed": min_filled_met,
            "filled": filled_count,
            "required": self.min_filled_keys
        }
        
        # Check 3: Critical keys have values
        critical_filled, missing_critical = self.check_critical_keys(checklist_store)
        details["checks"]["critical_keys"] = {
            "passed": critical_filled,
            "missing": missing_critical
        }
        
        # Check 4: Plateau detection
        is_plateau, plateau_info = self.check_plateau_detection(
            checklist_store, ledger, current_step
        )
        details["checks"]["plateau_detection"] = {
            "is_plateau": is_plateau,
            **plateau_info
        }
        
        # Check 5: Final checklist call
        has_final_checklist = self.check_final_checklist_call(
            ledger, current_step
        )
        details["checks"]["final_checklist"] = {
            "has_recent_call": has_final_checklist,
            "last_call_step": self.last_checklist_step
        }
        
        # Determine if stop is valid
        is_valid = False
        reasons = []
        
        # Valid stop conditions
        if all_filled:
            is_valid = True
            reasons.append("All keys have extracted values")
        elif min_filled_met and critical_filled:
            is_valid = True
            reasons.append(f"Minimum keys filled ({filled_count}/{self.min_filled_keys})")
        elif is_plateau and filled_count >= 15:  # Lower threshold with plateau
            is_valid = True
            reasons.append("Plateau reached with sufficient keys")
        
        # Additional requirement: final checklist
        if is_valid and self.require_final_checklist and not has_final_checklist:
            is_valid = False
            reasons.append("Need to call get_checklist() before stopping")
        
        # Generate validation message
        if is_valid:
            message = f"Stop validated: {', '.join(reasons)}"
        else:
            problems = []
            if not all_filled:
                problems.append(f"{len(empty_keys)} empty keys")
            if not min_filled_met:
                problems.append(f"Only {filled_count}/{self.min_filled_keys} keys filled")
            if not critical_filled:
                problems.append(f"Missing critical keys: {', '.join(missing_critical)}")
            if not has_final_checklist:
                problems.append("No recent get_checklist() call")
            message = f"Stop invalid: {', '.join(problems)}"
        
        details["validation_message"] = message
        details["is_valid"] = is_valid
        
        return is_valid, message, details
    
    def check_all_keys_filled(
        self,
        checklist_store: ChecklistStore
    ) -> Tuple[bool, List[str]]:
        """
        Check if all keys have extracted values.
        
        Returns:
            Tuple of (all_filled, list_of_empty_keys)
        """
        empty_keys = checklist_store.get_empty_keys()
        return len(empty_keys) == 0, empty_keys
    
    def count_filled_keys(self, checklist_store: ChecklistStore) -> int:
        """
        Count the number of keys with extracted values.
        
        Returns:
            Number of filled keys
        """
        checklist = checklist_store.get_checklist()
        return sum(1 for item in checklist if item.extracted)
    
    def check_critical_keys(
        self,
        checklist_store: ChecklistStore
    ) -> Tuple[bool, List[str]]:
        """
        Check if critical keys have values.
        
        Critical keys are those that should almost always have values.
        
        Returns:
            Tuple of (all_critical_filled, missing_critical_keys)
        """
        # Define critical keys that should usually have values
        critical_keys = [
            "Filing_Date",
            "Case_Name",
            "Court",
            "Docket_Number",
            "Plaintiff",
            "Defendant"
        ]
        
        checklist = checklist_store.get_checklist()
        missing = []
        
        # Convert list to dict for easier lookup
        checklist_dict = {item.key: item for item in checklist}
        
        for key in critical_keys:
            if key in checklist_dict:
                item = checklist_dict[key]
                if not item.extracted and not item.candidates:
                    missing.append(key)
        
        return len(missing) == 0, missing
    
    def check_plateau_detection(
        self,
        checklist_store: ChecklistStore,
        ledger: Ledger,
        current_step: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if progress has plateaued.
        
        Returns:
            Tuple of (is_plateau, plateau_info)
        """
        # Get current counts
        current_filled = self.count_filled_keys(checklist_store)
        current_evidence = self._count_total_evidence(checklist_store)
        
        # Update history
        self.keys_filled_history.append(current_filled)
        self.evidence_count_history.append(current_evidence)
        
        # Keep only recent history
        if len(self.keys_filled_history) > self.plateau_steps:
            self.keys_filled_history = self.keys_filled_history[-self.plateau_steps:]
        if len(self.evidence_count_history) > self.plateau_evidence_steps:
            self.evidence_count_history = self.evidence_count_history[-self.plateau_evidence_steps:]
        
        info = {
            "current_filled": current_filled,
            "current_evidence": current_evidence,
            "history_length": len(self.keys_filled_history)
        }
        
        # Need enough history to detect plateau
        if len(self.keys_filled_history) < self.plateau_steps:
            info["plateau_reason"] = "Insufficient history"
            return False, info
        
        # Check for key plateau (no new keys filled)
        keys_plateau = all(
            count == self.keys_filled_history[-1] 
            for count in self.keys_filled_history[-self.plateau_steps:]
        )
        
        # Check for evidence plateau (no new evidence)
        evidence_plateau = False
        if len(self.evidence_count_history) >= self.plateau_evidence_steps:
            evidence_plateau = all(
                count == self.evidence_count_history[-1]
                for count in self.evidence_count_history[-self.plateau_evidence_steps:]
            )
        
        is_plateau = keys_plateau or evidence_plateau
        
        if keys_plateau:
            info["plateau_reason"] = f"No new keys in {self.plateau_steps} steps"
        elif evidence_plateau:
            info["plateau_reason"] = f"No new evidence in {self.plateau_evidence_steps} steps"
        else:
            info["plateau_reason"] = "No plateau detected"
        
        return is_plateau, info
    
    def check_final_checklist_call(
        self,
        ledger: Ledger,
        current_step: int,
        recency_threshold: int = 3
    ) -> bool:
        """
        Check if get_checklist was called recently.
        
        Args:
            ledger: Action ledger
            current_step: Current step
            recency_threshold: How recent the call should be
            
        Returns:
            Whether get_checklist was called recently
        """
        # Get recent events from ledger
        recent_events = ledger.get_recent_actions(limit=recency_threshold)
        
        for event in recent_events:
            if hasattr(event, 'tool') and event.tool == 'get_checklist':
                self.last_checklist_step = getattr(event, 'step', current_step)
                return True
        
        return False
    
    def _count_total_evidence(self, checklist_store: ChecklistStore) -> int:
        """
        Count total evidence items across all keys.
        
        Returns:
            Total number of evidence items
        """
        checklist = checklist_store.get_checklist()
        total = 0
        
        for item in checklist:
            for extracted in item.extracted:
                total += len(extracted.evidence)
        
        return total
    
    def suggest_next_action(
        self,
        validation_details: Dict[str, Any]
    ) -> str:
        """
        Suggest what the agent should do based on validation results.
        
        Args:
            validation_details: Details from validation
            
        Returns:
            Suggested action message
        """
        checks = validation_details.get("checks", {})
        
        # If need final checklist
        final_check = checks.get("final_checklist", {})
        if not final_check.get("has_recent_call"):
            return "Call get_checklist() to review final state before stopping"
        
        # If missing critical keys
        critical = checks.get("critical_keys", {})
        if not critical.get("passed"):
            missing = critical.get("missing", [])
            if missing:
                return f"Search for critical keys: {', '.join(missing[:3])}"
        
        # If many unresolved
        all_keys = checks.get("all_keys_resolved", {})
        if not all_keys.get("passed"):
            unresolved = all_keys.get("unresolved_keys", [])
            if unresolved:
                return f"Resolve remaining keys: {', '.join(unresolved[:3])}"
        
        # If plateau but not enough keys
        min_keys = checks.get("minimum_keys_filled", {})
        if not min_keys.get("passed"):
            filled = min_keys.get("filled", 0)
            required = min_keys.get("required", 20)
            return f"Continue searching (need {required - filled} more keys)"
        
        return "Continue with extraction"