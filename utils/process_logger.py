# utils/process_logger.py

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class StageMetrics:
    """
    Holds timing and other metrics for a single, named processing stage.
    """
    name: str
    # REMOVED: The default_factory is removed to make time setting explicit.
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    children: List["StageMetrics"] = field(default_factory=list)

class ProcessLogger:
    """
    Tracks the timing and metrics of hierarchical processing stages.

    This class allows for the structured logging of how long each step and sub-step
    of a process takes, which is useful for performance analysis and debugging.
    """
    def __init__(self) -> None:
        """Initialises the ProcessLogger with a root stage."""
        self.root = StageMetrics(name="root")
        self._stage_stack: List[StageMetrics] = [self.root]

    @property
    def current_stage(self) -> StageMetrics:
        """Returns the currently active stage from the top of the stack."""
        return self._stage_stack[-1]

    def start_stage(self, name: str) -> None:
        """
        Starts a new stage, making it a child of the currently active stage.

        Args:
            name: The name of the new stage to start.
        """
        new_stage = StageMetrics(name=name)
        # ADDED: Set the start time explicitly when the stage is created.
        # This ensures that any mocks from the test will be correctly applied.
        new_stage.start_time = time.monotonic()
        
        self.current_stage.children.append(new_stage)
        self._stage_stack.append(new_stage)

    def end_stage(self) -> float:
        """
        Ends the currently active stage, calculates its duration, and returns it.

        Returns:
            The duration of the completed stage in seconds.
        """
        if len(self._stage_stack) <= 1:
            # Cannot end the root stage.
            return 0.0
            
        stage = self._stage_stack.pop()
        stage.end_time = time.monotonic()
        stage.duration = stage.end_time - stage.start_time
        return stage.duration

    def add_metric(self, name: str, value: Any) -> None:
        """
        Adds a key-value metric to the currently active stage.

        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        self.current_stage.metrics[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire logged process tree into a dictionary."""
        def stage_to_dict(stage: StageMetrics) -> Dict[str, Any]:
            return {
                "stage_name": stage.name,
                "duration_seconds": round(stage.duration, 3),
                "metrics": stage.metrics,
                "sub_stages": [stage_to_dict(child) for child in stage.children]
            }
        
        # We convert the children of the root, not the root itself.
        return {"process_tree": [stage_to_dict(child) for child in self.root.children]}

