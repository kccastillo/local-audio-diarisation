# utils/display_manager.py

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

# Import the TreeFormatter and its symbols
from .tree_formatter import TreeFormatter, TreeSymbols
# Import the ProcessLogger to integrate timing
from .process_logger import ProcessLogger

# Use TYPE_CHECKING block for imports that will exist later.
if TYPE_CHECKING:
    from .progress_visualiser import ProgressBar
    from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class DisplayManager:
    """
    Manages the hierarchical display of processing stages and information.

    This class uses a TreeFormatter to create structured, indented output in the console,
    making complex, multi-stage processes easier to visualise and debug.
    """

    def __init__(self, config: "ConfigManager | None" = None) -> None:
        """
        Initialises the DisplayManager.

        Args:
            config: An optional ConfigManager instance (used for future configuration).
        """
        self.tree = TreeFormatter()
        self.config = config
        self.warning_buffer: List[str] = []
        
        # Initialise the process logger for timing
        self.process_logger = ProcessLogger()

    @contextmanager
    def section(self, name: str, number: int, total: int) -> Iterator[None]:
        """
        A context manager for creating a main, top-level numbered section.
        This also logs the timing for the entire section.

        Example:
            [1/4] Audio Preprocessing
        """
        self.process_logger.start_stage(name)
        try:
            print(f"\n--- [{number}/{total}] {name} ---")
            yield
        finally:
            self._flush_warnings()
            duration = self.process_logger.end_stage()
            # Only show timing for sections that take a noticeable amount of time
            if duration >= 1.0:
                # Use a simple prefix for the final timing line
                print(f"  └─ Completed in {self._format_time(duration)}")


    @contextmanager
    def level(self, text: str, is_last: bool = False) -> Iterator["DisplayManager"]:
        """
        A context manager for creating a nested level within a section.
        This generates the tree structure prefix and logs the timing for the sub-step.
        """
        self.process_logger.start_stage(text)
        self.tree.push_level(is_last_item=is_last)
        prefix = self.tree.get_prefix(is_last_item=is_last)
        print(f"{prefix}{text}")

        try:
            yield self
        finally:
            self._flush_warnings()
            self.tree.pop_level()
            self.process_logger.end_stage() # End the timing for this specific level

    def add_warning(self, message: str) -> None:
        """
        Buffers a warning message to be displayed at the end of the current level.
        This prevents warnings from interrupting a clean progress display.

        Args:
            message: The warning message string.
        """
        logger.warning(message) # Also log it immediately
        self.warning_buffer.append(message)

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Prints a dictionary of key-value metrics, indented under the current level.

        Args:
            metrics: A dictionary where keys are metric names and values are the metric info.
        """
        self.tree.push_level(is_last_item=True)
        prefix = self.tree.get_prefix(is_last_item=False)
        self.tree.pop_level()

        for name, value in metrics.items():
            print(f"{prefix}{name}: {value}")
            # Also add the metric to the process logger for later analysis
            self.process_logger.add_metric(name, value)


    def create_progress(self, desc: str, total: int) -> "ProgressBar":
        """
        Creates a progress bar instance, indented under the current level.
        """
        from .progress_visualiser import ProgressFactory
        
        self.tree.push_level(is_last_item=True)
        prefix = self.tree.get_prefix(is_last_item=True)
        self.tree.pop_level()
        
        return ProgressFactory.create_bar(desc, total, prefix=prefix)


    def _flush_warnings(self) -> None:
        """Prints any buffered warnings with appropriate indentation."""
        if not self.warning_buffer:
            return

        self.tree.push_level(is_last_item=True)
        prefix = self.tree.get_prefix(is_last_item=False)
        self.tree.pop_level()
        
        print(f"{prefix.replace(TreeSymbols.TEE, TreeSymbols.PIPE)}")

        for warning in self.warning_buffer:
            print(f"{prefix}{TreeSymbols.WARNING_PREFIX} {warning}")
        
        print(f"{prefix.replace(TreeSymbols.TEE, TreeSymbols.PIPE)}")
        
        self.warning_buffer.clear()

    def _format_time(self, seconds: float) -> str:
        """
        Formats a duration in seconds into a human-readable string (MM:SS or HH:MM:SS).
        """
        if seconds < 0:
            seconds = 0
            
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"
