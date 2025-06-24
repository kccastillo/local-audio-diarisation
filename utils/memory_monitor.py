# utils/memory_monitor.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """
    Represents a single point-in-time snapshot of memory usage.

    Attributes:
        stage (str): The name of the processing stage when the snapshot was taken.
        timestamp (datetime): The exact time of the snapshot.
        allocated_gb (float): Allocated VRAM in gigabytes.
        peak_gb (float): Peak VRAM allocated by PyTorch's memory manager since the start.
    """
    stage: str
    timestamp: datetime = field(default_factory=datetime.now)
    allocated_gb: float = 0.0
    peak_gb: float = 0.0

    def to_dict(self) -> Dict[str, str | float]:
        """Converts the snapshot to a dictionary for JSON serialisation."""
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "allocated_gb": round(self.allocated_gb, 3),
            "peak_gb": round(self.peak_gb, 3),
        }

class MemoryMonitor:
    """
    A utility class to track GPU VRAM and system RAM usage at various stages
    of the processing pipeline.
    """
    def __init__(self) -> None:
        """Initialises the memory monitor."""
        self.history: List[MemorySnapshot] = []
        self.peak_vram_gb: float = 0.0

    def get_memory_status(self, stage: str = "current") -> MemorySnapshot:
        """
        Takes a snapshot of the current memory usage.

        Args:
            stage: A descriptive name for the current processing stage.

        Returns:
            A MemorySnapshot object containing the current memory usage details.
        """
        snapshot = MemorySnapshot(stage=stage)

        if torch.cuda.is_available():
            try:
                allocated_bytes = torch.cuda.memory_allocated()
                peak_bytes = torch.cuda.max_memory_allocated()

                snapshot.allocated_gb = allocated_bytes / 1e9
                snapshot.peak_gb = peak_bytes / 1e9

                # Update the overall peak VRAM usage tracked by this monitor instance
                if snapshot.allocated_gb > self.peak_vram_gb:
                    self.peak_vram_gb = snapshot.allocated_gb

                logger.debug(
                    f"Memory snapshot at stage '{stage}': "
                    f"Allocated={snapshot.allocated_gb:.2f}GB, "
                    f"Peak={snapshot.peak_gb:.2f}GB"
                )
            except Exception as e:
                logger.error(f"Failed to get CUDA memory status: {e}", exc_info=True)
        
        self.history.append(snapshot)
        return snapshot

    def save_history(self, output_path: Path) -> None:
        """
        Saves the recorded history of memory snapshots to a JSON file.

        Args:
            output_path: The file path to save the JSON history to.
        """
        logger.info(f"Saving memory usage history to: {output_path}")
        try:
            history_data = [snapshot.to_dict() for snapshot in self.history]
            
            # Get system RAM for context
            system_ram_total_gb = psutil.virtual_memory().total / 1e9
            
            final_data = {
                "summary": {
                    "peak_vram_usage_gb": round(self.peak_vram_gb, 3),
                    "total_system_ram_gb": round(system_ram_total_gb, 3),
                },
                "snapshots": history_data,
            }
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2)
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save memory history to {output_path}: {e}", exc_info=True)
        except ImportError:
            logger.warning("psutil is not installed. Cannot report system RAM.")


    def clear_history(self) -> None:
        """Clears the snapshot history and resets the peak memory counter."""
        self.history.clear()
        self.peak_vram_gb = 0.0
        # Reset PyTorch's peak memory counter for a clean slate on the next run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.debug("Memory monitor history cleared.")

