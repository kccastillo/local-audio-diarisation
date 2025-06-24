# utils/progress_visualiser.py

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ProgressType(Enum):
    """Enumeration for the available types of progress visualisation."""
    TQDM = auto()   # Use the tqdm library for rich progress bars.
    SIMPLE = auto() # Use simple text-based percentage updates.
    NONE = auto()   # No progress output.

class ProgressBar:
    """A wrapper class for progress bars to standardise their usage."""

    def __init__(
        self,
        desc: str,
        total: Optional[int],
        prefix: str = "",
        progress_type: ProgressType = ProgressType.TQDM
    ) -> None:
        """
        Initialises a ProgressBar instance.

        Args:
            desc: The description of the task for the progress bar.
            total: The total number of iterations. Can be None for indeterminate progress.
            prefix: A string to prepend to the description (used for indentation).
            progress_type: The type of progress visualisation to use.
        """
        self.desc = desc
        self.total = total
        self.prefix = prefix
        self.progress_type = progress_type
        self.current = 0
        self.bar: Optional[tqdm] = None

        if self.progress_type == ProgressType.TQDM:
            try:
                self.bar = tqdm(
                    total=total,
                    desc=f"{prefix}{desc}",
                    unit="it",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
            except Exception as e:
                logger.error(f"Failed to initialise tqdm progress bar: {e}", exc_info=True)
                self.progress_type = ProgressType.SIMPLE # Fallback to simple
        
        # No initialisation needed for SIMPLE or NONE types

    def update(self, amount: float = 1.0) -> None:
        """Updates the progress by a specified amount."""
        if self.total is not None:
            self.current = min(self.current + amount, self.total)

        if self.bar:
            self.bar.update(amount)
        elif self.progress_type == ProgressType.SIMPLE and self.total is not None and self.total > 0:
            percentage = (self.current / self.total) * 100
            # Use carriage return to keep progress on one line
            print(f"\r{self.prefix}{self.desc}: {percentage:.1f}%", end="")

    def set_description(self, desc: str) -> None:
        """Updates the description of the progress bar."""
        self.desc = desc
        if self.bar:
            self.bar.set_description(f"{self.prefix}{self.desc}")

    def close(self) -> None:
        """Closes and cleans up the progress bar instance."""
        if self.bar:
            self.bar.close()
        elif self.progress_type == ProgressType.SIMPLE:
            # Print a newline to move to the next line after progress is done
            print()

class ProgressFactory:
    """A factory class for creating and managing ProgressBar instances."""
    _default_type: ProgressType = ProgressType.TQDM

    @classmethod
    def set_progress_type(cls, progress_type: ProgressType) -> None:
        """
        Sets the default progress bar type for all subsequently created bars.
        
        Args:
            progress_type: The ProgressType to set as the default.
        """
        if not isinstance(progress_type, ProgressType):
            logger.warning(f"Invalid progress type '{progress_type}'. Defaulting to TQDM.")
            cls._default_type = ProgressType.TQDM
        else:
            cls._default_type = progress_type
            logger.debug(f"Default progress type set to {progress_type.name}")

    @classmethod
    def create_bar(
        cls,
        desc: str,
        total: Optional[int],
        prefix: str = "",
        progress_type: Optional[ProgressType] = None
    ) -> ProgressBar:
        """
        Creates a new ProgressBar instance using the factory's default type.

        Args:
            desc: The description for the progress bar.
            total: The total number of items.
            prefix: The indentation prefix.
            progress_type: Override the factory's default type for this specific bar.

        Returns:
            An initialised ProgressBar instance.
        """
        effective_type = progress_type if progress_type is not None else cls._default_type
        return ProgressBar(desc, total, prefix, effective_type)

