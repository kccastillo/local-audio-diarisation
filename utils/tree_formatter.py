# utils/tree_formatter.py

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TreeSymbols:
    """
    A dataclass holding the Unicode characters for drawing tree structures in the console.
    'frozen=True' makes instances of this class immutable.
    """
    PIPE: str = "│"
    TEE: str = "├─"
    LAST: str = "└─"
    SPACE: str = "  " # Two spaces for alignment
    WARNING_PREFIX: str = "[!]"

class TreeFormatter:
    """
    Manages the state and generation of prefixes for creating a tree-like
    structure in text output, which helps in visualising hierarchical processes.
    """
    def __init__(self) -> None:
        """Initialises the formatter with an empty level stack."""
        # This stack tracks whether the current level is the last item
        # at each depth, which determines whether to use a PIPE or SPACE.
        self._is_last_stack: list[bool] = []

    def get_prefix(self, is_last_item: bool) -> str:
        """
        Generates the appropriate prefix string (e.g., "│  ├─ ") for the current level.

        Args:
            is_last_item: A boolean indicating if the current item is the last
                          one in its parent's list of children.

        Returns:
            A string representing the tree prefix for the current line.
        """
        if not self._is_last_stack:
            return "" # No prefix for the root level

        parts: list[str] = []
        # Iterate through parent levels to build the connecting lines
        for is_last in self._is_last_stack[:-1]:
            # If the parent level was the last item, draw a space.
            # Otherwise, draw a vertical pipe to show the connection continues.
            parts.append(TreeSymbols.SPACE if is_last else TreeSymbols.PIPE + " ")

        # Add the connector for the current item
        parts.append(TreeSymbols.LAST if is_last_item else TreeSymbols.TEE)
        
        return "".join(parts) + " "

    def push_level(self, is_last_item: bool) -> None:
        """
        Enters a new, deeper level in the tree structure.

        Args:
            is_last_item: Whether the new level being entered is the last
                          child of its current parent.
        """
        self._is_last_stack.append(is_last_item)

    def pop_level(self) -> None:
        """Exits the current level, moving back up the tree."""
        if self._is_last_stack:
            self._is_last_stack.pop()

