# utils/datatypes.py

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TranscriptionSegment:
    """
    Represents a single, continuous segment of a transcript attributed to a speaker.

    This is an immutable data structure, which helps prevent accidental modification
    after it has been created.

    Attributes:
        speaker (str): The identified speaker label (e.g., "Speaker 1").
        start (float): The start time of the segment in seconds from the beginning of the audio.
        end (float): The end time of the segment in seconds.
        text (str): The transcribed text spoken during this segment.
    """
    speaker: str
    start: float
    end: float
    text: str

