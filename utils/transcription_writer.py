# utils/transcription_writer.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

# Import the core data type
from .datatypes import TranscriptionSegment

# Use TYPE_CHECKING to avoid circular import issues if DisplayManager ever needed to import from here.
# This makes the import available for type hinting but not at runtime.
if TYPE_CHECKING:
    from .display_manager import DisplayManager

logger = logging.getLogger(__name__)

class TranscriptionWriter:
    """
    Handles saving a list of TranscriptionSegment objects to a file
    in various formats (e.g., TXT, JSON).
    """

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Converts a duration in seconds into a standard HH:MM:SS format.

        Args:
            seconds: The duration in total seconds.

        Returns:
            A string formatted as "HH:MM:SS".
        """
        # Ensure seconds is a non-negative number
        if seconds < 0:
            seconds = 0
        
        hours, remainder = divmod(seconds, 3600)
        minutes, remainder = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(remainder):02d}"

    @classmethod
    def _save_txt(cls, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Saves the transcript in a human-readable .txt format."""
        logger.info(f"Saving transcript to TXT file: {output_path}")
        try:
            with output_path.open('w', encoding='utf-8') as f:
                for segment in segments:
                    # Format: [HH:MM:SS --> HH:MM:SS] Speaker X: Text
                    start_time = cls._format_timestamp(segment.start)
                    end_time = cls._format_timestamp(segment.end)
                    timestamp = f"[{start_time} --> {end_time}]"
                    f.write(f"{timestamp} {segment.speaker}: {segment.text}\n")
        except IOError as e:
            logger.error(f"IOError while writing to {output_path}: {e}", exc_info=True)
            # Re-raise to let the caller handle the file writing failure
            raise

    @classmethod
    def _save_json(cls, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Saves the transcript in a machine-readable .json format."""
        logger.info(f"Saving transcript to JSON file: {output_path}")
        try:
            # Convert the list of dataclass objects into a list of dictionaries
            json_data = [
                {
                    "speaker": segment.speaker,
                    "start": round(segment.start, 3), # Round for cleaner output
                    "end": round(segment.end, 3),
                    "text": segment.text
                }
                for segment in segments
            ]
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except (IOError, TypeError) as e:
            logger.error(f"Error while writing JSON to {output_path}: {e}", exc_info=True)
            # Re-raise to let the caller handle the failure
            raise

    @classmethod
    def save_transcript(
        cls,
        segments: List[TranscriptionSegment],
        output_path: Path,
        format: str = "txt",
        display: "DisplayManager | None" = None
    ) -> None:
        """
        Main method to save a transcript in the specified format.

        Args:
            segments: A list of TranscriptionSegment objects to save.
            output_path: The Path object for the output file.
            format: The desired output format ('txt' or 'json').
            display: An optional DisplayManager instance for logging metrics.

        Raises:
            ValueError: If an unsupported format is requested.
        """
        logger.info(f"Attempting to save transcript in '{format}' format to {output_path}")

        # Ensure the parent directory exists
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create parent directory for {output_path}: {e}", exc_info=True)
            # Re-raise as an IOError as it's a file system issue
            raise IOError(f"Failed to create directory for output file: {output_path.parent}") from e


        if format.lower() == "txt":
            cls._save_txt(segments, output_path)
        elif format.lower() == "json":
            cls._save_json(segments, output_path)
        else:
            error_msg = f"Unsupported transcript format requested: '{format}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if display:
            try:
                # Use a try-except block for display metrics so a display error
                # doesn't prevent the successful save from being recognised.
                file_size_kb = output_path.stat().st_size / 1024
                display.print_metrics({
                    "Transcript Saved": output_path.name,
                    "Format": format.upper(),
                    "Segments Saved": len(segments),
                    "File Size": f"{file_size_kb:.1f} KB"
                })
            except Exception as e:
                logger.warning(f"Failed to print display metrics after saving transcript: {e}")

