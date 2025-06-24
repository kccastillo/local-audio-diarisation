# processors/diarisation_processor.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from .base_processor import BaseProcessor

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from config.config_manager import ConfigManager
    from utils.display_manager import DisplayManager
    from utils.memory_monitor import MemoryMonitor

class DiarisationProcessor(BaseProcessor):
    """
    A processor dedicated to Speaker Diarisation.

    It loads a Pyannote diarisation model, processes an audio file to determine
    speaker turns, and unloads the model to conserve memory.
    """
    def __init__(
        self,
        config: ConfigManager,
        display: DisplayManager,
        memory_monitor: MemoryMonitor,
        device: torch.device,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Initialises the DiarisationProcessor.

        Args:
            auth_token: The Hugging Face authentication token for Pyannote models.
        """
        super().__init__(config, display, memory_monitor, device, processor_name="DiarisationProcessor")
        self.auth_token = auth_token
        # This can be made configurable in config.yaml if needed
        self.model_name = "pyannote/speaker-diarization-3.1"
        self.logger.info(f"Using diarisation model: {self.model_name}")

    def load_model(self) -> None:
        """
        Loads the Pyannote Diarisation pipeline from Hugging Face.
        """
        self.logger.info(f"Loading diarisation model: {self.model_name}")
        if self.auth_token is None:
            self.display.add_warning(f"Hugging Face auth token not provided. Model download may fail if not cached.")
            # Depending on strictness, you could raise an error here.
            # raise ValueError("Authentication token is required for Pyannote models.")

        try:
            self.model = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token
            ).to(self.device)
            self.display.print_metrics({"Diarisation Model": self.model_name})
            self.logger.info("Diarisation model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load diarisation model '{self.model_name}': {e}", exc_info=True)
            self.display.add_warning(f"Could not load diarisation model. Please check your Hugging Face token and internet connection.")
            raise

    def process(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Annotation:
        """
        Processes an audio file to determine speaker turns.

        Args:
            audio_path: The absolute path to the preprocessed audio file.
            min_speakers: An optional hint for the minimum number of speakers.
            max_speakers: An optional hint for the maximum number of speakers.

        Returns:
            A pyannote.core.Annotation object containing the speaker timeline.
        """
        if self.model is None:
            self.logger.error("Diarisation model is not loaded. Cannot process audio.")
            raise RuntimeError("Diarisation model must be loaded before processing.")

        self.logger.info(f"Running diarisation on: {audio_path}")
        self.display.print_metrics({
            "Min Speakers Hint": min_speakers if min_speakers is not None else "Auto",
            "Max Speakers Hint": max_speakers if max_speakers is not None else "Auto"
        })

        try:
            # The model call accepts num_speakers (for a fixed number) or min/max speakers.
            diarisation_result: Annotation = self.model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            num_speakers_found = len(diarisation_result.labels())
            self.logger.info(f"Diarisation processing complete. Found {num_speakers_found} speakers.")
            
            return diarisation_result
            
        except Exception as e:
            self.logger.error(f"An error occurred during diarisation processing: {e}", exc_info=True)
            self.display.add_warning(f"Diarisation processing failed with an error: {e}")
            # Return an empty Annotation object on failure
            return Annotation()
