# processors/vad_processor.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from pyannote.audio import Pipeline

from .base_processor import BaseProcessor

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from config.config_manager import ConfigManager
    from utils.display_manager import DisplayManager
    from utils.memory_monitor import MemoryMonitor

class VADProcessor(BaseProcessor):
    """
    A processor dedicated to Voice Activity Detection (VAD).

    It loads the Pyannote VAD model, processes an audio file to find speech
    segments, and unloads the model to conserve memory.
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
        Initialises the VADProcessor.

        Args:
            auth_token: The Hugging Face authentication token for Pyannote models.
        """
        super().__init__(config, display, memory_monitor, device, processor_name="VADProcessor")
        self.auth_token = auth_token
        self.model_name = "pyannote/voice-activity-detection"

    def load_model(self) -> None:
        """
        Loads the Pyannote VAD pipeline from Hugging Face.
        """
        self.logger.info(f"Loading VAD model: {self.model_name}")
        if self.auth_token is None:
            self.display.add_warning(f"Hugging Face auth token not provided. Model download may fail if not cached.")
            # Depending on strictness, you could raise an error here.
            # raise ValueError("Authentication token is required for Pyannote models.")

        try:
            self.model = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token
            ).to(self.device)
            self.display.print_metrics({"VAD Model": self.model_name})
            self.logger.info("VAD model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load VAD model '{self.model_name}': {e}", exc_info=True)
            self.display.add_warning(f"Could not load VAD model. Please check your Hugging Face token and internet connection.")
            raise

    def process(self, audio_path: str) -> List[Dict[str, float]]:
        """
        Processes an audio file to detect speech segments.

        Args:
            audio_path: The absolute path to the preprocessed audio file.

        Returns:
            A list of dictionaries, where each dictionary represents a speech
            segment with 'start' and 'end' keys in seconds.
        """
        if self.model is None:
            self.logger.error("VAD model is not loaded. Cannot process audio.")
            raise RuntimeError("VAD model must be loaded before processing.")

        self.logger.info(f"Running VAD on: {audio_path}")
        try:
            vad_result = self.model(audio_path)
            
            # The output of the VAD pipeline is a pyannote.core.Annotation object.
            # We convert its speech timeline into a simple list of dicts.
            speech_regions: List[Dict[str, float]] = [
                {"start": segment.start, "end": segment.end}
                for segment in vad_result.get_timeline().support()
            ]
            
            self.logger.info(f"VAD processing complete. Found {len(speech_regions)} speech segments.")
            return speech_regions
            
        except Exception as e:
            self.logger.error(f"An error occurred during VAD processing: {e}", exc_info=True)
            self.display.add_warning(f"VAD processing failed with an error: {e}")
            # Return an empty list or re-raise the exception depending on desired behavior
            return []

