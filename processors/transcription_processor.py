# processors/transcription_processor.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import whisper

from .base_processor import BaseProcessor

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from config.config_manager import ConfigManager
    from utils.display_manager import DisplayManager
    from utils.memory_monitor import MemoryMonitor

class TranscriptionProcessor(BaseProcessor):
    """
    A processor dedicated to speech-to-text transcription using OpenAI's Whisper.

    It loads a specified Whisper model, processes audio segments to generate text,
    and unloads the model to conserve memory.
    """
    def __init__(
        self,
        config: ConfigManager,
        display: DisplayManager,
        memory_monitor: MemoryMonitor,
        device: torch.device,
    ) -> None:
        """
        Initialises the TranscriptionProcessor.
        """
        super().__init__(config, display, memory_monitor, device, processor_name="TranscriptionProcessor")
        # CORRECTED: The default value must be passed as a keyword argument 'default='.
        self.model_size = self.config.get('processing', 'whisper', 'model_size', default='base')
        self.logger.info(f"Using Whisper model size: {self.model_size}")

    def load_model(self) -> None:
        """
        Loads the specified Whisper model into memory.
        """
        self.logger.info(f"Loading Whisper model '{self.model_size}' to device '{self.device.type}'.")
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.display.print_metrics({
                "Whisper Model": self.model_size,
                "Device": self.device.type,
                "Parameters": f"{sum(p.numel() for p in self.model.parameters()) // 1_000_000}M"
            })
            self.logger.info("Whisper model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model '{self.model_size}': {e}", exc_info=True)
            self.display.add_warning(f"Could not load Whisper model. Please check model name and network.")
            raise

    def process(
        self,
        audio_path: str,
        speech_regions: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Transcribes speech segments from an audio file.

        Args:
            audio_path: The absolute path to the preprocessed audio file.
            speech_regions: A list of speech segments (dicts with 'start' and 'end')
                            as detected by the VAD processor.

        Returns:
            A list of raw segment dictionaries as produced by Whisper, with
            timestamps adjusted to be relative to the original audio file.
        """
        if self.model is None:
            self.logger.error("Whisper model is not loaded. Cannot process audio.")
            raise RuntimeError("Whisper model must be loaded before processing.")

        self.logger.info(f"Running Whisper transcription on {len(speech_regions)} speech segments.")
        
        try:
            # Load the entire audio file into memory once for efficiency
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        except Exception as e:
            self.logger.error(f"Failed to read audio file {audio_path} for transcription: {e}", exc_info=True)
            self.display.add_warning(f"Could not read audio file: {audio_path}")
            return []

        all_segments: List[Dict[str, Any]] = []
        progress = self.display.create_progress("Transcribing Segments", len(speech_regions))

        for region in speech_regions:
            start_time, end_time = region['start'], region['end']
            
            # Convert start/end times to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract the audio chunk for this region
            audio_chunk = audio_data[start_sample:end_sample]

            if len(audio_chunk) < 100: # Skip tiny, likely false-positive chunks
                progress.update(1)
                continue

            try:
                # Transcribe the chunk
                result = self.model.transcribe(
                    audio_chunk,
                    language='en', # Can be made configurable
                    fp16=torch.cuda.is_available() # Use fp16 for speed on GPU
                )

                # The result contains segments with timestamps relative to the chunk.
                # We need to adjust them to be relative to the full audio file.
                if 'segments' in result:
                    for segment in result['segments']:
                        segment['start'] += start_time
                        segment['end'] += start_time
                        all_segments.append(segment)
            except Exception as e:
                self.logger.warning(
                    f"Whisper failed to transcribe a chunk from {start_time:.2f}s to {end_time:.2f}s. Error: {e}"
                )
            
            progress.update(1)

        progress.close()
        self.logger.info(f"Transcription complete. Generated {len(all_segments)} raw text segments.")
        return all_segments
