# audio/audio_cleaner.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

# Use TYPE_CHECKING block for type hints to avoid circular imports
if TYPE_CHECKING:
    from config.config_manager import ConfigManager
    from utils.display_manager import DisplayManager

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Handles the initial stages of audio processing: loading, cleaning,
    and standardising audio from various formats into a clean WAV file.
    """

    def __init__(
        self,
        config: ConfigManager,
        display: DisplayManager,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialises the AudioPreprocessor.

        Args:
            config: The application's ConfigManager instance.
            display: The application's DisplayManager instance.
            input_dir: The directory to read source files from. Defaults to config setting.
            output_dir: The directory to save processed files to. Defaults to config setting.
        """
        self.config = config
        self.display = display

        self.input_dir = input_dir or self.config.get_path('recordings_dir')
        self.output_dir = output_dir or self.config.get_path('input_dir')
        self.temp_dir = self.config.get_path('temp_dir')

        # Ensure directories exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        # Audio parameters from config
        self.target_sr = self.config.get('audio', 'capture', 'rate', default=16000)
        self.norm_peak = self.config.get('audio', 'preprocessing', 'normalization', 'target_peak', default=0.7)
        self.nr_prop_decrease = self.config.get('audio', 'preprocessing', 'noise_reduction', 'prop_decrease', default=0.6)
        self.nr_is_stationary = self.config.get('audio', 'preprocessing', 'noise_reduction', 'stationary', default=True)
        # ADDED: Load the frequency mask setting from config
        self.nr_freq_mask_hz = self.config.get('audio', 'preprocessing', 'noise_reduction', 'freq_mask_smooth_hz', default=500)


    def process_audio(self, filename: str) -> Path:
        """
        Orchestrates the full audio preprocessing pipeline.

        Args:
            filename: The name of the file in the input directory to process.

        Returns:
            The path to the final processed WAV file.
        """
        with self.display.level("Loading Audio File"):
            audio_data, original_sr = self._load_audio(filename)

        with self.display.level("Standardising Audio Format"):
            if audio_data.ndim > 1:
                audio_data = self._convert_to_mono(audio_data)
            
            # Resample to the target sample rate (e.g., 16kHz) required by models
            if original_sr != self.target_sr:
                with self.display.level("Resampling Audio"):
                    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.target_sr)
                    self.display.print_metrics({
                        "Original Rate": f"{original_sr} Hz",
                        "Target Rate": f"{self.target_sr} Hz"
                    })

        with self.display.level("Applying Noise Reduction"):
            audio_data = self._apply_noise_reduction(audio_data, self.target_sr)

        with self.display.level("Normalising Audio Volume"):
            audio_data = self._normalise_audio(audio_data)

        with self.display.level("Saving Processed Audio", is_last=True):
            output_path = self._save_processed_audio(audio_data, self.target_sr, filename)

        return output_path

    def _load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file from various formats (MP4, M4A, WAV, etc.),
        converting it to a NumPy array.
        """
        input_path = self.input_dir / filename
        logger.info(f"Loading audio from: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")

        file_suffix = input_path.suffix.lower()
        
        # For non-WAV files, convert to a temporary WAV first using pydub
        if file_suffix in ['.mp4', '.m4a', '.mp3', '.ogg', '.flac']:
            with self.display.level("Converting Media Format"):
                try:
                    audio_segment = AudioSegment.from_file(input_path)
                    temp_wav_path = self.temp_dir / f"{input_path.stem}_conversion.wav"
                    
                    progress = self.display.create_progress("Converting to WAV", 100)
                    audio_segment.export(temp_wav_path, format='wav')
                    progress.update(100)
                    progress.close()

                    self.display.print_metrics({"Original Format": file_suffix.strip('.').upper()})
                    load_path = temp_wav_path
                except Exception as e:
                    logger.error(f"Failed to convert {filename} using pydub: {e}", exc_info=True)
                    self.display.add_warning(f"Audio conversion failed for {filename}. Ensure FFmpeg is installed.")
                    raise IOError(f"Failed to process media file: {filename}") from e
        else:
            load_path = input_path

        try:
            with self.display.level("Reading Audio Data", is_last=True):
                audio_data, sr = librosa.load(load_path, sr=None, mono=False)
                self.display.print_metrics({
                    "Sample Rate": f"{sr} Hz",
                    "Channels": "Stereo" if audio_data.ndim > 1 else "Mono",
                    "Duration": f"{librosa.get_duration(y=audio_data, sr=sr):.1f}s"
                })
        except Exception as e:
            logger.error(f"Failed to load audio data from {load_path}: {e}", exc_info=True)
            raise IOError(f"Could not read audio data from file: {load_path}") from e
        finally:
            if 'temp_wav_path' in locals() and temp_wav_path.exists():
                temp_wav_path.unlink()

        return audio_data, sr

    def _convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Converts stereo numpy array to mono by averaging channels."""
        with self.display.level("Converting to Mono"):
            mono_data = np.mean(audio_data, axis=0)
            self.display.print_metrics({
                "Original Channels": audio_data.shape[0],
                "New Channels": 1
            })
            return mono_data

    def _apply_noise_reduction(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Reduces background noise in the audio data."""
        chunk_size = sr * 15
        processed_chunks = []
        
        progress = self.display.create_progress("Reducing Noise", len(audio_data))
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            try:
                reduced_chunk = nr.reduce_noise(
                    y=chunk,
                    sr=sr,
                    prop_decrease=self.nr_prop_decrease,
                    stationary=self.nr_is_stationary,
                    # ADDED: Use the frequency mask setting from the config
                    freq_mask_smooth_hz=self.nr_freq_mask_hz
                )
                processed_chunks.append(reduced_chunk)
            except Exception as e:
                logger.warning(f"Noise reduction failed on a chunk, using original chunk. Error: {e}")
                processed_chunks.append(chunk)
            progress.update(len(chunk))
        
        progress.close()
        return np.concatenate(processed_chunks)

    def _normalise_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalises the audio to a target peak amplitude."""
        current_peak = np.max(np.abs(audio_data))
        if current_peak == 0:
            self.display.add_warning("Audio is silent, cannot normalise.")
            return audio_data

        gain = self.norm_peak / current_peak
        normalised_data = audio_data * gain
        
        self.display.print_metrics({
            "Original Peak": f"{20 * np.log10(current_peak):.1f} dB",
            "Target Peak": f"{20 * np.log10(self.norm_peak):.1f} dB",
            "Applied Gain": f"{20 * np.log10(gain):.1f} dB"
        })
        return normalised_data

    def _save_processed_audio(self, audio_data: np.ndarray, sr: int, original_filename: str) -> Path:
        """Saves the fully processed audio data to a standard WAV file."""
        output_filename = f"{Path(original_filename).stem}_processed.wav"
        output_path = self.output_dir / output_filename

        try:
            sf.write(output_path, audio_data, sr, subtype='PCM_16')
            file_size_kb = output_path.stat().st_size / 1024
            self.display.print_metrics({
                "Saved Processed File": output_filename,
                "Format": "16-bit PCM WAV",
                "Size": f"{file_size_kb:.1f} KB"
            })
            return output_path
        except Exception as e:
            logger.error(f"Failed to save processed audio to {output_path}: {e}", exc_info=True)
            raise IOError(f"Could not write processed audio file: {output_path}") from e
