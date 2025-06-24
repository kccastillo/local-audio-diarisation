# run_diariser.py (Finalised Orchestrator)

from __future__ import annotations

import time
import subprocess
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
import psutil

# --- Core Utilities ---
from config.config_manager import ConfigManager, ConfigurationError
from utils.display_manager import DisplayManager
from utils.memory_monitor import MemoryMonitor
from utils.progress_visualiser import ProgressFactory, ProgressType

# --- Data Types ---
from utils.datatypes import TranscriptionSegment

# --- IO and Preprocessing ---
from audio.audio_cleaner import AudioPreprocessor
from utils.transcription_writer import TranscriptionWriter
# from audio.audio_recorder import AudioRecorder, list_devices # Optional: uncomment if needed

# --- NEW PROCESSOR IMPORTS ---
from processors.vad_processor import VADProcessor
from processors.diarisation_processor import DiarisationProcessor
from processors.transcription_processor import TranscriptionProcessor

# --- Pyannote Core (needed for attribution logic) ---
from pyannote.core import Segment, Annotation

# Setup basic logger in case manager fails early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptionManager:
    """
    Orchestrates the audio diarisation pipeline by sequentially invoking
    dedicated processors for VAD, diarisation, and transcription.
    Manages configuration, display, and overall workflow.
    """
    def __init__(self):
        """Initialises configuration, display, logging, paths, and GPU device."""
        try:
            self.config = ConfigManager()
            self.display = DisplayManager(self.config)
            self._setup_logging()
            self.logger.info("Transcription Manager initialised.")

            self.recordings_dir = self.config.get_path('recordings_dir')
            self.input_dir = self.config.get_path('input_dir')
            self.output_dir = self.config.get_path('output_dir')
            self.temp_dir = self.config.get_path('temp_dir')
            self.token_path = Path(self.config.get('auth', 'token_path'))
            self.memory_monitor = MemoryMonitor()

            self.device = self._setup_gpu()

            progress_type_str = self.config.get('display', 'progress_type', default='TQDM')
            progress_type = getattr(ProgressType, progress_type_str.upper(), ProgressType.TQDM)
            ProgressFactory.set_progress_type(progress_type)

        except (ConfigurationError, ImportError) as e:
            logger.error(f"Fatal Initialisation Error: {e}", exc_info=True)
            print(f"\nFatal Initialisation Error: {e}\nPlease ensure all modules exist and config is correct.", file=sys.stderr)
            sys.exit(1)

    def _setup_logging(self):
        """Configures logging using settings from ConfigManager."""
        self.logger = logging.getLogger(__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_level_str = self.config.get('logging', 'level', default='INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        log_format = self.config.get('logging', 'format', default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file_prefix = self.config.get('logging', 'file_prefix', default='diariser')
        timestamp_format = self.config.get('output', 'timestamp_format', default='%Y%m%d_%H%M%S')
        log_file = self.config.get_path('logs_dir') / f"{log_file_prefix}_{datetime.now().strftime(timestamp_format)}.log"

        formatter = logging.Formatter(log_format)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(log_level)
        self.logger.info(f"Logging configured. Level: {log_level_str}. File: {log_file}")

    def _setup_gpu(self) -> torch.device:
        """Sets up the GPU device for PyTorch operations."""
        with self.display.level("Setting up Compute Device"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.display.print_metrics({"Device": f"GPU ({gpu_name})", "VRAM": f"{total_mem_gb:.1f} GB"})
                self.logger.info(f"Using GPU: {gpu_name}")
                return device
            else:
                self.display.print_metrics({"Device": "CPU"})
                self.logger.warning("CUDA not available. Using CPU.")
                return torch.device("cpu")

    def _read_auth_token(self) -> str | None:
        """Reads the Hugging Face authentication token from file."""
        try:
            return self.token_path.read_text().strip() or None
        except FileNotFoundError:
            self.logger.warning(f"Auth token file not found at: {self.token_path}")
            return None

    def list_available_recordings(self) -> list[Path]:
        """Lists available audio/video recordings in the recordings directory."""
        patterns = ['*.mp4', '*.m4a', '*.wav', '*.mp3', '*.ogg', '*.flac']
        return sorted({file for p in patterns for file in self.recordings_dir.glob(p)})

    def get_audio_duration(self, file_path: Path) -> str:
        """Gets the duration of an audio/video file using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
                capture_output=True, text=True, check=True, timeout=15
            )
            duration_sec = float(result.stdout)
            hours, rem = divmod(duration_sec, 3600)
            mins, secs = divmod(rem, 60)
            return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"
        except Exception:
            return "??:??:??"

    def get_input_source(self) -> Optional[Path]:
        """Interactively prompts the user to select an input source."""
        files = self.list_available_recordings()
        print("\n--- Input Source Selection ---")
        options = {i + 1: file for i, file in enumerate(files)}
        options[0] = "exit"

        for i, file in options.items():
            if i > 0:
                print(f"  {i}. {file.name} ({self.get_audio_duration(file)})")
        print("\n  0. Exit")

        while True:
            try:
                choice = int(input(f"\nSelect option [0-{len(files)}]: "))
                if choice in options:
                    selection = options[choice]
                    if selection == "exit": return None
                    return selection
            except (ValueError, KeyError):
                print("Invalid choice. Please try again.")

    def process_file(self, input_file: Path, auth_token: str | None, **kwargs) -> Optional[Path]:
        """Processes a single audio file through the sequential pipeline."""
        total_steps = 5
        min_speakers = kwargs.get('min_speakers')
        max_speakers = kwargs.get('max_speakers')
        output_format = kwargs.get('format')

        self.logger.info(f"Starting processing for: {input_file.name}")
        self.memory_monitor.get_memory_status("start_processing")
        processed_audio_path = None

        try:
            with self.display.section("Audio Preprocessing", 1, total_steps):
                preprocessor = AudioPreprocessor(self.config, self.display)
                processed_audio_path = preprocessor.process_audio(input_file.name)
                self.memory_monitor.get_memory_status("post_preprocessing")

            with self.display.section("Voice Activity Detection", 2, total_steps):
                 vad_processor = VADProcessor(self.config, self.display, self.memory_monitor, self.device, auth_token)
                 try:
                     vad_processor.load_model()
                     speech_regions = vad_processor.process(str(processed_audio_path))
                 finally:
                     vad_processor.unload_model()

            if not speech_regions:
                 self.display.add_warning("VAD found no speech. Stopping processing.")
                 return None

            with self.display.section("Speaker Diarisation", 3, total_steps):
                diar_processor = DiarisationProcessor(self.config, self.display, self.memory_monitor, self.device, auth_token)
                try:
                    diar_processor.load_model()
                    diarisation_result = diar_processor.process(str(processed_audio_path), min_speakers, max_speakers)
                finally:
                    diar_processor.unload_model()

            with self.display.section("Speech Transcription", 4, total_steps):
                trans_processor = TranscriptionProcessor(self.config, self.display, self.memory_monitor, self.device)
                try:
                    trans_processor.load_model()
                    transcription_segments = trans_processor.process(str(processed_audio_path), speech_regions)
                finally:
                    trans_processor.unload_model()

            with self.display.section("Attribution & Saving", 5, total_steps):
                 final_segments = self._perform_speaker_attribution(diarisation_result, transcription_segments)
                 output_filename = f"{input_file.stem}_{datetime.now().strftime(self.config.get('output', 'timestamp_format'))}.{output_format}"
                 output_path = self.output_dir / output_filename
                 TranscriptionWriter.save_transcript(final_segments, output_path, output_format, self.display)
                 return output_path

        except Exception as e:
            self.logger.error(f"Critical error during processing: {e}", exc_info=True)
            return None
        finally:
            if processed_audio_path and processed_audio_path.exists():
                processed_audio_path.unlink()
            history_path = self.config.get_path('logs_dir') / f"memory_usage_{input_file.stem}.json"
            self.memory_monitor.save_history(history_path)

    def _perform_speaker_attribution(self, diarisation: Annotation, transcription: List[Dict]) -> List[TranscriptionSegment]:
        """Assigns speakers to transcribed segments."""
        final_segments = []
        for segment_data in transcription:
            text = segment_data.get('text', '').strip()
            if not text: continue
            
            current_segment = Segment(segment_data['start'], segment_data['end'])
            segment_diarisation = diarisation.crop(current_segment, mode='intersection')
            speakers = {label for _, _, label in segment_diarisation.itertracks(yield_label=True)}
            
            speaker = speakers.pop() if len(speakers) == 1 else "Unknown Speaker"
            final_segments.append(TranscriptionSegment(speaker, current_segment.start, current_segment.end, text))
            
        return final_segments

    def run(self, args: argparse.Namespace) -> Optional[Path]:
        """Main execution flow."""
        self.logger.info("Starting Diarisation Pipeline Run...")
        input_file = Path(args.input) if args.input else self.get_input_source()
        if not input_file: return None

        auth_token = args.auth_token or self._read_auth_token()
        output_path = self.process_file(
            input_file, auth_token,
            format=args.format, min_speakers=args.min_speakers, max_speakers=args.max_speakers
        )
        
        return output_path

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    try:
        temp_config = ConfigManager()
        # CORRECTED: Use keyword arguments for 'default' to call config.get correctly.
        default_format = temp_config.get('output', 'default_format', default='txt')
        default_min_speakers = temp_config.get('processing', 'diarisation', 'min_speakers', default=1)
        default_max_speakers = temp_config.get('processing', 'diarisation', 'max_speakers', default=5)
    except Exception:
        default_format, default_min_speakers, default_max_speakers = 'txt', 1, 5
    
    parser = argparse.ArgumentParser(description='Audio Transcription and Diarisation Pipeline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', type=str, help='Path to an input audio/video file. Bypasses interactive selection.')
    parser.add_argument('--auth-token', type=str, help='Hugging Face authentication token.')
    parser.add_argument('--format', choices=['txt', 'json'], default=default_format, help='Output transcript format.')
    parser.add_argument('--min-speakers', type=int, default=default_min_speakers, help='Minimum number of speakers expected.')
    parser.add_argument('--max-speakers', type=int, default=default_max_speakers, help='Maximum number of speakers expected.')
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    start_time = time.time()
    manager = None
    try:
        args = parse_args()
        manager = TranscriptionManager()
        output_path = manager.run(args)

        if manager and output_path:
            with manager.display.section("Processing Complete", 1, 1):
                manager.display.print_metrics({
                    "Status": "✓ Success",
                    "Output File": output_path.name,
                    "Processing Time": manager.display._format_time(time.time() - start_time),
                })
        elif manager:
            with manager.display.section("Processing Finished", 1, 1):
                manager.display.print_metrics({
                    "Status": "✗ Failed or Aborted",
                    "Info": "Check logs for details.",
                })
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}", file=sys.stderr)
        logging.getLogger(__name__).error("Fatal error in main execution:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
