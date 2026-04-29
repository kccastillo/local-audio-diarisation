# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based offline speaker diarisation and transcription pipeline. It combines OpenAI's Whisper for transcription with pyannote.audio for speaker diarisation to produce transcripts identifying "who said what". The pipeline runs entirely locally with a memory-efficient architecture that sequentially loads/unloads large AI models to work within typical GPU VRAM constraints (e.g., 8GB).

## Architecture & Core Design

The codebase follows a **processor-based pipeline pattern** with clear separation of concerns:

**Main Processing Pipeline** (`run_diariser.py`):
- `TranscriptionManager`: Orchestrates the entire workflow across 5 sequential stages:
  1. **Audio Preprocessing** (`audio/audio_cleaner.py`): Noise reduction and volume normalisation via FFmpeg
  2. **Voice Activity Detection** (VADProcessor): Identifies speech regions in audio
  3. **Speaker Diarisation** (DiarisationProcessor): Detects and labels different speakers
  4. **Speech Transcription** (TranscriptionProcessor): Converts speech to text using Whisper
  5. **Attribution & Output**: Assigns speakers to transcribed segments and saves results

**Processor Architecture** (`processors/`):
- All processors inherit from `BaseProcessor` (base_processor.py), which provides:
  - Common interface: `load_model()`, `unload_model()`, `process()`
  - Consistent logging and device management (CPU/GPU)
  - Memory tracking integration
- Individual processors: VADProcessor, DiarisationProcessor, TranscriptionProcessor
- Models are loaded/unloaded on-demand to conserve VRAM

**Configuration System** (`config/config_manager.py`):
- Singleton pattern ensures single config instance across app
- YAML-based configuration (config/config.yaml)
- Handles path resolution, directory creation, and Hugging Face authentication tokens

**Supporting Infrastructure**:
- `utils/datatypes.py`: Immutable `TranscriptionSegment` dataclass for transcription data
- `utils/memory_monitor.py`: Tracks GPU/CPU memory usage throughout pipeline
- `utils/display_manager.py`: Unified console output with progress sections and formatting
- `utils/transcription_writer.py`: Saves transcripts in TXT or JSON format

## Key Dependencies

- **pyannote.audio** (3.3.2): Speaker diarisation and VAD via Pyannote pipelines
- **openai-whisper** (20240930): Speech-to-text transcription
- **torch** (2.7.1+cu118): Deep learning backend with CUDA support
- **librosa**: Audio analysis and preprocessing
- **soundfile**: Audio I/O
- **pydub**: Audio format handling
- **pyyaml**: Configuration parsing
- **pytest** (8.4.1): Testing framework

## Commands

### Setup & Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Main Pipeline
```bash
# Interactive mode (select audio file from recordings_dir)
python run_diariser.py

# Direct file processing
python run_diariser.py --input path/to/audio.mp4

# With custom speaker count hints
python run_diariser.py --input path/to/audio.mp4 --min-speakers 2 --max-speakers 4

# Custom output format (txt or json)
python run_diariser.py --input path/to/audio.mp4 --format json

# With Hugging Face token (for model downloads)
python run_diariser.py --input path/to/audio.mp4 --auth-token your_hf_token
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_0_config.py -v

# Run single test
pytest tests/test_3_transcription_processor.py::TestTranscriptionProcessor::test_process_valid_audio -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

### Development & Debugging
```bash
# Run with debug logging
python run_diariser.py --input path/to/audio.mp4  # Logging level controlled in config.yaml

# Check GPU availability and VRAM
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Output
- Transcripts saved to `output/` directory (default)
- Logs saved to `logs/` directory with timestamp
- Memory usage history per file saved as JSON in `logs/`
- Temporary processed audio stored in `temp/` (cleaned up after processing)

## Configuration

The application reads `config/config.yaml` with sections for:
- **paths**: Base directory, recordings_dir, output_dir, temp_dir, logs_dir
- **auth**: Hugging Face token location
- **processing**: Whisper model size, Pyannote settings, speaker count hints
- **output**: Default transcript format, timestamp format
- **display**: Progress type (TQDM or other), verbosity
- **logging**: Log level, format, file prefix

Path values are resolved relative to the `base_dir` (typically project root).

## Testing Structure

- **Test Numbering**: Files are named `test_<number>_<module>.py` to indicate execution order (0-4). Tests in lower-numbered files should run before higher-numbered ones due to dependencies (e.g., config tests before processor tests).
- **Fixtures**: Use pytest fixtures for temporary directories, mock configs, and test audio files
- **Mocking**: pytest-mock used for isolating components; real audio/models avoided where possible in unit tests
- **End-to-End**: `test_4_end_to_end.py` tests the full pipeline (slowest, runs last)

## Important Implementation Notes

**VRAM Management**: The sequential load/unload pattern in `TranscriptionManager.process_file()` is critical. Each processor loads its model, processes, then immediately unloads and clears CUDA cache. Don't change this pattern without understanding the memory implications.

**Speaker Attribution**: The `_perform_speaker_attribution()` method in TranscriptionManager crops the diarisation result to each transcription segment's time window and selects the speaker with the highest overlap. Edge case: overlapping speakers default to "Unknown Speaker".

**Pyannote Models**: VAD and diarisation use Pyannote pipelines, which download from Hugging Face on first run. Requires `auth_token` for some models. Cached locally by default.

**Error Handling**: The pipeline continues through preprocessing/VAD failures but stops at critical errors. Check logs in `logs/` for detailed error context.

## Branch Information

- **Main branch**: `main`
- **Current feature branch**: `feature/speaker-separation` (working on enhanced speaker separation)

## Future Development Areas

- Streaming/incremental processing for very long audio files
- Multi-GPU support for parallel processing
- Custom speaker profiles/voice recognition
- Real-time transcription via live audio input
