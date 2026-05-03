# Diarizer (local-audio-diarisation)

Offline speaker diarisation and transcription pipeline written in Python. Combines [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarisation with [OpenAI Whisper](https://github.com/openai/whisper) for transcription to produce transcripts that identify "who said what".

The architecture is memory-efficient: AI models are loaded and unloaded sequentially so that large models (e.g. `whisper-large-v3`) can run on consumer-grade GPUs with as little as 8 GB of VRAM.

For design details see [ARCHITECTURE.md](ARCHITECTURE.md).

## Key features

- **High-accuracy transcription** — Whisper models (sizes from `tiny` to `large-v3`) for state-of-the-art speech-to-text.
- **Speaker diarisation** — pyannote.audio with optional hints for the number of speakers.
- **Offline and private** — runs entirely locally; no audio uploaded to the cloud.
- **VRAM-efficient** — sequential load/unload makes large models viable on consumer GPUs (≥ 8 GB VRAM).
- **Modular** — dedicated processors per stage (preprocessing, VAD, diarisation, transcription) with a shared base interface.
- **Audio preprocessing** — noise reduction and volume normalisation via FFmpeg before model inference.
- **Flexible input** — common audio and video formats (MP4, M4A, WAV, MP3) handled through FFmpeg.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

FFmpeg must be available on `PATH`. Some pyannote models require a Hugging Face access token (see *Configuration* below).

## Run the pipeline

```bash
# Interactive: select an audio file from recordings_dir
python run_diariser.py

# Direct file processing
python run_diariser.py --input path/to/audio.mp4

# Speaker-count hints
python run_diariser.py --input path/to/audio.mp4 --min-speakers 2 --max-speakers 4

# Output format (txt or json; default txt)
python run_diariser.py --input path/to/audio.mp4 --format json

# Hugging Face token (overrides config.yaml)
python run_diariser.py --input path/to/audio.mp4 --auth-token <hf_token>
```

## Outputs

- **Transcripts** → `output/` (configurable). Format: TXT or JSON.
- **Logs** → `logs/<timestamp>.log` per run.
- **Memory history** → `logs/<timestamp>_memory.json` per run (GPU/CPU usage timeline).
- **Temporary processed audio** → `temp/` (cleaned up after a successful run).

## Configuration

Configuration lives in `config/config.yaml`. Sections:

- **paths** — `base_dir`, `recordings_dir`, `output_dir`, `temp_dir`, `logs_dir`. All paths resolve relative to `base_dir` (typically the project root).
- **auth** — Hugging Face token location.
- **processing** — Whisper model size, pyannote settings, default speaker-count hints.
- **output** — default transcript format, timestamp format.
- **display** — progress bar type (TQDM or other), verbosity.
- **logging** — log level, format, file prefix.

The config is loaded once via a singleton `ConfigManager`; all modules share that instance.

## Testing

```bash
# All tests
pytest tests/ -v

# A specific file
pytest tests/test_0_config.py -v

# A single test
pytest tests/test_3_transcription_processor.py::TestTranscriptionProcessor::test_process_valid_audio -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

Test files are numbered (`test_<n>_<module>.py`) to encode dependency order — `test_0_config.py` runs before processor tests; `test_4_end_to_end.py` runs last because it exercises the full pipeline. Fixtures cover temp directories, mock configs, and test audio files; `pytest-mock` is used to isolate components from real models where possible.

## Project status

A v2 rewrite is planned as a clean replacement of v1 (no A/B coexistence). The v1 codebase will be tagged `v1-final` before the rewrite begins. See `Bus/202605040200_PLAN_diarizer-v2-rearchitecture.md` for the rewrite plan-of-plans.
