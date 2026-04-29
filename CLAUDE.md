# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based offline speaker diarisation and transcription pipeline. It combines OpenAI's Whisper for transcription with pyannote.audio for speaker diarisation to produce transcripts identifying "who said what". The pipeline runs entirely locally with a memory-efficient architecture that sequentially loads/unloads large AI models to work within typical GPU VRAM constraints (e.g., 8GB).

A v2 rewrite is being planned — the current code is the baseline. Tag `v1-final` (or equivalent) before rewriting.

## Working style

- AU spelling, usage, date formats.
- New conversation: check the request is well-specified; ask clarifying questions first when it isn't.
- No unprompted output of artefacts, illustrations, code, or longform sections. If producing one seems like the best move, ask permission first.
- Long output: ask whether the direction is right before continuing.
- No cross-chat references between projects unless prompted; isolate within projects by default.
- To Ken: plain language; do not compress, abbreviate, or elide. Ruthless token economy is for internal planning only. Always name the thing, the operation, the result.
- When offering options: describe each in full, then say which you lean toward and explain why.
- If a question is answerable from a tool's default behaviour, decide and proceed.
- If something depends on a prior decision or external input, state the dependency — Ken will not always remember.
- Requirement before solution — no mechanism design until requirement and process are agreed.
- Reviews: brief verification preamble (what was checked against — files, not just the document under review) → one-line overall verdict → priority-ordered numbered punch list (blockers first, nits last) → "Not blockers" subgroup → net verdict (what's ready, what needs fixing).

## Agent execution rules

**Plan execution protocol & model switch rules:** See [AGENT_RULES.md](AGENT_RULES.md) for plan phases, model responsibilities, and switch protocol.

**Operating rules:**
1. **All plans go to Bus/** — Every piece of planned work lives as a PLAN file, never chat-only.
2. **Respect model boundaries** — Sonnet thinks; Haiku writes; Opus reviews. Never cross lanes without Ken's model switch.
3. **Research and advice to Bus/** — RESEARCH (Haiku data) and ADVICE (Opus strategic notes) go via `write-bus-input` skill. Writing an input auto-clears blocked plans waiting on it.

## Skills

Skills live in `.claude/skills/<name>/SKILL.md`. Invoked via `Skill("<name>")`. See [.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md](.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md) for structure and conventions.

| Skill | What it does |
|---|---|
| `create-agent-skills` | Expert guidance for creating and refining skills — structure, principles, workflows, templates |
| `write-bus-plan` | Transcribe plans to `Bus/` files; manage monthly LOG and status tables |
| `write-bus-input` | Write RESEARCH/ADVICE files to `Bus/`; unblock plans waiting on input |
| `execute-plan` | Execute PLAN steps in order; populate Executor Notes; update LOG; commit + push |
| `retire` | Move files to gitignored `Retired/` folder when no longer needed |

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
- **librosa**, **soundfile**, **pydub**: Audio I/O and analysis
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
