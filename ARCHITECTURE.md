---
title: Diarizer architecture
type: design-reference
last_updated: 2026-05-04
related:
  - README.md
  - CLAUDE.md
  - config/config.yaml
---

# Diarizer architecture

Offline speaker-diarisation and transcription pipeline. Combines OpenAI Whisper (transcription) with pyannote.audio (voice activity detection + speaker diarisation) to produce "who said what" transcripts. Runs entirely locally; designed to fit typical consumer GPUs (e.g. 8 GB VRAM) by loading and unloading large models sequentially.

A v2 rewrite is planned. This document describes the v1 baseline. Tag `v1-final` (or equivalent) before rewriting.

---

## Pipeline overview

`run_diariser.py` exposes the entry point. `TranscriptionManager` orchestrates the workflow across five sequential stages:

1. **Audio preprocessing** (`audio/audio_cleaner.py`) — noise reduction and volume normalisation via FFmpeg.
2. **Voice activity detection** (`VADProcessor`) — identifies speech regions in the audio.
3. **Speaker diarisation** (`DiarisationProcessor`) — detects and labels distinct speakers.
4. **Speech transcription** (`TranscriptionProcessor`) — converts speech to text using Whisper.
5. **Attribution & output** — assigns speakers to transcribed segments and writes results.

Each stage produces output consumed by the next; failure modes are stage-local (see *Error handling* below).

---

## Processor architecture (`processors/`)

All processors inherit from `BaseProcessor` (`base_processor.py`), which provides a common interface:

- `load_model()` — bring the model into VRAM.
- `unload_model()` — release VRAM and clear CUDA cache.
- `process()` — run the stage's transformation.

Concrete processors:

- `VADProcessor` — pyannote VAD pipeline.
- `DiarisationProcessor` — pyannote speaker diarisation pipeline.
- `TranscriptionProcessor` — Whisper.

`BaseProcessor` also handles consistent logging, device selection (CPU vs CUDA), and integration with the memory monitor.

**VRAM management is the load-bearing design choice.** Each processor loads its model, runs `process()`, then immediately unloads and clears the CUDA cache before the next processor starts. This is what allows the full pipeline to run on an 8 GB GPU. Do not change this pattern without understanding the memory implications — concurrent model residency will OOM on consumer hardware.

---

## Configuration (`config/config_manager.py`)

`ConfigManager` is a singleton — a single config instance is shared across the application. Reads `config/config.yaml`. Responsibilities:

- Path resolution (all paths relative to `base_dir`, typically project root).
- Directory creation for `recordings_dir`, `output_dir`, `temp_dir`, `logs_dir`.
- Hugging Face authentication token discovery (required for some pyannote model downloads).

Config sections: `paths`, `auth`, `processing` (Whisper model size, pyannote settings, speaker-count hints), `output` (transcript format, timestamp format), `display`, `logging`.

---

## Supporting infrastructure (`utils/`)

- `datatypes.py` — immutable `TranscriptionSegment` dataclass (start, end, text, speaker).
- `memory_monitor.py` — tracks GPU and CPU memory usage throughout the pipeline; emits a per-file JSON history to `logs/`.
- `display_manager.py` — unified console output: progress sections, formatting, verbosity.
- `transcription_writer.py` — saves transcripts as TXT or JSON.

---

## Key dependencies

- **pyannote.audio** (3.3.2) — speaker diarisation and VAD pipelines.
- **openai-whisper** (20240930) — speech-to-text transcription.
- **torch** (2.7.1+cu118) — deep-learning backend with CUDA support.
- **librosa**, **soundfile**, **pydub** — audio I/O and analysis.
- **pyyaml** — configuration parsing.
- **pytest** (8.4.1) — testing.

Full pinned set in `requirements.txt`.

---

## Speaker attribution

`TranscriptionManager._perform_speaker_attribution()` is where Whisper's transcription segments meet pyannote's diarisation timeline:

- For each transcription segment, the diarisation result is cropped to that segment's time window.
- The speaker label with the highest overlap inside the window wins.
- Edge case: when speakers overlap with comparable weight, the segment is attributed to `"Unknown Speaker"` rather than guessing.

This is a deliberately simple heuristic. The v2 rewrite is the place to revisit it (e.g. forced alignment, softer overlap-weighting, or speaker-embedding consistency across segments).

---

## Pyannote model handling

VAD and diarisation models are pulled from Hugging Face on first run and cached locally. Some models require an `auth_token` (provided via CLI flag, env var, or `config.yaml`). The token is sourced through `ConfigManager`.

---

## Error handling

The pipeline distinguishes **continuable** from **fatal** failures:

- Preprocessing failures and VAD anomalies — the pipeline continues with a warning, falling back to the next-best input.
- Critical errors (e.g. Whisper OOM, missing auth token, corrupt audio) — the pipeline halts and surfaces the error.

Detailed error context lands in `logs/<timestamp>.log`. Memory history is written alongside as JSON.

---

## Outputs

- **Transcripts** → `output/` (default; configurable via `config.yaml`). Format: TXT or JSON.
- **Logs** → `logs/<timestamp>.log`.
- **Memory history** → `logs/<timestamp>_memory.json` (per file).
- **Temporary processed audio** → `temp/` (cleaned up after a successful run).

---

## Testing structure

Tests live in `tests/` and are numbered to encode dependency order:

- `test_0_config.py` — config / singleton behaviour. Runs first.
- `test_1_*` to `test_3_*` — per-processor unit tests.
- `test_4_end_to_end.py` — full-pipeline integration. Slowest; runs last.

Conventions: pytest fixtures for temp dirs, mock configs, and test audio files. `pytest-mock` for component isolation. Real audio and real models are avoided in unit tests where possible — only `test_4_end_to_end.py` exercises the genuine pipeline.

---

## v2 rewrite

A clean replacement of the v1 architecture is planned (no A/B coexistence). Constraints likely to carry over:

- Sequential model load/unload to fit consumer GPU VRAM.
- Pluggable processor interface (BaseProcessor or successor).
- Singleton config with YAML source.
- Speaker-attribution heuristic revisited rather than reused as-is.

Tag the v1 baseline (`v1-final` or equivalent) before the rewrite begins so it remains recoverable.
