# Diarizer (local-audio-diarisation)

Offline speaker diarisation and transcription pipeline written in Python. Combines [pyannote.audio 3.3](https://github.com/pyannote/pyannote-audio) for speaker diarisation with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper via CTranslate2) for transcription to produce transcripts that identify "who said what".

The architecture is memory-efficient: AI models are loaded and unloaded sequentially so that `whisper-large-v3-turbo` and pyannote 3.1 both run on consumer-grade GPUs with as little as 8 GB of VRAM.

For design details see [ARCHITECTURE.md](ARCHITECTURE.md).

## Key features

- **Adaptive preprocessing** — measures the input (RMS, noise floor, voiced fraction, spectral centroid) and applies only the FFmpeg filters the audio actually needs. Loudnorm only when too quiet; denoise default OFF.
- **Validation gates between stages** — six gates fail fast on cheap pre-checks (silent file, decode failure, sparse audio) before committing to the expensive ASR stage.
- **High-fidelity transcription** — `whisper-large-v3-turbo` with `compute_type=int8_float16` for ~3 GB VRAM at full accuracy.
- **Speaker diarisation** — pyannote.audio 3.1 with optional `--min-speakers` / `--max-speakers` hints.
- **Offline and private** — runs entirely locally; first run downloads model weights, subsequent runs need no network.
- **VRAM-efficient** — sequential load/unload bounds peak VRAM by the larger of the two models. OOM retry falls back to `int8` automatically.
- **Flexible input** — common audio and video formats (MP4, M4A, WAV, MP3) handled through FFmpeg.
- **Multiple output formats** — TXT (default), JSON (full structured dump), SRT (subtitle).

## Setup (Windows + CUDA)

PyPI's default `torch` wheel for Windows is CPU-only. Install torch from PyTorch's CUDA index first, then the rest from `requirements.txt`:

```bash
python -m venv venv
venv\Scripts\activate

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121
pip install -r requirements.txt
```

FFmpeg must be available on `PATH`. The pyannote diarisation models require a Hugging Face access token (see *Configuration* below).

## Run the pipeline

```bash
# Default config + auto-detected output path
python -m diarizer.cli --input "recordings/meeting.m4a"

# Override format and speaker hints
python -m diarizer.cli --input meeting.m4a --format srt --min-speakers 2 --max-speakers 6

# Skip preprocessing (e.g. for already-prepped Teams audio at 16k mono)
python -m diarizer.cli --input teams_recording.mp4 --no-preprocess

# Override Hugging Face token explicitly (otherwise resolved from config.auth.token_path
# or HF_TOKEN env var — see config/config.yaml for resolution order)
python -m diarizer.cli --input meeting.m4a --auth-token <hf_token>

# Transcription only (no diarisation)
python -m diarizer.cli --input speech.wav --no-diarisation

# Pick a different Whisper model
python -m diarizer.cli --input meeting.m4a --model medium
```

## Outputs

- **Transcripts** → `output/` (configurable). Format: TXT, JSON, or SRT.
- **Logs** → stderr (capture with shell redirect; configurable in future).
- **Temporary processed audio** → `temp/` (cleaned up by ffmpeg's `-y`-overwrite, but tempfiles retain).

## Configuration

Configuration lives in `config/config.yaml`. Every section is optional; absent sections use the dataclass defaults. Sections:

- **paths** — `recordings_dir`, `output_dir`, `temp_dir`, `logs_dir`.
- **model** — Whisper model name, compute_type, device, cpu_fallback, language.
- **diarisation** — `enabled`, `min_speakers`, `max_speakers`.
- **preprocessing** — `enabled`, `normalise`, `denoise`, `target_sample_rate`. Denoise default OFF.
- **output** — `format`, `include_timestamps`, `include_speakers`.
- **auth** — `hf_token` (inline) or `token_path` (file path). Resolution order: CLI flag → inline → token_path → `HF_TOKEN` env → `HUGGING_FACE_HUB_TOKEN` env.

Configuration is loaded per-call via `diarizer.config.load_config(path)` — no module-level singleton.

## Testing

```bash
pytest                       # discovers tests/diarizer/ via pyproject.toml
pytest tests/diarizer/test_gates.py -v
```

79 tests cover config, gates, measurement, output, pipeline (mocked), and preprocessing (mocked ffmpeg). Real-model end-to-end runs are exercised manually on GPU; see `ARCHITECTURE.md`.

## Project status

v2.0 (faster-whisper + pyannote.audio 3.3) is the canonical pipeline. The previous five-stage processor architecture was retired in PLAN 300 and is recoverable at `git checkout v1-final`.
