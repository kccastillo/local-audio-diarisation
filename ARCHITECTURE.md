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

Offline speaker-diarisation and transcription pipeline. A single `Pipeline` class orchestrates [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper via CTranslate2) and [pyannote.audio 3.3](https://github.com/pyannote/pyannote-audio) end-to-end, with adaptive preprocessing and validation gates between stages.

The pipeline is designed for the operator's primary use case: cybersecurity meeting transcription on consumer-grade GPUs (8 GB VRAM). Recovery point: the previous five-stage processor architecture is preserved at `git checkout v1-final`.

---

## Pipeline shape

`diarizer/pipeline.py` exposes a single `Pipeline` class with one public method, `transcribe(audio_path) -> TranscriptionResult`. Internally the call walks six stages with validation gates between them:

```
ingest  →  adaptive preprocess  →  voice presence  →  diarise  →  ASR  →  attribute  →  output
   ↓             ↓                       ↓                ↓         ↓        ↓
 Gate 0       Gate 1                  Gate 2           Gate 3    Gate 4  Gate 5
 BLOCK        BLOCK                   BLOCK            WARN      WARN    WARN
```

Gates 0–2 raise `PipelineGateFailure` on failure (cheap pre-checks block the run before GPU time is spent on bad input). Gates 3–5 warn and continue (transcript is still useful with messy speaker labels).

---

## Stages

### 1 — Ingest (measurement)

`diarizer.preprocessing.measure(path)` returns an `AudioMeasurement` dataclass capturing duration, sample rate, channels, RMS dBFS, peak dBFS, noise floor, speech floor, voiced fraction, and spectral centroid. Cost is roughly hundreds of milliseconds per minute of audio. Cheap relative to the rest of the pipeline; we run it once at ingest and reuse the result everywhere.

### 2 — Adaptive preprocessing

`diarizer.preprocessing.preprocess()` shells out to FFmpeg with filter selection driven by the measurement:

- **Always**: decode + resample to 16 kHz mono PCM (Whisper's preferred input shape).
- **Loudness normalisation** (`loudnorm=I=-23:LRA=7:TP=-2`): only when measured RMS is below −32 dBFS. Prevents over-cooking already-loud inputs.
- **Denoise** (`afftdn=nf=-25`): default OFF. Even when explicitly enabled, only applies if noise floor > −40 dBFS *and* SNR < 20 dB. Learned and FFT-based denoisers degrade Whisper WER on already-clean audio.
- **Spectral-centroid sanity warning**: if input centroid > 6 kHz, log a warning that the audio has likely been through aggressive AI denoise upstream (e.g. NVIDIA Broadcast, Krisp). Whisper will still cope but quality may suffer.

Output is a 16 kHz mono WAV in the configured `temp_dir`.

### 3 — Voice presence

A coarse measurement-based check (re-uses the voiced-fraction value from measurement). Below 5 % voiced, the run blocks at Gate 2 — refusing to spend ASR time on near-silence. Below 30 %, it warns ("sparse audio detected") but continues.

### 4 — Diarisation

`pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")` on CUDA. Optional `min_speakers` / `max_speakers` hints from config. Produces a speaker timeline `[(start, end, label), ...]`. Pipeline is unloaded and CUDA cache cleared before the ASR stage starts — sequential model residency keeps peak VRAM bounded by the larger of the two models.

Diarisation is skipped (graceful degradation) if no Hugging Face token is resolved.

### 5 — ASR (faster-whisper)

`WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")`. Run with:

- `beam_size=1` — lower hallucination per arXiv:2501.11378.
- `vad_filter=True` — built-in Silero VAD, no separate stage.
- `word_timestamps=True` — word-level alignment via Whisper's own internal timing.
- `condition_on_previous_text=False` — breaks hallucination chains across long files.

OOM retry: if first load fails with an out-of-memory error, retry with `compute_type="int8"`. If still OOMs, raise with a "lower the model size" message.

Model is unloaded and cache cleared after transcription.

### 6 — Attribution

For each ASR segment, find the diariser timeline turn with maximum overlap inside the segment's window. Tie or no-overlap → `"Unknown Speaker"`. Pure function in `Pipeline._attribute()`.

### 7 — Output

`diarizer.output.write_output()` dispatches to TXT, JSON, or SRT writers. All formats are driven from the `TranscriptionResult` dataclass (segments + speaker labels + timestamps + measurement metadata).

---

## Validation gates

`diarizer/gates.py` defines six pure-function gates:

| Gate | Check | Severity |
|---|---|---|
| 0 — `ingest_gate` | duration ≥ 10 s, sane SR, channels in [1,8], RMS above floor | BLOCK |
| 1 — `preprocess_damage_gate` | post-preprocess RMS within 6 dB of pre, centroid not eaten by denoise | BLOCK |
| 2 — `voice_presence_gate` | voiced fraction ≥ 5 % (block) or ≥ 30 % (warn-only) | BLOCK |
| 3 — `diarisation_sanity_gate` | speaker count within hints, no monopoly speaker (>98 %), no spurious clusters (<0.5 %) | WARN |
| 4 — `asr_sanity_gate` | non-empty output; ≥ 10 chars per voiced minute | WARN |
| 5 — `output_sanity_gate` | output file written, non-empty, segment count matches | WARN |

Each gate returns a `GateResult` carrying name / passed / severity / message / metrics. Gates never raise — the orchestrator (`Pipeline.transcribe()`) decides via `is_blocking_failure()`.

---

## Module layout

```
diarizer/
├── __init__.py          — exports Config, load_config, Pipeline, TranscriptionResult
├── config.py            — dataclass Config + load_config(path); no singleton
├── preprocessing.py     — measure() + preprocess()
├── pipeline.py          — Pipeline class (six-stage flow)
├── gates.py             — six pure-function gates
├── output.py            — TXT / JSON / SRT writers
└── cli.py               — argparse entrypoint, `python -m diarizer.cli`

config/
└── config.yaml          — example config

tests/diarizer/          — 79 unit tests (config, gates, measurement, output, pipeline-mocked, preprocessing-mocked)
scripts/
├── inspect_audio.py     — audio characterisation helper
├── extract_docx.py      — Teams transcript extractor (used during quality benchmarking)
└── compare_transcripts.py — named-entity cross-check
```

---

## Configuration

`config/config.yaml` is the example. All sections are optional; absent sections fall back to the dataclass defaults baked into `Config`. Sections:

- **paths** — recordings_dir, output_dir, temp_dir, logs_dir.
- **model** — name (default `large-v3-turbo`), compute_type (default `float16`), device (default `cuda`), cpu_fallback, language.
- **diarisation** — enabled, min_speakers, max_speakers.
- **preprocessing** — enabled, normalise (default true), denoise (default false), target_sample_rate.
- **output** — format, include_timestamps, include_speakers.
- **auth** — hf_token (inline) or token_path (file path). Resolution order at runtime: CLI flag → inline → token_path → `HF_TOKEN` env → `HUGGING_FACE_HUB_TOKEN` env.

`load_config(path)` returns a fresh `Config` instance per call — no module-level singleton.

---

## Memory and VRAM

The pipeline keeps GPU residency to **one model at a time**:

1. Pyannote diariser loads → diarises → unloads → `torch.cuda.empty_cache()`.
2. faster-whisper loads → transcribes → unloads → cache clear.

Measured peak on the operator's RTX 3070 (8 GB): ~5 GB residual mid-run, comfortable headroom. `large-v3-turbo` at `compute_type=float16` sits ~6 GB; at `int8_float16` sits ~3 GB. The OOM retry path falls back to `int8` if the chosen compute_type doesn't fit.

---

## Output formats

- **TXT** (default): `[SPEAKER_XX] HH:MM:SS Text segment.` per line. Speaker prefix and timestamp prefix are config-driven.
- **JSON**: full `TranscriptionResult` dump including segments, words (when `word_timestamps=True`), speaker labels, gate-result trail, and the input measurement.
- **SRT**: standard subtitle format with `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing.

The writers are model-agnostic — any future ASR/diariser swap that returns a compatible `TranscriptionResult` will continue to work.

---

## Testing

`pytest` (configured via `pyproject.toml`) discovers `tests/diarizer/` automatically. 79 tests covering:

- `test_config.py` (9) — defaults, YAML override, partial override, env-token resolution.
- `test_gates.py` (28) — pure-function tests per gate (pass / fail / edge cases).
- `test_measurement.py` (8) — synthetic WAV fixtures (silence, pure tone, stereo, sparse pulses).
- `test_output.py` (9) — TXT / JSON / SRT writers on synthetic results.
- `test_pipeline.py` (16) — orchestration, with `faster_whisper` and `pyannote.audio` mocked via `sys.modules` injection. Covers full pipeline, no-diarisation, no-preprocess, missing-token, blocking gate failures, OOM retry, attribution math.
- `test_preprocessing.py` (9) — FFmpeg shell tests with `subprocess.run` mocked.

Real-model end-to-end runs require GPU + HF token and are exercised manually (smoke evidence: `Retired/202605040250_RESEARCH_v2-smoke-run.md`).

---

## Transcript-review webapp (post-pipeline)

`diarizer serve <session-dir>` boots a local FastAPI app on `127.0.0.1:8765` (loopback only) for reviewing and editing finished transcripts. Each pipeline run produces a session directory under `output/<stem>_<timestamp>/` containing:

- `source.opus` — 16 kHz mono Opus/OGG (~16 kbps VBR), the **playback artefact**. Generated from the source file alongside the existing WAV. Browser-native, small.
- `source.wav` — 16 kHz mono PCM WAV, the **model-canonical input** (unchanged from `preprocessing.py`). The webapp does not read this file; it exists for reproducibility and re-runs.
- `transcript.json` — immutable copy of the original pipeline output. The webapp never overwrites this.
- `transcript_edit_YYYYMMDD_NN.json` — versioned edit sidecars (created on every Save in the webapp). `NN` is a per-day counter `00`–`99` written via O_EXCL atomic create with retry; on the 100th same-day save the counter wraps to `00` and overwrites (operator-locked behaviour, surfaced via a `wrapped: true` flag in the save response).
- `session.json` — manifest pointing to the artefacts above.

Edit scope in v1: speaker labels (global rename + per-segment reassign) and transcript text. Segment boundaries and audio zoom are deferred. Undo/redo within a session is deferred — versioned saves provide coarse undo via the Versions dropdown.

## Recovery

The pre-cutover v1 architecture (five-processor pipeline with sequential load/unload, singleton ConfigManager, separate VAD/diarisation/transcription stages) is preserved at `git checkout v1-final`. Reference if anything needs to be revived.
