---
title: "Diarizer v2 implementation (WhisperX wrapper)"
type: bus-plan
status: superseded
assigned_to: "opus"
priority: high
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
linked_decisions:
  - "D1 pipeline shape: single thin wrapper around WhisperX, no processor abstraction"
  - "D2 model: large-v3-turbo, compute_type=float16, fallback int8 via config"
  - "D3 diarisation: WhisperX-bundled pyannote (no swap)"
  - "D4 config: dataclass + YAML, no singleton, single load_config(path) function"
  - "D5 preprocessing: optional FFmpeg cleaning, on by default"
  - "D6 output: TXT (default), JSON, SRT"
  - "D7 testing: three files (test_config, test_io, test_pipeline); drop numbered ordering"
  - "D8 VRAM management: drop load/unload pattern; single-shot pipeline"
  - "D9 speaker hints: carry over --min-speakers / --max-speakers verbatim"
  - "D10 HF auth: env HF_TOKEN > config.yaml auth.hf_token > --auth-token CLI flag"
  - "D11 offline only: yes (after first model download)"
  - "D12 CLI only: yes (no API/HTTP)"
  - "D13 anchor: WhisperX confirmed"
  - "D14 dev strategy: build alongside v1 in this session; v1 deletion in follow-up PLAN after hardware validation"
linked_inputs:
  - "202605040210_RESEARCH_v2-requirements.md"
parent_plan_of_plans: "202605040200_PLAN_diarizer-v2-rearchitecture.md"
---

## Status: SUPERSEDED 2026-05-04

This PLAN anchored on WhisperX without alternatives analysis. Operator paused execution after install hit Windows-CUDA dependency hell. SOTA research run produced [`Bus/202605040230_RESEARCH_v2-stack-and-pipeline.md`](202605040230_RESEARCH_v2-stack-and-pipeline.md) recommending **faster-whisper + pyannote 3.3.x** instead. The replacement PLAN to be drafted is `202605040240_PLAN_v2-rebuild.md`. Do not execute this file.

The v2 module shapes (`v2/config.py`, `v2/output.py`, `v2/cli.py`, dataclass Config, YAML loader, output writers, 24 of the 33 tests) are stack-agnostic and survive into the rebuild — only `v2/pipeline.py` and `v2/preprocessing.py` need rewriting.

---

## Objective (original — retained for record)

Implement Diarizer v2 as a thin WhisperX wrapper, replacing v1's five-processor architecture. Build alongside v1 in this session (no A/B at runtime — v2 isn't wired until the operator hardware-validates and a follow-up PLAN deletes v1). Operator-confirmed scope: cybersecurity meeting transcription, 8 GB VRAM, all current audio/video formats, offline, CLI only.

## Decisions (all locked, see frontmatter)

See `linked_decisions:`. Rationale per decision recorded inline below where non-obvious.

**D2 (Whisper variant) rationale.** `large-v3-turbo` is the best speed/accuracy/VRAM compromise on consumer GPUs (~6 GB VRAM in fp16). `large-v3` (~10 GB fp16) blows the budget without int8 quantisation. Operator can override via config to a smaller model if VRAM-tight on lower-end 8 GB cards.

**D5 (preprocessing) rationale.** Cybersecurity meeting audio is typically noisy (laptop mics, conference rooms, screen-share audio). FFmpeg-based normalisation cheap and helps Whisper. Off-switch in config for clean studio audio.

**D14 (dev strategy) rationale.** Operator constraint: "no A/B testing". v1 isn't being kept as a fallback — it's being kept until hardware-validation confirms v2 works on the operator's GPU. Once validated (next session), follow-up PLAN deletes v1 atomically. v2 module names use a `v2/` package or distinct names to avoid colliding with v1 imports during dev.

## Steps

### Step 1 — Add WhisperX to dependencies

Append `whisperx>=3.1.1` to `requirements.txt`. WhisperX pulls Whisper, faster-whisper (ctranslate2), pyannote.audio, and torch as transitive deps. Keep v1 deps untouched.

**verify:** `grep -q "^whisperx" requirements.txt`.

### Step 2 — Create v2 package skeleton

New directory `v2/` at project root containing:
- `v2/__init__.py`
- `v2/config.py`
- `v2/preprocessing.py`
- `v2/pipeline.py`
- `v2/output.py`
- `v2/cli.py`

`v2/__init__.py` exports `Pipeline`, `Config`, `load_config`. No code yet — empty stubs created in this step.

**verify:** all six files exist; `python -c "import v2"` succeeds.

### Step 3 — Implement v2/config.py

Dataclass `Config` with fields:
- `paths` — recordings_dir, output_dir, temp_dir, logs_dir.
- `model` — name (default `large-v3-turbo`), compute_type (default `float16`), device (default `cuda` with cpu fallback).
- `diarisation` — enabled (default true), min_speakers (default null), max_speakers (default null).
- `preprocessing` — enabled (default true), denoise (default true), normalise (default true).
- `output` — format (default `txt`), include_timestamps (default true).
- `auth` — hf_token (default null; resolved at runtime from env if null).

Function `load_config(path: Path | None = None) -> Config`. Reads YAML if provided; falls back to baked-in defaults. No singleton — each caller owns the instance.

Default `config/config.v2.yaml` written alongside (separate from v1's `config/config.yaml`).

**verify:** `python -c "from v2.config import load_config; c = load_config(); print(c.model.name)"` outputs `large-v3-turbo`.

### Step 4 — Implement v2/preprocessing.py

Function `preprocess_audio(input_path: Path, output_path: Path, config: PreprocessingConfig) -> Path`. Wraps FFmpeg:
- Extract audio if input is video (any format ffmpeg recognises → 16 kHz mono WAV).
- Optional `arnndn` or `afftdn` denoise filter if `config.denoise`.
- Optional `loudnorm` if `config.normalise`.
- Returns the processed path.

If preprocessing disabled, just decode to WAV and return.

**verify:** `python -c "from v2.preprocessing import preprocess_audio"` imports cleanly. (Real-audio test deferred to hardware session.)

### Step 5 — Implement v2/pipeline.py

Class `Pipeline`:
- `__init__(self, config: Config)` — stores config, lazy-imports whisperx.
- `transcribe(self, audio_path: Path) -> TranscriptionResult` — single method. Internally:
  1. Calls `preprocess_audio` if `config.preprocessing.enabled`.
  2. Loads WhisperX model via `whisperx.load_model(...)`.
  3. Runs transcription.
  4. Loads alignment model via `whisperx.load_align_model(...)`.
  5. Aligns segments.
  6. If `config.diarisation.enabled`: loads diarisation pipeline, assigns speakers via `whisperx.assign_word_speakers(...)`.
  7. Returns a `TranscriptionResult` dataclass (segments + metadata).

`TranscriptionResult` is a simple dataclass — segment list (start, end, text, speaker), source path, model name, duration.

**verify:** import-only; full execution requires hardware.

### Step 6 — Implement v2/output.py

Three formatters:
- `write_txt(result, path)` — plain text, one segment per line, optional `[speaker] HH:MM:SS` prefix.
- `write_json(result, path)` — JSON dump of dataclass.
- `write_srt(result, path)` — SRT subtitle format (numbered entries with `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing).

Single dispatch function `write_output(result, path, format: str)`.

**verify:** unit-testable on synthetic `TranscriptionResult` fixtures.

### Step 7 — Implement v2/cli.py

Argparse-based CLI:
- `--input PATH` — audio/video file (required if not interactive).
- `--config PATH` — config YAML override.
- `--format {txt,json,srt}` — output format override.
- `--min-speakers INT`, `--max-speakers INT` — diarisation hints.
- `--auth-token TOKEN` — HF token override.
- `--no-preprocess` — skip FFmpeg preprocessing.
- `--model NAME` — model override.

Entry function `main()`. Loads config, resolves overrides, instantiates Pipeline, runs transcription, writes output, prints summary.

**verify:** `python -m v2.cli --help` lists the flags.

### Step 8 — Tests

Create `tests/v2/` with three files:
- `test_config.py` — defaults; YAML override; partial override.
- `test_output.py` — format writers on synthetic `TranscriptionResult`; round-trip JSON.
- `test_pipeline_smoke.py` — heavy-mocked pipeline (mock `whisperx.load_model` etc.); verifies orchestration without real models.

Drop numbered-file ordering — pytest discovers in alphabetical order which is fine for v2's smaller surface.

**verify:** `pytest tests/v2/ -v` collects ≥3 tests. Real-model E2E test deferred to hardware session.

### Step 9 — README v2 section

Add a `## Diarizer v2 (in development)` section to `README.md` describing the v2 CLI and config. Note: v1 still the operative pipeline until hardware validation completes.

**verify:** README section present; references `v2/cli.py`.

### Step 10 — Commit + push

Single commit at end with all v2 work; or progressive commits per logical step group. Operator preference: progressive, so failure points are easy to bisect.

**verify:** `git log --oneline` shows v2 commits; `git push` succeeds.

## Verification (full PLAN)

- [ ] `whisperx` in `requirements.txt`.
- [ ] `v2/` package importable.
- [ ] `v2/config.py` loads defaults and YAML.
- [ ] `v2/preprocessing.py`, `v2/pipeline.py`, `v2/output.py`, `v2/cli.py` import cleanly.
- [ ] `python -m v2.cli --help` succeeds.
- [ ] `tests/v2/` collects ≥3 tests; mocked tests pass.
- [ ] README has v2 section.
- [ ] Commits pushed to `feature/speaker-separation`.

## Acceptance (spec)

A user with the v2 dependencies installed can run `python -m v2.cli --input <audio>` and get a transcript that meets v2 RESEARCH success criteria — once hardware-validated. All v1 code remains intact and runnable. Follow-up PLAN handles v1 deletion atomically after operator validates v2 on real audio.

## Out of scope

- Hardware validation (separate session, separate PLAN).
- v1 deletion (follow-up PLAN).
- Performance benchmarking, model comparison, runtime tuning.

## Executor Notes

**Executed:** 2026-05-04 (autonomous opus execution after operator delegation).
**Outcome:** done (code-complete; hardware validation deferred per D14).

**What was done:**

- Step 1 — `requirements.v2.txt` created with `whisperx>=3.1.1`, `PyYAML`, `pytest`, `pytest-mock`. Deviation from PLAN: existing `requirements.txt` is UTF-16-encoded; editing it in place would risk encoding mangling. Used a separate v2 requirements file instead. Aligns with D14 (build alongside) — v1's deps are unaffected.
- Step 2 — `v2/` package created with all six files plus `config/config.v2.yaml`.
- Step 3 — `v2/config.py`: `Config` dataclass with `paths`, `model`, `diarisation`, `preprocessing`, `output`, `auth` sections. `load_config(path)` reads YAML with section-by-section dataclass instantiation. Unknown fields raise TypeError. `resolved_hf_token()` checks explicit > `HF_TOKEN` > `HUGGING_FACE_HUB_TOKEN`.
- Step 4 — `v2/preprocessing.py`: `preprocess_audio()` decodes to 16 kHz mono WAV via FFmpeg. Optional `afftdn=nf=-25` denoise and `loudnorm=I=-16:TP=-1.5:LRA=11` normalisation. Raises `FFmpegError` on missing ffmpeg or failed exec.
- Step 5 — `v2/pipeline.py`: `Pipeline.transcribe(audio_path)` runs preprocess → load_model → transcribe → load_align_model → align → diarisation (if HF token resolved) → segment build. Returns `TranscriptionResult` (segments, source, model, language, duration). Lazy-imports `whisperx` and `torch` so unit tests can mock without installing them. CPU fallback when CUDA unavailable.
- Step 6 — `v2/output.py`: `write_txt`, `write_json`, `write_srt`, plus `write_output()` dispatcher. Speaker prefixes and timestamps are config-driven.
- Step 7 — `v2/cli.py`: argparse-based; `--input`, `--config`, `--output`, `--format`, `--model`, `--min-speakers`, `--max-speakers`, `--auth-token`, `--no-preprocess`, `--no-diarisation`, `--verbose`. Output path auto-derived from input stem when not specified.
- Step 8 — `tests/v2/test_config.py` (9 tests), `tests/v2/test_output.py` (9 tests), `tests/v2/test_pipeline_smoke.py` (6 tests). All 24 pass under the v1 venv (no whisperx install needed because the smoke tests inject a fake whisperx module via `monkeypatch.setitem(sys.modules, "whisperx", fake)`).
- Step 9 — `README.md` got a `## Diarizer v2 (in development)` section with v2 layout, setup, usage examples, and pointer to `Bus/202605040220_PLAN_v2-implementation.md`.
- Step 10 — Commit + push (this commit).

**Verification results:**
- `python -c "import v2"` succeeds.
- `python -c "from v2.config import load_config; c = load_config(); print(c.model.name)"` outputs `large-v3-turbo`.
- `pytest tests/v2/ -v` — **24/24 passing** in 0.59 s.
- `python -m v2.cli --help` lists all flags.

**Deferred to hardware-validation session (next session):**
- Real-audio E2E run on operator's GPU.
- VRAM measurement on `large-v3-turbo` + `compute_type=float16` + 8 GB ceiling.
- Diarisation quality check on a representative cybersecurity meeting.

**Blockers:** none.

**Files added:**
- `requirements.v2.txt`
- `v2/__init__.py`, `v2/config.py`, `v2/preprocessing.py`, `v2/pipeline.py`, `v2/output.py`, `v2/cli.py`
- `config/config.v2.yaml`
- `tests/v2/__init__.py`, `tests/v2/test_config.py`, `tests/v2/test_output.py`, `tests/v2/test_pipeline_smoke.py`

**Files modified:**
- `README.md` (v2 section added)
- `Bus/202605040220_PLAN_v2-implementation.md` (this file)

**Files NOT touched (per D14):**
- `requirements.txt` (v1 deps untouched)
- `processors/`, `audio/`, `utils/`, `config/config_manager.py`, `config/config.yaml`, `run_diariser.py` (v1 code intact)
- `tests/test_0..4_*.py` (v1 tests intact)

**Next step (separate PLAN, next session):** hardware-validate v2 on a real meeting recording. After validation: follow-up PLAN deletes v1, rewrites ARCHITECTURE.md as v2, opens `feature/speaker-separation → main` PR.
