---
title: "v2 requirements — Diarizer re-architecture"
type: bus-research
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
source: "operator (Ken), 2026-05-04 conversation"
feeds_plan: "202605040200_PLAN_diarizer-v2-rearchitecture.md"
---

## Goals

The single most-important shift driving v2: **simplify**. Operator (Ken) judges v1 as overly complex. The technology has moved on — single-model pipelines (e.g. [WhisperX](https://github.com/m-bain/whisperX)) now bundle transcription + diarisation + alignment in one project, removing the need for the v1 five-stage processor architecture (audio_cleaner → VAD → diarisation → transcription → attribution).

Concrete v2 success criteria:
- **Runs on 8 GB VRAM** consumer GPU.
- **High-fidelity transcription** of meeting audio. Primary use case is **cybersecurity meeting transcription** — the consumer of the transcripts is Ken doing security work, so accuracy on technical jargon, speaker turns, and timestamping matters more than e.g. broadcast-quality post-production polish.
- **Architectural simplicity** — fewer moving parts than v1. Replace the bespoke five-processor pipeline with a single integrated tool (likely WhisperX) wrapped with the minimum project-specific glue (config, I/O, format conversion).

## Hard constraints

- **VRAM:** 8 GB maximum. v2 must fit within consumer-GPU memory.
- **Input formats:** must support every audio and video format v1 currently accepts (MP4, M4A, WAV, MP3, plus other formats v1 handles via FFmpeg). Video → audio conversion must continue to work.
- **Offline:** carried over from v1 — no cloud round-trip. (Operator did not re-state, but it is implied by "stay within the same scope as the current version" and is a defining v1 property; flagged for explicit confirmation in Step 3.)

## Soft preferences

None given. The operator explicitly declined to express model-line, library, or deployment preferences. **Implication for design:** Step 3 chooses based on technical merit and constraint-fit, not operator taste.

## Out of scope

**Stay within the same scope as the current version.** v2 does the same job as v1, better and simpler. Do not add:
- Real-time streaming.
- Multi-language UI.
- A web interface or HTTP API (CLI-only carries over).
- Model fine-tuning or training surfaces.
- Cloud or remote-model integration.

If a feature is not in v1, it is not in v2 unless Step 3 surfaces a specific reason.

## Anchor candidate: WhisperX

Operator named WhisperX as a representative example of "the technology has moved on". WhisperX is an open-source project that wraps:
- Whisper (transcription),
- wav2vec2-based forced phoneme alignment (sub-word timestamps),
- pyannote.audio (diarisation, same library v1 uses),

into a single Python package and CLI. It is **not** the only option (alternatives include NVIDIA NeMo's speaker-aware ASR, vanilla Whisper + lighter glue, and stable-ts), but it is the most direct fit for the operator's framing and is the strongest Step 3 candidate barring a specific disqualifier.

**Step 3 will confirm or replace this anchor based on:**
- 8 GB VRAM compatibility (WhisperX with `large-v2` typically needs ~10-12 GB unless `compute_type=int8` or `medium` is used; this needs verification, not assumption).
- Diarisation quality on multi-speaker meeting audio at the constraint VRAM.
- Maintenance status (last release date, open issue volume).
- License compatibility (WhisperX is BSD-2-Clause; pyannote.audio gating still applies for the underlying diarisation models).

If WhisperX disqualifies on any of these, Step 3 picks the next-best single-model candidate.

## Open questions for Step 3

These are not requirements; they are design questions Step 3 must answer before child PLANs can be drafted:

1. **Pipeline shape.** WhisperX wraps a five-stage equivalent into one package call. Confirm: is "one package call" the architecture, or do we still want a thin processor abstraction wrapping it?
2. **Whisper variant.** `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3` / `large-v3-turbo` / distil-whisper. 8 GB ceiling probably forces `medium` or `large-v3-turbo` with int8 quantisation.
3. **Diarisation backend.** Stay on pyannote (WhisperX default), switch to a lighter alternative, or accept a reduced-quality embedded diariser?
4. **Configuration shape.** Replace v1's singleton ConfigManager + YAML — keep YAML, switch to dataclass-based config, or push to env vars + CLI flags only?
5. **Audio/video preprocessing.** v1 does FFmpeg-based noise reduction and volume normalisation before models see the audio. WhisperX expects clean PCM input — keep the FFmpeg preprocessing step, drop it, or make it optional?
6. **Output format.** v1 supports TXT and JSON. Keep both? WhisperX emits its own JSON/SRT/VTT shapes — adopt those, or wrap into v1-compatible output?
7. **Testing.** v1 has numbered-file ordering with end-to-end test at the bottom. v2 with one model boundary needs a much smaller suite — design from scratch.
8. **VRAM management.** v1's load/unload-per-stage pattern is unnecessary if a single model owns the GPU end-to-end. Drop the pattern entirely?
9. **Speaker count hints.** v1 exposes `--min-speakers` / `--max-speakers`. WhisperX supports the same. Carry over verbatim?
10. **Hugging Face auth token handling.** Pyannote diarisation models still require gated HF access. Keep v1's config-based token handling, or simplify?

## Confirm before Step 3 closes

These should be re-checked with the operator at the top of Step 3:

- **Offline only?** (implied; not re-stated). Yes/no.
- **CLI only?** (implied; not re-stated). Yes/no.
- **Confirm WhisperX as anchor**, or open up the candidate list?

## References

- WhisperX: https://github.com/m-bain/whisperX
- v1 architecture baseline: [`ARCHITECTURE.md`](../ARCHITECTURE.md)
- v1 codebase recovery point: tag `v1-final` (commit a57f077)
