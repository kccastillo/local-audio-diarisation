---
title: "Diarizer Roadmap"
type: roadmap
scope: speaker diarisation + transcription pipeline — v1 → v2 rewrite and beyond
created: 2026-04-29
last_updated: 2026-04-29
status: drafting
---

# Diarizer Roadmap

## Mission

Efficient, accurate, fully offline speaker diarisation and transcription on consumer GPUs (8GB VRAM target). Clean orchestration that can extend to streaming / real-time inputs without rewriting the core. Word-level speaker attribution as the accuracy bar.

v1 is the working baseline. v2 is the in-flight rewrite — leaner architecture, faster backends, better attribution. Tag `v1-final` before v2 lands so v1 stays recoverable via git.

## Open strategic questions

Calibration items not yet resolved. Each shapes downstream thread decisions; answers belong in this file once settled.

- **Q1 — WhisperX wholesale, or piecewise improvements?** WhisperX bundles VAD + faster-whisper + forced alignment + Pyannote diarisation with word-level speaker labels. Adopting it collapses T01, T02, T03 into T04. Skipping it keeps the orchestrator under our control but means doing the integration work ourselves. Decision blocks committing to T01 vs T04.

## Threads

Threads keep stable IDs (T01..). Closed threads stay in this list with `Status: closed` for historical context — grep `Status:.*closed` for the inventory.

**T01 — Swap Whisper backend to faster-whisper**
- Status: identified
- Why: ~50% VRAM reduction and ~3-4× speed for the same model size vs `openai-whisper`. On an 8GB card, this is the difference between `medium` being comfortable and `large-v3` being feasible with headroom.
- Direction: drop-in replacement in `transcription_processor.py`. Config flag for backend selection during transition. Verify segment timing parity with v1 outputs on a test audio.
- Blocks: nothing. Linked: T04 (mooted if WhisperX adopted).

**T02 — Drop redundant VAD step**
- Status: identified
- Why: `pyannote/speaker-diarization-3.1` runs VAD internally as its first stage. Running a separate `pyannote/voice-activity-detection` model first is duplicate work — extra model load, extra inference, no information gained.
- Direction: extract speech regions from the diarisation pipeline output rather than running a dedicated VADProcessor. Preserve the early-exit check ("no speech, stop") on the diarisation result.
- Blocks: nothing. Linked: T04 (subsumed if WhisperX adopted).

**T03 — Word-level speaker attribution**
- Status: identified
- Why: current `_perform_speaker_attribution()` overlaps Whisper *segment* boundaries against diarisation segments and picks the dominant speaker — falls over on interruptions and cross-talk, defaults to "Unknown Speaker" whenever overlap is ambiguous. Word-level alignment fixes this by attributing each word to whichever speaker held the floor at that timestamp.
- Direction: forced alignment (e.g. wav2vec2 via `whisperx` or standalone `ctc-forced-aligner`). Output schema changes: segments carry per-word `(word, start, end, speaker)` tuples.
- Blocks: nothing. Linked: T04 (provided natively by WhisperX).

**T04 — Evaluate WhisperX as orchestrator replacement**
- Status: identified
- Why: WhisperX bundles VAD + faster-whisper + alignment + diarisation in one library with battle-tested integration. If adopted, it collapses T01 + T02 + T03 and replaces ~60% of the current orchestrator code. Cost is dependency weight and reduced control over the pipeline shape.
- Direction: time-boxed spike — run WhisperX against a representative audio file, compare accuracy and runtime against v1. Decision: adopt, partial-adopt (use as library, keep our orchestrator), or skip.
- Blocks: T01, T02, T03 (decision needed before committing effort to those if adoption is on the table). Linked: Q1.

**T05 — Long-audio chunked processing**
- Status: identified
- Why: a 3-hour podcast loads entirely into memory before Whisper transcription. Pyannote diarisation also holds the full audio. Memory grows with audio length even though models process in windows internally.
- Direction: chunk input audio at silence boundaries, run pipeline per chunk, stitch outputs with overlap handling for speaker continuity across chunk seams.
- Blocks: nothing. Linked: T01 (faster-whisper makes per-chunk overhead lower).

**T06 — Batch mode (multi-file processing)**
- Status: identified
- Why: load/unload-per-file is wasteful when processing N files. Each file pays the model-load cost three times (~15-45s overhead per file). Processing all files through one stage before moving to the next amortises loads.
- Direction: batch flag on CLI. Restructure `TranscriptionManager` so model load happens outside the per-file loop. Trade-off: holds intermediate state for all files in memory simultaneously — quantify before committing.
- Blocks: nothing. Linked: T05 (chunking changes what "per file" means).
