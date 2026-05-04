---
title: "v2 stack & pipeline decision — drop WhisperX, build on faster-whisper + pyannote 3.3"
type: bus-research
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
sources:
  - "Bus/202605040210_RESEARCH_v2-requirements.md (operator requirements)"
  - "Bus/202605040220_RESEARCH_audio-characteristics.md (concrete audio measurements)"
  - "Background-agent SOTA survey (general-purpose subagent, 2026-05-04)"
supersedes:
  - "Bus/202605040220_PLAN_v2-implementation.md (WhisperX-anchored — superseded)"
feeds_plan: "202605040240_PLAN_v2-rebuild.md (to be drafted)"
---

## TL;DR

**Drop WhisperX.** Build v2 on **`faster-whisper` + `pyannote.audio 3.3.x`** with a thin attribution layer we own. Same model components Whisper-X wraps; no third-party dependency-pin churn between us and the GPU. Pipeline: ingest → adaptive preprocessing → voice-presence gate → diarisation → ASR with built-in Silero VAD → attribution → output, with cheap fail-fast validation gates between each stage. Whisper variant: `large-v3-turbo`, `compute_type="int8_float16"`, fits in ~3 GB VRAM on the 8 GB ceiling. Concrete pin set documented below.

This decision overturns my earlier WhisperX choice (PLAN 202605040220, marked for supersession by this file).

## Why the earlier WhisperX pick was wrong

Recap of the criticism the operator extracted from me on 2026-05-04:

1. I anchored on WhisperX because the operator named it as an example, not a directive. No alternatives analysis happened.
2. WhisperX is "three models in a trench coat" — Whisper (via faster-whisper) + wav2vec2 alignment + pyannote diarisation. The "simplification" is at the call site, not the architecture.
3. The install path on Windows + CUDA is a known pain point (issues #1216, #1295, #954, #1158, PR #1182 in m-bain/whisperX). I confirmed this empirically: bare `pip install whisperx` pulled CPU-only torch 2.8.0 from PyPI; force-reinstalling CUDA torch 2.5.1+cu121 broke whisperx's `torch~=2.8` pin and pyannote-audio 4.0.4's `torch>=2.8` pin in one move.
4. v2 code was committed before the install was tested. Smoke tests with mocked `whisperx` modules do not prove the package actually works on the target hardware.

Net: the WhisperX choice paid for a dependency-hell tax with no architectural return.

## What changed in the SOTA between v1 (2024) and now (2026)

| Component | v1 | v2 candidate (recommended) | What's new |
|---|---|---|---|
| ASR | `openai-whisper==20240930` (reference impl) | `faster-whisper` 1.2.x | CTranslate2 backend, ~4× faster than reference, int8/int8_float16 support, BYO-torch (no pin from the library) |
| ASR model | `whisper-large-v3` (or smaller) | `whisper-large-v3-turbo` | Same accuracy class as v3, half the VRAM, ~4× faster decode |
| Diariser | `pyannote.audio==3.3.2` | **`pyannote.audio==3.3.2` (unchanged)** | 4.x exists with better quality (community-1, exclusive mode) but pins `torch==2.8.0` *exactly* — same trap as WhisperX. Stay on 3.x |
| VAD | `pyannote.VAD` separate stage | **faster-whisper's built-in Silero VAD** | Tighter integration, no separate stage, fewer timestamp-mismatch bugs |
| Forced alignment | None in v1 | None in v2 | Whisper-large-v3-turbo word timestamps are good enough for human reading; alignment was WhisperX polish, not accuracy |

Notable rejections (explored, declined):

- **NeMo Streaming Sortformer** — best published DER for diarisation, but officially WSL2-on-Windows. Not a clean pip install on bare Windows + CUDA. Hold as fallback for later.
- **NeMo Parakeet-TDT-0.6B-v3** — strong ASR with much better hallucination-on-silence resistance than Whisper. Same Windows install pain. Hold as future option.
- **pyannote 4.x** — better diarisation, but `torch==2.8.0` exact pin is the same anti-pattern we just escaped from. Revisit after issue [pyannote-audio#1976](https://github.com/pyannote/pyannote-audio/issues/1976) closes or they relax the pin.
- **Reverb (Rev.com open ASR)** — beats Whisper on technical-meeting WER, but non-commercial license. Internal-use-only might be OK for the operator; flag as revisitable.
- **distil-whisper-large-v3** — speed gain, but not the bottleneck on this use case; English-only.

## The recommended stack — concrete pins

```
Python 3.11
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
ctranslate2==4.5.0      # cuDNN9-compatible
faster-whisper==1.2.0   # no torch pin — BYO
pyannote.audio==3.3.2
soundfile, librosa, numpy>=2.0, pyyaml
ffmpeg (system, on PATH)
```

Install:

```
python -m venv venv
venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121
pip install ctranslate2==4.5.0 faster-whisper==1.2.0 pyannote.audio==3.3.2 soundfile librosa numpy pyyaml
```

This avoids the v1-attempt traps:
- `ctranslate2 4.5.0` is the cuDNN9 line that matches torch 2.5.1+cu121.
- `pyannote 3.3.2` is loose-pinned on torch.
- `faster-whisper` doesn't pin torch at all.
- The `--index-url` for PyTorch is explicit so we don't silently get the CPU wheel.

**Pin verification gate:** before any code is committed against this stack, run `pip install -r requirements.v2.txt` in a throwaway venv and confirm:
1. No dependency-resolver warnings.
2. `python -c "import torch; assert torch.cuda.is_available()"` succeeds.
3. `python -c "from faster_whisper import WhisperModel; from pyannote.audio import Pipeline"` succeeds.

This was my biggest miss the first time — I didn't actually run the install before locking the architecture.

## Pipeline shape

Replaces v1's `TranscriptionManager.process_file()` and the abandoned WhisperX-anchored v2.

```
ingest → preprocess (adaptive) → voice-presence gate → diarisation → ASR → attribution → output
   ↓           ↓                       ↓                    ↓           ↓        ↓
 GATE 0      GATE 1                  GATE 2               GATE 3      GATE 4   GATE 5
```

Each stage is a separate function with its own success/failure return; an orchestrator function walks them and applies validation gates. No singleton state, no five-stage processor inheritance hierarchy.

### Stage 0 — Ingest (cheap)
Load file metadata via `soundfile.info()` (or `librosa.get_duration()` for AAC).

### Gate 0 — Ingest sanity
- File opens without exception.
- Duration ≥ 10 s (else: probably corrupt; abort).
- Sample rate in [8 000, 192 000].
- Channels in [1, 8].

### Stage 1 — Adaptive preprocessing
Decode to PCM via FFmpeg subprocess. Decisions driven by measured input properties:

| Measurement | Action |
|---|---|
| Sample rate ≠ 16 000 Hz | Resample to 16 kHz |
| Channels > 1 | Mean to mono |
| RMS dBFS < −32 | Apply `loudnorm=I=-23:LRA=7:TP=-2` (FFmpeg native) |
| RMS dBFS ≥ −32 | Skip loudnorm |
| Noise floor dBFS > −40 AND SNR < 20 dB | Apply gentle `afftdn=nf=-25` |
| Otherwise | Skip denoise (default — denoising hurts Whisper WER on already-clean audio) |
| Spectral centroid > 6 kHz | Log warning: input has likely been pre-processed; transcription quality may suffer |

Output: a 16 kHz mono PCM tempfile.

### Gate 1 — Preprocessing didn't damage signal
- post-RMS within ±6 dB of pre-RMS (loudnorm should normalise, not destroy).
- post-spectral-centroid within reasonable range of pre (denoise shouldn't have eaten mid-band — if it did, abort and retry without denoise).

### Stage 2 — Voice-presence pass
Coarse Silero VAD over the whole file (CPU, ~1 s per hour). Compute voiced-fraction.

### Gate 2 — Voice presence
- voiced_fraction ≥ 0.05 (else: probably silence; abort with operator-friendly message).
- voiced_fraction < 0.80: log "sparse audio detected, VAD-trim will help" — informational.

### Stage 3 — Diarisation
`pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")` on GPU. Pass `min_speakers` / `max_speakers` if hinted. Save speaker timeline. Unload pipeline + clear CUDA cache.

### Gate 3 — Diarisation sanity
- 1 ≤ speaker_count ≤ (max_speakers or 10).
- No single speaker holds > 98 % of voiced time.
- No speaker holds < 0.5 % of voiced time (probably spurious cluster).

If any check fails: log warning but continue. Diarisation quality is informational, not a hard gate.

### Stage 4 — ASR
`faster_whisper.WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")`. Run with `vad_filter=True` (built-in Silero), `beam_size=1` (lower hallucination per arXiv 2501.11378), `word_timestamps=True`, `condition_on_previous_text=False` (lower hallucination chain risk). Stream segments. Unload model + clear CUDA cache.

### Gate 4 — ASR sanity
- Total transcribed text length ≥ 10 chars per minute of voiced audio (else: model is hallucinating empty / model loaded wrong).
- Detected language matches expectation (or `language=None` allows any — log it).

### Stage 5 — Attribution
For each ASR segment, find the diarisation speaker with maximum overlap inside the segment's window. v1's `_perform_speaker_attribution` heuristic, ported. On overlap-tied or no-overlap segments, attribute as `"Unknown Speaker"` (kept; documented limitation of pyannote 3.x; pyannote 4.x exclusive-mode would solve this).

### Gate 5 — Output sanity
- Output file written and non-empty.
- Segment count matches transcribed segment count.

### Stage 6 — Output
TXT (default), JSON, or SRT. Same writers as the abandoned v2 version (those are model-agnostic).

## What survives from the abandoned v2

The v2 module shapes (`v2/config.py`, `v2/output.py`, `v2/cli.py`, the dataclass-based Config, the YAML loader, the SRT/JSON/TXT writers) are stack-agnostic and survive. Only `v2/pipeline.py` and `v2/preprocessing.py` need to be rewritten against the faster-whisper + pyannote stack instead of WhisperX.

24 of the existing 33 v2 tests stay valid (config, output, preprocessing-as-FFmpeg-shell). The 6 pipeline-smoke tests need to be rewritten against the new module structure.

## Validation gate philosophy

Each gate has three properties:
1. **Cheap** — measured in milliseconds or single-digit seconds. No 30-min bills before any gate fires.
2. **Specific** — failure says exactly which stage produced the bad output and why.
3. **Recoverable where possible** — preprocessing damage retries without denoise; ASR hallucination doesn't auto-retry (it's expensive) but logs clearly.

Gates that *block* (abort the run): 0, 1, 2.
Gates that *warn* (log + continue): 3, 4, 5.

This split matches the cost asymmetry: if voice-presence fails, you're about to waste 30 minutes of GPU time on silence — block. If diarisation looks weird, you'll still get a transcript with awkward speaker labels — warn.

## Quality benchmark

The Teams-generated transcript for File 3 (`recordings/20260410_112033 - PVT - Cyber Security Control Exemption Form - Website Blocked.docx`) is the bar. v2 should produce a transcript with:
- Comparable speaker labelling (Teams uses real names; v2 will use SPEAKER_00 etc. — that's a UX gap, not an accuracy gap).
- Comparable technical-vocabulary accuracy on terms like "RITMs", "macro block", "MVP", "exemption", "control" — Whisper-large-v3-turbo should handle all of these.
- Comparable timestamp resolution (segment-level is fine; word-level is bonus).

Not a goal: lossless reproduction of Teams' formatting (paragraph breaks, capitalisation choices). v2 is a different tool.

## Open questions to surface before the next PLAN starts executing

1. **Operator confirmation that the new stack is the call.** This RESEARCH file recommends; the operator decides. After read-and-confirm, draft the rebuild PLAN.
2. **Should v2's existing module structure stay (`v2/` package) or rename to `diarizer/`?** Earlier decision was to defer until v1 is decommissioned. With WhisperX gone the deferral still makes sense — keep v2 naming until the decommission PLAN.
3. **Quality benchmark threshold.** What's "good enough" measured against the Teams reference? Suggest WER ≤ 15 % on words that appear in both transcripts as a minimum bar; ≤ 10 % as good. Operator to confirm.
4. **Reverb consideration.** If technical-jargon WER is the load-bearing requirement, Reverb ASR may beat Whisper on this use case (it does in the published benchmarks). Worth a 30-min experiment before locking ASR — or defer to a future PLAN.

## Sources (subset; full list in the background-agent transcript)

- [WhisperX issues #1216, #1295, #954, #1158](https://github.com/m-bain/whisperX/issues)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote-audio issue #1976 — torch==2.8.0 exact pin](https://github.com/pyannote/pyannote-audio/issues/1976)
- [HF deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [arXiv 2501.11378 — Whisper hallucinations from non-speech](https://arxiv.org/abs/2501.11378)
- [arXiv 2505.12969 — Calm-Whisper](https://arxiv.org/html/2505.12969v1)
- [DeepFilterNet issue #483 — degrades STT](https://github.com/Rikorose/DeepFilterNet/issues/483)
- [NVIDIA Streaming Sortformer blog](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/)
- [Reverb arXiv 2410.03930](https://arxiv.org/abs/2410.03930)
- [BrassTranscripts diarization comparison 2026](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)
- [Northflank — Best open-source STT 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
