---
title: "Pin verification — v2 stack installs cleanly on Windows + CUDA"
type: bus-research
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
source: "Direct install in throwaway venv venv-v2-test/ on the project machine"
feeds_plan: "Bus/202605040240_PLAN_v2-rebuild.md (Step 1 of 10)"
verdict: PASS
---

## Verdict

**PASS.** The recommended pin set from RESEARCH 230 installs cleanly on the operator's actual machine (Windows 11 + RTX 3070 + CUDA 12.x) with zero pip resolver warnings. All three required imports succeed. CUDA is available and the GPU is detected as expected.

This is the discipline gate that the abandoned WhisperX PLAN 220 lacked — confirming the dependency triple is real before committing code to it.

## Method

Throwaway venv created (`venv-v2-test/`, separate from the project venv). Python 3.12 (system default; project venv is 3.11 — see "Variance" below).

Install commands, in order:
```
python -m venv venv-v2-test
venv-v2-test\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121
pip install ctranslate2==4.5.0 faster-whisper==1.2.0 pyannote.audio==3.3.2 soundfile librosa "numpy>=2.0" pyyaml
```

## Resolver warnings observed

**None.** No `ERROR: pip's dependency resolver does not currently take into account...` lines in either install command. Contrast with the abandoned WhisperX install which produced four such warnings (`pyannote-audio 4.0.4 requires torch>=2.8.0`, `torchvision 0.23.0 requires torch==2.8.0`, etc.).

## Import-check results

| Check | Command | Result |
|---|---|---|
| torch + CUDA | `python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"` | `torch 2.5.1+cu121 cuda OK NVIDIA GeForce RTX 3070` |
| faster-whisper | `python -c "from faster_whisper import WhisperModel"` | `faster-whisper OK` |
| pyannote.audio | `python -c "from pyannote.audio import Pipeline"` | `pyannote OK` |

GPU detected: **NVIDIA GeForce RTX 3070, 8 GB VRAM**. Matches the design constraint exactly (the RESEARCH 230 stack was sized for an 8 GB ceiling).

## Resolved pin set (durable record)

The four load-bearing pins:

| Package | Pinned | Resolved |
|---|---|---|
| torch | 2.5.1+cu121 | 2.5.1+cu121 |
| torchaudio | 2.5.1+cu121 | 2.5.1+cu121 |
| ctranslate2 | 4.5.0 | 4.5.0 |
| faster-whisper | 1.2.0 | 1.2.0 |
| pyannote.audio | 3.3.2 | 3.3.2 |
| numpy | >=2.0 | 2.4.4 |
| soundfile | latest | 0.13.1 |
| librosa | latest | 0.11.0 |
| pyyaml | latest | 6.0.3 |

Notable transitive resolutions:
- `pyannote-core` → 6.0.1 (newer than the v1 5.0.0; pyannote.audio 3.3.2 accepts both)
- `pyannote-pipeline` → 4.0.0
- `pyannote-database` → 6.1.1
- `pyannote-metrics` → 4.0.0
- `lightning` → 2.6.1
- `pytorch-lightning` → 2.6.1
- `tokenizers` → 0.23.1
- `huggingface_hub` → 1.13.0
- `protobuf` → 7.34.1
- `tqdm` → 4.67.3

Full `pip list --format=freeze` output captured separately (committed alongside this file as evidence — 100+ lines).

## Variance vs. plan

- **Python version**: throwaway venv used Python 3.12.x (system default on the operator's Windows 11). The project venv is Python 3.11.0. PyPI wheels exist for both Python lines for every load-bearing package — the only thing to watch is `cp312-cp312` vs `cp311-cp311` wheel availability when re-installing in the project venv. All four load-bearing packages have cp311 wheels (verified in earlier WhisperX install logs).

- **Pyannote sub-packages drift forward**: `pyannote.audio==3.3.2` accepts the 6.x major bumps of `pyannote.core / database / metrics / pipeline`. This is intentional — those sub-packages diverged from `pyannote.audio`'s versioning. Not an error; flag in case future audits see "pyannote 6.x" and assume something is broken.

- **No CUDA pin in the requirements file is silent.** The `--index-url https://download.pytorch.org/whl/cu121` is a runtime pip flag, not a package version. Step 2 of PLAN 240 will encode this as a comment + a separate install instruction in `requirements.v2.txt`. A bare `pip install -r requirements.v2.txt` against PyPI would still pull CPU-only torch wheels — the operator-facing install docs must be explicit.

## What this does NOT verify

- That the **models** load. `WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")` would download ~1.5 GB and exercise the cuDNN runtime. Step 10 of PLAN 240 (smoke run) handles this.
- That **diarisation** runs. Pyannote 3.1 needs an HF token (already on disk at `D:/Projects/Tokens/kc_diariser.txt`); first download is several hundred MB. Step 10 handles this too.
- **VRAM** at runtime. Theoretical headroom is ~3 GB ASR + ~2 GB diariser; actual depends on batch size and sequence length. Measured at Step 10.
- **WER quality** vs. the Teams reference transcript. Out of scope for Step 1 — a separate quality-evaluation question.

## Implication for PLAN 240

Step 1: **DONE**. Step 2 (update project venv with verified pins) can proceed with confidence. The remaining nine steps no longer have an install-failure risk hanging over them.

## Sources

- Direct install + import verification on the operator's machine, 2026-05-04.
- Recommended pin set: `Bus/202605040230_RESEARCH_v2-stack-and-pipeline.md` § "The recommended stack — concrete pins".
