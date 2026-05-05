---
title: "ASR knob panel — App Control meeting, 7 configs scored vs manual corrections"
type: bus-research
created: 2026-05-06
created_by: opus
last_updated: 2026-05-06
sources:
  - "scripts/run_asr_panel.py"
  - "scripts/score_panel.py"
  - "output/app-control-meeting-28apr manual corrections.txt (operator ground truth)"
  - "output/panel/app-control/*.txt (7 panel outputs — gitignored under output/)"
verdict: "Baseline retained as default. F (cond=True + prompt) available as opt-in via CLI flags."
---

## Why this exists

After the v2 cutover, operator did a manual pass on the App Control meeting transcript and flagged three patterns of remaining error: (1) accent-related infidelities concentrated on Ejaz's speech (`incident` → `instant`, `Cyber` → `Saiba`, `audit` → `auto/order`), (2) proper-noun losses across all speakers (`Pradeep`, `Sadeed`, `Declan`, `Steve Russell`, `Dave Jaremin`, `Leigh`, `Oliver`), (3) trailing soft-spoken decision points dropped from sentence ends.

Operator's hunch: diminishing returns on further tweaking. Asked whether targeted ASR-config knobs could close the gap before declaring v2 good enough.

This RESEARCH captures the panel that tested that hunch, on 2026-05-06, against the operator-corrected transcript as ground truth.

## Setup

Three new ModelConfig fields shipped in `c5447f5` (defaults preserve baseline):

- `beam_size: int = 1`
- `condition_on_previous_text: bool = False`
- `initial_prompt: Optional[str] = None` (with `initial_prompt_path` file alternative)

Operator-supplied prompt (`prompts/app-control.txt`):

> Cybersecurity meeting transcript on Application Control at Seqwater. Names: Ken, Peter, Ejaz, Pradeep, Sadeed, Declan, Bryan, Leigh, Oliver, Dave Jaremin, Steve Russell. Terms: Airlock, Citrix, Rapid7, Intune, OTP, RITM, BAU, Ring Zero, out-of-hours, blacklisting, audit, incident, configuration, executive, unexecuted, fiddle.

Panel matrix (run via `scripts/run_asr_panel.py`):

| Config | beam_size | condition_on_previous_text | initial_prompt | Purpose |
|---|---|---|---|---|
| A_baseline | 1 | F | — | reference |
| B_prompt | 1 | F | yes | does the prompt help? |
| C_beam5 | 5 | F | — | does beam search help disambiguate accent? |
| D_beam5_prompt | 5 | F | yes | additive? |
| E_cond | 1 | T | — | does carrying context forward help? |
| F_cond_prompt | 1 | T | yes | the only way to make the prompt persist past 30s |
| G_full_stack | 5 | T | yes | best plausible config |

Diarisation and preprocessing run once per audio (deterministic and identical across configs); ASR varied per config.

## Results

| Config | PN recall | DT recall | Jaccard | Short-line % | Wall (s) |
|---|---|---|---|---|---|
| A_baseline | 62.9% (39/62) | 74.0% (111/150) | 90.2% | 6.4% | 39.8 |
| B_prompt | 62.9% | 74.0% | 90.2% | 6.4% | 41.7 |
| C_beam5 | **69.4%** (43/62) ↑ | 67.3% (101/150) ↓ | 85.4% | 11.0% | 54.0 |
| D_beam5_prompt | 69.4% | 67.3% | 85.4% | 11.1% | 52.7 |
| E_cond | 59.7% (37/62) ↓ | **81.3%** (122/150) ↑↑ | 85.3% | 29.4% ⚠ | 52.0 |
| F_cond_prompt | 66.1% (41/62) ↑ | 76.0% (114/150) ↑ | 83.7% | 20.0% ⚠ | 53.4 |
| G_full_stack | 59.7% ↓ | 66.0% ↓ | 84.0% | 94.8% 💀 | 111.4 |

PN = proper-noun recall (62 ground-truth occurrences); DT = domain-term recall (150 ground-truth occurrences); Jaccard = lowercase-token-set overlap with the manual-corrections file; Short-line % = fraction of segments ≤4 words (degeneration signal).

## Key learnings

### 1. `initial_prompt` alone has zero effect past the first audio chunk

A_baseline and B_prompt are byte-identical from 00:00:31 onwards. The prompt biases Whisper's first decode pass and then fades completely — its influence does not persist into subsequent chunks unless `condition_on_previous_text=True` is also enabled. None of the 62 proper-noun mentions in the ground truth fall in those first 30 seconds, so the prompt's bias is invisible at the recall metric.

**Implication.** For long meetings, `initial_prompt` is essentially ornamental on its own. To use it meaningfully you must pair it with `condition_on_previous_text=True` so the prompt context gets carried forward.

### 2. `beam_size=5` is a trade, not an upgrade

Beam search at width 5 recovers some rare proper nouns Whisper's greedy decode missed:

- Declan: 0/3 → 2/3
- Steve Russell: 0/1 → 1/1
- Pradeep's: 0/1 → 1/1
- Emma: 4/5 → 5/5

But it **simultaneously hurts common jargon-with-accent**, especially Ejaz's signature mistakes:

- incident: 12 → 8 (Ejaz's "incident" sounds like "instant" to the decoder; beam search makes that worse because "instant" is a more common English word and outscores "incident" under longer-context probability)
- incident management: 9 → 5
- Airlock: 6 → 4
- P1: 9 → 7, P2: 9 → 8
- Rapid7: 1 → 0

Net across both vocabularies: −7 hits. Beam search is **fundamentally a search-strategy lever**, not an acoustic-modelling lever. The "incident → instant" problem is acoustic; beam search can't fix it.

### 3. `condition_on_previous_text=True` is high-reward, structurally risky

Cond=True alone (E) recovers domain terms substantially (+11 hits, biggest gain in the panel) because the prompt vocabulary persists forward across chunks. But it causes structural degeneration: by the latter half of the audio, output fragments into 1–4 word per-segment chunks. 29% of E's lines are short.

Cond=True + prompt (F) is the only config that moves both axes positively (+2 PN, +3 DT) while keeping the body structurally intact. But the **tail hallucinates**: F's last segments are "And shave it. / And shave it. / Not these shi '." — classic Whisper end-of-audio repetition-collapse near trailing silence. ~20% short lines.

The arXiv:2501.11378 concern about hallucination chains was real on this audio class, even with `vad_filter=True` and `prompt_reset_on_temperature=0.5` defaults active.

### 4. Stacking knobs degenerates more than it adds

G_full_stack (beam=5 + cond=True + prompt) breaks down severely: 95% short lines, content recall worse than baseline. The two knobs interact badly when combined — beam search's lookahead amplifies the cond-chain's degeneration tendency.

### 5. Real remaining gains live outside the ASR config space

The accent-driven errors (`incident → instant`, `Cyber → Saiba`, `OTP → OTB`) are at the **acoustic boundary** of `large-v3-turbo`'s training distribution. No knob in faster-whisper closes that gap. Three theoretical directions that would:

- **Acoustic-model swap.** `large-v3` (full, non-turbo) for marginal gain; NeMo Parakeet-TDT-0.6B-v3 for documented accent-robustness improvements. Both are install/integration jobs we punted on (Parakeet is WSL2-only on Windows, large-v3 is VRAM-tight at 8 GB).
- **Hot-word forcing.** NeMo supports hard vocabulary biasing (forced token re-weighting); faster-whisper does not. This would solve "incident vs instant" cleanly.
- **LLM post-processing pass.** Feed the transcript + a known-mistake correction list into a small local LLM for clean-up. Doesn't touch the ASR stack; cheapest remaining lever.

## Decision (2026-05-06)

Baseline retained as default. The new knobs (beam_size, condition_on_previous_text, initial_prompt) ship as **opt-in** via CLI flags or config — defaults preserve the baseline behaviour. If a future meeting really needs the +5% PN / +2% DT recall, F config is available:

```
python -m diarizer.cli --input meeting.m4a \
    --condition-on-previous-text \
    --initial-prompt-file prompts/app-control.txt
```

Caveat: tail may need manual trimming, body may have ~20% short segments. Acceptable for capture-points-matter meetings, not a default.

The LLM post-processing pass is the **next research candidate** if recall on accent-mangled jargon becomes load-bearing. Out of scope for this RESEARCH.

## Operator-confirmed transcription patterns that survive even the best config

These mistakes persist across ALL panel configs and are characteristics of `large-v3-turbo`'s acoustic prior on this speaker mix, not anything tunable in faster-whisper. Logged here so future calibration knows where the floor is:

| Speaker | Manual corrections word | Whisper consistently hears | Config that helps |
|---|---|---|---|
| Ejaz | "incident" | "instant" | none |
| Ejaz | "Cyber" | "Saiba" / "Sive" | none |
| Ejaz | "audit" | "auto" / "order" | none |
| Ejaz | "OTP" | "OTB" / "OTN" | none |
| Ejaz | "out-of-hours" | "autofower" / "EOS" / "four hours" | none |
| Peter (AU) | "Declan" | "techman" / "Doctor" | beam=5 partially |
| Peter (AU) | "Pradeep" | "potential" / "pretty" / "Pradeep" (mixed) | none consistent |
| Peter (AU) | "Steve Russell" | "stream brussel" | beam=5 |
| All | "Dave Jaremin" | "Dave Jarman" | none (close miss) |

## Files

- Code: `diarizer/config.py` (new fields), `diarizer/pipeline.py` (plumbing), `diarizer/cli.py` (flags). Tests in `tests/diarizer/test_config.py` (+5 new). Commit `c5447f5`.
- Panel runner: `scripts/run_asr_panel.py`
- Scorer: `scripts/score_panel.py`
- Per-output diff helper: `scripts/diff_transcripts.py`
- Operator-supplied prompt: `prompts/app-control.txt`
- Panel outputs: `output/panel/app-control/*.txt` (gitignored under `output/`; reproducible from the runner script)
