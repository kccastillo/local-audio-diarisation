---
title: "Crosstalk indicator: heuristic flag on segments + waveform stripe for rapid speaker swaps"
type: bus-plan
status: ready
assigned_to: sonnet
priority: medium
created: 2026-05-06
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
pipeline_phase: drafted
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
linked_decisions:
  - "Crosstalk detection mechanism: heuristic on existing transcript timings (option (a)). Pyannote OSD model (option (b)) is deferred — heuristic works on every existing session immediately and answers the operator's actual question (where in this meeting does it get hard to follow)."
  - "Heuristic: sliding 5-second window across segments. Count speaker-changes (transitions where seg[i].speaker != seg[i-1].speaker) within window. Threshold 4+ swaps → all segments whose [start, end] intersect that window are flagged crosstalk."
  - "Computed entirely on the frontend at transcript-load time. No backend endpoint, no cached state in session.json. Works instantly on existing sessions including bridged ones."
  - "Visual indicator: dual-channel — (1) a coloured dot/stripe on the left edge of every flagged segment row in the transcript pane; (2) a faint coloured stripe drawn in the waveform background covering the time spans of all flagged regions."
  - "Single warning colour for all crosstalk indicators (suggested orange #f0a000, reusing the existing --warn variable so palette stays at 4 colours)."
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
---

## Objective

Add a visual indicator that highlights regions of the recording where rapid speaker-swapping suggests crosstalk or interruption. The indicator appears in two synchronised places: the affected transcript rows show a coloured stripe on their left edge, and the waveform shows a faint coloured stripe in its background covering the time spans of flagged regions. Detection is a pure-frontend heuristic on the existing transcript timings — no audio re-analysis, no new backend state, no model load. Works on every existing session.

## Context

**What exists today.**
- The webapp lives at `diarizer/webapp/` with backend (`app.py`), peaks (`peaks.py`), and frontend (`static/{app.js,index.html,style.css}`).
- Transcript JSON contains segments with `start`, `end`, `speaker`, `text`. Segments are non-overlapping by construction (the pipeline does max-overlap diarisation lookup per segment, picking the dominant speaker — overlap information is discarded at attribution time).
- The current pipeline cannot directly report overlapped speech because the diarisation output has been collapsed to a single speaker per segment by `pipeline.py`.
- The waveform is rendered from `waveform_peaks.json` to a canvas; the transcript is rendered as a vertical list of `.segment` rows in `#transcript`.

**Why a heuristic, not a real overlap detector.**
- Real overlap detection requires running pyannote's `pyannote/overlapped-speech-detection` model — a separate model load + inference pass at finalisation. ~20s extra on CPU, ~5s on GPU per 30 min of audio. For sessions already on disk (no overlap data), this would either need lazy-compute on first webapp request (slow) or a backfill subcommand.
- The heuristic answers the operator's *actual* question — "where in this meeting does it get hard to follow" — which correlates strongly with rapid speaker-swap density even when raw simultaneous-speech detection is not available.
- The heuristic is cheap (sub-millisecond per session in the browser) and works on every existing session including bridged ones with no source-Opus.

**What is being added.**
- New JS function `computeCrosstalkRegions(segments, windowSec=5, threshold=4) -> Set<segmentIndex>` that returns the indices of segments flagged as crosstalk.
- Frontend renders flagged segments with a `.crosstalk` CSS class (left-edge stripe).
- Waveform render extended to draw a faint background stripe over the time span of flagged regions, painted before the amplitude bars so it sits behind them.

**Decisions locked this session (Human, 2026-05-06):**
- Heuristic mechanism (option (a) over OSD).
- 5-second sliding window.
- 4+ speaker-swap threshold.
- Both row stripe and waveform stripe (dual-channel indicator).
- Single warning colour reusing `--warn` (#f0a000 default).
- Frontend-computed, no backend changes.

**Out of scope.**
- Pyannote OSD integration (deferred — option (b) in the design conversation).
- Per-segment confidence scores or fuzzy "intensity" colouring (single binary flag for v1).
- Configurable window/threshold via UI knobs (locked-default for v1).
- Backend `crosstalk_regions.json` sidecar (defer to OSD work if/when it lands).

## Steps

### Step 1 — Heuristic implementation in `app.js`

**Files touched:** `diarizer/webapp/static/app.js`.

**Function:** `computeCrosstalkRegions(segments, windowSec=5, threshold=4) -> {flaggedSegments: Set<int>, flaggedRanges: Array<{startTime, endTime}>}`.

Algorithm:
1. Walk all segments in order. For each starting index `i`, find the maximal contiguous run `[i, j]` such that `segments[j].start - segments[i].start <= windowSec`.
2. Within that run, count speaker-swaps: number of `k` in `[i+1, j]` where `segments[k].speaker !== segments[k-1].speaker`.
3. If swap-count >= threshold → mark every segment in `[i, j]` as flagged AND merge `[segments[i].start, segments[j].end]` into a flagged-range list (merge overlapping ranges into a single span).
4. Return `{flaggedSegments, flaggedRanges}`.

**Wiring:** call once after `segments` is loaded in `init()` (after the original or latest-edit assignment). Re-run after any edit that mutates `segments[*].speaker` (Save handler, Speaker modal Submit, version-load). Store result in module-scope `crosstalk = {flaggedSegments, flaggedRanges}` so render functions can read it.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'computeCrosstalkRegions' in p and 'flaggedSegments' in p and 'flaggedRanges' in p"`

### Step 2 — Transcript row indicator

**Files touched:** `diarizer/webapp/static/{app.js,style.css}`.

**JS change in `renderTranscript`:** when building each `.segment` row, check if its index is in `crosstalk.flaggedSegments`; if so, add class `crosstalk` to the row.

**CSS:** new rule `.segment.crosstalk { border-left: 3px solid var(--warn); }`. The existing `.segment.diff-changed` already uses `border-left: 3px solid var(--accent)` — to avoid the two collisions, add a tooltip via `title="rapid speaker-swapping in this region"` on the row when crosstalk is flagged. If a row is BOTH diff-changed AND crosstalk, the diff border wins (more specific in the user's current task); document this resolution in app.js comments.

`verify:` `python -c "p = open('diarizer/webapp/static/style.css', encoding='utf-8').read(); assert '.segment.crosstalk' in p and 'var(--warn)' in p"`

### Step 3 — Waveform stripe indicator

**Files touched:** `diarizer/webapp/static/app.js`.

**JS change in `drawWaveform`:** before drawing the amplitude bars, iterate `crosstalk.flaggedRanges`. For each range `{startTime, endTime}` that intersects the visible window, compute pixel x-extents via the existing `timeToX` helper, and fill a faint warning-coloured rectangle from x1 to x2 across the full canvas height. Use a low-alpha colour (e.g. `rgba(240, 160, 0, 0.15)`) so the amplitude bars remain dominant.

The stripe must redraw on zoom/pan changes (which `requestAnimationFrame` already drives). No new render-loop wiring needed.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'flaggedRanges' in p and 'fillRect' in p and 'rgba(240' in p or 'crosstalk' in p.lower()"`

### Step 4 — Smoke tests

**Files touched:** `tests/diarizer/test_webapp_smoke.py`.

Crosstalk computation lives entirely in JS (no Python surface). Add the following Python-level tests anyway, to lock in invariants the JS code must satisfy:

- `test_app_js_has_crosstalk_helper` — grep-style: `computeCrosstalkRegions` symbol present in app.js, accepts `(segments, windowSec, threshold)`, returns `{flaggedSegments, flaggedRanges}`.
- `test_app_js_has_crosstalk_render_hooks` — grep that `renderTranscript` references `crosstalk.flaggedSegments` AND `drawWaveform` references `crosstalk.flaggedRanges`.

These are static checks — they prove the wiring exists, not that the algorithm is correct. Manual walkthrough item 5 below covers algorithm correctness.

`acceptance:` `pytest tests/diarizer/test_webapp_smoke.py -q`

### Step 5 — Manual UI walkthrough (verify: human)

Operator-visible affordance checklist for the new behaviour. Each item must be confirmed working in Chrome and Firefox. Any failure → outcome-verification reverts to drafted.

1. Loading a session with rapid speaker swaps (e.g. the existing 28-apr meeting) shows orange left-edge stripes on the affected transcript rows.
2. The same time spans appear as faint orange background stripes in the waveform canvas.
3. Hovering a flagged row shows the tooltip "rapid speaker-swapping in this region".
4. Loading a session with calm, single-speaker stretches shows NO crosstalk indicators on those segments — no false positives in obvious quiet regions.
5. Editing a speaker label via the Speaker modal (e.g. globally renaming `SPEAKER_00` to `Alice`) re-evaluates crosstalk: indicators stay correct because identity-only relabelling does not change swap counts.
6. Saving and reloading the latest edit-version preserves indicator placement (the heuristic is deterministic on a given segments array).
7. Zoom-in on the waveform — the orange background stripes still cover the correct time spans (no clipping or drift).
8. The crosstalk colour is visually distinct from the diff-mode `--accent` border on changed segments — a row that is both crosstalk-flagged and edit-changed shows the diff border (accent), not the warn border, per the documented resolution.

`verify: human — operator runs items 1–8; all must pass.`

## Verification

- [ ] `computeCrosstalkRegions`, `flaggedSegments`, and `flaggedRanges` present in `app.js`. (Step 1)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'computeCrosstalkRegions' in p and 'flaggedSegments' in p and 'flaggedRanges' in p"`
- [ ] `.segment.crosstalk` rule with `var(--warn)` present in `style.css`. (Step 2)
      `verify: python -c "p = open('diarizer/webapp/static/style.css', encoding='utf-8').read(); assert '.segment.crosstalk' in p and 'var(--warn)' in p"`
- [ ] `flaggedRanges`, `fillRect`, and crosstalk-fill colour present in `app.js`. (Step 3)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'flaggedRanges' in p and 'fillRect' in p and 'rgba(240' in p or 'crosstalk' in p.lower()"`
- [ ] Full smoke test suite passes. (Step 4)
      `acceptance: pytest tests/diarizer/test_webapp_smoke.py -q`
- [ ] Manual UI walkthrough: operator runs affordance checklist items 1–8; all must pass. (Step 5)
      `verify: human — operator runs items 1–8; all must pass.`

## Executor Notes

(Empty — to be populated by the executor.)

## Notes

Previous webapp PLAN: `Retired/202605060300_PLAN_webapp-v2-affordances.md` (retired after successful execution 2026-05-06).
