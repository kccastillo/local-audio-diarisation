---
title: "Crosstalk indicator: heuristic flag on segments + waveform stripe for rapid speaker swaps"
type: bus-plan
status: in-progress
assigned_to: sonnet
priority: medium
created: 2026-05-06
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
pipeline_phase: executing
audit_state:
  sufficiency_iterations: 2
  plan_safety_iterations: 1
  last_stage: plan_safety
  last_outcome: success
linked_decisions:
  - "Crosstalk detection mechanism: heuristic on existing transcript timings (option (a)). Pyannote OSD model (option (b)) is deferred — heuristic works on every existing session immediately and answers the operator's actual question (where in this meeting does it get hard to follow)."
  - "Heuristic: sliding 5-second window across segments. Count speaker-changes (transitions where seg[i].speaker != seg[i-1].speaker) within window. Threshold 4+ swaps → all segments whose [start, end] intersect that window are flagged crosstalk."
  - "Computed entirely on the frontend at transcript-load time. No backend endpoint, no cached state in session.json. Works instantly on existing sessions including bridged ones."
  - "Visual indicator: dual-channel — (1) a coloured dot/stripe on the right edge of every flagged segment row in the transcript pane; (2) a faint coloured stripe drawn in the waveform background covering the time spans of all flagged regions."
  - "Single warning colour for all crosstalk indicators (suggested orange #f0a000, reusing the existing --warn variable so palette stays at 4 colours)."
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
---

## Objective

Add a visual indicator that highlights regions of the recording where rapid speaker-swapping suggests crosstalk or interruption. The indicator appears in two synchronised places: the affected transcript rows show a coloured stripe on their right edge, and the waveform shows a faint coloured stripe in its background covering the time spans of flagged regions. Detection is a pure-frontend heuristic on the existing transcript timings — no audio re-analysis, no new backend state, no model load. Works on every existing session.

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
- New Python reference implementation `diarizer/webapp/crosstalk.py` with `compute_crosstalk_regions()` — canonical algorithm definition that pytest exercises with hand-crafted fixtures.
- New JS function `computeCrosstalkRegions(segments, windowSec=5, threshold=4) -> {flaggedSegments: Set<int>, flaggedRanges: Array<{startTime, endTime}>}` that mirrors the Python reference.
- Frontend renders flagged segments with a `.crosstalk` CSS class (right-edge stripe).
- Waveform render extended to draw a faint background stripe over the time span of flagged regions, painted before the amplitude bars so it sits behind them.

**Decisions locked this session (Human, 2026-05-06):**
- Heuristic mechanism (option (a) over OSD).
- 5-second sliding window.
- 4+ speaker-swap threshold.
- Both row stripe and waveform stripe (dual-channel indicator).
- Single warning colour reusing `--warn` (#f0a000 default).
- Frontend-computed, no backend changes.
- Crosstalk uses border-RIGHT; diff uses border-LEFT (orthogonal channels, no specificity conflict).

**Out of scope.**
- Pyannote OSD integration (deferred — option (b) in the design conversation).
- Per-segment confidence scores or fuzzy "intensity" colouring (single binary flag for v1).
- Configurable window/threshold via UI knobs (locked-default for v1).
- Backend `crosstalk_regions.json` sidecar (defer to OSD work if/when it lands).

## Steps

### Step 1 — Python reference implementation + heuristic in `app.js`

**Files touched:** `diarizer/webapp/crosstalk.py`, `diarizer/webapp/static/app.js`.

**Python reference implementation** at `diarizer/webapp/crosstalk.py`:

```python
# Reference implementation. Mirror of diarizer/webapp/static/app.js:computeCrosstalkRegions.
# Keep the two algorithms in lockstep — Step 4's pytest is the canonical correctness gate.
```

Pure function signature:

```python
def compute_crosstalk_regions(segments: list[dict], window_sec: float = 5.0, threshold: int = 4) -> dict:
```

Returns `{"flagged_segments": [int, ...], "flagged_ranges": [{"start": float, "end": float}, ...]}`.

Algorithm (identical to JS version):
1. Walk segments in order. For each starting index `i`, find maximal `[i, j]` such that `segments[j].start - segments[i].start <= window_sec`.
2. Within that run, count speaker-swaps: number of `k` in `[i+1, j]` where `segments[k]["speaker"] != segments[k-1]["speaker"]`.
3. If swap-count >= threshold → flag every segment in `[i, j]` AND merge `[segments[i]["start"], segments[j]["end"]]` into `flagged_ranges` (maintain as a sorted list; merge with last entry if they overlap or touch — `next.start <= last.end` — else append).
4. Return `{"flagged_segments": sorted(list(flagged_set)), "flagged_ranges": flagged_ranges}`.

**No CLI surface, no FastAPI endpoint** — module is reference-only, imported by tests.

`verify:` `python -c "from diarizer.webapp.crosstalk import compute_crosstalk_regions; r = compute_crosstalk_regions([], 5.0, 4); assert r == {'flagged_segments': [], 'flagged_ranges': []}; print('OK empty')"`

**JS function** `computeCrosstalkRegions(segments, windowSec=5, threshold=4) -> {flaggedSegments: Set<int>, flaggedRanges: Array<{startTime, endTime}>}` in `app.js`:

Algorithm mirrors the Python reference exactly. Include this note in the JS comment block:

```
// Algorithm assumes segments[].start is monotonically non-decreasing; the pipeline guarantees this.
// If a future edit path ever re-orders segments, sort by start before calling.
```

**Wiring:** call once after `segments` is loaded in `init()` (after the original or latest-edit assignment). Re-run after any edit that mutates `segments[*].speaker`:
- after `btn-save` click handler completes successfully (before `loadVersions()`),
- after `speaker-ok` click handler's `renderTranscript()` call,
- and after `versions-dropdown` change handler's segment-load.

Store result in module-scope `crosstalk = {flaggedSegments, flaggedRanges}` so render functions can read it.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'computeCrosstalkRegions' in p and 'flaggedSegments' in p and 'flaggedRanges' in p"`

### Step 2 — Transcript row indicator

**Files touched:** `diarizer/webapp/static/{app.js,style.css}`.

**JS change in `renderTranscript`:** when building each `.segment` row, check if its index is in `crosstalk.flaggedSegments`; if so, add class `crosstalk` to the row. Also add `title="rapid speaker-swapping in this region"` on the row when crosstalk is flagged.

**CSS:** new rule `.segment.crosstalk { border-right: 3px solid var(--warn); }`.

Resolution: orthogonal channels. Crosstalk uses border-RIGHT (not border-left); diff uses border-left (existing). When a row is both crosstalk and diff-changed, both edges show simultaneously — visually clean, no cascade competition.

`verify:` `python -c "p = open('diarizer/webapp/static/style.css', encoding='utf-8').read(); assert '.segment.crosstalk' in p and 'border-right' in p and 'var(--warn)' in p"`

### Step 3 — Waveform stripe indicator

**Files touched:** `diarizer/webapp/static/app.js`.

**JS change in `drawWaveform`:** before drawing the amplitude bars, iterate `crosstalk.flaggedRanges`. For each range `{startTime, endTime}` that intersects the visible window, compute pixel x-extents via the existing `timeToX` helper, and fill a faint warning-coloured rectangle from x1 to x2 across the full canvas height. Use a low-alpha colour (e.g. `rgba(240, 160, 0, 0.15)`) so the amplitude bars remain dominant.

The stripe must redraw on zoom/pan changes (which `requestAnimationFrame` already drives). No new render-loop wiring needed.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'flaggedRanges' in p and 'fillRect' in p; assert 'rgba(240, 160, 0' in p or 'var(--warn)' in p"`

### Step 4 — Algorithm-correctness tests

**Files touched:** `tests/diarizer/test_crosstalk.py` (new file, or append to `tests/diarizer/test_webapp_smoke.py`).

Tests exercise the Python reference implementation `compute_crosstalk_regions()` with hand-crafted fixtures. The JS production path must mirror the Python reference; these tests are the canonical correctness gate.

- `test_crosstalk_empty_returns_empty` — `compute_crosstalk_regions([])` returns `{"flagged_segments": [], "flagged_ranges": []}`.
- `test_crosstalk_single_speaker_session_no_flags` — 10 segments, all `SPEAKER_00`. No swaps possible. `flagged_segments == []`, `flagged_ranges == []`.
- `test_crosstalk_two_segments_below_threshold` — 2 segments with one swap. Below threshold 4. No flags.
- `test_crosstalk_dense_swaps_flagged` — 5 segments at start times [0, 0.5, 1, 1.5, 2] with alternating speakers (4 swaps in a 2s window). With threshold=4 and window=5s, all 5 segments should be flagged; `flagged_ranges == [{"start": 0, "end": <last_segment_end>}]`.
- `test_crosstalk_sparse_pause_no_flags` — 2 quick swaps then 30s of silence then 2 more quick swaps. Each cluster has only 1–2 swaps within its 5s window. No flags.
- `test_crosstalk_long_segment_does_not_break_window_walk` — single 60s segment followed by a few rapid swaps. The long segment does not artificially extend the window's swap count.
- `test_crosstalk_threshold_boundary_exactly_at_threshold` — 5 segments with exactly 4 swaps in 5s. With threshold=4 (>=, not >), should flag.
- `test_crosstalk_just_below_threshold` — 4 segments with 3 swaps in 5s. Below threshold. No flags.
- `test_crosstalk_overlapping_windows_merge_ranges` — two crosstalk clusters within 1s of each other (both individually pass threshold). Resulting `flagged_ranges` should be a single merged range covering both clusters.

Also keep a single grep-style test verifying the JS function exists with the correct signature:

- `test_app_js_has_crosstalk_helper` — grep that `computeCrosstalkRegions` symbol is present in `app.js`, accepts `(segments, windowSec, threshold)`, and returns `{flaggedSegments, flaggedRanges}`.

`acceptance:` `pytest tests/diarizer/test_crosstalk.py -q` (or `tests/diarizer/test_webapp_smoke.py -q` if appended there).

### Step 5 — Manual UI walkthrough (verify: human)

Operator-visible affordance checklist for the new behaviour. Each item must be confirmed working in Chrome and Firefox. Any failure → outcome-verification reverts to drafted.

The human walkthrough serves as the JS-vs-Python parity gate, not as the algorithm-correctness gate (Step 4 covers correctness).

0. The flagged segments visible in the browser MATCH the segments returned by `compute_crosstalk_regions()` in Python on the same transcript JSON. Operator may verify by running `python -c 'from diarizer.webapp.crosstalk import compute_crosstalk_regions; import json; print(compute_crosstalk_regions(json.load(open("<session>/transcript.json"))["segments"]))'` and comparing the `flagged_segments` list against the orange-stripe rows in the browser.
1. Loading a session with rapid speaker swaps (e.g. the existing 28-apr meeting) shows orange right-edge stripes on the affected transcript rows.
2. The same time spans appear as faint orange background stripes in the waveform canvas.
3. Hovering a flagged row shows the tooltip "rapid speaker-swapping in this region".
4. Loading a session with calm, single-speaker stretches shows NO crosstalk indicators on those segments — no false positives in obvious quiet regions.
5. Editing a speaker label via the Speaker modal (e.g. globally renaming `SPEAKER_00` to `Alice`) re-evaluates crosstalk: indicators stay correct because identity-only relabelling does not change swap counts.
6. Saving and reloading the latest edit-version preserves indicator placement (the heuristic is deterministic on a given segments array).
7. Zoom-in on the waveform — the orange background stripes still cover the correct time spans (no clipping or drift).
8. A row that is BOTH crosstalk-flagged AND edit-changed shows a coloured border on the LEFT edge (diff, accent blue) AND a coloured border on the RIGHT edge (crosstalk, warn orange) simultaneously.

`verify: human — operator runs items 0–8; all must pass.`

## Verification

- [ ] Python reference `compute_crosstalk_regions()` present in `diarizer/webapp/crosstalk.py`; empty-input smoke passes. (Step 1)
      `verify: python -c "from diarizer.webapp.crosstalk import compute_crosstalk_regions; r = compute_crosstalk_regions([], 5.0, 4); assert r == {'flagged_segments': [], 'flagged_ranges': []}; print('OK empty')"`
- [ ] `computeCrosstalkRegions`, `flaggedSegments`, and `flaggedRanges` present in `app.js`. (Step 1)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'computeCrosstalkRegions' in p and 'flaggedSegments' in p and 'flaggedRanges' in p"`
- [ ] `.segment.crosstalk` rule with `border-right` and `var(--warn)` present in `style.css`. (Step 2)
      `verify: python -c "p = open('diarizer/webapp/static/style.css', encoding='utf-8').read(); assert '.segment.crosstalk' in p and 'border-right' in p and 'var(--warn)' in p"`
- [ ] `flaggedRanges`, `fillRect`, and crosstalk-fill colour present in `app.js`. (Step 3)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'flaggedRanges' in p and 'fillRect' in p; assert 'rgba(240, 160, 0' in p or 'var(--warn)' in p"`
- [ ] Full algorithm-correctness test suite passes. (Step 4)
      `acceptance: pytest tests/diarizer/test_crosstalk.py -q`
- [ ] Manual UI walkthrough: operator runs affordance checklist items 0–8; all must pass. (Step 5)
      `verify: human — operator runs items 0–8; all must pass.`

## Executor Notes

(Empty — to be populated by the executor.)

## Notes

Previous webapp PLAN: `Retired/202605060300_PLAN_webapp-v2-affordances.md` (retired after successful execution 2026-05-06).
