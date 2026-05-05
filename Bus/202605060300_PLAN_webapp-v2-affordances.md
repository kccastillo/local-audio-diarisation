---
title: "Webapp v2 affordances: waveform scrub, TXT export, play/sync toggle, segment-nav arrows"
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
  sufficiency_iterations: 0
  plan_safety_iterations: 0
  last_stage: none
  last_outcome: none
linked_decisions:
  - "Replace spectrogram with scrolling mono waveform — amplitude is the operator's main affordance; mono source means stereo split is moot"
  - "Drag horizontally on waveform canvas → audio seeks; pixel-to-time ratio 2px = 1 second; live preview during drag"
  - "Play/Pause is now a SYNC toggle. Play resumes audio + enables sync (auto-highlight active line, auto-scroll transcript). Pause halts both."
  - "Auto-pause on edit: clicking any transcript line OR pressing Up/Down arrow keys pauses audio AND disables sync. Cursor lands on that segment for editing."
  - "Up/Down arrow keys navigate between adjacent segments. Left/Right arrows are LEFT ALONE — browsers use them for caret movement in contenteditable text."
  - "Export TXT format matches the existing pipeline's [SPEAKER] hh:mm:ss text format (one line per segment, hours always shown)"
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
---

## Objective

Add four affordances to the transcript-review webapp:
1. Replace the spectrogram with a scrolling, high-contrast amplitude waveform.
2. Make the waveform draggable to scrub the audio.
3. Add a TXT-export endpoint + button that emits the existing `[SPEAKER] hh:mm:ss text` format.
4. Refactor the play/pause button into a SYNC toggle: Play resumes audio AND enables auto-highlight + auto-scroll-to-active-line; Pause halts both. Clicking any segment OR pressing Up/Down arrows auto-pauses (drops out of sync into edit mode).

## Context

**What exists today.**
- Webapp under `diarizer/webapp/` — FastAPI backend (`app.py`), vanilla JS frontend (`static/app.js`, `index.html`, `style.css`). Built in `Retired/202605060200_PLAN_transcript-review-webapp.md`; decisions D1–D8 from that PLAN remain in force.
- Spectrogram is rendered each frame from a Web Audio AnalyserNode (FFT 1024, viridis colour ramp, 1px-per-frame horizontal scroll). Operator feedback: hard to read amplitude.
- Active-segment highlight follows the playhead unconditionally (no concept of "sync mode" — it just always tracks).
- Save and Diff buttons exist; no TXT export.
- No keyboard navigation between segments. Up/Down do nothing in the current frontend.
- Audio source is 16 kHz mono Opus (locked D6 in `Retired/202605060200_PLAN_transcript-review-webapp.md`). Stereo channel-split visualisation is therefore not available.

**What is being changed.**
- Visualisation swap: AnalyserNode's `getByteTimeDomainData()` (PCM samples in [-1, 1] equivalents) replaces `getByteFrequencyData()`. Render as a centred waveform with peak envelope per pixel column, single accent colour on dark background.
- New drag handler on the waveform canvas: `mousedown` → start drag; `mousemove` (during drag) → seek `audio.currentTime` by `Δx / 2` seconds (preview); `mouseup`/`mouseleave` → finalise. Touch events mirror this.
- New endpoint `GET /api/transcript/export.txt` returns `text/plain` content with one line per segment in `[SPEAKER] hh:mm:ss text` format. Frontend "Export TXT" button triggers a browser download of the current in-memory edit state.
- Frontend state machine refactor: introduce a single `syncMode` boolean. Play handler sets `syncMode = true`, scrolls active row into view, calls `audio.play()`. Pause handler sets `syncMode = false`, calls `audio.pause()`. Active-highlight logic only runs when `syncMode` is true. Clicking any segment row OR pressing Up/Down (when focus is anywhere in the transcript) calls a `dropOutOfSync()` helper that pauses + disables sync + focuses the target row's text cell. Up/Down move focus to the previous/next segment row's text cell.

**Decisions locked this session (Human, 2026-05-06):**
- Visualisation: scrolling mono waveform (operator chose option (a) over higher-contrast spectrogram or hybrid).
- Drag-to-scrub pixel ratio: 2px = 1 second.
- Up/Down navigate segments; Left/Right untouched.
- TXT export format: identical to existing pipeline's TXT writer (`[SPEAKER] hh:mm:ss text`).

**Out of scope.**
- Audio zoom on the waveform (deferred from `Retired/202605060200_PLAN_transcript-review-webapp.md`).
- Segment boundary editing (still deferred).
- Frequency-domain visualisation (operator chose amplitude-only).
- In-session undo/redo stack (still deferred).

## Steps

### Step 1 — Replace spectrogram render with waveform

- File: `diarizer/webapp/static/app.js`.
- In the existing `drawSpectrogram` render loop, swap the per-column rendering: replace `analyser.getByteFrequencyData()` with `analyser.getByteTimeDomainData()`. Compute the peak deviation from 128 (the silent zero-line in unsigned-byte PCM) per frame; map to a vertical line length centred on `canvas.height / 2`. Draw with a single high-contrast accent colour (`#5fd97a` or similar bright green — confirm in style.css's accent variable).
- Remove the viridis ramp helper.
- Rename the function to `drawWaveform` (search-and-replace; only used in the requestAnimationFrame chain and `init()`).
- `style.css`: ensure `--accent-wave: #5fd97a` (or chosen colour) is defined and used.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'getByteTimeDomainData' in p and 'getByteFrequencyData' not in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'drawWaveform' in p"`

### Step 2 — Add drag-to-scrub on waveform canvas

- File: `diarizer/webapp/static/app.js`.
- Attach `mousedown` to the canvas. On mousedown, capture initial `audio.currentTime` and `e.clientX`. Set `dragging = true`. On `mousemove` while dragging, compute `Δx = e.clientX - initialX`; new currentTime = clamp(`initialTime + Δx / 2`, 0, audio.duration). Seek live during drag (preview). On `mouseup` or `mouseleave`, `dragging = false`.
- Mirror with `touchstart`/`touchmove`/`touchend` (single-touch only; ignore multi-touch).
- Add `cursor: ew-resize;` on the canvas in `style.css`.
- The drag must not interfere with play/pause state. If the user starts a drag while audio is playing, audio continues playing at the new time. If paused, it stays paused.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'mousedown' in p and 'mousemove' in p and 'currentTime' in p"`

### Step 3 — TXT export endpoint + button

- File: `diarizer/webapp/app.py`. New endpoint `GET /api/transcript/export.txt?from=<filename>` (the `from` parameter is optional; default = latest edit, or original if no edits). Returns `text/plain; charset=utf-8` with `Content-Disposition: attachment; filename=<source_basename>_export_<YYYYMMDD>.txt`.
- Format: one line per segment, `[SPEAKER] hh:mm:ss text` — hours always shown (matching existing `output.py:_format_timestamp`). Reuse the same formatter logic; do NOT call into `diarizer.output` directly to avoid coupling — copy the small `_format_timestamp` helper into `app.py`.
- Edge cases: missing speaker → omit `[<speaker>]` prefix; missing text → empty line.
- File: `diarizer/webapp/static/index.html`. Add `<button id="btn-export" type="button">Export TXT</button>` next to Save.
- File: `diarizer/webapp/static/app.js`. Wire `btn-export` to POST the current in-memory edit state as the body of a new `POST /api/transcript/export.txt` (operator may have unsaved edits — we don't want to require saving first). Backend: accept optional JSON body containing `{segments: [...]}`; if absent, fall back to the latest saved sidecar.
- Browser-download trick: receive the response as a Blob, create an object URL, programmatic `<a href download>` click.

`verify:` unit test in `tests/diarizer/test_webapp_smoke.py` exercises the endpoint with both the GET (latest-edit) form and POST (in-memory body) form, asserts `text/plain` content type and `[SPEAKER]` lines.

### Step 4 — Play/Pause as sync toggle + auto-pause-on-edit

- File: `diarizer/webapp/static/app.js`.
- Introduce module-scoped `let syncMode = false;`.
- New helpers:
  - `enterSync()` — sets `syncMode = true`, calls `audio.play()`, button text → "Pause", scrolls active row (if any) into view via `row.scrollIntoView({block: 'nearest'})`.
  - `exitSync()` — sets `syncMode = false`, calls `audio.pause()`, button text → "Play".
  - `dropOutOfSync(targetIndex?)` — calls `exitSync()`; if `targetIndex` provided, focuses that row's `.text` element and places cursor at end.
- Refactor existing play/pause click handler to toggle between `enterSync` and `exitSync`.
- Refactor `tickHighlight` to early-return when `syncMode === false`. Active-highlight class is removed at the moment sync is exited (visual cue for current state is via the button label, not via the highlight).
- Click handler on each segment row → `dropOutOfSync(rowIndex)` (preserves current behaviour where clicking timestamp seeks audio — keep that on the timestamp element specifically; don't fire `dropOutOfSync` from a timestamp click since that's a navigational action; only fire it when the click lands on the speaker pill, the row body, or the text cell).
- Keyboard: `keydown` listener on `document`. If `key === 'ArrowUp'` or `key === 'ArrowDown'` AND the active element is inside `#transcript`, prevent default, compute target index (`current ± 1`, clamped), call `dropOutOfSync(target)`. Do NOT intercept arrow keys when focus is outside `#transcript` (e.g. inside the volume slider).
- Verify Left/Right are NOT bound — keep contenteditable cursor movement intact.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'syncMode' in p and 'enterSync' in p and 'exitSync' in p and 'dropOutOfSync' in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'ArrowUp' in p and 'ArrowDown' in p"`

### Step 5 — Smoke tests for the new endpoints + frontend invariants

- File: `tests/diarizer/test_webapp_smoke.py`. Append:
  - `test_export_txt_get_returns_plaintext` — GET `/api/transcript/export.txt`, asserts 200, `text/plain` content type, `[SPEAKER_00]` substring present.
  - `test_export_txt_post_uses_body_segments` — POST with `{segments: [{start, end, text, speaker}]}` containing a custom speaker label, asserts that label appears in the response (proves the body wins over saved files).
  - `test_export_txt_format_includes_timestamps` — assert lines match a regex `^\[.+\] \d{2}:\d{2}:\d{2} `.

`acceptance:` `pytest tests/diarizer/test_webapp_smoke.py -q`

### Step 6 — Manual UI walkthrough (verify: human)

Operator-visible affordance checklist for the new behaviours. Each item must be confirmed working in Chrome and Firefox. Any failure → outcome-verification reverts to drafted.

1. Visualisation is now a centred waveform (no longer a colour-ramp spectrogram); amplitude varies visibly with speech vs silence.
2. Waveform colour is high-contrast and easily readable on the dark background.
3. Click + drag on the waveform → audio currentTime moves; release → final position holds. Drag works in both directions.
4. Drag of ~200 pixels moves audio by ~100 seconds (2px = 1s ratio).
5. Pressing Play resumes audio AND begins auto-highlighting the active segment AND scrolls the transcript pane to keep the active line visible.
6. The Play button's label changes to Pause while sync is active.
7. Pressing Pause halts audio AND removes the active-highlight class (no segment is highlighted in pause/edit mode).
8. Clicking on a transcript line auto-pauses (audio stops, button reverts to Play, active-highlight is cleared).
9. Pressing Up arrow while focused inside the transcript moves focus to the previous segment's text cell, auto-pauses if needed.
10. Pressing Down arrow moves to the next segment's text cell.
11. Left and Right arrows still move the text caret inside a contenteditable cell (no segment navigation hijack).
12. Export TXT button triggers a browser file download.
13. Downloaded file contains lines of the form `[SPEAKER_00] 00:00:04 …`.
14. Export TXT respects unsaved in-memory edits (rename a speaker, click Export TXT before saving, downloaded file shows the new label).

`verify: human — operator runs items 1–14; all must pass.`

## Verification

- [ ] `getByteTimeDomainData` is present in `app.js`; `getByteFrequencyData` is absent.
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'getByteTimeDomainData' in p and 'getByteFrequencyData' not in p"`
- [ ] `drawWaveform` function present in `app.js`.
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'drawWaveform' in p"`
- [ ] Drag-to-scrub event wiring in place.
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'mousedown' in p and 'mousemove' in p and 'currentTime' in p"`
- [ ] TXT export endpoint + button tests pass (GET and POST forms; content type; `[SPEAKER]` lines).
      `verify: unit test in tests/diarizer/test_webapp_smoke.py`
- [ ] Sync-mode state machine identifiers present in `app.js`.
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'syncMode' in p and 'enterSync' in p and 'exitSync' in p and 'dropOutOfSync' in p"`
- [ ] Arrow-key segment navigation identifiers present in `app.js`.
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'ArrowUp' in p and 'ArrowDown' in p"`
- [ ] Full smoke test suite passes.
      `acceptance: pytest tests/diarizer/test_webapp_smoke.py -q`
- [ ] Manual UI walkthrough: operator runs affordance checklist items 1–14; all must pass.
      `verify: human — operator runs items 1–14; all must pass.`

## Executor Notes

*(Empty — to be populated by the executor.)*
