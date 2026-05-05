---
title: "Webapp v2 affordances: pre-rendered waveform, TXT export, sync toggle, segment-nav, unified speaker modal"
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
  sufficiency_iterations: 4
  plan_safety_iterations: 1
  last_stage: plan_safety
  last_outcome: success
linked_decisions:
  - "Visualisation: server-side pre-rendered peak-amplitude waveform. Peaks computed eagerly at pipeline finalisation (or at session-bridge time) and persisted as <session-dir>/waveform_peaks.json — webapp loads it on init."
  - "Click on the waveform seeks audio AND auto-plays from that point (matches transcript-timestamp click for a single mental model). No drag affordance."
  - "Vertical playhead line drawn at audio.currentTime within the visible window."
  - "Zoom: 1× = whole file fits; zoom in by 2× steps up to 32×. Controls: +/- buttons in the player bar, plus Ctrl+wheel on the canvas."
  - "Pan-on-play (when zoomed): jump-pan model. When playhead crosses 75% of visible window, window jumps forward by 50%. No smooth scrolling — DAW-style."
  - "Play/Pause is now a SYNC toggle. Play resumes audio + enables sync (auto-highlight active line, auto-scroll transcript). Pause halts both."
  - "Auto-pause on edit: clicking any transcript line OR pressing Up/Down arrow keys pauses audio AND disables sync. Cursor lands on that segment for editing."
  - "Up/Down arrow keys navigate between adjacent segments. Left/Right arrows are LEFT ALONE — browsers use them for caret movement in contenteditable text."
  - "Export TXT format matches the existing pipeline's [SPEAKER] hh:mm:ss text format (one line per segment, hours always shown)"
  - "Clicking a segment's timestamp seeks audio to that point AND enters sync mode (auto-plays from there). Distinct from clicking the row body or speaker pill, which auto-pauses for editing."
  - "Speaker pill click opens a unified Speaker modal: pick from existing speakers OR type a new label, with a 'replace all instances of <current speaker>' checkbox for global rename. New labels typed here become first-class speakers (appear in the dropdown for future segments). The hidden Shift-R global-rename modal is removed — discoverability."
  - "Pre-existing sessions get lazy peak generation: if waveform_peaks.json is missing on a /api/waveform request, the backend computes it from source.opus and writes it back to the session dir. No backfill CLI subcommand needed."
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
---

## Objective

Add five affordances to the transcript-review webapp:
1. Replace the live spectrogram with a server-side pre-rendered amplitude waveform; click to seek+play; zoom in/out; auto-pan when zoomed and playing.
2. Add a TXT-export endpoint + button emitting the existing `[SPEAKER] hh:mm:ss text` format.
3. Refactor play/pause into a SYNC toggle: Play resumes audio AND enables auto-highlight + auto-scroll-to-active-line; Pause halts both.
4. Auto-pause on edit: clicking any transcript line (body / speaker pill / text) OR pressing Up/Down arrows pauses audio AND disables sync, focusing that segment for editing. Up/Down navigates between adjacent segments. Left/Right are left alone for caret movement.
5. Unified Speaker modal: clicking a speaker pill opens a single dialog with existing-speaker dropdown + new-speaker text field + replace-all checkbox; new labels become first-class speakers in subsequent dropdowns.

## Context

**What exists today.**
- Webapp under `diarizer/webapp/` — FastAPI backend (`app.py`), vanilla JS frontend (`static/app.js`, `index.html`, `style.css`). Built in `Retired/202605060200_PLAN_transcript-review-webapp.md`; decisions D1–D8 from that PLAN remain in force.
- Spectrogram is rendered each frame from a Web Audio AnalyserNode (FFT 1024, viridis colour ramp, 1px-per-frame horizontal scroll). Operator feedback: hard to read amplitude.
- Active-segment highlight follows the playhead unconditionally (no concept of "sync mode" — it just always tracks).
- Save and Diff buttons exist; no TXT export.
- No keyboard navigation between segments. Up/Down do nothing in the current frontend.
- Audio source is 16 kHz mono Opus (locked D6 in `Retired/202605060200_PLAN_transcript-review-webapp.md`). Stereo channel-split visualisation is therefore not available.

**What is being changed.**
- Visualisation swap: server-side pre-rendered peak-amplitude waveform replaces the live AnalyserNode spectrogram. Peaks are computed at pipeline finalisation, stored as `waveform_peaks.json`, loaded by the webapp on init, rendered to a canvas once (re-drawn on zoom/pan changes). No Web Audio graph needed. Pre-existing sessions (without peaks files) lazily generate the file on first webapp request — no manual backfill required.
- Click-to-seek-and-play on the waveform canvas. No drag handler.
- Zoom controls (buttons + Ctrl+wheel) and jump-pan model for auto-panning while playing at zoom > 1×.
- New endpoint `GET /api/transcript/export.txt` returns `text/plain` content with one line per segment in `[SPEAKER] hh:mm:ss text` format. Frontend "Export TXT" button triggers a browser download of the current in-memory edit state.
- Frontend state machine refactor: introduce a single `syncMode` boolean. Play handler sets `syncMode = true`, scrolls active row into view, calls `audio.play()`. Pause handler sets `syncMode = false`, calls `audio.pause()`. Active-highlight logic only runs when `syncMode` is true. Clicking any segment row OR pressing Up/Down (when focus is anywhere in the transcript) calls a `dropOutOfSync()` helper that pauses + disables sync + focuses the target row's text cell. Up/Down move focus to the previous/next segment row's text cell.

**Decisions locked this session (Human, 2026-05-06):**
- Visualisation: server-side pre-rendered peak-amplitude waveform (operator redirected from live AnalyserNode waveform with drag).
- Zoom: 1×–32× by 2× steps; jump-pan (DAW-style) when playing at zoom > 1×.
- Click-to-seek-and-play; no drag affordance.
- Up/Down navigate segments; Left/Right untouched.
- TXT export format: identical to existing pipeline's TXT writer (`[SPEAKER] hh:mm:ss text`).

**Out of scope.**
- Segment boundary editing (still deferred).
- Frequency-domain visualisation (operator chose amplitude-only).
- In-session undo/redo stack (still deferred).

## Steps

### Step 1 — Server-side pre-rendered waveform peaks

**Goal:** generate a per-bin peak-amplitude array from `source.opus` and persist it as `<session-dir>/waveform_peaks.json` so the webapp can render the full file's waveform instantly without any Web Audio analysis.

**Files touched:**
- `diarizer/webapp/peaks.py` (new) — peak-extraction logic.
- `diarizer/cli.py` — wire the peaks generator into the `_run` finalisation block (after Opus encode, before "Session dir:" print).
- `scripts/build_session_from_txt.py` — same wiring so bridged sessions also get peaks.
- `diarizer/webapp/app.py` — small endpoint addition (see Step 2).

**Library choice:** use `soundfile` (already in `requirements.txt`) to decode the Opus file. `soundfile.read(path)` returns `(samples, sample_rate)` as a numpy array. Mono so `.ndim == 1`. Compute peaks via `numpy.abs(samples).reshape(num_bins, -1).max(axis=1)` then normalise by max-abs to [0, 1].

**Defensive mono squash** — D6 locks the source as mono Opus, but soundfile's behaviour on accidentally-stereo input would silently corrupt the peak array. The mean-axis-1 squash is cheap insurance. In `peaks.compute_peaks`, before reshape: `if samples.ndim > 1: samples = samples.mean(axis=1)`.

**Bin count:** target 4000 bins for any duration ≤ 60 minutes (≈90ms per bin at 1 hour, ≈14ms at 5 min — fine resolution at all zoom levels up to 32×). For very short audio (< 30 s), drop to fewer bins so each bin holds at least 100 samples.

**Output schema:** `waveform_peaks.json`:
```
{
  "version": 1,
  "bins": <int>,
  "duration_s": <float>,
  "peaks": [<float>, ...]   // length == bins, each value in [0, 1]
}
```

**API:** `peaks.compute_peaks(opus_path: Path, target_bins: int = 4000) -> dict` returns the schema above. `peaks.write_peaks(opus_path, out_json_path, target_bins=4000) -> Path` writes it.

`verify:` `python -c "from diarizer.webapp.peaks import compute_peaks; from pathlib import Path; import subprocess; tmp = Path('temp/_verify_peaks.opus'); tmp.parent.mkdir(parents=True, exist_ok=True); subprocess.run(['ffmpeg','-y','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','2','-c:a','libopus','-b:a','16k',str(tmp)], check=True, capture_output=True); d = compute_peaks(tmp); assert d['bins'] == len(d['peaks']) > 0; assert d['duration_s'] >= 1.5; assert all(0.0 <= p <= 1.0 for p in d['peaks']); print('OK')"`

### Step 2 — Frontend waveform render with click-to-seek + zoom + jump-pan

**Goal:** draw the pre-rendered peaks to a canvas, draw a vertical playhead line, support click-to-seek-and-play, support zoom (1×–32× by 2× steps), and pan automatically when zoomed and playing.

**Files touched:** `diarizer/webapp/static/{index.html,app.js,style.css}`, `diarizer/webapp/app.py`.

**Backend endpoint:** `GET /api/waveform` returns the contents of `<session-dir>/waveform_peaks.json` with `Content-Type: application/json`. If the file is missing, attempt lazy generation:
- If `source.opus` exists → call `compute_peaks(source.opus, target_bins=4000)`, write the result to `<session-dir>/waveform_peaks.json` (best-effort; tolerate write-permission failures by serving the in-memory result), then return it. Add a structured log line: `logger.info("Lazily generated waveform_peaks.json for %s", session_dir)`.
- If `source.opus` is also missing → return 404 with `detail="source.opus missing — cannot generate waveform peaks"`.

**State held in app.js:**
- `peaks: Float32Array | null` — the peak array, length `bins`.
- `peaksDuration: number` — total audio length in seconds.
- `zoomLevel: number` — integer in {1, 2, 4, 8, 16, 32}; 1 = full file fits.
- `windowStartTime: number` — left edge of visible window in seconds.

**Visible window math:**
- `visibleSeconds = peaksDuration / zoomLevel`
- `windowEndTime = windowStartTime + visibleSeconds`
- A given `t` maps to canvas X via `(t - windowStartTime) / visibleSeconds * canvas.width`.

**Render:**
- Per-pixel column: pick the peak bin whose midpoint time is closest to that column's time. Draw a vertical line centred on `canvas.height / 2`, height = `peak * canvas.height`, single accent colour (`#5fd97a` or chosen accent).
- Vertical white playhead line at the X position of `audio.currentTime` (only drawn when `windowStartTime ≤ currentTime ≤ windowEndTime`).
- Redraw via requestAnimationFrame whenever the audio is playing OR the window/zoom changes.

**Click-to-seek-and-play:**
- `mousedown` → `clickT = windowStartTime + (e.offsetX / canvas.width) * visibleSeconds`.
- Set `audio.currentTime = clickT`. Call `enterSync()` (defined in Step 4) to start playing + enable sync.

**Zoom:**
- Two buttons in the player bar: `Zoom +`, `Zoom -`.
- Zoom in: `zoomLevel = min(32, zoomLevel * 2)`. Re-centre the visible window on `audio.currentTime` (so the zoom focuses on the current playhead).
- Zoom out: `zoomLevel = max(1, zoomLevel / 2)`. Re-centre similarly. At zoomLevel=1, `windowStartTime = 0`.
- `Ctrl+wheel` on the canvas: deltaY < 0 → zoom in; deltaY > 0 → zoom out. Centre on the cursor's time, not the playhead.
- Display the current zoom level as text next to the buttons (e.g., `1×`, `2×`, `4×`).
- All `windowStartTime` mutations must clamp >= 0 (lower bound) AND <= `peaksDuration - visibleSeconds` (upper bound). Add explicit `windowStartTime = Math.max(0, windowStartTime)` after every zoom-recentre and pan calculation.

**Jump-pan when playing + zoomed:**
- Each frame, if `zoomLevel > 1` AND `audio.paused === false`:
  - If `audio.currentTime > windowStartTime + visibleSeconds * 0.75` → `windowStartTime = audio.currentTime - visibleSeconds * 0.25` (jump forward by 50% of the visible window; new window has the playhead at 25% from the left).
  - Clamp `windowStartTime + visibleSeconds <= peaksDuration`; if it would exceed, set `windowStartTime = peaksDuration - visibleSeconds`.

**No drag handler at all.** The operator's redirected design uses click-only navigation.

**No AnalyserNode.** Strip out `audioCtx`, `analyser`, `source`, `ensureAudioCtx()`, the `getByteFrequencyData`/`getByteTimeDomainData` rendering. The `<audio>` element handles playback directly; volume is controlled via `audio.volume`. **This is a simplification** — Web Audio gesture-policy concerns disappear.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'AnalyserNode' not in p and 'createMediaElementSource' not in p and 'getByteFrequencyData' not in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'zoomLevel' in p and 'windowStartTime' in p and 'jumpPan' in p.replace(' ','').lower() or 'pan' in p.lower()"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'peaks' in p and '/api/waveform' in p"`
`verify:` `python -c "import json, tempfile, shutil, subprocess; from pathlib import Path; from diarizer.webapp.app import create_app; from httpx import ASGITransport, AsyncClient; import asyncio; tmp = Path(tempfile.mkdtemp()); subprocess.run(['ffmpeg','-y','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','2','-c:a','libopus','-b:a','16k', str(tmp/'source.opus')], check=True, capture_output=True); (tmp/'transcript.json').write_text('{\"segments\":[]}'); (tmp/'session.json').write_text('{\"version\":1,\"source_basename\":\"x\",\"created\":\"\",\"audio_opus\":\"source.opus\",\"audio_wav\":\"source.wav\",\"transcript_original\":\"transcript.json\",\"edits\":[]}'); app = create_app(tmp); transport = ASGITransport(app=app); async def go(): async with AsyncClient(transport=transport, base_url='http://127.0.0.1') as c: r = await c.get('/api/waveform'); return r; r = asyncio.run(go()); assert r.status_code == 200; assert (tmp/'waveform_peaks.json').exists(); print('OK lazy-gen')"`

### Step 3 — TXT export endpoint + button

- File: `diarizer/webapp/app.py`. New endpoint `GET /api/transcript/export.txt?from=<filename>` (the `from` parameter is optional; default = latest edit, or original if no edits). Returns `text/plain; charset=utf-8` with `Content-Disposition: attachment; filename=<source_basename>_export_<YYYYMMDD>.txt`.
- Format: one line per segment, `[SPEAKER] hh:mm:ss text` — hours always shown (matching existing `output.py:_format_timestamp`). Reuse the same formatter logic; do NOT call into `diarizer.output` directly to avoid coupling — copy the small `_format_timestamp` helper into `app.py`. Add a `# keep in sync with diarizer/output.py:_format_timestamp` comment.
- Edge cases: missing speaker → omit `[<speaker>]` prefix (do NOT include `[ ]` brackets — `write_txt` builds prefix from a `prefix_parts` list, so missing speaker yields just the timestamp prefix or empty prefix). Missing text → emit just the prefix (no trailing space). Mirror `diarizer/output.py:write_txt` — format the line as `f'{prefix} {text}'.strip()` exactly to keep the two writers aligned. Reference: `diarizer/output.py:write_txt` lines 66-79 (formatting loop at 70-77) — emits `f"{prefix} {seg.text}".strip() if prefix else seg.text`; the webapp formatter must produce the same bytes.
- The byte-identity test in Step 6 is the canonical enforcement of this decision — if the pipeline's `write_txt` format ever changes, this test will fail and force the webapp's formatter to be updated in lockstep.
- File: `diarizer/webapp/static/index.html`. Add `<button id="btn-export" type="button">Export TXT</button>` next to Save.
- File: `diarizer/webapp/static/app.js`. Wire `btn-export` to POST the current in-memory edit state as the body of a new `POST /api/transcript/export.txt` (operator may have unsaved edits — we don't want to require saving first). Backend: accept optional JSON body containing `{segments: [...]}`; if absent, fall back to the latest saved sidecar.
- Browser-download trick: receive the response as a Blob, create an object URL, programmatic `<a href download>` click.

`verify:` unit test in `tests/diarizer/test_webapp_smoke.py` exercises the endpoint with both the GET (latest-edit) form and POST (in-memory body) form, asserts `text/plain` content type and `[SPEAKER]` lines.
`verify:` `pytest tests/diarizer/test_webapp_smoke.py::test_export_txt_matches_write_txt_byte_for_byte -q`

### Step 4 — Play/Pause as sync toggle + auto-pause-on-edit

- File: `diarizer/webapp/static/app.js`.
- Introduce module-scoped `let syncMode = false;`.
- New helpers:
  - `enterSync()` — sets `syncMode = true`, calls `audio.play()`, button text → "Pause", scrolls active row (if any) into view via `row.scrollIntoView({block: 'nearest'})`. Note: AudioContext is no longer in play — Step 2's redesign uses the pre-rendered waveform, removing the analyser graph entirely. `enterSync()` is just `audio.play()` + button label update + `scrollIntoView`.
  - `exitSync()` — sets `syncMode = false`, calls `audio.pause()`, button text → "Play".
  - `dropOutOfSync(targetIndex?)` — calls `exitSync()`; if `targetIndex` provided, focuses that row's `.text` element and places cursor at end.
- Refactor existing play/pause click handler to toggle between `enterSync` and `exitSync`.
- Refactor `tickHighlight` to early-return when `syncMode === false`. Active-highlight class is removed at the moment sync is exited (visual cue for current state is via the button label, not via the highlight).
- Click handlers on segment elements split by target:
  - **Timestamp click** → set `audio.currentTime = seg.start`, then call `enterSync()`. This both seeks AND auto-plays from that point. (No `dropOutOfSync` call — the timestamp is a 'play from here' action.)
  - **Speaker pill / row body / text cell click** → `dropOutOfSync(rowIndex)` (auto-pause, focus that row's text for editing). Unchanged from prior behaviour.
- Keyboard: `keydown` listener on `document`. If `key === 'ArrowUp'` or `key === 'ArrowDown'` AND the active element is inside `#transcript`, prevent default, compute target index (`current ± 1`, clamped), call `dropOutOfSync(target)`. Do NOT intercept arrow keys when focus is outside `#transcript` (e.g. inside the volume slider).
- Verify Left/Right are NOT bound — keep contenteditable cursor movement intact.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'syncMode' in p and 'enterSync' in p and 'exitSync' in p and 'dropOutOfSync' in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'ArrowUp' in p and 'ArrowDown' in p"`

### Step 5 — Unified Speaker modal

**Goal:** consolidate per-segment reassign + global rename + new-speaker creation into a single affordance triggered by clicking any speaker pill. Remove the Shift-R global-rename keybinding (undiscoverable).

**Files touched:** `diarizer/webapp/static/index.html`, `diarizer/webapp/static/app.js`, `diarizer/webapp/static/style.css`.

**HTML changes (`index.html`):**
- Delete the existing `<dialog id="rename-modal">` (the Shift-R modal).
- Replace the existing `<dialog id="reassign-modal">` with a new `<dialog id="speaker-modal">` containing:
  - A `<select id="speaker-existing">` populated from current unique speakers — first option is "(new label below)" sentinel for the empty/new-label case.
  - An `<input id="speaker-new" type="text" placeholder="or type new label">` — when non-empty, takes precedence over the dropdown selection.
  - A `<label><input id="speaker-replace-all" type="checkbox"> Replace all segments currently labelled "<current>"</label>` — text shows the segment's current speaker name dynamically, defaults UNCHECKED.
  - Submit + Cancel buttons.

**JS changes (`app.js`):**
- Remove the `Shift-R` global keydown listener and the `openRenameModal()` / `rename-ok` handler entirely.
- Replace `openReassignModal(idx)` with a new `openSpeakerModal(idx)` that:
  - Captures the current segment's speaker into a `currentSpeaker` variable.
  - Rebuilds `#speaker-existing` from `uniqueSpeakers(segments)` plus a leading `<option value="">(new label below)</option>` sentinel.
  - Pre-selects the current speaker in the dropdown.
  - Clears `#speaker-new` to empty.
  - Updates the `replace-all` checkbox label text to include the current speaker, e.g. `Replace all segments currently labelled "SPEAKER_00"`.
  - Resets the checkbox UNCHECKED.
  - `showModal()`.
- Submit handler:
  1. Resolve target label: `newLabel = speakerNew.value.trim() || speakerExisting.value`. If both empty, do nothing (close modal).
  2. **Uniqueness rule:** if `speakerNew.value.trim()` is non-empty, normalise (trim) and check it is not already an existing speaker (case-sensitive match against `uniqueSpeakers(segments)`). If it collides, surface a toast `"Speaker '<X>' already exists — pick from the dropdown"` and re-open the modal. (Do NOT silently swallow — the operator's intent matters.)
  3. If `replace-all` is CHECKED → walk all segments where `seg.speaker === currentSpeaker` and set their speaker to `newLabel`. (Global replacement.)
  4. If UNCHECKED → set only the clicked segment's speaker to `newLabel`. (Per-segment reassign.)
  5. Re-render transcript. New label automatically appears in the next modal's dropdown via `uniqueSpeakers(segments)`.
  6. After the modal closes (whether Submit or Cancel), call `row.querySelector('.text').focus()` where `row` is the row that opened the modal. Reason: `<dialog>.showModal()` moves focus into the dialog and browsers do NOT restore focus to a contenteditable on close — without explicit restoration, Up/Down arrow navigation breaks after the modal closes.
- The speaker pill click handler now calls `openSpeakerModal(rowIndex)` (not `openReassignModal`). Per the auto-pause-on-edit rule from Step 4: clicking the speaker pill ALSO triggers `dropOutOfSync(rowIndex)` before the modal opens.

**CSS (`style.css`):** ensure the new modal's spacing is consistent with the existing dialog styles. No new colour additions.

`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'openSpeakerModal' in p and 'openRenameModal' not in p and 'openReassignModal' not in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/index.html', encoding='utf-8').read(); assert 'speaker-modal' in p and 'rename-modal' not in p and 'reassign-modal' not in p"`
`verify:` `python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert '.text\\').focus()' in p or '.text\").focus()' in p or 'querySelector(' in p and '.focus()' in p"`

### Step 6 — Smoke tests for the new endpoints + frontend invariants

- File: `tests/diarizer/test_webapp_smoke.py`. Append:
  - `test_export_txt_get_returns_plaintext` — GET `/api/transcript/export.txt`, asserts 200, `text/plain` content type, then loads `tests/diarizer/conftest.py`'s `stub_session` and reads the fixture's `transcript.json` to extract the actual speaker labels, asserts one of those labels appears in the export body. (This pins the assertion to whatever the fixture actually emits — avoids hidden coupling to a hardcoded `SPEAKER_00`.)
  - `test_export_txt_post_uses_body_segments` — POST with `{segments: [{start, end, text, speaker}]}` containing a custom speaker label, asserts that label appears in the response (proves the body wins over saved files).
  - `test_export_txt_format_includes_timestamps` — assert lines match a regex `^\[.+\] \d{2}:\d{2}:\d{2} `.
  - `test_waveform_endpoint_returns_peaks` — pre-create a `waveform_peaks.json` in the stub session (small array, 100 bins, all 0.5), GET `/api/waveform`, assert 200 + JSON shape (`bins`, `duration_s`, `peaks` keys present).
  - `test_waveform_404_when_missing` — fixture session has neither `waveform_peaks.json` nor `source.opus`; GET `/api/waveform`, assert 404 with `detail` containing `"source.opus missing"`.
  - `test_waveform_lazy_generates_when_missing` — fixture session has `source.opus` but no `waveform_peaks.json`; first `GET /api/waveform` returns 200 with peaks AND writes `waveform_peaks.json` to disk; second `GET /api/waveform` returns 200 (now from disk).
  - `test_export_txt_matches_write_txt_byte_for_byte` — construct an in-memory `TranscriptionResult`-shaped object with 3–4 segments (mix of speakers, mix of speaker-present and speaker-empty); run `diarizer.output.write_txt(result, tmp/'a.txt', config_with_speakers_and_timestamps)` to get the pipeline's bytes; POST the same segments to the webapp's `/api/transcript/export.txt`, capture the response bytes; assert the two byte sequences are identical.

`acceptance:` `pytest tests/diarizer/test_webapp_smoke.py -q`

### Step 7 — Manual UI walkthrough (verify: human)

Operator-visible affordance checklist for the new behaviours. Each item must be confirmed working in Chrome and Firefox. Any failure → outcome-verification reverts to drafted.

1. Page loads with a waveform fully visible across the canvas (1× zoom). Amplitude variation is clearly visible (peaks during speech, troughs during silence).
2. A vertical playhead line is visible on the waveform, anchored at the current `audio.currentTime`.
3. Clicking anywhere on the waveform seeks audio to that position AND begins playback automatically (button shows Pause).
4. The Zoom + button doubles the zoom level (max 32×); the visible window narrows around the current playhead.
5. The Zoom - button halves the zoom level (min 1×); at 1× the entire file is visible.
6. Holding Ctrl and scrolling the mouse wheel over the canvas zooms in/out; zoom centres on the cursor's time.
7. When zoomed (e.g., 8×) and playing, the visible window auto-pans forward — when the playhead reaches ~75% of the right edge, the window jumps forward by 50%.
8. Pressing Play resumes audio AND enables sync mode (active segment highlighted, transcript pane scrolls to keep active line visible).
9. Pause stops audio AND clears the active-highlight (no segment is highlighted in pause/edit mode).
10. Clicking on a transcript line (body / speaker pill / text) auto-pauses (audio stops, button reverts to Play, highlight cleared).
11. Clicking a segment's start-time timestamp seeks audio AND auto-plays from that point.
12. Pressing Up/Down arrow inside the transcript moves focus to the previous/next segment's text cell, auto-pausing if needed.
13. Left/Right arrows still move the text caret inside a contenteditable cell (no segment-nav hijack).
14. Export TXT button triggers a browser file download.
15. Downloaded file contains lines of `[SPEAKER_00] 00:00:04 …` form. (Use whatever speaker label the current transcript actually has — check before assertion.)
16. Export TXT respects unsaved in-memory edits (rename a speaker, click Export TXT before saving — downloaded file shows the new label).
17. Clicking any speaker pill opens the unified Speaker modal. Dropdown shows every current speaker; current segment's speaker is preselected. Replace-all checkbox is unchecked by default and shows the current speaker's name in its label.
18. Typing a new label + Submit (replace-all UNCHECKED) changes only that segment's speaker. The new label appears in the dropdown the next time the modal opens.
19. Selecting an existing speaker + Submit (replace-all UNCHECKED) reassigns only that segment.
20. Selecting a target + ticking replace-all + Submit replaces every segment with the same original speaker. Confirm by scrolling — no occurrences of the old label remain.
21. Typing a new label that collides (case-sensitive) with an existing speaker → Submit shows a "Speaker already exists" toast; modal stays open.
22. Typing a new label with leading/trailing whitespace → Submit trims it before applying. Whitespace alone or empty-with-empty-dropdown selection → Submit is a no-op (modal closes, no change).
23. After the Speaker modal closes (Submit or Cancel), focus returns to the originating row's text cell — Up/Down arrow navigation still works.
24. The Shift-R keybinding does nothing (the old global-rename modal is gone).

`verify: human — operator runs items 1–24; all must pass.`

## Verification

- [ ] No AnalyserNode, `createMediaElementSource`, or `getByteFrequencyData` in `app.js`. (Step 2)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'AnalyserNode' not in p and 'createMediaElementSource' not in p and 'getByteFrequencyData' not in p"`
- [ ] Zoom state and pan logic present in `app.js`. (Step 2)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'zoomLevel' in p and 'windowStartTime' in p and 'jumpPan' in p.replace(' ','').lower() or 'pan' in p.lower()"`
- [ ] Peak loading and waveform endpoint referenced in `app.js`. (Step 2)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'peaks' in p and '/api/waveform' in p"`
- [ ] Peaks compute function passes contract assertions. (Step 1)
      `verify: python -c "from diarizer.webapp.peaks import compute_peaks; from pathlib import Path; import subprocess; tmp = Path('temp/_verify_peaks.opus'); tmp.parent.mkdir(parents=True, exist_ok=True); subprocess.run(['ffmpeg','-y','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','2','-c:a','libopus','-b:a','16k',str(tmp)], check=True, capture_output=True); d = compute_peaks(tmp); assert d['bins'] == len(d['peaks']) > 0; assert d['duration_s'] >= 1.5; assert all(0.0 <= p <= 1.0 for p in d['peaks']); print('OK')"`
- [ ] TXT export endpoint + button tests pass (GET and POST forms; content type; speaker lines). (Step 3)
      `verify: unit test in tests/diarizer/test_webapp_smoke.py`
- [ ] Sync-mode state machine identifiers present in `app.js`. (Step 4)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'syncMode' in p and 'enterSync' in p and 'exitSync' in p and 'dropOutOfSync' in p"`
- [ ] Arrow-key segment navigation identifiers present in `app.js`. (Step 4)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'ArrowUp' in p and 'ArrowDown' in p"`
- [ ] Unified Speaker modal identifiers present — `openSpeakerModal` in `app.js`; `speaker-modal` in `index.html`; old `rename-modal`, `reassign-modal`, `openRenameModal`, `openReassignModal` absent. (Step 5)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert 'openSpeakerModal' in p and 'openRenameModal' not in p and 'openReassignModal' not in p"`
      `verify: python -c "p = open('diarizer/webapp/static/index.html', encoding='utf-8').read(); assert 'speaker-modal' in p and 'rename-modal' not in p and 'reassign-modal' not in p"`
- [ ] Focus restoration after Speaker modal present in `app.js`. (Step 5)
      `verify: python -c "p = open('diarizer/webapp/static/app.js', encoding='utf-8').read(); assert '.text\\').focus()' in p or '.text\").focus()' in p or 'querySelector(' in p and '.focus()' in p"`
- [ ] Waveform endpoint, lazy-generation, and 404-when-missing tests pass. (Step 6)
      `verify: pytest tests/diarizer/test_webapp_smoke.py::test_waveform_endpoint_returns_peaks tests/diarizer/test_webapp_smoke.py::test_waveform_404_when_missing tests/diarizer/test_webapp_smoke.py::test_waveform_lazy_generates_when_missing -q`
- [ ] TXT export byte-identity test passes. (Step 6)
      `verify: pytest tests/diarizer/test_webapp_smoke.py::test_export_txt_matches_write_txt_byte_for_byte -q`
- [ ] Full smoke test suite passes. (Step 6)
      `acceptance: pytest tests/diarizer/test_webapp_smoke.py -q`
- [ ] Manual UI walkthrough: operator runs affordance checklist items 1–24; all must pass. (Step 7)
      `verify: human — operator runs items 1–24; all must pass.`

## Executor Notes

*(Empty — to be populated by the executor.)*
