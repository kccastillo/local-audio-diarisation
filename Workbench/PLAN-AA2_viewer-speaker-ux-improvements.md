---
schema_version: 2
id: PLAN-AA2
title: Viewer speaker UX improvements — checkbox selection, popover, anchored dialogue,
  empty-row fix
type: plan
status: ready
assigned_to: sonnet
priority: medium
created: 2026-07-12
created_by: plan-writer
created_month: 202607
log_month: 202607
due: ''
repeatable: false
repeat_cadence: ''
linked_decisions: []
linked_inputs: []
blocked_by: ''
rollover_count: 0
triggers_plans: []
closes_thread: ''
advances_thread: ''
parent_plan_of_plans: ''
pipeline_phase: drafted
ideate_phase: complete
ideate_critique_addressed: []
ideate_iteration_count:
  self_critique: 0
  spec_refine: 0
ideate_reconcile_outcome: ''
tags:
- webapp
- viewer
- ui
- ux
- speaker-editing
files_touched:
- diarizer/webapp/static/app.js
- diarizer/webapp/static/index.html
- diarizer/webapp/static/style.css
substrate_files:
- diarizer/webapp/static/app.js
- diarizer/webapp/static/index.html
- diarizer/webapp/static/style.css
push_policy: manual
audit_acknowledgements: []
audit_disputes: []
audit_overrides: []
audit_extracted: null
pipeline_overrides: []
halt_log: []
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
  last_audit_commit: ''
  preferred_model_override: ''
verification_state:
  state_pass: 0
  state_fail: 0
  acceptance_pass: 0
  acceptance_fail: 0
  human_pending: []
  human_verdict: pending
  human_diagnostics: ''
  human_acknowledged_failures: []
  failure_logs: {}
  human_passed: false
---

## Objective

Improve the transcript viewer's speaker-editing and row-selection UX with four coupled front-end changes, reducing mouse travel and clicks during speaker cleanup and eliminating the dead-zone where an emptied row becomes unusable. Front-end only: no backend, API, or data-model changes. The `segments` data model and the `POST /api/transcript/save` payload shape are unchanged.

The four features are:

- **F1 — Multi-row selection and mass speaker assignment:** a checkbox per row enabling bulk speaker relabelling via a sticky bottom bar, without displacing the per-pill interaction.
- **F2 — Empty-row grabbability:** a CSS `min-height` + visible `:empty` outline on `.segment .text` so a fully-deleted row remains clickable and focusable. Pure CSS; no injected characters.
- **F3 — Fast speaker popover:** a compact popover anchored beside the pill, opened by right-click, replacing the current screen-centred modal for the common single-row rename case.
- **F4 — Anchored advanced dialogue:** the existing advanced dialogue converted from centred `showModal()` to an anchored non-modal `show()`, reachable via a "More…" link in the popover or via Shift+right-click / Alt+click on the pill.

## Context

### Current state (verified against substrate files; executor should re-read to confirm exact line numbers, which may drift)

All three target files live in `diarizer/webapp/static/`: `index.html`, `app.js`, `style.css`.

**Row layout:** rows are rendered by `renderTranscript()` in app.js (lines 293–365). Each row is a `<div class="segment">` laid out as a CSS grid in style.css (line 75): `grid-template-columns: 110px 70px 1fr` (speaker pill | timestamp | contentEditable text). `row.dataset.index = i` stores the segment index on the DOM node.

**Speaker change today:** RIGHT-click the speaker pill fires `contextmenu` → `openSpeakerModal(idx)` (app.js lines 318–322, function at lines 512–529), which populates and calls `$("speaker-modal").showModal()` on the native `<dialog id="speaker-modal">` (index.html lines 35–53). `showModal()` centres the dialog on screen — this is the mouse-fatigue problem for the common rename case. The Apply handler is at app.js lines 531–565 and contains a `#speaker-replace-all` checkbox (the only current global-rename op).

**Left-click pill:** fires `setSelectedSpeaker(seg.speaker)` for waveform speaker highlight (app.js lines 314–317). THIS MUST BE PRESERVED unchanged.

**Text editing:** the `.text` cell is `contentEditable`; its `input` handler syncs `segments[i].text = text.textContent` (app.js lines 339–342). BUG: when text is fully deleted, the `1fr` text cell collapses to zero height, leaving no click target to re-select the row and type back in.

**Speaker list:** `uniqueSpeakers(segs)` (app.js lines 160–164) derives the current speaker list from segments; the `speakers` global is recomputed on every `renderTranscript()`.

**No multi-row selection exists today.**

**Save flow:** `POST /api/transcript/save` with `{ segments }` (app.js lines 581–595). Payload shape MUST stay unchanged.

**Behaviours to PRESERVE:** waveform speaker-highlight on left-click pill (`setSelectedSpeaker`); sync-mode exit on edit/focus (`exitSync`); arrow-key row navigation (app.js lines 494–508); dirty-button tracking (`updateDirtyButtons()`); active-row highlight during playback (`tickHighlight()`); `restoreFocus()` behaviour after a speaker change.

### H4 — Calling-convention checklist

**Not triggered.** The Steps body involves browser-side vanilla JavaScript with no test-runner keywords (`pytest`, `unittest`, `async def`, `asyncio`), no third-party API client patterns, and no platform-specific async/sync boundary concerns. There is no automated JS test runner in this repo. Verification relies on `verify: human` browser checks and static Python/grep-based structural assertions on the static files. H4 checklist is omitted entirely per the conditional-skip rule.

### H2 — Capacity ceiling check

This PLAN introduces no MCP tools, schema tables, or context-window slice deliverables. No threshold in `.claude/skills/_shared/capacity-thresholds.md` is approached. H2 is clear.

## Design Decisions Classification

**Already locked** (operator pinned each via structured Q&A during the ideate cadence):

- **F1:** Multi-row selection via a CHECKBOX PER ROW (new left-most column), NOT shift/ctrl-click.
- **F1:** A DEDICATED MASS ACTION BAR (separate from the per-pill popover) that appears when >=1 row is ticked, showing "N rows selected" + an existing-speaker dropdown + a type-new text field + Apply. Apply sets the chosen speaker on every ticked row at once; affects ONLY ticked rows (no global/replace-all semantics).
- **F1:** After Apply, the selection is KEPT (ticks persist).
- **F1:** Ticked rows get a visible selected-state highlight but remain fully editable.
- **F1:** The mass action bar is a sticky strip at the BOTTOM of the viewport.
- **F2:** Empty-row grabbability via MIN-HEIGHT EMPTY BOX — CSS `min-height` + a visible outline on `.segment .text` when `:empty`. No placeholder words. Pure CSS; MUST NOT alter saved text (no injected characters).
- **F3:** A compact SPEAKER POPOVER anchored BESIDE the pill becomes the DEFAULT speaker-change path, opened by right-click (taking over the gesture that opens the modal today). Contents: existing-speaker dropdown + type-new field + Apply on Enter. NO global/replace-all. Single-row only.
- **F3/F4:** The advanced (large) dialogue is reachable BOTH via a "More…" link inside the popover AND via a modifier-click on the pill (Shift+right-click, or Alt+click).
- **F4:** The advanced dialogue opens ANCHORED NEAR the clicked pill as a non-modal panel, clamped to stay on-screen — NOT centred. All its current advanced options (incl. replace-all) are retained.

**Mechanically forced** (downstream consequences of locked decisions; no meaningful alternative):

- Adding the checkbox column forces `.segment` `grid-template-columns` to gain a leading column: `110px 70px 1fr` → `auto 110px 70px 1fr` (executor confirms exact width token; `auto` or a fixed small px value — pick to match existing row padding).
- The popover's existing-speaker dropdown and the mass-bar dropdown both reuse `uniqueSpeakers(segments)`.
- Converting the advanced dialogue from `showModal()` (centred, modal) to an anchored non-modal panel means replacing `showModal()` with `show()` (or a positioned custom element) plus anchor/clamp positioning math; the dialog must still be dismissable (Escape / click-away / Cancel).
- A shared anchor-and-clamp positioning helper is needed by both the popover (F3) and the advanced panel (F4); implement once, use twice.
- Selection state needs a client-side store (a `Set` of selected segment indices) that survives `renderTranscript()` re-renders, since re-render rebuilds all rows.

**Real judgement calls:** NONE remain — all resolved during ideation.

## Steps

1. **F2 — Empty-row grabbability (CSS only).** In `style.css`, add `min-height: 1.4em` to `.segment .text` and add a `.segment .text:empty` rule with a visible outline (e.g. `outline: 1px dashed var(--muted)`) so an emptied text cell keeps a visible, clickable bounding box. The `:empty` outline MUST NOT apply when the cell has content. Do NOT inject `content`, `::before`/`::after` pseudo-elements with characters, or `placeholder` attributes — saved text must be untouched. Verify the selector is literally `.segment .text:empty` (H8).

2. **F1 structural — Checkbox column and grid update.** In `style.css`, update `.segment { grid-template-columns: ... }` from `110px 70px 1fr` to `auto 110px 70px 1fr` (or a fixed small value matching existing row padding; executor's judgement). Add a `.segment .row-select` rule for the checkbox cell (alignment, no unnecessary padding). Add a `.segment.row-selected` rule for the selected-row highlight (e.g. a left-border accent or a tinted background distinct from `.segment.active`). In `renderTranscript()` in `app.js`, render `<input type="checkbox" class="row-select">` as the FIRST child of each row element (before the speaker pill). Read and restore `checked` state from `selectedRows` (the module-level `Set` introduced in Step 3) on each render so selection survives re-render. Ensure the checkbox `click` / `change` event does NOT propagate to row-body click or text-focus handlers.

3. **F1 state — Selection store.** At module level in `app.js`, declare `let selectedRows = new Set();`. Wire the checkbox `change` handler on each row: on check, add the segment index to `selectedRows` and add class `row-selected` to the row; on uncheck, delete from `selectedRows` and remove `row-selected`. After any `renderTranscript()` call that rebuilds rows, re-apply `checked` state and `row-selected` class from `selectedRows` so selection is durable across re-renders. Ensure ticking a row does NOT interfere with text editing (contentEditable focus), arrow-key navigation, or playback highlight (`tickHighlight()` keys off `.segment.active`, not `row-selected`).

4. **F1 UI — Mass action bar.** In `index.html`, add `<div id="mass-action-bar" class="hidden">` as a direct child of `<body>` (after `<main id="transcript">`). Contents: a `<span id="mass-selection-count">` for "N rows selected", a `<select id="mass-speaker-existing">` (existing speakers), an `<input id="mass-speaker-new" type="text">` (type-new field), and `<button id="mass-apply">Apply</button>`. In `style.css`, add `#mass-action-bar` as a `position: fixed; bottom: 0; left: 0; right: 0` strip with sufficient `z-index` to sit above rows. Add `#mass-action-bar.hidden { display: none; }`. Populate `#mass-speaker-existing` from `uniqueSpeakers(segments)` whenever the bar is shown. Show the bar when `selectedRows.size >= 1`; hide it when `selectedRows.size === 0`. Update `#mass-selection-count` text on every selection change.

5. **F1 behaviour — Mass apply handler.** In `app.js`, add a `click` listener on `#mass-apply`. On click: (a) resolve target speaker — `#mass-speaker-new` value (trimmed) takes precedence over `#mass-speaker-existing` value, mirroring the precedence logic in the existing `speaker-ok` handler (app.js lines 544–551); (b) if no target resolved, return early; (c) for every `idx` in `selectedRows`, set `segments[idx].speaker = target`; (d) call `recomputeCrosstalk()`, `refreshSelectedSpeakerRanges()`, `renderTranscript()`, `updateDirtyButtons()`; (e) KEEP `selectedRows` unchanged after apply (ticks persist); (f) re-show the bar with the updated count.

6. **Shared anchor+clamp positioning helper (supports F3 + F4).** In `app.js`, add a function `positionFloating(anchorEl, floatingEl)` that: reads `anchorEl.getBoundingClientRect()`, positions `floatingEl` beside the anchor (preferred: immediately to the right; fallback: left if right would overflow), and clamps the element within the current viewport bounds so it never clips. The function writes `floatingEl.style.left` and `floatingEl.style.top` directly and sets `floatingEl.style.position = "fixed"`. Used by both the popover (Step 7) and the advanced panel (Step 8).

7. **F3 — Fast speaker popover.** In `index.html`, add `<div id="speaker-popover" class="hidden">` as a direct child of `<body>`. Contents: a `<select id="popover-speaker-existing">`, an `<input id="popover-speaker-new" type="text" placeholder="new label">`, and a `<button id="popover-apply">Apply</button>`, plus a `<button id="popover-more">More…</button>` link. In `style.css`, style `#speaker-popover` as `position: fixed; z-index` above rows; add `#speaker-popover.hidden { display: none; }`. In `app.js`: (a) Replace the `contextmenu` binding on the speaker pill (currently calling `openSpeakerModal(i)`) with a new handler that populates `#popover-speaker-existing` via `uniqueSpeakers(segments)`, sets `#popover-speaker-new` to `""`, records the target row index in a module-level `let popoverRow = null`, calls `positionFloating(speaker, $("speaker-popover"))`, and removes the `hidden` class. (b) Wire `#popover-apply` click (and Enter keypress on `#popover-new`) to resolve the target speaker using the same precedence logic as Step 5, call `segments[popoverRow].speaker = target`, then call `recomputeCrosstalk()`, `refreshSelectedSpeakerRanges()`, `renderTranscript()`, `updateDirtyButtons()`, `restoreFocus()`, and hide the popover. (c) Wire Escape keydown and a document `click` listener (click-outside check) to hide the popover. (d) Wire `#popover-more` click to close the popover and open the advanced dialogue at the same anchor (Step 8). The popover has NO replace-all option — single-row only. Left-click behaviour on the pill (`setSelectedSpeaker`) is UNCHANGED.

8. **F4 — Anchored advanced dialogue + modifier entry.** In `app.js`: (a) Replace `$("speaker-modal").showModal()` in `openSpeakerModal()` with `$("speaker-modal").show()` followed by `positionFloating(anchorEl, $("speaker-modal"))` — where `anchorEl` is passed as a new second parameter to `openSpeakerModal(idx, anchorEl)` (callers: the "More…" button in the popover and the modifier-click handler below). (b) Add a modifier-click path on the speaker pill: detect Shift+contextmenu or Alt+click (executor's judgement on the exact event; avoid conflicts with browser-native Alt+click behaviour) to call `openSpeakerModal(i, speaker)` directly, bypassing the popover. (c) Ensure `$("speaker-modal")` is dismissable via Escape, Cancel button, and click-outside (add a document `click` listener that closes it when the click target is outside the dialog). (d) All existing options inside `#speaker-modal` (incl. `#speaker-replace-all`) are RETAINED — the advanced path keeps its full feature set. (e) In `style.css`, update the `dialog` rule to remove any browser-default centering (`margin: 0` or `top/left` explicit) so `positionFloating`'s fixed positioning takes effect, and set `max-width` to prevent overflow on narrow viewports.

9. **Regression sweep.** Explicitly re-verify all preserved behaviours. For each behaviour, note the file/function/line range and confirm it is unmodified or explicitly handled: (a) left-click pill → `setSelectedSpeaker` waveform highlight (unchanged path); (b) sync-mode exit on text edit/focus (`exitSync` called from `focus` handler — unchanged); (c) arrow-key row navigation (`keydown` handler — unchanged); (d) dirty-button tracking — `updateDirtyButtons()` called in every speaker-apply path (Steps 5, 7, 8) and on every text `input`; (e) active-row playback highlight — `tickHighlight()` keys off `.segment.active`, independent of `row-selected`; (f) save payload shape — `{ segments }` structure at app.js lines 581–595 is unmodified; no new fields added to `segments` array items.

## Verification

- [ ] F2 CSS present
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/style.css').read_text(encoding='utf-8'); sys.exit(0 if 'min-height' in s and ':empty' in s else 1)"`

- [ ] F1 checkbox + selection store present in app.js
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/app.js').read_text(encoding='utf-8'); sys.exit(0 if 'row-select' in s and 'selectedRows' in s else 1)"`

- [ ] Mass action bar element present in index.html
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/index.html').read_text(encoding='utf-8'); sys.exit(0 if 'mass-action-bar' in s else 1)"`

- [ ] Speaker popover element present in index.html
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/index.html').read_text(encoding='utf-8'); sys.exit(0 if 'speaker-popover' in s else 1)"`

- [ ] positionFloating helper present in app.js
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/app.js').read_text(encoding='utf-8'); sys.exit(0 if 'positionFloating' in s else 1)"`

- [ ] Advanced dialogue uses show() not showModal() for the primary open path (F4) — note: if showModal is legitimately retained for a non-centred use-case, annotate with a comment; the check below fails on any bare showModal call so executor must annotate or remove
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/app.js').read_text(encoding='utf-8'); sys.exit(0 if 'showModal(' not in s else 1)"`

- [ ] Grid template updated for checkbox column in style.css
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/style.css').read_text(encoding='utf-8'); sys.exit(0 if '110px 70px 1fr' not in s else 1)"`

- [ ] Save payload unchanged — segments array still POSTed with no extra fields
  `verify: python -c "import pathlib,sys; s=pathlib.Path('diarizer/webapp/static/app.js').read_text(encoding='utf-8'); sys.exit(0 if 'segments }' in s or '{ segments }' in s else 1)"`

- [ ] Acceptance — popover UX
  `acceptance: verify: human — Right-click a speaker pill opens the small popover anchored beside the pill (not centred on screen); picking/typing a speaker + Enter re-labels only that row; the popover closes; left-click on the same pill still highlights the speaker on the waveform without opening any popover.`

- [ ] Acceptance — mass action bar
  `acceptance: verify: human — Tick two or more row checkboxes; the bottom mass action bar appears showing the correct row count; choose/type a speaker + Apply re-labels exactly the ticked rows and the ticks remain after apply; unticking all rows hides the bar.`

- [ ] Empty-row fix
  `verify: human — Delete all text in a row; the row's text cell keeps a visible min-height box with a dashed outline and remains clickable and focusable so text can be typed back in; the dashed outline disappears once text is present.`

- [ ] Advanced dialogue anchor
  `verify: human — Shift+right-click (or Alt+click) a speaker pill opens the advanced dialogue anchored near the pill (not centred); the replace-all checkbox is present and functional; Escape, Cancel button, and click-outside all dismiss it.`

- [ ] Save round-trip
  `verify: human — After speaker edits and a text deletion+retype, Save succeeds; the saved transcript_edit JSON reflects the new speaker labels and the retyped text; no injected characters appear in rows that were emptied and retyped.`

## Executor Notes

**Executed:**
**Outcome:**
**What was done:**
**Blockers (if any):**
**Files modified:**
