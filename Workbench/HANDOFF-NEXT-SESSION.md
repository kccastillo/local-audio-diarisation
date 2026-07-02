---
created: 2026-07-02
last_updated: 2026-07-02
scope: NEXT-SESSION
author: Claude (pre-plan_foundry-install session)
---

# Handoff — diarizer webapp/packaging work + project-context preservation

*plan_foundry has been installed over a bespoke `Bus/`-based (older-plan_foundry) harness and the swap is COMPLETE (bundle sha 03dd46a2). This file carries forward (a) four work requests to turn into PLANs, (b) the project-only parts of the old `CLAUDE.md`, and (c) project memories/conventions. If you are reading this post-restart via `rehydrate-handoff`: the harness is ready — proceed to author the PLANs in Next steps.*

## Next steps

**Workflow (locked 2026-07-02):** Ken runs **one conversation per PLAN**. This handoff is the **living ledger** — its status markers are updated incrementally as each PLAN is authored / executed, so every new conversation opens with an accurate what's-done / what's-left picture. Do NOT retire this handoff until the whole slate is complete.

**Status legend:** `scoped` = requirement agreed · `PLAN authored` = written to Workbench/, awaiting its execution conversation · `executing` · `complete`.

Slate: A, B, C, D (webapp/packaging), then E (orphans tie-off — runs last, per Ken "after all of this"). Author each via plan_foundry's `write-plan` / `plan-pipeline`.

1. **[A — scoped ✅ · PLAN authored → `Workbench/PLAN-AA0_just-run-packaging.md`] Package the pipeline to "just run" against source audio.**
   - Human's intent: "packaged to 'just run' and be aimed at some kind of source audio", output to the usual project folders in the current project.
   - Current reality: one file at a time — `python -m diarizer.cli run --input <file> --config config/config.yaml`. During this session, batching two `./input/*.m4a` files was fully manual (one background run each), and Windows console encoding choked on ` ` (narrow no-break space) in filenames.
   - **Requirement agreed with Ken (2026-07-02):** (a) drag an audio file onto a launcher → processes that **one** file (not batch); (b) `config/config.yaml` becomes the **built-in default** so `--config` is never required (closes the silent-no-diarisation footgun); (c) if the file already has completed output, **skip it** — `--force` reprocesses.
   - **Mechanism (in PLAN-AA0):** `DEFAULT_CONFIG` resolved from `__file__` in `diarizer/cli.py`; `find_completed_session` skip-helper in `diarizer/session.py`; `--force` flag; new `Transcribe.cmd` drag-drop launcher at project root; README note.
   - **Next:** open a dedicated conversation, run `plan-pipeline` against PLAN-AA0 (review → execute → verify). Human-verify item needs a real `.m4a` dragged on Windows. Item A's earlier Blocking-decision (the "just run" scope) is now RESOLVED.

2. **[B — scoped] Right-click speaker reassignment on the name tag.**
   - Human's words: the central name-change dialog is in the middle while name tags are on the left, so fixing mis-attributions one-by-one is tedious. Wants: right-click on the name → change the speaker for THAT line, offering a dropdown of all existing names PLUS an option that launches the primary central dialog.
   - Code grounding: `diarizer/webapp/static/app.js:318-322` — the name pill ALREADY binds `contextmenu` → `openSpeakerModal(i)` (jumps straight to the full central modal). The request is to insert a lightweight context menu first: quick-pick list of `uniqueSpeakers(segments)` for one-click single-line reassignment, plus a "More…/Rename…" entry that opens the existing modal (`openSpeakerModal`, app.js:512). Single-line reassign already exists as the modal's non-replace-all path (`segments[idx].speaker = target`, app.js:558).

3. **[C — scoped] Deleted line becomes hard to re-select for input.**
   - Human's words: if the whole dialogue line is deleted, it's difficult to select that line again to type into; would like the whole line space to be a box that can receive input.
   - Code grounding: rows render an editable `<div class="text" contentEditable>` (app.js:334-338). When emptied, the contentEditable collapses to near-zero height so there's almost nothing to click. Row-body click routes to `dropOutOfSync(i)` which focuses the text cell (app.js:351-358, 457-475). Likely fix lives in `style.css` (min-height/padding on `.text` and/or making the whole `.segment` row a click target that focuses `.text`).

4. **[D — scoped, needs root-cause] Caret jumps to end of line on click/input.**
   - Human's words: when selecting a dialogue line and clicking at specific words, the active cursor sometimes jumps to the final character of the line on input — very annoying.
   - Code grounding: `dropOutOfSync` deliberately places the caret at the END via `range.collapse(false)` (app.js:465-472); it's meant to fire only for row-body / arrow-nav entry, NOT for direct clicks inside the text (see the guard + comment at app.js:351-356). The bug is that clicking specific words still lands the caret at end — investigate whether the row/focus handlers are stealing the native click caret, or whether an `input`-triggered re-render is resetting selection. Reproduce first; do not guess the fix.

5. **[E — to scope · runs LAST, per Ken "after all of this"] Tie off the harness-swap orphans.**
   - Leftovers from the plan_foundry swap that still need a decision + cleanup, gathered here so nothing is lost:
     - **`.claude/_retired_old_harness/`** — 8 superseded old skills + `maintenance-agent.md`, moved aside (not deleted) during the swap. Delete once confirmed nothing is missed.
     - **`Bus/`** — git-tracked old PLAN history (the pre-plan_foundry audit trail). Decide: migrate anything still useful into `Workbench/`, then retire — do NOT bulk-delete.
     - **`.claude/CONSTITUTION.md`** — its Plan-lifecycle / Skill-registry sections are now stale (plan_foundry owns those); prune them but KEEP the working-style substrate (Discussion-vs-work-order, Review shape, Bash compounds, Decision autonomy) that CLAUDE.md still references.
     - **`plan_foundry/` clone at project root** — the source bundle used for install; likely removable now the bundle lives under `.claude/`. Confirm the installer isn't referenced anywhere live, then retire.
     - **This handoff's Appendices 1 & 2** — now redundant (folded into the current `CLAUDE.md` + auto-memory); trim when the slate is done.
     - **This handoff file itself** — retire via `rehydrate-handoff` once A–E are all complete.
   - **Next:** its own conversation — scope which orphans to delete vs migrate vs keep (a `decide-and-proceed` pass), then author the cleanup PLAN. Requirement-before-solution: agree the disposition of each item before touching it.

## Blocking decisions

- **[A] "Just run" scope** — ✅ RESOLVED 2026-07-02. Agreed: drag-drop single-file launcher + default `--config` + skip-already-done. Captured in PLAN-AA0. (Was: batch vs watch-folder vs entry point vs default-config.)
- **Harness swap confirmation** — confirm plan_foundry install completed AND the old bespoke `Bus/`-harness leftovers were reconciled (see Constraints). If the old harness was only partially removed, dangling references may remain in `.claude/CONSTITUTION.md` and the old `CLAUDE.md`.

## Constraints & do-nots

- **Always pass `--config config/config.yaml` to any `diarizer.cli run`** — without it the CLI loads baked-in defaults, `auth.token_path` (HF token at `D:/Projects/Tokens/kc_diariser.txt`) is ignored, and diarisation silently skips with no speaker labels. Project-critical. (from project memory `feedback_diarizer_config_flag`)
- **Uncommitted working-tree changes exist** — this session edited `diarizer/cli.py` (stdout/stderr `reconfigure(encoding="utf-8")` in `main()`) and `diarizer/webapp/app.py` (`_header_safe_filename()` applied to export `Content-Disposition`) to fix ` ` crashes. Pre-existing modifications also sit in `config/config.yaml`, `diarizer/webapp/{app.py,static/app.js,static/index.html,static/style.css}`. Commit or stash these BEFORE the harness swap touches `CLAUDE.md`/`.gitignore`, so pipeline fixes don't get entangled with harness churn.
- **AU spelling, usage, date formats** throughout.
- **To the human (Ken): plain language** — name the thing, the operation, the result; no compression or abbreviation in human-facing surfaces. (NB: plan_foundry's own convention refers to "the human", not by name, and forbids personal names in docs — reconcile if adopting its style wholesale.)
- **Discussion-vs-work-order** — treat "X is broken" / "we should Y" / "how about Z" as discussion openers, not directives; ask before acting.
- **Requirement before solution** — no mechanism design until requirement and process are agreed (directly gates item A).
- **Single-operation Bash only** — no `&&` / `;` / pipes / redirects chaining distinct ops; they trip permission prompts. (from project memory `feedback_compound_bash`)
- **Reviews follow the fixed five-element shape** — verification preamble → one-line verdict → priority-ordered punch list → "Not blockers" subgroup → net verdict.

### Post-swap reconciliation flags (from the install session)
- **Old harness was NOT in git** (`.claude/` is gitignored). The 8 superseded old skills + `maintenance-agent.md` were **moved aside, not deleted**, to `.claude/_retired_old_harness/` — reversible on disk. Delete that folder once you're satisfied nothing is missed.
- **Kept active on purpose:** `decide-and-proceed` skill and `.claude/CONSTITUTION.md` (your working-style substrate; referenced by CLAUDE.md). CONSTITUTION's Plan-lifecycle / Skill-registry sections are now stale — plan_foundry owns those.
- **`Bus/` is git-tracked and left intact** — old PLAN history. plan_foundry uses `Workbench/`. Decide later whether to migrate/retire `Bus/`; do not bulk-delete (it's the project's audit trail).
- **New PLANs go to `Workbench/`** via `write-plan` / `plan-pipeline`, not `Bus/`.

## Where things live

- **Preserved project `CLAUDE.md` (project-only) + memories/conventions** — the two appendix sections at the bottom of THIS file. Fold them into the new `CLAUDE.md` (outside plan_foundry's sentinel block) after install.
- **Diarizer CLI** — `diarizer/cli.py` (subcommands `run`, `serve`). HF token resolves via `config/config.yaml` → `auth.token_path`.
- **Webapp** — `diarizer/webapp/app.py` (FastAPI) + `diarizer/webapp/static/{app.js,index.html,style.css}`. Served with `python -m diarizer.cli serve <session-dir>` on `127.0.0.1:8765`.
- **Session outputs** — `output/<source-stem>_<timestamp>/` each hold `session.json`, `source.opus`, `transcript.json`, `waveform_peaks.json`. Two produced this session: the "OT Vulnerability Controls Specification…" and "OT VM Project Meeting…" recordings from `./input/`.
- **plan_foundry clone** — `plan_foundry/` at project root (source bundle; installer is `plan_foundry/.claude/skills/init-plan-foundry/lib/run_install.py`, procedure in `plan_foundry/BOOTSTRAP.md`).

---

## APPENDIX 1 — Preserved project CLAUDE.md (PROJECT-ONLY; harness rules intentionally excluded)

> Verbatim-faithful copy of the project-pertaining directives from the pre-swap `CLAUDE.md`. The old harness "Agent execution rules / Skills" sections tied to the `Bus/` mechanism are deliberately NOT copied — plan_foundry supplants them.

**Project Overview**
Offline speaker-diarisation + transcription pipeline (faster-whisper + pyannote.audio 3.3). See `README.md` for usage and `ARCHITECTURE.md` for design.

**Working style**
- AU spelling, usage, date formats.
- New conversation: ask clarifying questions if the request is under-specified.
- Treat "X is broken" / "we should Y" / "how about Z" as discussion openers, not work orders.
- No unprompted output of artefacts, illustrations, code, or longform sections — ask permission first.
- Long output: ask whether the direction is right before continuing.
- No cross-chat references between projects unless prompted.
- To Ken: plain language; name the thing, the operation, the result. No compression, no abbreviation.
- When offering options: describe each in full; state recommendation with reason.
- If a question is answerable from a tool's default behaviour, decide and proceed.
- State dependencies on prior decisions or external input explicitly.
- Requirement before solution — no mechanism design until requirement and process are agreed.
- Reviews follow the fixed five-element shape (see Constraints above).

**Durable operating rules worth keeping (harness-agnostic)**
- Single-operation Bash only (no `&&`/`;`/pipe chains across distinct ops).
- Delegate read-heavy searches (>5 files or >1500 lines) to a subagent.
- Cross-session state lives in the Claude auto-memory (`memory/MEMORY.md`).

**SUPPLANTED by plan_foundry (recorded for reconciliation, not to be re-adopted verbatim):**
- "All plans go to `Bus/`" → plan_foundry uses `Workbench/`.
- "RESEARCH/ADVICE via `write-bus-input`" → plan_foundry uses `write-input`.
- "PLAN-or-not triage / Bus PLAN mechanism" → plan_foundry's `plan-pipeline` + PLAN lifecycle.
- "`.claude/` is stable upstream; do not modify" → no longer applies now that plan_foundry manages `.claude/`.
- The `.claude/CONSTITUTION.md` cross-references (Discussion-vs-work-order, Review shape, Bash compounds, Decision autonomy) — verify whether plan_foundry ships equivalents or whether CONSTITUTION.md content must be re-homed.

## APPENDIX 2 — Preserved project memories & conventions

> From the Claude auto-memory for this project (`…/D--projects-diarizer/memory/`). These live outside the repo `.claude/` and so should survive the swap, but are duplicated here for durability.

**Memory: always pass `--config config/config.yaml` (feedback, 2026-05-26)**
Every `diarizer.cli run` invocation — interactive, scripted, or background — must include `--config config/config.yaml`. The CLI's `--config` defaults to `None`, which loads baked-in dataclass defaults instead of the project config, so `auth.token_path` is ignored, diarisation logs "no HF token resolved", and the transcript has no speaker attribution. Applies to `serve` too if it grows config-dependent behaviour.

**Memory: avoid compound Bash constructions (feedback, 2026-05-04)**
Don't chain shell ops with `&&` / `;` / pipes / redirects — they trigger Claude Code permission prompts and interrupt flow. Use single-operation Bash calls or the dedicated tools (Glob, Grep, Read, Edit, Write). One bare command per Bash call; run recovery as a separate call after seeing an error. Heredocs to a single command are fine (one semantic operation).

**Project conventions observed this session**
- Filenames from the recorder contain ` ` (narrow no-break space) and `.` — these break: (i) Windows cp1252 console `print()`, (ii) latin-1 HTTP `Content-Disposition` headers. Fixes applied this session reconfigure stdout to UTF-8 and sanitise the export filename; keep these when packaging (item A).
- HF token file: `D:/Projects/Tokens/kc_diariser.txt`, referenced by `config/config.yaml` → `auth.token_path`. No interactive HF auth was needed for pyannote or Whisper in this session.
- venv Python: `D:\projects\diarizer\venv\Scripts\python.exe`.
- Webapp default bind: `127.0.0.1:8765`; free the port with the owning-PID kill before restarting.
