---
title: "Integrate updated harness CLAUDE.md; retire CLAUDE copy.md"
type: bus-plan
status: done
assigned_to: ""
priority: high
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
linked_decisions:
  - "Diarizer-specific sections → README.md (usage) + ARCHITECTURE.md (design)"
  - "mechanism-notes.md rules → fold into CONSTITUTION.md (no new file)"
  - "Project Overview → one-line pointer in CLAUDE.md; full overview in README.md"
  - "Skills table → drop; pointer to CONSTITUTION § Skill registry"
linked_inputs:
  - "CLAUDE copy.md (operator-imported harness CLAUDE.md, 2026-05-04)"
blocked_by: ""
rollover_count: 0
---

## Objective

Apply the updated harness CLAUDE.md (currently sitting at root as `CLAUDE copy.md`) onto this project's `CLAUDE.md`, preserving Diarizer identity. Move all explanatory content out of `CLAUDE.md` per CONSTITUTION § "CLAUDE.md content rule" (directives only). Fold the three behavioural rules currently referenced from a non-existent `mechanism-notes.md` into the existing `CONSTITUTION.md`. Move usage/commands/testing content into `README.md`. Delete `CLAUDE copy.md` once integrated.

`ARCHITECTURE.md` is already written (Diarizer-focused) and does not need changes from this PLAN.

## Context

**Harness state.** The harness (`.claude/`) is treated as production / stable. This project does not modify harness internals; it only consumes them. CLAUDE.md sits at the seam between harness conventions and project content — it is the one piece this project edits.

**Current CLAUDE.md (166 lines)** mixes directives with substantive Diarizer documentation: Project Overview, Architecture & Core Design, Key Dependencies, Commands, Configuration, Testing Structure, Important Implementation Notes. CONSTITUTION § "CLAUDE.md content rule" forbids this — CLAUDE.md is directives only.

**Imported harness CLAUDE.md (`CLAUDE copy.md`, 40 lines)** is directives-only and references `.claude/references/mechanism-notes.md` (does not exist in this project) and `ARCHITECTURE.md` (now exists, written by this session).

**Three rules currently referenced from `mechanism-notes.md`:**
1. **Discussion-vs-work-order** — treat "X is broken" / "we should Y" / "how about Z" as discussion openers, not work orders.
2. **Review shape** — verification preamble → one-line verdict → priority-ordered punch list → "Not blockers" → net verdict.
3. **Bash compounds** — avoid `&&` / `;` / pipes in a single Bash call (PowerShell parser, hook brittleness).

These three rules belong in `CONSTITUTION.md` as new HOT sections. CLAUDE.md gets terse directive lines pointing to them.

**Decisions locked (operator, 2026-05-04):**
- D1: Diarizer-specific sections → README.md (usage: commands, configuration, testing) + ARCHITECTURE.md (design — already done).
- D2: `mechanism-notes.md` references → fold into existing CONSTITUTION.md, do not create a new file.
- D3: Project Overview → one-line pointer in CLAUDE.md; full overview lives in README.md.
- D4: Skills table → drop; pointer to CONSTITUTION § Skill registry.
- D5: ARCHITECTURE.md → already written and committed (`44c265d`).

## Steps

### Step 1 — Add three rules to CONSTITUTION.md

Append three new HOT sections to `.claude/CONSTITUTION.md`:

- **Discussion-vs-work-order** — directive + rationale (operator phrasing patterns; pause before acting; ask whether it's a request or a thought).
- **Review shape** — directive + the five-element shape (preamble, verdict, punch list, not-blockers, net verdict).
- **Bash compounds** — directive + rationale (PowerShell parser quirks; hook brittleness; one operation per Bash call).

Place after § "Communication with the operator" and before § "Decision autonomy". Update the HOT-classification list at the top of the file to include the three new sections.

**verify:** `grep -E "^## (Discussion-vs-work-order|Review shape|Bash compounds)" .claude/CONSTITUTION.md` returns three matches.

**acceptance:** All three sections present, each ≤25 lines, each tagged `**IMPORTANT:**` on the directive line.

### Step 2 — Expand README.md with usage content

Move into `README.md` from current `CLAUDE.md`:
- **Commands** section (Setup & Environment, Run the Main Pipeline, Testing) — verbatim move with light formatting.
- **Configuration** section (config.yaml structure: paths/auth/processing/output/display/logging) — verbatim move.
- **Testing Structure** notes (test numbering convention, fixtures, mocking, end-to-end) — verbatim move.

Keep existing README content (Description, Key Features) above. Add an **Outputs** subsection (transcripts/logs/memory-history paths) drawn from current CLAUDE.md.

**verify:** `grep -E "^## (Setup|Run|Testing|Configuration|Outputs)" README.md` returns ≥5 matches; `wc -l README.md` shows substantially more than 15.

**acceptance:** A new contributor can run the pipeline using only README.md (no CLAUDE.md read required).

### Step 3 — Rewrite CLAUDE.md to directives-only

Replace `D:/projects/Diarizer/CLAUDE.md` with a directives-only file modelled on `CLAUDE copy.md` but Diarizer-identified. Required content:

- **Project Overview** — one line: "Offline speaker-diarisation + transcription pipeline (Whisper + pyannote.audio). See [README.md](README.md) for usage, [ARCHITECTURE.md](ARCHITECTURE.md) for design."
- **Working style** — operator-facing directive list. Adopt the harness copy's bullets, but reference `.claude/CONSTITUTION.md` § <section-name> instead of `mechanism-notes.md`. AU spelling rule, "no cross-chat references", "options-with-recommendation", "decide and proceed", etc. Keep terse; one line per directive.
- **Agent execution rules** — operator-facing list. Pointer to `.claude/CONSTITUTION.md` § "Plan lifecycle" for phases/halt-conditions. Operating rules: plans go to `Bus/`; RESEARCH/ADVICE via `write-bus-input`; bash compounds (pointer to CONSTITUTION § "Bash compounds"); delegate read-heavy searches; cross-session state in `memory/MEMORY.md`; PLAN-or-not triage threshold.
- **Skills** — one-line pointer to `.claude/CONSTITUTION.md` § Skill registry. Drop the per-skill table.

**Drop entirely** (now lives elsewhere): Architecture & Core Design, Key Dependencies, Commands, Configuration, Testing Structure, Important Implementation Notes.

**verify:** `wc -l CLAUDE.md` returns ≤80 lines; `grep -E "^## (Architecture|Commands|Configuration|Testing Structure|Implementation Notes|Key Dependencies)" CLAUDE.md` returns 0 matches.

**acceptance:** CLAUDE.md contains only directives; every section either is a directive list or points to README.md / ARCHITECTURE.md / CONSTITUTION.md.

### Step 4 — Delete CLAUDE copy.md

Remove `D:/projects/Diarizer/CLAUDE copy.md` from the working tree.

**verify:** `test ! -f "CLAUDE copy.md"` succeeds (file absent).

**acceptance:** Repo root has only `CLAUDE.md`, `README.md`, `ARCHITECTURE.md` at the top level (no `CLAUDE copy.md`).

### Step 5 — Cross-check internal links

Walk `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`, `.claude/CONSTITUTION.md` and confirm every relative link target resolves. Specifically check:
- `CLAUDE.md` → `README.md`, `ARCHITECTURE.md`, `.claude/CONSTITUTION.md`, `memory/MEMORY.md`
- `ARCHITECTURE.md` → `README.md`, `CLAUDE.md`, `config/config.yaml`
- `.claude/CONSTITUTION.md` references that previously pointed at `mechanism-notes.md` — update to in-file section pointers.

**verify:** For each relative link in those four files, the target file or anchor exists.

**acceptance:** No broken internal links.

### Step 6 — Commit and push

Stage the four edited files (`CLAUDE.md`, `README.md`, `.claude/CONSTITUTION.md`, deletion of `CLAUDE copy.md`). Commit with message describing the integration. Push to `feature/speaker-separation`.

**verify:** `git log -1 --name-status` shows the four expected paths; `git status` shows clean (relative to staged set).

**acceptance:** Commit pushed; remote in sync.

## Verification (full PLAN)

- [ ] CONSTITUTION.md has three new HOT sections (Discussion-vs-work-order, Review shape, Bash compounds).
- [ ] README.md has Setup/Run/Testing/Configuration/Outputs sections; pipeline runnable from README alone.
- [ ] CLAUDE.md ≤80 lines, directives-only, Diarizer-identified in the overview line.
- [ ] CLAUDE copy.md deleted.
- [ ] All internal links resolve.
- [ ] Commit pushed to `feature/speaker-separation`.

## Acceptance (spec)

A fresh session reading only `CLAUDE.md` understands what this project is, how to behave (working style + agent rules), and where to look for everything else. A human developer reading only `README.md` can install and run the pipeline. A maintainer reading only `ARCHITECTURE.md` understands the v1 design well enough to plan the v2 rewrite. No content duplicated across the three files.

## Out of scope

- Any changes to `.claude/` skills, agents, scripts, or settings — harness is in production.
- The Diarizer v2 re-architecture PLAN — separate, follow-up PLAN.
- Touching `memory/MEMORY.md` or `Retired/`.

## Executor Notes

**Executed:** 2026-05-04 (same-session execution by opus, after PLAN draft)
**Outcome:** done

**What was done:**
- Step 1: Added three HOT sections to `.claude/CONSTITUTION.md` (Discussion-vs-work-order, Review shape, Bash compounds), inserted between `Communication with the operator` and `Decision autonomy`. Updated the HOT classification list at the top of the file. Removed broken `mechanism-notes.md` reference from line 3.
- Step 2: Rewrote `README.md` (15 → 92 lines) with Setup, Run, Outputs, Configuration, Testing, and Project status sections. Pipeline runnable from README alone.
- Step 3: Rewrote `CLAUDE.md` (166 → 39 lines), directives-only. Project Overview is now a one-line pointer to README.md and ARCHITECTURE.md. Working style references CONSTITUTION § Discussion-vs-work-order and § Review shape. Operating rule 3 references CONSTITUTION § Bash compounds. Operating rule 7 added: harness treated as production / stable upstream. Skills section is a one-line pointer to CONSTITUTION § Skill registry.
- Step 4: Deleted `CLAUDE copy.md` from working tree (was untracked; nothing to stage).
- Step 5: Verified internal links in CLAUDE.md, README.md, ARCHITECTURE.md, CONSTITUTION.md all resolve. Remaining `mechanism-notes.md` references in `.claude/skills/_shared/decisions.md` and `MAINTENANCE.md` are out-of-scope (harness internals).
- Step 6: Commit + push (this commit).

**Files modified:**
- `CLAUDE.md` (rewrite, 166→39 lines)
- `README.md` (expand, 15→92 lines)
- `.claude/CONSTITUTION.md` (3 new sections + HOT list update + broken-link fix)
- `CLAUDE copy.md` (deleted; was untracked)
- `Bus/202605010000_LOG_202605.md` (status update)
- `Bus/202605040100_PLAN_integrate-harness-claude-md.md` (this file: status=done, Executor Notes)

**Blockers:** none.

**Notes:** PLAN 2 (`202605040200_PLAN_diarizer-v2-rearchitecture.md`) was `blocked_by: 202605040100_PLAN_integrate-harness-claude-md.md`; that block can now be cleared.
