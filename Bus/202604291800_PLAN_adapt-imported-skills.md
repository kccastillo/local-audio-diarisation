---
title: "Adapt imported skills from BA→code; import debug-like-expert"
type: bus-plan
status: ready
assigned_to: haiku
priority: high
created: 2026-04-29
created_by: haiku
created_month: 202604
log_month: 202604
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: ""
rollover_count: 0
triggers_plans: []
closes_thread: "H01"
advances_thread: "H02"
parent_plan_of_plans: ""
---

## Objective
Trim and re-fit the 5 skills imported from `seqwater-app-control-2026` (`create-agent-skills`, `write-bus-plan`, `write-bus-input`, `execute-plan`, `retire`) so they match this project's coding workflow rather than BA/Wiki/State conventions, and import `debug-like-expert` from `gsd-build/gsd-2`. This is the dogfooding run of the bus — first PLAN through the new harness implements the harness fixes.

## Context
Source audit: `.claude/SKILLS_AUDIT.md` (5 sections — rename proposal, per-skill BA-isms, cross-cutting changes, gap analysis, suggested PLAN structure). The audit's Section 5 is the basis for the Steps below. Closes H01 in HARNESS.md. Step 11 also effectively closes H02 (debug-like-expert import); recorded as `advances_thread: H02` because the conventions cap one closer per PLAN — Haiku should still mark H02 closed in HARNESS.md when step 11 lands.

Key BA-isms to remove during execution:
- `Wiki/`, `[[wikilinks]]`, `_index.md`, `Antigravity`, `atomise`, `librarian`, `pandoc-convert`, `Production/Wiki/`
- `State/` directory pattern (does not exist here)
- `linked_decisions` frontmatter field (no decision-tracking system in this project)

Path corrections:
- ROADMAP.md is at repo root, not `.claude/ROADMAP.md`
- HARNESS.md is at `.claude/HARNESS.md` (new) — execute-plan must handle both files via T-/H- prefix on thread IDs

## Steps
[Numbered steps for Haiku to execute via the `execute-plan` skill. Each step is independently verifiable. No `[Ken]` or `[blocked-on-input]` markers — all steps are mechanical edits informed by `.claude/SKILLS_AUDIT.md`.]

1. Rename `create-agent-skills` → `create-skill`. Rename the directory `.claude/skills/create-agent-skills/` to `.claude/skills/create-skill/`. Update SKILL.md frontmatter `name:` field. Grep `.claude/` for any `create-agent-skills` references and update them.
2. Update `.claude/skills/write-bus-plan/templates/plan-template.md`:
   - Drop `linked_decisions` field.
   - Add fields: `branch: ""`, `files_touched: []`, `verification_commands: []`, `tag_after: ""`.
   - Replace Wiki/`_index.md` verification example bullets with code-style examples (pytest invocations, `python run_diariser.py` smoke runs, frontmatter `last_updated` checks on ROADMAP.md/HARNESS.md/CLAUDE.md).
   - In the Steps section guidance, add the vertical-slice note: "Aim for vertical-slice steps — each step delivers a verifiable end-to-end change rather than a horizontal layer."
   - In the Context section, replace `[[wikilinks]]` example phrasing with neutral wording.
   - Recurring Task cadence list: extend to include `after-merge` and `after-release` alongside `monthly | quarterly | after-event`.
3. Update `.claude/skills/write-bus-input/templates/research-template.md`:
   - Add frontmatter fields: `research_kind: ""` (one of: library-probe | api-behaviour | benchmark | bug-repro | docs-summary), `target_versions: []`.
   - Add a `## Reproduction` section between `## Findings` and `## Sources` with the placeholder text from SKILLS_AUDIT.md §2.2.
4. Update `.claude/skills/write-bus-input/templates/advice-template.md`: change `from: "Opus via Ken"` default to `from: ""`.
5. Update `.claude/skills/execute-plan/SKILL.md` and `.claude/skills/execute-plan/workflows/execute-steps.md`:
   - SKILL.md: replace Atomise/Wiki examples in `<skill_invocation_semantics>` with code-relevant examples (e.g. invoking `retire` on a deprecated module). Update the roadmap-sync line to: "Apply roadmap sync to whichever file (ROADMAP.md or HARNESS.md) contains the referenced thread, disambiguated by the T-/H- prefix on the thread ID."
   - execute-steps.md Step 4.5: read `ROADMAP.md` (root) for T-prefixed threads and `.claude/HARNESS.md` for H-prefixed threads.
   - execute-steps.md Step 5: drop the "For every State/ file modified..." block. Replace with: "If CLAUDE.md, ROADMAP.md, or HARNESS.md was modified, ensure their `last_updated` frontmatter is bumped to today."
   - `.claude/skills/execute-plan/references/log-rules.md`: drop the "State File Updates" section.
6. Update `.claude/skills/write-bus-plan/references/bus-conventions.md`:
   - Roadmap Linkage section: change "ROADMAP.md" references to: "PLANs may reference threads in either `ROADMAP.md` (T-prefix IDs) or `.claude/HARNESS.md` (H-prefix IDs). The prefix on the ID disambiguates which file holds the thread."
   - Add a short subsection documenting the T-/H- prefix convention.
7. Project-wide search-and-replace pass under `.claude/`:
   - Remove or rephrase any remaining occurrences of `Wiki/`, `Wiki_Map`, `wikilinks`, `[[...]]` patterns, `Antigravity`, `State/` directory references, `atomise`, `librarian`, `pandoc-convert`, `Production/Wiki/`.
8. Update `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`: replace the "Existing Project Skills" table (currently 13 seqwater entries) with the actual 5 imported skills (post-rename: `create-skill`, `write-bus-plan`, `write-bus-input`, `execute-plan`, `retire`) plus `debug-like-expert` (added in step 11).
9. Update `CLAUDE.md` skills table to reflect the rename (`create-agent-skills` → `create-skill`) and add the `debug-like-expert` row.
10. Confirm step 2's vertical-slice guidance landed in `plan-template.md`'s Steps section (this step is a verification cross-check; if missing, add it).
11. Import `debug-like-expert` from `gsd-build/gsd-2/src/resources/skills/debug-like-expert/`: copy SKILL.md + selected references into `.claude/skills/debug-like-expert/`, trim GSD-specific framing, ensure `name:` and `description:` frontmatter match the imported-skills convention. After this lands, mark H02 in `.claude/HARNESS.md` as `Status: closed` and append a closure bullet.
12. Verify:
    - `Skill("create-skill")` invocation resolves (manual smoke).
    - `plan-template.md` renders with the new frontmatter fields when used.
    - Grep `.claude/` returns zero hits for: `Wiki`, `Antigravity`, `State/`, `atomise`, `linked_decisions`.
    - `.claude/skills/debug-like-expert/SKILL.md` exists and has valid frontmatter.
13. Retire `.claude/SKILLS_AUDIT.md` via the `retire` skill once all preceding steps verify.

## Verification
- [ ] Directory `.claude/skills/create-skill/` exists; `.claude/skills/create-agent-skills/` does not.
- [ ] `plan-template.md` frontmatter contains `branch`, `files_touched`, `verification_commands`, `tag_after`; does NOT contain `linked_decisions`.
- [ ] `research-template.md` frontmatter contains `research_kind` and `target_versions`; body has `## Reproduction` section.
- [ ] `advice-template.md` `from:` default is `""`.
- [ ] `execute-plan/workflows/execute-steps.md` Step 4.5 reads both ROADMAP.md and HARNESS.md; Step 5 has no State/ block.
- [ ] `bus-conventions.md` documents T-/H- prefix convention.
- [ ] Grep under `.claude/` returns zero hits for `Wiki`, `Antigravity`, `State/`, `atomise`, `linked_decisions`.
- [ ] `SKILLS_IMPLEMENTATION_GUIDE.md` "Existing Project Skills" table lists only the 6 actual skills (5 imported + debug-like-expert).
- [ ] `CLAUDE.md` skills table reflects rename and lists debug-like-expert.
- [ ] `.claude/skills/debug-like-expert/SKILL.md` exists with valid frontmatter.
- [ ] `.claude/HARNESS.md` H01 status set to `closed` with closure bullet; H02 status set to `closed` with closure bullet.
- [ ] `.claude/SKILLS_AUDIT.md` moved to `Retired/`.

## Executor Notes
*Populated by Haiku after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
