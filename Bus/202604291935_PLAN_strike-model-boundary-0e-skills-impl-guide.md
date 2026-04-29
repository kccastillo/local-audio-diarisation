---
title: "Strike model-boundary 0e: SKILLS_IMPLEMENTATION_GUIDE.md"
type: bus-plan
status: needs-revision
assigned_to: ""
priority: high
created: 2026-04-29
created_by: sonnet
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
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Strike model-role protocol from `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`. The "Existing Project Skills" table currently lists `Haiku` as the "Callable By" for several skills — change to model-agnostic phrasing. Note: the create-agent-skills references discuss Haiku/Sonnet/Opus only in **model-tier testing** context (e.g. "test your skill against Haiku, Sonnet, Opus") — that is NOT the model-role protocol and is out of scope. This plan does not touch those references.

## Context
Sub-plan 0e — last of the strike-model-boundary split. Smallest scope.

## Steps

### File 1: `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`

1. Replace the "Existing Project Skills" table. Find this exact block:
   ```
   ## Existing Project Skills

   | Skill | Purpose | Callable By |
   |---|---|---|
   | `retire` | Move files to gitignored Retired/ folder | plans, other skills |
   | `consolidate` | Find canonical content, replace with pointer refs | plans, other skills |
   | `authority-detect` | Detect if content is authoritative | atomise |
   | `create-agent-skills` | Guide creation of new skills | user (invoked manually) |
   | `create-commands` | Guide creation of new commands | user (invoked manually) |
   | `write-bus-plan` | Transcribe plans into Bus/ structure | Haiku on Ken request |
   | `write-bus-input` | Write RESEARCH/ADVICE to Bus/ | Ken/Sonnet on handoff |
   | `pandoc-convert` | Convert Clippings source files to MD | Haiku in atomise pipeline |
   | `atomise` | Split MD into atomic notes | Haiku after pandoc-convert |
   | `librarian` | Route atomic notes into Wiki/ | Haiku after atomise |
   | `wiki-map-update` | Rebuild Wiki_Map.md Mermaid diagram | librarian, consolidate (after migration) |
   | `execute-plan` | Run a PLAN step-by-step | Haiku on Ken request |
   ```
   Replace with:
   ```
   ## Existing Project Skills

   | Skill | Purpose | Callable By |
   |---|---|---|
   | `retire` | Move files to gitignored Retired/ folder | plans, other skills |
   | `consolidate` | Find canonical content, replace with pointer refs | plans, other skills |
   | `authority-detect` | Detect if content is authoritative | atomise |
   | `create-agent-skills` | Guide creation of new skills | user (invoked manually) |
   | `create-commands` | Guide creation of new commands | user (invoked manually) |
   | `write-bus-plan` | Transcribe plans into Bus/ structure | active session on Ken request |
   | `write-bus-input` | Write RESEARCH/ADVICE to Bus/ | active session on handoff |
   | `pandoc-convert` | Convert Clippings source files to MD | active session in atomise pipeline |
   | `atomise` | Split MD into atomic notes | active session after pandoc-convert |
   | `librarian` | Route atomic notes into Wiki/ | active session after atomise |
   | `wiki-map-update` | Rebuild Wiki_Map.md Mermaid diagram | librarian, consolidate (after migration) |
   | `execute-plan` | Run a PLAN step-by-step | active session on Ken request |
   ```

2. Verify: `grep -in "haiku\|sonnet" .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` returns no matches.

### Verify across whole repo and commit

3. Run final verification across all active instructions: `grep -rin "haiku transcribes\|sonnet thinks\|opus reviews\|respect model boundaries" CLAUDE.md AGENT_RULES.md .claude/`. Expected: no matches.

4. Run grep for `haiku` across instruction files (broader check): `grep -rin "haiku" CLAUDE.md AGENT_RULES.md .claude/skills/write-bus-plan/ .claude/skills/write-bus-input/ .claude/skills/execute-plan/ .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`. Expected: no matches.

5. Note: `grep -rin "haiku" .claude/skills/create-agent-skills/` IS expected to match (model-tier testing context — out of scope per Objective). Confirm the matches are only in those files.

6. `git add .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`

7. Commit with this exact message:
   ```
   Strike model-boundary from SKILLS_IMPLEMENTATION_GUIDE.md (0e)

   - Replace "Haiku" callers in Existing Project Skills table with "active session"
   - Final part 5 of 5 (sub-plans 0a–0e); strike-model-boundary complete
   - Note: create-agent-skills references retain Haiku/Sonnet/Opus mentions
     in model-tier testing context (out of scope per directive)

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

8. Hold push for Ken's confirmation.

## Verification
- [ ] `grep -in "haiku\|sonnet" .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` returns no matches
- [ ] Final repo-wide check: `grep -rin "haiku transcribes\|sonnet thinks\|opus reviews" CLAUDE.md AGENT_RULES.md .claude/` returns no matches (Bus/ historical records and Retired/ excluded by default scope of grep targets above)
- [ ] One commit; push held

## Executor Notes

**Executed:** 2026-04-29
**Outcome:** needs-revision
**What was done:**
- Replaced "Existing Project Skills" table in `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`: 7 rows with `Haiku ...` / `Ken/Sonnet ...` "Callable By" entries swapped to `active session ...` phrasing per the Steps block.
- Step 2 verification passed: `grep -in "haiku\|sonnet" .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` returns no matches.
- Step 3 repo-wide verification FAILED — matches still present:
  - `CLAUDE.md:31` → "Respect model boundaries — Sonnet thinks; Haiku writes; Opus reviews."
  - `.claude/skills/write-bus-plan/SKILL.md:9` → "Haiku transcribes..."
  - `.claude/skills/write-bus-input/SKILL.md:7` → "Haiku transcribes..."
- Step 5 (create-agent-skills check) confirmed: matches in `core-principles.md`, `iteration-and-testing.md`, `use-xml-tags.md` — model-tier testing context, out of scope.
- Halted before commit per halt-on-failure protocol.

**Blockers:**
- Sub-plans 0a (CLAUDE.md/AGENT_RULES.md), 0b (write-bus-plan), 0c (write-bus-input), 0d (execute-plan) all still `status: ready` in the LOG. They must execute before 0e's repo-wide verification (Step 3) can pass. 0e was authored as the *final* sub-plan but was invoked first.

**Files modified:**
- `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` (table rewrite, kept on disk; uncommitted pending Ken's call)
- `Bus/202604291935_PLAN_strike-model-boundary-0e-skills-impl-guide.md` (status, Executor Notes)

**Recommended path forward (for Ken/Sonnet):**
- Execute 0a → 0b → 0c → 0d in order, then re-run 0e. The 0e file edit is already on disk; the re-run can either keep it (and skip Step 1) or revert and re-apply.
- Alternatively: revise 0e to drop the repo-wide Step 3 check (turning it into a same-file-only plan) and accept that the cross-cutting verification will live in a separate closeout plan after 0a–0d.
