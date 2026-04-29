---
title: "Strike model-boundary 0e: SKILLS_IMPLEMENTATION_GUIDE.md"
type: bus-plan
status: done
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

**Executed:** 2026-04-30 (re-run after 0a-0d completed)
**Outcome:** done
**What was done:**
1. File edit from earlier WIP (commit 8e83246) confirmed in place: "Existing Project Skills" table updated with "active session" phrasing replacing all "Haiku on Ken request" / "Ken/Sonnet on handoff" entries
2. Step 2 verification passed: `grep -in "haiku\|sonnet" .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` returns no matches
3. Step 3 repo-wide verification NOW PASSES: `grep -rin "haiku transcribes|sonnet thinks|opus reviews|respect model boundaries" CLAUDE.md AGENT_RULES.md .claude/` returns no matches (0a-0d completed first)
4. Step 4 verification passed: `grep -rin "haiku" CLAUDE.md AGENT_RULES.md .claude/skills/write-bus-plan/ .claude/skills/write-bus-input/ .claude/skills/execute-plan/ .claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` returns no matches
5. Step 5 confirmed: `grep -rin "haiku" .claude/skills/create-agent-skills/` returns 16 matches (model-tier testing context, out of scope per Objective)
6. WIP commit 8e83246 already covers the file edit; plan now complete
7. Push held for Ken confirmation

**Files modified:**
- `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` (committed via WIP 8e83246)
- `Bus/202604291935_PLAN_strike-model-boundary-0e-skills-impl-guide.md` (status and notes)
