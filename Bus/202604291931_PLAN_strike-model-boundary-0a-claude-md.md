---
title: "Strike model-boundary 0a: CLAUDE.md and AGENT_RULES.md"
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
triggers_plans: ["202604291950_PLAN_rationalise-claude-md.md"]
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Strike "Sonnet thinks / Haiku transcribes / Opus reviews" model-role protocol from CLAUDE.md and AGENT_RULES.md. Replace AGENT_RULES.md content entirely with a model-agnostic plan-execution lifecycle reference.

## Context
First of five sub-plans (0a–0e) splitting the original 202604291930_PLAN_strike-model-boundary.md (retired). Each sub-plan handles one file family with exact diffs. This sub-plan owns the two top-level instruction files. Plan 202604291950 (rationalise CLAUDE.md) depends on 0a completing first because both touch CLAUDE.md's "Agent execution rules" section.

## Steps

### CLAUDE.md edits

1. Open `CLAUDE.md`. In the "## Agent execution rules" section, find this exact block:

   ```
   **Operating rules:**
   1. **All plans go to Bus/** — Every piece of planned work lives as a PLAN file, never chat-only.
   2. **Respect model boundaries** — Sonnet thinks; Haiku writes; Opus reviews. Never cross lanes without Ken's model switch.
   3. **Research and advice to Bus/** — RESEARCH (Haiku data) and ADVICE (Opus strategic notes) go via `write-bus-input` skill. Writing an input auto-clears blocked plans waiting on it.
   ```

   Replace with this exact block:

   ```
   **Operating rules:**
   1. **All plans go to Bus/** — Every piece of planned work lives as a PLAN file, never chat-only.
   2. **Research and advice to Bus/** — RESEARCH (data drops) and ADVICE (strategic notes) go via `write-bus-input` skill. Writing an input auto-clears blocked plans waiting on it.
   ```

2. Verify by grep: `grep -n "Sonnet thinks\|Haiku writes\|Opus reviews\|Respect model boundaries" CLAUDE.md` returns no matches.

### AGENT_RULES.md replacement

3. Replace the entire contents of `AGENT_RULES.md` with this exact content:

   ```markdown
   # Agent Operating Rules

   Foundational rules for plan execution. Referenced from CLAUDE.md.

   ## Plan Execution Protocol

   Non-trivial work runs through a PLAN in `Bus/`, never chat-only.

   ### Lifecycle

   ```
   ready → in-progress → done | partially-complete | blocked | needs-revision
   ```

   - `ready` — transcribed, not yet started.
   - `in-progress` — execution has started.
   - `done` — all verification criteria pass. Terminal.
   - `partially-complete` — some steps done, others blocked or deferred. Terminal for this cycle.
   - `blocked` — cannot proceed; `blocked_by` holds the reason. Cleared automatically by `write-bus-input` when the resolving RESEARCH/ADVICE lands, or manually by Ken.
   - `needs-revision` — plan itself is faulty; halt and surface to Ken.

   ### Phases

   1. **Draft** — Converse with Ken; finalise PLAN content.
   2. **Transcribe** — Write PLAN file to `Bus/` via the `write-bus-plan` skill.
   3. **Review** *(optional)* — Review PLAN for gaps, dependencies, breakage risk.
   4. **Revise** *(if needed)* — Fold review feedback into PLAN.
   5. **Execute** — Run steps via the `execute-plan` skill; populate Executor Notes.
   6. **Closeout** — Update monthly LOG; git commit + push.
   7. **Post-execution review** — Append hiccup entries to `.claude/skills/_hiccups.md` for any deviation, silent failure, or hard error observed during the run.

   ### Halt conditions

   - **Interrupted runs:** if Ken halts a run mid-execution, his description of what he caught is a hiccup entry — write it to `.claude/skills/_hiccups.md` before resuming.
   - **Ambiguity:** if a step is ambiguous, unsafe, or marked [Ken]: halt, set `status: needs-revision`, surface to Ken. Do not improvise.
   - **Verification failure:** if any verification check fails, halt before commit; do not tick boxes that have not been verified.
   ```

4. Verify: `grep -i "sonnet\|haiku\|opus" AGENT_RULES.md` returns no matches.

### Commit

5. `git status` to confirm only `CLAUDE.md` and `AGENT_RULES.md` are modified.
6. `git add CLAUDE.md AGENT_RULES.md`
7. Commit with this exact message (HEREDOC form):
   ```
   Strike model-boundary protocol from CLAUDE.md and AGENT_RULES.md (0a)

   - Remove "Sonnet thinks / Haiku writes / Opus reviews" rule from CLAUDE.md
   - Remove "(Haiku data)" / "(Opus strategic notes)" qualifiers
   - Replace AGENT_RULES.md with model-agnostic plan execution lifecycle
   - Part 1 of 5 (sub-plans 0a–0e) splitting strike-model-boundary

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```
8. **Hold push** until Ken confirms — per push policy, do not push without authorisation.

## Verification
- [ ] `grep -n "Respect model boundaries\|Sonnet thinks\|Haiku writes\|Opus reviews" CLAUDE.md` returns no matches
- [ ] `grep -in "sonnet\|haiku\|opus" AGENT_RULES.md` returns no matches
- [ ] CLAUDE.md "Operating rules" list has exactly 2 items (was 3)
- [ ] AGENT_RULES.md exists, ≤50 lines, contains "Plan Execution Protocol" section with Lifecycle and Phases
- [ ] One commit created; push held for Ken's confirmation
- [ ] LOG row for this plan flipped to `done` after Executor Notes populated

## Executor Notes

**Executed:** 2026-04-30
**Outcome:** done
**What was done:** 
1. Edited CLAUDE.md: removed "Respect model boundaries" rule and updated Operating rules section (3 items → 2 items)
2. Replaced AGENT_RULES.md entire contents with model-agnostic plan execution lifecycle reference
3. Verified no model-role phrases remain via grep (both files clean)
4. Committed changes: b26de54
5. Push held pending Ken confirmation per plan requirements

**Files modified:** CLAUDE.md, AGENT_RULES.md
