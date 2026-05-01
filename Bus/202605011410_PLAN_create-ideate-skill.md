---
title: "Create ideate skill (three-phase ideation arc + write-bus-input handoff)"
type: bus-plan
status: blocked
assigned_to: sonnet
priority: high
created: 2026-05-01
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: "Parent PLAN 202605011400 must complete steps 1-6a (shared plan-safe.md, execute-plan and write-bus-plan mutations, bus-conventions update, agent files) before this child runs"
rollover_count: 0
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: 202605011400_PLAN_build-plan-pipeline-orchestrator.md
pipeline_phase: drafted
---

## Objective
Create the `ideate` skill at `.claude/skills/ideate/SKILL.md` — a conversational arc that runs in the parent session (never as a subagent) and walks the Human through Clarify → Survey → Converge to produce a plan-ready idea. The skill must also recognise when a question requires external data (RESEARCH) or a strategic decision worth persisting (ADVICE) and invoke `write-bus-input` to drop the appropriate Bus file.

## Context

Spawned from parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md`. See that PLAN's design decisions 6, 10, 12 for the `ideate` design constraints:

- **Decision 6:** three explicit phases (Clarify → Survey → Converge), imposing the working-style discipline already in CLAUDE.md (requirement before mechanism; options in full with a recommendation).
- **Decision 10 (hard rule):** ideation runs only in the parent session, never as a subagent. There is no `planning-ideator` agent.
- **Decision 12:** ideation may produce RESEARCH/ADVICE inputs, not just PLAN content. The skill should recognise these moments and invoke `write-bus-input`. Resulting filenames land in the eventual PLAN's `linked_inputs: []` array.

This PLAN is itself design-heavy — the child of a parent that flagged it for decomposition. It will likely need its own check-plan pass before execution, which may further sub-decompose if the SKILL.md authoring proves non-trivial.

## Steps

1. **[Human or Sonnet — design decision]** Decide the SKILL.md structure: single SKILL.md with embedded phase descriptions, or SKILL.md plus `workflows/` files for each phase. Lean: single SKILL.md (skill is small; phases are conversational guidance not procedural workflow).

2. Compose the SKILL.md frontmatter (`name: ideate`, `description: ...` covering trigger phrases like "let's ideate", "help me think through X", and noting parent-session-only).

3. Compose the body documenting:
   - The three phases with explicit entry/exit conditions.
   - The Human-signals-done exit (no automatic detection).
   - Non-output: skill itself does not write to disk for plan content.
   - **Side-channel outputs:** when to invoke `write-bus-input` for RESEARCH (need external data) or ADVICE (decision worth recording). Include illustrative trigger conditions.
   - Handoff: at Converge close, the orchestrator (`plan-pipeline`) takes the converged idea and dispatches to `plan-writer` agent to transcribe.
   - **Decision-triage at Converge close** (parent decision 15): the Converge phase output must classify every design decision touched during ideation into Already-locked / Mechanically-forced / Real-judgement-call. The handoff to `plan-writer` carries this classification so it lands in the PLAN's Context section and informs the subsequent `[Human — design-review checkpoint]` triage.

4. Write the file at `.claude/skills/ideate/SKILL.md`.

5. **[Human]** review the drafted skill against the design decisions in the parent PLAN. Iterate if needed.

## Verification

- [ ] `.claude/skills/ideate/SKILL.md` exists with valid frontmatter
- [ ] Three phases (Clarify, Survey, Converge) explicitly documented with entry/exit conditions
- [ ] Decision 10 (parent-session-only) stated in skill body
- [ ] Decision 12 (write-bus-input handoff for RESEARCH/ADVICE) documented with trigger conditions
- [ ] Decision 15 (decision-triage at Converge close) documented; classification categories and handoff format specified
- [ ] Decision 16 compliance: SKILL.md contains `<preconditions>` block (input requirements) and `<output_schema>` block (what Converge returns to plan-writer)
- [ ] Handoff to plan-writer at Converge close documented

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
