---
title: "Create ideate skill (three-phase ideation arc + write-bus-input handoff)"
type: bus-plan
status: done
assigned_to: opus
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
blocked_by: ""
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

**Executed:** 2026-05-01 (authored directly in Opus during bootstrap, per option-2 path)
**Outcome:** done
**What was done:**
- Created `.claude/skills/ideate/SKILL.md` with frontmatter (trigger phrases, parent-session-only note), essential_principles, preconditions, inputs, output_schema (notes that this skill is conversational, not transactional — no `<pipeline-result>` block), exception_conditions, constraints, success_criteria
- Created `.claude/skills/ideate/workflows/ideate-arc.md` with phase-by-phase guidance: Clarify (requirement before mechanism), Survey (≥2 options with stated lean per Working style), Converge (sharpen + decision-triage at close per parent decision 15)
- Side-channel handoff to `write-bus-input` documented for both Clarify and Survey (RESEARCH for missing data, ADVICE for persisted decisions per parent decision 12)
- Checkpoint dispatch to `plan-writer` documented at clarify-locked and survey-converged moments (per parent decisions 4 and 18)
- Decision 15 triage requirement at Converge close documented (Already-locked / Mechanically-forced / Real-judgement-call) so subsequent `[Human]` checkpoints don't re-prompt for already-settled decisions
- Hard rule (decision 10) baked in: skill returns `outcome: exception` if invoked from a subagent context; runs only in parent session

**Blockers (if any):** None.

**Files modified:**
- Created: `.claude/skills/ideate/SKILL.md`
- Created: `.claude/skills/ideate/workflows/ideate-arc.md`

**Smoke test deferred:** real exercise will happen the first time a Human invokes ideation against a real problem (probably during dogfood child 1440 or some real PLAN).

**last_executor_outcome:**
  outcome: success
  outcome_subtype: done
  executed: 2026-05-01
  diagnostics_summary: ""
