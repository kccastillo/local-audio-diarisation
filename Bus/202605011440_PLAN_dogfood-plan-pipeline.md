---
title: "Dogfood plan-pipeline against a real small target (note-jot utility skill)"
type: bus-plan
status: ready
assigned_to: sonnet
priority: medium
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
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
---

## Objective
Exercise the freshly-built `plan-pipeline` orchestrator end-to-end against a small but real target — designing and creating a tiny utility skill called `note-jot` (writes a one-line note with a timestamp to a daily log file). The dogfood validates phase transitions, subagent dispatch, background execution, parent/children interaction, and the retire-as-final-step contract.

## Context

Spawned from parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md`. The dogfood target is `note-jot` rather than a contrived no-op so that the ideation phase (Clarify → Survey → Converge) has real surface to exercise — what does the skill take as input? where does the daily log live? what format? — exactly the kinds of questions `ideate` is designed to discipline.

**Why `note-jot`:** small enough to complete in one dogfood run, but real enough that the Human will have at least one Survey-phase decision (e.g. log file location) and the `ideate → write-bus-plan → check-plan → execute-plan → retire` sequence produces a real artefact at the end (a working `note-jot` skill).

## Steps

1. Invoke `Skill("plan-pipeline")` (or trigger via "let's make a plan for a note-jot utility skill"). Confirm the orchestrator activates and reads phase state.

2. **Phase: drafting** — orchestrator runs `ideate` in the parent session. Walk Clarify → Survey → Converge with the Human deciding scope and shape of `note-jot`. Verify: at clarify-locked and survey-converged moments, orchestrator dispatches `plan-writer` agent to write/update the dogfood PLAN file incrementally. Verify the dogfood PLAN file's `pipeline_phase: drafting` updates correctly.

3. **Phase transition: drafting → drafted** — Human signals ideation is done. Orchestrator flips dogfood PLAN's `pipeline_phase: drafted`.

4. **Phase: drafted → checked** — orchestrator dispatches `plan-checker` agent. Verify the agent reads the dogfood PLAN, returns a two-section review. If review surfaces blockers: confirm orchestrator stays at `drafted` and surfaces findings (this is a valid dogfood outcome — it tests the unhappy path). Iterate until checked.

5. **Phase transition: checked → executing** — orchestrator flips `pipeline_phase: executing` and dispatches `plan-executor` agent with `run_in_background: true`. Verify parent stays responsive during execution.

6. **Phase: executing → complete** — on completion notification, orchestrator flips `pipeline_phase: complete`. Verify `note-jot` skill exists and works (smoke test it).

7. **Phase: complete → retired** — orchestrator invokes `Skill("retire", "<dogfood PLAN path>")`. Verify dogfood PLAN moves to `Retired/` and a commit/push lands.

8. Capture findings in this PLAN's Executor Notes:
   - Did each phase transition work cleanly?
   - Any orchestrator bugs surfaced?
   - Any subagent dispatch issues?
   - Did `pipeline_phase` track correctly throughout?
   - Did `run_in_background:true` work as expected?
   - Was the `ideate` skill's RESEARCH/ADVICE handoff exercised? (If `note-jot` design didn't need it, note that — but design something else if dogfood doesn't naturally exercise this path.)
   - Recommendations for fixes/iterations on the new skills.

## Verification

- [ ] Dogfood PLAN file (separate, for `note-jot`) created via the pipeline and exists in `Bus/`
- [ ] All five `pipeline_phase` transitions completed: drafting → drafted → checked → executing → complete
- [ ] `note-jot` skill exists at `.claude/skills/note-jot/SKILL.md` and smoke-tests pass
- [ ] Dogfood PLAN retired to `Retired/`
- [ ] Findings captured in this PLAN's Executor Notes with explicit pass/fail for each transition
- [ ] Any bugs/issues filed as follow-up items (new PLANs or notes in parent PLAN's Executor Notes)

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
