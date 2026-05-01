---
title: "Create plan-pipeline orchestrator skill (phase dispatch via Agent tool)"
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
blocked_by: "Parent PLAN 202605011400 must complete steps 1-6a (shared file, mutations, conventions, agents) AND children PLAN_create-ideate-skill (202605011410), PLAN_create-audit-haiku-safe-skill (202605011420), PLAN_create-audit-sufficiency-skill (202605011425) must be done before this child runs"
rollover_count: 0
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: 202605011400_PLAN_build-plan-pipeline-orchestrator.md
pipeline_phase: drafted
---

## Objective
Create the `plan-pipeline` orchestrator skill at `.claude/skills/plan-pipeline/SKILL.md` — the end-to-end orchestrator that walks a planning workflow through five phases (ideation → write → check → execute → retire), dispatches non-interactive phases to pinned-model subagents in `.claude/agents/`, and tracks state via the `pipeline_phase` frontmatter field on PLAN files.

## Context

Spawned from parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md`. This is the most design-heavy of the three skill-creation children — contains complex dispatch logic, agent invocation patterns, background-mode handling, and several edge cases surfaced during parent's check-plan review.

**Design constraints from parent (decisions 1, 10, 11, 13):**
- Phase state lives in `pipeline_phase` frontmatter; orchestrator reads it and dispatches.
- Interactive phases (ideation only) run in parent session, never as subagent.
- Activation triggers: "let's make a plan", "let's plan X", "I want to plan Y", "plan out Z".
- Execution dispatched with `run_in_background: true` so parent stays responsive.
- **Orchestrator owns all git operations.** Subagents do filesystem work only — they do not commit or push. The orchestrator commits and pushes after each phase that produces filesystem changes (write phase, execute phase, retire phase). Design decision: one commit per phase, OR one commit at pipeline completion — to be settled in this PLAN's design pass.

- **Hybrid background mode** (parent decision 18). Only `plan-executor` runs background; all other subagents (plan-writer, sufficiency-auditor, plan-safety-auditor, plan-retirer) run foreground. Audit loops, writes, and retires are fully synchronous — orchestrator dispatches, parent blocks, agent returns, orchestrator continues. Background mode is reserved for the one phase (`executing`) where parent responsiveness during a long run actually matters.

- **Subagent-driven exception handling** (parent decision 19). Orchestrator does NOT process mid-execution messages or attempt to cancel agents. Each subagent has agency to self-terminate cleanly on its declared exception conditions. Orchestrator receives one transactional result per dispatch and routes on the `outcome` enum:
  - `success` → advance phase or proceed in loop.
  - `revision_needed` → audit found blockers; surface to Human, await revision, re-dispatch.
  - `exception` → **kanban full stop**: halt pipeline, surface diagnostics, do not advance.

- **`outcome` enum is the load-bearing contract** (parent decision 20). Routing table for the orchestrator's SKILL.md: `(phase, outcome) → next action`. Exhaustive — every (phase, outcome) pair has a defined next action. Inter-skill handoffs validate that upstream `payload` matches downstream `inputs` schema.

- **Audit loops are synchronous** (parent decision 21). Foreground audits make the loop deterministic and easy to follow. Iteration count tracked, max iterations (default 5) triggers `outcome: exception` to escalate. Each iteration announced to Human (no silent re-runs).

- **One conversational re-entry point: executor completion.** When `plan-executor` (background) returns its completion message, parent Claude re-invokes `plan-pipeline` against the PLAN. Idempotent re-entry from on-disk `pipeline_phase` makes a missed cue recoverable. All other transitions are synchronous and don't require re-entry.

**Design constraints from parent's Learnings section:**
- Trigger-phrase collision risk vs. `write-bus-plan` ("create plan file") — verify Skill matcher routes correctly; disambiguate if needed.
- `parent_plan_of_plans` traversal — orchestrator must know how to dispatch from parent to children, wait, and resume parent.
- `pipeline_phase` advancement vs. children blocking — if parent has spawned children, parent must not auto-advance until children complete. Need a "waiting on children" state or equivalent.
- Ad-hoc PLAN handling: PLAN with no `pipeline_phase` field defaults to `drafted` (per bus-conventions update in parent step 5), not `drafting`.

## Steps

1. **[Human or Sonnet]** Decide structure: single SKILL.md, or SKILL.md plus `workflows/dispatch.md` and `references/phase-state-machine.md`. Lean: split into workflows/references — the dispatch logic and edge cases are too dense for one file.

2. Compose SKILL.md frontmatter:
   - `name: plan-pipeline`
   - `description:` with activation triggers per decision 11; clearly states it is the orchestrator and references the phase state machine.

3. Compose the dispatch state machine (in SKILL.md or `references/phase-state-machine.md`). The state machine is a (phase, audit_state, outcome) → action lookup. Document each transition exhaustively:
   - `(absent | empty)` pipeline_phase → treat as `drafted` (per bus-conventions ad-hoc default).
   - `drafting` → run `ideate` in parent session (interactive, per decision 10); checkpoint via `plan-writer` foreground dispatch at clarify-locked and survey-converged moments. After write, commit+push (decision 22). On Converge close: flip `pipeline_phase: drafted`.
   - `drafted` (audit loop, per decision 21): orchestrator behaviour is fully spelled out in decision 21 of parent — implement that exact state machine. Audits run foreground; outcome-driven branching; durable `audit_state` frontmatter; per-stage iteration counters with MAX_ITERATIONS=5.
   - `checked` → flip `pipeline_phase: executing`; commit+push the phase flip; dispatch `plan-executor` **background** (the only background dispatch); orchestrator returns control to parent.
   - `executing` → orchestrator skill returns. The parent Claude must re-invoke `plan-pipeline` when the executor's completion message arrives (the single conversational re-entry point — see decision 18). On re-entry: read `outcome` from PLAN frontmatter (`last_executor_outcome`).
     - `outcome: success` → flip `pipeline_phase: outcome-verifying` (NEW — per decision 25); run state and acceptance verifications; do NOT advance to complete until verifications pass.
     - `outcome: revision_needed` → revert `pipeline_phase: drafted`; reset `audit_state` if appropriate (because content changed); surface findings; commit+push.
     - `outcome: exception` → kanban full stop; commit+push WIP state with diagnostics.
   - `outcome-verifying` (NEW phase per decision 25) → run all `verify:` and `acceptance:` shell commands from PLAN's Verification section; tally pass/fail; collect `verify: human` items into `verification_state.human_pending`; commit+push the state update.
     - Any verify/acceptance shell command failed → outcome = revision_needed, override executor's success, revert pipeline_phase to drafted, surface diagnostics.
     - All shell checks passed AND no human_pending → flip `pipeline_phase: complete`, advance to retire.
     - All shell checks passed AND human_pending non-empty → surface human items, return control. On Human reply re-invocation: interpret reply, set `verification_state.human_verdict`, branch (all_pass → complete; rejected → revision_needed → drafted).
   - `complete` → invoke `Skill("retire", "<path>")` directly OR dispatch `plan-retirer` foreground (skill design choice). After successful retire: commit+push.

   **Re-entry idempotency:** any orchestrator invocation must read `pipeline_phase` and `audit_state` from disk first to determine the current state. Re-invocation on the same state must not double-dispatch (e.g. if parent Claude accidentally invokes orchestrator twice on the same return message, the second invocation should detect "no state change since last action" and no-op). Implement via a simple "if outcome was already recorded for this stage and state hasn't advanced, do nothing" check.

4. Document the **parent/children interaction**:
   - When the active PLAN has non-empty `triggers_plans:`, orchestrator pauses parent advancement until all listed children reach `status: done`.
   - Parent's `pipeline_phase` does not flip to `executing` while children are running.
   - Mechanism: orchestrator reads `triggers_plans:` on parent and polls children's `status` (or waits for explicit signal).

5. Document **trigger-phrase disambiguation**: confirm "create plan file" (existing `write-bus-plan` trigger) does not collide with "let's make a plan" (new `plan-pipeline` trigger). If collision exists, adjust one or the other; document the decision.

5a. Document **git operation handling at every milestone** (parent decisions 13, 22). Orchestrator commits + pushes at:
   - Plan-writer success → commit message: `plan-pipeline: drafted <plan-filename>` (write phase).
   - Human revision detected (file mtime advanced OR explicit re-invocation after surfaced revision_needed) → commit before re-dispatching audit, message: `plan-pipeline: human-revised <plan-filename> (audit_state.{stage}_iterations={N})`.
   - Each audit `outcome` write to `audit_state` frontmatter → commit so loop state is durable in origin. Message: `plan-pipeline: audit_state update — <stage>:<outcome>`.
   - Plan-executor success → commit + push the execution work, message: `plan-pipeline: executed <plan-filename>`.
   - Plan-retirer success → commit + push, message: `plan-pipeline: retired <plan-filename>`.
   - On `outcome: exception` halt → commit + push current state with `WIP: pipeline halted at <phase> for <plan-filename> — see diagnostics`.

   **Bootstrap exception:** during the bootstrap (parent PLAN 202605011400 execution), the Human commits manually after each `execute-plan` invocation. The orchestrator only owns git for *future* PLANs.

   **Failure handling:** if commit or push fails, orchestrator emits `outcome: exception` with diagnostics; pipeline halts. Never use `--no-verify`, `--force`, `--force-with-lease`, or bypass signing.

5b. Document **Human-facing surfacing with decision-triage** (parent decision 15): every point at which the orchestrator surfaces a question or decision-list to the Human (e.g. `drafted` blockers, `[Human]` checkpoints, halt-on-failure escalations) must apply the triage classification first. Implementation: orchestrator wraps any Human-prompt with a triage pre-pass; only Real-judgement-call items become questions, the rest are listed as "noted" without requesting an answer.

5c. Document the **routing table** (parent decision 20): for each phase, name the agent dispatched, the input fields it receives (sourced from PLAN frontmatter, prior-phase output, or Human input), and the output consumer (where outputs feed next). Validate upstream→downstream contract match before each dispatch.

5d. Document the **subagent message processing protocol** (parent decisions 18, 19): structured message format (lean JSON: `{type: "milestone"|"anomaly"|"completion", agent: "<name>", plan_path: "...", payload: {...}}`); message router that maps message → orchestrator action; `pipeline_phase` updates triggered by completion messages; full-stop semantics on anomaly messages (halt all in-flight subagents — mechanism TBD per probe-2; if no clean cancel, mark PLAN halted and ignore subsequent returns).

5e. Document **conversational re-entry** (parent decision 18): the executing→complete transition (and any other transition triggered by a background-agent completion) happens when the agent's return message arrives at the parent conversation. The orchestrator's SKILL.md must instruct the parent Claude that on receiving such a message, it should re-invoke `plan-pipeline` against the affected PLAN. Idempotency: re-entry must read `pipeline_phase` from disk and resume correctly even if multiple messages have queued up.

6. Write the file(s).

7. **[Human]** Smoke test: invoke `Skill("plan-pipeline")` with a contrived empty PLAN; verify it correctly identifies the phase and dispatches (without actually creating real artefacts).

## Verification

- [ ] `.claude/skills/plan-pipeline/SKILL.md` exists with valid frontmatter and activation triggers
- [ ] Phase state machine documented (inline or in `references/phase-state-machine.md`)
- [ ] All five phases covered with success/failure transitions
- [ ] Ad-hoc PLAN default (`absent → drafted`) documented
- [ ] Background execution (`run_in_background: true`) documented for `executing` phase
- [ ] Parent/children interaction (triggers_plans, waiting state) documented
- [ ] Trigger-phrase collision check performed; outcome documented
- [ ] Git operation handling documented: granularity, bootstrap exception, failure handling (per parent decision 13)
- [ ] Human-facing surfacing applies decision-triage (decision 15): only Real-judgement-call items become questions
- [ ] Decision 16 compliance: SKILL.md contains `<preconditions>`, `<output_schema>`, AND a documented set of inter-skill handoff schemas (what each subagent receives and returns)
- [ ] Decision 18: hybrid background mode documented (only plan-executor is background); single conversational re-entry point (executor completion); idempotent re-entry from disk-state
- [ ] Decision 19: subagent-driven exception handling — orchestrator routes on `outcome` enum; no streaming, no cancellation; kanban full stop on `outcome: exception`
- [ ] Decision 20: `outcome` enum (success | revision_needed | exception) is the load-bearing contract; routing table documented for each (phase, outcome) pair
- [ ] Decision 21: explicit audit-loop state machine documented (durable audit_state frontmatter; per-stage iteration counters; MAX_ITERATIONS=5; rerun-vs-continue signals from outcome enum)
- [ ] Decision 22: orchestrator commits + pushes at every milestone (write, Human revision, audit_state update, executor success, retire, exception halt) with descriptive commit messages
- [ ] Decision 25: outcome-verifying phase implemented between executing→complete; runs verify:/acceptance: shell commands; surfaces verify: human items via structured prompt; verification_state durable on disk; human_verdict re-entry flow documented (pending → all_pass | rejected)
- [ ] Visual feedback: phase transitions echoed to Human conversationally; LOG Status Table updated per phase
- [ ] Smoke test passes

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
