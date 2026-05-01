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
  sufficiency_iterations: 3
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: success
---

## Objective
Exercise the freshly-built `plan-pipeline` orchestrator end-to-end against a small but real target — designing and creating a tiny utility skill called `note-jot` (writes a one-line note with a timestamp to a daily log file). The dogfood validates phase transitions (including the transient `outcome-verifying` phase per decision 25), the two-stage audit loop with deliberate failure-path exercise (decision 21), pinned-model subagent dispatch with hybrid background mode (decision 18), and the orchestrator-owned git milestone discipline (decision 22).

## Context

Spawned from parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md`. The dogfood target is `note-jot` rather than a contrived no-op so that the ideation phase (Clarify → Survey → Converge) has real surface to exercise — what does the skill take as input? where does the daily log live? what format? — exactly the kinds of questions `ideate` is designed to discipline.

**Why `note-jot`:** small enough to complete in one dogfood run, but real enough that the Human will have at least one Survey-phase decision (e.g. log file location) and the `ideate → plan-writer → audit-sufficiency → audit-haiku-safe → plan-executor → outcome-verifying → plan-retirer` sequence produces a real artefact at the end (a working `note-jot` skill).

**Execution context** (parent decision 10, hard rule): the drafting phase of this dogfood requires interactive `ideate` in the parent session. A background `plan-executor` subagent CANNOT drive the dogfood end-to-end because ideate is parent-session-only. The dogfood is therefore executed by the Human-in-parent-Claude pairing acting AS orchestrator, walking the steps below by hand. This is the correct execution mode for any dogfood that exercises ideate; subagent execution is reserved for non-interactive PLANs.

## Steps

1. **Trigger plan-pipeline activation.** The Human, in the parent session, types a natural-language trigger (e.g. "let's plan a note-jot utility skill"). Confirm the Skill matcher routes to `plan-pipeline` (not `write-bus-plan`'s "create plan file" trigger). Confirm the orchestrator activates and reads phase state from disk. Since no note-jot PLAN exists yet, the orchestrator enters branch 4A (`drafting`) and invokes `Skill("ideate")` directly in the parent session — never via a subagent.

2. **Phase: drafting** — orchestrator runs `ideate` in the parent session. Walk Clarify → Survey → Converge with the Human deciding scope and shape of `note-jot`.
   - At clarify-locked and survey-converged moments, the orchestrator dispatches `plan-writer` (foreground subagent) to write/update the note-jot PLAN file incrementally.
   - Verify the new note-jot PLAN's `pipeline_phase: drafting` is set on the first plan-writer write.
   - Each plan-writer return → orchestrator commits + pushes (`plan-pipeline: drafting checkpoint <plan-filename>`).
   - **Deliberate audit-loop exercise (per decision 21):** in the first complete draft of the note-jot PLAN (at survey-converged), intentionally seed a sufficiency-grade blocker — e.g. omit the Objective's deliverable claim, or leave Verification with no `acceptance:` item. This is so Step 4 below forces the audit loop's `revision_needed` re-dispatch path on the first iteration; the second iteration after revision should converge.

3. **Phase transition: drafting → drafted** — Human signals ideation is done (any phrase that re-triggers `plan-pipeline`). On re-invocation, the orchestrator reads disk, sees the note-jot PLAN has all expected sections, flips `pipeline_phase: drafted`, commits + pushes (`plan-pipeline: drafted <plan-filename>`).

4. **Phase: drafted (audit loop, two stages, with forced failure path)** — orchestrator drives the audit loop per decision 21:
   - Iteration 1, sufficiency stage: dispatch `sufficiency-auditor` (foreground, opus). Because the seed blocker from Step 2 is present, the auditor returns `outcome: revision_needed` with `Blockers: ≥1`. Orchestrator writes `audit_state` (sufficiency_iterations: 1, last_stage: sufficiency, last_outcome: revision_needed); commits + pushes (`plan-pipeline: audit_state update — sufficiency:revision_needed`); surfaces the review with decision-15 triage; returns. Verify: counter incremented, frontmatter durable, Human surface clean.
   - Human revises the seed blocker on the note-jot PLAN.
   - Iteration 2, sufficiency stage (after PLAN mtime advances): dispatch `sufficiency-auditor` again. This time `outcome: success`. Orchestrator updates `audit_state` (sufficiency_iterations: 2, last_stage: sufficiency, last_outcome: success); commits + pushes; chains to the plan-safety stage WITHIN the same invocation (allowed, no `pipeline_phase` boundary crossed).
   - Iteration 1, plan-safety stage: dispatch `plan-safety-auditor` (foreground, sonnet). Verify it refuses to run if sufficiency hasn't passed (precondition); since sufficiency just passed, it proceeds. On `outcome: success`: orchestrator updates `audit_state`, flips `pipeline_phase: drafted → checked`, commits + pushes (`plan-pipeline: checked <plan-filename>`), returns. (If plan-safety returns `revision_needed` instead, exercise that loop too; sufficiency does NOT re-run.)
   - Verify throughout: `audit_state` frontmatter is durable on disk (re-readable across invocations); MAX_ITERATIONS=5 not breached; orchestrator never silently re-runs.

5. **Phase transition: checked → executing** — orchestrator reads note-jot's `assigned_to:` (default haiku for note-jot's mechanical work), routes to `plan-executor` (haiku tier per the executor-tier table), flips `pipeline_phase: executing`, sets `status: in-progress`, commits + pushes (`plan-pipeline: executing <plan-filename>`). Dispatches via `Agent({subagent_type: "plan-executor", ..., run_in_background: true})` — passing the `run_in_background: true` parameter on the Agent tool call. Per parent PLAN 202605011400 decision 18 (amended 2026-05-01): the Agent tool's `run_in_background` parameter is the load-bearing mechanism for backgrounding; the agent's frontmatter `background: true` is preserved as authoring documentation but not relied upon for behaviour. Confirm the dispatch returns control to parent immediately. Surface "Executor dispatched in background; resume on completion" to the Human.

6. **Re-entry on executor completion** — when the `plan-executor` background subagent returns its completion message, parent-Claude observes it and re-invokes `Skill("plan-pipeline")` against the note-jot PLAN. The orchestrator, on re-entry, reads `last_executor_outcome` frontmatter (decision 24) — populated by `execute-plan`'s workflow on completion. Branch:
   - `outcome: success` → flip `pipeline_phase: outcome-verifying` (decision 25), commit + push, continue into Step 7.
   - `outcome: revision_needed` → revert to `drafted`, reset `audit_state`, set `status: needs-revision`, commit + push, surface diagnostics, return. (For dogfood, expect success.)
   - `outcome: exception` → kanban halt; commit + push WIP; surface diagnostics; return.

7. **Phase: outcome-verifying** (per decision 25) — orchestrator runs all `verify:` and `acceptance:` shell commands from the note-jot PLAN's Verification section; tallies pass/fail; collects any `verify: human` items into `verification_state.human_pending`; writes `verification_state` to note-jot PLAN frontmatter; commits + pushes (`plan-pipeline: outcome-verification ran for <plan-filename>`). Branch:
   - All shell pass + no human_pending → flip `pipeline_phase: complete`; commit + push.
   - All shell pass + human_pending non-empty → surface the structured prompt ("Outcome-verification: N auto-checks passed. M items need your eyeball: [list]. Reply 'all good' or describe what is wrong."); apply decision-15 triage; return; on Human reply re-entry, set `human_verdict: all_pass | rejected`.
   - Any shell failure → override executor success; outcome = revision_needed; revert to `drafted`; reset `audit_state`; commit + push.
   - Verify the dogfood: at least one `acceptance:` item on note-jot's Verification was actually executed (greppable in the dogfood's git history via the commit message templates).

8. **Phase: complete → retire** — orchestrator dispatches `plan-retirer` (foreground subagent, haiku) on the note-jot PLAN. Verify note-jot PLAN moves to `Retired/`, `.gitignore` excludes the path, orchestrator commits + pushes (`plan-pipeline: retired <plan-filename>`).

9. **Capture findings in this PLAN's Executor Notes:**
   - Trigger-routing: did "let's plan a note-jot utility skill" route to `plan-pipeline` or get misrouted to `write-bus-plan`?
   - Each phase transition: clean / dirty? frontmatter durable across invocations?
   - Audit loop failure path: did iteration 1 force `revision_needed`? did iteration 2 converge? did counters increment? did sufficiency NOT re-run if only plan-safety failed?
   - `plan-writer` dispatch: any failures? any duplicate-filename / month-mismatch issues?
   - Background dispatch: did the `run_in_background: true` parameter on the Agent invocation make the call non-blocking? Did parent stay responsive? Did the completion message arrive cleanly? Did re-entry pattern actually fire (or did parent-Claude need explicit prompting)? (This exercises decision 18's amended mechanism.)
   - Outcome-verification: did `verify:`/`acceptance:` shell commands actually run? was `verification_state` written durably?
   - `<pipeline-result>` parsing: any malformed returns? any agent that produced parse-tripping prose before the block?
   - Subagent skill-preload (`skills:` frontmatter): did each agent have what it needed, or did anything fail because a transitive-closure skill wasn't preloaded?
   - Decision-15 triage application: was every Human-facing surface actually triaged?
   - Git milestone discipline: did the orchestrator commit + push at every documented milestone? any missed?
   - Recommendations for fixes/iterations on the new skills (these feed the post-dogfood fix-PLAN bundling re-entry hook, slim, frontmatter-ownership, outcome-verify-safe-default).

## Verification

- [ ] Note-jot PLAN file (separate, for the dogfood target skill) created via the pipeline and exists in `Bus/`
      `verify: ls Bus/*PLAN_note-jot*.md 2>/dev/null | head -1`
- [ ] All six `pipeline_phase` transitions completed on the note-jot PLAN: drafting → drafted → checked → executing → outcome-verifying → complete (then retired)
      `verify: human`
- [ ] note-jot PLAN's `audit_state.sufficiency_iterations` reached at least 2 (forced failure path was exercised)
      `verify: grep -E "^\s+sufficiency_iterations: [2-9]" Bus/*PLAN_note-jot*.md Retired/*PLAN_note-jot*.md 2>/dev/null | head -1`
- [ ] note-jot skill exists at `.claude/skills/note-jot/SKILL.md` and a smoke invocation produces a timestamped line in the daily log
      `acceptance: test -f .claude/skills/note-jot/SKILL.md && grep -q "^name: note-jot" .claude/skills/note-jot/SKILL.md`
- [ ] note-jot PLAN retired to `Retired/`
      `verify: ls Retired/*PLAN_note-jot*.md 2>/dev/null | head -1`
- [ ] orchestrator's milestone commits are visible in git history (sample three: drafted, checked, retired)
      `acceptance: git log --oneline -n 50 | grep -E "plan-pipeline: (drafted|checked|retired)" | wc -l | awk '{exit !($1>=3)}'`
- [ ] dogfood findings captured in this PLAN's Executor Notes with explicit pass/fail per Step 9 question
      `verify: human`
- [ ] any bugs surfaced are filed as content in this PLAN's Executor Notes "Recommendations" subsection (will feed the post-dogfood fix-PLAN — re-entry hook, slim, frontmatter-ownership, outcome-verify-safe-default)
      `verify: human`

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
