# Plan-Pipeline Dispatch Procedure

One invocation = one walk through this procedure. The orchestrator reads disk, decides what to do, executes one or more phase transitions, commits + pushes after each milestone, and returns control.

**Phase-boundary chaining (F9 from PLAN 202605011900, 2026-05-01):** the orchestrator MAY chain phase transitions within a single invocation (e.g. `drafted → checked → executing`) when running in a continuous parent session, provided each transition produces a milestone commit + push. Audit-loop iterations stay within `drafted`. Re-entry idempotency is preserved by reading disk on every invocation — the chaining relaxation does not weaken the on-disk-state contract. The Human or parent-Claude triggers the next invocation if the chain pauses (e.g. background executor dispatched, Human-pending verification surfaced).

Per-phase routing tables, commit-message templates, and frontmatter mutation cheat sheet live in [../references/phase-state-machine.md](../references/phase-state-machine.md). This file is the procedural narrative; the reference is the lookup.

---

## Step 1: Resolve target PLAN

**If `plan_path` was supplied:**
- Read the file. If unreadable → emit `exception` (orchestrator-side; commit not required since nothing changed).
- Parse frontmatter. If malformed → `exception`.

**If only `request` was supplied (fresh entry):**
- Detect intent. If the request contains an existing PLAN reference (e.g. "resume the pipeline on PLAN_xyz"), resolve to its path and proceed as above.
- Otherwise, treat as a fresh planning request → enter Step 4 (`drafting` phase) with no PLAN file yet.

**If both supplied:** prefer `plan_path`; treat `request` as additional context only.

**If neither:** emit `exception`. The orchestrator does not invent a target.

---

## Step 2: Read durable state

Extract from PLAN frontmatter (when a PLAN exists):

```
pipeline_phase             # absent or empty → treat as drafted (bus-conventions ad-hoc default)
audit_state:
  sufficiency_iterations    # default 0
  plan_safety_iterations    # default 0
  last_stage                # none | sufficiency | plan_safety
  last_outcome              # none | success | revision_needed | exception
last_executor_outcome:
  outcome                   # success | revision_needed | exception
  outcome_subtype           # done | partially-complete | blocked | needs-revision
  executed
  diagnostics_summary
verification_state:
  state_pass / state_fail
  acceptance_pass / acceptance_fail
  human_pending             # list
  human_verdict             # pending | all_pass | rejected
status                       # ready | in-progress | done | partially-complete | blocked | needs-revision
triggers_plans               # list of child PLAN filenames
assigned_to                  # haiku (default) | sonnet | opus | other
```

Treat any absent field as the documented default.

---

## Step 3: Idempotency and parent/children gates

**Idempotency check.** If the durable state shows the action for the current `pipeline_phase` has already been taken since the last meaningful change AND no new outcome is recorded since, return: "already at <phase>; nothing to do." This protects against parent-Claude double-invocation on a single trigger.

Concrete checks (applied in order):
- `pipeline_phase: complete` AND PLAN already retired → no-op.
- `pipeline_phase: executing` AND no `last_executor_outcome` recorded → executor still running (or no completion message arrived); no-op.
- `pipeline_phase: outcome-verifying` AND `verification_state.human_verdict: pending` AND no fresh `human_reply` supplied → no-op (waiting on the Human).
- `pipeline_phase: drafted` AND `audit_state.last_outcome: revision_needed` AND PLAN file mtime not advanced since the audit_state write → no-op (waiting on Human revision).

**Children gate.** If `triggers_plans` is non-empty, read each child PLAN's `status`. If any child is non-terminal (not in {`done`, `partially-complete`, `cancelled`, `closed`}), surface "Paused for children: <list>" with decision-15 triage, return. Do NOT advance the parent's `pipeline_phase`.

If gates pass, proceed to Step 4.

---

## Step 4: Dispatch by phase

Each branch below mutates frontmatter and commits as documented in [../references/phase-state-machine.md](../references/phase-state-machine.md). Only one branch executes per invocation (except the audit loop within `drafted`, which may dispatch one audit + record outcome + return).

### 4A. `drafting` (or no PLAN yet)

**Entry condition:** request without an existing PLAN, or a PLAN with `pipeline_phase: drafting`.

1. Invoke `Skill("ideate", request_or_existing_path)` directly in the parent session. **Never via a subagent.**
2. The ideate arc walks Clarify → Survey → Converge with the Human. At checkpoint moments (clarify-locked, survey-converged), dispatch `plan-writer` foreground via the Agent tool to write or update the PLAN file. Each successful `plan-writer` return → commit + push (template `plan-pipeline: drafted <plan-filename>` or `... drafting checkpoint <plan-filename>`).
3. When the Human signals ideation closed (any phrase that re-triggers the pipeline matcher, or an explicit "ready to audit"), the next orchestrator invocation flips `pipeline_phase: drafted` (Step 4 reads disk and falls into branch 4B). The ideate skill itself does not flip the phase.
4. RESEARCH/ADVICE escapes during ideate: ideate invokes `write-bus-input` directly; resulting filenames land in the PLAN's `linked_inputs:` on the next `plan-writer` checkpoint. Commit + push after `write-bus-input` returns (template `plan-pipeline: drafting input <filename>`).

Return control to parent after each `plan-writer` checkpoint (the arc itself is conversational; the orchestrator re-enters when the Human's next turn re-triggers the pipeline).

### 4B. `drafted` (audit loop)

**Read `audit_state` and follow the lookup in references/phase-state-machine.md → "Audit-loop dispatch table".** The decision tree:

| `last_stage` | `last_outcome`     | Next action                                                                                     |
|--------------|--------------------|-------------------------------------------------------------------------------------------------|
| none         | none               | Dispatch `sufficiency-auditor` (foreground).                                                    |
| sufficiency  | success            | Dispatch `plan-safety-auditor` (foreground).                                                    |
| sufficiency  | revision_needed    | Awaiting Human revision; if PLAN mtime advanced since audit_state write, re-dispatch `sufficiency-auditor`; else no-op. |
| plan_safety  | success            | Both passed → flip `pipeline_phase: checked`; commit + push; return. (No further dispatch this invocation.) |
| plan_safety  | revision_needed    | Awaiting revision; re-dispatch `plan-safety-auditor` after PLAN mtime advances. (Sufficiency does not re-run.) |
| any          | exception          | Kanban halt — already committed when exception was first emitted. Re-entry is a no-op until Human takes action. |

**Per dispatch:**
1. Increment the appropriate iteration counter (`audit_state.sufficiency_iterations++` or `plan_safety_iterations++`).
2. **If counter > 5:** do NOT dispatch. Orchestrator emits its own `outcome: exception` ("audit loop did not converge after 5 iterations — Human review required"). Update `audit_state` accordingly, commit + push WIP, surface diagnostics, return.
3. Otherwise: announce the iteration to the Human ("Audit iteration <N>: dispatching <agent>"), dispatch foreground via `Agent({subagent_type: "<agent>", ...})`. Block until return.
4. Parse the LAST `<pipeline-result>` block in the agent's return (text-scan for opening tag, find matching closing tag, extract the JSON code fence, parse). If absent or malformed → orchestrator emits `exception` (commit + push WIP, surface diagnostics, return).
5. Branch on `outcome`:
   - `success` → write `audit_state.last_stage = <stage>`, `last_outcome: success`. Commit + push (template `plan-pipeline: audit_state update — <stage>:success`). Re-enter Step 4B from the top within the same invocation only if the next stage is the OTHER audit (sufficiency → plan-safety transition). Once `plan_safety` succeeds, flip `pipeline_phase: checked` and commit + push under that template, then return.
   - `revision_needed` → write `audit_state.last_stage = <stage>`, `last_outcome: revision_needed`. Commit + push (`plan-pipeline: audit_state update — <stage>:revision_needed`). Apply decision-15 triage to `payload.triaged_human_items` if present, surface `payload.review_text` + iteration counts, instruct the Human "Revise the PLAN and re-invoke me to re-audit." Return.
   - `exception` → kanban halt. Commit + push WIP with `payload.diagnostics`. Surface diagnostics. Return.

**One exception to "one phase per invocation":** within `drafted`, the orchestrator may chain sufficiency-success → plan-safety-dispatch in a single invocation (commit between). This is allowed because no `pipeline_phase` boundary is crossed — both stages are within `drafted`.

### 4C. `checked`

**Single transition: dispatch executor.**

1. Read PLAN's `assigned_to` frontmatter. Map per the executor-tier table in references/phase-state-machine.md:
   - `haiku`, empty, or unrecognised → `plan-executor`.
   - `sonnet` → `plan-executor-sonnet`.
   - `opus` → `plan-executor-opus`.
2. Flip `pipeline_phase: executing`. Set `status: in-progress`. Commit + push (`plan-pipeline: executing <plan-filename>`).
3. Dispatch the executor with `run_in_background: true`. The Agent call returns an agent ID; the orchestrator does NOT wait. Surface to the Human: "Executor <name> dispatched in background. I'll resume when it completes."
4. Return control to parent. The executor's completion message arriving in parent's conversation is the re-entry cue.

### 4D. `executing`

**Re-entry path.** Parent-Claude observed the executor's completion message and re-invoked this skill with `plan_path`. The orchestrator does not poll; this branch only runs on re-entry.

1. Read `last_executor_outcome.outcome` from PLAN frontmatter (the executor's `execute-plan` workflow writes this — decision 24).
2. If absent or stale (executed date older than the dispatch) → no-op. The completion message likely fired before the executor finished; wait for the next message.
3. Parse the most recent `<pipeline-result>` block from the executor's return (parent-Claude supplies it implicitly via the conversation; if not directly available, treat the frontmatter `last_executor_outcome` as canonical — that is the durable record per decision 24).
4. Branch:
   - `outcome: success` → flip `pipeline_phase: outcome-verifying`. Commit + push (`plan-pipeline: outcome-verifying <plan-filename>`). Continue into branch 4E within the same invocation.
   - `outcome: revision_needed` → revert `pipeline_phase: drafted`. Reset `audit_state` to fresh (`last_stage: none`, `last_outcome: none`, counters to 0) — content has changed materially, so prior audits are stale. Set `status: needs-revision`. Commit + push (`plan-pipeline: executor revision_needed <plan-filename>`). Surface executor's `diagnostics_summary` + Executor Notes pointer with decision-15 triage. Return.
   - `outcome: exception` → kanban halt. Commit + push WIP (`WIP: pipeline halted at executing for <plan-filename> — see diagnostics`). Surface diagnostics. Return.

### 4E. `outcome-verifying` (per decision 25)

1. Read PLAN's `## Verification` section. Extract every line annotated with `verify:`, `acceptance:`, or `verify: human`.
2. For each `verify:` shell command: run via Bash. Tally pass (exit 0) / fail.
3. For each `acceptance:` shell command: run via Bash. Tally pass / fail.
4. For each `verify: human` item: capture the prose description into `verification_state.human_pending`.
5. Write `verification_state` frontmatter:
   ```
   verification_state:
     state_pass: <int>
     state_fail: <int>
     acceptance_pass: <int>
     acceptance_fail: <int>
     human_pending: [<list of prose descriptions>]
     human_verdict: pending
   ```
   Commit + push (`plan-pipeline: outcome-verification ran for <plan-filename>`).
6. Branch:
   - **Any `verify:`/`acceptance:` shell failed** → outcome = revision_needed (overrides executor's success). Revert `pipeline_phase: drafted`, reset `audit_state` to fresh, set `status: needs-revision`. Commit + push (`plan-pipeline: outcome-verification failed — reverting to drafted for <plan-filename>`). Surface failed-assertion list (commands + exit codes) with decision-15 triage. Return.
   - **All shell checks passed AND `human_pending` empty** → flip `pipeline_phase: complete`. Commit + push (`plan-pipeline: complete <plan-filename>`). Continue into branch 4F.
   - **All shell checks passed AND `human_pending` non-empty** → surface the structured prompt:
     ```
     Outcome-verification: <N> auto-checks passed. <M> items need your eyeball:
     - [item 1 prose]
     - [item 2 prose]
     ...
     Reply 'all good' to mark them passed, or describe what is wrong.
     ```
     Apply decision-15 triage to the items first (most are likely Real-judgement-call here, but classify and surface only the genuine asks). Return. The Human's reply re-triggers the orchestrator (`human_reply` input or, if parent-Claude re-invokes without it, parse the most recent Human message). On re-entry while `pipeline_phase: outcome-verifying` AND `human_verdict: pending`:
     - Affirming reply ("all good", "yes", "approved", "pass") → set `human_verdict: all_pass`. Commit + push (`plan-pipeline: human verification passed for <plan-filename>`). Flip `pipeline_phase: complete`, continue to 4F.
     - Rejecting/critique reply → set `human_verdict: rejected` with the Human's text in `verification_state.human_diagnostics`. Commit + push. Treat as revision_needed: revert `pipeline_phase: drafted`, reset `audit_state`, set `status: needs-revision`. Surface diagnostics. Return.
     - Ambiguous reply → re-prompt once: "Please reply with 'pass' or 'fail [reason]'." Return. Next re-entry that still finds ambiguity → emit `exception` with diagnostics ("could not interpret Human verdict after one clarifying prompt").

### 4F. `complete` (retire)

1. Dispatch `plan-retirer` foreground via the Agent tool with the PLAN path.
2. Parse the `<pipeline-result>` block.
3. Branch:
   - `success` → commit + push (`plan-pipeline: retired <plan-filename>`). Surface "Pipeline complete: <plan-filename> retired." Return.
   - `exception` → kanban halt. Commit + push WIP. Surface diagnostics. Return.

---

## Step 5: Children-aware advancement (when this PLAN spawns children)

If the active PLAN's `triggers_plans:` was populated during `drafting` (e.g. ideation produced a parent-of-plans), the children-gate in Step 3 prevents the parent from auto-advancing past `checked` until children reach terminal status. The Human typically:
1. Runs the orchestrator on the parent through `drafted → checked` (parent's own design is audited).
2. Manually flips children's `status: blocked → ready` (or runs the orchestrator on each child if they themselves use the pipeline).
3. Once all children terminal, re-invokes the orchestrator on the parent — Step 3's children gate now passes, and 4C dispatches the parent's executor.

The orchestrator does not poll children. Re-entry is the cue.

---

## Step 6: Bootstrap exception

While the parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md` is itself executing (its own `pipeline_phase` is non-`complete`), the orchestrator does NOT manage git for it — the Human commits manually after each `execute-plan` call, per the bootstrap instructions in that PLAN's Constraints section. This branch is detected by inspecting whether the active PLAN is the parent bootstrap file.

For all other PLANs, full git milestone discipline applies.

---

## Step 7: Always commit (or fail loud) and return

After any phase work in Step 4, before returning control:
- If any frontmatter or filesystem change happened, the appropriate `git add -A && git commit -m "<template>" && git push` was already run inline (per the templates).
- If `git push` failed at any point, the orchestrator already emitted `exception` and committed WIP locally (push will be retried on Human action).
- If nothing changed (idempotent no-op), no commit needed.

Return control to parent-Claude with a one- or two-sentence summary of what just happened (which phase advanced, or which prompt is now in front of the Human, or that this was a no-op).
