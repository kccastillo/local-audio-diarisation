# Plan-Pipeline Phase State Machine

Lookup tables and templates for the orchestrator. The procedural narrative is in [../workflows/dispatch.md](../workflows/dispatch.md); this file is the reference.

---

## Phase enum

```
drafting → drafted → checked → executing → outcome-verifying → complete
```

`outcome-verifying` is a transient phase that always sits between `executing` and `complete` (per parent PLAN 202605011400 decision 25). Never skipped.

**Ad-hoc PLAN default** (per `bus-conventions.md`): `pipeline_phase` absent or empty → treat as `drafted`. Do NOT treat as `drafting` — that would re-ideate an already-authored PLAN.

---

## (Phase, Outcome) → Action routing table

The load-bearing contract (decisions 19, 20). `outcome` always comes from a `<pipeline-result>` block (subagent return) or from the orchestrator itself (MAX_ITERATIONS, git failure, malformed return).

| Phase                | Outcome           | Next action                                                                                                  |
|----------------------|-------------------|--------------------------------------------------------------------------------------------------------------|
| drafting             | (n/a — interactive) | Continue arc; `plan-writer` checkpoint commits.                                                            |
| drafted              | success           | Audit-loop dispatch table below.                                                                             |
| drafted              | revision_needed   | Update `audit_state`; commit; surface review; await Human revision; return.                                  |
| drafted              | exception         | Kanban halt: commit + push WIP; surface diagnostics; return.                                                 |
| checked              | (n/a)             | Flip → `executing`; commit; dispatch executor (background); return.                                          |
| executing            | success           | Flip → `outcome-verifying`; commit; continue into outcome-verification.                                      |
| executing            | revision_needed   | Revert → `drafted`; reset `audit_state`; set `status: needs-revision`; commit; surface; return.              |
| executing            | exception         | Kanban halt: commit + push WIP; surface; return.                                                             |
| outcome-verifying    | all-pass, no human_pending | Flip → `complete`; commit; dispatch `plan-retirer`.                                                |
| outcome-verifying    | shell-fail        | Override executor success → revision_needed; revert → `drafted`; reset `audit_state`; commit; surface; return. |
| outcome-verifying    | human_pending     | Surface structured prompt; commit `verification_state` write; return; await Human reply.                     |
| outcome-verifying    | human_verdict: all_pass | Flip → `complete`; commit; dispatch `plan-retirer`.                                                    |
| outcome-verifying    | human_verdict: rejected | Override → revision_needed; revert → `drafted`; reset `audit_state`; commit; surface; return.          |
| complete             | success           | `plan-retirer` returned success → commit retire; surface "pipeline complete"; return.                        |
| complete             | exception         | Retire failed → kanban halt; commit WIP; surface; return.                                                    |
| any                  | malformed return  | Orchestrator emits `exception`; commit + push WIP; surface "subagent return malformed"; return.              |
| any                  | git failure       | Orchestrator emits `exception`; commit (local) WIP if possible; surface "git operation failed: <op>"; return. |

---

## Audit-loop dispatch table (within `drafted`)

Driven by the durable `audit_state` frontmatter. Per parent decision 21.

| `audit_state.last_stage` | `audit_state.last_outcome` | Next dispatch                                | Iteration counter incremented |
|--------------------------|----------------------------|----------------------------------------------|-------------------------------|
| none                     | none                       | `sufficiency-auditor`                        | `sufficiency_iterations`      |
| sufficiency              | revision_needed            | `sufficiency-auditor` (after Human revision) | `sufficiency_iterations`      |
| sufficiency              | success                    | `plan-safety-auditor`                        | `plan_safety_iterations`      |
| plan_safety              | revision_needed            | `plan-safety-auditor` (after Human revision; sufficiency NOT re-run) | `plan_safety_iterations` |
| plan_safety              | success                    | (none) — flip `pipeline_phase: checked`; exit loop. | (n/a)                  |
| any                      | exception                  | (none) — kanban halt already in effect.      | (n/a)                         |

**MAX_ITERATIONS = 5** per stage. If incrementing would exceed 5, do NOT dispatch — orchestrator emits `outcome: exception` itself with diagnostics_summary `"audit loop did not converge after 5 iterations on <stage>"`. Commit + push WIP. Return.

**Re-revision detection.** "After Human revision" means PLAN file mtime is later than the last `audit_state` commit timestamp. If mtime is NOT advanced and `last_outcome: revision_needed`, the orchestrator no-ops (waiting on Human).

**On commit after Human revision** (before re-dispatching the audit): `git add -A && git commit -m "plan-pipeline: human-revised <plan-filename> (audit_state.<stage>_iterations=<N>)" && git push`.

---

## Agent dispatch table

| Phase / sub-phase                                | Agent name                                                                | Mode       | Input fields                                              | Output payload (relevant fields used by orchestrator) |
|--------------------------------------------------|---------------------------------------------------------------------------|------------|-----------------------------------------------------------|------------------------------------------------------|
| drafting (checkpoints)                           | `plan-writer`                                                             | foreground | `{plan_content, target_filename, mode: create|update}`    | `{filename_written, action, log_updated}`            |
| drafting (RESEARCH/ADVICE)                       | (skill `write-bus-input`, parent-session direct, NOT a subagent)          | n/a        | `{type: RESEARCH|ADVICE, topic, body}`                     | written filename                                     |
| drafted (audit, sufficiency)                     | `sufficiency-auditor`                                                     | foreground | `{plan_path}`                                              | `{blockers_count, review_text, triaged_human_items}` |
| drafted (audit, plan-safety)                     | `plan-safety-auditor`                                                     | foreground | `{plan_path}`                                              | `{blockers_count, review_text}`                      |
| executing (executor dispatch by `assigned_to`)   | `plan-executor` (haiku, default) / `plan-executor-sonnet` / `plan-executor-opus` | **background** | `{plan_path}`                                              | `{outcome_subtype, executor_notes, files_modified}`  |
| outcome-verifying (shell + human)                | (orchestrator runs Bash directly — no subagent)                            | n/a        | n/a                                                        | n/a                                                  |
| complete (retire)                                | `plan-retirer`                                                            | foreground | `{plan_path}`                                              | `{retired_path, gitignore_updated}`                  |

**Executor tier selection** (PLAN-driven, decision 8):

| PLAN frontmatter `assigned_to:`        | Agent dispatched               |
|----------------------------------------|--------------------------------|
| `haiku`, empty, absent, or unrecognised | `plan-executor` (haiku)        |
| `sonnet`                               | `plan-executor-sonnet`         |
| `opus`                                 | `plan-executor-opus`           |

`assigned_to` is a guideline — Human may override per PLAN. Per decision 16's anti-monolithic principle, prefer decomposition over `opus` execution; `plan-executor-opus` is an escape hatch, not a default.

---

## Frontmatter mutation cheat sheet

The orchestrator may write these fields directly (without going through `plan-writer`) because they are orchestration-owned, not body content. Use the Edit tool on the PLAN file's YAML frontmatter block.

### `pipeline_phase`
String enum. Mutated at every phase transition. Values: `drafting | drafted | checked | executing | outcome-verifying | complete`.

### `audit_state`
```yaml
audit_state:
  sufficiency_iterations: 0
  plan_safety_iterations: 0
  last_stage: none           # none | sufficiency | plan_safety
  last_outcome: none         # none | success | revision_needed | exception
```
Reset to fresh (all zeros / `none`) when reverting from `executing` or `outcome-verifying` back to `drafted` (PLAN content has changed; prior audits are stale).

### `last_executor_outcome`
Written by `execute-plan` workflow on completion (per decision 24). Read-only for the orchestrator:
```yaml
last_executor_outcome:
  outcome: success            # success | revision_needed | exception
  outcome_subtype: done       # done | partially-complete | blocked | needs-revision
  executed: 2026-05-01
  diagnostics_summary: ""
```

### `verification_state`
```yaml
verification_state:
  state_pass: 0
  state_fail: 0
  acceptance_pass: 0
  acceptance_fail: 0
  human_pending: []
  human_verdict: pending      # pending | all_pass | rejected
  human_diagnostics: ""       # populated when human_verdict: rejected
```

### `status`
Existing field, mirrored to LOG. Orchestrator mutates on phase boundaries:
- Entering `executing` → `status: in-progress`.
- Reverting from `executing`/`outcome-verifying` → `drafted` → `status: needs-revision`.
- After successful retire → `status: done` (already set by `execute-plan`; just confirm).

---

## Commit-message templates

Every commit is `git add -A && git commit -m "<template>" && git push`. Never `--no-verify`, `--force`, or `--force-with-lease`. If push fails → orchestrator emits `exception` (commit is already local).

| Trigger                                          | Template                                                                                            |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `plan-writer` returned success during drafting   | `plan-pipeline: drafting checkpoint <plan-filename>`                                                |
| `plan-writer` returned success at draft-close    | `plan-pipeline: drafted <plan-filename>`                                                            |
| `write-bus-input` lands during drafting          | `plan-pipeline: drafting input <input-filename>`                                                    |
| Human-revised PLAN, before re-audit              | `plan-pipeline: human-revised <plan-filename> (audit_state.<stage>_iterations=<N>)`                 |
| Audit `audit_state` write (success or rev_needed) | `plan-pipeline: audit_state update — <stage>:<outcome>`                                            |
| Both audits passed → `checked`                   | `plan-pipeline: checked <plan-filename>`                                                            |
| Dispatch executor (`checked → executing`)        | `plan-pipeline: executing <plan-filename>`                                                          |
| Executor success → `outcome-verifying`           | `plan-pipeline: outcome-verifying <plan-filename>`                                                  |
| Outcome-verification ran (state recorded)        | `plan-pipeline: outcome-verification ran for <plan-filename>`                                       |
| Outcome-verification shell failed (revert)       | `plan-pipeline: outcome-verification failed — reverting to drafted for <plan-filename>`             |
| Human verification passed                        | `plan-pipeline: human verification passed for <plan-filename>`                                      |
| Flip → `complete`                                | `plan-pipeline: complete <plan-filename>`                                                           |
| Retire success                                   | `plan-pipeline: retired <plan-filename>`                                                            |
| Executor revision_needed (revert)                | `plan-pipeline: executor revision_needed <plan-filename>`                                           |
| Any kanban halt                                  | `WIP: pipeline halted at <phase> for <plan-filename> — see diagnostics`                             |

**Bootstrap exception:** the parent PLAN `202605011400_PLAN_build-plan-pipeline-orchestrator.md` is git-managed by the Human, not the orchestrator. Detect by filename comparison; skip all of the above for that one file.

---

## `<pipeline-result>` parsing

Every subagent's return ends with:

```
<pipeline-result>
```json
{
  "outcome": "success" | "revision_needed" | "exception",
  "payload": { /* skill-specific */ },
  "diagnostics": { /* populated when outcome != success */ }
}
```
</pipeline-result>
```

**Parse procedure:**
1. Text-scan the agent's return string. Find the LAST `<pipeline-result>` opening tag.
2. From that position, find the matching `</pipeline-result>` closing tag.
3. Within that span, extract the content of the JSON code fence (```json ... ```).
4. JSON-parse. Validate `outcome` is in the enum.
5. On any failure (no opening tag, no closing tag, no JSON fence, parse error, outcome not in enum) → orchestrator emits its own `outcome: exception` with `diagnostics: { reason: "subagent return malformed", agent: "<name>", excerpt: "<last 500 chars of return>" }`.

The agent's prose body before the block is for Human reading; the orchestrator does NOT mine it for state.

---

## Decision-15 triage helper

Applied at every Human-facing surface. For each item the orchestrator is about to surface:

1. **Already locked** — was this Human-proposed earlier in this conversation, or affirmed in a prior PLAN? List for transparency only ("noted: <item>"); do NOT ask.
2. **Mechanically forced** — is there a single mechanically-correct answer (downstream of a locked decision, an infrastructure constraint, or established working style)? List for transparency only; do NOT ask.
3. **Real judgement call** — does the Human plausibly have alternatives they'd prefer? Surface as a question.

Format the surface as:

```
**Already locked** (no action needed):
- <item>

**Mechanically forced** (no action needed):
- <item>

**Real judgement calls** (please respond):
- <question>
```

If all items are non-judgement-call, surface only the transparency lists with a one-line "no questions for you" close.

---

## Concrete `Agent({...})` invocation snippets

For reference (the procedural file uses these implicitly).

**Plan-writer (foreground, draft checkpoint):**
```
Agent({
  subagent_type: "plan-writer",
  description: "Write PLAN drafting checkpoint",
  prompt: "<plan_content>\n\nTarget filename: <target_filename>\nMode: <create|update>\n\nReturn structured <pipeline-result> per your SKILL.md."
})
```

**Sufficiency / plan-safety auditor (foreground):**
```
Agent({
  subagent_type: "sufficiency-auditor",   // or "plan-safety-auditor"
  description: "Audit PLAN <filename>",
  prompt: "Audit the PLAN at: <plan_path>\n\nApply your skill's procedure end-to-end. Return structured <pipeline-result> per your SKILL.md."
})
```

**Plan-executor (background, tier-selected):**
```
Agent({
  subagent_type: "<plan-executor | plan-executor-sonnet | plan-executor-opus>",
  description: "Execute PLAN <filename>",
  prompt: "Execute the PLAN at: <plan_path>\n\nRun execute-plan skill end-to-end. Write last_executor_outcome to PLAN frontmatter on completion. Do NOT commit or push (orchestrator owns git). Return structured <pipeline-result>.",
  run_in_background: true
})
```

**Plan-retirer (foreground):**
```
Agent({
  subagent_type: "plan-retirer",
  description: "Retire PLAN <filename>",
  prompt: "Retire the PLAN at: <plan_path>\n\nMove to Retired/. Do NOT commit or push (orchestrator owns git). Return structured <pipeline-result>."
})
```

---

## Idempotent-no-op summary

Re-entry into the orchestrator on the same disk-state with no new outcome should return immediately. Quick checks (in order):

1. `pipeline_phase: complete` AND PLAN already moved to `Retired/` → "already retired".
2. `pipeline_phase: executing` AND no `last_executor_outcome` since dispatch → "executor still running".
3. `pipeline_phase: outcome-verifying` AND `verification_state.human_verdict: pending` AND no fresh `human_reply` → "awaiting Human verification reply".
4. `pipeline_phase: drafted` AND `audit_state.last_outcome: revision_needed` AND PLAN mtime ≤ `audit_state` commit time → "awaiting Human revision".
5. Children gate: `triggers_plans` non-empty with any non-terminal child → "paused for children: <list>".

If none of those apply, proceed into Step 4 of the dispatch procedure.
