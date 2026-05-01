---
title: "Post-dogfood fixes for plan-pipeline (9 findings from PLAN 1440 run)"
type: bus-plan
status: ready
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
pipeline_phase: drafting
---

## Objective

Resolve 9 findings surfaced during the plan-pipeline dogfood run (PLAN 202605011440 against PLAN 202605011700_PLAN_note-jot) before re-running the pipeline on any real target. The dogfood halted at `executing` due to a hard architectural finding (#1 below); the other 8 findings are smaller but compound. This PLAN bundles them so a single coordinated revision pass closes them, after which note-jot is re-attempted (or replaced with a different small target).

## Context

**Why this PLAN exists.** The dogfood (1440) successfully exercised drafting, both audit stages, and the executor dispatch. It halted at executing because the haiku subagent's Bash permission context did not honour `.claude/settings.json` allowlist additions made by the parent. That is a load-bearing assumption of the entire `plan-executor` design — every Haiku-executable PLAN ships through that subagent — and resolving it is a precondition for the orchestrator delivering real value.

**Status of the bigger work.** Parent PLAN 202605011400 reached `partially-complete` with all skill-creation children done (1410, 1420, 1425, 1430). Dogfood 1440 reached `partially-complete` with strong findings. note-jot 1700 is `needs-revision` (blocked on the same subagent permission issue). After this fix-PLAN lands, 1440 should be re-run (or a successor dogfood authored).

## Findings catalogue

Surfaced during the 2026-05-01 dogfood run. Priority is engineering-effort × severity, not user-impact.

### F1 — Subagent permission context (HIGH severity)

**Symptom:** plan-executor (haiku, dispatched in background) attempted `mkdir` and got `Permission('Bash(mkdir *)') denied` despite the project `.claude/settings.json` containing `Bash(mkdir *)` in `permissions.allow`. The parent session can run mkdir fine; the subagent could not.

**Implication:** the `plan-executor` subagent design assumes that subagents inherit the parent's allowlist resolution from `.claude/settings.json`. They don't (or at least, they don't via the same mechanism). This blocks every Haiku-executable PLAN flowing through the pipeline.

**Unknowns:** is there a different config file that subagents consult? Is there a per-subagent permission scope? Does it require explicit pass-through from the Agent invocation? Probe needed.

### F2 — Wire-format inconsistency across subagent variants (HIGH severity)

**Symptom:** three different `<pipeline-result>` shapes observed:
- `audit-sufficiency`, `plan-safety-auditor` — correct JSON code fence inside literal `<pipeline-result>` tags.
- `plan-writer` (one run) — XML-style payload (`<outcome>success</outcome>`) inside `<pipeline-result>` tags, no JSON fence.
- `plan-executor` — JSON code fence with HTML-escaped angle brackets (`&lt;pipeline-result&gt;`).

**Implication:** decision 23's "fail loud on malformed" parser would have rejected two of three variants. The orchestrator pragmatically accepted them all to keep the dogfood moving, but production runs need consistent format.

**Cause hypothesis:** subagents are picking up format from their preloaded SKILL.md, which states the expected wire format inconsistently across our skill files. Likely a docs-and-discipline fix, not a code fix.

### F3 — Decision 25 skills-as-deliverable carve-out (MED severity)

**Symptom:** any PLAN whose deliverable is a Claude skill cannot have a true behavioural `acceptance:` shell command — Claude skills are invoked by Claude, not by a shell. Best achievable: artefact-property `acceptance:` (frontmatter parses, workflow content present) plus `verify: human` for skill-invocation behaviour.

**Implication:** decision 25's "every PLAN must have at least one acceptance: that exercises the deliverable" is too strict for skill-deliverable PLANs. Either broaden the decision or carve out an exception.

### F4 — Orchestrator's no-direct-edit constraint too strict for surgical edits (MED severity)

**Symptom:** the orchestrator constraint "Never write PLAN content to disk yourself. Dispatch plan-writer for every create-or-update of a PLAN file" is heavy for one-line surgical edits (commit-hash citation update; small audit-resolution note). Dispatching plan-writer for a 1-line change is wasteful tokens-and-time.

**Implication:** the constraint makes sense for *body* content (Steps, Verification, etc.) but is overkill for surgical revisions to existing body content. Need to define the boundary or relax it.

### F5 — Phase-flip + content-write are not atomic (MED severity)

**Symptom:** plan-writer writes body content but does not touch `pipeline_phase` (orchestration-owned). The orchestrator must do a separate Edit + commit for the phase flip. Two commits where one logical milestone exists.

**Implication:** either plan-writer should accept a `target_phase:` argument and write both atomically, or the orchestrator should bundle the content-write commit + phase-flip into a single git commit.

### F6 — Decision 18 was probe-stale within hours (LOW severity, already addressed inline)

**Symptom:** decision 18 was authored 2026-05-01 morning based on probe-2 finding "`run_in_background` parameter is undocumented." Within the same day, the parameter became documented; the audit caught the contradiction.

**Status:** parent PLAN 1400 was amended inline during dogfood. Lesson captured in Learnings: "probe findings are perishable; re-verify before treating them as load-bearing."

### F7 — Audit-loop caught a silent Edit-tool failure (LOW severity, positive)

**Symptom:** orchestrator-author tried to edit `.claude/settings.json` via the Edit tool without first using the Read tool — Edit refused; orchestrator missed the refusal and moved on; iteration 2 of sufficiency-audit caught the on-disk-vs-claimed mismatch.

**Status:** positive signal — the audit loop is real and catches real issues. Lesson: orchestrator should verify-then-edit, not assume Edit succeeded after a Bash-cat-only inspection.

### F8 — Seed-blocker mechanism unreliable (LOW severity)

**Symptom:** dogfood 1440 Step 2 said "seed a sufficiency-grade blocker by omitting the Objective's deliverable claim." Orchestrator-author wrote "Build note-jot, a small utility skill" which the auditor read as a *complete* deliverable claim. Seed didn't fire.

**Implication:** the seed mechanism depends on the orchestrator-author judging "what counts as deliverable-light," which is exactly the judgement call the audit is supposed to catch. The audit-loop got exercised anyway via organic blockers — but a more deterministic seed (e.g. "strip the most recent Verification item") would test the loop reliably.

### F9 — Subagent stop semantics across phase boundaries unclear (LOW severity)

**Symptom:** the strict "one phase per invocation" rule (drafted → checked → executing should be three invocations) was bypassed in this session for efficiency. No consequences observed but the design intent is to maintain re-entry idempotency across crashes.

**Status:** worth tightening or relaxing explicitly, depending on whether crash-recovery is a real concern in single-session continuous use.

## Steps

*[To be populated at Converge-close. The findings catalogue above feeds Survey; converge will pick fix-shape per finding. F1 is the keystone — without it, F2-F9 are polish on a non-running engine.]*

## Verification

*[To be populated at Converge-close per parent decision 25.]*

## Executor Notes

*Populated after execution. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
