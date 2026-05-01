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

Six phases (A-F), each gated on the prior. Phase A is critical-path: F1's fix-shape can't be designed without probe data. Phases C-E are independent of each other once Phase B lands.

### Phase A — Probe and design F1 (subagent permission context)

1. **Probe subagent permission scope.** Spawn an Agent task (general-purpose) that dispatches a haiku subagent and has it attempt several Bash commands: one allowlist-matched in parent `.claude/settings.json` (e.g. `mkdir`), one not-matched (e.g. `curl`), one with explicit working directory, one without. Capture: which commands worked, exact denial messages, any documented config the subagent's permission scope reads. Land findings as a RESEARCH file via `write-bus-input` so 1900's `linked_inputs` references it.

2. **Survey F1 fix options based on probe.** At least 2 options. Common candidates: (a) `permissions:` field in agent frontmatter (per-agent scoping), (b) project-level inherit-into-subagent config, (c) explicit pass-through on the Agent invocation, (d) restructure plan-executor to depend only on already-allowed commands (`python -c` stdlib for filesystem ops). Document tradeoffs; state lean.

3. **Converge on F1 design.** Decision-15 triage; capture in this PLAN's Context.

### Phase B — Implement F1

4. Apply the chosen F1 fix per Phase A. Likely affects: `.claude/agents/plan-executor.md` (and tier variants), possibly `.claude/settings.json`, possibly `execute-plan` skill workflow.

5. **Smoke-test F1.** Author a trivial dummy PLAN (one mechanical Step: create a single file). Dispatch plan-executor on it. Confirm subagent runs the Bash command, file lands, executor returns `outcome: success`. If not, return to Phase A.

### Phase C — Wire format consistency (F2)

6. Update each subagent skill's SKILL.md to include a single canonical wire-format example (literal angle brackets, JSON code fence). Files: `.claude/skills/audit-sufficiency/SKILL.md`, `.claude/skills/audit-haiku-safe/SKILL.md`, `.claude/skills/write-bus-plan/SKILL.md`, `.claude/skills/execute-plan/SKILL.md`, `.claude/skills/retire/SKILL.md`. Add a non-negotiable line in each `<essential_principles>`: "Wire format: end response with literal `<pipeline-result>` containing JSON code fence per parent decision 23. No XML payload, no HTML escaping."

### Phase D — Medium-severity findings

7. **F3 — decision 25 skills-as-deliverable carve-out.** Amend parent PLAN 202605011400 decision 25 to add: "Exception: PLANs whose deliverable is a Claude skill (or other artefact not invokable from a shell) MAY satisfy the at-least-one-`acceptance:` requirement via an artefact-property check (frontmatter parses, workflow content present, gate-clauses present) PLUS a `verify: human` for the actual invocation behaviour." Mirror the carve-out in `.claude/skills/_shared/plan-safe.md`.

8. **F4 — orchestrator no-direct-edit boundary.** Update `.claude/skills/plan-pipeline/SKILL.md` `<constraints>` to define the boundary explicitly: surgical body-content edits (one-line revisions, audit-resolution notes, commit-hash citations) MAY use the Edit tool directly; full body re-authoring (Objective/Steps/Verification rewrite) MUST route through plan-writer. The current "never write PLAN content" prohibition applies only to the second.

9. **F5 — phase-flip + content-write atomicity.** Add `target_phase:` (optional) to plan-writer's input contract in `.claude/agents/plan-writer.md` and `.claude/skills/write-bus-plan/SKILL.md` `<inputs>`. When supplied, plan-writer writes both body content and the new `pipeline_phase` value in one file write. Orchestrator passes it for transitions where content-write and phase-flip naturally coincide (drafting checkpoints, drafting→drafted close, executor-success→outcome-verifying).

### Phase E — Lower-severity housekeeping

10. **F8 — deterministic seed-blocker.** Update dogfood spec 1440 Step 2 to replace "omit the Objective's deliverable claim" with a deterministic injection rule: "after the converge-close plan-writer dispatch, the orchestrator runs an Edit removing the LAST Verification item from the note-jot PLAN; this guarantees a sufficiency-grade blocker fires regardless of the Objective's wording." (The seed becomes orchestrator-mechanical, not author-judgement.)

11. **F9 — phase-boundary stop semantics.** Update `.claude/skills/plan-pipeline/SKILL.md` and `workflows/dispatch.md` to state explicitly: "The orchestrator MAY chain phase transitions within a single invocation (drafted → checked → executing) when running in a continuous parent session, provided each transition produces a milestone commit. Re-entry idempotency is preserved by reading disk on every invocation." (Relaxes the strict one-phase-per-invocation; keeps re-entry safety.)

### Phase F — Re-dogfood and parent closeout

12. **Re-run dogfood.** Either re-execute 1440 against a fresh target, or unblock note-jot 1700 (`status: ready`, `pipeline_phase: checked`, `blocked_by: ""`) and re-dispatch plan-executor. Verify ALL six phases complete cleanly: drafting → drafted → checked → executing → outcome-verifying → complete → retired. Capture re-dogfood findings in 1440's Executor Notes (or a new dogfood successor PLAN if the second run also surfaces enough material).

13. **Parent PLAN 1400 step 8 — CLAUDE.md skill table update.** Add four rows per parent step 8 (ideate, audit-sufficiency, audit-haiku-safe, plan-pipeline). Already specified in parent.

14. **Close parent PLAN 1400.** Run parent's Verification list end-to-end. Mark `status: done` and `pipeline_phase: complete`. Retire parent + 1440 + 1700 + 1900 (this PLAN) via `plan-retirer`.

## Verification

- [ ] F1 probe RESEARCH file landed in `Bus/`
      `verify: ls Bus/*RESEARCH*subagent*permission*.md 2>/dev/null | grep -q .`
- [ ] Phase A converges with chosen F1 fix-shape documented in this PLAN's Context
      `verify: human`
- [ ] F1 fix applied and smoke-test passes
      `acceptance: bash -c 'ls .claude/skills/plan-pipeline-smoke-test* 2>/dev/null && echo "smoke target present" || echo "smoke target missing"' | grep -q "present"`
- [ ] Every subagent SKILL.md contains the canonical wire-format example (F2)
      `verify: grep -l "literal \`<pipeline-result>\`" .claude/skills/audit-sufficiency/SKILL.md .claude/skills/audit-haiku-safe/SKILL.md .claude/skills/execute-plan/SKILL.md .claude/skills/retire/SKILL.md .claude/skills/write-bus-plan/SKILL.md 2>/dev/null | wc -l | awk '{exit !($1>=5)}'`
- [ ] Parent PLAN 1400 decision 25 carries the skills-as-deliverable carve-out (F3)
      `verify: grep -q "skills-as-deliverable\|skills-as-exception\|Claude skill" Bus/202605011400_PLAN_build-plan-pipeline-orchestrator.md`
- [ ] `_shared/plan-safe.md` mirrors the F3 carve-out
      `verify: grep -q "skills-as-deliverable\|skills-as-exception" .claude/skills/_shared/plan-safe.md`
- [ ] plan-pipeline SKILL.md surgical-edit boundary documented (F4)
      `verify: grep -qE "(surgical|one-line)" .claude/skills/plan-pipeline/SKILL.md`
- [ ] plan-writer accepts target_phase argument (F5)
      `verify: grep -q "target_phase" .claude/agents/plan-writer.md .claude/skills/write-bus-plan/SKILL.md`
- [ ] dogfood spec 1440 Step 2 uses deterministic seed (F8)
      `verify: grep -E "removes the LAST Verification item|deterministic injection" Bus/202605011440_PLAN_dogfood-plan-pipeline.md`
- [ ] plan-pipeline phase-boundary chaining rule documented (F9)
      `verify: grep -E "MAY chain phase transitions|continuous parent session" .claude/skills/plan-pipeline/SKILL.md .claude/skills/plan-pipeline/workflows/dispatch.md 2>/dev/null | grep -q .`
- [ ] re-dogfood completes all six phases without a kanban halt
      `acceptance: git log --oneline -n 100 | grep -E "plan-pipeline: retired" | wc -l | awk '{exit !($1>=1)}'`
- [ ] CLAUDE.md skill table updated (parent step 8)
      `verify: grep -cE "^\| \`(ideate|audit-sufficiency|audit-haiku-safe|plan-pipeline)\`" CLAUDE.md | awk '{exit !($1>=4)}'`
- [ ] Parent PLAN 1400 status is done and parent + dogfood + note-jot + this PLAN are retired
      `verify: ls Retired/202605011400_*.md Retired/202605011440_*.md Retired/202605011700_*.md Retired/202605011900_*.md 2>/dev/null | wc -l | awk '{exit !($1>=4)}'`
- [ ] Re-dogfood + closeout feels coherent — no missed findings, no surprise halts
      `verify: human`

## Executor Notes

*Populated after execution. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
