---
title: "Close audit-haiku-safe blind spot: PLANs can pass audits while asking executor to perform excluded operations"
type: bus-plan
status: ready
assigned_to: opus
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
parent_plan_of_plans: ""
pipeline_phase: drafted
audit_state:
  sufficiency_iterations: 2
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: success
---

## Objective

Close a blind spot in `audit-haiku-safe`: a PLAN whose Step asks `plan-executor` to perform an architecturally-excluded operation (currently: invoke `retire`, use `Bash`) currently passes both audits. The runtime then either silently no-ops (case observed in F/14 closeout PLAN 202605012200, 2026-05-01: `Skill("retire")` from inside executor returned no error but moved zero files) or fails opaquely. Two complementary fixes prevent the class:
1. **Audit-time prevention** — extend `_shared/plan-safe.md` with executor capability-boundary rules; audit-haiku-safe already checks against plan-safe.md so the new rules flow through automatically.
2. **Runtime safety net** — extend `plan-executor.md` (and the sonnet/opus tier variants) with explicit exception-on-excluded-operation handling, so a PLAN that slipped through audits still fails loud rather than silent.

Also: codify the closeout-style-retire carve-out in `plan-pipeline/SKILL.md` so future PLANs that retire OTHER PLANs (not their own self-retire via 4F) have a documented pattern.

## Context

**The triggering observation.** During F/14 closeout (PLAN 202605012200), Step 4 instructed the executor to invoke `Skill("retire")` on eight PLANs. Both audits (sufficiency 1 iter, plan-safety 3 iters) passed. The executor reported `outcome: success`. Outcome-verification then caught that 0 of 8 retire moves had landed: all eight files remained in `Bus/`. Cause: subagents do not inherit the parent's skill registry; `Skill()` calls from inside subagents fail without raising. Even with skill preload via `skills:` frontmatter, the retire skill needs Bash for `mv`, which is denied to plan-executor by F1 Option C (decision in PLAN 202605011900). And independent of that, parent PLAN 1400 decision 3 already says retire is orchestrator-owned. The agent file even states "Does not retire (decision 3)" in its prose.

**Two architectural truths converged here.** Decision 3 (retire is orchestrator-owned) AND F1 Option C (executor has no Bash) BOTH independently prevent the executor from retiring. The PLAN's Step 4 violated both. Neither audit caught it — sufficiency-auditor focuses on conceptual validity; audit-haiku-safe focuses on annotation format and atomicity. Neither cross-references the executor's capability boundaries.

**Why this matters beyond F/14.** The same blind spot will admit other invalid Steps: any Step asking executor to use Bash directly, any Step asking executor to invoke other orchestrator-only skills (`write-bus-input` runs in parent during ideate; `plan-pipeline` is the orchestrator itself). As more skills are added, the exclusion list grows; without an audit-time check, future PLANs will keep tripping this.

**Recovery pattern from F/14.** Parent session performed the eight `mv` operations directly after the executor failed; this is the correct pattern for closeout-style PLANs (PLANs whose deliverable retires OTHER PLANs). The orchestrator's normal `complete` phase (4F) retires the current PLAN via `plan-retirer` agent — that doesn't help when a PLAN's work itself is mass-retire of other PLANs. This pattern needs documenting so future closeout PLANs are authored correctly.

## Design Decisions Classification

### Already-locked
- Don't remove subagents (Human directive 2026-05-01: "direction shouldn't change").
- Skill-preload via `skills:` frontmatter remains the workaround for runtime skill access in subagents.
- Decision 3 (retire is orchestrator-owned) stays as-is.
- F1 Option C (executor has no Bash) stays as-is.
- Framing of new exclusion rule: (β) — split by mechanism into 4 sub-clauses (Human chose 2026-05-01 during sufficiency iteration 1).
- Enumerate vs reference exclusion list: this section in plan-safe.md is canonical; agent file documents tool-permission denials only.

### Mechanically-forced
- Audit-haiku-safe checks against plan-safe.md, so encoding rules in plan-safe.md is the lowest-friction prevention path.
- Runtime defence in plan-executor.md must use existing `outcome: exception` mechanism (no new wire format).

### Real-judgement-calls
- Whether the closeout-style-retire carve-out lives in plan-pipeline SKILL.md `<constraints>` (parallel to F4, F9) or in `references/phase-state-machine.md` as a routing-table entry.

## Steps

0. **Pre-fix smoke capture (orchestrator-side, parent session).** Author a tiny throwaway PLAN at `Bus/_tmp_audit-blind-spot-smoke.md` whose Step 1 says "Invoke `Skill('retire')` from inside plan-executor on `tmp/audit-smoke-target.md`". Do NOT advance it through the pipeline yet — instead, the orchestrator (parent session) manually invokes `sufficiency-auditor` and `plan-safety-auditor` foreground via Agent tool. Capture both audit `<pipeline-result>` JSON outputs verbatim into `tmp/audit-smoke-prefix-results.txt`. Confirm both return `outcome: success` (this is the bug we're closing).

1. **Augment `_shared/plan-safe.md`.** Add a new section after the F1 Option C rules with the following content as the target wording:

   > **Executor capability boundaries.** Steps must not ask `plan-executor` (or its tier variants) to perform any of the following. Each has a different mechanism but the rule is uniform: route through orchestrator (parent session) instead.
   >
   > - **(a) Raw `Bash`.** Denied at the tool-permission layer via `disallowedTools` (F1 Option C, PLAN 202605011900). Use Read/Edit/Write/Glob/Grep filesystem tools, or `python -c` for shell-equivalent operations within the executor's `python` allowance, instead. Audit-haiku-safe blocker if a Step's prose or `verify:`/`acceptance:` shell name `bash`/`sh` directly.
   > - **(b) Skills excluded by orchestrator-ownership decisions.** Currently: `retire` (parent PLAN 1400 decision 3 — orchestrator owns retire via 4F or, for closeout-style PLANs, via direct parent-session retire). Audit-haiku-safe blocker if a Step says "invoke `retire`" / "use `Skill('retire')`" / "call retire skill".
   > - **(c) Skills that are parent-session-only by structural convention.** Currently: `ideate` (decision 10 — no agent file exists; cannot be dispatched). Audit-haiku-safe blocker if a Step asks executor to run ideate; route ideation to parent session.
   > - **(d) Skills that orchestrate other skills.** Currently: `plan-pipeline` (the orchestrator itself), `write-bus-input` (runs in parent during ideation per decision 12). These run in parent context by design. Audit-haiku-safe blocker if a Step asks executor to invoke them.
   >
   > Source of truth for the per-skill list: this section. Source of truth for tool-permission exclusions: `.claude/agents/plan-executor.md` `disallowedTools`. When the lists drift, this section is canonical for audit purposes; update it whenever an exclusion changes.

2. **Extend `plan-executor.md` (and tier variants `plan-executor-sonnet.md`, `plan-executor-opus.md`) with runtime exception clause.** Add to each agent's exception conditions: "If a Step requires invocation of an excluded skill (`retire`, `write-bus-input`, `plan-pipeline`, `ideate`) or raw Bash, terminate `outcome: exception` with `diagnostics: { reason: 'Step requires excluded operation X — route through orchestrator (parent session)', step_number: N }`. Do not silent no-op."

3. **Codify closeout-style-retire pattern in `plan-pipeline/SKILL.md`.** Add to `<constraints>` (parallel to F4 and F9): "Closeout-style PLANs (PLANs whose Steps retire OTHER PLANs, distinct from the orchestrator's normal 4F self-retire) must express Step 4 as 'orchestrator retires Bus/X.md, Bus/Y.md, ...' — the orchestrator (parent session) performs the moves directly because plan-executor is structurally unable to (decision 3 + F1 Option C). The orchestrator may invoke `Skill("retire")` from parent context for each file, or use direct `mv` if appropriate."

4. **Author or update audit-haiku-safe regression test material.** If audit-haiku-safe maintains test fixtures, add a fixture: a PLAN whose Step 4 says "Invoke retire skill from inside executor"; assert that audit-haiku-safe returns `revision_needed` with the new capability-boundary blocker. If no test fixtures exist yet, document the regression-test scenario in audit-haiku-safe's SKILL.md as a worked example.

5. **Post-fix smoke verification (orchestrator-side, parent session).** Orchestrator re-invokes `plan-safety-auditor` foreground on the same throwaway PLAN at `Bus/_tmp_audit-blind-spot-smoke.md`. Capture output into `tmp/audit-smoke-postfix-results.txt`. Confirm `outcome: revision_needed` with at least one blocker citing "Executor capability boundaries" or "excluded operation". Then test the runtime defence: dispatch `plan-executor` foreground (force-bypass-of-audit only for this smoke test, since the audit now correctly catches it) and confirm executor returns `outcome: exception` with diagnostic referencing "excluded operation". Finally, retire the throwaway PLAN: orchestrator (parent) `mv Bus/_tmp_audit-blind-spot-smoke.md Retired/_tmp_audit-blind-spot-smoke.md`.

   Note: the throwaway PLAN's retire MUST happen via orchestrator (parent) since the smoke-test PLAN's content asks for the very excluded operation being tested — the executor structurally cannot retire it.

## Verification

- [ ] verify: `grep -F "Executor capability boundaries" .claude/skills/_shared/plan-safe.md`
- [ ] verify: `grep -F "excluded operation" .claude/agents/plan-executor.md`
- [ ] verify: `grep -F "excluded operation" .claude/agents/plan-executor-sonnet.md`
- [ ] verify: `grep -F "excluded operation" .claude/agents/plan-executor-opus.md`
- [ ] verify: `grep -F "Closeout-style PLANs" .claude/skills/plan-pipeline/SKILL.md`
- [ ] verify: `test -f tmp/audit-smoke-prefix-results.txt`
- [ ] verify: `test -f tmp/audit-smoke-postfix-results.txt`
- [ ] verify: `! test -e Bus/_tmp_audit-blind-spot-smoke.md && test -e Retired/_tmp_audit-blind-spot-smoke.md`
- [ ] acceptance: `grep -F "Executor capability boundaries" .claude/skills/_shared/plan-safe.md && grep -F "excluded operation" .claude/agents/plan-executor.md && grep -F "excluded operation" .claude/agents/plan-executor-sonnet.md && grep -F "excluded operation" .claude/agents/plan-executor-opus.md && grep -F "Closeout-style PLANs" .claude/skills/plan-pipeline/SKILL.md`
- [ ] verify: human — the new plan-safe.md rule wording is unambiguous and matches the executor agent file's actual exclusion list (no drift); and the throwaway PLAN's pre-fix results confirm both audits passed (bug reproduced) while post-fix results confirm audit-haiku-safe now blocks and executor raises exception (fix validated)

## Executor Notes

(Populated by plan-executor during executing phase.)
