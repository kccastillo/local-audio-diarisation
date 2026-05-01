---
title: "Create audit-haiku-safe skill (mechanical plan-safety review, Sonnet-pinned)"
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
Create the `audit-haiku-safe` skill at `.claude/skills/audit-haiku-safe/SKILL.md` — a one-shot **mechanical** review skill (Sonnet-pinned) that takes a PLAN file path and verifies each step against the shared plan-safe definition. Returns a structured review with a machine-readable `Blockers: N` summary line. This is the second of two audits in the `drafted → checked` gate; runs only after `audit-sufficiency` returns `Blockers: 0`.

## Context

Spawned from parent PLAN `202605011400`. Per parent's revised decision 5: sufficiency audit (Opus, conceptual) and haiku-safe audit (Sonnet, mechanical) are split into two skills with different model tiers. This skill is the mechanical leaf.

**What it checks** (single lens — mechanical contract):
- Each step is concrete: specific file paths, exact command syntax, no "likely"/"probably".
- Each step is unambiguous: no judgement calls; executor runs steps, doesn't redesign.
- Each step is atomic: one operation; clear success/failure condition.
- Each step is safe: no destructive operations without explicit Human approval; no `--no-verify`/`--force`.
- Each step is testable: verification criteria are independent and checkable.
- All this is the canonical plan-safe definition at `.claude/skills/_shared/plan-safe.md` — the skill **defers** to that file rather than duplicating.

**What it does NOT check** (sufficiency lens — that's `audit-sufficiency`'s job):
- Whether assumptions are verified.
- Whether edge cases are handled.
- Whether the validation path is sensible.
- Whether the design is over-engineered.
- Whether dogfood targets exercise real friction.

**Output format** (parent decisions 14, 15, 16):
- Brief verification preamble (what was checked against — files, not just the PLAN under review).
- One-line overall verdict.
- Single section (no Sufficiency section — that's the other skill).
- `**Blockers**` and `**Not blockers**` subgroups.
- Net verdict.
- Machine-readable summary: `Blockers: N` — orchestrator reads this and applies the gate mechanically.
- If any item requires Human input (rare — this skill is mostly mechanical), apply decision 15 triage before surfacing.

**Definition of blocker** (parent decision 14): any finding that would cause Haiku to halt, error, or be forced into a judgement call mid-execution.

## Steps

1. Compose SKILL.md frontmatter: `name: audit-haiku-safe`, `description` covering trigger phrases ("haiku-safe check this", "is this plan haiku-safe", "verify plan-safety"); note Sonnet pinning is a guideline.

2. Compose body documenting:
   - **Single lens:** mechanical plan-safety per the shared definition.
   - Per-step verification procedure: read each step, evaluate against the five plan-safe criteria, classify as Blocker or Not-blocker.
   - Cross-step checks: ordering coherent, line numbers consistent (line numbers in step N still valid after step N-1's edits).
   - Output format and machine-readable summary line.
   - Decision 15 triage if any Human-input items.
   - **Sequencing reminder:** this skill runs *after* `audit-sufficiency` passes. If invoked on a PLAN that hasn't passed sufficiency, halt and return `Blockers: 1 (precondition: sufficiency audit not run or not passing)`.

3. Externalise the per-step procedure into `workflows/audit-haiku-safe-steps.md` if SKILL.md exceeds ~80 lines.

4. Write the file(s).

5. **[Human]** Smoke test: invoke `Skill("audit-haiku-safe", "<path-to-this-PLAN>")` against this PLAN itself (which should pass since this PLAN is itself authored Haiku-safe). Verify output format and that the `Blockers: N` line is parseable.

## Verification

- [ ] `.claude/skills/audit-haiku-safe/SKILL.md` exists with valid frontmatter
- [ ] Body references `.claude/skills/_shared/plan-safe.md` (does not inline the definition)
- [ ] Single lens (mechanical only) documented; clarifies it does NOT do sufficiency
- [ ] Output ends with machine-readable `Blockers: N` summary
- [ ] Decision 14 (gate semantics) and decision 15 (triage) applied
- [ ] Decision 16 compliance: SKILL.md contains `<preconditions>` and `<output_schema>` blocks
- [ ] Decision 19: SKILL.md contains `<exception_conditions>` block (e.g. PLAN unreadable, sufficiency-auditor not yet passed, shared plan-safe.md missing); subagent self-terminates with `outcome: exception` on any condition; no mid-execution milestones (not platform-supported)
- [ ] Decision 20: explicit `<inputs>` block AND `<output_schema>` block in SKILL.md body; output schema includes `outcome: enum[success, revision_needed, exception]` + `payload: {blockers_count, review_text}` + `diagnostics`
- [ ] Decision 25: skill checks every PLAN Verification item is shell-runnable (annotated `verify:`, `acceptance:`, or `verify: human`) AND at least one `acceptance:` item exists per PLAN. Missing format or absent acceptance item → flag as blocker.
- [ ] Sequencing precondition documented: requires audit-sufficiency to have run and passed first
- [ ] Smoke test against self produces a substantive, mechanical review

## Executor Notes

**Executed:** 2026-05-01 (created directly by Opus during bootstrap, on Human request — out-of-band of the planned execute-plan invocation)
**Outcome:** done
**What was done:**
- Created `.claude/skills/audit-haiku-safe/SKILL.md` with frontmatter, essential_principles, preconditions, inputs/output_schema (decision 20), exception_conditions (decision 19), output_format (decisions 14, 15, CLAUDE.md "Reviews" rule), constraints, success_criteria
- Created `.claude/skills/audit-haiku-safe/workflows/audit-haiku-safe-steps.md` with 7-step procedure: validate preconditions → per-step plan-safety review → cross-step coherence → verification format check → compose review output → decision-triage → emit pipeline-result block
- Skill body explicitly references shared `_shared/plan-safe.md` (single source of truth, decision 2)
- Skill checks every Verification item has shell-runnable annotation AND at least one `acceptance:` per PLAN (decision 25)
- Skill output ends with `<pipeline-result>` JSON block (decision 23)
- Sequencing precondition documented: returns `outcome: exception` if invoked before audit-sufficiency passed

**Blockers (if any):** None.

**Files modified:**
- Created: `.claude/skills/audit-haiku-safe/SKILL.md`
- Created: `.claude/skills/audit-haiku-safe/workflows/audit-haiku-safe-steps.md`

**Smoke test deferred:** Per child verification "smoke test against self produces a substantive, mechanical review" — this is deferred. Will be exercised naturally during child 1430 (plan-pipeline) creation when the orchestrator's loop dispatches plan-safety-auditor.

**last_executor_outcome:**
  outcome: success
  outcome_subtype: done
  executed: 2026-05-01
  diagnostics_summary: ""
