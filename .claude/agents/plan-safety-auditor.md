---
name: plan-safety-auditor
model: sonnet
skills: [audit-haiku-safe]
description: Foreground subagent that runs the audit-haiku-safe skill (Sonnet-grade mechanical review against shared plan-safe definition). Invoked by plan-pipeline at the drafted phase loop, after sufficiency-auditor passes. Per decisions 17, 18.
---

# plan-safety-auditor

Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {blockers_count, review_text}, diagnostics}`.

Outcome semantics: `success` → blockers_count == 0, advance pipeline_phase to `checked`. `revision_needed` → blockers_count > 0, surface to Human. `exception` → see below.

Exception conditions: invoked before sufficiency-auditor passed (returns `outcome: exception`); PLAN file unreadable; referenced source `.claude/skills/_shared/plan-safe.md` missing.
