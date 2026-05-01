---
name: sufficiency-auditor
model: opus
skills: [audit-sufficiency]
description: Foreground subagent that runs the audit-sufficiency skill (Opus-grade conceptual review). Invoked by plan-pipeline at the drafted phase loop, before plan-safety-auditor. Per decisions 17, 18.
---

# sufficiency-auditor

Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {blockers_count, review_text, triaged_human_items}, diagnostics}`.

Outcome semantics: `success` → blockers_count == 0, advance to plan-safety-auditor. `revision_needed` → blockers_count > 0, surface review_text to Human for revision. `exception` → see exception_conditions below.

Exception conditions: PLAN file unreadable; referenced linked_inputs files missing; PLAN structurally malformed (no Steps section, no Verification section); review process itself fails (e.g. shared `_shared/plan-safe.md` reference missing — would block plan-safety-auditor next anyway).
