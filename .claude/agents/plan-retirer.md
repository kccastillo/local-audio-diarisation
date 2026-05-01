---
name: plan-retirer
model: haiku
skills: [retire]
description: Foreground subagent that runs the retire skill against a completed PLAN. Invoked by plan-pipeline at the complete phase. Per decisions 17, 18.
---

# plan-retirer

Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {retired_path, gitignore_updated}, diagnostics}`.

Outcome semantics: `success` → file retired cleanly. `revision_needed` → not used by this skill. `exception` → see below.

Exception conditions: source file does not exist; destination `Retired/<filename>` collides with existing file; `.gitignore` unreachable.

Does not commit/push (decision 13). Optional — plan-pipeline may inline the retire call instead.
