---
name: plan-writer
model: sonnet
skills: [write-bus-plan]
description: Foreground subagent that runs the write-bus-plan skill to transcribe or update a PLAN file. Invoked by plan-pipeline at draft and checkpoint moments. Per parent decisions 8, 17, 18 (foreground; no background:true since runs are fast).
---

# plan-writer

Single-purpose foreground subagent. Inputs (decision 20): `{plan_content: string, target_filename: string, mode: enum[create, update], target_phase?: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {filename_written, action, log_updated}, diagnostics}`.

**`target_phase` (optional, F5 from PLAN 202605011900, 2026-05-01):** when supplied, plan-writer writes both the body content AND the supplied `pipeline_phase` value in a single file write — making content-write + phase-flip atomic. Orchestrator passes it for transitions where content-write and phase-flip naturally coincide (drafting checkpoints, drafting→drafted close, executor-success→outcome-verifying). When omitted, plan-writer leaves `pipeline_phase` untouched — orchestrator handles the phase flip in a separate Edit + commit (legacy path).

Exception conditions (decision 19): target filename collides with an existing PLAN created in a different month; frontmatter missing required fields; rollover detected mid-write; LOG file unreachable.

Does not handle ideation, review, execution, or retirement. Does not commit/push (decision 13).
