---
title: "Create audit-sufficiency skill (conceptual review, Opus-pinned)"
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
linked_inputs:
  - 202605011500_ADVICE_sufficiency-audit-exemplar.md
blocked_by: ""
rollover_count: 0
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: 202605011400_PLAN_build-plan-pipeline-orchestrator.md
pipeline_phase: drafted
---

## Objective
Create the `audit-sufficiency` skill at `.claude/skills/audit-sufficiency/SKILL.md` — a one-shot **conceptual** review skill (Opus-pinned) that interrogates a PLAN for assumptions, edge cases, validation-path risk, dogfood fidelity, stale references, and meta-design concerns. Returns a structured review with a machine-readable `Blockers: N` summary line. This is the first of two audits in the `drafted → checked` gate; runs *before* `audit-haiku-safe`.

## Context

Spawned from parent PLAN `202605011400`. The need for this skill was surfaced during the dogfood when six rounds of mechanical check returned `Blockers: 0` but a Human-led Opus pass identified six architectural issues none of the prior reviews touched. See ADVICE 202605011500 for the worked exemplar of what an Opus-pass output looks like — that is the design reference for this skill.

**What it checks** (seven lenses — borrowed from the exemplar; the seventh added per parent decision 25):
1. **Assumptions** — what is the design depending on but hasn't verified? Surface load-bearing assumptions before they become silent failures.
2. **Validation path** — when does the design first get tested? If late, what's at risk if foundational assumptions are wrong?
3. **Test fidelity** — does the dogfood / smoke test exercise real friction, or is it a contrived smoke test?
4. **Edge cases at orchestration layer** — missing referenced artefact? malformed sub-output? downstream component fails? Are these handled?
5. **Freshness** — anything elsewhere in codebase / memory / docs that this design contradicts? Anything that will become misleading after execution?
6. **Meta** — is the design over-engineered? Could it be smaller? What's the minimum viable version?
7. **Spec-acceptance fidelity** (decision 25) — does the PLAN's `acceptance:` sample(s) actually exercise what the Objective claims to deliver, or do they just incidentally box-tick? Are they too narrow (one branch covered, others not) or too broad (untestable as written)? An acceptance sample that doesn't relate to the Objective is a blocker.

**What it does NOT check** (mechanical lens — that's `audit-haiku-safe`'s job):
- Per-step concreteness, atomicity, exact-text completeness, line-number accuracy.

**Output format** (parent decisions 14, 15, 16):
- Brief verification preamble (what was checked against — files, not just the PLAN under review).
- One-line overall verdict.
- Single section (Sufficiency).
- `**Blockers**` and `**Not blockers**` subgroups.
- Net verdict.
- Machine-readable summary: `Blockers: N`.
- Decision 15 triage applied to any Human-input items (Already-locked / Mechanically-forced / Real-judgement-call).

## Steps

1. **[Opus]** Read the exemplar ADVICE file (`202605011500_ADVICE_sufficiency-audit-exemplar.md`) to ground the design in a worked example.

2. Compose SKILL.md frontmatter: `name: audit-sufficiency`, `description` covering trigger phrases ("opus pass", "sufficiency audit", "course-correct check", "check sufficiency"); note Opus pinning is a recommendation (decision 8).

3. Compose body documenting:
   - **Six-lens framework:** assumptions / validation / fidelity / edges / freshness / meta. Each lens with a concrete prompt (e.g. for assumptions: "what does the design depend on that hasn't been confirmed in this Claude Code version / this codebase / this team's tooling?").
   - Worked example excerpt referencing the ADVICE file.
   - Output format with `Blockers: N` summary.
   - Decision 15 triage requirement.
   - **Sequencing:** runs first in the gate; `audit-haiku-safe` only runs if this skill returns `Blockers: 0`.

4. Externalise the lens-by-lens checklist into `references/sufficiency-lenses.md` if SKILL.md exceeds ~100 lines (this skill is conceptually richer than haiku-safe; expect more body content).

5. Write the file(s).

6. **[Human]** Smoke test: invoke `Skill("audit-sufficiency", "<path-to-parent-PLAN-202605011400>")` against the parent PLAN itself. Expected outcome: review surfaces some of the same six observations that the original Opus pass produced (see ADVICE), validating the skill captures that quality of pass. If the review is shallow or mechanical, iterate the SKILL.md.

## Verification

- [ ] `.claude/skills/audit-sufficiency/SKILL.md` exists with valid frontmatter (model: opus or note Opus-recommended)
- [ ] Body documents the six-lens framework with concrete prompts per lens
- [ ] References ADVICE 202605011500 as worked exemplar
- [ ] Output ends with machine-readable `Blockers: N` summary
- [ ] Decision 14 (gate semantics) and decision 15 (triage) applied
- [ ] Decision 16 compliance: SKILL.md contains `<preconditions>` and `<output_schema>` blocks
- [ ] Decision 19: SKILL.md contains `<exception_conditions>` block (PLAN unreadable, linked_inputs missing, PLAN structurally malformed); subagent self-terminates with `outcome: exception` on any condition; no mid-execution milestones (not platform-supported)
- [ ] Decision 20: explicit `<inputs>` block AND `<output_schema>` block in SKILL.md body; output schema includes `outcome: enum[success, revision_needed, exception]` + `payload: {blockers_count, review_text, triaged_human_items}` + `diagnostics`
- [ ] Decision 25 (seventh lens): skill checks that PLAN's `acceptance:` sample(s) actually map to the Objective; an unrelated/incidental acceptance is flagged as a blocker.
- [ ] Sequencing documented: this skill runs first; `audit-haiku-safe` runs after
- [ ] Smoke test against parent PLAN produces substantive output reflective of the exemplar's quality

## Executor Notes

**Executed:** 2026-05-01 (authored directly in Opus during bootstrap, per option-2 path)
**Outcome:** done
**What was done:**
- Created `.claude/skills/audit-sufficiency/SKILL.md` with frontmatter (trigger phrases include "opus pass"), reference to shared plan-safe.md, reference to ADVICE 1500 exemplar, essential_principles (conceptual lens, not mechanical), preconditions, inputs, output_schema (outcome enum + payload + diagnostics + pipeline-result wire format), exception_conditions, output_format following CLAUDE.md "Reviews" rule, constraints, success_criteria
- Created `.claude/skills/audit-sufficiency/workflows/audit-sufficiency-steps.md` with 6-step procedure: validate preconditions → read referenced inputs → apply seven lenses (assumptions, validation path, test fidelity, edges, freshness, meta, spec-acceptance fidelity) → triage Human-input items per decision 15 → compose review per CLAUDE.md format → emit pipeline-result block
- Each lens has a concrete prompt + Blocker/Not-blocker examples
- Workflow explicitly references the ADVICE exemplar as the calibration target ("if your review is shallower than the exemplar, you missed lenses; if heavier, you may have ventured into mechanical territory")
- Stays out of mechanical territory (defers per-step concreteness checks to audit-haiku-safe)

**Blockers (if any):** None.

**Files modified:**
- Created: `.claude/skills/audit-sufficiency/SKILL.md`
- Created: `.claude/skills/audit-sufficiency/workflows/audit-sufficiency-steps.md`

**Smoke test deferred:** Real exercise will happen the first time `plan-pipeline` orchestrator (child 1430) dispatches `sufficiency-auditor` against a real PLAN. The exemplar (ADVICE 1500) was generated against the parent PLAN and should reproduce when the skill is invoked there.

**last_executor_outcome:**
  outcome: success
  outcome_subtype: done
  executed: 2026-05-01
  diagnostics_summary: ""
