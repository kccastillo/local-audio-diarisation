---
title: "ADVICE: Sufficiency audit exemplar — Opus-pass observations on plan-pipeline orchestrator design"
type: bus-advice
status: standing
created: 2026-05-01
created_by: opus
created_month: 202605
log_month: 202605
feeds_plan: 202605011425_PLAN_create-audit-sufficiency-skill.md
linked_pkg: 202605011400_PLAN_build-plan-pipeline-orchestrator.md
---

## Purpose

This ADVICE file is the worked exemplar for the `audit-sufficiency` skill (to be created via PLAN 202605011425). It captures the kind of output that an Opus-pass sufficiency audit produces — qualitatively different from the mechanical, contract-checking work of a Sonnet-pinned `audit-haiku-safe` skill. The skill's authors should use this as the reference for the *kind* of reasoning a sufficiency audit performs.

## Context

Generated during the dogfood of `plan-pipeline` design (PLAN 202605011400). After 16 design decisions had been locked and an automated check-plan v6 returned `Blockers: 0`, the Human asked for "another Opus pass to make sure we capture any opportunities to course-correct before we have written anything." The pass surfaced six observations that the prior reviews (v1 through v6, each a more mechanical check) had missed.

The lesson: mechanical plan-safety review (verifying each step is concrete, testable, atomic) and conceptual sufficiency review (interrogating assumptions, surfacing missing edge cases, asking whether the design is over-engineered) are different cognitive operations. They benefit from different model tiers and should be different skills.

## The exemplar audit (verbatim — what an `audit-sufficiency` pass produces)

### 1. Foundational: framework support unverified

We've designed around three load-bearing assumptions that have never been confirmed in the actual Claude Code version in use:

- **Pinned-model agent definitions in `.claude/agents/`** with `model: haiku|sonnet|opus` frontmatter. The Agent tool's `subagent_type` parameter — does it accept custom names from `.claude/agents/`?
- **`Skill("...")` calls from inside a subagent.** Does a dispatched subagent inherit the parent's skill registry?
- **`run_in_background: true` semantics.** Does notification handling work in a *skill* context, not just chat?

If any is wrong, parts of the architecture need redesign. Verify before committing to building.

### 2. Bootstrap delays validation until everything is built

We modify execute-plan, retire, write-bus-plan, bus-conventions; create 4 agents and 3 skills; only THEN exercise the pipeline at child 4. If a foundational assumption is wrong, we discover it after ~10 file mutations. Consider a smaller MVP path: ship `audit-haiku-safe` standalone first, validate pinned-model subagents with one tiny agent, defer the full orchestrator.

### 3. Dogfood target is contrived

`note-jot` is synthetic — building it doesn't prove the pipeline survives real planning friction. The next real PLAN already in the LOG (e.g. `rationalise-claude-md`) would be a more honest test.

### 4. Orchestrator edge cases unspecified

- Missing agent file → fall back to parent-session invocation, not crash.
- Malformed check-plan output (no `Blockers: N` line) → halt with "review unparseable", not silently advance.
- Child PLAN fails its own check-plan → blocks parent advancement; explicit handling needed.

### 5. Memory note staleness

`feedback_retire_push.md` literally says retire commits/pushes; after decision 13, retire no longer does. User-observed invariant holds (commit-after-retire still happens, one layer up) but the literal text misleads. Update.

### 6. Meta: 16 decisions before any file is written

A future executor (or future Claude) reading the PLAN cold has to absorb 16 decisions before understanding what we're building. Consider a "decisions digest" header — 3–4 bullets — with the full list as appendix. Faster onboarding.

## What this exemplar teaches the `audit-sufficiency` skill

Pattern of the review:
1. **Question load-bearing assumptions** — find the bits the design depends on but hasn't verified. Surface them before they become silent failures.
2. **Audit the validation path** — when does the design first get tested? If late, what's at risk if foundational assumptions are wrong?
3. **Interrogate test fidelity** — does the dogfood/test target actually exercise real friction, or is it a contrived smoke test?
4. **Surface unhandled edge cases at the orchestration layer** — what if a referenced artefact is missing? What if a sub-output is malformed? What if a downstream component fails?
5. **Find stale references** — anything elsewhere in the codebase / memory / docs that this design contradicts? Update or it'll mislead.
6. **Step back to the meta level** — is the design over-engineered? Could it be smaller? What's the minimum viable version?

Lenses: assumption, validation, fidelity, edges, freshness, meta.

The Sonnet/Haiku `audit-plan-safety` skill operates on a different lens: per-step concreteness, atomicity, judgement-call detection, line-number accuracy, exact-text completeness. It runs *after* sufficiency passes (no point Haiku-safety-checking an insufficient plan).
