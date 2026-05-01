---
title: "Build skill-usage-audit skill: read usage log, produce findings PLAN"
type: bus-plan
status: blocked
assigned_to: ""
priority: low
created: 2026-04-30
created_by: opus
created_month: 202604
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: "deferred until either (a) project has 15+ skills, OR (b) Ken hits a felt misfire (Claude invoking the wrong skill at a noticeable rate). When unblocked, Sonnet must revise Steps 2-6 below using the actual log data accumulated by then."
rollover_count: 1
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Create a `skill-usage-audit` skill at `.claude/skills/skill-usage-audit/` that reads `.claude/_skill_usage.jsonl` (populated by PLAN 202604300210), computes per-skill stats, and produces a Bus PLAN with findings — e.g., "skill X: 0 calls in N days, retire or broaden description?"; "skill Y: high call rate but description appears too narrow — broaden?"; "skill Z: invoked when intent was W — tighten description."

## Context

**This PLAN is intentionally blocked.** Logging will start as soon as PLAN 202604300210 lands, but the *audit* needs:
- Enough skills that Ken can no longer hold all trigger phrases in his head (current count: ~10; threshold: ~15+).
- OR a felt pain signal — Ken keeps invoking the wrong skill, or notices Claude misfiring.

Until one of those, the audit's value-to-effort ratio is poor: with 10 skills and < 1 month of data, every "finding" would be noise.

When unblocked, Sonnet should:
1. Read the accumulated `.claude/_skill_usage.jsonl` to see what *actually* matters in the data (call distribution, repeat-misfire patterns, dead skills).
2. Revise the Steps in this PLAN based on those patterns — the steps below are a *spec sketch*, not Haiku-safe yet.
3. Update `blocked_by` to empty and `status` to `ready`.

## Steps

> **DEFERRED — these steps are a sketch, not Haiku-safe.** Sonnet must rewrite them when the trigger condition hits and real log data is available. Do NOT execute as-is.

### Step 1 (Sonnet, on unblock): Refresh the design
1. Read 30+ days of `.claude/_skill_usage.jsonl`.
2. Note actual call patterns: most-called, least-called, never-called, common args, error rate.
3. Decide: what counts as a "finding"? Likely candidates — dead skill (0 calls in 30 days), drifted description (call rate >> success rate), narrow trigger (specific args dominate).
4. Rewrite Steps 2–6 below as Haiku-safe steps with file paths, exact contents, and verification criteria.

### Step 2 (placeholder): Create skill structure
1. `mkdir -p .claude/skills/skill-usage-audit/{references,workflows,templates}`
2. Write `SKILL.md` (router pattern: read log → compute stats → produce findings PLAN).
3. Write `references/audit-rules.md` — definition of dead skill / drifted description / over-broad trigger.
4. Write `workflows/produce-audit.md` — step-by-step procedure.
5. Write `templates/audit-plan-template.md` — PLAN structure for findings output.

### Step 3 (placeholder): Register the skill
1. Add row to CLAUDE.md Skills table.
2. Add row to `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` Existing Project Skills table.

### Step 4 (placeholder): Test on real data
1. Invoke `Skill("skill-usage-audit")`.
2. Verify it reads the live log without error.
3. Verify it produces a Bus PLAN with reasonable findings (manual sanity check — no false positives on actively-used skills).

### Step 5 (placeholder): Commit, hold push
Standard commit + push-hold per project convention.

### Step 6 (placeholder): Decide cadence
Either:
- Make it manual-only (Ken invokes when curious), or
- Add a RECUR- PLAN (e.g., quarterly audit) — modelled on `RECUR-monthly-claude-md-audit`.

## Verification

> **Placeholder — to be detailed when unblocked.**

- [ ] `.claude/skills/skill-usage-audit/SKILL.md` exists with valid frontmatter
- [ ] Skill can be invoked and reads the live log without error
- [ ] Output PLAN contains plausible findings (no obvious false positives on actively-used skills)
- [ ] Skill registered in CLAUDE.md and SKILLS_IMPLEMENTATION_GUIDE.md
- [ ] One commit; push held

## Executor Notes
*Do not execute while `status: blocked`. When unblocked, Sonnet rewrites Steps 2–6 first, then a Haiku run via execute-plan completes execution.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
