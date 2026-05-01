---
name: audit-haiku-safe
description: Mechanical plan-safety audit (Sonnet-pinned). Reviews a PLAN file step-by-step against the shared plan-safe definition; checks each step is concrete, atomic, unambiguous, safe, testable; checks Verification items have valid verify:/acceptance:/human format with at least one acceptance: per PLAN. Returns structured review with machine-readable Blockers count. Trigger phrases: "haiku-safe check this", "is this plan haiku-safe", "verify plan-safety", "audit-haiku-safe".
---

**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md) — single source of truth shared with execute-plan.

<essential_principles>
Mechanical only — no conceptual review. Sufficiency is `audit-sufficiency`'s job.
Read each PLAN step. Evaluate against the five plan-safe criteria. Classify as Blocker or Not-blocker.
Cross-step coherence: ordering, line-number consistency after upstream edits.
Verification format: every item must have `verify:` / `acceptance:` / `verify: human`. At least one `acceptance:` per PLAN.
Output a machine-readable `Blockers: N` summary so the orchestrator can apply the gate.
Do not fix the PLAN. Surface findings; the Human revises and re-audits.
Wire format: end response with literal `<pipeline-result>` containing JSON code fence per parent decision 23. No XML payload, no HTML escaping.
</essential_principles>

<preconditions>
Before starting:
- PLAN file path is provided as input.
- PLAN file exists and is readable.
- Shared plan-safe definition exists at `.claude/skills/_shared/plan-safe.md`.
- audit-sufficiency has run and returned `outcome: success` (no point haiku-safety-checking an insufficient plan). If not yet run, return `outcome: exception`.
</preconditions>

<inputs>
- `plan_path: string` — absolute or repo-relative path to the PLAN file under review.
</inputs>

<output_schema>
- `outcome: enum[success, revision_needed, exception]`
  - `success` → Blockers: 0; PLAN is mechanically executable; orchestrator advances `pipeline_phase: drafted → checked`.
  - `revision_needed` → Blockers: N>0; PLAN has plan-safety issues; orchestrator surfaces findings, awaits revision.
  - `exception` → preconditions failed (e.g. invoked before sufficiency passed; PLAN unreadable; shared plan-safe.md missing).
- `payload:`
  - `blockers_count: int`
  - `review_text: string` — formatted review per the CLAUDE.md "Reviews" rule.
- `diagnostics:` — populated when outcome != success; otherwise empty.

The agent's response message ends with a `<pipeline-result>` block (decision 23 of PLAN 202605011400):
```
<pipeline-result>
```json
{ "outcome": "...", "payload": { ... }, "diagnostics": { ... } }
```
</pipeline-result>
```
</output_schema>

<exception_conditions>
- PLAN file unreadable.
- audit-sufficiency has not yet returned `outcome: success` (precondition violation).
- Shared `_shared/plan-safe.md` reference missing.
- PLAN structurally malformed (no Steps section, no Verification section).
</exception_conditions>

**Review procedure:** See [workflows/audit-haiku-safe-steps.md](workflows/audit-haiku-safe-steps.md).

<output_format>
Brief verification preamble (what was checked against — files, not just the PLAN under review)
→ one-line overall verdict
→ for the single Plan-safety section:
   - **Blockers** subgroup (each item: prose + the criterion violated + suggested fix shape)
   - **Not blockers** subgroup
→ Net verdict
→ Machine-readable summary line: `Blockers: N` (e.g. `Blockers: 0` or `Blockers: 3`).

**Definition of "blocker"** (per PLAN 202605011400 decision 14): any finding that would cause Haiku to halt, error, or be forced into a judgement call mid-execution. Operator-procedural recommendations are nits, not blockers.

**Decision-triage of any Human-input items** (per parent decision 15): if a finding requires Human input, classify as Already-locked / Mechanically-forced / Real-judgement-call before surfacing. Only Real-judgement-call items become questions.
</output_format>

<constraints>
- Never modify the PLAN under review.
- Never run any of the PLAN's verify:/acceptance: commands at this phase — that's the orchestrator's outcome-verification phase, not this audit.
- Never recommend conceptual changes (sufficiency lens). Stay mechanical.
- Always include the machine-readable `Blockers: N` line.
- Subagent context: invoked via the `plan-safety-auditor` agent, which preloads this skill via `skills:` frontmatter (decision 17). Do not assume access to the parent's wider skill registry.
</constraints>

<success_criteria>
- Review prose follows CLAUDE.md "Reviews" rule format.
- Every PLAN step has been classified Blocker or Not-blocker.
- Verification items have been checked for shell-runnable format (verify: / acceptance: / verify: human) AND at least one acceptance: per PLAN.
- Output ends with `Blockers: N` machine-readable summary.
- Output ends with a `<pipeline-result>` block (decision 23).
- Decision-triage applied to any Human-input items.
</success_criteria>
