---
name: audit-sufficiency
description: Conceptual audit of a PLAN (Opus-pinned). Interrogates assumptions, validation path, test fidelity, orchestration edge cases, freshness, meta-design, and spec-acceptance fidelity. The kind of pass that catches what mechanical review misses — "we can build it, but is it the right thing?". Returns structured review with machine-readable Blockers count. Runs FIRST in the drafted-phase audit loop; plan-safety-auditor only runs after this passes. Trigger phrases: "opus pass", "sufficiency audit", "course-correct check", "check sufficiency", "audit-sufficiency".
---

**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md).

**Worked exemplar:** See [`Bus/202605011500_ADVICE_sufficiency-audit-exemplar.md`](../../../Bus/202605011500_ADVICE_sufficiency-audit-exemplar.md) — captures what an Opus-pass output looks like, generated from the parent PLAN of plan-pipeline itself. Authors and future iterators of this skill should ground in this exemplar.

<essential_principles>
Conceptual lens — NOT mechanical. Per-step concreteness, atomicity, exact-text checks belong to `audit-haiku-safe`; do not duplicate them here.
Apply seven lenses in order: assumptions, validation path, test fidelity, orchestration edge cases, freshness, meta, spec-acceptance fidelity. Each lens has a concrete prompt; each finding gets classified Blocker or Not-blocker.
Read the PLAN AND its referenced inputs (`linked_inputs`) AND any source files the PLAN cites — sufficiency review needs context the PLAN itself can't fully express.
Output ends with a machine-readable `Blockers: N` summary so the orchestrator can apply the gate (parent decision 14).
Apply decision-triage (parent decision 15) to any Human-input items: classify Already-locked / Mechanically-forced / Real-judgement-call before surfacing.
Do not fix the PLAN. Surface findings; the Human revises and re-audits.
Sufficiency must pass before plan-safety runs (parent decision 21 loop) — no point checking Haiku-safety on an insufficient plan.
</essential_principles>

<preconditions>
Before starting:
- PLAN file path is provided as input.
- PLAN file exists and is readable.
- PLAN has the expected sections: Objective, Context, Steps, Verification.
- Any files referenced in PLAN's `linked_inputs:` exist and are readable (RESEARCH/ADVICE inputs).
- Any source files cited in PLAN's Context or Steps are accessible (best-effort — orchestrator can't verify all code paths upfront).
</preconditions>

<inputs>
- `plan_path: string` — absolute or repo-relative path to the PLAN file under review.
</inputs>

<output_schema>
- `outcome: enum[success, revision_needed, exception]`
  - `success` → Blockers: 0; PLAN is sufficient; orchestrator advances to dispatch `plan-safety-auditor`.
  - `revision_needed` → Blockers: N>0; PLAN has sufficiency issues; orchestrator surfaces findings, awaits revision, re-audits sufficiency on next iteration.
  - `exception` → preconditions failed (PLAN unreadable, malformed, linked_inputs missing); orchestrator pulls kanban full-stop.
- `payload:`
  - `blockers_count: int`
  - `review_text: string` — formatted review per the CLAUDE.md "Reviews" rule, with each lens's findings.
  - `triaged_human_items: list` — Human-input items classified per decision 15 (Already-locked, Mechanically-forced, Real-judgement-call).
- `diagnostics:` — populated when outcome != success; otherwise empty.

The agent's response message ends with a `<pipeline-result>` block (parent decision 23):
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
- PLAN structurally malformed (no Steps section, no Verification section, no Objective).
- One or more `linked_inputs:` files missing on disk.
- Source files cited in Steps reference paths that don't exist (best-effort; not a strict precondition — flag in diagnostics rather than always halting).
</exception_conditions>

**Audit procedure:** See [workflows/audit-sufficiency-steps.md](workflows/audit-sufficiency-steps.md) for the seven-lens procedure.

<output_format>
Per the CLAUDE.md "Reviews" rule:
1. **Verification preamble** — what files were inspected (PLAN, linked_inputs, source files cited). Be specific: "Reviewed PLAN against [files A, B, C]" not "reviewed against project files".
2. **One-line overall verdict** — "Sufficient and ready for plan-safety audit" / "N sufficiency blockers — revise before re-auditing" / "Pre-condition violation — see diagnostics".
3. **Lens-by-lens findings**, each lens has its own subsection with:
   - `**Blockers**` (numbered, priority-ordered): finding + lens that surfaced it + suggested fix shape (without authoring it).
   - `**Not blockers**` (nits): brief list.
4. **Triaged Human-input items** (decision 15): if any blocker requires Human input, classify Already-locked / Mechanically-forced / Real-judgement-call. Only Real-judgement-call items become questions for the Human.
5. **Net verdict** — "ready to advance to plan-safety" / "revise the N blockers and re-audit".
6. **Machine-readable summary** — `Blockers: N` on its own line (e.g. `Blockers: 0`, `Blockers: 3`).

**Definition of "blocker"** (parent decision 14): any finding that would cause downstream work to halt, error, or be forced into a judgement call. Operator-procedural recommendations are nits.
</output_format>

<constraints>
- Never modify the PLAN under review.
- Never run any of the PLAN's `verify:`/`acceptance:` commands at this phase — that's the orchestrator's outcome-verification phase, not this audit.
- Never make mechanical observations (line numbers, exact-text matching, per-step concreteness). That's `audit-haiku-safe`'s job.
- Always include the machine-readable `Blockers: N` line.
- Subagent context: invoked via the `sufficiency-auditor` agent, which preloads this skill via `skills:` frontmatter (decision 17). Do not assume access to the parent's wider skill registry.
- Sequencing: this skill is intended to run FIRST in the audit loop. If invoked after `audit-haiku-safe` has already passed, that's fine — runs are idempotent — but the natural order is sufficiency → plan-safety.
</constraints>

<success_criteria>
- Review prose follows CLAUDE.md "Reviews" rule format.
- All seven lenses were applied with at least one observation each (or explicit "no findings under this lens").
- Findings classified Blocker or Not-blocker.
- Decision-15 triage applied to any Human-input items.
- Output ends with `Blockers: N` machine-readable summary.
- Output ends with a `<pipeline-result>` block (decision 23).
- Verification preamble names the specific files inspected, not just "the PLAN".
- For Blockers, suggested fix shape is sketched — not authored, but enough that the Human knows what direction the revision should take.
</success_criteria>
