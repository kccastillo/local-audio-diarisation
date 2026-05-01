# Audit-Haiku-Safe Procedure

## Process

### Step 1: Validate preconditions
- Read the PLAN file at `plan_path`.
  - If unreadable → return `outcome: exception` with diagnostics.
- Confirm `_shared/plan-safe.md` exists and is readable.
  - If missing → `outcome: exception`.
- Check that audit-sufficiency has returned `outcome: success` for this PLAN. Inspect PLAN frontmatter `audit_state.last_stage` and `audit_state.last_outcome`:
  - If `last_stage == sufficiency AND last_outcome == success` (or any later stage with success): preconditions met, proceed.
  - Otherwise → `outcome: exception` with diagnostics: "audit-sufficiency precondition not satisfied; run audit-sufficiency first".

### Step 2: Per-step plan-safety review
For each numbered step in the PLAN's `## Steps` section:

For each step, evaluate against the five plan-safe criteria from `_shared/plan-safe.md`:
1. **Concrete:** specific file paths, exact command syntax, no "likely" / "probably" / "should".
2. **Unambiguous:** no judgement calls; executor runs the steps, doesn't redesign them.
3. **Atomic:** one operation per step; clear success/failure condition.
4. **Safe:** no destructive operations without explicit Human approval; no `--no-verify`/`--force` bypasses.
5. **Testable:** each step has independent, checkable verification criteria.

Classify each finding:
- **Blocker** (decision 14): would cause Haiku to halt, error, or require judgement mid-execution. Examples: "improve X", "audit Y", "if appropriate, Z".
- **Not blocker** (nit): would benefit from sharpening but doesn't block mechanical execution. Examples: line-number anchors instead of text matching; clearer commit message templates.

### Step 3: Cross-step coherence
- Sequencing: do steps reference earlier steps' outputs correctly? Do line-number references account for upstream edits (e.g. step N+1 says "delete lines 23-36" but step N already deleted lines 5-10)?
- File path consistency: do steps reference the same file using the same path style?
- Dependencies: does each step's preconditions hold given prior steps' postconditions?

### Step 4: Verification format check
For each item in the PLAN's `## Verification` section:
- Item must have a shell-runnable annotation directly below the prose:
  - `verify: <shell command>` (state assertion), OR
  - `acceptance: <shell command>` (behavioural check), OR
  - `verify: human` (subjective; surfaced for eyeball).
- At least one `acceptance:` item is required per PLAN (decision 25).
- Items missing the annotation entirely → Blocker.
- Items where the shell command is malformed (e.g. backticks not closed) → Blocker.
- PLAN with zero acceptance: items → Blocker (the Objective cannot be sampled-verified).

### Step 5: Compose review output
Format per CLAUDE.md "Reviews" rule:
1. **Verification preamble** (1-2 sentences): what files were inspected (the PLAN, _shared/plan-safe.md, any referenced source files).
2. **Verdict** (one line): "Mechanically executable" / "N plan-safety blockers — revision needed" / "Pre-condition violation".
3. **Plan-safety section** with two subgroups:
   - `**Blockers**` (numbered, priority-ordered): prose + which plan-safe criterion violated + suggested fix shape (without authoring it).
   - `**Not blockers**`: brief list.
4. **Net verdict**: "ready to advance" / "revise the N blockers and re-audit".
5. **Machine-readable summary**: `Blockers: N` on its own line.

### Step 6: Decision-triage of Human-input items (if any)
If any finding asks the Human for input (rare for plan-safety, common for sufficiency), classify each per parent PLAN 202605011400 decision 15:
- **Already locked** — Human proposed/affirmed earlier; no question.
- **Mechanically forced** — only one alternative; no question.
- **Real judgement call** — surface as a question.

### Step 7: Emit the pipeline-result block
End the agent's response with:
```
<pipeline-result>
```json
{
  "outcome": "success" | "revision_needed" | "exception",
  "payload": {
    "blockers_count": <int>,
    "review_text": "<the formatted review>"
  },
  "diagnostics": { /* if outcome != success */ }
}
```
</pipeline-result>
```

The orchestrator parses this block to drive the audit-loop state machine (decision 21).
