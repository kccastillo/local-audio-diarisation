# Audit-Sufficiency Procedure

This is conceptual review, not mechanical. Each lens has a concrete prompt; apply all seven, classify findings as Blocker or Not-blocker, then compose the structured review.

## Step 1: Validate preconditions
- Read the PLAN file at `plan_path`.
  - Unreadable → `outcome: exception` with diagnostics.
- Check structural completeness: Objective, Context, Steps, Verification sections all exist.
  - Missing → `outcome: exception` with diagnostics.
- For each filename in PLAN frontmatter `linked_inputs:`, verify the file exists at `Bus/<filename>`.
  - Missing → `outcome: exception`.
- Best-effort scan of Steps section for cited source-file paths; note any that don't exist as `freshness` lens findings (not exceptions).

## Step 2: Read the PLAN's referenced inputs
- For each `linked_inputs:` file: read it. RESEARCH files inform the Context; ADVICE files often capture decisions you need to know about during sufficiency review.
- For source files cited in Steps (when paths look real): spot-check 2-3 of them. Don't try to read everything — the goal is sampling, not exhaustive context loading.

## Step 3: Apply the seven lenses

For each lens, use the prompt below. Generate findings; classify each as Blocker or Not-blocker.

### Lens 1 — Assumptions
**Prompt:** What does this PLAN's design depend on that hasn't been confirmed in this Claude Code version, this codebase, this team's tooling, or this environment? Surface load-bearing assumptions before they become silent failures.

**Examples of blockers:** "Step 3 assumes feature X is supported but probe-1 didn't verify"; "Step 7 depends on Bash command Y being available — not in `.claude/settings.json` permissions"; "Decision N references behaviour that isn't documented anywhere".

**Examples of not-blockers:** "Step 5 assumes a tool returns JSON; usually it does but format may vary".

### Lens 2 — Validation path
**Prompt:** When does this PLAN's design first get tested? If late, what's at risk if a foundational assumption is wrong? Look for the gap between "we built it" and "we verified it works".

**Examples of blockers:** "PLAN builds 6 things and only validates at the very end via dogfood — if assumption X is wrong, we discover after building all 6"; "No smoke test exists for the orchestrator's re-entry pattern".

**Examples of not-blockers:** "Validation is at end-of-PLAN rather than per-step, but the steps are tightly coupled so per-step doesn't add much".

### Lens 3 — Test fidelity
**Prompt:** Does the dogfood / smoke test exercise real friction, or is it a contrived synthetic that doesn't surface real-world issues?

**Examples of blockers:** "Dogfood target is `note-jot` — synthetic skill with no real ideation surface; building it doesn't prove the pipeline survives real planning friction"; "Test data is hand-crafted and won't trigger edge cases that real data would".

**Examples of not-blockers:** "Test target is small but exercises the critical path".

### Lens 4 — Edge cases at the orchestration layer
**Prompt:** What if a referenced artefact is missing, malformed, or fails? Are these handled or silent?

**Examples of blockers:** "Orchestrator re-entry on a missing PLAN file isn't specified — would silently advance"; "Loop doesn't have max-iteration guard — could runaway"; "If a subagent returns malformed `<pipeline-result>` block, behaviour undefined".

**Examples of not-blockers:** "Subagent dispatch error reporting could be more verbose".

### Lens 5 — Freshness
**Prompt:** Anything elsewhere in the codebase, memory, or docs that this design contradicts? Anything that will become misleading after this PLAN executes?

**Examples of blockers:** "Memory note says X always commits and pushes; this PLAN decouples X from git but doesn't update the memory"; "CLAUDE.md skill table will be stale after this PLAN — but no step updates it".

**Examples of not-blockers:** "Workflow file has a typo from a previous edit; harmless but worth fixing".

### Lens 6 — Meta
**Prompt:** Is the design over-engineered? Could it be smaller? What's the minimum viable version of this PLAN?

**Examples of blockers:** "PLAN has 14 decisions but only 3 are real choices — others are mechanically forced; the 14-decision header obscures the real design surface"; "Three steps could be one if we accept slightly less granular Verification".

**Examples of not-blockers:** "PLAN is comfortable with bootstrap chicken-and-egg; that's fine — it's a one-time cost".

### Lens 7 — Spec-acceptance fidelity (per parent decision 25)
**Prompt:** Does the PLAN's `acceptance:` sample(s) actually exercise what the Objective claims to deliver? Or are they incidental box-ticks that pass without the deliverable working?

**Examples of blockers:** "Objective says 'ship a working note-jot skill' but Verification only checks the SKILL.md file exists — never invokes the skill on a real input"; "PLAN has zero `acceptance:` items — only `verify:` state assertions".

**Examples of not-blockers:** "Acceptance check could be more thorough but covers the critical path".

## Step 4: Triage Human-input items (decision 15)

Across all lens findings, identify any Blocker that requires Human input to resolve (typical for sufficiency: requirement-clarification, decision-on-tradeoff, scope-question). Classify each per parent PLAN 202605011400 decision 15:

- **Already locked** — Human proposed/affirmed earlier in conversation/PLAN history. List for transparency, no question.
- **Mechanically forced** — only one alternative; downstream consequence of locked decisions. List for transparency, no question.
- **Real judgement call** — surface as a question.

Only Real-judgement-call items appear as questions in the surfaced review.

## Step 5: Compose review output

Format per CLAUDE.md "Reviews" rule:

```markdown
## Verification preamble
Reviewed PLAN [filename] against:
- [list of inspected files: linked_inputs, source files]
- Cross-checked decisions [list] against [files where they're documented]

## Verdict
[One line]

## Sufficiency findings

### Lens 1 — Assumptions
**Blockers**
1. [Finding] — suggested fix shape: [direction, not full text]

**Not blockers**
- [Nit]

### Lens 2 — Validation path
[same shape]

[... repeat for all 7 lenses ...]

## Triaged Human-input items
**Already locked** (transparency only):
- [item]

**Mechanically forced** (transparency only):
- [item]

**Real judgement calls** (Human input requested):
- [question]

## Net verdict
[ready to advance / revise the N blockers and re-audit]

Blockers: N
```

## Step 6: Emit the pipeline-result block

End the agent's response with:
```
<pipeline-result>
```json
{
  "outcome": "success" | "revision_needed" | "exception",
  "payload": {
    "blockers_count": <int>,
    "review_text": "<the formatted review>",
    "triaged_human_items": [
      { "class": "real_judgement_call", "item": "..." },
      ...
    ]
  },
  "diagnostics": { /* if outcome != success */ }
}
```
</pipeline-result>
```

The orchestrator parses this block to drive the audit-loop state machine (parent decision 21).

## Operating notes

- **The exemplar (`Bus/202605011500_ADVICE_sufficiency-audit-exemplar.md`) is the calibration target.** If your review is shallower than the exemplar, you missed lenses or didn't read referenced files. If it's heavier, you may have ventured into mechanical territory — refactor those findings out (they belong to `audit-haiku-safe`).
- **Token economy:** sufficiency audit is the most expensive phase per dispatch (Opus reading PLAN + multiple inputs + sampled source files). One pass per loop iteration is the goal — don't redo work the previous iteration already covered unless the revision actually invalidates prior findings.
- **Stay conceptual.** If you find yourself counting commas in a step, you're in the wrong skill. Stop and refactor that finding out.
- **The exemplar surfaced 6 issues across 6 lenses** in a single Opus pass. Don't expect to find issues at every lens every time — sometimes a lens reports "no findings" and that's the correct output.
