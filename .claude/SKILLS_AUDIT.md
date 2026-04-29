---
title: "Skills Audit — BA → Code Workflow"
type: audit
created: 2026-04-29
status: drafting
---

# Skills Audit — BA → Code Workflow

Purpose: identify every BA-specific assumption in the 5 imported skills (and the 4 supporting docs we already touched) and propose the code-equivalent. Output of this audit becomes the first PLAN through the new bus.

## 1. Renaming proposal

The imported skill names are mostly generic — `execute-plan`, `retire`, `write-bus-plan`. Two options:

**Option A — Minimal trim (recommended).** Drop redundant qualifiers; leave names mostly as-is.

| Current | Proposed | Rationale |
|---|---|---|
| `create-agent-skills` | `create-skill` | "agent" qualifier is redundant; all skills are agent-callable |
| `write-bus-plan` | `write-bus-plan` (keep) | "bus" describes the destination convention, not BA jargon |
| `write-bus-input` | `write-bus-input` (keep) | same |
| `execute-plan` | `execute-plan` (keep) | already generic |
| `retire` | `retire` (keep) | already generic |

**Option B — Explicit code prefix.** Add a `code-` prefix to flag they're for software work.

| Current | Proposed |
|---|---|
| `create-agent-skills` | `create-code-skill` |
| `write-bus-plan` | `write-code-plan` |
| `write-bus-input` | `write-code-input` |
| `execute-plan` | `execute-code-plan` |
| `retire` | `retire-code` (awkward — keep as `retire`) |

**Recommendation: Option A.** This project is single-purpose (coding work) — there's no parallel BA family to disambiguate from, so the `code-` prefix adds verbosity without payoff. The skill *content* is what carries the code-orientation, not the file name. Option B makes sense if you ever run BA work and code work in the same repo.

---

## 2. Per-skill BA-isms and proposed code-equivalents

### 2.1 `write-bus-plan`

**SKILL.md**
| Line | Current | Issue | Proposed |
|---|---|---|---|
| 24 | `Never write to Wiki/ — that is Antigravity's domain` | "Wiki/" and "Antigravity" are seqwater concepts | Delete the constraint entirely |
| 28 | LOG Status Table sort rule | Generic, fine | Keep as-is |

**templates/plan-template.md** — major rework needed. Current frontmatter is BA-flavoured; verification examples reference Wiki and `_index.md`.

Proposed code-flavoured frontmatter (additions in **bold**):
```yaml
---
title: "[Plan title]"
type: bus-plan
status: ready
assigned_to: haiku
priority: medium
created: YYYY-MM-DD
created_by: haiku
created_month: YYYYMM
log_month: YYYYMM
due: ""
repeatable: false
repeat_cadence: ""
linked_inputs: []
blocked_by: ""
rollover_count: 0
triggers_plans: []
closes_thread: ""        # ROADMAP.md or HARNESS.md thread ID this PLAN closes
advances_thread: ""      # ROADMAP.md or HARNESS.md thread ID this PLAN advances
parent_plan_of_plans: ""
**branch: ""**           # git branch the work happens on (empty = current branch)
**files_touched: []**    # files expected to change — informational, helps reviewers
**verification_commands: []**  # exact commands to run during Step 3 verification
**tag_after: ""**        # git tag to apply after success (e.g. "v1-final")
---
```

Rationale: drop `linked_decisions` (no decision-tracking system here). Drop verification examples that reference Wiki. Add code-relevant fields.

Proposed body section changes:
- **Steps section** — keep `[Ken]` and `[blocked-on-input]` markers; they're generic enough.
- **Verification** — replace Wiki/_index.md examples with code-style:
  - `pytest tests/test_<module>.py passes`
  - `python -m mypy <module>/ clean` (if applicable)
  - `python run_diariser.py --input <test-audio> --format json` produces expected output
  - frontmatter check (e.g. ROADMAP.md `last_updated` bumped)
- **Recurring Task section** — keep, but cadence phrasing should accept "after-merge" / "after-release" alongside monthly/quarterly.

**workflows/write-plan.md** — generic enough. No changes.

**references/bus-conventions.md** — minor: thread linkage section says "ROADMAP.md" but in our setup we have both ROADMAP.md (product) and HARNESS.md (harness). Update to: "PLANs may reference threads in either `ROADMAP.md` (T-prefix IDs) or `HARNESS.md` (H-prefix IDs). The prefix on the ID disambiguates which file holds the thread."

**references/bus-naming-convention.md** and **references/assigned_to-field.md** — not yet read; flag for review pass.

---

### 2.2 `write-bus-input`

**SKILL.md**
| Line | Current | Issue | Proposed |
|---|---|---|---|
| 27 | `Never write to Wiki/ — that is the Wiki skills' domain` | seqwater-specific | Delete |

**templates/research-template.md** — too thin for code-relevant research. Proposed additions:

```yaml
---
title: "Research — [topic]"
type: bus-research
created: YYYY-MM-DD
feeds_plan: ""
from: "Haiku"
question_asked: ""
integration_status: pending
**research_kind: ""**  # one of: library-probe | api-behaviour | benchmark | bug-repro | docs-summary
**target_versions: []**  # versions of libraries/tools probed (e.g. ["faster-whisper==1.0.3"])
---

## Findings
[Research findings — paste verbatim. One concept per section heading if multiple.]

## Reproduction
[**NEW.** If this is a bug-repro or benchmark: minimal commands or code snippet to reproduce. If library-probe: the actual probe code/commands run.]

## Sources
[URLs, files, library docs, GitHub issues consulted. Include access dates.]

## Caveats
[Data quality, gaps, version-specific behaviour, known unknowns.]
```

**templates/advice-template.md** — generic enough. Minor: change `from: "Opus via Ken"` default to `from: ""` (we may have advice from sources other than Opus).

**workflows/write-input.md** — generic. No changes.

---

### 2.3 `execute-plan`

**SKILL.md**
| Line | Current | Issue | Proposed |
|---|---|---|---|
| 49–53 | Atomise/Wiki examples in `<skill_invocation_semantics>` | seqwater-specific examples | Replace with code-relevant: e.g. invoking `retire` on a deprecated module, invoking `debug-like-expert` to investigate a failing test |
| 62 | `If outcome is not "done": still commit and push, but clearly mark blockers in the commit body` | Generic, fine | Keep |
| 73–74 | Co-Authored-By trailer with Haiku 4.5 | Model-specific but our convention | Keep |
| 77 | Roadmap sync mentions ROADMAP.md only | Need to handle HARNESS.md too | "Apply roadmap sync to whichever file (ROADMAP.md or HARNESS.md) contains the referenced thread (disambiguated by T-/H- prefix)" |

**workflows/execute-steps.md**
| Step | Issue | Proposed |
|---|---|---|
| 4.5 | Reads `.claude/ROADMAP.md` only | Read `ROADMAP.md` (root) for T-threads, `.claude/HARNESS.md` for H-threads |
| 5 | "For every State/ file modified..." | Drop entirely. Diarizer has no State/ pattern. Replace with: "If CLAUDE.md, ROADMAP.md, or HARNESS.md was modified, ensure their `last_updated` frontmatter is bumped to today." |
| 7 | Commit format spec | Generic, fine |

**references/log-rules.md**
- "State File Updates" section — drop (no State/ in this project).

---

### 2.4 `retire`

**SKILL.md** — fully generic, no changes.

**workflows/retire-file.md** — fully generic, no changes.

(Already correctly identified `Retired/` for gitignore in our `.gitignore`.)

---

### 2.5 `create-agent-skills` (proposed: `create-skill`)

**SKILL.md** — mostly generic, teaches HOW to write skills. No BA-isms in the body.

References (`references/*.md`) — 13 files, not deep-read in this audit. Likely all generic skill-authoring guidance. Flag for spot-check during the PLAN execution.

Proposed change: rename only.

---

## 3. Cross-cutting changes

### 3.1 Paths
- ROADMAP.md is now at root, not `.claude/ROADMAP.md`. Two skill files reference the old path: `execute-plan/SKILL.md` line 77 and `execute-plan/workflows/execute-steps.md` Step 4.5. Update both.
- HARNESS.md is at `.claude/HARNESS.md` (new). Same Step 4.5 needs to handle it.

### 3.2 Thread ID prefix convention
Adopt: `T01..` for product threads in ROADMAP.md, `H01..` for harness threads in HARNESS.md. The prefix disambiguates which file `closes_thread` / `advances_thread` refers to. Document this in `bus-conventions.md`.

### 3.3 References to drop project-wide
Search-and-replace pass during execution:
- `Wiki/`, `Wiki_Map`, `wikilinks`, `[[...]]` patterns — delete or rephrase
- `Antigravity` (seqwater agent name) — delete
- `State/` directory references — delete
- `atomise`, `librarian`, `pandoc-convert`, etc. (skills not imported) — remove from examples
- `Production/Wiki/` paths — delete

### 3.4 SKILLS_IMPLEMENTATION_GUIDE.md update
The "Existing Project Skills" table currently lists 13 seqwater skills. Replace with our actual 5 (or 6 once `debug-like-expert` lands).

### 3.5 CLAUDE.md skills table
Already updated to reflect imported skills. Re-update after rename and `debug-like-expert` addition.

---

## 4. Gap analysis — what's missing?

Reviewed `gsd-build/gsd-2` (~30 skills). Most are not relevant (frontend/UI, MCP, GSD-specific machinery).

### Strong recommendation: import `debug-like-expert`
- Methodical read-only debugging skill: evidence → hypothesis → verify → propose fix → decision gate
- Decision-gate pattern ("ANALYSIS COMPLETE — what would you like to do?") aligns with Ken's "no unprompted artefacts" preference
- Read-only constraint protects against drive-by fixes during investigation
- Will be useful for: T01 (faster-whisper regression hunts), Q1 (WhisperX spike), any v2 debugging session
- Effort: copy SKILL.md + references, trim GSD framing, register in CLAUDE.md
- Tracked as **H02** in HARNESS.md

### Pattern to bake into PLAN template (not a separate skill)
From `decompose-into-slices`: **vertical slices, not horizontal layers**. A PLAN that delivers "schema + integration + tests for one capability end-to-end" beats one that delivers "all schemas first, all integrations later." The principle should appear in the plan-template's Steps section as guidance, not as a separate skill — gsd-2's full skill is too coupled to its `.gsd/milestones/` machinery.

Proposed addition to `plan-template.md` Steps section:
```
## Steps
[Numbered steps for Haiku to execute. Aim for vertical-slice steps —
each step delivers a verifiable end-to-end change rather than a
horizontal layer. Mark [Ken] for human-action steps and [blocked-on-input]
for steps waiting on RESEARCH/ADVICE.]
```

### Considered and skipped
- **`spike-wrap-up`** — relevant concept (Q1 is a spike) but tightly coupled to `.gsd/workflows/spikes/`. Cheaper to handle Q1's wrap-up as steps in the resolving PLAN: "if findings recommend WhisperX adoption, append outcome to ROADMAP.md, close T04, open replacement threads."
- **`review`** — built-in `/review` command exists, no need for a skill version
- **`security-review`** — built-in `/security-review` command exists
- **`handoff`** — possibly useful for model-switch context handoffs, but our AGENT_RULES.md already covers this with the pause block. Revisit if friction appears.
- **`forensics`, `observability`, `lint`, `code-optimizer`, `dependency-upgrade`** — overkill for a single-developer Python project of this size. Reconsider when relevant.
- **`decompose-into-slices`** as a full skill — pattern is valuable, full skill is too GSD-coupled. Bake into plan template instead (above).

### Skills NOT needed
- Anything frontend/UI (accessibility, react, frontend-design, core-web-vitals, make-interfaces-feel-better)
- Anything domain-specific to GSD or MCP (create-mcp-server, create-gsd-extension, gsd-headless)
- Anything CI/release-focused until we have CI (github-workflows)

---

## 5. Suggested PLAN structure

The first PLAN through the bus implements this audit. Suggested step decomposition:

1. Rename `create-agent-skills` → `create-skill`. Update directory, SKILL.md frontmatter, all internal references.
2. Update `write-bus-plan/templates/plan-template.md` — drop `linked_decisions`, add `branch`, `files_touched`, `verification_commands`, `tag_after`. Replace Wiki/`_index.md` verification examples with code-style.
3. Update `write-bus-input/templates/research-template.md` — add `research_kind`, `target_versions`, `## Reproduction` section.
4. Update `write-bus-input/templates/advice-template.md` — generalise `from` field.
5. Update `execute-plan/SKILL.md` and `execute-plan/workflows/execute-steps.md` — fix ROADMAP path, add HARNESS.md handling, drop State/ section, replace seqwater examples.
6. Update `bus-conventions.md` — document T-/H- prefix convention; update thread linkage path notes.
7. Strip Wiki/Antigravity/atomise/State/ references project-wide (search-and-replace).
8. Update `SKILLS_IMPLEMENTATION_GUIDE.md` "Existing Project Skills" table.
9. Update CLAUDE.md skills table.
10. Add bake-in to plan-template Steps section guiding vertical-slice authoring.
11. Import `debug-like-expert` from gsd-2 (skill body + selected references), trim GSD framing.
12. Verify: `Skill("create-skill")` invocation works (manual smoke); plan-template renders with new fields; no remaining grep hits for "Wiki", "Antigravity", "State/" in `.claude/`.
13. Retire `.claude/SKILLS_AUDIT.md` once consumed.

Each step is small, verifiable, and code-only — no judgment calls. Haiku-safe.
