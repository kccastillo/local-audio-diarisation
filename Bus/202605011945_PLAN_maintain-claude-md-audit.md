---
title: "Maintain CLAUDE.md — audit findings 2026-05-01"
type: bus-plan
status: ready
assigned_to: ""
priority: medium
created: 2026-05-01
created_by: maintain-claude-md
created_month: 202605
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: ""
rollover_count: 0
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Apply audit findings from `maintain-claude-md` against `CLAUDE.md` and `.claude/CONTEXT_CONSTITUTION.md` (run on 2026-05-01). User asked for audit + prune candidates; this PLAN bundles both because all the prune candidates surfaced as audit blockers/warns.

## Context
- `CLAUDE.md`: **163 lines** (over soft cap 150, well under hard cap 300).
- `.claude/CONTEXT_CONSTITUTION.md`: **missing** (file does not exist).
- 3 blockers, 5 warns, 4 suggestions.
- All file/skill pointers in CLAUDE.md resolve except CONTEXT_CONSTITUTION.md (which isn't currently referenced — separate finding).

## Findings

### Blockers

1. **CONTEXT_CONSTITUTION.md missing** — checks H4, C3 — `.claude/CONTEXT_CONSTITUTION.md` does not exist on disk, and CLAUDE.md has no pointer to it. The audit checklist requires both. **Fix:** either (a) create a stub from `initiate-harness/templates/` and add a top-of-file pointer in CLAUDE.md, or (b) explicitly document an exception (skill scope says these are required, so option (a) is preferred).

2. **Codebase duplication** — anti-pattern *Codebase duplication* + checks F1, G1 — the following sections document facts authoritative elsewhere and will drift:
   - `CLAUDE.md:47-76` "Architecture & Core Design" — module-by-module description; derivable from `processors/`, `run_diariser.py`, `utils/`, `config/`.
   - `CLAUDE.md:78-85` "Key Dependencies" — duplicates `requirements.txt`, with pinned versions (pyannote.audio 3.3.2, torch 2.7.1+cu118, openai-whisper 20240930, pytest 8.4.1) that will silently drift.
   - `CLAUDE.md:135-145` "Configuration" — duplicates `config/config.yaml` schema.
   - `CLAUDE.md:147-152` "Testing Structure" — derivable from `tests/` directory layout.
   **Fix:** move each to `.claude/references/<topic>.md` and replace with a one-line pointer; or delete and let Claude read source.

3. **Skills table drift** — anti-pattern *Codebase duplication* + check E2 — `CLAUDE.md:37-45` table lists 7 skills but `.claude/skills/` contains 11 (missing entries: `plan-pipeline`, `ideate`, `audit-haiku-safe`, `audit-sufficiency`). The table duplicates the directory. **Fix:** replace the table with a one-line pointer to `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md` plus a directory listing instruction; or accept the table and add the four missing rows (lower-effort, but drift recurs next time skills are added).

### Warns

4. **Soft cap exceeded** — A1 — 163 > 150 lines. Prune pass (blockers 2 + 3) takes file to ~80 lines, well under cap.

5. **Commands block over trinity budget** — B2 — `CLAUDE.md:87-127` is ~40 lines vs. checklist rule "≤3 commands inline; rest as pointer". **Fix:** keep the 2-3 most-used (`pytest tests/ -v`, `python run_diariser.py --input ...`); move the rest to `.claude/references/commands.md`.

6. **Subagent / delegation triggers missing** — H2 + anti-pattern *Subagent / delegation rules missing* — no documented threshold for when to delegate to a subagent vs. handle inline. Default behaviour blows context on big searches. **Fix:** add a 2-3 line rule in Agent execution rules section.

7. **Scratchpad pattern half-documented** — H1 — Bus/ for plans is documented (Operating rules), but `memory/` cross-session pattern is not. **Fix:** add one bullet to Operating rules pointing at `memory/MEMORY.md` for cross-session state.

8. **Inline version numbers** — G1 + anti-pattern *Static maintenance smell* — pyannote.audio 3.3.2, torch 2.7.1+cu118, openai-whisper 20240930, pytest 8.4.1. Resolved automatically if blocker 2 is applied (versions move to a reference or are deleted).

### Suggestions

9. **Add IMPORTANT/MUST markers** — C2 — Working-style rules (`CLAUDE.md:13-23`) are all flat bullets. The load-bearing ones ("No unprompted output of artefacts", "Requirement before solution", "Reviews format") deserve IMPORTANT markers for instruction weighting.

10. **Move v2-rewrite note out of Project Overview** — `CLAUDE.md:9` is a status note ("v2 rewrite is being planned") that will become stale. Move to `ROADMAP.md` and reference; or delete once v2 starts.

11. **Consolidate "Important Implementation Notes" with Caveats** — `CLAUDE.md:154-162` are project-specific imperatives (VRAM management, speaker attribution edge case, pyannote auth). These ARE trinity caveats (B3) and should sit higher in the file with IMPORTANT markers, not buried at the end.

12. **Consider whether CONTEXT_CONSTITUTION.md is needed for this project** — the harness `initiate-harness` skill provides templates for it; if the constitution adds value here (rot recovery protocol, subagent rules, etc.), create it. If the working-style + agent-execution sections of CLAUDE.md already cover the constitution's role for a single-developer project, document an exception in CLAUDE.md and update the audit checklist (via a future add-mode PLAN against the checklist).

## Steps

Execute in order. Steps 1–3 resolve all blockers; 4–7 resolve warns; 8–11 resolve suggestions. Each step is independently verifiable.

1. **Decide on CONTEXT_CONSTITUTION.md** (resolves blocker 1, suggestion 12). Pick option (a) create stub from `initiate-harness/templates/` and add `> See [.claude/CONTEXT_CONSTITUTION.md](.claude/CONTEXT_CONSTITUTION.md) for context-rot recovery protocol.` in CLAUDE.md top 10 lines, OR option (b) add a "CONTEXT_CONSTITUTION.md not used in this project — single-developer scope" line to CLAUDE.md and open a future PLAN to relax the checklist's H4/I-section requirements for solo projects.

2. **Migrate codebase-duplication sections to `.claude/references/`** (resolves blockers 2, warn 8, suggestion 9 partially). Create:
   - `.claude/references/architecture.md` — move CLAUDE.md:47-76 verbatim.
   - `.claude/references/configuration.md` — move CLAUDE.md:135-145.
   - `.claude/references/testing.md` — move CLAUDE.md:147-152.
   - `.claude/references/dependencies.md` — move CLAUDE.md:78-85, OR delete entirely (requirements.txt is authoritative).
   Replace each removed section in CLAUDE.md with a one-line pointer.

3. **Resolve skills-table drift** (resolves blocker 3). Pick: replace the table at CLAUDE.md:37-45 with `> Skills live in .claude/skills/<name>/SKILL.md. See [.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md](.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md) for the registry and conventions.` and delete the table; OR add the four missing rows (`plan-pipeline`, `ideate`, `audit-haiku-safe`, `audit-sufficiency`). Recommend the pointer (eliminates recurring drift).

4. **Trim Commands block** (resolves warn 5). Keep 2 commands inline (the pytest one-liner and the `python run_diariser.py --input ...` invocation). Move the rest to `.claude/references/commands.md`. Add pointer.

5. **Add subagent/delegation rule** (resolves warn 6). Add to "Agent execution rules" section, e.g. "Delegate to a subagent (general-purpose / Explore) for any search expected to read >5 files or >1500 lines; handle inline otherwise."

6. **Add memory pointer** (resolves warn 7). Add bullet to Operating rules: "Cross-session state lives in `memory/MEMORY.md` (auto-loaded). Save user preferences and project facts there, not chat-only."

7. **Apply IMPORTANT markers** (resolves suggestion 9). Mark these Working-style rules with `**IMPORTANT:**` prefix: lines 15 (no unprompted output), 22 (requirement before solution), 23 (reviews format).

8. **Decide on v2-rewrite note** (resolves suggestion 10). Move CLAUDE.md:9 to `ROADMAP.md` and delete from CLAUDE.md, OR keep but add a date-stamp `(as of 2026-05-01)` and a follow-up to revisit.

9. **Consolidate Caveats higher in file** (resolves suggestion 11). Move CLAUDE.md:154-162 ("Important Implementation Notes") into a new "Caveats" subsection under "Project Overview" near the top; add IMPORTANT markers.

10. **Recount lines** — verify CLAUDE.md is now ≤150 (target ~80–100).

11. **Run audit checklist again** — confirm zero blockers remain.

## Verification

- verify: `wc -l CLAUDE.md` returns ≤150 — acceptance: yes.
- verify: `test -f .claude/CONTEXT_CONSTITUTION.md && grep -q "CONTEXT_CONSTITUTION.md" CLAUDE.md` — acceptance: 0 exit (option a) OR explicit exception line present in CLAUDE.md (option b).
- verify: every pointer in CLAUDE.md resolves — `grep -oE '\[.*\]\(([^)]+)\)' CLAUDE.md` paths all exist.
- verify: no inline pinned version strings remain in CLAUDE.md (`grep -E '\b[0-9]+\.[0-9]+\.[0-9]+' CLAUDE.md` returns empty or only AU date formats).
- verify: skills table either deleted or contains all 11 skills present in `.claude/skills/`.
- human: Ken re-reads top 30 and bottom 20 lines of CLAUDE.md and confirms load-bearing rules are at the edges with IMPORTANT markers.
- acceptance: zero blockers when `maintain-claude-md` audit re-runs.

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
