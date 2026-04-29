---
title: "Diarizer Harness Roadmap"
type: roadmap
scope: .claude/ harness — skills, conventions, agent operating model
created: 2026-04-29
last_updated: 2026-04-29
status: drafting
---

# Diarizer Harness Roadmap

## Mission

Keep the `.claude/` harness aligned with this project's actual workflow (code, not BA/knowledge work). Skills imported from sibling projects start BA-flavoured; trim and re-fit them to coding conventions. Add only the skills we'll actually use.

## Threads

Threads keep stable IDs (H01..). Closed threads stay in this list with `Status: closed` for historical context.

**H01 — Adapt imported skills for coding workflow**
- Status: identified
- Why: 5 skills (`create-agent-skills`, `write-bus-plan`, `write-bus-input`, `execute-plan`, `retire`) were imported from `seqwater-app-control-2026`, where the workflow centres on BA deliverables, a Wiki/atomise pipeline, and State/ files. Several conventions (Wiki references, State/ paths, BA-flavoured templates) don't apply here. Skills need rename, content tweaks, and code-relevant frontmatter before the first proper PLAN runs through them.
- Direction: see `.claude/SKILLS_AUDIT.md` for the itemised findings and proposed changes. First PLAN through the bus will execute these tweaks (dogfooding run).
- Blocks: every other PLAN. Linked: H02.

**H02 — Add `debug-like-expert` skill**
- Status: identified
- Why: a methodical, read-only debugging skill (port from `gsd-build/gsd-2`) is well-suited to v2 rewrite work — especially the WhisperX spike and any backend-swap regression hunts. Decision-gate pattern (no auto-implement) aligns with Ken's working style.
- Direction: copy `gsd-2/src/resources/skills/debug-like-expert/` (SKILL.md + references), trim GSD-specific framing, register in CLAUDE.md skills table.
- Blocks: nothing. Linked: H01.
