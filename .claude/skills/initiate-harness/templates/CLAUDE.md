# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<!-- PROJECT: one-paragraph description of what this project is, its primary stack, and the constraint that shapes its architecture. -->

## Working style

- AU spelling, usage, date formats.
- New conversation: check the request is well-specified; ask clarifying questions first when it isn't.
- No unprompted output of artefacts, illustrations, code, or longform sections. If producing one seems like the best move, ask permission first.
- Long output: ask whether the direction is right before continuing.
- No cross-chat references between projects unless prompted; isolate within projects by default.
- Plain language; do not compress, abbreviate, or elide. Ruthless token economy is for internal planning only. Always name the thing, the operation, the result.
- When offering options: describe each in full, then say which you lean toward and explain why.
- If a question is answerable from a tool's default behaviour, decide and proceed.
- If something depends on a prior decision or external input, state the dependency.
- Requirement before solution — no mechanism design until requirement and process are agreed.
- Reviews: brief verification preamble (what was checked against — files, not just the document under review) → one-line overall verdict → priority-ordered numbered punch list (blockers first, nits last) → "Not blockers" subgroup → net verdict (what's ready, what needs fixing).

## Agent execution rules

**Plan execution protocol & model switch rules:** See [AGENT_RULES.md](AGENT_RULES.md) for plan phases, model responsibilities, and switch protocol.

**Operating rules:**
1. **All plans go to Bus/** — Every piece of planned work lives as a PLAN file, never chat-only.
2. **Research and advice to Bus/** — RESEARCH (data drops) and ADVICE (strategic notes) go via `write-bus-input` skill. Writing an input auto-clears blocked plans waiting on it.

## Skills

Skills live in `.claude/skills/<name>/SKILL.md`. Invoked via `Skill("<name>")`. See [.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md](.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md) for structure and conventions.

| Skill | What it does |
|---|---|
| `initiate-harness` | Bootstrap a fresh project with the Bus/PLAN harness — creates Bus/, Retired/, AGENT_RULES.md, CLAUDE.md, ROADMAP.md, current-month LOG from templates |
| `create-agent-skills` | Expert guidance for creating and refining skills — structure, principles, workflows, templates |
| `write-bus-plan` | Transcribe plans to `Bus/` files; manage monthly LOG and status tables |
| `write-bus-input` | Write RESEARCH/ADVICE files to `Bus/`; unblock plans waiting on input |
| `execute-plan` | Execute PLAN steps in order; populate Executor Notes; update LOG; commit + push |
| `maintain-claude-md` | Audit CLAUDE.md and CONTEXT_CONSTITUTION.md against camelCase + rot principles; propose adds/prunes; output as Bus PLANs |
| `retire` | Move files to gitignored `Retired/` folder when no longer needed |
| `ideate` | Three-phase ideation arc (Clarify → Survey → Converge) for shaping a problem into a plan-ready idea |
| `audit-haiku-safe` | Mechanical plan-safety audit (Sonnet-pinned) — checks each step is concrete, atomic, unambiguous, safe, testable |
| `audit-sufficiency` | Conceptual plan audit (Opus-pinned) — interrogates assumptions, validation path, test fidelity |
| `plan-pipeline` | End-to-end planning orchestrator — walks a PLAN through drafting → drafted → checked → executing → verifying → complete |

## Architecture & Core Design

<!-- PROJECT: describe the high-level architecture — main entry points, processor / pipeline / module pattern, the design decision that shapes everything else. Reference key files with relative paths. -->

## Key Dependencies

<!-- PROJECT: list the load-bearing libraries with versions and what they do. -->

## Commands

<!-- PROJECT: setup, run, test commands. Block fenced. -->

```bash
# Setup
# Run
# Test
```

## Configuration

<!-- PROJECT: configuration system — file location, sections, path resolution behaviour. -->

## Testing Structure

<!-- PROJECT: test layout, naming convention, fixtures, mocking approach, end-to-end vs unit. -->

## Important Implementation Notes

<!-- PROJECT: gotchas, invariants, "don't change this without understanding why" notes that future agents need to respect. -->
