# CLAUDE.md

Directives only. Rationale, examples, and standing rules with context live in [.claude/CONSTITUTION.md](.claude/CONSTITUTION.md).

## Project Overview

Offline speaker-diarisation + transcription pipeline (Whisper + pyannote.audio). See [README.md](README.md) for usage and [ARCHITECTURE.md](ARCHITECTURE.md) for design.

## Working style

- AU spelling, usage, date formats.
- New conversation: ask clarifying questions if the request is under-specified.
- **IMPORTANT:** Treat "X is broken" / "we should Y" / "how about Z" as discussion openers, not work orders. See `.claude/CONSTITUTION.md` § Discussion-vs-work-order.
- No unprompted output of artefacts, illustrations, code, or longform sections — ask permission first.
- Long output: ask whether the direction is right before continuing.
- No cross-chat references between projects unless prompted.
- To Ken: plain language; name the thing, the operation, the result. No compression, no abbreviation.
- When offering options: describe each in full; state recommendation with reason.
- If a question is answerable from a tool's default behaviour, decide and proceed.
- State dependencies on prior decisions or external input explicitly.
- **IMPORTANT:** Requirement before solution — no mechanism design until requirement and process are agreed.
- **IMPORTANT:** Reviews follow the fixed five-element shape. See `.claude/CONSTITUTION.md` § Review shape.

## Agent execution rules

See [.claude/CONSTITUTION.md](.claude/CONSTITUTION.md) § Plan lifecycle for plan phases, halt conditions, and the skill registry.

**Operating rules:**
1. All plans go to `Bus/` — never chat-only.
2. RESEARCH and ADVICE go to `Bus/` via the `write-bus-input` skill.
3. **Bash:** avoid `&&` / `;` / pipes that combine distinct operations in one call. See `.claude/CONSTITUTION.md` § Bash compounds.
4. Delegate read-heavy searches (>5 files or >1500 lines) to a subagent — protects parent context from rot.
5. Cross-session state lives in `memory/MEMORY.md`.
6. **PLAN-or-not triage:** use the Bus PLAN mechanism for tasks that need audit-trail + reproducibility + multi-step coordination (typically: ≥3 Steps, ≥3 files touched, real-judgement-calls present, or audit/sign-off required). Below that bar, use direct edits; the PLAN overhead outweighs its benefit. When in doubt, prefer direct work and only escalate to PLAN if the task grows.
7. The `.claude/` harness is treated as production / stable upstream. This project does not modify harness internals.

## Skills

Skills live in `.claude/skills/<name>/SKILL.md`. Invoke via `Skill("<name>")`. See [.claude/CONSTITUTION.md](.claude/CONSTITUTION.md) § Skill registry for the full list.
