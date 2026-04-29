# Agent Operating Rules

Foundational rules for all agent work in this project. Referenced in CLAUDE.md.

## Plan Execution Protocol

Non-trivial work runs through a PLAN in `Bus/`, never chat-only. Each phase runs under a specific model:

| Phase | Model | What happens |
|---|---|---|
| 1. Draft | Sonnet | Converse with Ken; finalise PLAN content |
| 2. Transcribe | Haiku | Write PLAN file to `Bus/` via `write-bus-plan` skill |
| 3. Review *(optional)* | Opus | Review PLAN for gaps, dependencies, breakage risk |
| 4. Revise *(if needed)* | Sonnet | Fold review feedback into PLAN |
| 5. Execute | Haiku | Run steps via `execute-plan` skill; populate Executor Notes |
| 6. Closeout | Haiku | Update monthly LOG; git commit + push |
| 7. Post-execution review | Opus | Review execution; append hiccup entries to `.claude/skills/_hiccups.md` for any deviation, silent failure, or hard error observed during the run |

**Phase transitions require a model switch.** Never cross a phase boundary without Ken prompting the switch.

**Interrupted runs:** if Ken halts a Haiku run mid-execution, he will describe what he caught. That description is a hiccup entry — Sonnet or Opus writes it to `.claude/skills/_hiccups.md`.

## Model Switch Protocol

When a model switch is needed, output this pause block:

```
⏸ MODEL SWITCH NEEDED
Switch to: [HAIKU | OPUS | SONNET]
Why: [one sentence]
Give it: [exact prompt or file path]
Return here after: [what to bring back — the main driver]
```

Stop and wait for Ken's response. Never continue past a model switch.
