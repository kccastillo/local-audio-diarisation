# Agent Operating Rules

Foundational rules for plan execution. Referenced from CLAUDE.md.

## Plan Execution Protocol

Non-trivial work runs through a PLAN in `Bus/`, never chat-only.

### Lifecycle

```
ready → in-progress → done | partially-complete | blocked | needs-revision
```

- `ready` — transcribed, not yet started.
- `in-progress` — execution has started.
- `done` — all verification criteria pass. Terminal.
- `partially-complete` — some steps done, others blocked or deferred. Terminal for this cycle.
- `blocked` — cannot proceed; `blocked_by` holds the reason. Cleared automatically by `write-bus-input` when the resolving RESEARCH/ADVICE lands, or manually by Ken.
- `needs-revision` — plan itself is faulty; halt and surface to Ken.

### Phases

1. **Draft** — Converse with Ken; finalise PLAN content.
2. **Transcribe** — Write PLAN file to `Bus/` via the `write-bus-plan` skill.
3. **Review** *(optional)* — Review PLAN for gaps, dependencies, breakage risk.
4. **Revise** *(if needed)* — Fold review feedback into PLAN.
5. **Execute** — Run steps via the `execute-plan` skill; populate Executor Notes.
6. **Closeout** — Update monthly LOG; git commit + push.
7. **Post-execution review** — Append hiccup entries to `.claude/skills/_hiccups.md` for any deviation, silent failure, or hard error observed during the run.

### Halt conditions

- **Interrupted runs:** if Ken halts a run mid-execution, his description of what he caught is a hiccup entry — write it to `.claude/skills/_hiccups.md` before resuming.
- **Ambiguity:** if a step is ambiguous, unsafe, or marked [Ken]: halt, set `status: needs-revision`, surface to Ken. Do not improvise.
- **Verification failure:** if any verification check fails, halt before commit; do not tick boxes that have not been verified.
