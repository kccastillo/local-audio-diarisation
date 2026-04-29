# Git hooks

This directory holds version-controlled git hooks for the Diarizer repo.

## One-time activation (per clone)

```
git config core.hooksPath .githooks
```

After running this, git will use the hooks here instead of `.git/hooks/`.

## Hooks

- **pre-commit** — enforces line caps on `CLAUDE.md` and `.claude/CONTEXT_CONSTITUTION.md`. Soft warn at 150 lines, hard block at 300 lines. Treats absent files as PASS. Only checks files staged in the current commit.

If the hook blocks a legitimate commit, run `/maintain-claude-md prune` to produce a removal PLAN, execute it, then re-commit.
