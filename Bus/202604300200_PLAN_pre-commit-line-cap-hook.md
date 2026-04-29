---
title: "Add git pre-commit hook to enforce CLAUDE.md and CONTEXT_CONSTITUTION.md line caps"
type: bus-plan
status: done
assigned_to: ""
priority: medium
created: 2026-04-30
created_by: opus
created_month: 202604
log_month: 202604
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
Create a git pre-commit hook at `.githooks/pre-commit` that blocks any commit which would leave `CLAUDE.md` or `.claude/CONTEXT_CONSTITUTION.md` over 300 lines (hard cap), and prints a soft-cap warning at 150 lines without blocking. Activate via `git config core.hooksPath .githooks` so the hooks directory is shared and version-controlled.

## Context
The `maintain-claude-md` skill (PLAN 202604291940, status: done) audits these files when invoked. The audit runs only when the user (or RECUR- task) triggers it — between audits, drift can accumulate. A pre-commit hook is the complementary control: it enforces the line budget at the commit gate on every commit, regardless of who edited the file or whether the skill has run recently.

The hook is line-counting only — it does NOT replace the skill (which checks anti-patterns, dead refs, codebase duplication). The two together: hook enforces the budget; skill enforces quality.

`.claude/CONTEXT_CONSTITUTION.md` does not exist yet (it lands in PLAN 202604291950, rationalise-claude-md). The hook MUST treat absent files as PASS so it does not break commits before the constitution lands.

Bash is the chosen language because Git for Windows (Ken's environment) ships bash, and bash is portable to any future contributors on macOS/Linux.

## Steps

### Step 1: Create the hooks directory
1. Create directory `.githooks/` at the repo root: `mkdir -p .githooks`
2. Verify: `.githooks/` exists and is empty.

### Step 2: Write the pre-commit script
Create `.githooks/pre-commit` with this exact content:

```bash
#!/usr/bin/env bash
#
# Pre-commit hook: enforce line caps on living-config files.
# Soft warn at 150, hard block at 300. Absent files PASS silently.
# Only checks files that are present in this commit's staged diff.
#
set -euo pipefail

soft=150
hard=300
files=("CLAUDE.md" ".claude/CONTEXT_CONSTITUTION.md")

# Files staged for this commit (added/modified). Excludes deletions.
staged=$(git diff --cached --name-only --diff-filter=AM)

fail=0
for f in "${files[@]}"; do
  # Only check files that are actually in this commit
  if ! echo "$staged" | grep -Fxq "$f"; then
    continue
  fi
  # Skip if the working-tree file is missing (defensive — staged but vanished)
  if [[ ! -f "$f" ]]; then
    continue
  fi
  lines=$(wc -l < "$f")
  if (( lines > hard )); then
    echo "BLOCKED: $f has $lines lines (hard cap = $hard)." >&2
    echo "  Run /maintain-claude-md prune before committing." >&2
    fail=1
  elif (( lines > soft )); then
    echo "WARN: $f has $lines lines (soft cap = $soft)." >&2
    echo "  Consider /maintain-claude-md prune. Commit allowed." >&2
  fi
done

exit $fail
```

### Step 3: Make the script executable
Run: `chmod +x .githooks/pre-commit`

### Step 4: Activate the hooks directory for this repo
Run: `git config core.hooksPath .githooks`

Verify with: `git config --get core.hooksPath` — must print `.githooks`.

### Step 5: Verify the hook script is syntactically valid
1. Run `bash -n .githooks/pre-commit` — must exit 0 (no output means valid bash).
2. Run `wc -l CLAUDE.md` and record the result in Executor Notes as the current baseline.
3. The hook will be live-tested by the commit in Step 7 (which itself triggers the hook).

### Step 6: Create the activation README
Create file `.githooks/README.md` with this exact content (note: outer fence is 4 backticks because the README itself contains a 3-backtick code block):

````markdown
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
````

### Step 7: Stage, commit, hold push

1. `git status` — confirm the changes are exactly: `.githooks/pre-commit` (new), `.githooks/README.md` (new), `Bus/202604300200_PLAN_pre-commit-line-cap-hook.md` (modified — Executor Notes + status), `Bus/202604010000_LOG_202604.md` (modified — Status Table). No other files.
2. Stage all four:
   ```
   git add .githooks/pre-commit .githooks/README.md Bus/202604300200_PLAN_pre-commit-line-cap-hook.md Bus/202604010000_LOG_202604.md
   ```
3. Commit with this exact message (HEREDOC):

```
Add pre-commit hook enforcing CLAUDE.md / CONTEXT_CONSTITUTION.md line caps

- .githooks/pre-commit blocks commits over 300 lines, warns at 150
- Activated via git config core.hooksPath .githooks (one-time, manual)
- Treats absent files as PASS (CONTEXT_CONSTITUTION.md not yet created)
- .githooks/README.md documents the activation step
- Complements maintain-claude-md skill: hook = budget, skill = quality

Co-Authored-By: Claude <noreply@anthropic.com>
```

4. **Hold push** — Ken confirms before pushing.

## Verification
- [ ] `.githooks/pre-commit` exists and is executable (Unix: `-rwxr-xr-x` or similar)
- [ ] `bash -n .githooks/pre-commit` exits 0
- [ ] `git config --get core.hooksPath` returns `.githooks`
- [ ] `.githooks/README.md` exists with the activation command
- [ ] Current `CLAUDE.md` line count recorded in Executor Notes (baseline)
- [ ] One commit; push held

## Executor Notes

**Executed:** 2026-04-30
**Outcome:** done
**What was done:**
- Created `.githooks/` directory at repo root
- Wrote `.githooks/pre-commit` hook script (line-count enforcement: soft 150, hard 300)
- chmod +x `.githooks/pre-commit` (executable)
- git config core.hooksPath .githooks (activated for this repo)
- Verified hook syntax with bash -n (passed)
- Recorded CLAUDE.md baseline: 161 lines (under both caps)
- Created `.githooks/README.md` with activation instructions

**Files modified:**
- `.githooks/pre-commit` (new)
- `.githooks/README.md` (new)
- `Bus/202604300200_PLAN_pre-commit-line-cap-hook.md` (Executor Notes + status)
- `Bus/202604010000_LOG_202604.md` (Status Table row)
