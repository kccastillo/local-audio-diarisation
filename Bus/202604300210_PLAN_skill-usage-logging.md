---
title: "Set up skill-usage logging via post_tool_use hook"
type: bus-plan
status: ready
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
triggers_plans: ["202604300220_PLAN_skill-usage-audit-skill.md"]
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Add a `post_tool_use` hook to `.claude/settings.json` that fires after every `Skill` tool invocation and appends a JSON line to `.claude/_skill_usage.jsonl`. This produces the telemetry needed by the future skill-usage-audit skill (PLAN 202604300220, currently `blocked` / deferred).

## Context
Logging skill invocations is cheap and one-shot: a small bash script + one hook entry. The *value* of the data scales with skill count and elapsed time — by setting up logging now, the future audit skill has a real history to read instead of starting from zero. The audit skill itself is deferred until either 15+ project skills exist OR a felt misfire signal hits — see PLAN 202604300220.

The Claude Code hook payload schema must be discovered during execution (the executor uses `update-config` skill for the `settings.json` edit; it knows the current event names and payload format). This PLAN specifies *intent and shape* of the log row; the executor wires up the exact JSON.

The log file is gitignored — it grows unboundedly and contains session-local context that does not belong in version history.

## Steps

### Step 1: Create the scripts directory
1. Create directory: `mkdir -p .claude/scripts`
2. Verify: `.claude/scripts/` exists.

### Step 2: Write the logger script
Create `.claude/scripts/log-skill-usage.sh` with this exact content:

```bash
#!/usr/bin/env bash
#
# post_tool_use hook for the Skill tool.
# Reads the hook payload (JSON) from stdin, appends one JSONL row to the log.
#
# Expected payload fields (per Claude Code post_tool_use schema):
#   - tool_name       (string)  — should be "Skill"
#   - tool_input      (object)  — { skill: "...", args: "..." }
#   - tool_response   (object)  — present if hook fires after tool returned
#   - session_id      (string)  — Claude Code session UUID
#
# Output row schema (one JSON object per line):
#   { "ts": "<iso8601>", "skill": "<name>", "args": "<string>",
#     "session_id": "<uuid>", "ok": <bool> }
#
set -euo pipefail

LOG_FILE=".claude/_skill_usage.jsonl"
payload=$(cat)

# Only log Skill tool invocations
tool_name=$(echo "$payload" | jq -r '.tool_name // empty')
if [[ "$tool_name" != "Skill" ]]; then
  exit 0
fi

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
skill=$(echo "$payload" | jq -r '.tool_input.skill // ""')
args=$(echo "$payload" | jq -r '.tool_input.args // ""')
session_id=$(echo "$payload" | jq -r '.session_id // ""')
# Did the call succeed? Default to true if response missing.
ok=$(echo "$payload" | jq -r '
  if .tool_response == null then true
  elif (.tool_response.error // null) != null then false
  else true end
')

mkdir -p "$(dirname "$LOG_FILE")"
jq -nc \
  --arg ts "$ts" \
  --arg skill "$skill" \
  --arg args "$args" \
  --arg session_id "$session_id" \
  --argjson ok "$ok" \
  '{ts:$ts, skill:$skill, args:$args, session_id:$session_id, ok:$ok}' \
  >> "$LOG_FILE"
```

### Step 3: Make the script executable
Run: `chmod +x .claude/scripts/log-skill-usage.sh`

### Step 4: Register the hook in settings.json (via update-config skill)
The exact JSON shape for Claude Code hooks (event-name capitalisation, key names) is owned by the `update-config` skill — that skill is the canonical source for `settings.json` schema. Do NOT hand-author the JSON.

1. Invoke `Skill("update-config")` and pass exactly this verbatim request:

   > Add a hook to project-level `.claude/settings.json` that fires after every invocation of the `Skill` tool. The hook should run the command `bash .claude/scripts/log-skill-usage.sh` and pass the hook payload via stdin. If a `.claude/settings.json` does not exist yet, create one. Do not modify any other hook entries.

2. Follow `update-config`'s returned instructions exactly. If `update-config` asks Haiku to make a configuration choice (e.g. "user-level or project-level?", "merge with existing hook X?"), HALT — set status `needs-revision` and report. Do not improvise.

3. After `update-config` completes, verify:
   - `.claude/settings.json` exists: `[[ -f .claude/settings.json ]]`.
   - File is valid JSON: `jq empty .claude/settings.json` exits 0.
   - File contains the string `"Skill"`: `grep -F '"Skill"' .claude/settings.json` exits 0.
   - File contains the string `log-skill-usage.sh`: `grep -F 'log-skill-usage.sh' .claude/settings.json` exits 0.

### Step 5: Add the log file to .gitignore
Insert one new line `.claude/_skill_usage.jsonl` into `.gitignore` directly after the existing line `.claude/settings.local.json` (which sits inside the `# Claude Code harness` section). The result should look like:

```
# Claude Code harness
.claude/settings.local.json
.claude/_skill_usage.jsonl
Retired/
```

Verify:
- `grep -Fxn '.claude/_skill_usage.jsonl' .gitignore` returns exactly one match.
- `git check-ignore .claude/_skill_usage.jsonl` exits 0 and prints the path (gitignored confirmed).

### Step 6: Smoke-test the logger script
1. Confirm `jq` is installed: `command -v jq` must return a path. If absent, HALT with `needs-revision` (jq is required by the script; install or Sonnet picks an alternative).

2. Test the script in isolation against a synthetic payload:
   ```bash
   echo '{"tool_name":"Skill","tool_input":{"skill":"retire","args":"foo"},"session_id":"test-session"}' | bash .claude/scripts/log-skill-usage.sh
   ```

3. Verify a row exists: `wc -l .claude/_skill_usage.jsonl` must report 1.

4. Verify the row content: `cat .claude/_skill_usage.jsonl | jq -e 'select(.skill == "retire" and .session_id == "test-session" and .ok == true)'` must exit 0.

5. Clean up the synthetic row: `rm .claude/_skill_usage.jsonl` (the file is gitignored; it will be re-created on first real Skill invocation).

### Step 7: Live-test in the next session
The hook only fires in real Claude Code sessions, so cannot be smoke-tested mid-execution. Document this caveat in Executor Notes — Ken verifies on next session start by invoking any skill and checking the log file.

### Step 8: Stage, commit, hold push

1. `git status` — confirm the changes are exactly: `.claude/scripts/log-skill-usage.sh` (new), `.claude/settings.json` (new or modified), `.gitignore` (modified), `Bus/202604300210_PLAN_skill-usage-logging.md` (modified — Executor Notes + status), `Bus/202604010000_LOG_202604.md` (modified — Status Table). `.claude/_skill_usage.jsonl` should NOT appear (gitignored). No other files.
2. Stage all five:
   ```
   git add .claude/scripts/log-skill-usage.sh .claude/settings.json .gitignore Bus/202604300210_PLAN_skill-usage-logging.md Bus/202604010000_LOG_202604.md
   ```
3. Commit with this exact message (HEREDOC):

```
Add skill-usage logging via post_tool_use hook

- .claude/scripts/log-skill-usage.sh appends JSONL rows for every Skill call
- post_tool_use hook in .claude/settings.json wires it up (via update-config)
- Log file .claude/_skill_usage.jsonl is gitignored (grows unbounded)
- Telemetry source for future skill-usage-audit skill (PLAN 202604300220)
- Smoke-tested against synthetic payload; live verification deferred to
  next session start

Co-Authored-By: Claude <noreply@anthropic.com>
```

4. **Hold push** — Ken confirms before pushing.

## Verification
- [ ] `.claude/scripts/log-skill-usage.sh` exists and is executable
- [ ] `.claude/settings.json` exists, is valid JSON, and contains both `"Skill"` and `log-skill-usage.sh`
- [ ] `.gitignore` contains the exact line `.claude/_skill_usage.jsonl`
- [ ] `git check-ignore .claude/_skill_usage.jsonl` returns the path (gitignored)
- [ ] Synthetic payload smoke test produced exactly one valid JSONL row, then was cleaned (file removed)
- [ ] One commit; push held
- [ ] Live verification step recorded in Executor Notes (Ken invokes any skill on next session, confirms `.claude/_skill_usage.jsonl` populated)

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
