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
Logging skill invocations is cheap and one-shot: a small Python script + one hook entry. The *value* of the data scales with skill count and elapsed time — by setting up logging now, the future audit skill has a real history to read instead of starting from zero. The audit skill itself is deferred until either 15+ project skills exist OR a felt misfire signal hits — see PLAN 202604300220.

**Revision history:** This PLAN was originally bash + jq. The first execution (commit 8a35ab9, WIP) halted at the smoke-test step because jq was installed via winget but not visible on the bash session's PATH. Decision: rewrite in Python instead — Python is already a hard dependency of this project (it's a Python project), so adding a jq runtime dep for one trivial script was unnecessary tooling drift. This revision swaps `.sh` → `.py` and updates the hook command accordingly. The previous WIP work (`.gitignore` entry, `.claude/scripts/` directory) is preserved.

The hook is wired up by editing `.claude/settings.json` directly. The script's parsing of the hook payload (Step 2) uses Opus's best-guess of Claude Code's payload field names (`tool_name`, `tool_input.skill`, `tool_input.args`, `tool_response`, `session_id`). If the actual payload uses different names, log rows will appear with empty values — that's a calibration signal, not a failure: Sonnet adjusts the script in a follow-up plan once we see what real rows look like.

The log file is gitignored — it grows unboundedly and contains session-local context that does not belong in version history.

## Steps

### Step 1: Confirm the scripts directory exists
1. Run: `[[ -d .claude/scripts ]] && echo OK || echo FAIL`. Expected: `OK` (directory was created in the prior WIP run).
2. If `FAIL`: `mkdir -p .claude/scripts`.

### Step 2: Remove the old bash script

Run:
```
rm -f .claude/scripts/log-skill-usage.sh
```

Verify: `[[ ! -e .claude/scripts/log-skill-usage.sh ]] && echo OK || echo FAIL`. Expected: `OK`.

### Step 2b: Write the Python script

Create `.claude/scripts/log-skill-usage.py` with this exact content (note: every line in the code block below starts at column 0 — do NOT add any indentation when writing the file):

```python
#!/usr/bin/env python3
"""PostToolUse hook for the Skill tool.

Reads the hook payload (JSON) from stdin, appends one JSONL row to the log.

Expected payload fields (per Claude Code PostToolUse schema):
  - tool_name       (string)  — should be "Skill"
  - tool_input      (object)  — { skill: "...", args: "..." }
  - tool_response   (object | null) — present if hook fires after tool returned
  - session_id      (string)  — Claude Code session UUID

Output row schema (one JSON object per line):
  { "ts": "<iso8601>", "skill": "<name>", "args": "<string>",
    "session_id": "<uuid>", "ok": <bool> }
"""
import datetime
import json
import pathlib
import sys

LOG_FILE = pathlib.Path(".claude/_skill_usage.jsonl")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if payload.get("tool_name") != "Skill":
        return 0

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response")

    if tool_response is None:
        ok = True
    elif isinstance(tool_response, dict) and tool_response.get("error") is not None:
        ok = False
    else:
        ok = True

    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "skill": tool_input.get("skill", ""),
        "args": tool_input.get("args", ""),
        "session_id": payload.get("session_id", ""),
        "ok": ok,
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Verify after writing:
- `[[ -f .claude/scripts/log-skill-usage.py ]] && echo OK`
- `python -c "import ast; ast.parse(open('.claude/scripts/log-skill-usage.py').read()); print('OK')"` exits 0 (Python syntax check).
- `head -1 .claude/scripts/log-skill-usage.py` prints exactly `#!/usr/bin/env python3` (no leading whitespace).

### Step 3: Make the Python script executable
Run: `chmod +x .claude/scripts/log-skill-usage.py` (harmless on Windows; helpful on Unix).

### Step 4: Update the hook command in settings.json

`.claude/settings.json` exists from the prior WIP run AND has been hand-edited by Ken to add a `permissions` block. Make a TARGETED edit — do NOT rewrite the whole file.

1. Pre-flight: confirm the file exists and currently points at the .sh script.
   ```
   [[ -f .claude/settings.json ]] && grep -F '"command": "bash .claude/scripts/log-skill-usage.sh"' .claude/settings.json && echo OK_FOUND_OLD || echo HALT
   ```
   - If `OK_FOUND_OLD`: proceed.
   - If `HALT`: set status `needs-revision` and report. The file is in an unexpected state — Sonnet investigates.

2. Use the Edit tool to replace exactly this string in `.claude/settings.json`:
   - old_string: `"command": "bash .claude/scripts/log-skill-usage.sh"`
   - new_string: `"command": "python .claude/scripts/log-skill-usage.py"`

3. Verify:
   - File is valid JSON: `python -c "import json; json.load(open('.claude/settings.json'))"` exits 0.
   - New command present: `grep -F '"command": "python .claude/scripts/log-skill-usage.py"' .claude/settings.json` exits 0.
   - Old command absent: `! grep -F 'log-skill-usage.sh' .claude/settings.json` (i.e., grep exits 1).
   - Structural check: `python -c "import json; c=json.load(open('.claude/settings.json')); assert c['hooks']['PostToolUse'][0]['matcher']=='Skill'; print('OK')"` exits 0.
   - Permissions block preserved: `python -c "import json; c=json.load(open('.claude/settings.json')); assert 'permissions' in c; print('OK')"` exits 0.

### Step 5: Verify .gitignore entry (carryover from prior WIP)
The line `.claude/_skill_usage.jsonl` was added to `.gitignore` in the prior WIP commit and should already be present. Verify:
- `grep -Fxn '.claude/_skill_usage.jsonl' .gitignore` returns exactly one match.
- `git check-ignore .claude/_skill_usage.jsonl` exits 0 and prints the path.

If either fails: HALT (`needs-revision`) — the prior commit's state is unexpected.

### Step 6: Smoke-test the Python logger
1. Confirm `python` is available: `command -v python` must return a path. If absent, HALT with `needs-revision`.

2. Test the script in isolation against a synthetic payload:
   ```
   echo '{"tool_name":"Skill","tool_input":{"skill":"retire","args":"foo"},"session_id":"test-session"}' | python .claude/scripts/log-skill-usage.py
   ```

3. Verify exactly one row was written: `wc -l .claude/_skill_usage.jsonl` must report 1.

4. Verify row content with python:
   ```
   python -c "import json; r=json.loads(open('.claude/_skill_usage.jsonl').readline()); assert r['skill']=='retire' and r['session_id']=='test-session' and r['ok'] is True; print('OK')"
   ```
   Must print `OK` and exit 0.

5. Negative test (non-Skill tool should be ignored):
   ```
   echo '{"tool_name":"Bash","tool_input":{"command":"ls"},"session_id":"test-session"}' | python .claude/scripts/log-skill-usage.py
   ```
   Then verify `wc -l .claude/_skill_usage.jsonl` STILL reports 1 (no new row).

6. Clean up: `rm .claude/_skill_usage.jsonl` (file is gitignored; will be re-created on first real Skill invocation).

### Step 7: Live-test in the next session
The hook only fires in real Claude Code sessions, so cannot be smoke-tested mid-execution. Document this caveat in Executor Notes — Ken verifies on next session start by invoking any skill and checking that `.claude/_skill_usage.jsonl` is created and populated.

### Step 8: Stage, commit, push

1. `git status` — confirm the changes are exactly:
   - `.claude/scripts/log-skill-usage.sh` (deleted)
   - `.claude/scripts/log-skill-usage.py` (new)
   - `.claude/settings.json` (modified — command field only)
   - `Bus/202604300210_PLAN_skill-usage-logging.md` (modified — Executor Notes + status)
   - `Bus/202604010000_LOG_202604.md` (modified — Status Table)
   - `.claude/_skill_usage.jsonl` should NOT appear (gitignored).
   - No other files modified.

2. Stage all five (the deleted `.sh` is staged via its path):
   ```
   git add .claude/scripts/log-skill-usage.sh .claude/scripts/log-skill-usage.py .claude/settings.json Bus/202604300210_PLAN_skill-usage-logging.md Bus/202604010000_LOG_202604.md
   ```

3. Commit with this exact message (HEREDOC):

```
Skill-usage logging: rewrite hook in Python (drop jq dep)

- Replace .claude/scripts/log-skill-usage.sh with .py (no jq required)
- Update .claude/settings.json hook command (bash → python)
- Preserve Ken's permissions block in settings.json
- .gitignore entry for .claude/_skill_usage.jsonl unchanged from prior WIP
- Smoke-tested with synthetic payload (Skill match + non-Skill ignored);
  live verification deferred to next session start
- Closes plan that was halted at 8a35ab9 (jq not on bash PATH)

Co-Authored-By: Claude <noreply@anthropic.com>
```

4. `git push` to origin. Push is NOT held this time — the prior WIP is already on origin and this commit closes it out.

## Verification
- [ ] `.claude/scripts/log-skill-usage.py` exists and is executable
- [ ] `.claude/scripts/log-skill-usage.sh` no longer exists
- [ ] `.claude/settings.json` is valid JSON, contains `"command": "python .claude/scripts/log-skill-usage.py"`, no longer references the .sh script, and Ken's `permissions` block is preserved
- [ ] `.gitignore` contains the exact line `.claude/_skill_usage.jsonl`
- [ ] `git check-ignore .claude/_skill_usage.jsonl` returns the path (gitignored)
- [ ] Synthetic Skill payload produced exactly one valid JSONL row; non-Skill payload added zero rows; log file cleaned afterwards
- [ ] One commit; pushed to origin
- [ ] Live verification step recorded in Executor Notes (Ken invokes any skill on next session, confirms `.claude/_skill_usage.jsonl` populated)

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**

---
*Prior run (commit 8a35ab9): WIP, halted on jq missing from bash PATH. Setup work preserved (gitignore entry, scripts dir, settings.json with permissions block). This re-run replaces .sh→.py and updates the hook command.*
