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
