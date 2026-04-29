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