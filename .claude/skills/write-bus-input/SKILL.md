---
name: write-bus-input
description: Bus input transcription skill. Writes RESEARCH (data drops) and ADVICE (strategic notes, e.g. Opus output pasted by Ken) files into Bus/, updates the monthly LOG's Context Inputs table, and clears any PLAN's blocked state that was waiting on this input. Trigger phrases: "write this research to bus", "create research file", "write this advice", "paste advice", "record this input".
---

<essential_principles>
Transcribe input content accurately. Do not interpret, summarise, or modify content.
RESEARCH and ADVICE are the same shape: context inputs that unblock PLANs. One skill handles both; the Type field distinguishes them.
Always check the input's `feeds_plan` (RESEARCH) or `advises_plan` (ADVICE) frontmatter — if that PLAN is `blocked` waiting on this input, clear the blocked state.
Always update the monthly LOG's Context Inputs table after writing an input file.
Report back: filename written, LOG updated, PLAN(s) unblocked if any.
</essential_principles>

<preconditions>
Before writing, confirm:
- Type is RESEARCH or ADVICE (exact uppercase)
- Content, target filename, and target plan (if any) have been provided
- Current month LOG exists (if not, create it via write-bus-plan's Step 2 first)
</preconditions>

**Input writing procedure:** See [workflows/write-input.md](workflows/write-input.md)

<constraints>
- Never modify input content — transcribe exactly as provided
- Never flip a PLAN's status unless the input clearly resolves its blocked_by reason — if in doubt, leave status unchanged and flag to Ken
- Never clear `blocked_by` without also flipping `status: blocked` → `ready` in the same edit
- Never write to Wiki/ — that is the Wiki skills' domain
- An input file's `integration_status` stays `pending` until it is integrated into a PLAN; this skill does not mark integration complete
</constraints>

<success_criteria>
- Input file exists at the correct path with valid frontmatter
- Monthly LOG Context Inputs table has the new row
- If the input resolved a PLAN block: that PLAN's `status` is `ready`, `blocked_by` is empty, and the input filename appears in its `linked_inputs`
- Ken has been given the report including which PLAN (if any) was unblocked
</success_criteria>