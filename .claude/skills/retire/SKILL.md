---
name: retire
description: Move files to a gitignored Retired/ folder when they are no longer needed. Plan should note invocation.
---

<objective>
Move a file to a gitignored `Retired/` folder, removing it from the active codebase while preserving it locally. Plan execution should log that this skill was invoked.
</objective>

<quick_start>
Invoke with: `Skill("retire", "path/to/file.md")`

Returns: Confirmation that file has been retired.
</quick_start>

**Retirement procedure:** See [workflows/retire-file.md](workflows/retire-file.md)

<success_criteria>
- File no longer exists in original location
- File exists in `Retired/[filename]`
- `Retired/` is in .gitignore
- Confirmation returned to user
- If part of plan execution, plan LOG notes the invocation
</success_criteria>