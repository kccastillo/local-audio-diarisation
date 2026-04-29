---
name: retire
description: Move files to a gitignored Retired/ folder when they are no longer needed, redundant, or superseded. Use proactively whenever an artefact has served its purpose — audit docs, temp research, replaced configs, completed working files. Always commits and pushes after moving.
---

<objective>
Move a file to a gitignored `Retired/` folder, removing it from the active codebase while preserving it locally. Commit and push the removal so the source of truth reflects the current state. Plan execution should log that this skill was invoked.
</objective>

<quick_start>
Invoke with: `Skill("retire", "path/to/file.md")`

Returns: Confirmation that file has been retired, committed, and pushed.
</quick_start>

**Retirement procedure:** See [workflows/retire-file.md](workflows/retire-file.md)

<success_criteria>
- File no longer exists in original location
- File exists in `Retired/[filename]`
- `Retired/` is in .gitignore
- Git commit created and pushed to origin
- Confirmation returned to user
- If part of plan execution, plan LOG notes the invocation
</success_criteria>