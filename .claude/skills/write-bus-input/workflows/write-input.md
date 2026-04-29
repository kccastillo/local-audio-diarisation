# Write Input File to Bus

## Process

Step 1: Receive from Sonnet/Ken:
  - Type: RESEARCH | ADVICE
  - Content (full body)
  - Target filename (Sonnet generates per bus naming convention)
  - Target plan filename (if this input feeds/advises a specific plan)
  - Frontmatter values: question_asked, from (for RESEARCH), etc.

Step 2: Confirm monthly LOG exists.
  Path: Bus/{YYYYMM}010000_LOG_{YYYYMM}.md
  If missing: defer to write-bus-plan Step 2 (LOG creation + rollover) before proceeding.

Step 3: Write the input file.
  - RESEARCH → use templates/research-template.md
  - ADVICE → use templates/advice-template.md
  - Fill all frontmatter exactly from Sonnet's content
  - Set `integration_status: pending`
  - For RESEARCH: set `feeds_plan` to the target PLAN filename (or "" if none)
  - For ADVICE: set `advises_plan` to the target PLAN filename (or "" if none)

Step 4: Update the monthly LOG's Context Inputs table.
  Add a row:
  | [input filename] | RESEARCH or ADVICE | [from — e.g. "Haiku", "Opus via Ken"] | [target plan filename or —] | pending |

  Update `last_updated` in the LOG frontmatter to today.

Step 5: Unblock the target PLAN (if applicable).
  If `feeds_plan` / `advises_plan` is set:
    a. Open that PLAN file.
    b. Check if this input resolves the block:
       - PLAN's `status: blocked`, AND
       - the input's topic matches the PLAN's `blocked_by` reason (semantic match — Sonnet set `feeds_plan`/`advises_plan` pointing at this PLAN, so the intent is already confirmed).
    c. If resolved:
       - Flip `status: blocked` → `status: ready`
       - Clear `blocked_by: ""`
       - Add the new input's filename to `linked_inputs` if not already present
       - If `assigned_to: ken` was set because of this block, flip back to `assigned_to: haiku`
    d. If NOT resolved (PLAN blocked on a different input, or status is not blocked):
       - Still add the filename to `linked_inputs`
       - Do not change `status` or `blocked_by`
       - Note this in the report to Ken

Step 6: Report to Ken:
  ```
  Written:     {input filename} ({TYPE})
  LOG:         {LOG filename} → Context Inputs table updated
  Unblocked:   {PLAN filename} → status: ready   (or "no PLAN unblocked")
  Ready for:   {next step — e.g. "Sonnet integrates and re-runs the PLAN"}
  ```