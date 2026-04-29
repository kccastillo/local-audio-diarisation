# Write Plan File to Bus

## Process

Step 1: Receive plan content:
  - Plan content (frontmatter values + body sections)
  - Target filename (Sonnet generates this)
  - Any LOG updates needed

Step 2: Check if current month LOG exists.
  Path: Bus/{YYYYMM}010000_LOG_{YYYYMM}.md
  If it does not exist: create it from templates/log-template.md, then perform rollover (Step 2a) before proceeding.

Step 2a: Rollover from prior month (only when creating a new LOG).
  a. Find the most recent prior LOG in Bus/ (largest YYYYMM filename < current month). If none exists, skip rollover.
  b. Open the prior LOG's Status Table. For every row whose Status is NOT one of {done, cancelled, closed}:
     - Add a row to the NEW LOG's "Rollover from Prior Month" table:
       | [plan filename] | [created_month from plan frontmatter] | [new rollover_count] | [one-line reason — e.g. "not started", "in progress", "blocked on X"] |
     - Also add the plan to the NEW LOG's Status Table (same row format as a fresh plan) so active work stays visible in the current month's table.
     - Open the plan file and update its frontmatter:
       * log_month = current YYYYMM  (this field always moves to the active LOG)
       * rollover_count = previous rollover_count + 1
       * created_month = UNCHANGED (immutable — never edit on rollover)
  c. Do NOT move or rename plan files. Plans stay at their original path; only the frontmatter and LOG references update.
  d. Close out the prior LOG: set its frontmatter status to "closed" and last_updated to today. Do not edit its Status Table — it reflects end-of-month state.

Step 3: Write the PLAN file.
  Use templates/plan-template.md as structure.
  Fill all frontmatter fields from Sonnet's content exactly.
  Set created_month = current YYYYMM (this field is immutable — never change it on rollover).
  Set log_month = current YYYYMM (this changes on rollover).

  Special case — recurring PLAN (slug starts with RECUR-):
  - Check if a PLAN file with this RECUR- slug already exists.
  - If it exists: do NOT create a new file. Instead append a new row to the ## History table.
  - If it does not exist: create the file normally, then add a blank ## History table at the bottom.

Step 4: Update the monthly LOG.
  Add a new row to the Status Table:
  | [filename] | [title] | [assigned_to] | [priority] | [status] | [due or —] |

  If this is a recurring task: also add/update a row in the Recurring Task Tracker:
  | [task name] | [RECUR- slug] | [cadence] | [last done or —] | [next due] | [ACTIVE if applicable] |

  After adding or updating any row: reorder the entire Status Table. Rows with Status not in {done, cancelled, closed} appear first, sorted by filename descending; rows with Status in {done, cancelled, closed} follow, sorted by filename descending.

Step 5: Report to Ken:
  ```
  Written: [filename]
  LOG updated: [LOG filename]
  Ready for: [next step — e.g. 'hand to Antigravity for execution']
  ```