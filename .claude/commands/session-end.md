We are ending this session. Before stopping, complete all of the following in order:

1. **Update HANDOFF.md** — state only, ~150 line ceiling. Exactly these 5 sections, nothing else:
   - **Current state** — 3 sentences. What's running, the headline number, the one open thread.
   - **Next task** — one explicit task with file paths or exact commands. Not a list.
   - **Open decisions** — table (Decision · Options · Owner · Due). Only OPEN rows; remove DONE rows on commit.
   - **Blockers / waiting-on** — external dependency / person-action / system state. "None." if none.
   - **First task for next session** — one sentence, specific and actionable.

   **Rule:** *If you're writing narrative in HANDOFF, it belongs in CHANGELOG.* HANDOFF is state-only; never re-narrate what happened this session, never copy Files-changed lists, never keep "Session N summary" sub-sections. Use the active plan doc for planning detail. The HANDOFF/CHANGELOG content boundary is mandatory — drift back to a fat HANDOFF is a regression.

2. **Update CHANGELOG.md** — add entries for everything meaningful that changed

3. **Sync the docs** (only what changed this session):
   - **Regenerate the leaderboard** if any technique or the eval harness changed:
     `python scripts/run_eval.py` (writes results/leaderboard.{md,csv}). Keep it
     regenerable in one command — never hand-edit the leaderboard.
   - **docs/TRADEOFFS.md** — update the technique × {quality, speed, interpretability,
     cost, cold-start, complexity} matrix if a technique was added or changed.
   - **docs/RESULTS.md / docs/METHODOLOGY.md** — update if findings or eval method changed.
   - **README.md** — update if the run commands or project scope changed.
   - These are judgment edits — never bulldoze hand-written content; surface conflicts instead.

4. **Run pytest** — report results. If tests are failing, note them explicitly. Do not paper over failures.

5. **Write any pending ADRs** — if a non-obvious architectural decision was made this session and no ADR exists, write it now. Update docs/adr/README.md index.

6. **Commit** with a conventional commit message:
   Format: {type}({scope}): {description}
   Stage specific files — not git add -A

7. **Tell me the first task for next session** — one sentence, specific, actionable.

Do not skip any of these steps. If something can't be completed, note why in HANDOFF.md.
