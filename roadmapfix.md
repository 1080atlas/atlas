# Atlas – Claude Fix Roadmap (Natural‑Language Instructions)

*Purpose – guide Claude to complete the MVP loop based on the repo audit dated 2025‑08‑01.*
*No code blocks – describe tasks in plain English; Claude will generate code & tests.*

---

## 1 Database layer tasks

1. **Add lineage helpers** in `src/database.py`:
     • `get_top_k(k)` – return top‑k strategies ordered by `metrics->test_sharpe DESC` where `status = 'candidate'`.
     • `get_children(parent_id)` – return all rows where `parent_id = :parent_id`.
2. Ensure the `parent_id` column is declared `INTEGER` (typo currently `INTEGE R`).
3. Expose both helpers to the pipeline via `DatabaseManager` class.

## 2 Knowledge base glue

1. In `src/knowledge_base.py`, implement `retrieve_top_n(query, n)`:
     • Embed the query with the same MiniLM model used for snippets.
     • Fetch the `n` nearest snippets from `knowledge.db` using cosine similarity.
     • Return each snippet as `{filepath, text}`.
2. Add a short helper `load_snippet_text(filepath)` that returns raw Markdown text so Planner can cite it.

## 3 Planner enhancements

1. Complete `plan_strategy` in `src/planner.py`:
     • Sample parent: call `DatabaseManager.get_top_k(1)`; pass its code & motivation to the prompt.
     • Before sending prompt to GPT, fetch three knowledge snippets via `retrieve_top_n` and insert under "Relevant knowledge snippets:" section.
     • Include previous Analyzer’s **Next Action** bullet if available.
     • Return structured object `{code, motivation}`.

## 4 Guard‑rail expansion

1. Extend AST checks in `src/guard_rail.py` to flag:
     • Network calls (`urllib`, `requests`, `aiohttp`).
     • Order size > 5 % ADV – detect constant `size` or percentage > 0.05.
2. Return a dataclass or dict: `{passed: bool, errors: [str]}`.

## 5 Backtester improvements

1. Implement **purged walk‑forward**: drop overlapping rows between train and validate/test windows.
2. Add Sharpe threshold check: if any test fold Sharpe < 0.3 set flag `unstable = True` and pass to Analyzer.

## 6 Analyzer completion

1. Write Markdown report with required headings:
     1. Summary
     2. Metrics table (train, val, test)
     3. Stability Check (flag test folds)
     4. Strengths & Weaknesses
     5. **Next Action for Planner** – single bullet suggestion
2. Save the report to `/reports/{strategy_id}.md` and store identical text in the DB `analysis` field.

## 7 Pipeline runner wiring

1. Replace dummy `parent_code = ""` with real call to `get_top_k`.
2. Pass Analyzer’s `unstable` flag back to Planner if applicable.

## 8 Unit tests to add / update

1. **Guard‑rail tests**: strategy with `requests.get()` must fail; strategy leverage = 3× must fail.
2. **Database tests**: inserting two children then calling `get_children(parent_id)` returns both.
3. **Knowledge retrieve test**: query "market structure" returns at least one snippet containing that phrase.

## 9 README\_dev.md updates

After each major module change Claude must append a bullet under *"Current modules"* summarising purpose and key functions.

## 10 Acceptance criteria

* Cron‑run of `pipeline_runner.py` completes sample → plan → guard → backtest → analyze → store without exceptions.
* New strategy row appears in DB with populated metrics and analysis; corresponding `/reports/<id>.md` exists.
* All PyTest tests pass locally.

---

**Reminder to Claude**: Ask for clarification in a comment if any requirement is ambiguous before generating code.
