# Atlas – Claude Audit Tasks (2025‑08‑02)

*This markdown supersedes the earlier **Claude Fix Roadmap** and lists the follow‑up tasks discovered in the 1 Aug 2025 repo audit.*


---

## A. Priority P1 – fix immediately

1. **`src/planner.py`** – complete `plan_strategy` so it returns `{code, motivation}`; remove file truncation.
2. **`src/analyzer.py`** – repair truncated f‑string; ensure report always includes headings 3 (Stability), 4 (Strengths), 5 (Next Action).
3. **`src/guard_rail.py`** –
   • Add ADV ≤ 5 % check.
   • Handle `node.module is None` in import checker.
4. **`src/backtester.py`** – compute `unstable = True` if any test fold Sharpe < 0.3; return flag.
5. **`src/pipeline_runner.py`** –
   • Replace dummy parent sampling with `db.get_top_k(1)`.
   • Pass `unstable` flag into Analyzer.
6. **`src/database.py`** –
   • Use `json_extract(metrics,'$.test_sharpe')` in `get_top_k`.
   • `json.dumps(metrics)` before insert.
   • Add `UNIQUE(version)` to prevent duplicates.

## B. Priority P2 – quality & robustness

1. **`src/knowledge_base.py`** – move embedding build into `KnowledgeBase.build()`; import should be fast.
2. **Lineage helpers tests** – add PyTest that inserting children then calling `get_children` returns correct list.
3. **Guard‑rail tests** – ensure banned import/network call and leverage >2× cause failure.
4. **Knowledge retrieval test** – query "market structure" returns at least one snippet from `knowledge/`.
5. **README\_dev.md** – Claude must append a brief bullet to *Current modules* after each completed P1 task.

## C. Acceptance criteria

* A single run of `pipeline_runner.py` completes the full loop and inserts a new row into the DB with populated `metrics` and `analysis` plus `/reports/<id>.md` present.
* All PyTest tests (including new ones) pass.
* Guard‑rail rejects a test strategy using `requests.get()`.

---

**Reminder to Claude**: Confirm uncertainties in a comment before generating code.  Follow folder conventions strictly, and update docs/tests alongside code changes.
