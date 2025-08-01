Atlas - Claude Implementation Guide

Atlas is an autonomous trading‑strategy research loop.  It uses GPT‑based agents to generate, validate, and store algorithmic ideas for a single asset (BTC‑USD in the MVP).  The pipeline runs nightly, logging everything to a local database and citing snippets from a small knowledge base.  Once the loop is stable it will expand to multiple assets.

---

## 1  Objective

Build a **single‑asset prototype** of the Atlas pipeline that can:
• iterate over strategy ideas for one trading pair (BTC‑USD)
• back‑test them with walk‑forward validation and basic cost modelling
• log results in a small Mongo (or SQLite) database
• pass guard‑rail checks for leverage, data leakage and API safety

Once this loop runs unattended for several weeks, we will extend it to multi‑asset.

---

## 2  Folder conventions

* **`src/`** – Claude’s Python modules (one module per role).
* **`prompts/`** – text prompt templates (planner, analyzer, guard‑rail checklist).
* **`tests/`** – PyTest files Claude will maintain.
* **`README_dev.md`** – living document summarising current architecture (Claude updates when adding modules).
* **`pipeline_runner.py`** – nightly driver script orchestrated by cron.

No other top‑level folders without explicit instruction.

---

## 3  Roles & minimum responsibilities

1. **Planner**
   • Receives parent strategy code and motivation summary.
   • Returns modified strategy plus a short rationale citing at least one research snippet.
   • Must keep syntax valid Python 3.10.

2. **Static Guard‑Rail**
   • Scans planner output for banned libraries, forward‑looking timestamps, leverage >2× or order size > 5 % ADV.
   • On violation: write a short error explanation; do not auto‑fix unless explicitly instructed.

3. **Backtester**
   • Runs walk‑forward (train 3 years, validate 6 months, test 1 month, monthly roll).
   • Models fixed bid‑ask spread and 10 bps slippage.
   • Outputs Sharpe, MaxDD, Turnover, Beta.

4. **Analyzer**
   • Summarises backtest results in plain English.
   • Flags instability if Sharpe in any test window < 0.3.
   • Suggests one improvement idea for next iteration.

5. **Database Layer**
   • Stores strategy code, metrics, and analyzer text.
   • Provides simple `get_top_k` query for planner sampling.

---

## 3A Database Schema & Lineage

**Setup** — Use **SQLite** for the MVP. Create a table `strategies` with columns: id (PK), timestamp, parent\_id, version, code, motivation, metrics (JSON), analysis, status.

Lineage helpers Claude must expose:

* `get_top_k(k)` – returns best K by test Sharpe.
* `get_children(parent_id)` – returns descendants for visualization.

**Iteration logic** — Planner always receives the **top‑k** codes *plus* the analyzer’s “Next Action” field from the parent. The new strategy’s `parent_id` links lineage.

---

## 3B Research Report Template (Analyzer Output)

After each back‑test Analyzer writes a Markdown report saved both to `/reports/{strategy_id}.md` **and** the `analysis` field in DB.

Headings required in every report:

1. **Summary** – one paragraph.
2. **Metrics** – table (train / val / test Sharpe, MaxDD, Turnover).
3. **Stability Check** – note any test folds with Sharpe < 0.3.
4. **Strengths & Weaknesses** – bullet list.
5. **Next Action for Planner** – single bullet that seeds the next iteration.

---

## 3C Cognition Base (Knowledge Database)

Purpose – supply the Planner and Analyzer with quick, citation-ready snippets from finance literature, risk rules, and regulatory texts.

**Minimal MVP design**

1. **Folder layout:** create a top‑level `knowledge/` directory; each source is one Markdown file (e.g., `001_market_structure.md`, `002_margin_rules.md`).  Keep content short, evergreen, and text‑only.
2. **Embedding loader:** a helper script reads every file in `knowledge/`, generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and stores `{id, filepath, embedding}` rows in `knowledge.db` (SQLite).  Run this script at pipeline start‑up or whenever new knowledge files are added.
3. **Retrieval helper:** expose `retrieve_top_n(query, n)` that returns the top‑N Markdown snippets by cosine similarity.
4. **Planner & Analyzer usage:** before composing output, call `retrieve_top_n` with the task prompt and cite at least one snippet in the motivation or analysis text.
5. **Updating knowledge:** when a useful article is clipped into Notion, copy the evergreen portion into `knowledge/`, commit, rerun the embedding loader.

*No external vector DB is required for the MVP; SQLite keeps things portable.*

---

## 4  Guard‑rail checklist (enforced by Static Guard‑Rail)

* Only standard libraries plus pandas, numpy, vectorbt.
* No `datetime.now()` inside strategy logic.
* Position size per trade ≤ 5 % average daily volume.
* Gross leverage ≤ 2× notional.
* No network calls from strategy code.
* File writes restricted to `/tmp`.

On any violation, return `STATUS = failed` and reason.

---

## 5  Implementation order for Claude

1. Create **data loader** that fetches daily candles for BTC‑USD from a free API and caches locally.
2. Build **backtester** using vectorbt with walk‑forward loop.
3. Implement **Static Guard‑Rail** as a Python function that inspects AST.
4. Draft **Planner prompt template** and generate first sample strategy (e.g., moving‑average crossover).
5. Wire **pipeline\_runner** to execute: sample → plan → guard‑rail → backtest → analyze → store.
6. Write **unit tests** covering: guard‑rail catches leverage > 2×, backtester returns Sharpe, database saves entry.
7. Add **README\_dev.md section** documenting new modules after each milestone.

---

## 6  Journal workflow

* After each Claude session, append to **Notion “Progress Journal”**: date, what changed, issues found, next step.
* If a concept explanation was needed, add a one‑sentence takeaway plus link in **Concept Cheats**.

---

## 7  Expansion trigger

When the nightly loop runs for 30 consecutive calendar days with:
• no guard‑rail violations
• average test‑window Sharpe ≥ 0.7
• MaxDD ≤ 15 %
then begin the Multi‑Asset Shared redesign.

---

## 8  Final reminders for Claude

* **Always read the Notion page “Concept Cheats” (https://www.notion.so/Concept-Cheats-2429f3d3bc5680368412e01e48865567?source=copy_link) first.**  Treat it as ground truth definitions and the latest guard‑rail thresholds.  Summarise any newly added bullets in your own words before proceeding.
* Use plain Python 3.10.
* Follow folder conventions strictly.
* Update `README_dev.md` and tests whenever you create or change modules.
* Confirm passing PyTest before closing each task.
* Seek clarification in a comment block if requirements feel ambiguous.
* Claude will **NEVER** do anything outside the scope of their instructions without asking permission first. 
