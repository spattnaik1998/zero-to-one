# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Testing Zero to One": a claim-by-claim empirical test of Peter Thiel's book against public data from before and after its publication (`BOOK_DATE = 2014-09-16`). One Jupyter notebook per hypothesis (`notebooks/01_H1_…` to `08_H8_…`), a shared helper module (`thiel_common.py`), and a PDF report builder (`findings_report.py`).

On top of it, "Ask Zero to One" (`rag/`) is an agentic RAG assistant being built for a client over the analyst's notes and this analysis, with the data the notebooks pulled through the API keys (SEC, Census, FRED/BLS, OEWS) as the primary evidence. Its design is `docs/agentic_rag_spec.md` and its reference architecture `docs/architecture_v3.png`. The spec's decision log records what has been settled; read it before changing the design. Everything in `docs/` except `docs/README.md` is private engagement material: it is git-ignored and must never be committed or quoted in tracked files, because the GitHub repo is public. Don't name the client in tracked files.

Note: `../../AGENTS.md` (in `Downloads/`) describes an unrelated CrewAI project and does not apply here.

## Commands

Windows; the virtualenv is `.venv/` (`.venv/Scripts/python.exe`, `.venv/Scripts/jupyter.exe`).

```bash
pip install -r requirements.txt
# Run a notebook end to end (equivalent to Kernel → Restart & Run All)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_H2_competition_profits.ipynb
# Run on generated data, no network or API keys (testing only)
THIEL_SYNTHETIC=1 jupyter nbconvert --to notebook --execute --inplace notebooks/02_H2_competition_profits.ipynb
# Rebuild output/Zero_to_One_Findings.pdf from whatever summaries exist
python findings_report.py
```

The notebooks have no test suite; their "check" is the last cell, which prints verdict inputs and writes `output/hN_summary.json`.

For `rag/`:

```bash
pip install -r requirements-rag.txt
pytest -q                       # tests in tests/; CI runs these on every PR
python -m rag doctor            # keys, models, sources, analysis outputs; --offline downgrades missing keys to warnings
python -m rag manifest          # build/manifest.json: source hashes, git commit, models
python -m rag ingest [notes|notebooks|registry|tables|index|all]   # build/candidates/; all runs every step; stops if the registry sources disagree or the DuckDB copy differs from output/
                                # `index` calls OpenAI (embeddings are cached in build/cache/, so rebuilds of unchanged text are free)
python -m rag release          # gate the candidates, then publish build/releases/<id>/ and point CURRENT at it
                                # --check (gate only), --list, --pin <id> (rollback), --allow-dirty (uncommitted inputs, local trials)
```

Parser tests run on a synthetic notes PDF built in `tests/conftest.py` with the same typography as the real notes, and on synthetic notebooks and repos built inside the tests. `tests/test_real_sources.py` checks the real notes, and `tests/test_registry.py::test_the_real_repository_is_consistent` checks the real notebooks, README and `findings_report.py`; both run in CI because those files are committed.

Everything under `output/` (figures, tables, summary JSONs, the PDF) is committed. A notebook run, including a `THIEL_SYNTHETIC=1` run, overwrites those tracked files in place, because synthetic mode writes to the same paths. After a synthetic test, restore them with `git checkout -- output/` and don't commit them.

When printing notebook contents from a shell, set `PYTHONIOENCODING=utf8`: the notebooks contain characters (−, —) that crash the cp1252 console.

## Architecture

**Data flow:** notebook → `thiel_common` fetchers (cached to `data/raw/`) → analysis → `tc.save_fig` (`output/figures/*.png` + `.pdf`), `tc.save_table` (`output/tables/*.csv`), and a final `output/hN_summary.json` → `findings_report.py` reads only the summary JSONs and figure PNGs.

**`thiel_common.py`** (imported as `tc`) holds everything shared:
- Keys come from the nearest `.env` walking up from the repo (in practice `Recent/.env`): `SEC_USER_AGENT`, `CENSUS_API_KEY`, `FRED_API_KEY`. `_need()` raises with a clear message if one is missing.
- Fetchers: SEC XBRL frames (`sec_frame`, `public_float_panel`, `annual_financials`, `financial_panel`, `cash_flows`), SEC financial-statement SIC codes, Census, FRED/BLS, Yahoo/Stooq prices, YC directory, OEWS/O*NET/AIOE occupation data. Every download is cached once in `data/raw/`; delete the file to refresh. `_download` detects HTML error pages and prints a manual-download hint instead of failing.
- SEC data cleaning matters: `check_against_assets` screens public float against total assets, `fix_scale_errors` corrects unit-scale errors (log kept in `tc.SCALE_FIXES`), and `link_successors`/`SUCCESSOR` merge renamed or acquired CIKs.
- Every fetcher has a `_synthetic_*` counterpart used when `THIEL_SYNTHETIC=1`.
- Shared stats: `tail_fit`/`compare_tails` (power-law via the `powerlaw` package), `top_share`, `ccdf`, `fe_ols` (fixed-effects OLS with clustered SEs), `wls`.
- Plot style: `tc.style()`, fixed palette `tc.C` plus `GREY`/`INK`/`MUTED`. Use these colors in new figures.

**`findings_report.py`** (reportlab): one `section_hN(summary)` function per hypothesis, registered in `SECTIONS`; verdict text is hardcoded in `VERDICTS`. A hypothesis shows "Pending" until its JSON exists, and "Results in, write-up pending" until it also has a `SECTIONS` entry. Adding a hypothesis therefore means: a new notebook that writes `output/hN_summary.json`, a `section_hN` function, and entries in `SECTIONS`, `VERDICTS` and the scorecard list in `build()`. If a notebook changes its summary keys, update the matching `section_hN` too.

**Inputs that cannot be downloaded** live in `reference/` (Ritter IPO tables, delisted prices). `data/*.tgz` are archived snapshots of raw data and outputs.

**`rag/`** reads settings through `rag/config.py` from the same nearest `.env` as `thiel_common`. Keep every key (SEC, Census, FRED, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) in that one file: a closer second `.env` would hide the notebooks' keys. Template: `.env.example`. Claude handles reasoning (`CLAUDE_MODEL`, default `claude-opus-5-5`; `CLAUDE_PROVIDER` selects the platform). OpenAI handles embeddings only, with the model pinned in `OPENAI_EMBEDDING_MODEL`. Changing that model means re-embedding every index.

**Releases.** Nothing serves from `build/candidates/`. `rag/release.py` gates a build: summaries must be live, not synthetic (each carries a `_provenance` stamp from `tc.save_summary`, or, for summaries that predate the stamp, matches the hash in `output/provenance.json`, and no output may contain the synthetic generators' `FIRM <n>` names); release inputs must be committed; the candidates must match a fresh build from today's sources; the DuckDB copy must match `output/`; the indexes must match the candidates and the pinned embedding model; every retrieval smoke question must be answered in the top 5. A release is immutable (read-only files, content-derived ID, re-hashed on open). Notebooks must write their summary with `tc.save_summary(summary, "hN")`, never by hand, or the gate cannot tell a live run from a synthetic one.

**Tools.** `rag/tools/toolset.py`: `Toolset(open_release(settings), embedder)` binds the agent's tools to one verified release; `call(name, input)` validates arguments in code (strict tool schemas cannot express ranges) and returns JSON with a `source_id` and `voice` on every result. `query_table` accepts exactly one SELECT on a read-only DuckDB connection with external access disabled and the configuration locked, a row cap and a time limit; keep all four guards. Verdicts and one-line findings appear only for `final` claims. Return paths inside the release (`figures/<name>.png`), never local filesystem paths.

**The book is out of the corpus and out of the code** (scope decision): the assistant works from the notes and the analysis, and Thiel's words appear only as the notebooks' claim-cell quotations. **`sources/book/`** stays git-ignored so that a local copy of the book is never committed: the GitHub repo is public and the book is copyrighted. **`sources/notes/`** holds the analyst's reading notes, which the analyst chose to publish.

## Conventions

- New notebooks go in `notebooks/` and follow the existing shape: a markdown cell quoting the claim and chapter, setup (the two `sys.path` lines that find the repo root, then `import thiel_common as tc; tc.style()`), analysis cells, and a final "Verdict inputs and check cell" that writes the summary with `tc.save_summary(summary, "hN")`.
- Figure and table names are prefixed `hN_` (`h2_fig3_airlines_bigtech`, `h2a_cross_section_slopes`).
- Verdicts appear in three places that must agree: `VERDICTS` in `findings_report.py`, the "Findings" table in `README.md`, and the scorecard claim text in `build()`. Chapters must also agree between each notebook's heading, the README row and the scorecard. `rag/ingest/registry.py` checks both, and `tests/test_registry.py::test_the_real_repository_is_consistent` fails CI on any mismatch. `README.md` also has the notebook/data table. Update all of them when a verdict changes or a notebook is added.
- Identify figures by file name (`h1_fig1_ccdf`), never by figure number: the report and the notebooks number figures differently (the report's "Figure H1-1" is `h1_fig3_wealth_creation`, the notebook's is `h1_fig1_ccdf`).
- `H1_PROVISIONAL` at the top of `findings_report.py` switches the H1 section and scorecard to provisional wording. The RAG claims registry must carry the same status, and the answer verifier must give no verdict for a provisional or untested claim.
- Each `rag/` subpackage's `__init__.py` lists its planned modules and the spec section they implement. Keep those lists current as modules land.
