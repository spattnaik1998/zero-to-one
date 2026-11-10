# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Testing Zero to One": a claim-by-claim empirical test of Peter Thiel's book against public data from before and after its publication (`BOOK_DATE = 2014-09-16`). One Jupyter notebook per hypothesis (`01_H1_…` to `08_H8_…`), a shared helper module (`thiel_common.py`), and a PDF report builder (`findings_report.py`).

Note: `../../AGENTS.md` (in `Downloads/`) describes an unrelated CrewAI project and does not apply here.

## Commands

Windows; the virtualenv is `.venv/` (`.venv/Scripts/python.exe`, `.venv/Scripts/jupyter.exe`).

```bash
pip install -r requirements.txt
# Run a notebook end to end (equivalent to Kernel → Restart & Run All)
jupyter nbconvert --to notebook --execute --inplace 02_H2_competition_profits.ipynb
# Run on generated data, no network or API keys (testing only)
THIEL_SYNTHETIC=1 jupyter nbconvert --to notebook --execute --inplace 02_H2_competition_profits.ipynb
# Rebuild output/Zero_to_One_Findings.pdf from whatever summaries exist
python findings_report.py
```

There is no test suite or linter. The "check" is the last cell of each notebook, which prints verdict inputs and writes `output/hN_summary.json`.

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

## Conventions

- New notebooks follow the existing shape: a markdown cell quoting the claim and chapter, setup (`import thiel_common as tc; tc.style()`), analysis cells, and a final "Verdict inputs and check cell" that writes the summary JSON.
- Figure and table names are prefixed `hN_` (`h2_fig3_airlines_bigtech`, `h2a_cross_section_slopes`).
- `README.md` holds the hypothesis/data table. Keep it in sync when adding a notebook (H8 is currently missing from it).
