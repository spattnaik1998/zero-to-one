"""Offline CI build: sources to candidate stores (architecture band A, spec section 6.1).

Scope: the corpus is the analyst's notes and the H1-H8 analysis; the primary evidence is
the data the notebooks pulled through the API keys. The book is not part of the corpus.

Built (`python -m rag ingest` writes build/candidates/):
  notes.py      notes PDF -> bullets by chapter (voice=analyst, "My view" = opinion) and the
                corrections table, rows split on the table's rules
  notebooks.py  notebooks/*.ipynb -> chunks tagged claim / method / result / caveat; the claim
                cell's blockquote is Thiel's own words (voice=thiel), everything else voice=evidence
  registry.py   claims registry + chapter map from the notebooks, findings_report.py, README
                and output/; returns problems when they disagree, and ingest then stops
  chapters.py   the book's 16 parts (preface 0, chapters 1-14, conclusion 15)
  chunking.py   section-bounded chunks, <= 375 words (~500 tokens), 15% sentence overlap
  pdf.py        word -> line extraction by baseline
  tables.py     output/ -> DuckDB: the 46 results tables as-is, every summary value addressed by
                path (summary_values), schema cards and figure captions; verify_database()
                compares every cell and value with the source files
  embed.py      OpenAI embeddings with the pinned model, cached on disk by (model, text)
  index.py      three search indexes (notes, analysis, catalog) in indexes.duckdb; each
                document keeps its cited text and a context-prefixed text for search
  run.py        runs the steps and writes the candidate files

Planned:
  panels.py     phase 3: cleaned thiel_common panels as read-only views

Rules: candidates are for development; only `python -m rag release` (rag/release.py) turns a
passing build into a release, and it refuses any `output/` produced in synthetic mode.
"""
