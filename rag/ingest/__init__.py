"""Offline CI build: sources to candidate stores (architecture band A, spec section 6.1).

Planned modules:
  book.py       book PDF -> chapter/section/paragraph chunks with PDF page anchors
  notes.py      notes PDF -> bullet chunks (voice=analyst, "My view" = opinion) and the corrections table
  notebooks.py  notebooks/*.ipynb -> cells tagged claim / method / result / caveat
  registry.py   VERDICTS + README + summaries -> claims registry and chapter map; fails on disagreement
  tables.py     output/ summaries, tables and figures -> DuckDB with schema cards
  alignment.py  each note bullet vs. book passages (Claude), with exact checks on numbers
  embed.py      chunk vectors with the pinned OPENAI_EMBEDDING_MODEL
  panels.py     phase 3: cleaned thiel_common panels as read-only views

Rules: refuse to build a client corpus while RAG_BOOK_LICENSED is false, and refuse
any `output/` produced in synthetic mode (THIEL_SYNTHETIC=1).
"""
