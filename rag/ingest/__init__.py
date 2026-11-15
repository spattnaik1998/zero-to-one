"""Offline CI build: sources to candidate stores (architecture band A, spec section 6.1).

Built (`python -m rag ingest` writes build/candidates/*.jsonl):
  pdf.py        word -> line extraction by baseline (handles drop caps, small caps, font splits)
  chapters.py   the book's 16 parts (preface 0, chapters 1-14, conclusion 15), shared by both parsers
  book.py       book PDF -> paragraphs -> chunks with chapter, section, subsection and PDF pages;
                check_structure() fails unless all 16 parts are found in order
  notes.py      notes PDF -> bullets by chapter (voice=analyst, "My view" = opinion) and the
                corrections table, rows split on the table's rules
  chunking.py   section-bounded chunks, <= 375 words (~500 tokens), 15% sentence overlap
  run.py        runs the parsers and writes the candidate files

Planned:
  notebooks.py  notebooks/*.ipynb -> cells tagged claim / method / result / caveat
  registry.py   VERDICTS + README + summaries -> claims registry and chapter map; fails on disagreement
  tables.py     output/ summaries, tables and figures -> DuckDB with schema cards
  alignment.py  each note bullet vs. book passages (Claude), with exact checks on numbers
  embed.py      chunk vectors with the pinned OPENAI_EMBEDDING_MODEL
  panels.py     phase 3: cleaned thiel_common panels as read-only views

Rules: candidates are for development until the release gate passes; publishing a client
corpus must refuse while RAG_BOOK_LICENSED is false, and must refuse any `output/`
produced in synthetic mode (THIEL_SYNTHETIC=1).
"""
