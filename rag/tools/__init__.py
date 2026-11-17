"""Tools the agent calls, each scoped to one released store (spec section 6.3).

  search_notes      -> notes index         get_chapter      -> chapter map (notes + hypotheses)
  search_analysis   -> analysis index (notebook chunks; Thiel's quotations carry voice=thiel)
  list_claims / get_claim -> claims registry (the only source of verdicts)
  query_table / get_figure -> DuckDB + figures (read-only, allow-listed views, row cap)
  run_analysis (phase 3) -> isolated compute over immutable panel views; results are
    session-scoped, never written to the corpus or the registry, and logged to audit
    with recipe, parameters and release ID.

No search_book: the book is out of scope. Every tool reads the release pinned for the
session, never a CI candidate.
"""
