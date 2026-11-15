"""Tools the agent calls, each scoped to one released store (spec section 6.3).

  search_book       -> book index          get_chapter      -> chapter map
  search_notes      -> notes index         search_analysis  -> analysis index
  list_claims / get_claim -> claims registry
  query_table / get_figure -> DuckDB + figures (read-only, allow-listed views, row cap)
  run_analysis (phase 3) -> isolated compute over immutable panel views; results are
    session-scoped, never written to the corpus or the registry, and logged to audit
    with recipe, parameters and release ID.

Every tool reads the release pinned for the session, never a CI candidate.
"""
