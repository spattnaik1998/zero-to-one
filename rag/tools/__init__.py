"""Tools the agent calls, each reading one store of a pinned, verified release (spec section 6.3).

Built (toolset.py; `Toolset(release, embedder).call(name, input)`; `tool_definitions()`):
  search_notes      -> notes index         get_chapter      -> chapter map (notes + claims)
  search_analysis   -> analysis index (notebook chunks; Thiel's quotations carry voice=thiel)
  find_data         -> catalog index (which table or figure holds the data)
  list_claims / get_claim -> claims registry (the only source of verdicts; final claims only)
  describe_table / query_table -> DuckDB results database (one read-only SELECT, no file or
                       network access, locked configuration, row cap, time limit)
  get_figure        -> figure caption and its file inside the release

Planned (phase 3): run_analysis -> isolated compute over immutable panel views; results are
session-scoped, never written to the corpus or the registry, and logged to audit.

No search_book: the book is out of scope. Every tool reads the release pinned for the
session (rag.release.open_release), never a CI candidate.
"""
from .toolset import TOOL_DEFINITIONS, ToolError, Toolset, tool_definitions

__all__ = ["TOOL_DEFINITIONS", "ToolError", "Toolset", "tool_definitions"]
