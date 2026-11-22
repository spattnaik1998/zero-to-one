"""Ask Zero to One: agentic RAG over the analyst's notes and the H1-H8 analysis.

Design: docs/agentic_rag_spec.md. Reference architecture: docs/architecture_v3.png.

Package layout follows the three bands of the architecture:
  rag.ingest  offline CI build: parsers, alignment, candidate stores (band A)
  rag.tools   read-only tools over a released, version-pinned corpus (bands B/C)
  rag.agent   orchestrator, answer verifier, audit log (band C)
  rag.serve   the web chat and its API (band C)
  rag.eval    golden set, B0 baseline, authorization checks and the evaluation gate for promotion
Shared: rag.config (settings), rag.release (gate, publish, promote), rag.retrieval (hybrid search),
rag.units (column units from name suffixes), rag.sources (source files and manifest).
"""
