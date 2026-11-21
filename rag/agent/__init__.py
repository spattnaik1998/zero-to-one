"""Online serving (architecture band C, spec sections 6.4-6.5).

Built:
  orchestrator.py  Session: Claude (Opus 5.5) tool loop pinned to one release; tool results go
                   back as search-result blocks so every result is citable by its source_id;
                   turn and tool-call budgets end in a tools-off final answer; returns an Answer
                   (text, citations, tool calls, stop reason, usage). open_session(settings)
                   builds one; `python -m rag ask "..."` runs one question.

Planned:
  verifier.py      answer contract: numbers match tool outputs (allowing for the summaries'
                   rounding), verdicts come from the registry, voice labels, a citation per
                   paragraph, no verdict for untested/provisional claims, no investment advice.
                   FAIL -> revise once, then flag.
  audit.py         append-only log: question, tool calls and outputs, cited source ids,
                   model, release ID, verification outcome
"""
