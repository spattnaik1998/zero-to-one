"""Online serving (architecture band C, spec sections 6.4-6.5).

Planned modules:
  orchestrator.py  Claude tool loop (Messages API + tool use), session pinned to a release ID
  verifier.py      answer contract: numbers match tool outputs, verdicts come from the registry,
                   voice labels, a citation per paragraph, no verdict for untested/provisional
                   claims, no investment advice. FAIL -> revise or flag.
  audit.py         append-only log: question, tool calls and outputs, chunk IDs,
                   model, release ID, verification outcome
"""
