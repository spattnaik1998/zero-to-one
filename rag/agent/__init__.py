"""Online serving (architecture band C, spec sections 6.4-6.5).

Built:
  orchestrator.py  Session: Claude (Opus 5.5) tool loop pinned to one release; tool results go
                   back as search-result blocks so every result is citable by its source_id;
                   turn and tool-call budgets end in a tools-off final answer; returns an Answer
                   (text, per-segment citations, tool calls, stop reason, usage).
                   open_session(settings) builds one.
  verifier.py      the answer contract, checked in code: numbers supported by tool results
                   (rounding, percent and scale aware), verdicts only from the registry and only
                   for final claims, untested chapters called untested, citations on every
                   paragraph that states a fact, the analyst's and Thiel's voices labelled, no
                   investment advice. ask_checked(session, q): verify, revise once, then flag.
                   `python -m rag ask "..."` uses it.
  audit.py         append-only, hash-chained log of every question: each attempt with its tool
                   calls and their results, citations, both verifications, release id and
                   digest, model, tokens, timing, session and user; errors too. Cross-process
                   lock; verify() detects edited, deleted, reordered or truncated records.
                   ask_audited(session, q, log): an answer that cannot be logged is not returned.
"""
