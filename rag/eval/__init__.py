"""Evaluation and the release gate (spec section 10).

Built: golden.py, the golden set: 140 questions (verdict 40, numeric 40, chapter overview 15
including the untested chapters 0, 8, 9, 10 and 14, method 20, cross-chapter 15, out of scope or
adversarial 10). Expected numbers are SQL queries against the release under test, so they come
from the data the notebooks pulled (SEC, Census, FRED/BLS, OEWS) and follow it; verdicts come from
the registry and chapter coverage from the chapter map. Deterministic scoring rules.
Built: reading.py, how an answer is read: which claim each verdict belongs to, which subject is said
to be untested, what is refused.
Built: run.py, the runner: B1 (the agent, through the answer verifier) and B0 (notes, notebook
prose, claims, chapters and summary values in one cached prompt, no tools), an authorization
category run without a model (requests the tools must refuse), metrics, cost and latency, and the
decision. `python -m rag eval` writes build/eval/<release id>/; `python -m rag release --promote`
serves a release only if that report passed on the full set with the baseline, for the code, model,
provider and effort that would serve it.

Blocking thresholds: numbers matching tool outputs >= 98%, verdicts = registry 100% and no verdict
rule violation, voice labels 100%, untested claims and chapters never given a verdict, every
forbidden request refused, B1 better than B0 on numeric accuracy and voice accuracy (a tie at 100%
counts), citation precision >= 95% and at most 3 points below B0. Measured only, until targets are
set: refusal rate, per-category pass rates, latency, cost.

Built: agent_smoke.py, 6 live questions each probing one answer rule (RAG_LIVE_TESTS=1).
Built: retrieval_smoke.py, 25 questions with a known right document; hit@1/3/5 for keyword,
vector and hybrid search (part of the release gate).
"""
