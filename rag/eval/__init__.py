"""Evaluation and the CI release gate (spec section 10).

Built: agent_smoke.py, 6 live questions each probing one answer rule (exact numbers, an untested
chapter, no investment advice, prompt injection, voices kept apart); tests/test_agent.py runs it
only with RAG_LIVE_TESTS=1, since it calls Claude.
Built: retrieval_smoke.py, 25 questions with a known right document; reports hit@1/3/5
for keyword, vector and hybrid search (tests/test_retrieval.py runs it where an OpenAI
key and the built indexes exist).

Golden set (~140, planned): verdict 40, numeric lookup 40, method 20, chapter overview 15
(incl. untested chapters 0, 8, 9, 10, 14), cross-chapter 15, out of scope/adversarial 10,
plus an authorization category (tool ACL negatives).

Answer thresholds (planned): numbers matching tool outputs >= 98% (allowing for the
summaries' rounding), verdicts = registry 100%, voice labels 100%, untested/provisional
claims never get a verdict. Any failure will block publication.
Today the release gate (rag/release.py) runs retrieval_smoke and requires every
question's answer in the top 5.
Open (set from the first baseline): citation precision, out-of-scope refusal rate.
"""
