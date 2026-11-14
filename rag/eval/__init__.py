"""Evaluation and the CI release gate (spec section 10).

Golden set (~180): verdict 40, numeric lookup 40, book quote/definition 25, method 20,
chapter overview 15, notes vs. book 15, cross-chapter 15, out of scope/adversarial 10,
plus an authorization category (tool ACL negatives).

Gate thresholds: numbers matching tool outputs >= 98%, verdicts = registry 100%,
voice labels 100%, alignment labels >= 90%, alignment numerics 100%,
untested/provisional claims never get a verdict. Any failure blocks publication.
Open (set from the first baseline): citation precision, out-of-scope refusal rate.
"""
