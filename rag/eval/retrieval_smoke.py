"""Retrieval smoke test: realistic questions with a known right document (spec section 10).

Each case names the index, a question as a user would phrase it, and how to recognise
the right document (by id, by text it contains, and/or by metadata). `run` reports
hit@1/@3/@5 for keyword-only, vector-only and hybrid search, so the choice of hybrid
rests on numbers. It needs the built indexes and an embedder for the questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval import HybridIndex


@dataclass(frozen=True)
class Case:
    index: str
    query: str
    doc_id: str | None = None
    contains: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    def matches(self, hit) -> bool:
        if self.doc_id and hit.doc_id != self.doc_id:
            return False
        if any(c.lower() not in hit.text.lower() for c in self.contains):
            return False
        return all(hit.meta.get(k) == v for k, v in self.meta.items())


CASES = (
    # notes: exact facts and paraphrases of the analyst's bullets
    Case("notes", "How much did airlines make per passenger compared with Google's profit margin?", contains=("37 cents",)),
    Case("notes", "What is the ideal size of a startup's board of directors?", contains=("board of three",)),
    Case("notes", "How much should a startup CEO pay himself?", contains=("$150,000",)),
    Case("notes", "Thiel's four attitudes to the future, definite versus indefinite optimism", contains=("indefinite optimism",)),
    Case("notes", "What is my own view on whether computers replace people?", meta={"opinion": True}),
    Case("notes", "When does a distribution channel work, lifetime value versus acquisition cost", contains=("CLV",)),
    Case("notes", "The seven questions every business must answer", contains=("seven questions every business",)),
    # analysis: claims, methods, caveats in the notebooks
    Case("analysis", "Thiel's quote that the best investment in a fund outperforms the rest combined",
         meta={"hypothesis": "H1", "voice": "thiel"}),
    Case("analysis", "How were unit-scale errors in the SEC public float data corrected?", contains=("scale",), meta={"hypothesis": "H1"}),
    Case("analysis", "Does a firm's market share predict its operating margin?", meta={"hypothesis": "H2"}),
    Case("analysis", "US life expectancy at birth before and after the book", meta={"hypothesis": "H6"}, contains=("life expectancy",)),
    Case("analysis", "Did occupations exposed to AI lose jobs after ChatGPT?", meta={"hypothesis": "H8"}),
    Case("analysis", "What are the limitations of the named-companies scorecard?", meta={"hypothesis": "H4", "kind": "caveat"}),
    Case("analysis", "Do IPOs of companies without profits perform worse?", meta={"hypothesis": "H5"}),
    # catalog: finding the right table or figure
    Case("catalog", "Which table lists the 20 largest firms in 2014 and whether they are still in the top 20?", doc_id="table:h3c_top20_2014"),
    Case("catalog", "Airline versus big tech operating margin by year", doc_id="table:h2c_airlines_vs_bigtech"),
    Case("catalog", "Chart of the employment rate and the unemployment rate over time", doc_id="figure:h8_fig1_aggregate"),
    Case("catalog", "Returns of cleantech stocks since the book", doc_id="table:h5c_cleantech_returns"),
    Case("catalog", "Plot of the power-law tail of firm sizes", doc_id="figure:h1_fig1_ccdf"),
    Case("catalog", "Sales and marketing spending as a share of revenue by sector and year", doc_id="table:h7a_intensity_by_sector_year"),
    # paraphrases that avoid the target's wording, so keyword overlap cannot carry them
    Case("notes", "Is it better to be first into a market or the last one standing?", contains=("last mover",)),
    Case("notes", "Why should co-founders know each other well before starting out?", contains=("prehistory",)),
    Case("analysis", "Did the richest public companies pull further ahead of everyone else?", meta={"hypothesis": "H1"}),
    Case("analysis", "Are we still stuck making progress only in software rather than physical things?", meta={"hypothesis": "H6"}),
    Case("catalog", "Which occupations shrank or grew fastest?", doc_id="table:h8b_movers"),
)


def run(db: Path, embedder, cases=CASES, ks=(1, 3, 5)) -> dict:
    """{mode: {k: hits}} plus the per-case first rank for each mode."""
    indexes = {name: HybridIndex.load(db, name) for name in {c.index for c in cases}}
    vectors = embedder.embed([c.query for c in cases])
    modes = ("bm25", "dense", "hybrid")
    totals = {m: {k: 0 for k in ks} for m in modes}
    per_case = []
    for case, qv in zip(cases, vectors):
        ranks = {}
        for m in modes:
            hits = indexes[case.index].search(case.query, qv, k=max(ks), mode=m)
            rank = next((i + 1 for i, h in enumerate(hits) if case.matches(h)), None)
            ranks[m] = rank
            for k in ks:
                totals[m][k] += rank is not None and rank <= k
        per_case.append((case, ranks))
    return {"n": len(cases), "totals": totals, "per_case": per_case}
