import json
from types import SimpleNamespace

import pytest

from rag.config import ROOT
from rag.ingest.registry import build_registry, read_report

REPORT = '''
H1_PROVISIONAL = {provisional}

def section_h1(s): return []
def section_h2(s): return []

VERDICTS = {{1: "Partly holds", 2: "Holds"}}
SECTIONS = {{1: section_h1, 2: section_h2}}

def build():
    rows = []
    for n, c in [(1, "Power law (7)"), (2, "Competition destroys profits ({h2_chapters})"),
                 (9, "Founders and organisation (9\\u201310)")]:
        rows.append(n)
'''

README = """# Title

## Findings

| # | Claim (chapter) | Verdict | In one line |
|---|---|---|---|
| H1 | Power law (7) | Partly holds | A few winners. |
| H2 | Competition destroys profits (3–4) | {h2_verdict} | Dominance pays. |

## Notebooks and data

| H1 | not | the | findings |
"""


def make_repo(tmp_path, h2_verdict="Holds", h2_chapters="3–4", provisional=False, h2_summary=True):
    (tmp_path / "findings_report.py").write_text(
        REPORT.format(provisional=provisional, h2_chapters=h2_chapters), encoding="utf-8")
    (tmp_path / "README.md").write_text(README.format(h2_verdict=h2_verdict), encoding="utf-8")
    nbs = tmp_path / "notebooks"
    nbs.mkdir()
    for name, head in (("01_H1.ipynb", "# H1 · Power law (*Zero to One*, ch. 7)\n\n> \"Quote.\"\n"),
                       ("02_H2.ipynb", "# H2 · Competition (*Zero to One*, ch. 3–4)\n\n**Caveats**\n* One.\n")):
        (nbs / name).write_text(json.dumps({"cells": [{"cell_type": "markdown", "metadata": {}, "source": head}]}),
                                encoding="utf-8")
    out = tmp_path / "output"
    for d in ("tables", "figures"):
        (out / d).mkdir(parents=True)
    (out / "h1_summary.json").write_text('{"alpha": 1, "top10": 2}', encoding="utf-8")
    if h2_summary:
        (out / "h2_summary.json").write_text('{"slope": 1}', encoding="utf-8")
    for f in ("h1a_dist.csv", "h1_fixes.csv", "h10_other.csv", "h2b_regs.csv"):
        (out / "tables" / f).write_text("x\n1\n")
    for f in ("h1_fig1_ccdf.png", "h2_fig1_margins.png", "h2_fig1_margins.pdf"):
        (out / "figures" / f).write_bytes(b"")
    return tmp_path


NOTES = [SimpleNamespace(id="note-07-01", chapter=7), SimpleNamespace(id="note-03-01", chapter=3),
         SimpleNamespace(id="note-09-01", chapter=9)]


def test_read_report_without_importing_it(tmp_path):
    r = read_report(make_repo(tmp_path) / "findings_report.py")
    assert r["verdicts"] == {1: "Partly holds", 2: "Holds"}
    assert r["sections"] == {1, 2} and r["h1_provisional"] is False
    assert r["scorecard"][9] == "Founders and organisation (9–10)"


def test_consistent_sources_build_cleanly(tmp_path):
    claims, chapter_map, problems = build_registry(make_repo(tmp_path), NOTES)
    assert problems == []
    by_id = {c.id: c for c in claims}
    h1 = by_id["H1"]
    assert (h1.status, h1.verdict, h1.chapters, h1.thiel_quote) == ("final", "Partly holds", (7,), '"Quote."')
    assert h1.tables == ("h1_fixes", "h1a_dist") and h1.figures == ("h1_fig1_ccdf",)
    assert h1.summary_keys == ("alpha", "top10") and h1.note_ids == ("note-07-01",)
    assert by_id["H2"].caveats == ("One.",) and by_id["H2"].figures == ("h2_fig1_margins",)
    h9 = by_id["H9"]
    assert (h9.status, h9.verdict, h9.chapters, h9.notebook) == ("untested", None, (9, 10), None)
    entries = {e.number: e for e in chapter_map}
    assert entries[7].hypotheses == ("H1",) and entries[7].tested
    assert entries[9].untested == ("H9",) and not entries[9].tested and entries[9].note_ids == ("note-09-01",)
    assert len(chapter_map) == 16


def test_verdict_disagreement_is_a_problem(tmp_path):
    _, _, problems = build_registry(make_repo(tmp_path, h2_verdict="Partly holds"))
    assert problems == ["H2: README verdict 'Partly holds' != VERDICTS 'Holds'"]


def test_chapter_disagreement_is_a_problem(tmp_path):
    _, _, problems = build_registry(make_repo(tmp_path, h2_chapters="3"))
    assert problems == ["H2: findings_report scorecard gives chapters (3,), notebook heading gives (3, 4)"]


def test_provisional_and_pending_claims_carry_no_verdict(tmp_path):
    claims, _, problems = build_registry(make_repo(tmp_path, provisional=True, h2_summary=False))
    by_id = {c.id: c for c in claims}
    assert (by_id["H1"].status, by_id["H1"].verdict) == ("provisional", None)
    assert (by_id["H2"].status, by_id["H2"].verdict) == ("pending", None)
    assert problems == []


def test_the_real_repository_is_consistent():
    """Runs in CI: the notebooks, README, findings_report.py and output/ are all committed."""
    claims, chapter_map, problems = build_registry(ROOT)
    assert problems == []
    assert [(c.id, c.status) for c in claims] == [(f"H{n}", "final") for n in range(1, 9)] + [("H9", "untested")]
    assert all(c.verdict for c in claims if c.status == "final")
    assert [e.number for e in chapter_map if not e.tested] == [0, 8, 9, 10, 14]
