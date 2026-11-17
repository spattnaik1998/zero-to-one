import json

import pytest

from rag.ingest.notebooks import notebook_chunks, parse_chapters, parse_claim_cell

CLAIM = """# H3 · Durability and the last-mover advantage (*Zero to One*, ch. 5)

> "First quoted line."
> "Second quoted line."

Context paragraph about the claim.

| Claim | Test | Data |
|---|---|---|
| **H3a.** Profits last. | Top-decile persistence. | SEC XBRL |
| **H3b.** Leaders stay leaders. | Five-year survival. | SEC + SIC |

**Caveats**
* First caveat.
* Second caveat.
"""


def write_nb(path, cells):
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    return path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [{"text": "big output"}],
            "execution_count": 1}


@pytest.mark.parametrize("text, expected", [
    ("*Zero to One*, ch. 3–4", (3, 4)),
    ("ch. 3, 5, 6, 12, 13", (3, 5, 6, 12, 13)),
    ("*Zero to One*, ch. 1 and conclusion", (1, 15)),
    ("2, 11", (2, 11)),
    ("1, conclusion", (1, 15)),
    ("9–10", (9, 10)),
])
def test_parse_chapters(text, expected):
    assert parse_chapters(text) == expected


def test_parse_chapters_rejects_chapters_outside_the_book():
    with pytest.raises(ValueError):
        parse_chapters("ch. 21")


def test_claim_cell_parts():
    c = parse_claim_cell(CLAIM)
    assert (c.hypothesis, c.title, c.chapters) == ("H3", "Durability and the last-mover advantage", (5,))
    assert c.quote == '"First quoted line." "Second quoted line."'
    assert [s["id"] for s in c.sub_claims] == ["H3a", "H3b"]
    assert c.sub_claims[1] == {"id": "H3b", "claim": "Leaders stay leaders.", "test": "Five-year survival.",
                               "data": "SEC + SIC"}
    assert c.caveats == ("First caveat.", "Second caveat.")
    assert "Caveats" not in c.context and "First quoted" not in c.context


def test_inline_caveat_paragraph():
    c = parse_claim_cell("# H4 · Named companies (ch. 3, 5)\n\nIntro.\n\n"
                         "**Interpretation caveat.** This is a scorecard, not a test of cause.\n")
    assert c.caveats == ("This is a scorecard, not a test of cause.",) and c.quote == ""


def test_text_after_the_caveat_list_returns_to_the_context():
    c = parse_claim_cell("# H8 · Computers (ch. 12)\n\nIntro.\n\n**Caveats**\n* Only caveat.\n\n"
                         "**Before running:** install openpyxl.\n")
    assert c.caveats == ("Only caveat.",)
    assert c.context.endswith("**Before running:** install openpyxl.")


def test_claim_cell_needs_a_heading_with_chapters():
    with pytest.raises(ValueError):
        parse_claim_cell("# H3 · No chapters here\n")


def test_notebook_chunks_kinds_voices_and_sections(tmp_path):
    nb = write_nb(tmp_path / "03_H3.ipynb", [
        md(CLAIM),
        code("import thiel_common as tc"),
        md("## H3a · Does high profitability last?\n\nMethod text."),
        code("x = 1"),
        md("### Figure H3-1 · Persistence\n\nWhat the figure shows."),
        md("## H3b · Do leaders stay leaders?\n\nMore method."),
        code(""),
        md("## Verdict inputs and check cell\n"),
    ])
    claim, chunks = notebook_chunks(nb, tmp_path)
    rows = [(c.kind, c.voice, c.section, c.sub_claim, c.cell, c.cell_type) for c in chunks]
    assert rows == [
        ("claim", "thiel", None, None, 0, "markdown"),
        ("claim", "evidence", None, None, 0, "markdown"),
        ("caveat", "evidence", None, None, 0, "markdown"),
        ("method", "evidence", None, None, 1, "code"),
        ("method", "evidence", "H3a · Does high profitability last?", "H3a", 2, "markdown"),
        ("method", "evidence", "H3a · Does high profitability last?", "H3a", 3, "code"),
        ("result", "evidence", "H3a · Does high profitability last?", "H3a", 4, "markdown"),
        ("method", "evidence", "H3b · Do leaders stay leaders?", "H3b", 5, "markdown"),
        ("result", "evidence", "Verdict inputs and check cell", None, 7, "markdown"),
    ]
    assert chunks[0].id == "nb-h3-001" and chunks[0].notebook == "03_H3.ipynb"
    assert all(c.chapters == (5,) for c in chunks)
    assert not any("big output" in c.text for c in chunks)       # outputs are not indexed


def test_long_code_cell_is_split_at_blank_lines(tmp_path):
    block = "\n".join(f"v{i} = {i}" for i in range(150))          # 450 words
    nb = write_nb(tmp_path / "01_H1.ipynb", [md("# H1 · Power law (ch. 7)\n"), code(block + "\n\n" + block)])
    _, chunks = notebook_chunks(nb, tmp_path)
    code_chunks = [c for c in chunks if c.cell_type == "code"]
    assert len(code_chunks) == 2 and all(c.n_words <= 450 for c in code_chunks)
