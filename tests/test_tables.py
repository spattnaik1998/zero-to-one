import json

import duckdb
import pytest

from rag.config import ROOT
from rag.ingest.tables import build_database, flatten, report_captions, verify_database

REPORT = '''
def fig(name, caption): return []

def section_h2(s):
    E, L = s["end_year"], s["last_year"]
    end = s.get("end_year", 0)
    return fig("h2_fig1_margins", f"Margins to {E} (data to {L}), ranked {end}.") + \\
        fig("h2_fig2_shares", "A constant caption.") + fig("h2_fig3_bad", f"Unknown {missing}.")
'''


def nb(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


@pytest.fixture
def repo(tmp_path):
    out = tmp_path / "output"
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir()
    (out / "tables" / "h2a_slopes.csv").write_text(",slope,n\nall,0.5,10\nsmall,,3\n")
    (out / "tables" / "h2b_quint_2014_2025.csv").write_text("q,growth\n1,1.5\n2,2.5\n")
    (out / "h2_summary.json").write_text(json.dumps(
        {"end_year": 2025, "last_year": 2024, "flag": True, "none": None,
         "groups": [{"name": "tech", "payback": 1.038}, {"name": "rest", "payback": 0.74}],
         "by_year": {"2025": {"margin": 27.4}}}))
    for f in ("h2_fig1_margins", "h2_fig2_shares", "h2_fig3_bad", "h2_fig4_notebook_only"):
        (out / "figures" / f"{f}.png").write_bytes(b"png")
    (out / "figures" / "h2_fig1_margins.pdf").write_bytes(b"pdf")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "02_H2.ipynb").write_text(nb([
        {"cell_type": "markdown", "metadata": {}, "source": "# H2 · Competition (ch. 3)\n"},
        {"cell_type": "markdown", "metadata": {}, "source": "## H2a · Slopes\n"},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": 1,
         "source": 'tc.save_table(t, "h2a_slopes")\nq(df, f"h2b_quint_2014_{END}")'},
        {"cell_type": "markdown", "metadata": {}, "source": "### Figure H2-4 · Only in the notebook\n"},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": 2,
         "source": 'tc.save_fig(fig, "h2_fig4_notebook_only")'},
    ]))
    (tmp_path / "findings_report.py").write_text(REPORT)
    return tmp_path


def test_flatten_paths():
    assert flatten({"a": [{"b": 1}, 2], "c": {"d": None}}) == [("a[0].b", 1), ("a[1]", 2), ("c.d", None)]


def test_tables_load_exactly_with_provenance(repo, tmp_path):
    db = tmp_path / "a.duckdb"
    cards, figures, n_values = build_database(repo, db)
    assert verify_database(repo, db) == []
    by_name = {c.name: c for c in cards}

    slopes = by_name["h2a_slopes"]
    assert (slopes.hypothesis, slopes.sub_claim, slopes.rows, slopes.renamed_index) == ("H2", "H2a", 2, True)
    assert [c.name for c in slopes.columns] == ["row_label", "slope", "n"]
    assert slopes.grain == "row_label"
    slope = slopes.columns[1]
    assert (slope.nulls, slope.min, slope.max) == (1, 0.5, 0.5)
    assert (slopes.notebook, slopes.cell, slopes.section) == ("notebooks/02_H2.ipynb", 2, "H2a · Slopes")
    assert by_name["h2b_quint_2014_2025"].cell == 2        # name written as an f-string with {END}

    con = duckdb.connect(str(db), read_only=True)
    assert con.execute('SELECT row_label, slope FROM h2a_slopes ORDER BY n DESC').fetchall() == [("all", 0.5), ("small", None)]
    values = dict(con.execute("SELECT path, value_num FROM summary_values").fetchall())
    texts = dict(con.execute("SELECT path, value_text FROM summary_values").fetchall())
    con.close()
    assert n_values == len(values) == 9          # end_year, last_year, flag, none, 4 in groups, 1 in by_year
    assert values["groups[0].payback"] == 1.038 and values["by_year.2025.margin"] == 27.4
    assert (values["flag"], texts["flag"]) == (1.0, "true")
    assert (values["none"], texts["none"]) == (None, None)


def test_figure_captions_and_sources(repo, tmp_path):
    _, figures, _ = build_database(repo, tmp_path / "a.duckdb")
    by_name = {f.name: f for f in figures}
    assert by_name["h2_fig1_margins"].caption == "Margins to 2025 (data to 2024), ranked 2025."
    assert by_name["h2_fig1_margins"].pdf == "output/figures/h2_fig1_margins.pdf"
    assert by_name["h2_fig2_shares"].caption_source == "report"
    assert by_name["h2_fig3_bad"].unresolved == ("missing",)
    nb_only = by_name["h2_fig4_notebook_only"]
    assert (nb_only.caption, nb_only.caption_source) == ("Figure H2-4 · Only in the notebook", "notebook heading")


def test_verification_catches_a_changed_value(repo, tmp_path):
    db = tmp_path / "a.duckdb"
    build_database(repo, db)
    con = duckdb.connect(str(db))
    con.execute("UPDATE h2a_slopes SET slope = 0.6 WHERE row_label = 'all'")
    con.execute("UPDATE summary_values SET value_num = 1.04 WHERE path = 'groups[0].payback'")
    con.close()
    problems = verify_database(repo, db)
    assert "h2a_slopes.slope: values differ from the CSV" in problems
    assert any("groups[0].payback" in p for p in problems)


def test_the_real_outputs_load_exactly(tmp_path):
    """Runs in CI: output/, the notebooks and findings_report.py are committed."""
    db = tmp_path / "real.duckdb"
    cards, figures, n_values = build_database(ROOT, db)
    assert verify_database(ROOT, db) == []
    assert len(cards) == len(list((ROOT / "output" / "tables").glob("*.csv")))
    assert all(c.cell is not None for c in cards), [c.name for c in cards if c.cell is None]
    assert len(figures) == len(list((ROOT / "output" / "figures").glob("*.png")))
    assert all(f.caption and not f.unresolved for f in figures)
    assert report_captions(ROOT / "findings_report.py", ROOT / "output")["h3_fig3_top20"][0].endswith("rank in 2025.")
