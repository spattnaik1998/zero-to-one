"""A synthetic PDF with the same typography as the analyst's notes.

The parser tests run on it so they cover layout cases the real notes do not
contain. Text is placeholder; only the layout matters.
"""
import json
import zlib

import numpy as np
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

H = letter[1]


def put(c, x, base, text, font="Times-Roman", size=15.0):
    """Draw text with its baseline `base` points from the top; return the x where it ends."""
    c.setFont(font, size)
    c.drawString(x, H - base, text)
    return x + c.stringWidth(text, font, size)


def _rule(c, top):
    c.setLineWidth(0.5)
    c.line(71, H - top, 540, H - top)


@pytest.fixture
def notes_pdf(tmp_path) -> Path:
    path = tmp_path / "notes.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    bullet = lambda base, text: (put(c, 71, base, "•", "Helvetica", 10), put(c, 83, base, text, size=9.8))  # noqa: E731

    put(c, 71, 88, "Zero to One", "Times-Bold", 20)
    put(c, 71, 104, "Notes by chapter.", "Times-Italic", 10)
    put(c, 71, 134, "1 · The Challenge of the Future", "Times-Bold", 12.5)
    bullet(150, "First bullet starts here and")
    put(c, 83, 163, "continues on a second line.", size=9.8)
    bullet(180, "Second bullet with a list:")
    put(c, 89, 193, "1) first item", size=9.8)
    put(c, 89, 206, "2) second item", size=9.8)
    put(c, 71, 236, "12 · Man and Machine", "Times-Bold", 12.5)
    bullet(252, "My view: computers complement people.")
    put(c, 71, 282, "Conclusion · Stagnation or Singularity?", "Times-Bold", 12.5)
    bullet(298, "Four possible futures.")
    put(c, 71, 330, "Changes from the original notes", "Times-Bold", 12.5)
    for top in (348, 366, 396, 424):
        _rule(c, top)
    put(c, 75, 360, "Original note", "Times-Bold", 8.6)
    put(c, 254, 360, "What changed", "Times-Bold", 8.6)
    put(c, 75, 378, "Left cell line one", size=8.6)
    put(c, 75, 389, "left cell line two.", size=8.6)
    put(c, 254, 378, "Right one line.", size=8.6)
    put(c, 75, 410, "Short left.", size=8.6)
    put(c, 254, 405, "Right cell first,", size=8.6)
    put(c, 254, 416, "right cell second.", size=8.6)
    put(c, 464, 760, "Zero to One · page 1", size=8.0)
    c.showPage()

    for top in (60, 78, 100):
        _rule(c, top)
    put(c, 75, 72, "Original note", "Times-Bold", 8.6)
    put(c, 254, 72, "What changed", "Times-Bold", 8.6)
    put(c, 75, 92, "Third original.", size=8.6)
    put(c, 254, 92, "Third change.", size=8.6)
    put(c, 464, 760, "Zero to One · page 2", size=8.0)
    c.showPage()
    c.save()
    return path


# ---------------------------------------------------------------- shared by retrieval and release tests
from rag.ingest.embed import normalise  # noqa: E402
from rag.retrieval import tokenize  # noqa: E402


class HashEmbedder:
    """Deterministic stand-in for OpenAI: hashed bag of words, so shared words mean similarity.
    Uses CRC32, not hash(), which Python randomises per process."""
    model, dims = "hash-test", 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.dims), np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t) or ["empty"]:
                out[i, zlib.crc32(tok.encode()) % self.dims] += 1.0
        return normalise(out)


@pytest.fixture
def candidates(tmp_path):
    c = tmp_path / "candidates"
    c.mkdir()
    notes = [
        {"id": "note-03-01", "chapter": 3, "chapter_title": "3 All Happy Companies Are Different", "page": 1,
         "opinion": False, "voice": "analyst", "text": "Airlines made 37 cents per passenger; Google's margin was 21%."},
        {"id": "note-09-01", "chapter": 9, "chapter_title": "9 Foundations", "page": 2,
         "opinion": False, "voice": "analyst", "text": "A board of three is ideal. CEO pay under $150,000."},
        {"id": "note-12-01", "chapter": 12, "chapter_title": "12 Man and Machine", "page": 2,
         "opinion": True, "voice": "analyst", "text": "My view: computers complement people."},
    ]
    (c / "notes_chunks.jsonl").write_text("\n".join(json.dumps(n) for n in notes), encoding="utf-8")
    chunks = [
        {"id": "nb-h3-001", "hypothesis": "H3", "sub_claim": "H3c", "kind": "method", "voice": "evidence",
         "section": "H3c · Are the giants durable?", "cell": 8, "cell_type": "markdown", "chapters": [5],
         "notebook": "notebooks/03.ipynb", "text": "The largest firms by public float, five-year survival."},
        {"id": "nb-h3-002", "hypothesis": "H3", "sub_claim": None, "kind": "caveat", "voice": "evidence",
         "section": None, "cell": 0, "cell_type": "markdown", "chapters": [5], "notebook": "notebooks/03.ipynb",
         "text": "Firms that stop filing may have been acquired or failed."},
    ]
    (c / "notebook_chunks.jsonl").write_text("\n".join(json.dumps(x) for x in chunks), encoding="utf-8")
    (c / "schema_cards.json").write_text(json.dumps([{
        "name": "h3c_top20_2014", "hypothesis": "H3", "sub_claim": "H3c", "section": "H3c · Are the giants durable?",
        "rows": 20, "columns": [{"name": "firm", "type": "str", "min": None, "max": None, "examples": ["Apple Inc."]},
                                {"name": "rank_2025", "type": "float64", "min": 2.0, "max": 221.0, "examples": []}]}]))
    (c / "figures.json").write_text(json.dumps([{"name": "h1_fig1_ccdf", "hypothesis": "H1",
        "caption": "The tail in the year of the book", "caption_source": "notebook heading",
        "png": "output/figures/h1_fig1_ccdf.png"}]))
    (c / "claims_registry.json").write_text(json.dumps([{"id": "H1", "title": "The power law"},
                                                        {"id": "H3", "title": "Durability"}]))
    return c


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


# ---------------------------------------------------------------- a complete synthetic release (tools, agent)
import duckdb  # noqa: E402

from rag.config import load_settings  # noqa: E402
from rag.ingest.index import build_indexes  # noqa: E402
from rag.release import Check, GateReport, open_release, publish  # noqa: E402
from rag.tools.toolset import Toolset  # noqa: E402


CLAIMS = [
    {"id": "H1", "title": "The power law", "claim": "Power law", "chapters": [7], "status": "final",
     "verdict": "Partly holds", "one_line": "A few winners.", "thiel_quote": "q", "sub_claims": [], "caveats": ["c"],
     "notebook": "notebooks/01.ipynb", "summary": "output/h1_summary.json", "summary_keys": ["alpha"],
     "tables": ["h3c_top20_2014"], "figures": ["h1_fig1_ccdf"], "note_ids": []},
    {"id": "H3", "title": "Durability", "claim": "Durability", "chapters": [5], "status": "final",
     "verdict": "Holds, with a twist", "one_line": "Leaders last.", "thiel_quote": "", "sub_claims": [],
     "caveats": [], "notebook": "notebooks/03.ipynb", "summary": None, "summary_keys": [], "tables": [],
     "figures": [], "note_ids": []},
    {"id": "H9", "title": "Founders and organisation", "claim": "Founders", "chapters": [9, 10], "status": "untested",
     "verdict": None, "one_line": None, "thiel_quote": "", "sub_claims": [], "caveats": [], "notebook": None,
     "summary": None, "summary_keys": [], "tables": [], "figures": [], "note_ids": ["note-09-01"]},
]


def _chapter_map():
    out = []
    for n in range(16):
        hyps = [c["id"] for c in CLAIMS if n in c["chapters"] and c["status"] == "final"]
        placeholders = [c["id"] for c in CLAIMS if n in c["chapters"] and c["status"] == "untested"]
        notes = {3: ["note-03-01"], 9: ["note-09-01"], 12: ["note-12-01"]}.get(n, [])
        out.append({"number": n, "title": f"Chapter {n}", "hypotheses": hyps, "untested": placeholders,
                    "tested": bool(hyps), "note_ids": notes})
    return out


@pytest.fixture
def toolset(candidates, hash_embedder, tmp_path):
    (candidates / "claims_registry.json").write_text(json.dumps(CLAIMS))
    (candidates / "chapter_map.json").write_text(json.dumps(_chapter_map()))
    (candidates / "notes_corrections.jsonl").write_text("")
    con = duckdb.connect(str(candidates / "analysis.duckdb"))
    con.execute("CREATE TABLE h3c_top20_2014 AS SELECT * FROM (VALUES ('Apple', 1, 3, true), ('Exxon', 2, 16, true), "
                "('GE', 3, 90, false), ('Odd', 4, NULL, false)) t(firm, rank_2014, rank_2025, still_top20)")
    con.execute("CREATE TABLE summary_values AS SELECT * FROM (VALUES "
                "('H1', 'alpha', 'alpha', 'float', 1.8, '1.8', 'h1_summary.json'), "
                "('H1', 'n_firms', 'n_firms', 'int', 39363.0, '39363', 'h1_summary.json'), "
                "('H1', 'ok', 'ok', 'bool', 1.0, 'true', 'h1_summary.json'), "
                "('H1', 'years[0]', 'years', 'int', 2011.0, '2011', 'h1_summary.json'), "
                "('H1', 'top5[0].firm', 'top5', 'str', NULL, 'NVIDIA', 'h1_summary.json')) "
                "t(hypothesis, path, top_key, value_type, value_num, value_text, source)")
    con.close()
    build_indexes(candidates, candidates / "indexes.duckdb", hash_embedder)
    (tmp_path / "output" / "figures").mkdir(parents=True)
    (tmp_path / "output" / "figures" / "h1_fig1_ccdf.png").write_bytes(b"png")
    settings = load_settings(environ={"RAG_BUILD_DIR": str(candidates.parent), "OPENAI_EMBEDDING_MODEL": "hash-test"},
                             dotenv={})
    publish(settings, GateReport([Check("all", True, "ok")]), root=tmp_path, make_current=True)
    return Toolset(open_release(settings), hash_embedder)


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path, monkeypatch):
    """Every test writes audit records to its own folder, never to the real build/audit/."""
    monkeypatch.setenv("RAG_AUDIT_DIR", str(tmp_path / "audit"))
