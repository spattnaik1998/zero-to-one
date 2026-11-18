"""Search indexes from the candidate files (spec section 6.2).

Three indexes, kept apart so an interpretation, a finding and a table are never mixed:
  notes     the analyst's bullets (voice=analyst)
  analysis  notebook chunks (voice=evidence, or thiel for the claim-cell quotations)
  catalog   table schema cards and figure captions, for finding the right table or chart

Each document keeps two texts: `text`, shown and cited as-is, and `embed_text`, the
same content with its context (chapter, hypothesis, section) prepended, which is what
gets embedded and keyword-indexed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

from .embed import Embedder

INDEXES = ("notes", "analysis", "catalog")


@dataclass(frozen=True)
class Doc:
    index: str
    doc_id: str
    text: str
    embed_text: str
    meta: dict


def read_jsonl(path: Path) -> list[dict]:
    """Records of a JSON Lines file; blank lines are skipped."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _context(*parts) -> str:
    """Join context labels, dropping one the next label already starts with
    ("H3c" before "H3c · Are the giants durable?")."""
    parts = [p for p in parts if p]
    return " · ".join(p for i, p in enumerate(parts) if not (i + 1 < len(parts) and parts[i + 1].startswith(p)))


def hypothesis_titles(candidates: Path) -> dict[str, str]:
    """H-number -> claim title from the claims registry ("H1" -> "The power law"), if it was built."""
    f = candidates / "claims_registry.json"
    return {c["id"]: c["title"] for c in json.loads(f.read_text(encoding="utf-8"))} if f.is_file() else {}


def note_docs(candidates: Path) -> list[Doc]:
    out = []
    for n in read_jsonl(candidates / "notes_chunks.jsonl"):
        label = "Analyst's view" if n["opinion"] else "Analyst's note"
        out.append(Doc("notes", n["id"], n["text"], f"{n['chapter_title']}. {label}: {n['text']}",
                       {"chapter": n["chapter"], "chapter_title": n["chapter_title"], "opinion": n["opinion"],
                        "voice": n["voice"], "page": n["page"], "source": "notes"}))
    return out


def analysis_docs(candidates: Path) -> list[Doc]:
    titles, out = hypothesis_titles(candidates), []
    for c in read_jsonl(candidates / "notebook_chunks.jsonl"):
        where = _context(f'{c["hypothesis"]} {titles.get(c["hypothesis"], "")}'.strip(), c["sub_claim"], c["section"])
        who = "Thiel, quoted in the claim" if c["voice"] == "thiel" else c["kind"]
        out.append(Doc("analysis", c["id"], c["text"], f"{where} ({who}): {c['text']}",
                       {k: c[k] for k in ("hypothesis", "sub_claim", "kind", "voice", "section", "cell",
                                          "cell_type", "chapters", "notebook")}))
    return out


def catalog_docs(candidates: Path) -> list[Doc]:
    titles, out = hypothesis_titles(candidates), []
    for t in json.loads((candidates / "schema_cards.json").read_text(encoding="utf-8")):
        cols = []
        for c in t["columns"]:
            detail = f"{c['min']} to {c['max']}" if c["min"] is not None else ", ".join(c["examples"])
            cols.append(f"{c['name']} ({c['type']}{'; ' + detail if detail else ''})")
        where = _context(f'{t["hypothesis"]} {titles.get(t["hypothesis"], "")}'.strip(), t["sub_claim"], t["section"])
        text = f"Table {t['name']}: {t['rows']} rows. Columns: " + "; ".join(cols)
        out.append(Doc("catalog", f"table:{t['name']}", text, f"{where}. {text}",
                       {"type": "table", "name": t["name"], "hypothesis": t["hypothesis"],
                        "sub_claim": t["sub_claim"], "section": t["section"], "rows": t["rows"]}))
    for f in json.loads((candidates / "figures.json").read_text(encoding="utf-8")):
        text = f"Figure {f['name']}: {f['caption']}"
        out.append(Doc("catalog", f"figure:{f['name']}", text,
                       f"{f['hypothesis']} {titles.get(f['hypothesis'], '')}. {text}",
                       {"type": "figure", "name": f["name"], "hypothesis": f["hypothesis"],
                        "caption_source": f["caption_source"], "png": f["png"]}))
    return out


def all_docs(candidates: Path) -> list[Doc]:
    docs = note_docs(candidates) + analysis_docs(candidates) + catalog_docs(candidates)
    ids = [(d.index, d.doc_id) for d in docs]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise ValueError(f"duplicate document ids: {dupes}")
    return docs


def content_hash(docs: list[Doc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: (d.index, d.doc_id)):
        h.update(f"{d.index}\x00{d.doc_id}\x00{d.embed_text}\x00".encode("utf-8"))
    return h.hexdigest()


def build_indexes(candidates: Path, out: Path, embedder: Embedder) -> dict:
    docs = all_docs(candidates)
    vectors = embedder.embed([d.embed_text for d in docs])
    if vectors.shape[0] != len(docs):
        raise ValueError(f"got {vectors.shape[0]} vectors for {len(docs)} documents")
    dims = vectors.shape[1]
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embeddings are not unit length")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    try:
        con.execute(f"""CREATE TABLE documents (index_name VARCHAR, doc_id VARCHAR, text VARCHAR,
                        embed_text VARCHAR, meta VARCHAR, embedding FLOAT[{dims}],
                        PRIMARY KEY (index_name, doc_id))""")
        con.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                        [(d.index, d.doc_id, d.text, d.embed_text, json.dumps(d.meta, ensure_ascii=False), v.tolist())
                         for d, v in zip(docs, vectors)])
        con.execute("""CREATE TABLE index_info (index_name VARCHAR PRIMARY KEY, model VARCHAR, dims INTEGER,
                       n_docs INTEGER, content_hash VARCHAR, built_utc VARCHAR)""")
        built = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name in INDEXES:
            part = [d for d in docs if d.index == name]
            con.execute("INSERT INTO index_info VALUES (?, ?, ?, ?, ?, ?)",
                        [name, embedder.model, dims, len(part), content_hash(part), built])
    finally:
        con.close()
    return {"output": str(out), "dims": dims, "model": embedder.model,
            "counts": {name: sum(d.index == name for d in docs) for name in INDEXES},
            "content_hash": content_hash(docs)}
