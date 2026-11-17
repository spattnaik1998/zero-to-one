"""Parse the sources and write candidate chunk files to build/candidates/.

These are CI candidates: nothing here is served until the release gate passes
(architecture band A). Each file is JSON Lines, one record per line.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import ROOT, Settings
from ..sources import find_sources, sha256
from .notebooks import notebook_chunks
from .notes import missing_chapters, read_notes
from .registry import build_registry, write_json
from .tables import build_database, verify_database
from .embed import EmbeddingCache, OpenAIEmbedder
from .index import build_indexes


class RegistryError(RuntimeError):
    """The registry's sources disagree; the build must stop."""


def write_jsonl(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path


def _one(paths: list[Path], role: str, folder: Path) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"expected exactly one {role} PDF in {folder}, found {len(paths)}")
    return paths[0]


def ingest_notes(settings: Settings) -> dict:
    path = _one(find_sources(settings.sources_dir)["notes"], "notes", settings.sources_dir / "notes")
    chunks, corrections = read_notes(path)
    out = write_jsonl(settings.build_dir / "candidates" / "notes_chunks.jsonl", chunks)
    out_c = write_jsonl(settings.build_dir / "candidates" / "notes_corrections.jsonl", corrections)
    return {
        "source_sha256": sha256(path), "output": [str(out), str(out_c)],
        "bullets": len(chunks), "opinion_bullets": sum(c.opinion for c in chunks),
        "chapters_without_notes": missing_chapters(chunks), "corrections": len(corrections),
    }


def ingest_notebooks(settings: Settings, root: Path = ROOT) -> dict:
    chunks, per = [], {}
    for nb in sorted((root / "notebooks").glob("*.ipynb")):
        claim, cs = notebook_chunks(nb, root)
        chunks += cs
        per[claim.hypothesis] = len(cs)
    out = write_jsonl(settings.build_dir / "candidates" / "notebook_chunks.jsonl", chunks)
    kinds = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    return {"output": str(out), "chunks": len(chunks), "per_hypothesis": per, "kinds": kinds,
            "thiel_quotes": sum(c.voice == "thiel" for c in chunks)}


class IndexBuildError(RuntimeError):
    """The search indexes cannot be built (missing key or model, or missing candidate files)."""


def ingest_index(settings: Settings) -> dict:
    """Embed the candidate documents and write build/candidates/indexes.duckdb."""
    if not settings.openai_api_key or not settings.embedding_model:
        raise IndexBuildError("OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL must both be set (see python -m rag doctor)")
    candidates = settings.build_dir / "candidates"
    # claims_registry.json supplies the hypothesis titles used as search context; without it the
    # index would silently differ from a full build.
    needed = ["notes_chunks.jsonl", "notebook_chunks.jsonl", "schema_cards.json", "figures.json",
              "claims_registry.json"]
    missing = [n for n in needed if not (candidates / n).is_file()]
    if missing:
        raise IndexBuildError(f"run the earlier ingest steps first; missing {missing}")
    cache = EmbeddingCache(settings.build_dir / "cache" / "embeddings.duckdb")
    try:
        embedder = OpenAIEmbedder(settings.embedding_model, settings.openai_api_key, cache)
        report = build_indexes(candidates, candidates / "indexes.duckdb", embedder)
    finally:
        cache.close()
    report.update(api_calls=embedder.api_calls, tokens=embedder.tokens, cache_hits=embedder.cache_hits)
    return report


class TablesError(RuntimeError):
    """The database does not match the files it was built from."""


def ingest_tables(settings: Settings, root: Path = ROOT) -> dict:
    """Load output/ into DuckDB, then check every value against the source files."""
    db = settings.build_dir / "candidates" / "analysis.duckdb"
    cards, figures, n_values = build_database(root, db)
    problems = verify_database(root, db)
    if problems:
        raise TablesError("the database differs from output/:\n  - " + "\n  - ".join(problems[:20]))
    out_cards = write_json(settings.build_dir / "candidates" / "schema_cards.json", cards)
    out_figs = write_json(settings.build_dir / "candidates" / "figures.json", figures)
    return {
        "output": [str(db), str(out_cards), str(out_figs)],
        "tables": len(cards), "rows": sum(c.rows for c in cards), "summary_values": n_values,
        "figures": len(figures), "untraced_tables": [c.name for c in cards if c.cell is None],
        "caption_sources": {s: sum(f.caption_source == s for f in figures) for s in ("report", "notebook heading", "none")},
        "unresolved_captions": [f.name for f in figures if f.unresolved],
    }


def ingest_registry(settings: Settings, root: Path = ROOT) -> dict:
    """Build the claims registry and chapter map. Raises RegistryError if the sources disagree."""
    notes_pdf = find_sources(settings.sources_dir)["notes"]
    notes = read_notes(notes_pdf[0])[0] if len(notes_pdf) == 1 else []
    claims, chapter_map, problems = build_registry(root, notes)
    if problems:
        raise RegistryError("the registry's sources disagree:\n  - " + "\n  - ".join(problems))
    out = write_json(settings.build_dir / "candidates" / "claims_registry.json", claims)
    out_map = write_json(settings.build_dir / "candidates" / "chapter_map.json", chapter_map)
    return {
        "output": [str(out), str(out_map)],
        "claims": [(c.id, c.status, c.verdict) for c in claims],
        "untested_chapters": [e.number for e in chapter_map if not e.tested],
        "notes_linked": bool(notes),
    }
