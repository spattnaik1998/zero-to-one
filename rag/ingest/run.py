"""Parse the sources and write candidate chunk files to build/candidates/.

These are CI candidates: nothing here is served until the release gate passes
(architecture band A). Each file is JSON Lines, one record per line.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from ..config import Settings
from ..sources import find_sources, sha256
from .book import book_chunks, check_structure, read_book
from .notes import missing_chapters, read_notes


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


def ingest_book(settings: Settings) -> dict:
    path = _one(find_sources(settings.sources_dir)["book"], "book", settings.sources_dir / "book")
    parts, paras = read_book(path)
    check_structure(parts)
    chunks = book_chunks(paras)
    out = write_jsonl(settings.build_dir / "candidates" / "book_chunks.jsonl", chunks)
    sizes = [c.n_words for c in chunks]
    return {
        "source_sha256": sha256(path), "output": str(out),
        "parts": [{"number": p.number, "title": p.title, "pages": [p.page_start, p.page_end]} for p in parts],
        "paragraphs": len(paras), "chunks": len(chunks),
        "words": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)},
        "licensed": settings.book_licensed,
    }


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
