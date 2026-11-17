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
