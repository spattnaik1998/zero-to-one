"""Source corpora and the corpus manifest.

The book and the analyst's notes live under `sources/` and are never committed
(see sources/README.md). Every build records what it read in a manifest: file
hashes, the git commit of the analysis, the models used and the book's licence
status. The CI release gate publishes this manifest with each corpus version.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, Settings

ROLES = ("book", "notes")


@dataclass(frozen=True)
class SourceFile:
    role: str
    path: str      # relative to the sources directory
    sha256: str
    bytes: int


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_sources(sources_dir: Path) -> dict[str, list[Path]]:
    """PDFs found for each role, sorted by name. Missing folders give empty lists."""
    out: dict[str, list[Path]] = {}
    for role in ROLES:
        d = sources_dir / role
        out[role] = sorted(p for p in d.glob("*.pdf") if p.is_file()) if d.is_dir() else []
    return out


def describe(sources_dir: Path) -> list[SourceFile]:
    return [SourceFile(role, p.relative_to(sources_dir).as_posix(), sha256(p), p.stat().st_size)
            for role, paths in find_sources(sources_dir).items() for p in paths]


def git_state(repo: Path = ROOT) -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def build_manifest(settings: Settings) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "book_licensed": settings.book_licensed,
        "models": {
            "claude_provider": settings.claude_provider,
            "claude_model": settings.claude_model,
            "embedding_model": settings.embedding_model,
        },
        "sources": [asdict(s) for s in describe(settings.sources_dir)],
    }


def write_manifest(settings: Settings, path: Path | None = None) -> Path:
    path = path or settings.build_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(settings), indent=2) + "\n", encoding="utf-8")
    return path
