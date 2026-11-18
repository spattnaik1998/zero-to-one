"""Release gate and published corpus versions (architecture band B, spec sections 6 and 10).

`python -m rag ingest` writes *candidates*. Nothing serves from them. `run_gate` checks a
candidate build; `publish` copies a passing build into an immutable, versioned release:

  build/releases/<release_id>/   the artifacts (read-only files) and manifest.json
  build/releases/CURRENT         the release new sessions pin to

`open_release` re-hashes every file against the manifest, so a modified release is refused.
A release ID is derived from the content, so publishing the same build twice is a no-op.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import ROOT, Settings
from .sources import describe, git_state, sha256

ARTIFACTS = ("notes_chunks.jsonl", "notes_corrections.jsonl", "notebook_chunks.jsonl", "claims_registry.json",
             "chapter_map.json", "schema_cards.json", "figures.json", "analysis.duckdb", "indexes.duckdb")
GATED_PATHS = ("notebooks", "output", "findings_report.py", "README.md", "sources/notes", "thiel_common.py")
ATTESTATION = Path("output") / "provenance.json"
SYNTHETIC_NAME = re.compile(r"\bFIRM \d+\b")          # how thiel_common's synthetic generators name firms


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class GateReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, ok, detail))


# ---------------------------------------------------------------- checks
def check_provenance(root: Path) -> Check:
    """Every summary is stamped non-synthetic, or matches the attested hash of a known-real run;
    and no output carries the synthetic generators' firm names."""
    attested = {}
    if (root / ATTESTATION).is_file():
        attested = json.loads((root / ATTESTATION).read_text(encoding="utf-8")).get("summaries", {})
    bad = []
    summaries = sorted((root / "output").glob("h*_summary.json"))
    for f in summaries:
        prov = json.loads(f.read_text(encoding="utf-8")).get("_provenance")
        if prov is not None:
            if prov.get("synthetic") is not False:
                bad.append(f"{f.name} was written by a synthetic run")
        elif attested.get(f.name, {}).get("sha256") != sha256(f):
            bad.append(f"{f.name} has no provenance stamp and does not match {ATTESTATION.as_posix()}")
    for f in sorted((root / "output").glob("h*_summary.json")) + sorted((root / "output" / "tables").glob("*.csv")):
        if SYNTHETIC_NAME.search(f.read_text(encoding="utf-8", errors="ignore")):
            bad.append(f"{f.relative_to(root).as_posix()} contains synthetic firm names (FIRM <n>)")
    if not summaries:
        bad.append("no summaries in output/")
    return Check("provenance", not bad, "; ".join(bad) or f"{len(summaries)} summaries from live runs")


def check_git_clean(root: Path, allow_dirty: bool) -> Check:
    r = subprocess.run(["git", "status", "--porcelain", "--", *GATED_PATHS], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        return Check("git", False, "git status failed: " + r.stderr.strip())
    changed = [l[3:] for l in r.stdout.splitlines() if l.strip()]
    if changed and not allow_dirty:
        return Check("git", False, f"uncommitted changes in release inputs: {changed[:5]}; commit them, "
                                   "or pass --allow-dirty for a local trial release")
    return Check("git", True, "release inputs committed" if not changed else f"dirty, allowed: {changed[:5]}")


def _canonical(records) -> str:
    return json.dumps(json.loads(json.dumps(records, default=str)), sort_keys=True)


def check_candidates_current(root: Path, candidates: Path, settings: Settings) -> Check:
    """Re-derive the text candidates from today's sources; they must equal the files on disk."""
    from .ingest.notebooks import notebook_chunks
    from .ingest.notes import read_notes
    from .ingest.registry import build_registry
    from .sources import find_sources

    stale = []
    notes_pdf = find_sources(settings.sources_dir)["notes"]
    notes, corrections = read_notes(notes_pdf[0]) if len(notes_pdf) == 1 else ([], [])
    chunks = [c for nb in sorted((root / "notebooks").glob("*.ipynb")) for c in notebook_chunks(nb, root)[1]]
    claims, chapter_map, problems = build_registry(root, notes)
    if problems:
        stale.append("registry sources disagree: " + "; ".join(problems))
    expected = {
        "notes_chunks.jsonl": [asdict(x) for x in notes],
        "notes_corrections.jsonl": [asdict(x) for x in corrections],
        "notebook_chunks.jsonl": [asdict(x) for x in chunks],
        "claims_registry.json": [asdict(x) for x in claims],
        "chapter_map.json": [asdict(x) for x in chapter_map],
    }
    for name, want in expected.items():
        f = candidates / name
        if not f.is_file():
            stale.append(f"{name} missing")
            continue
        text = f.read_text(encoding="utf-8")
        have = [json.loads(l) for l in text.splitlines() if l.strip()] if name.endswith(".jsonl") else json.loads(text)
        if _canonical(have) != _canonical(want):
            stale.append(f"{name} is out of date")
    return Check("candidates", not stale, "; ".join(stale) + ("; rerun python -m rag ingest" if stale else "")
                 or "text candidates match the current sources")


def check_database(root: Path, candidates: Path) -> Check:
    """The database matches output/, and the schema cards and figure list match a fresh build."""
    import tempfile
    from .ingest.tables import build_database, verify_database
    db = candidates / "analysis.duckdb"
    if not db.is_file():
        return Check("database", False, "analysis.duckdb missing")
    problems = verify_database(root, db)
    with tempfile.TemporaryDirectory() as tmp:
        cards, figures, _ = build_database(root, Path(tmp) / "fresh.duckdb")
    for name, fresh in (("schema_cards.json", cards), ("figures.json", figures)):
        have = json.loads((candidates / name).read_text(encoding="utf-8"))
        if _canonical(have) != _canonical([asdict(x) for x in fresh]):
            problems.append(f"{name} does not match a fresh build from output/")
    return Check("database", not problems, "; ".join(problems[:3])
                 or "every table cell and summary value matches output/; schema cards and figures current")


def check_index(candidates: Path, settings: Settings) -> Check:
    from .ingest.index import INDEXES, all_docs, content_hash
    db = candidates / "indexes.duckdb"
    if not db.is_file():
        return Check("index", False, "indexes.duckdb missing")
    con = duckdb.connect(str(db), read_only=True)
    try:
        info = {r[0]: r[1:] for r in con.execute("SELECT index_name, model, n_docs, content_hash FROM index_info").fetchall()}
    finally:
        con.close()
    docs = all_docs(candidates)
    bad = []
    for name in INDEXES:
        part = [d for d in docs if d.index == name]
        if name not in info:
            bad.append(f"{name} index missing")
            continue
        model, n, h = info[name]
        if model != settings.embedding_model:
            bad.append(f"{name} built with {model}, settings pin {settings.embedding_model}")
        if (n, h) != (len(part), content_hash(part)):
            bad.append(f"{name} index is out of date with the candidate files")
    return Check("index", not bad, "; ".join(bad) or f"3 indexes match the candidates ({len(docs)} documents)")


def check_retrieval(candidates: Path, settings: Settings) -> Check:
    if not settings.openai_api_key:
        return Check("retrieval", False, "needs OPENAI_API_KEY to embed the test questions")
    from .eval.retrieval_smoke import run
    from .ingest.embed import EmbeddingCache, OpenAIEmbedder
    cache = EmbeddingCache(settings.build_dir / "cache" / "embeddings.duckdb")
    try:
        r = run(candidates / "indexes.duckdb", OpenAIEmbedder(settings.embedding_model, settings.openai_api_key, cache))
    finally:
        cache.close()
    n, t = r["n"], r["totals"]["hybrid"]
    ok = t[5] == n
    return Check("retrieval", ok, f"hybrid hit@1 {t[1]}/{n}, hit@5 {t[5]}/{n}" + ("" if ok else " (all must be in the top 5)"))


def run_gate(settings: Settings, root: Path = ROOT, allow_dirty: bool = False) -> GateReport:
    candidates = settings.build_dir / "candidates"
    report = GateReport()
    report.checks.append(check_provenance(root))
    report.checks.append(check_git_clean(root, allow_dirty))
    missing = [a for a in ARTIFACTS if not (candidates / a).is_file()]
    if missing:
        report.add("artifacts", False, f"missing {missing}; run python -m rag ingest")
        return report
    report.add("artifacts", True, f"all {len(ARTIFACTS)} candidate files present")
    report.checks.append(check_candidates_current(root, candidates, settings))
    report.checks.append(check_database(root, candidates))
    report.checks.append(check_index(candidates, settings))
    if report.passed:                                 # the only check that calls an API runs last
        report.checks.append(check_retrieval(candidates, settings))
    return report


# ---------------------------------------------------------------- publish and open
def _release_files(root: Path, candidates: Path) -> dict[str, Path]:
    files = {name: candidates / name for name in ARTIFACTS}
    for png in sorted((root / "output" / "figures").glob("*.png")):
        files[f"figures/{png.name}"] = png
    return files


def publish(settings: Settings, report: GateReport, root: Path = ROOT) -> dict:
    """Copy a passing build into build/releases/<release_id>/ and point CURRENT at it."""
    if not report.passed:
        raise RuntimeError("the release gate did not pass; nothing was published")
    candidates = settings.build_dir / "candidates"
    files = _release_files(root, candidates)
    hashes = {name: {"sha256": sha256(p), "bytes": p.stat().st_size} for name, p in sorted(files.items())}
    digest = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in hashes.items()}, sort_keys=True).encode()).hexdigest()
    release_id = f"r{datetime.now(timezone.utc):%Y%m%d}-{digest[:10]}"
    releases = settings.build_dir / "releases"
    existing = next((d for d in releases.glob(f"r*-{digest[:10]}") if (d / "manifest.json").is_file()), None) \
        if releases.is_dir() else None
    if existing:
        set_current(settings, existing.name)
        return {"release_id": existing.name, "path": str(existing), "new": False}

    target = releases / release_id
    tmp = releases / f".{release_id}.partial"
    if tmp.exists():
        shutil.rmtree(tmp, onerror=_force_remove)
    for name, src in files.items():
        dst = tmp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    claims = json.loads((candidates / "claims_registry.json").read_text(encoding="utf-8"))
    manifest = {
        "release_id": release_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_digest": digest,
        "git": git_state(root),
        "models": {"claude_provider": settings.claude_provider, "claude_model": settings.claude_model,
                   "embedding_model": settings.embedding_model},
        "sources": [asdict(s) for s in describe(settings.sources_dir)],
        "claims": [{"id": c["id"], "status": c["status"], "verdict": c["verdict"]} for c in claims],
        "gate": [asdict(c) for c in report.checks],
        "files": hashes,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in tmp.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    tmp.rename(target)
    set_current(settings, release_id)
    return {"release_id": release_id, "path": str(target), "new": True}


def _force_remove(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def set_current(settings: Settings, release_id: str) -> None:
    """Point new sessions at a release (also how to roll back). The release must open cleanly."""
    open_release(settings, release_id)
    releases = settings.build_dir / "releases"
    tmp = releases / "CURRENT.tmp"
    tmp.write_text(release_id + "\n", encoding="utf-8")
    os.replace(tmp, releases / "CURRENT")


def list_releases(settings: Settings) -> list[dict]:
    releases = settings.build_dir / "releases"
    current = current_release_id(settings)
    out = []
    for d in sorted(releases.glob("r*")) if releases.is_dir() else []:
        m = d / "manifest.json"
        if m.is_file():
            data = json.loads(m.read_text(encoding="utf-8"))
            out.append({"release_id": d.name, "created_utc": data["created_utc"], "git": data["git"],
                        "current": d.name == current})
    return out


def current_release_id(settings: Settings) -> str | None:
    f = settings.build_dir / "releases" / "CURRENT"
    return f.read_text(encoding="utf-8").strip() if f.is_file() else None


@dataclass(frozen=True)
class Release:
    release_id: str
    path: Path
    manifest: dict

    def file(self, name: str) -> Path:
        if name not in self.manifest["files"]:
            raise KeyError(f"{name} is not part of release {self.release_id}")
        return self.path / name


def open_release(settings: Settings, release_id: str | None = None) -> Release:
    """Open a release (CURRENT by default) after re-hashing every file against its manifest."""
    release_id = release_id or current_release_id(settings)
    if not release_id:
        raise FileNotFoundError("no release published yet; run python -m rag release")
    path = settings.build_dir / "releases" / release_id
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"release {release_id} not found")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    present = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {"manifest.json"}
    if present != set(manifest["files"]):
        raise ValueError(f"release {release_id} files differ from its manifest: "
                         f"extra {sorted(present - set(manifest['files']))}, missing {sorted(set(manifest['files']) - present)}")
    for name, meta in manifest["files"].items():
        if sha256(path / name) != meta["sha256"]:
            raise ValueError(f"release {release_id} has been modified: {name} does not match its manifest")
    return Release(release_id, path, manifest)
