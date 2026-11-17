"""Command line for the Ask Zero to One build.

    python -m rag doctor            check keys, models, sources and analysis outputs
    python -m rag doctor --offline  same, but missing API keys are warnings (CI, no secrets)
    python -m rag manifest          write build/manifest.json for the current sources
    python -m rag ingest [notes|notebooks|registry|all]
                                    build candidate files in build/candidates/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

from . import sources
from .config import ROOT, env_conflicts, find_dotenv, load_settings

PACKAGES = ["anthropic", "openai", "duckdb", "pdfplumber", "nbformat", "rank_bm25"]


def doctor(offline: bool) -> int:
    rows: list[tuple[str, str, str]] = []
    add = lambda status, name, detail: rows.append((status, name, detail))  # noqa: E731

    add("OK" if sys.version_info >= (3, 11) else "FAIL", "python", sys.version.split()[0])
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    add("FAIL" if missing else "OK", "packages",
        "missing: " + ", ".join(missing) + " (pip install -r requirements-rag.txt)" if missing else "all present")

    env_file = find_dotenv()
    add("OK" if env_file else "WARN", ".env", str(env_file) if env_file else "none found; copy .env.example")
    for name in env_conflicts():
        add("WARN", name, "set in both the environment and .env with different values; the environment's "
                          "value is used. Remove the stale environment variable (or the .env line) and restart.")
    try:
        s = load_settings()
    except ValueError as e:
        add("FAIL", "settings", str(e))
        return _report(rows)

    key_missing = "WARN" if offline else "FAIL"
    add("OK", "claude", f"{s.claude_model} via {s.claude_provider}")
    if s.claude_provider == "anthropic":
        add("OK" if s.anthropic_api_key else key_missing, "ANTHROPIC_API_KEY", "set" if s.anthropic_api_key else "missing")
    else:
        add("WARN", "provider credentials", f"{s.claude_provider} uses cloud credentials; not checked here")
    add("OK" if s.openai_api_key else key_missing, "OPENAI_API_KEY", "set" if s.openai_api_key else "missing")
    add("OK" if s.embedding_model else "FAIL", "OPENAI_EMBEDDING_MODEL",
        s.embedding_model or "missing; pin an exact model name")

    found = sources.find_sources(s.sources_dir)
    notes = found["notes"]
    add("OK" if notes else "FAIL", "notes source", f"{len(notes)} PDF in {s.sources_dir.name}/notes")

    have = [n for n in range(1, 9) if (ROOT / "output" / f"h{n}_summary.json").is_file()]
    add("OK" if len(have) == 8 else "WARN", "analysis outputs",
        f"{len(have)}/8 summaries in output/" + ("" if len(have) == 8 else "; run the notebooks"))
    return _report(rows)


def _report(rows) -> int:
    width = max(len(name) for _, name, _ in rows)
    for status, name, detail in rows:
        print(f"[{status:<4}] {name:<{width}}  {detail}")
    failed = sum(status == "FAIL" for status, _, _ in rows)
    print(f"\n{failed} failed, {sum(s == 'WARN' for s, _, _ in rows)} warnings")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rag", description="Ask Zero to One build tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check that the build can run")
    d.add_argument("--offline", action="store_true", help="treat missing API keys as warnings")
    sub.add_parser("manifest", help="write build/manifest.json")
    i = sub.add_parser("ingest", help="build candidate files in build/candidates/")
    i.add_argument("what", nargs="?", choices=["notes", "notebooks", "registry", "all"], default="all")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return doctor(args.offline)
    settings = load_settings()
    if args.cmd == "manifest":
        print(f"wrote {sources.write_manifest(settings)}")
        return 0
    return ingest(settings, args.what)


def ingest(settings, what: str) -> int:
    from .ingest.run import RegistryError, ingest_notebooks, ingest_notes, ingest_registry
    if what in ("notes", "all"):
        r = ingest_notes(settings)
        print(f"notes: {r['bullets']} bullets ({r['opinion_bullets']} opinion), {r['corrections']} corrections; "
              f"chapters without notes: {r['chapters_without_notes'] or 'none'}")
        for out in r["output"]:
            print(f"  -> {out}")
    if what in ("notebooks", "all"):
        r = ingest_notebooks(settings)
        print(f"notebooks: {r['chunks']} chunks {r['kinds']}, {r['thiel_quotes']} Thiel quotations; "
              f"per hypothesis {r['per_hypothesis']}")
        print(f"  -> {r['output']}")
    if what in ("registry", "all"):
        try:
            r = ingest_registry(settings)
        except RegistryError as e:
            print(f"registry: FAILED, {e}")
            return 1
        print("registry: " + ", ".join(f"{cid} {status}" + (f" ({verdict})" if verdict else "")
                                       for cid, status, verdict in r["claims"]))
        print(f"  untested chapters: {r['untested_chapters']}"
              + ("" if r["notes_linked"] else "; notes not found, so no note links"))
        for out in r["output"]:
            print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
