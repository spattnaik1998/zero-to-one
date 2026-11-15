"""Command line for the Ask Zero to One build.

    python -m rag doctor            check keys, models, sources and analysis outputs
    python -m rag doctor --offline  same, but missing API keys are warnings (CI, no secrets)
    python -m rag manifest          write build/manifest.json for the current sources
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

from . import sources
from .config import ROOT, find_dotenv, load_settings

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
    book, notes = found["book"], found["notes"]
    add("OK" if len(book) == 1 else ("FAIL" if not book else "WARN"), "book source",
        f"{len(book)} PDF in {s.sources_dir.name}/book" + ("" if len(book) == 1 else " (expected exactly 1)"))
    add("OK" if notes else "FAIL", "notes source", f"{len(notes)} PDF in {s.sources_dir.name}/notes")
    add("OK" if s.book_licensed else "WARN", "book licence",
        "confirmed" if s.book_licensed else "not confirmed: development only, client builds blocked (phase 0)")

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
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return doctor(args.offline)
    path = sources.write_manifest(load_settings())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
