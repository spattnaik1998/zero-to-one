"""Command line for the Ask Zero to One build.

    python -m rag doctor            check keys, models, sources and analysis outputs
    python -m rag doctor --offline  same, but missing API keys are warnings (CI, no secrets)
    python -m rag manifest          write build/manifest.json for the current sources
    python -m rag ingest [notes|notebooks|registry|tables|index|all]
                                    build candidate files in build/candidates/
    python -m rag release [--check | --list | --pin ID] [--allow-dirty]
                                    gate a candidate build and publish it as an immutable release
    python -m rag ask "question" [--json] [--release ID] [--effort low|medium|high|xhigh|max]
                                    answer a question with the agent on the CURRENT release;
                                    exits 1 unless the answer completed (refusal, max_tokens, ...)
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
    i.add_argument("what", nargs="?", choices=["notes", "notebooks", "registry", "tables", "index", "all"], default="all")
    r = sub.add_parser("release", help="run the release gate and publish an immutable corpus version")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="run the gate only; publish nothing")
    g.add_argument("--list", action="store_true", help="list published releases")
    g.add_argument("--pin", metavar="RELEASE_ID", help="point CURRENT at an existing release (rollback)")
    r.add_argument("--allow-dirty", action="store_true", help="allow uncommitted release inputs (local trials)")
    a = sub.add_parser("ask", help="answer a question with the agent")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full answer record as JSON")
    a.add_argument("--release", metavar="RELEASE_ID", help="a release other than CURRENT")
    a.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return doctor(args.offline)
    settings = load_settings()
    if args.cmd == "ask":
        return ask(settings, args)
    if args.cmd == "release":
        return release(settings, args)
    if args.cmd == "manifest":
        print(f"wrote {sources.write_manifest(settings)}")
        return 0
    return ingest(settings, args.what)


def ask(settings, args) -> int:
    import json as _json
    from .agent import orchestrator
    from .agent.verifier import ask_checked
    session = orchestrator.open_session(settings, args.release, effort=args.effort)
    checked = ask_checked(session, args.question)
    a, v = checked.answer, checked.verification
    good = a.ok and v.passed
    if args.json:
        print(_json.dumps(checked.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0 if good else 1
    if checked.flagged and a.ok:
        print("WARNING: this answer failed the automated checks after one revision:")
        for x in v.errors:
            print(f"  - [{x.rule}] {x.detail}")
        print()
    print(a.text or f"(no answer; stop reason {a.stop_reason})")
    if a.citations:
        print()
        print("Sources:")
        for c in dict.fromkeys((c.source_id, c.title) for c in a.citations):
            print(f"  [{c[0]}] {c[1] or ''}")
    calls = ", ".join(f"{t.name}{'' if t.ok else ' (error)'}" for t in a.tool_calls) or "none"
    print()
    print(f"{a.turns} model turn(s); tools: {calls}; release {a.release_id}; model {a.model}; "
          f"tokens in {a.usage['input_tokens']} (cache read {a.usage['cache_read_input_tokens']}), "
          f"out {a.usage['output_tokens']}" + ("; budget exhausted" if a.budget_exhausted else ""))
    print(f"checks: {'passed' if v.passed else 'FAILED'} ({v.numbers_supported}/{v.numbers_checked} numbers "
          f"supported" + ("; revised once" if checked.revised else "") + ")"
          + "".join(f"\n  warning: {x.detail}" for x in v.violations if x.severity == "warning"))
    return 0 if good else 1


def release(settings, args) -> int:
    from .release import list_releases, publish, run_gate, set_current
    if args.list:
        rows = list_releases(settings)
        for r in rows:
            print(f"{'*' if r['current'] else ' '} {r['release_id']}  {r['created_utc']}  git {str(r['git']['commit'])[:7]}"
                  + ("  (dirty)" if r["git"]["dirty"] else ""))
        print(f"{len(rows)} release(s); * = CURRENT" if rows else "no releases yet")
        return 0
    if args.pin:
        set_current(settings, args.pin)
        print(f"CURRENT -> {args.pin}")
        return 0
    report = run_gate(settings, allow_dirty=args.allow_dirty)
    for c in report.checks:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name:<11} {c.detail}")
    if not report.passed:
        print("gate FAILED: nothing published")
        return 1
    if args.check:
        print("gate passed (check only; nothing published)")
        return 0
    r = publish(settings, report)
    print(f"{'published' if r['new'] else 'already published'} {r['release_id']} -> {r['path']}; CURRENT updated")
    return 0


def ingest(settings, what: str) -> int:
    from .ingest.run import (IndexBuildError, RegistryError, TablesError, ingest_index, ingest_notebooks,
                             ingest_notes, ingest_registry, ingest_tables)
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
    if what in ("tables", "all"):
        try:
            r = ingest_tables(settings)
        except TablesError as e:
            print(f"tables: FAILED, {e}")
            return 1
        print(f"tables: {r['tables']} tables ({r['rows']} rows), {r['summary_values']} summary values, "
              f"{r['figures']} figures; verified against output/ with no differences")
        print(f"  captions: {r['caption_sources']}; unresolved: {r['unresolved_captions'] or 'none'}; "
              f"tables not traced to a notebook cell: {r['untraced_tables'] or 'none'}")
        for out in r["output"]:
            print(f"  -> {out}")
    if what in ("index", "all"):
        try:
            r = ingest_index(settings)
        except IndexBuildError as e:
            print(f"index: FAILED, {e}")
            return 1
        print(f"index: {r['counts']} documents, {r['dims']}-dim {r['model']} vectors; "
              f"{r['api_calls']} API call(s), {r['tokens']} tokens, {r['cache_hits']} from cache")
        print(f"  -> {r['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
