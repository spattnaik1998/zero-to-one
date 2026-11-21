"""Command line for the Ask Zero to One build.

    python -m rag doctor            check keys, models, sources and analysis outputs
    python -m rag doctor --offline  same, but missing API keys are warnings (CI, no secrets)
    python -m rag manifest          write build/manifest.json for the current sources
    python -m rag ingest [notes|notebooks|registry|tables|index|all]
                                    build candidate files in build/candidates/
    python -m rag release [--check | --list | --promote ID | --pin ID] [--allow-dirty] [--skip-eval]
                                    gate a build and publish it as a release (not yet served);
                                    --promote serves a release whose evaluation passed; --pin rolls back
    python -m rag eval [--release ID] [--baseline] [--limit N] [--category C] [--workers N] [--max-cost USD]
                                    run the golden set (B1 agent, B0 baseline) and write build/eval/<id>/
                                    gate a candidate build and publish it as an immutable release
    python -m rag ask "question" [--json] [--release ID] [--effort low|medium|high|xhigh|max]
                                    answer a question with the agent on the CURRENT release;
                                    exits 1 unless the answer completed and passed its checks;
                                    every answer is written to the audit log first (exit 2 and no
                                    answer if it cannot be)
    python -m rag audit [--verify | --tail N | --show RECORD_ID]
                                    check the audit log's hash chain, list or show records
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
    g.add_argument("--promote", metavar="RELEASE_ID", help="serve a release whose evaluation passed")
    g.add_argument("--pin", metavar="RELEASE_ID", help="point CURRENT at an earlier release (rollback); "
                                                       "it must have passed its evaluation too")
    r.add_argument("--skip-eval", action="store_true",
                   help="promote or pin without a passing evaluation (local trials only; stated in the output)")
    e = sub.add_parser("eval", help="run the golden set on a release")
    e.add_argument("--release", metavar="RELEASE_ID", help="the release to evaluate (default CURRENT)")
    e.add_argument("--baseline", action="store_true", help="also run B0 (needed to promote)")
    e.add_argument("--limit", type=int, help="only the first N questions (a trial; cannot promote)")
    e.add_argument("--category", action="append", default=[], choices=list(("verdict", "numeric", "chapter",
                   "method", "cross", "adversarial")), help="only these categories (a trial; cannot promote)")
    e.add_argument("--workers", type=int, default=4, help="questions answered in parallel")
    e.add_argument("--max-cost", type=float, default=40.0, help="stop starting questions after this many USD")
    e.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    r.add_argument("--allow-dirty", action="store_true", help="allow uncommitted release inputs (local trials)")
    a = sub.add_parser("ask", help="answer a question with the agent")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full answer record as JSON")
    a.add_argument("--release", metavar="RELEASE_ID", help="a release other than CURRENT")
    a.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    a.add_argument("--user", help="who is asking (recorded in the audit log)")
    au = sub.add_parser("audit", help="check or read the audit log")
    ag = au.add_mutually_exclusive_group()
    ag.add_argument("--verify", action="store_true", help="re-walk the hash chain (the default)")
    ag.add_argument("--tail", type=int, metavar="N", help="list the last N records")
    ag.add_argument("--show", metavar="RECORD_ID", help="print one record as JSON")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return doctor(args.offline)
    settings = load_settings()
    if args.cmd == "ask":
        return ask(settings, args)
    if args.cmd == "audit":
        return audit(settings, args)
    if args.cmd == "eval":
        return evaluate(settings, args)
    if args.cmd == "release":
        return release(settings, args)
    if args.cmd == "manifest":
        print(f"wrote {sources.write_manifest(settings)}")
        return 0
    return ingest(settings, args.what)


def audit(settings, args) -> int:
    import json as _json
    from .agent.audit import AuditLog
    log = AuditLog(settings.audit_dir)
    if args.show:
        r = log.find(args.show)
        if r is None:
            print(f"no record {args.show} in {settings.audit_dir}")
            return 1
        print(_json.dumps(r, indent=2, ensure_ascii=False))
        return 0
    if args.tail is not None:
        rows = [r for _, _, r in log.records() if isinstance(r, dict)][-args.tail:] if args.tail > 0 else []
        for r in rows:
            status = "error" if r.get("event") == "error" else ("FLAGGED" if r.get("flagged") else "passed")
            print(f"#{r.get('seq', '?')!s:<5} {r.get('ts', '?')}  {r.get('record_id', '?')}  {status:7}  "
                  f"{str(r.get('question', ''))[:70]}")
        print(f"{len(rows)} record(s) shown from {settings.audit_dir}")
        return 0
    problems = log.verify()
    count = sum(1 for _ in log.records())
    for p in problems:
        print(f"PROBLEM  {p}")
    print(f"{count} record(s) in {settings.audit_dir}: " + ("chain intact" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


def ask(settings, args) -> int:
    import json as _json
    import sys as _sys
    from .agent import orchestrator
    from .agent.audit import AuditError, AuditLog, ask_audited
    session = orchestrator.open_session(settings, args.release, effort=args.effort)
    try:
        checked, record = ask_audited(session, args.question, AuditLog(settings.audit_dir), user=args.user)
    except AuditError as e:
        print(f"answer withheld: {e}", file=_sys.stderr)
        return 2
    a, v = checked.answer, checked.verification
    good = a.ok and v.passed
    if args.json:
        print(_json.dumps({**checked.to_dict(), "audit_record_id": record["record_id"]},
                          indent=2, ensure_ascii=False, default=str))
        return 0 if good else 1
    if checked.flagged:
        how = ("after one revision" if checked.revised else
               f"and could not be revised ({checked.revision_error})" if checked.revision_error else "")
        print(f"WARNING: this answer failed the automated checks {how}:".replace("  ", " ").replace(" :", ":"))
        for x in v.errors:
            print(f"  - [{x.rule}] {x.detail}")
        print()
    print(a.text or f"(no answer; stop reason {a.stop_reason})")
    if a.citations:
        print()
        print("Sources:")
        for c in dict.fromkeys((c.source_id, c.title) for c in a.citations):
            print(f"  [{c[0]}] {c[1] or ''}")
    calls = ", ".join(f"{t.name}{'' if t.ok else ' (error)'}" for t in checked.all_tool_calls) or "none"
    u = checked.usage
    print()
    print(f"{a.turns} model turn(s) in the final attempt; tools: {calls}; release {a.release_id}; model {a.model}; "
          f"tokens in {u['input_tokens']} (cache read {u['cache_read_input_tokens']}), "
          f"out {u['output_tokens']}" + ("; budget exhausted" if a.budget_exhausted else ""))
    print(f"checks: {'passed' if v.passed else 'FAILED'} ({v.numbers_supported}/{v.numbers_checked} numbers "
          f"supported" + ("; revised once" if checked.revised else "") + ")"
          + "".join(f"\n  warning: {x.detail}" for x in v.violations if x.severity == "warning"))
    print(f"audit record: {record['record_id']} (#{record['seq']})")
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
    if args.promote or args.pin:
        from .eval.run import promotable
        target = args.promote or args.pin
        ok, why = promotable(settings, target)
        if not ok and not args.skip_eval:
            print(f"not served: {why}")
            return 1
        set_current(settings, target)
        print(f"CURRENT -> {target} ({why}" + ("; evaluation SKIPPED by --skip-eval" if not ok else "") + ")")
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
    r = publish(settings, report, make_current=False)
    print(f"{'published' if r['new'] else 'already published'} {r['release_id']} -> {r['path']} (not served yet)")
    print(f"next: python -m rag eval --release {r['release_id']} --baseline, "
          f"then python -m rag release --promote {r['release_id']}")
    return 0


def evaluate(settings, args) -> int:
    import anthropic

    from .agent import orchestrator
    from .eval.run import Baseline, evaluate as run_eval, print_report, report_dir
    session = orchestrator.open_session(settings, args.release, effort=args.effort)
    baseline = Baseline(session.toolset, anthropic.Anthropic(api_key=settings.anthropic_api_key), settings.claude_model,
                        effort=args.effort) if args.baseline else None
    report = run_eval(session, baseline=baseline, limit=args.limit, categories=tuple(args.category),
                      workers=args.workers, max_cost=args.max_cost, settings=settings,
                      out_dir=report_dir(settings, session.release_id))
    print_report(report)
    print(f"report: {report_dir(settings, session.release_id) / 'report.json'}")
    return 0 if report["decision"]["passed"] else 1


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
