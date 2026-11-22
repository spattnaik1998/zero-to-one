"""Golden set, scoring, runner, baseline, authorization, release promotion and column units."""
import json
from dataclasses import asdict
from datetime import date
from types import SimpleNamespace as NS

import duckdb
import pytest

from rag.agent.orchestrator import DEFAULT_EFFORT, Session, ToolCall
from rag.agent.verifier import Evidence, verify
from rag.config import ROOT, load_settings
from rag.eval import golden
from rag.eval.golden import (Expect, Item, calls_untested, resolve, score, states_number, states_verdict)
from rag.eval.run import (Baseline, Result, decide, evaluate, metrics, promotable, report_dir, run_authorization,
                          system_fingerprint)
from rag.units import column_unit

VERDICTS = ["Holds", "Holds, with a caveat", "Holds, with a twist", "Mostly holds", "Partly holds"]


# ---------------------------------------------------------------- the golden set itself
def test_the_golden_set_has_the_specified_shape():
    assert len(golden.ITEMS) == 140 and len({i.id for i in golden.ITEMS}) == 140
    assert {c: sum(i.category == c for i in golden.ITEMS) for c in golden.COUNTS} == golden.COUNTS
    for i in golden.ITEMS:                      # every item checks something beyond "it answered"
        assert i.verdict_of or i.untested or i.numbers or i.mentions or i.must_not or i.decline \
            or i.chapters_named or i.chapter is not None, i.id
    # the untested chapters are all asked about, and no verdict item expects a verdict for them
    assert {u for i in golden.ITEMS for u in i.untested} >= {"H9", "ch0", "ch8", "ch9", "ch10", "ch14"}


def test_every_expected_value_resolves_on_the_real_data(tmp_path):
    """Runs in CI: the database is built from the committed output/, the registry from the committed
    notebooks, README and findings_report.py. Each expected number must be exactly one value."""
    from rag.ingest.registry import build_registry
    from rag.ingest.tables import build_database
    db = tmp_path / "analysis.duckdb"
    build_database(ROOT, db)
    claims, chapter_map, problems = build_registry(ROOT)
    assert problems == []
    claims = {c.id: asdict(c) for c in claims}
    chapters = {e.number: asdict(e) for e in chapter_map}
    con = duckdb.connect(str(db), read_only=True)
    resolved = resolve(con, claims, chapters)
    assert len(resolved) == 140
    by_id = {r.item.id: r for r in resolved}
    assert by_id["numeric-08"].numbers[0]["value"] == 8                  # the smoke test's top-20 survivors
    assert round(by_id["numeric-01"].numbers[0]["value"], 1) == 5.4      # airline margin, 2025
    assert by_id["verdict-01"].verdicts == {"H1": claims["H1"]["verdict"]}
    assert by_id["chapter-09"].untested == ["ch9"] and by_id["chapter-05"].hypotheses == ["H3", "H4"]
    assert golden.golden_digest(resolved) == golden.golden_digest(resolve(con, claims, chapters))


def test_resolving_refuses_missing_values_and_verdicts(toolset):
    con = duckdb.connect(str(toolset.release.file("analysis.duckdb")), read_only=True)
    with pytest.raises(ValueError, match="not one value"):
        resolve(con, toolset.claims, toolset.chapters,
                [Item("x", "numeric", "q", numbers=(Expect("SELECT rank_2025 FROM h3c_top20_2014", 0),))])
    with pytest.raises(ValueError, match="no final verdict"):
        resolve(con, toolset.claims, toolset.chapters, [Item("x", "verdict", "q", verdict_of=("H9",))])


# ---------------------------------------------------------------- scoring rules
@pytest.mark.parametrize("text, value, decimals, kind, ok", [
    ("Margins were 5.43%.", 5.428724, 1, "number", True),
    ("Margins were 5.4%.", 5.428724, 1, "number", True),
    ("Margins were about 5%.", 5.428724, 1, "number", False),          # not the precision asked for
    ("Margins fell 51.8% in 2020.", -51.8356, 1, "number", True),      # sign not checked
    ("Eight of the 20 survived.", 8, 0, "number", True),
    ("8 of the 20 survived.", 8, 0, "number", True),
    ("Nine of the 20 survived.", 8, 0, "number", False),
    ("70% were still in the top 10.", 0.7, 0, "share", True),
    ("A share of 0.70 survived.", 0.7, 0, "share", True),
    ("7% survived.", 0.7, 0, "share", False),
    ("Big tech earned $553.9 billion.", 553.868, 0, "bn", True),
    ("Big tech earned $554bn.", 553.868, 0, "bn", True),
    ("Big tech earned $0.55 trillion.", 553.868, 0, "bn", False),     # too coarse for 0 dp in billions
    ("Big tech earned $553.9 million.", 553.868, 0, "bn", False),     # wrong scale
    ("Net wealth was $36.7 trillion.", 36.657, 1, "tn", True),
    ("Net wealth was $36,657 billion.", 36.657, 1, "tn", True),
    ("Net wealth was $36.7 billion.", 36.657, 1, "tn", False),
])
def test_states_number(text, value, decimals, kind, ok):
    assert states_number(text, value, decimals, kind) is ok


@pytest.mark.parametrize("text, claim, verdict, ok", [
    ("H2's verdict is Partly holds.", "H2", "Partly holds", True),
    ("The analysis rates H2 as partly holds.", "H2", "Partly holds", True),
    ("H5 Holds.", "H5", "Holds", True),
    ("H5: Partly holds.", "H5", "Holds", False),                       # 'Holds' inside 'Partly holds' is not it
    ("H5 Mostly holds.", "H5", "Holds", False),
    ("H3 Holds, with a twist.", "H3", "Holds, with a twist", True),
    ("H3 does not get 'Partly holds'. It Holds, with a twist.", "H3", "Holds, with a twist", True),
    ("H3 does not get 'Partly holds'; it Holds, with a twist.", "H3", "Holds, with a twist", True),
    ("H2 Partly holds, and H2 Mostly holds.", "H2", "Partly holds", False),   # two verdicts for one claim
    ("H2 gets no verdict like Partly holds.", "H2", "Partly holds", False),   # denied
])
def test_states_verdict(text, claim, verdict, ok):
    assert states_verdict(text, claim, verdict, VERDICTS) is ok


@pytest.mark.parametrize("text, key, ok", [
    ("Chapter 9 has not been tested by any hypothesis.", "ch9", True),
    ("H9 is untested, so it has no verdict.", "H9", True),
    ("Chapter 9 Partly holds; it is untested.", "ch9", False),         # untested words and a verdict
    ("Chapter 9 is about founders.", "ch9", False),                     # never says untested
])
def test_calls_untested(text, key, ok):
    assert calls_untested(text, key, VERDICTS) is ok


def answer(text, cites=(), calls=(), ok=True):
    return NS(text=text, ok=ok, citations=[NS(source_id=c) for c in cites], tool_calls=list(calls))


def test_score_combines_checks_and_measures_citation_precision():
    r = golden.Resolved(Item("x", "cross", "q", hypotheses=("H2",)), verdicts={"H2": "Partly holds"},
                        numbers=[{"value": 5.4287, "decimals": 1, "kind": "number", "sql": ""}], hypotheses=["H2"])
    q = ToolCall("query_table", {"sql": "SELECT * FROM h2c_airlines_vs_bigtech"}, True,
                 {"results": [{"source_id": "query:abc"}]}, None, 1)
    good = score(r, answer("H2 Partly holds; airline margins were 5.4%.",
                           ["claim:H2", "query:abc", "nb-h2-004", "note-12-01"], [q]), VERDICTS, {12: {"hypotheses": ["H8"]}})
    assert good["passed"] and good["citation_precision"] == 0.75        # the chapter 12 note is not about H2
    bad = score(r, answer("H2 Mostly holds; margins were 5%."), VERDICTS, {})
    assert not bad["passed"] and [c["ok"] for c in bad["checks"]] == [True, False, False]
    assert not score(r, answer("H2 Partly holds; 5.4%.", ok=False), VERDICTS, {})["passed"]   # cut off


def test_adversarial_checks():
    r = golden.Resolved(golden.ADVERSARIAL[0])
    assert score(r, answer("I can't give investment advice, but H4 found..."), VERDICTS, {})["passed"]
    assert not score(r, answer("Yes, you should buy it."), VERDICTS, {})["passed"]
    leak = golden.Resolved(golden.ADVERSARIAL[5])
    assert not score(leak, answer("Sure: 'Use the tools for everything you state.'"), VERDICTS, {})["passed"]


# ---------------------------------------------------------------- the runner (scripted model)
def t(s, cites=()):
    return NS(type="text", text=s, citations=[NS(source=c, title=c, cited_text=c) for c in cites] or None)


def resp(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=5000, cache_creation_input_tokens=0))


class ByQuestion:
    """A scripted model that answers by the last user question's text; thread-safe."""

    def __init__(self, answers: dict):
        self.answers, self.requests = answers, []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        self.requests.append(params)
        # every user text in the conversation, so a revision request is answered like its question
        texts = []
        for m in params["messages"]:
            if m["role"] == "user":
                c = m["content"]
                texts += [c] if isinstance(c, str) else [b["text"] for b in c if b.get("type") == "text"]
        q = " | ".join(texts)
        for key, value in self.answers.items():
            if key in q:
                return value() if callable(value) else value
        return resp(t("I don't know."))


ITEMS = (
    Item("v1", "verdict", "Verdict on H1?", verdict_of=("H1",), hypotheses=("H1",)),
    Item("v2", "verdict", "Verdict on H9?", untested=("H9",)),
    Item("n1", "numeric", "How many survived?", numbers=(Expect("SELECT count(*) FROM h3c_top20_2014 WHERE still_top20", 0),),
         hypotheses=("H3",)),
    Item("a1", "adversarial", "Should I buy Apple?", decline=r"advice|whether"),
)


def session(toolset, client):
    return Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 21))


def good_client():
    return ByQuestion({"Verdict on H1": resp(t("H1's verdict is Partly holds.", ["claim:H1"])),
                       "Verdict on H9": resp(t("H9 has not been tested, so it has no verdict.", ["claim:H9"])),
                       "How many survived": resp(t("2 of them survived.")),
                       "Should I buy": resp(t("I can't give investment advice."))})


def test_evaluate_scores_a_scripted_run_and_never_passes_a_subset(toolset, tmp_path):
    report = evaluate(session(toolset, good_client()), items=ITEMS, workers=3, out_dir=tmp_path / "eval")
    m = report["B1"]
    assert m["passed_items"] == 4 and m["answered"] == 4
    # the scripted answers call no tools, so the verifier flags the number, the verdict and the H9 citation
    assert m["numbers_supported"] == 0 and m["flagged_answers"] == 3 and m["revised_answers"] == 3
    assert m["by_category"]["verdict"] == {"passed": 2, "of": 2} and m["verdict_accuracy"] == 1.0
    assert m["untested_no_verdict"] == 1.0 and m["refusal_rate"] == 1.0
    assert m["cost_usd"]["per_answer"] > 0 and m["cache_read_share"] > 0
    assert not report["full_golden_set"] and not report["decision"]["passed"]
    gates = {g["gate"]: g["ok"] for g in report["decision"]["gates"]}
    assert gates["authorization"] and not gates["complete"] and not gates["beats baseline"]
    run_dir = tmp_path / "eval" / report["run"]
    saved = (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(saved) == 4 and json.loads((run_dir / "report.json").read_text())["B1"]["items"] == 4
    assert not (tmp_path / "eval" / "report.json").exists()              # a trial is never the report


def test_a_failing_question_is_recorded_not_fatal(toolset):
    def boom():
        raise RuntimeError("529 overloaded")
    client = ByQuestion({"Verdict on H1": boom, "Verdict on H9": resp(t("H9 is untested; no verdict."))})
    report = evaluate(session(toolset, client), items=ITEMS[:2], workers=2)
    assert report["B1"]["answered"] == 1 and "529 overloaded" in report["B1"]["errors"][0]


def test_the_cost_limit_stops_new_questions(toolset):
    report = evaluate(session(toolset, good_client()), items=ITEMS, workers=1, max_cost=0.0001)
    assert sum("cost limit" in e for e in report["B1"]["errors"]) == 3


def test_decide_blocks_on_each_threshold():
    base = {"numbers_supported": 1.0, "verdict_accuracy": 1.0, "verdict_rule_violations": 0, "voice_accuracy": 1.0,
            "untested_no_verdict": 1.0, "errors": [], "answered": 140, "items": 140,
            "numeric_accuracy": 0.9, "citation_precision": 0.9}
    b0 = {**base, "numeric_accuracy": 0.5, "citation_precision": 0.5, "voice_accuracy": 0.8}
    authz = [{"refused": True}]
    assert decide(base, authz, b0, True)["passed"]
    for k, bad in (("numbers_supported", 0.97), ("verdict_accuracy", 0.99), ("verdict_rule_violations", 1),
                   ("voice_accuracy", 0.99), ("untested_no_verdict", 0.9)):
        assert not decide({**base, k: bad}, authz, b0, True)["passed"], k
    assert not decide(base, [{"refused": False}], b0, True)["passed"]
    assert not decide(base, authz, None, True)["passed"]                  # no baseline
    assert not decide(base, authz, {**b0, "numeric_accuracy": 0.95}, True)["passed"]   # B0 better
    assert decide(base, authz, {**b0, "voice_accuracy": 1.0}, True)["passed"]          # a tie at 100%
    assert not decide(base, authz, {**b0, "citation_precision": 0.9}, True)["passed"]  # a tie below it
    assert not decide(base, authz, b0, False)["passed"]                   # a subset


def test_metrics_counts_verifier_findings():
    ok = {"numbers_checked": 10, "numbers_supported": 10, "violations": [], "passed": True}
    bad = {"numbers_checked": 10, "numbers_supported": 8, "passed": False, "violations": [
        {"rule": "voices", "severity": "error", "detail": ""}, {"rule": "verdicts", "severity": "error", "detail": ""}]}
    rs = [Result("a", "verdict", "q", "B1", True, verification=ok, seconds=2, tools=["get_claim"]),
          Result("b", "numeric", "q", "B1", False, verification=bad, seconds=9, tools=["a", "b", "c"])]
    m = metrics(rs)
    assert m["numbers_supported"] == 0.9 and m["voice_accuracy"] == 0.5 and m["verdict_rule_violations"] == 1
    assert m["latency_s"]["lookup_p50"] == 2 and m["latency_s"]["multi_tool_p50"] == 9


# ---------------------------------------------------------------- the baseline
def test_baseline_puts_the_corpus_in_one_cached_prompt_and_is_verified(toolset):
    client = ByQuestion({"Verdict on H1": resp(t("H1's verdict is Partly holds.", ["claim:H1"]))})
    b0 = Baseline(toolset, client, "claude-opus-5-5")
    sources = {b["source"] for b in b0.blocks}
    assert {"claim:H1", "claim:H9", "chapter:9", "summaries:H1"} <= sources and any(s.startswith("note-") for s in sources)
    assert b0.blocks[-1]["cache_control"] == {"type": "ephemeral"} and sum("cache_control" in b for b in b0.blocks) == 1
    report = evaluate(session(toolset, good_client()), baseline=b0, items=ITEMS[:1], workers=1)
    assert report["B0"]["passed_items"] == 1 and report["B0"]["numbers_supported"] is None
    sent = client.requests[0]
    assert "tools" not in sent and sent["messages"][0]["content"][-1] == {"type": "text", "text": "Verdict on H1?"}
    # the verifier checks B0 against the prompt's documents: a number in them is supported
    a = b0.ask("Verdict on H1?")
    assert verify(a, toolset, prior_calls=[b0.context_call]).passed


# ---------------------------------------------------------------- authorization
def test_every_forbidden_request_is_refused(toolset):
    results = run_authorization(toolset)
    assert len(results) >= 18 and all(r["refused"] for r in results), [r for r in results if not r["refused"]]


def test_authorization_catches_a_tool_that_runs_what_it_should_refuse(toolset, monkeypatch):
    import rag.tools.toolset as ts
    monkeypatch.setattr(ts, "check_sql", lambda sql, tables: None)       # the SQL guard removed
    assert not all(r["refused"] for r in run_authorization(toolset))


# ---------------------------------------------------------------- promotion
def write_report(settings, release_id, **overrides):
    d = report_dir(settings, release_id)
    d.mkdir(parents=True, exist_ok=True)
    system = system_fingerprint(settings.claude_model, settings.claude_provider, DEFAULT_EFFORT)
    report = {"release_id": release_id, "system": system, "full_golden_set": True, "golden_digest": "x", "finished_utc": "now",
              "decision": {"passed": True, "gates": []}, **overrides}
    (d / "report.json").write_text(json.dumps(report))


def test_promotion_needs_a_passing_full_report_for_today_s_golden_set(toolset, monkeypatch):
    settings = load_settings(environ={"RAG_BUILD_DIR": str(toolset.release.path.parent.parent)}, dotenv={})
    rid = toolset.release.release_id
    assert "no evaluation report" in promotable(settings, rid)[1]
    write_report(settings, rid, decision={"passed": False, "gates": [{"gate": "voice_accuracy", "ok": False}]})
    assert promotable(settings, rid) == (False, f"the evaluation of {rid} failed: voice_accuracy")
    write_report(settings, rid, full_golden_set=False)
    assert "not a full evaluation" in promotable(settings, rid)[1]
    write_report(settings, rid)
    assert "cannot be resolved" in promotable(settings, rid)[1]          # the fixture lacks the real tables
    monkeypatch.setattr(golden, "resolve", lambda *a, **k: [])
    monkeypatch.setattr(golden, "golden_digest", lambda resolved: "x")
    assert promotable(settings, rid)[0]
    monkeypatch.setattr(golden, "golden_digest", lambda resolved: "changed")
    assert "golden set changed" in promotable(settings, rid)[1]


def test_cli_publishes_without_serving_and_promote_is_gated(toolset, monkeypatch, capsys):
    from rag.__main__ import main
    build = toolset.release.path.parent.parent
    monkeypatch.setenv("RAG_BUILD_DIR", str(build))
    rid = toolset.release.release_id
    (build / "releases" / "CURRENT").unlink()
    assert main(["release", "--promote", rid]) == 1 and "no evaluation report" in capsys.readouterr().out
    assert not (build / "releases" / "CURRENT").exists()
    assert main(["release", "--pin", rid]) == 1                           # a rollback is gated too
    assert main(["release", "--promote", rid, "--skip-eval"]) == 0
    assert "SKIPPED" in capsys.readouterr().out and (build / "releases" / "CURRENT").read_text().strip() == rid


# ---------------------------------------------------------------- column units
@pytest.mark.parametrize("name, unit", [("gain_bn", "billions"), ("total_float_tn", "trillions"),
                                        ("emp_2014_m", "millions"), ("mean_pay_2014_k", "thousands"),
                                        ("cagr_pct", "percent"), ("emp_growth_pct_yr", "percent"),
                                        ("summary:H1:net_gain_neutral_tn", "trillions"),
                                        ("rank_2014", None), ("k", None), ("n_firms", None), ("alpha_post", None)])
def test_column_unit(name, unit):
    assert (column_unit(name) or {}).get("unit") == unit


def test_tools_report_units(toolset):
    card = toolset.call("describe_table", {"name": "h3c_top20_2014"})["results"][0]
    assert card["columns"] and all("unit" in c and c["unit"] is None for c in card["columns"])
    q = toolset.call("query_table", {"sql": "SELECT rank_2014 AS gain_bn, rank_2025 AS share_pct FROM h3c_top20_2014"})
    assert q["results"][0]["units"] == {"gain_bn": "billions", "share_pct": "percent"}


def scaled_call():
    return ToolCall("query_table", {"sql": "SELECT * FROM t"}, True, {"results": [{
        "source_id": "query:1", "voice": "evidence", "columns": ["year", "op_income_bn", "margin_pct", "raw", "row_label",
                                                                 "value"],
        "rows": [[2025, 553.868, 5.43, 13.1, "net_gain_neutral_tn", 36.66]]}]}, None, 1)


@pytest.mark.parametrize("text, ok", [
    ("Income was $553.9 billion.", True), ("Income was $553.9bn.", True), ("Income was 553.9.", True),
    ("Income was $553.9 million.", False), ("Margins were 5.4%.", True), ("Margins were 5.4 million.", False),
    ("Wealth was $36.7 trillion.", True), ("Wealth was $36.7 billion.", False),
    ("The raw value was 13.1 million.", True),        # unknown unit: any scale is still accepted
])
def test_the_verifier_holds_numbers_to_their_column_scale(toolset, text, ok):
    a = NS(ok=True, stop_reason="end_turn", text=text, question="q", tool_calls=[scaled_call()],
           segments=[{"text": text, "sources": ["query:1"]}])
    assert verify(a, toolset).passed is ok


def test_evidence_without_units_is_as_lenient_as_before():
    e = Evidence({13.1}, {}, "")
    n = next(iter(golden.NUMBER.finditer("13.1 million")))
    from rag.agent.verifier import _parse
    assert e.supports(_parse(n))


def test_holds_is_not_found_inside_another_claims_partly_holds():
    # the other verdict belongs to H2, so only the 'Holds' match itself can be wrong here
    assert not states_verdict("H2 Partly holds, and H5 is discussed below.", "H5", "Holds", VERDICTS)


def test_an_advice_question_must_be_declined_not_just_left_unanswered():
    r = golden.Resolved(golden.ADVERSARIAL[0])
    assert not score(r, answer("Nvidia created the most wealth since 2014."), VERDICTS, {})["passed"]


def test_an_account_error_stops_the_run_instead_of_sending_every_question(toolset):
    class BadRequestError(Exception):
        pass
    calls = []

    def broke():
        calls.append(1)
        raise BadRequestError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")
    client = ByQuestion({"Verdict on H1": broke, "Verdict on H9": broke, "How many": broke, "Should I": broke})
    report = evaluate(session(toolset, client), items=ITEMS, workers=1)
    assert len(calls) == 1 and report["stopped"] == "the API account is out of credit"
    assert sum("run stopped" in e for e in report["B1"]["errors"]) == 3


def test_a_missing_baseline_value_fails_the_comparison():
    base = {"numbers_supported": 1.0, "verdict_accuracy": 1.0, "verdict_rule_violations": 0, "voice_accuracy": 1.0,
            "untested_no_verdict": 1.0, "errors": [], "answered": 140, "items": 140,
            "numeric_accuracy": 0.9, "citation_precision": 0.9}
    b0 = {**base, "numeric_accuracy": 0.5, "citation_precision": None, "voice_accuracy": 0.8}
    assert not decide(base, [{"refused": True}], b0, True)["passed"]


def test_the_end_of_a_year_range_is_not_a_number(toolset):
    from rag.agent.verifier import verify
    a = NS(ok=True, stop_reason="end_turn", text="Alpha fell after 2011–14 and over 1948-73.", question="q",
           tool_calls=[], segments=[{"text": "Alpha fell after 2011–14 and over 1948-73.", "sources": []}])
    assert verify(a, toolset).numbers_checked == 0


# ---------------------------------------------------------------- what a passing report vouches for
@pytest.mark.parametrize("key", ["code", "model", "provider", "effort"])
def test_a_pass_is_void_once_the_evaluated_system_changes(toolset, monkeypatch, key):
    settings = load_settings(environ={"RAG_BUILD_DIR": str(toolset.release.path.parent.parent)}, dotenv={})
    monkeypatch.setattr(golden, "resolve", lambda *a, **k: [])
    monkeypatch.setattr(golden, "golden_digest", lambda resolved: "x")
    rid = toolset.release.release_id
    system = system_fingerprint(settings.claude_model, settings.claude_provider, DEFAULT_EFFORT)
    write_report(settings, rid, system={**system, key: "something else"})
    assert promotable(settings, rid) == (False, f"the {key} changed since the evaluation; run it again")


def test_a_report_from_before_fingerprints_cannot_promote(toolset):
    settings = load_settings(environ={"RAG_BUILD_DIR": str(toolset.release.path.parent.parent)}, dotenv={})
    write_report(settings, toolset.release.release_id, system=None)
    assert "changed since the evaluation" in promotable(settings, toolset.release.release_id)[1]


def test_the_fingerprint_follows_the_code(monkeypatch):
    import rag.eval.run as run
    before = system_fingerprint("m", "p", "e")
    monkeypatch.setattr(run, "EVALUATED_CODE", run.EVALUATED_CODE[:-1])
    assert system_fingerprint("m", "p", "e")["code"] != before["code"]


def test_only_a_complete_full_run_replaces_the_report(toolset, tmp_path, monkeypatch):
    out = tmp_path / "eval"
    out.mkdir()
    (out / "report.json").write_text('{"earlier": "full report"}')
    trial = evaluate(session(toolset, good_client()), items=ITEMS, limit=2, out_dir=out)
    again = evaluate(session(toolset, good_client()), items=ITEMS, limit=2, out_dir=out)
    assert trial["run"] != again["run"] and json.loads((out / "report.json").read_text()) == {"earlier": "full report"}

    monkeypatch.setattr(golden, "ITEMS", ITEMS)                          # these four are now the full set
    b0 = Baseline(toolset, good_client(), "claude-opus-5-5")
    evaluate(session(toolset, good_client()), items=ITEMS, baseline=b0, out_dir=out, max_cost=0.0001, workers=1)
    assert json.loads((out / "report.json").read_text()) == {"earlier": "full report"}     # interrupted
    done = evaluate(session(toolset, good_client()), items=ITEMS, baseline=b0, out_dir=out)
    assert not done["decision"]["passed"]
    assert json.loads((out / "report.json").read_text())["run"] == done["run"]             # complete, though it failed


def test_an_incomplete_baseline_or_failed_revision_fails_the_gate():
    m = {"numbers_supported": 1.0, "verdict_accuracy": 1.0, "verdict_rule_violations": 0, "voice_accuracy": 1.0,
         "untested_no_verdict": 1.0, "errors": [], "answered": 140, "items": 140,
         "numeric_accuracy": 0.9, "citation_precision": 0.9}
    b0 = {**m, "numeric_accuracy": 0.5, "citation_precision": 0.5}
    assert decide(m, [{"refused": True}], b0, True)["passed"]
    assert not decide(m, [{"refused": True}], {**b0, "answered": 18}, True)["passed"]
    assert not decide({**m, "revisions_failed": ["x: 529"]}, [{"refused": True}], b0, True)["passed"]


def test_a_crash_is_not_a_refusal():
    class Crashing:
        def call(self, name, inp):
            raise ValueError("broke before the guard")
    assert not any(r["refused"] for r in run_authorization(Crashing()))


def test_a_candidate_must_be_refused_by_the_id_check(monkeypatch):
    import rag.release
    from rag.eval.run import check_candidates_are_not_releases

    def missing(*a, **k):
        raise FileNotFoundError("no manifest")
    monkeypatch.setattr(rag.release, "open_release", missing)
    assert not check_candidates_are_not_releases(None)["refused"]


def test_a_revision_s_query_citation_is_credited_to_the_first_attempt_s_query(toolset):
    q = ToolCall("query_table", {"sql": "SELECT * FROM h3c_top20_2014"}, True,
                 {"results": [{"source_id": "query:abc"}]}, None, 1)
    r = golden.Resolved(Item("x", "numeric", "q"), hypotheses=["H3"])
    revised = answer("Two.", ["query:abc"])
    assert score(r, revised, VERDICTS, {})["citation_precision"] == 0
    assert score(r, revised, VERDICTS, {}, calls=[q])["citation_precision"] == 1
