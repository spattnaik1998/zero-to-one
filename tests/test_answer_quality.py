"""#8's groundwork: readable tool results (no raw data in answers), the checker's false alarms, a
withheld answer forgotten, progress events, and the agreed citation rule. Every loosened rule has
a wrong twin that must still fail."""
from datetime import date
from types import SimpleNamespace as NS

import pytest

from chat_support import ScriptedClaude, call, reply, said
from rag.agent import render
from rag.agent.audit import TOOL_CODE, AuditError, AuditLog, ask_audited
from rag.agent.orchestrator import Session, ToolCall, to_tool_result
from rag.agent.verifier import Evidence, RAW_DATA, _checked_numbers, ask_checked, verify
from rag.eval.run import CITATION_FLOOR, CITATION_MARGIN, EVALUATED_CODE, decide

MARGINS = ToolCall("query_table", {"sql": "SELECT * FROM margins"}, True, {"results": [{
    "source_id": "query:m", "voice": "evidence", "label": "Query result", "sql": "SELECT * FROM margins",
    "columns": ["year", "air_margin_pct", "air_income_bn"],
    "rows": [[2024, 5.577544648381332, 13.336958], [2025, 5.428724140694802, 13.134042]]}]}, None, 1)
OTHER = ToolCall("query_table", {"sql": "SELECT * FROM other"}, True, {"results": [{
    "source_id": "query:o", "voice": "evidence", "label": "Query result", "sql": "SELECT * FROM other",
    "columns": ["share"], "rows": [[0.332], [0.047], [0.061], [0.398]]}]}, None, 1)


def check(toolset, *segments, calls=(MARGINS, OTHER), question="q"):
    segs = [{"text": s if isinstance(s, str) else s[0], "sources": [] if isinstance(s, str) else list(s[1:])}
            for s in segments]
    answer = NS(ok=True, stop_reason="end_turn", text="".join(s["text"] for s in segs), question=question,
                tool_calls=list(calls), segments=segs)
    return verify(answer, toolset)


def rules(result):
    return {x.rule for x in result.errors}


# ---------------------------------------------------------------- readable results
def test_a_query_row_is_one_readable_line_without_json():
    blocks = to_tool_result("t", MARGINS.result)["content"]
    text = [c["text"] for c in blocks[0]["content"]]
    assert text[2] == "year: 2025; air_margin_pct: 5.42872%; air_income_bn: $13.134 billion."
    assert len(text) == 3 and not any(RAW_DATA.search(t) or "{" in t for t in text)


@pytest.mark.parametrize("value, name, shown", [
    (5.428724140694802, "margin_pct", "5.42872%"),
    (38785.0, "n", "38,785"),
    (2025, "year", "2025"),
    (320193, "cik", "320193"),
    (13100000000.0, "op_income", "13,100,000,000"),
    (123456.789, "x", "123,456.79"),
    (2.3120226167757434e-80, "p", "2.31202e-80"),
    (None, "x", "not available"),
    (True, "still_top20", "yes"),
    (-38.105654, "op_income_bn", "-$38.1057 billion"),
    (21.971, "mean_pay_2014_k", "$21.971 thousand"),
    (31.91, "emp_2025_m", "31.91 million"),
    ([3, 5, 6], "chapters", "3, 5 and 6"),
    ("NVIDIA CORP", "firm_pct", "NVIDIA CORP"),
])
def test_values_are_written_for_reading(value, name, shown):
    assert render.quantity(value, name) == shown


@pytest.mark.parametrize("value", [5.428724140694802, 553.868, 0.00028465333668718285, 3991.0, 123456.789, 21.971])
def test_a_rendered_number_is_still_supported_by_the_raw_result(value):
    evidence = Evidence(set(), {}, "", free=set())
    evidence.add(value)
    assert all(evidence.supports(n) for n in _checked_numbers(render.number(value)))


def test_a_row_label_carries_its_unit():
    item = {"columns": ["row_label", "value"], "rows": [["net_gain_neutral_tn", 36.657]]}
    assert render.lines(item)[1] == "row_label: net_gain_neutral_tn; value: $36.657 trillion."


def test_a_verdict_is_rendered_only_for_a_final_claim():
    base = {"id": "H1", "title": "T", "claim": "C", "verdict": "Partly holds", "one_line": "Secret finding"}
    assert '"Partly holds"' in " ".join(render.lines({**base, "status": "final"}))
    for status in ("untested", "provisional"):
        text = " ".join(render.lines({**base, "status": status}))
        assert "Partly holds" not in text and "Secret finding" not in text and "no verdict" in text


def test_every_tool_s_results_render_without_raw_data_or_empty_blocks(toolset):
    calls = [("list_claims", {}), ("get_claim", {"id": "H1"}), ("get_claim", {"id": "H9"}),
             ("get_chapter", {"chapter": 0}), ("get_chapter", {"chapter": 9}),
             ("get_figure", {"name": "h1_fig1_ccdf"}), ("describe_table", {"name": "h3c_top20_2014"}),
             ("search_notes", {"query": "computers"}), ("search_analysis", {"query": "power law"}),
             ("find_data", {"query": "top 20"}),
             ("query_table", {"sql": "SELECT * FROM h3c_top20_2014"}),
             ("query_table", {"sql": "SELECT * FROM h3c_top20_2014 WHERE false"})]
    for name, tool_input in calls:
        for block in to_tool_result("t", toolset.call(name, tool_input))["content"]:
            for content in block["content"]:
                assert content["text"].strip() and not RAW_DATA.search(content["text"]), (name, content)


# ---------------------------------------------------------------- raw data in an answer
@pytest.mark.parametrize("text", [
    'H4\'s verdict is "status": "final".',
    'Its record is marked "tested": false.',
    "Margins: [2025, 5.428724140694802, 13.134042].",
    "H4 covers [3, 5, 6, 12, 13].",
    'Growth: ["2014-2025", 2014, 1.868].',
    "productivity[4].period gives the years.",
])
def test_pasted_raw_data_fails_the_check(toolset, text):
    assert "format" in rules(check(toolset, (text, "query:m")))


@pytest.mark.parametrize("text", [
    'The verdict: "Partly holds".',
    "H4 covers chapters 3, 5, 6, 12 and 13.",
    "Margins fell [in 2020] sharply.",
    "The interval was [-0.2, 0.4].",
    'He calls it "the last mover": the firm that keeps the market.',
])
def test_ordinary_writing_is_not_raw_data(toolset, text):
    assert "format" not in rules(check(toolset, (text, "query:m")))


# ---------------------------------------------------------------- numbers: false alarms and their twins
@pytest.mark.parametrize("text, supported", [
    ("It beat the S&P 500 and the Russell 2000.", True),
    ("The share was 4.7–6.1%.", True),
    ("The share was 4.7 to 6.1%.", True),
    ("It added about 40 cents per dollar.", True),
    ("It had 500 firms.", False),
    ("The S&P had 500 winners.", False),
    ("The share was 4.2–6.1%.", False),
    ("It added about 45 cents per dollar.", False),
    ("It added 40 firms.", False),
])
def test_names_ranges_and_cents(toolset, text, supported):
    assert ("numbers" not in rules(check(toolset, (text, "query:o")))) is supported


# ---------------------------------------------------------------- citations are not relaxed
def test_a_table_beside_a_cited_paragraph_still_needs_its_facts_cited(toolset):
    lead = ("Margins have fallen since 2024.", "query:m")
    table = "\n\n| Year | Margin |\n|---|---|\n| 2024 | 5.58% |\n| 2025 | 5.43% |"
    assert "citations" in rules(check(toolset, lead, table))
    stated = ("Margins fell from 5.58% to 5.43%.", "query:m")
    assert "citations" not in rules(check(toolset, stated, table))


# ---------------------------------------------------------------- untested
@pytest.mark.parametrize("question, text, ok", [
    ("What does chapter 9 argue?", "Chapter 9 hasn’t been tested.", True),
    ("What does chapter 9 argue?", "It hasn't been tested.", False),
    ("Was the preface tested?", "The preface wasn’t tested.", True),
    ("What does chapter 9 argue?", "Chapter 9 wasn't tested.", True),
    ("Compare chapters 9 and 12.", "It hasn't been tested.", False),
    ("What does chapter 9 argue?", "Chapter 8 hasn't been tested.", False),
    ("What does chapter 9 argue?", "It is about founders.", False),
])
def test_untested_said_naturally(toolset, question, text, ok):
    number = 0 if "preface" in question else 9
    chapter = ToolCall("get_chapter", {"chapter": number}, True, toolset.call("get_chapter", {"chapter": number}), None, 1)
    result = check(toolset, (text, f"chapter:{number}"), calls=[chapter], question=question)
    assert ("untested" not in rules(result)) is ok


# ---------------------------------------------------------------- withheld answers and progress
def session(toolset, client):
    return Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 22))


class BrokenLog(AuditLog):
    def append(self, record):
        raise AuditError("disk full")


def test_a_withheld_answer_is_forgotten_by_the_session(toolset, tmp_path):
    client = ScriptedClaude({"H1?": [reply(call("a", "get_claim", id="H1"), stop="tool_use"),
                                     reply(said("The secret answer: H1 Partly holds.", "claim:H1"))],
                             "What did you just say?": [reply(said("I have no earlier answer."))]})
    s = session(toolset, client)
    with pytest.raises(AuditError):
        ask_audited(s, "H1?", BrokenLog(tmp_path / "audit"))
    assert s.messages == [] and s.calls_history == [] and s.context_tokens == 0
    ask_audited(s, "What did you just say?", AuditLog(tmp_path / "audit"))
    assert "secret answer" not in str(client.requests[-1]["messages"])


def test_a_withheld_follow_up_keeps_the_earlier_logged_history(toolset, tmp_path):
    client = ScriptedClaude({"one": [reply(said("First."))], "two": [reply(said("Second."))]})
    s = session(toolset, client)
    ask_audited(s, "one", AuditLog(tmp_path / "audit"))
    kept = list(s.messages)
    with pytest.raises(AuditError):
        ask_audited(s, "two", BrokenLog(tmp_path / "audit"))
    assert s.messages == kept


def test_progress_reports_stages_and_never_content(toolset, tmp_path):
    client = ScriptedClaude({"H1?": [reply(call("a", "get_claim", id="H1"), stop="tool_use"),
                                     reply(said("H1 Partly holds, from 999 firms.", "claim:H1")),
                                     reply(said("H1's verdict is Partly holds.", "claim:H1"))]})
    s = session(toolset, client)
    seen = []
    s.on_event = lambda stage, detail: seen.append((stage, detail))
    ask_audited(s, "H1?", AuditLog(tmp_path / "audit"))
    assert seen == [("thinking", None), ("tool", "get_claim"), ("thinking", None), ("checking", None),
                    ("revising", None), ("thinking", None), ("checking", None), ("recording", None)]


def test_a_broken_progress_display_does_not_fail_the_question(toolset):
    s = session(toolset, ScriptedClaude({"q": [reply(said("Fine."))]}))
    s.on_event = lambda stage, detail: 1 / 0
    assert ask_checked(s, "q").answer.text == "Fine."


def test_rendering_is_part_of_what_is_evaluated_and_audited():
    assert "agent/render.py" in EVALUATED_CODE
    assert {"agent/render.py", "agent/orchestrator.py"} <= set(TOOL_CODE)


# ---------------------------------------------------------------- the citation rule
@pytest.mark.parametrize("agent, baseline, passes", [
    (0.95, 0.95, True), (0.95, 0.98, True), (0.9524, 0.98, True), (0.97, 1.0, True), (1.0, 0.50, True),
    (0.9499, 0.90, False), (0.95, 0.9801, False), (0.96, 1.0, False), (None, 0.9, False), (0.99, None, False),
])
def test_citation_rule_is_a_floor_and_a_margin(agent, baseline, passes):
    assert (CITATION_FLOOR, CITATION_MARGIN) == (0.95, 0.03)
    m = {"numbers_supported": 1.0, "verdict_accuracy": 1.0, "verdict_rule_violations": 0, "voice_accuracy": 1.0,
         "untested_no_verdict": 1.0, "errors": [], "answered": 140, "items": 140, "numeric_accuracy": 1.0}
    b1, b0 = {**m, "citation_precision": agent}, {**m, "numeric_accuracy": 0.8, "citation_precision": baseline}
    assert decide(b1, [{"refused": True}], b0, True)["passed"] is passes


@pytest.mark.parametrize("text", ["It rose from 2014 to 75%.", "It moved over 2011–14 to 75%."])
def test_a_year_before_a_percentage_is_still_a_year(text):
    assert [n.literal for n in _checked_numbers(text)] == ["75"]
