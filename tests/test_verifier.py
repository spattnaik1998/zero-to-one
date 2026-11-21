"""Answer verifier tests. Answers are built from real tool calls on the synthetic release (so the
evidence is what the agent would really have), with hand-written text and citations."""
import json
from datetime import date
from types import SimpleNamespace as NS

import pytest

from rag.agent.orchestrator import Answer, Session, ToolCall
from rag.agent.verifier import Evidence, ask_checked, revision_request, verify


def call(toolset, name, inp):
    return ToolCall(name, inp, True, toolset.call(name, inp), None, 1)


def answer(segments, calls, question="q", stop="end_turn"):
    """segments: list of (text, [source_ids])."""
    segs = [{"text": t, "sources": list(s)} for t, s in segments]
    return Answer(question=question, text="".join(t for t, _ in segments).strip(), citations=[], tool_calls=calls,
                  stop_reason=stop, release_id="r", model="m", turns=1, segments=segs)


def errors(v, rule=None):
    return [x.detail for x in v.errors if rule is None or x.rule == rule]


@pytest.fixture
def query(toolset):
    # FROM a real table: results of literal-only queries are deliberately not evidence
    sql = ("SELECT 5.428724 AS airlines_pct, 27.438 AS bigtech_pct, 0.7 AS survival, 13100000000 AS op_income, "
           "3991 AS gain FROM h3c_top20_2014 LIMIT 1")
    c = call(toolset, "query_table", {"sql": sql})
    return c, c.result["results"][0]["source_id"]


# ---------------------------------------------------------------- numbers
@pytest.mark.parametrize("written", ["5.4%", "5.43%", "27.4", "70%", "0.7", "$13.1bn", "13.1 billion", "3,991", "$4.0 trillion"])
def test_supported_numbers_at_their_precision(toolset, query, written):
    c, sid = query
    v = verify(answer([(f"The figure was {written}.", [sid])], [c]), toolset)
    assert errors(v, "numbers") == [] and v.numbers_checked == 1


@pytest.mark.parametrize("written", ["5.9%", "27.5", "71%", "3,990"])
def test_unsupported_numbers_are_errors(toolset, query, written):
    c, sid = query
    v = verify(answer([(f"The figure was {written}.", [sid])], [c]), toolset)
    assert len(errors(v, "numbers")) == 1 and v.numbers_supported == 0


def test_years_chapters_claim_ids_and_list_markers_are_not_checked(toolset, query):
    c, sid = query
    text = ("Between 2014 and 2025, chapter 7 and chapters 9 and 10 matter; H3 and H3c too.\n"
            "1. First point is 5.4%.\n2) Second point.")
    v = verify(answer([(text, [sid])], [c]), toolset)
    assert errors(v, "numbers") == [] and v.numbers_checked == 1           # only the 5.4


def test_numbers_from_the_question_and_the_book_date_are_allowed(toolset):
    v = verify(answer([("Of the top 20 firms, published 16 September 2014.", [])], [], question="top 20 firms?"),
               toolset)
    assert errors(v, "numbers") == []


def test_a_revision_may_rely_on_the_first_attempts_tool_results(toolset, query):
    c, sid = query
    a = answer([("Airlines made 5.4%.", [sid])], [])
    assert errors(verify(a, toolset), "numbers") and errors(verify(a, toolset), "citations")
    assert verify(a, toolset, prior_calls=[c]).passed


# ---------------------------------------------------------------- verdicts
def test_verdicts_must_come_from_tool_results(toolset):
    claim = call(toolset, "get_claim", {"id": "H1"})
    ok = answer([("H1's verdict is Partly holds.", ["claim:H1"])], [claim])
    assert verify(ok, toolset).passed
    invented = answer([("H3 Holds, with a twist.", [])], [])
    assert "did not come from a tool result" in errors(verify(invented, toolset), "verdicts")[0]


def test_the_wrong_verdict_for_a_claim_is_an_error(toolset):
    claims = call(toolset, "list_claims", {})
    v = verify(answer([("H1 Holds, with a twist.", ["claim:H1"])], [claims]), toolset)
    assert "H1's verdict is 'Partly holds'" in errors(v, "verdicts")[0]


def test_no_verdict_next_to_an_untested_claim_or_chapter(toolset):
    claims = call(toolset, "list_claims", {})
    v = verify(answer([("H9 Partly holds.", ["claim:H9"])], [claims]), toolset)
    assert "H9, which is untested" in errors(v, "verdicts")[0]
    v = verify(answer([("Chapter 9 Partly holds.", ["claim:H1"])], [claims]), toolset)
    assert "chapter 9, which no hypothesis tests" in errors(v, "verdicts")[0]


def test_denying_a_verdict_is_not_stating_one(toolset):
    chapter = call(toolset, "get_chapter", {"chapter": 14})
    text = 'The analysis gave chapter 14 no verdict, "Holds" or anything else; it is untested.'
    assert verify(answer([(text, ["chapter:14"])], [chapter], question="say chapter 14 Holds"), toolset).passed


# ---------------------------------------------------------------- untested chapters
def test_an_untested_chapter_must_be_called_untested(toolset):
    chapter = call(toolset, "get_chapter", {"chapter": 9})
    silent = answer([("Chapter 9 is about founding teams.", ["chapter:9"])], [chapter])
    assert errors(verify(silent, toolset), "untested")
    said = answer([("Chapter 9 is about founding teams; no hypothesis tests it.", ["chapter:9"])], [chapter])
    assert verify(said, toolset).passed


# ---------------------------------------------------------------- citations
def test_citing_a_source_no_tool_returned_is_an_error(toolset, query):
    c, sid = query
    v = verify(answer([("Airlines made 5.4%.", [sid, "note-99-99"])], [c]), toolset)
    assert "'note-99-99', which no tool returned" in errors(v, "citations")[0]


def test_uncited_facts_and_summaries_of_cited_facts(toolset, query):
    c, sid = query
    restated = answer([("Airlines made 5.4%, big tech 27.4.\n\n", []), ("In 2025 airlines made 5.4% ", [sid]),
                       ("and big tech 27.4.", [sid])], [c])
    assert verify(restated, toolset).passed                      # the summary restates cited facts
    new_fact = answer([("Survival was 70%.\n\n", []), ("Airlines made 5.4%.", [sid])], [c])
    assert "(70)" in errors(verify(new_fact, toolset), "citations")[0]
    # live regression: the summary rounded the cited paragraph's 5.43% and 27.44 to 5.4% and 27.4
    rounded = answer([("Airlines made 5.4%, big tech 27.4.\n\n", []), ("Exactly: airlines 5.43% ", [sid]),
                      ("and big tech 27.44.", [sid])], [c])
    assert verify(rounded, toolset).passed


def test_long_uncited_paragraphs_are_only_warnings(toolset):
    long = " ".join(["word"] * 45)
    v = verify(answer([(long, [])], []), toolset)
    assert v.passed and [x.severity for x in v.violations] == ["warning"]


# ---------------------------------------------------------------- voices
def test_notes_and_thiel_must_be_labelled(toolset):
    notes = call(toolset, "search_notes", {"query": "board"})
    unlabelled = answer([("Boards should have three members.", ["note-09-01"])], [notes])
    assert errors(verify(unlabelled, toolset), "voices")
    labelled = answer([("The analyst's notes say boards should have three members.", ["note-09-01"])], [notes])
    assert verify(labelled, toolset).passed
    claim = call(toolset, "get_claim", {"id": "H1"})
    assert errors(verify(answer([("He said: q", ["quote:H1"])], [claim]), toolset), "voices")
    assert verify(answer([("Thiel said: q", ["quote:H1"])], [claim]), toolset).passed


# ---------------------------------------------------------------- advice and incomplete answers
@pytest.mark.parametrize("text, bad", [
    ("You should buy Tesla.", True), ("Investors should sell now.", True), ("I'd recommend buying it.", True),
    ("It is a strong buy.", True), ("I can't recommend buying or selling any stock.", False),
    ("This is research evidence, not investment advice.", False)])
def test_investment_advice(toolset, text, bad):
    assert bool(errors(verify(answer([(text, [])], []), toolset), "advice")) is bad


def test_an_incomplete_answer_fails_without_further_checks(toolset):
    v = verify(answer([("partial", [])], [], stop="max_tokens"), toolset)
    assert [x.rule for x in v.violations] == ["incomplete"]


def test_evidence_collects_nested_sources_and_voices(toolset):
    e = Evidence.from_calls([call(toolset, "get_chapter", {"chapter": 9}), call(toolset, "get_claim", {"id": "H1"})])
    assert e.sources["note-09-01"] == "analyst" and e.sources["quote:H1"] == "thiel"
    assert e.sources["summary:H1:n_firms"] == "evidence" and 39363.0 in e.numbers


# ---------------------------------------------------------------- verify, revise once, flag (integration)
def t(s, cites=()):
    return NS(type="text", text=s, citations=[NS(source=c, title=c, cited_text=c) for c in cites] or None)


def resp(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0))


class FakeClaude:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        self.requests.append({**params, "messages": list(params["messages"])})
        return self.responses.pop(0)


def tool(id, name, inp):
    return NS(type="tool_use", id=id, name=name, input=inp)


def session(toolset, client):
    return Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 21))


def test_a_passing_answer_is_returned_without_revision(toolset):
    client = FakeClaude(resp(tool("a", "get_claim", {"id": "H1"}), stop="tool_use"),
                        resp(t("H1's verdict is Partly holds.", ["claim:H1"])))
    r = ask_checked(session(toolset, client), "H1?")
    assert r.verification.passed and not r.revised and not r.flagged and len(client.requests) == 2


def test_a_failing_answer_is_revised_once_with_the_problems_listed(toolset):
    client = FakeClaude(
        resp(tool("a", "get_claim", {"id": "H1"}), stop="tool_use"),
        resp(t("H1's verdict is Partly holds, from 999 firms.", ["claim:H1"])),          # 999 is invented
        resp(t("H1's verdict is Partly holds.", ["claim:H1"])))
    r = ask_checked(session(toolset, client), "H1?")
    assert r.revised and not r.flagged and r.answer.text == "H1's verdict is Partly holds."
    assert not r.first_verification.passed and "'999'" in r.first_verification.errors[0].detail
    feedback = client.requests[2]["messages"][-1]
    assert feedback["role"] == "user" and "[numbers] '999'" in feedback["content"]
    d = r.to_dict()
    assert d["question"] == "H1?" and d["revised"] and d["verification"]["passed"] and not d["flagged"]


def test_an_answer_that_still_fails_is_flagged(toolset):
    client = FakeClaude(resp(t("Survival was 42%.")), resp(t("Survival was still 42%.")))
    r = ask_checked(session(toolset, client), "q")
    assert r.revised and r.flagged and r.verification.errors[0].rule == "numbers"
    assert len(client.requests) == 2


def test_a_refusal_is_not_revised(toolset):
    client = FakeClaude(resp(stop="refusal"))
    r = ask_checked(session(toolset, client), "q")
    assert not r.revised and r.flagged and len(client.requests) == 1


def test_revision_request_lists_only_errors():
    from rag.agent.verifier import Verification, Violation
    req = revision_request(Verification([Violation("numbers", "'9' unsupported"),
                                         Violation("citations", "long", "warning")]))
    assert "[numbers] '9' unsupported" in req and "long" not in req and "Do not mention this check" in req


def test_cli_exits_nonzero_for_a_flagged_answer(toolset, monkeypatch, capsys):
    import rag.agent.orchestrator as orch
    from rag.__main__ import main
    monkeypatch.setattr(orch, "open_session", lambda *a, **k: session(
        toolset, FakeClaude(resp(t("Survival was 42%.")), resp(t("Still 42%.")))))
    assert main(["ask", "q"]) == 1
    out = capsys.readouterr().out
    assert "WARNING: this answer failed the automated checks" in out and "checks: FAILED" in out
    monkeypatch.setattr(orch, "open_session", lambda *a, **k: session(
        toolset, FakeClaude(resp(t("Survival was 42%.")), resp(t("Still 42%.")))))
    assert main(["ask", "q", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["flagged"] is True


# ---------------------------------------------------------------- code review findings (PR #12, whole-app review)
def test_a_follow_up_may_rely_on_earlier_questions_tool_results(toolset):
    """Same session: the second answer uses get_claim's result from the first question, with no new call."""
    client = FakeClaude(resp(tool("a", "get_claim", {"id": "H1"}), stop="tool_use"),
                        resp(t("H1's verdict is Partly holds.", ["claim:H1"])),
                        resp(t("H1, rated Partly holds, covers chapter 7.", ["claim:H1"])))
    s = session(toolset, client)
    ask_checked(s, "H1?")
    follow = ask_checked(s, "Which chapter?")
    assert follow.verification.passed and not follow.revised and len(s.calls_history) == 1


def test_calls_history_rolls_back_with_the_messages(toolset):
    class Boom(Exception):
        pass
    client = FakeClaude(resp(tool("a", "list_claims", {}), stop="tool_use"))
    s = session(toolset, client)
    real = client._create

    def create(**params):
        if client.requests:
            raise Boom()
        return real(**params)
    client.beta.messages.create = create
    with pytest.raises(Boom):
        s.ask("q")
    assert s.calls_history == [] and s.messages == []


@pytest.mark.parametrize("text, checked, unsupported", [
    ("The slope was -0.45.", 1, ["0.45"]), ("Margins fell −12.3%.", 1, ["12.3"]),
    ("Returns ranged 5-40%.", 2, ["40"]), ("Shares fell .45 points.", 1, [".45"])])
def test_negative_ranged_and_leading_dot_numbers_are_checked(toolset, query, text, checked, unsupported):
    c, sid = query
    v = verify(answer([(text, [sid])], [c]), toolset)
    assert v.numbers_checked == checked and [x.split("'")[1] for x in errors(v, "numbers")] == unsupported


def test_a_unit_word_must_match_the_scale(toolset, query):
    c, sid = query                       # op_income = 13,100,000,000
    assert verify(answer([("Income was $13.1 billion.", [sid])], [c]), toolset).passed
    assert errors(verify(answer([("Income was $13.1 million.", [sid])], [c]), toolset), "numbers")
    assert errors(verify(answer([("Survival was 700.", [sid])], [c]), toolset), "numbers")   # 0.7 is not 700


def test_the_publication_date_does_not_support_other_numbers(toolset):
    v = verify(answer([("It was published 16 September 2014. Margins were 16%, about 0.2 or 20%.", [])], []), toolset)
    assert sorted(x.split("'")[1] for x in errors(v, "numbers")) == ["0.2", "16", "20"]


def test_literal_only_queries_and_echoed_sql_are_not_evidence(toolset):
    literal = call(toolset, "query_table", {"sql": "SELECT 42 AS survival_pct"})
    sid = literal.result["results"][0]["source_id"]
    assert errors(verify(answer([("Survival was 42%.", [sid])], [literal]), toolset), "numbers")
    where = call(toolset, "query_table", {"sql": "SELECT firm FROM h3c_top20_2014 WHERE rank_2014 < 77"})
    sid = where.result["results"][0]["source_id"]
    assert errors(verify(answer([("The cutoff was 77.", [sid])], [where]), toolset), "numbers")


def test_one_untested_word_no_longer_disables_the_verdict_checks(toolset):
    claims = call(toolset, "list_claims", {})
    v = verify(answer([("H1 Holds, with a twist, while H9 has not been tested.", ["claim:H1"])], [claims]), toolset)
    assert "H1's verdict is 'Partly holds'" in errors(v, "verdicts")[0]
    v = verify(answer([("H9 Partly holds; the write-up is pending.", ["claim:H9"])], [claims]), toolset)
    assert "given to H9, which is untested" in errors(v, "verdicts")[0]


def test_decimals_do_not_cut_the_sentence(toolset, query):
    claims = call(toolset, "list_claims", {})
    c, _ = query
    v = verify(answer([("H1 (r = 0.7) Holds, with a twist.", ["claim:H1"])], [claims, c]), toolset)
    assert "H1's verdict is 'Partly holds'" in errors(v, "verdicts")[0]


def test_lowercase_multi_word_verdicts_are_caught(toolset):
    claims = call(toolset, "list_claims", {})
    v = verify(answer([("Overall, H9 partly holds up in the data.", ["claim:H9"])], [claims]), toolset)
    assert errors(v, "verdicts")
    v = verify(answer([("Chapter 9's advice partly holds.", ["claim:H1"])], [claims]), toolset)
    assert "chapter 9, which no hypothesis tests" in errors(v, "verdicts")[0]


def test_a_verdict_is_judged_against_the_claim_it_is_attached_to(toolset):
    claims = call(toolset, "list_claims", {})
    ok = answer([("H9 is untested, but H1 Partly holds.", ["claim:H1"])], [claims])
    assert verify(ok, toolset).passed
    wrong = answer([("Unlike H9, H3 Partly holds.", ["claim:H3"])], [claims])
    assert "H3's verdict is 'Holds, with a twist'" in errors(verify(wrong, toolset), "verdicts")[0]


def test_untested_chapter_rule_only_for_chapters_the_answer_is_about(toolset):
    nine = call(toolset, "get_chapter", {"chapter": 9})
    claims = call(toolset, "get_claim", {"id": "H1"})
    about_h1 = answer([("H1's verdict is Partly holds.", ["claim:H1"])], [nine, claims], question="What about H1?")
    assert verify(about_h1, toolset).passed                               # chapter 9 fetched incidentally
    elsewhere = answer([("Chapter 9 covers founders. H1's write-up is pending.", ["chapter:9"])], [nine],
                       question="Chapter 9?")
    assert errors(verify(elsewhere, toolset), "untested")                 # 'pending' is not about chapter 9


def test_violations_are_not_duplicated_across_attempts(toolset):
    nine = call(toolset, "get_chapter", {"chapter": 9})
    a = answer([("Chapter 9 covers founders.", ["chapter:9"])], [nine], question="Chapter 9?")
    v = verify(a, toolset, prior_calls=[nine])
    assert len(errors(v, "untested")) == 1


@pytest.mark.parametrize("text", ["I can't tell you whether you should buy Tesla stock.",
                                  "Nobody can say if investors should sell now.",
                                  "Thiel says investors should avoid competitive industries."])
def test_refusals_and_paraphrases_are_not_advice(toolset, text):
    assert not errors(verify(answer([(text, [])], []), toolset), "advice")


def test_a_failed_revision_returns_the_first_answer_flagged(toolset):
    class Overloaded(Exception):
        pass
    client = FakeClaude(resp(t("Survival was 42%.")))
    real = client._create

    def create(**params):
        if client.requests:
            raise Overloaded("529")
        return real(**params)
    client.beta.messages.create = create
    r = ask_checked(session(toolset, client), "q")
    assert r.flagged and not r.revised and r.answer.text == "Survival was 42%." and "Overloaded" in r.revision_error


def test_the_record_keeps_both_attempts(toolset):
    client = FakeClaude(resp(tool("a", "get_claim", {"id": "H1"}), stop="tool_use"),
                        resp(t("H1 Partly holds, from 999 firms.", ["claim:H1"])),
                        resp(t("H1's verdict is Partly holds.", ["claim:H1"])))
    r = ask_checked(session(toolset, client), "H1?")
    d = r.to_dict()
    assert d["first_answer"]["text"] == "H1 Partly holds, from 999 firms."
    assert [c["name"] for c in d["first_answer"]["tool_calls"]] == ["get_claim"]
    assert d["usage_total"]["input_tokens"] == 3 and [c.name for c in r.all_tool_calls] == ["get_claim"]


def test_cli_warns_even_when_the_revision_was_cut_off(toolset, monkeypatch, capsys):
    import rag.agent.orchestrator as orch
    from rag.__main__ import main
    monkeypatch.setattr(orch, "open_session", lambda *a, **k: session(
        toolset, FakeClaude(resp(t("Survival was 42%.")), resp(t("Partial"), stop="max_tokens"))))
    assert main(["ask", "q"]) == 1
    out = capsys.readouterr().out
    assert "WARNING: this answer failed the automated checks after one revision" in out and "[incomplete]" in out
