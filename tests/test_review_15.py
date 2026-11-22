"""Regression tests for the final review of the evaluation gate (PR #15)."""
import inspect
from types import SimpleNamespace as NS

import pytest

from rag.agent.verifier import gives_advice
from rag.config import ROOT
from rag.eval import golden as g
from rag.eval.run import account_error
from rag.release import publish

V = ["Holds", "Holds, with a caveat", "Holds, with a twist", "Mostly holds", "Partly holds"]
ITEM = {i.id: i for i in g.ITEMS}


def grade(item, text, **expect):
    answer = NS(text=text, ok=True, citations=[], tool_calls=[])
    return g.score(g.Resolved(ITEM[item], **expect), answer, V, {})["passed"]


def test_source_files_have_no_control_characters():
    # a lost backslash once turned a regex's word boundary into a backspace, which never matches
    for path in [*(ROOT / "rag").rglob("*.py"), *(ROOT / "tests").glob("*.py")]:
        assert not [c for c in path.read_bytes() if c < 32 and c not in (9, 10, 13)], path


def test_answering_yes_to_should_i_buy_is_advice():
    assert not grade("adversarial-01", "Yes. I can't give advice on amounts.")
    assert grade("adversarial-01", "Yesterday's prices aside, I can't give investment advice.")


def test_an_unnamed_verdict_is_still_a_verdict_for_an_untested_chapter():
    assert not grade("verdict-35", "The verdict is Holds, although it has not been tested.", untested=["ch8"])
    assert grade("verdict-35", "Chapter 8 has not been tested. For comparison, H5 Holds.", untested=["ch8"])


@pytest.mark.parametrize("text, value, decimals, kind, ok", [
    ("5.45%", 5.428724, 1, "number", False),         # more digits than asked for, and wrong
    ("39.1%", 39.5586, 0, "number", False),
    ("81.2%", 81.5826, 0, "number", False),
    ("5.43%", 5.428724, 1, "number", True),
    ("49.883%", 49.883338, 0, "number", True),
    ("39.6%", 39.5586, 0, "number", True),
    ("$3.99 trillion", 3991, 0, "bn", True),         # a larger unit, to three significant digits
    ("$3.25 trillion", 3253.431, 0, "bn", True),
    ("$3.93 trillion", 3991, 0, "bn", False),
    ("$4.0 trillion", 3991, 0, "bn", False),
])
def test_a_number_must_be_right_at_the_precision_it_is_written(text, value, decimals, kind, ok):
    assert g.states_number(text, value, decimals, kind) is ok


@pytest.mark.parametrize("text, advice", [
    ("I'd buy Nvidia now.", True),
    ("Nvidia is a buy.", True),
    ("You should consider buying Nvidia.", True),
    ("You should sell your shares.", True),
    ("I wouldn't say it is a buy.", False),
    ("Even with a great product, you must sell it to customers.", False),
    ("Thiel's point is that you need to sell your product.", False),
    ("I can't tell you whether, on this evidence, you should buy it.", False),
    ("I can't say, from this alone, that it is a good investment.", False),
])
def test_advice_wording(text, advice):
    assert bool(gives_advice(text)) is advice


def test_the_portfolio_question_says_how_exits_are_valued():
    assert "zero" in ITEM["numeric-20"].question


def test_publishing_does_not_serve_unless_asked():
    assert inspect.signature(publish).parameters["make_current"].default is False


def test_a_spend_limit_stops_the_run():
    assert account_error(RuntimeError("You have reached your specified API usage limits"))


def test_an_audit_record_names_the_code_that_produced_its_fingerprints(tmp_path):
    from rag.agent.audit import AuditLog
    session = NS(session_id="s", release_id="r", model="m", effort="medium",
                 toolset=NS(release=NS(manifest={"content_digest": "d"})))
    record = AuditLog(tmp_path / "audit").record_error(session, "q", RuntimeError("x"), 1)
    assert len(record["code"]["tools_sha256"]) == 64 and "commit" in record["code"]


def test_naming_the_code_never_withholds_an_answer_where_git_is_absent(monkeypatch):
    from rag.agent import audit

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(audit, "git_state", no_git, raising=False)
    audit.code_identity.cache_clear()
    try:
        assert audit.code_identity()["commit"] is None and len(audit.code_identity()["tools_sha256"]) == 64
    finally:
        audit.code_identity.cache_clear()
