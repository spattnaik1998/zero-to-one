"""Agent loop tests with a scripted stand-in for the Claude API: deterministic, free, run in CI.
A live test (real Claude and OpenAI calls, a few cents) runs only with RAG_LIVE_TESTS=1."""
import json
import os
from datetime import date
from types import SimpleNamespace as NS

import pytest

from rag.agent.orchestrator import FALLBACK_BETA, Session, to_tool_result


def text(t, cites=()):
    return NS(type="text", text=t, citations=[NS(type="search_result_location", source=s, title=f"title {s}",
                                                  cited_text=f"cited {s}") for s in cites] or None)


def tool_use(id, name, inp):
    return NS(type="tool_use", id=id, name=name, input=inp)


def response(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=0))


class FakeClaude:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        self.requests.append({**params, "messages": list(params["messages"])})
        return self.responses.pop(0)


def session(toolset, client, **kw):
    return Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 18), **kw)


def test_tool_loop_answer_citations_and_request_shape(toolset):
    thinking = NS(type="thinking", thinking="", signature="sig")
    client = FakeClaude(
        response(thinking, tool_use("t1", "search_notes", {"query": "CEO pay"}), stop="tool_use"),
        response(text("CEO pay should stay under $150,000.", cites=["note-09-01"])))
    a = session(toolset, client).ask("  What does the analyst say about CEO pay?  ")

    assert a.text == "CEO pay should stay under $150,000." and a.stop_reason == "end_turn" and not a.refused
    assert [(c.source_id, c.cited_text) for c in a.citations] == [("note-09-01", "cited note-09-01")]
    assert [(t.name, t.ok) for t in a.tool_calls] == [("search_notes", True)] and a.turns == 2
    assert a.release_id == toolset.release.release_id and a.question == "What does the analyst say about CEO pay?"
    assert a.usage == {"input_tokens": 200, "output_tokens": 40, "cache_read_input_tokens": 100,
                       "cache_creation_input_tokens": 0}

    first, second = client.requests
    assert first["model"] == "claude-opus-5-5" and first["output_config"] == {"effort": "medium"}
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["cache_control"] == {"type": "ephemeral"} and "tool_choice" not in first
    assert all(t["strict"] for t in first["tools"]) and "search_book" not in {t["name"] for t in first["tools"]}
    assert toolset.release.release_id in first["system"] and "2026-11-18" in first["system"]
    # the assistant turn goes back unchanged (thinking block included), then the tool results
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0] is thinking
    result = second["messages"][2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    first_block = result["content"][0]
    assert first_block["type"] == "search_result" and first_block["source"] == "note-09-01"
    assert first_block["citations"] == {"enabled": True} and first_block["title"].startswith("Analyst's note")
    assert first_block["content"][0]["text"].startswith("A board of three")      # the note itself is citable


def test_parallel_tool_calls_return_in_one_message(toolset):
    client = FakeClaude(
        response(tool_use("a", "list_claims", {}), tool_use("b", "get_chapter", {"chapter": 9}), stop="tool_use"),
        response(text("done")))
    session(toolset, client).ask("q")
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_tool_errors_go_back_to_the_model_and_the_loop_continues(toolset):
    client = FakeClaude(
        response(tool_use("a", "query_table", {"sql": "DELETE FROM h3c_top20_2014"}), stop="tool_use"),
        response(text("I could not run that.")))
    a = session(toolset, client).ask("q")
    err = client.requests[1]["messages"][2]["content"][0]
    assert err["is_error"] is True and "only SELECT" in err["content"]
    assert a.tool_calls[0].ok is False and "only SELECT" in a.tool_calls[0].error and a.text


def test_turn_budget_forces_a_final_answer_without_tools(toolset):
    loop = [response(tool_use(f"t{i}", "list_claims", {}), stop="tool_use") for i in range(2)]
    client = FakeClaude(*loop, response(text("Best answer from what I gathered.")))
    a = session(toolset, client, max_turns=3).ask("q")
    assert [r.get("tool_choice") for r in client.requests] == [None, None, {"type": "none"}]
    assert a.budget_exhausted and a.text == "Best answer from what I gathered." and a.turns == 3


def test_tool_call_budget_also_forces_a_final_answer(toolset):
    client = FakeClaude(
        response(*(tool_use(f"t{i}", "list_claims", {}) for i in range(3)), stop="tool_use"),
        response(text("final")))
    a = session(toolset, client, max_tool_calls=3).ask("q")
    assert client.requests[1]["tool_choice"] == {"type": "none"} and a.budget_exhausted


def test_refusal_and_pause_turn(toolset):
    a = session(toolset, FakeClaude(response(stop="refusal"))).ask("q")
    assert a.refused and a.text == "" and a.tool_calls == []
    client = FakeClaude(response(text("Partial. "), stop="pause_turn"), response(text("Complete.")))
    a = session(toolset, client).ask("q")
    assert a.text == "Partial. Complete." and len(client.requests) == 2      # paused text is kept


def test_history_carries_across_questions(toolset):
    client = FakeClaude(response(text("first")), response(text("second")))
    s = session(toolset, client)
    s.ask("one")
    s.ask("two")
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"] and client.requests[1]["messages"][2]["content"] == "two"


def test_other_providers_skip_server_side_fallbacks(toolset):
    client = FakeClaude(response(text("ok")))
    s = session(toolset, client)
    s.provider = "bedrock"
    s.ask("q")
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


def test_empty_questions_are_rejected(toolset):
    with pytest.raises(ValueError):
        session(toolset, FakeClaude()).ask("   ")


def test_tool_results_are_citable_with_the_toolsets_labels(toolset):
    notes = to_tool_result("t", toolset.call("search_notes", {"query": "computers", "opinion_only": True}))
    assert [(b["source"], b["title"]) for b in notes["content"]] == [("note-12-01", "Analyst's opinion, chapter 12")]
    assert notes["content"][0]["content"][0]["text"] == "My view: computers complement people."   # the note itself
    query = to_tool_result("t", toolset.call("query_table", {"sql": "SELECT 8 AS n"}))
    block = query["content"][0]
    assert block["source"].startswith("query:") and block["title"] == "Query result: SELECT 8 AS n"
    assert json.loads(block["content"][0]["text"])["rows"] == [[8]]      # structured results cited as JSON
    assert to_tool_result("t", {"results": []})["content"] == "No results."


def test_nested_items_become_their_own_blocks_in_their_own_voice(toolset):
    """Review finding: a chapter's notes and a claim's Thiel quote were packed into one evidence block."""
    chapter = to_tool_result("t", toolset.call("get_chapter", {"chapter": 9}))["content"]
    by_source = {b["source"]: b for b in chapter}
    assert by_source["note-09-01"]["title"] == "Analyst's note, chapter 9"
    assert by_source["note-09-01"]["content"][0]["text"].startswith("A board of three")
    parent = json.loads(by_source["chapter:9"]["content"][0]["text"])
    assert parent["notes"] == ["note-09-01"] and "board of three" not in by_source["chapter:9"]["content"][0]["text"]

    claim = to_tool_result("t", toolset.call("get_claim", {"id": "H1"}))["content"]
    by_source = {b["source"]: b for b in claim}
    assert by_source["quote:H1"]["title"] == "Thiel, quoted in H1" and by_source["quote:H1"]["content"][0]["text"] == "q"
    assert by_source["summary:H1:n_firms"]["title"] == "Summary value H1 n_firms"
    parent = json.loads(by_source["claim:H1"]["content"][0]["text"])
    assert parent["thiel_quote"] == "quote:H1" and "summary:H1:alpha" in parent["headline_values"]
    assert by_source["claim:H1"]["title"] == "Claim H1: The power law (status final)"


# ---------------------------------------------------------------- review findings: session integrity
def _valid_history(messages):
    """Every assistant tool_use is answered by a tool_result in the next message."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ids = {b.id for b in m["content"] if getattr(b, "type", None) == "tool_use"}
        if ids:
            nxt = messages[i + 1]["content"] if i + 1 < len(messages) else []
            answered = {r["tool_use_id"] for r in nxt if isinstance(r, dict) and r.get("type") == "tool_result"}
            if ids - answered:
                return False
    return True


def test_an_unexpected_tool_failure_becomes_an_error_result(toolset, monkeypatch):
    real_call = toolset.call

    def flaky(name, inp):
        if name == "search_notes":
            raise RuntimeError("OpenAI rate limit")
        return real_call(name, inp)
    monkeypatch.setattr(toolset, "call", flaky)
    client = FakeClaude(response(tool_use("a", "search_notes", {"query": "x"}), stop="tool_use"),
                        response(text("Answered without the notes.")))
    s = session(toolset, client)
    a = s.ask("q")
    err = client.requests[1]["messages"][2]["content"][0]
    assert err["is_error"] and "internal reason" in err["content"] and "rate limit" not in err["content"]
    assert a.ok and a.tool_calls[0].error == "internal error: RuntimeError" and _valid_history(s.messages)


def test_an_api_failure_leaves_the_history_as_it_was(toolset):
    class Boom(Exception):
        pass

    client = FakeClaude(response(text("first answer")),
                        response(tool_use("a", "list_claims", {}), stop="tool_use"))
    s = session(toolset, client)
    s.ask("one")
    before = list(s.messages)
    real = client._create

    def create(**params):
        if len(client.requests) >= 2:
            client.requests.append(params)
            raise Boom("overloaded")
        return real(**params)
    client.beta.messages.create = create
    with pytest.raises(Boom):
        s.ask("two")                                # turn 1 calls a tool, turn 2 fails at the API
    assert s.messages == before and _valid_history(s.messages)


def test_a_reply_cut_off_mid_tool_call_keeps_the_history_valid(toolset):
    client = FakeClaude(response(text("Let me check"), tool_use("a", "list_claims", {}), stop="max_tokens"),
                        response(text("next answer")))
    s = session(toolset, client)
    a = s.ask("one")
    assert a.stop_reason == "max_tokens" and not a.ok and a.tool_calls == []
    assert _valid_history(s.messages) and "Not run" in s.messages[-1]["content"][0]["content"]
    assert s.ask("two").ok                          # the session still works


def test_the_tool_budget_applies_per_call_not_per_turn(toolset):
    client = FakeClaude(response(*(tool_use(f"t{i}", "list_claims", {}) for i in range(4)), stop="tool_use"),
                        response(text("final")))
    s = session(toolset, client, max_tool_calls=2)
    a = s.ask("q")
    results = client.requests[1]["messages"][2]["content"]
    assert len(a.tool_calls) == 2 and [r.get("is_error", False) for r in results] == [False, False, True, True]
    assert "budget" in results[2]["content"] and _valid_history(s.messages)


def test_a_full_session_asks_for_a_new_one(toolset):
    from rag.agent.orchestrator import CONTEXT_LIMIT, ContextLimitError
    s = session(toolset, FakeClaude())
    s.context_tokens = CONTEXT_LIMIT + 1
    with pytest.raises(ContextLimitError, match="new session"):
        s.ask("q")


def test_new_session_shares_the_release_and_clients(toolset):
    client = FakeClaude(response(text("a")))
    s = session(toolset, client, effort="high")
    s.ask("one")
    fresh = s.new_session()
    assert fresh.messages == [] and fresh.toolset is s.toolset and fresh.client is client and fresh.effort == "high"


def test_ask_json_exits_nonzero_unless_the_answer_completed(toolset, monkeypatch, capsys):
    import rag.agent.orchestrator as orch
    from rag.__main__ import main
    monkeypatch.setattr(orch, "open_session", lambda *a, **k: session(toolset, FakeClaude(response(stop="refusal"))))
    assert main(["ask", "q", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["refused"] is True


def test_smoke_verdict_patterns_come_from_the_report():
    import re
    from rag.eval.agent_smoke import verdict_patterns
    pats = verdict_patterns()
    hit = lambda s: any(re.search(p, s, re.I) for p in pats)  # noqa: E731
    assert hit("The verdict is Holds.") and hit("It Mostly holds") and hit('"Holds, with a twist"')
    assert not hit("whether chapter 9's advice holds up") and not hit("it holds")


def test_smoke_forbids_a_verdict_only_next_to_the_untested_chapter():
    import re
    from rag.eval.agent_smoke import CASES
    pats = next(c for c in CASES if c.name.startswith("untested")).must_not_match
    hit = lambda s: any(re.search(p, s, re.I | re.M) for p in pats)  # noqa: E731
    assert hit("Chapter 9 Partly holds.") and hit("H9 is rated Holds, with a twist.")
    assert not hit("The tested claims differ: H2 Partly holds. Chapter 9 is untested.")
    assert not hit("For comparison, H7 Holds, with a caveat.")


# ---------------------------------------------------------------- live (opt-in, costs a few cents)
@pytest.mark.skipif(os.environ.get("RAG_LIVE_TESTS") != "1", reason="set RAG_LIVE_TESTS=1 to call Claude and OpenAI")
def test_live_agent():
    from rag.agent.orchestrator import open_session
    from rag.agent.verifier import ask_checked
    from rag.config import load_settings
    from rag.eval.agent_smoke import run_all
    base = open_session(load_settings())
    failures = [r for r in run_all(base) if not r["ok"]]
    assert not failures, failures
    # a follow-up question in the same session resends the whole history, including the first final
    # answer; this failed with "text content blocks must be non-empty" before empty blocks were dropped
    s = base.new_session()
    first = ask_checked(s, "What is the verdict on H2?")
    follow = ask_checked(s, "And which chapters does it come from?")
    assert first.answer.ok and follow.answer.ok and follow.verification.passed
