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
    client = FakeClaude(response(text("partial"), stop="pause_turn"), response(text("complete")))
    assert session(toolset, client).ask("q").text == "complete" and len(client.requests) == 2


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


def test_tool_results_are_citable_with_voice_in_the_title():
    passages = to_tool_result("t", {"results": [
        {"source_id": "note-12-01", "voice": "analyst", "opinion": True, "chapter": 12, "text": "My view: x"},
        {"source_id": "nb-h1-001", "voice": "thiel", "hypothesis": "H1", "kind": "claim", "section": None, "text": "q"}]})
    assert [b["title"] for b in passages["content"]] == ["Analyst's opinion, chapter 12", "Thiel, quoted in H1"]
    query = to_tool_result("t", {"results": [{"source_id": "query:abc", "voice": "evidence", "sql": "SELECT 1",
                                              "columns": ["x"], "rows": [[8]], "row_count": 1, "truncated": False}]})
    block = query["content"][0]
    assert block["source"] == "query:abc" and block["title"] == "Query result: SELECT 1"
    assert json.loads(block["content"][0]["text"])["rows"] == [[8]]      # structured results cited as JSON
    claim = to_tool_result("t", {"results": [{"source_id": "claim:H9", "voice": "evidence", "id": "H9",
                                              "title": "Founders", "status": "untested"}]})
    assert claim["content"][0]["title"] == "Claim H9: Founders (status untested)"
    assert to_tool_result("t", {"results": []})["content"] == "No results."


# ---------------------------------------------------------------- live (opt-in, costs a few cents)
@pytest.mark.skipif(os.environ.get("RAG_LIVE_TESTS") != "1", reason="set RAG_LIVE_TESTS=1 to call Claude and OpenAI")
def test_live_agent():
    from rag.config import load_settings
    from rag.eval.agent_smoke import CASES, run_case
    from rag.agent.orchestrator import open_session
    s = load_settings()
    failures = []
    for case in CASES:
        result = run_case(open_session(s), case)
        if not result["ok"]:
            failures.append(result)
    assert not failures, failures
