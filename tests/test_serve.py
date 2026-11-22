"""The chat: unit tests, integration over a real socket (real agent loop, tools, checker and audit
log, scripted model), and the real command line as a separate process."""
import json
import os
import re
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from chat_support import Http, ScriptedClaude, call, finished, make_chat, reply, said, serving
from rag.agent.audit import AuditLog
from rag.agent.orchestrator import Session
from rag.config import ROOT
from rag.serve import chat as chat_module
from rag.serve.app import ADDRESS, ASSETS, STATIC
from rag.serve.auth import SIGNED_IN_S, Auth
from rag.serve.chat import Chat, ChatError
from rag.tools import tool_definitions

GOOD = [reply(call("a", "get_claim", id="H1"), stop="tool_use"),
        reply(said("H1's verdict is Partly holds.", "claim:H1"))]


# ---------------------------------------------------------------- sign-in
def test_each_user_gets_their_own_code_and_name():
    auth = Auth(["ana", "ben"])
    assert auth.codes["ana"] != auth.codes["ben"] and len(auth.codes["ana"]) >= 30
    token, user = auth.sign_in(auth.codes["ben"])
    assert user == "ben" and auth.user(token) == "ben"


@pytest.mark.parametrize("code", ["", "wrong", None, 5, ["x"], "x" * 500, "ünicode"])
def test_a_wrong_code_signs_nobody_in(code):
    assert Auth(["ana"]).sign_in(code) is None


def test_sign_in_expires_and_signs_out():
    now = [1000.0]
    auth = Auth(["ana"], clock=lambda: now[0])
    token, _ = auth.sign_in(auth.codes["ana"])
    other, _ = auth.sign_in(auth.codes["ana"])
    auth.sign_out(other)
    assert auth.user(other) is None and auth.user(token) == "ana"
    now[0] += SIGNED_IN_S + 1
    assert auth.user(token) is None


@pytest.mark.parametrize("users", [[], ["has space"], ["<script>"], ["x" * 65], [""], ["a\nb"]])
def test_bad_user_lists_are_refused(users):
    with pytest.raises(ValueError):
        Auth(users)


# ---------------------------------------------------------------- conversations
def test_an_answer_is_checked_logged_and_only_then_shown(toolset, tmp_path):
    chat = make_chat(toolset, ScriptedClaude({"H1?": list(GOOD)}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "H1?")
    turn = finished(chat, "ana", c["id"])
    a = turn["answer"]
    record = AuditLog(tmp_path / "log").find(a["record_id"])
    assert turn["state"] == "done" and not a["flagged"] and record["answer"] == "H1's verdict is Partly holds."
    assert record["user"] == "ana" and record["release_id"] == a["release_id"] == c["release_id"]
    assert a["segments"] == [{"text": "H1's verdict is Partly holds.", "refs": [1]}]
    assert a["sources"][0] == {"n": 1, "id": "claim:H1", "title": "Source claim:H1", "voice": "evidence",
                               "excerpts": ["What claim:H1 says"]}
    assert turn["steps"] == ["Waiting to start", "Working out what to look up", "Reading the claim",
                             "Working out what to look up", "Checking numbers, verdicts and citations",
                             "Saving to the audit log"]
    chat.close()


def test_no_draft_reaches_the_browser_while_it_is_being_checked(toolset, tmp_path, monkeypatch):
    chat = make_chat(toolset, ScriptedClaude({"H1?": list(GOOD)}), tmp_path / "log")
    reached, release = threading.Event(), threading.Event()
    real = chat.audit.record_answer

    def slow(*args, **kwargs):
        reached.set()
        assert release.wait(20)
        return real(*args, **kwargs)
    monkeypatch.setattr(chat.audit, "record_answer", slow)
    c = chat.create("ana")
    chat.ask("ana", c["id"], "H1?")
    assert reached.wait(20)                                  # the answer exists, and is not logged yet
    shown = json.dumps([chat.get("ana", c["id"]), chat.turn("ana", c["id"], 0)])
    assert "Partly holds" not in shown and "working" in shown
    release.set()
    assert "Partly holds" in json.dumps(finished(chat, "ana", c["id"]))
    chat.close()


def test_an_answer_that_cannot_be_logged_is_withheld_and_forgotten(toolset, tmp_path):
    (tmp_path / "blocked").write_text("a file where the folder should be")
    chat = make_chat(toolset, ScriptedClaude({"H1?": list(GOOD)}), tmp_path / "blocked")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "H1?")
    turn = finished(chat, "ana", c["id"])
    assert turn["state"] == "error" and "withheld" in turn["error"] and turn["answer"] is None
    assert "Partly holds" not in json.dumps(chat.get("ana", c["id"]))
    assert chat._conversations[c["id"]].session.messages == []
    chat.close()


def test_a_failure_is_logged_and_shows_no_internals(toolset, tmp_path):
    error = RuntimeError(r"sk-secret-key leaked in C:\Users\someone\file.py")
    chat = make_chat(toolset, ScriptedClaude({"q": [error]}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "q")
    turn = finished(chat, "ana", c["id"])
    assert turn["state"] == "error" and "sk-secret" not in json.dumps(turn) and "Users" not in json.dumps(turn)
    [(_, _, record)] = list(AuditLog(tmp_path / "log").records())
    assert record["event"] == "error" and record["user"] == "ana"
    chat.close()


def test_a_flagged_answer_says_which_checks_failed_and_hides_the_first_draft(toolset, tmp_path):
    chat = make_chat(toolset, ScriptedClaude({"q": [reply(said("Survival was 42%.")),
                                                    reply(said("Survival was 43%."))]}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "q")
    a = finished(chat, "ana", c["id"])["answer"]
    assert a["flagged"] and a["revised"] and "42" not in json.dumps(a["segments"])
    assert "A number isn't backed by the results" in [f["check"] for f in a["failed_checks"]]
    chat.close()


@pytest.mark.parametrize("stop, words", [("max_tokens", "cut off"), ("refusal", "declined")])
def test_an_unfinished_answer_carries_a_notice(toolset, tmp_path, stop, words):
    chat = make_chat(toolset, ScriptedClaude({"q": [reply(said("Partial"), stop=stop)]}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "q")
    assert words in finished(chat, "ana", c["id"])["answer"]["notice"]
    chat.close()


def test_the_three_voices_are_kept_apart(toolset, tmp_path):
    script = [reply(call("a", "get_claim", id="H1"), call("b", "get_chapter", chapter=12), stop="tool_use"),
              reply(said("The verdict on H1 is Partly holds.", "claim:H1"),
                    said(" Thiel says it himself.", "quote:H1"),
                    said(" The analyst's note agrees.", "note-12-01"))]
    chat = make_chat(toolset, ScriptedClaude({"q": script}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "q")
    a = finished(chat, "ana", c["id"])["answer"]
    assert [(s["n"], s["voice"]) for s in a["sources"]] == [(1, "evidence"), (2, "thiel"), (3, "analyst")]
    chat.close()


def test_a_follow_up_keeps_context_and_voices_from_earlier_questions(toolset, tmp_path):
    client = ScriptedClaude({"H1?": list(GOOD),
                             "Who says so?": [reply(said("Thiel says it himself.", "quote:H1"))]})
    chat = make_chat(toolset, client, tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "H1?")
    finished(chat, "ana", c["id"], 0)
    chat.ask("ana", c["id"], "Who says so?")
    a = finished(chat, "ana", c["id"], 1)["answer"]
    assert a["sources"][0]["voice"] == "thiel" and not a["flagged"]
    assert "H1?" in str(client.requests[-1]["messages"])
    chat.close()


def test_a_chart_is_shown_when_fetched_or_cited(toolset, tmp_path):
    script = [reply(call("a", "get_figure", name="h1_fig1_ccdf"), stop="tool_use"),
              reply(said("The chart shows the tail.", "figure:h1_fig1_ccdf"))]
    chat = make_chat(toolset, ScriptedClaude({"chart": script}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "chart")
    figures = finished(chat, "ana", c["id"])["answer"]["figures"]
    assert [f["name"] for f in figures] == ["h1_fig1_ccdf"] and figures[0]["caption"]
    assert chat.figure("ana", c["id"], "h1_fig1_ccdf") == b"png"
    chat.close()


def test_every_tool_has_a_progress_label():
    assert {d["name"] for d in tool_definitions()} == set(chat_module.LOOKUPS)


def test_one_question_at_a_time_per_user_across_conversations(toolset, tmp_path):
    client = ScriptedClaude({"slow": [reply(said("Done."))], "other": [reply(said("Mine."))]})
    gate = client.hold["slow"] = threading.Event()
    chat = make_chat(toolset, client, tmp_path / "log")
    a, b, theirs = chat.create("ana"), chat.create("ana"), chat.create("ben")
    chat.ask("ana", a["id"], "slow")
    for cid in (a["id"], b["id"]):
        with pytest.raises(ChatError) as e:
            chat.ask("ana", cid, "another")
        assert e.value.status == 409
    chat.ask("ben", theirs["id"], "other")                   # another user isn't blocked
    assert finished(chat, "ben", theirs["id"])["state"] == "done"
    gate.set()
    assert finished(chat, "ana", a["id"])["state"] == "done"
    chat.close()


def test_parallel_users_get_their_own_answers_and_one_intact_log(toolset, tmp_path):
    users = [f"user{i}" for i in range(8)]
    client = ScriptedClaude({f"question of {u}": [reply(said(f"Reply for {u[-1] * 3}x."))] * 2 for u in users})
    chat = make_chat(toolset, client, tmp_path / "log")
    ids = {u: chat.create(u)["id"] for u in users}
    for u in users:
        chat.ask(u, ids[u], f"question of {u}")
    records = {}
    for u in users:
        a = finished(chat, u, ids[u])["answer"]
        assert u[-1] * 3 in a["segments"][0]["text"]
        records[a["record_id"]] = u
    log = AuditLog(tmp_path / "log")
    assert log.verify() == [] and {r["record_id"]: r["user"] for _, _, r in log.records()} == records
    chat.close()


@pytest.mark.parametrize("question, status", [
    (None, 400), (5, 400), ("", 400), ("  \n ", 400), ("\x00\x07", 400), ("x" * 2001, 400), ("bad \ud800", 400)])
def test_bad_questions_are_refused_before_anything_runs(toolset, tmp_path, question, status):
    client = ScriptedClaude({})
    chat = make_chat(toolset, client, tmp_path / "log")
    c = chat.create("ana")
    with pytest.raises(ChatError) as e:
        chat.ask("ana", c["id"], question)
    assert e.value.status == status and client.requests == [] and chat.get("ana", c["id"])["turns"] == []
    chat.close()


def test_someone_else_s_conversation_looks_like_none_at_all(toolset, tmp_path):
    chat = make_chat(toolset, ScriptedClaude({}), tmp_path / "log")
    c = chat.create("ana")
    for attempt in (lambda: chat.get("ben", c["id"]), lambda: chat.turn("ben", c["id"], 0),
                    lambda: chat.ask("ben", c["id"], "hi"), lambda: chat.figure("ben", c["id"], "h1_fig1_ccdf")):
        with pytest.raises(ChatError) as e:
            attempt()
        assert e.value.status == 404
    chat.close()


def test_limits_drop_idle_conversations_but_never_a_working_one(toolset, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module, "MAX_CONVERSATIONS", 3)
    now = [0.0]
    client = ScriptedClaude({"slow": [reply(said("Done."))]})
    gate = client.hold["slow"] = threading.Event()
    chat = make_chat(toolset, client, tmp_path / "log", clock=lambda: now[0])
    working = chat.create("ana")["id"]
    chat.ask("ana", working, "slow")
    made = []
    for i in range(6):
        now[0] += 1
        made.append(chat.create("ana")["id"])
    assert working in chat._conversations and made[0] not in chat._conversations and made[-1] in chat._conversations
    assert sum(c.user == "ana" and not c.working for c in chat._conversations.values()) == 3
    now[0] += chat_module.IDLE_S + 1
    chat.create("ben")
    assert working in chat._conversations and made[-1] not in chat._conversations
    gate.set()
    finished(chat, "ana", working)
    chat.close()


def test_a_full_conversation_asks_for_a_new_one(toolset, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module, "MAX_TURNS", 1)
    chat = make_chat(toolset, ScriptedClaude({"q": [reply(said("Fine."))]}), tmp_path / "log")
    c = chat.create("ana")
    chat.ask("ana", c["id"], "q")
    finished(chat, "ana", c["id"])
    with pytest.raises(ChatError, match="full"):
        chat.ask("ana", c["id"], "again")
    chat.close()


def test_a_conversation_keeps_its_release_when_a_new_one_is_promoted(toolset, tmp_path, monkeypatch):
    first, second = toolset.release.release_id, "r20991231-0000000000"
    bases = {first: Session(toolset, None, "m"),
             second: NS(release_id=second, new_session=lambda: NS(release_id=second))}
    current, opened = [first], []
    monkeypatch.setattr(chat_module, "current_release_id", lambda settings: current[0])
    chat = Chat(None, AuditLog(tmp_path / "log"), evaluated=lambda rid: rid == first,
                open_base=lambda settings, rid: opened.append(rid) or bases[rid])
    old = chat.create("ana")
    current[0] = second
    new = chat.create("ana")
    assert (new["release_id"], new["evaluated"]) == (second, False)
    kept = chat.get("ana", old["id"])
    assert (kept["release_id"], kept["evaluated"]) == (first, True)
    chat.create("ben")
    assert opened == [first, second]                         # each release is opened and re-hashed once
    chat.close()


def test_an_unreadable_evaluation_counts_as_not_evaluated(toolset, tmp_path):
    chat = make_chat(toolset, ScriptedClaude({}), tmp_path / "log")

    def unreadable(release_id):
        raise ValueError("bad json")
    chat._check_evaluated = unreadable
    assert chat.status()["evaluated"] is False
    chat.close()


def test_without_a_promoted_release_there_is_no_chat(toolset, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module, "current_release_id", lambda settings: None)
    chat = make_chat(toolset, ScriptedClaude({}), tmp_path / "log")
    with pytest.raises(ChatError) as e:
        chat.create("ana")
    assert e.value.status == 503
    chat.close()


# ---------------------------------------------------------------- over HTTP
def test_the_whole_path_over_http(toolset, tmp_path):
    script = [reply(call("a", "get_figure", name="h1_fig1_ccdf"), call("b", "get_claim", id="H1"), stop="tool_use"),
              reply(said("H1's verdict is Partly holds.", "claim:H1"))]
    chat = make_chat(toolset, ScriptedClaude({"H1 and its chart?": script}), tmp_path / "log", evaluated=False)
    with serving(chat) as (port, auth):
        web = Http(port)
        cookie = web.sign_in(auth.codes["ana"])
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
        assert web.get("/api/me")[2] == {"user": "ana", "release_id": toolset.release.release_id, "evaluated": False}
        status, _, c = web.post("/api/conversations")
        assert status == 201
        status, _, turn = web.post(f"/api/conversations/{c['id']}/questions", {"question": "H1 and its chart?"})
        assert status == 202 and turn["state"] == "working"
        answer = web.wait(c["id"])["answer"]
        assert "Partly holds" in answer["segments"][0]["text"] and answer["figures"][0]["name"] == "h1_fig1_ccdf"
        status, headers, image = web.get(f"/api/conversations/{c['id']}/figures/h1_fig1_ccdf.png")
        assert status == 200 and image == b"png" and headers["Content-Type"] == "image/png"
        assert web.get(f"/api/conversations/{c['id']}")[2]["turns"][0]["state"] == "done"
        assert AuditLog(tmp_path / "log").find(answer["record_id"])["user"] == "ana"
        assert web.post("/api/sign-out")[0] == 200 and web.get("/api/me")[0] == 401


def test_a_name_in_the_request_cannot_change_who_is_logged(toolset, tmp_path):
    chat = make_chat(toolset, ScriptedClaude({"q": [reply(said("Fine."))]}), tmp_path / "log")
    with serving(chat) as (port, auth):
        web = Http(port)
        web.sign_in(auth.codes["ana"])
        cid = web.post("/api/conversations", {"user": "ben"}, headers={"X-User": "ben"})[2]["id"]
        web.post(f"/api/conversations/{cid}/questions", {"question": "q", "user": "ben"})
        record = AuditLog(tmp_path / "log").find(web.wait(cid)["answer"]["record_id"])
        assert record["user"] == "ana"


@pytest.fixture
def site(toolset, tmp_path):
    client = ScriptedClaude({})
    chat = make_chat(toolset, client, tmp_path / "log")
    with serving(chat) as (port, auth):
        ana, ben = Http(port), Http(port)
        ana.sign_in(auth.codes["ana"])
        ben.sign_in(auth.codes["ben"])
        yield NS(port=port, ana=ana, ben=ben, client=client, cid=ana.post("/api/conversations")[2]["id"])


@pytest.mark.parametrize("method, path", [
    ("GET", "/api/me"), ("POST", "/api/conversations"), ("GET", "/api/conversations/{cid}"),
    ("POST", "/api/conversations/{cid}/questions"), ("GET", "/api/conversations/{cid}/questions/0"),
    ("GET", "/api/conversations/{cid}/figures/h1_fig1_ccdf.png"), ("POST", "/api/sign-out")])
def test_every_endpoint_needs_a_sign_in(site, method, path):
    for cookie in (None, f"ask_z21_{site.port}=forged"):
        assert Http(site.port, cookie).request(method, path.format(cid=site.cid))[0] == 401


@pytest.mark.parametrize("method, path", [
    ("GET", "/api/conversations/{cid}"), ("POST", "/api/conversations/{cid}/questions"),
    ("GET", "/api/conversations/{cid}/questions/0"), ("GET", "/api/conversations/{cid}/figures/h1_fig1_ccdf.png")])
def test_one_user_cannot_reach_another_s_conversation(site, method, path):
    body = {"question": "hi"} if method == "POST" else None
    assert site.ben.request(method, path.format(cid=site.cid), body=body)[0] == 404
    assert site.client.requests == []


@pytest.mark.parametrize("host", ["evil.example", "evil.example:{port}", "127.0.0.1", "127.0.0.1:1", "0.0.0.0:{port}"])
def test_a_rebound_host_name_is_refused(site, host):
    headers = {"Host": host.format(port=site.port)}
    for path in ("/", "/api/me", "/static/app.js"):
        assert site.ana.get(path, headers=headers)[0] == 403


@pytest.mark.parametrize("headers, status", [
    ({"Origin": "https://evil.example"}, 403), ({"Origin": "null"}, 403), ({"Origin": "http://127.0.0.1:1"}, 403),
    ({"Content-Type": "text/plain"}, 415), ({"Content-Type": "application/x-www-form-urlencoded"}, 415)])
def test_another_site_cannot_post(site, headers, status):
    path = f"/api/conversations/{site.cid}/questions"
    assert site.ana.post(path, {"question": "hi"}, headers=headers)[0] == status
    assert site.client.requests == []


@pytest.mark.parametrize("raw", ["{", "[]", "null", '"text"', "[" * 8000 + "]" * 8000, "{}"])
def test_bad_bodies_are_a_plain_400(site, raw):
    status, _, data = site.ana.post(f"/api/conversations/{site.cid}/questions", raw=raw)
    assert status == 400 and "Traceback" not in str(data) and site.client.requests == []


def test_an_oversized_body_never_reaches_the_agent(site):
    try:
        status = site.ana.post(f"/api/conversations/{site.cid}/questions", {"question": "x" * 200_000})[0]
    except (ConnectionError, OSError):
        status = 400                                         # cut off while sending
    assert status >= 400 and site.client.requests == []


@pytest.mark.parametrize("path", [
    "/static/../app.py", "/static/%2e%2e/app.py", "/static/..%2f..%2f.env", "/.env", "/static/app.py",
    "/api/conversations/{cid}/figures/../../../manifest.json",
    "/api/conversations/{cid}/figures/..%2f..%2fmanifest.png",
    "/api/conversations/{cid}/figures/h9_no_such_chart.png",
    "/api/conversations/../../candidates", "/api/conversations/{cid}/questions/7"])
def test_nothing_outside_the_page_and_the_release_is_served(site, path):
    status, _, data = site.ana.get(path.format(cid=site.cid))
    assert status == 404 and "Traceback" not in str(data)


@pytest.mark.parametrize("path", ["/", "/api/me", "/static/app.js", "/static/app.css", "/nothing"])
def test_security_headers_are_on_every_response(site, path):
    _, headers, _ = site.ana.get(path)
    csp = headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp and "unsafe" not in csp
    assert "frame-ancestors 'none'" in csp and headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"


def test_the_page_works_under_its_own_security_policy():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"\sstyle=|\son[a-z]+=|<style|<script(?![^>]*\bsrc=)", html)
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function", script)
    for name in ASSETS:
        text = (STATIC / name).read_text(encoding="utf-8")
        assert not re.search(r"(?:src|href)=\"(?:https?:)?//|url\(\s*[\"']?https?:|@import", text), name


# ---------------------------------------------------------------- the command line, as a separate process
def test_the_serve_command(toolset, tmp_path):
    """No API is called: signing in, the release, a conversation and a chart need no model."""
    env = {**os.environ, "RAG_BUILD_DIR": str(toolset.release.path.parent.parent),
           "RAG_AUDIT_DIR": str(tmp_path / "log"), "ANTHROPIC_API_KEY": "test-not-a-key",
           "OPENAI_API_KEY": "test-not-a-key", "OPENAI_EMBEDDING_MODEL": "hash-test", "CLAUDE_PROVIDER": "anthropic",
           "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen([sys.executable, "-m", "rag", "serve", "--user", "ana", "--port", "0"], cwd=ROOT,
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    killer = threading.Timer(60, process.kill)
    killer.start()
    try:
        output = []
        for line in process.stdout:
            output.append(line)
            if "Ctrl+C" in line:
                break
        text = "".join(output)
        found = re.search(r"sign-in link for ana: http://127\.0\.0\.1:(\d+)/#code=(\S+)", text)
        assert found, text
        assert "NOT EVALUATED" in text
        web = Http(int(found.group(1)))
        web.sign_in(found.group(2))
        assert web.get("/api/me")[2]["release_id"] == toolset.release.release_id
        cid = web.post("/api/conversations")[2]["id"]
        assert web.get(f"/api/conversations/{cid}/figures/h1_fig1_ccdf.png")[2] == b"png"
    finally:
        killer.cancel()
        process.kill()
        process.communicate(timeout=20)


@pytest.mark.parametrize("extra", [["--host", "0.0.0.0"], []])
def test_the_chat_cannot_be_exposed_and_needs_a_named_user(extra):
    from rag.__main__ import main
    argv = ["serve", *(["--user", "ana", *extra] if extra else [])]
    with pytest.raises(SystemExit):
        main(argv)
    assert ADDRESS == "127.0.0.1"
