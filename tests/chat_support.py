"""Helpers for the chat tests: a scripted Claude that is safe across threads, and a real server on a
free port of this computer."""
from __future__ import annotations

import asyncio
import http.client
import json
import threading
import time
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace as NS

from rag.agent.audit import AuditLog
from rag.agent.orchestrator import Session
from rag.config import load_settings
from rag.serve.app import listen
from rag.serve.auth import Auth
from rag.serve.chat import Chat

REVISION = "An automated check of your previous answer"


def said(text, *cites, title=None, excerpt=None):
    return NS(type="text", text=text, citations=[
        NS(source=c, title=title or f"Source {c}", cited_text=excerpt or f"What {c} says") for c in cites] or None)


def call(use_id, name, /, **tool_input):
    return NS(type="tool_use", id=use_id, name=name, input=tool_input)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0))


class ScriptedClaude:
    """Replies by the question being answered, in order; `hold[question]` blocks its first request
    until the event is set."""

    def __init__(self, script: dict):
        self.script = {q: list(steps) for q, steps in script.items()}
        self.hold: dict[str, threading.Event] = {}
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        questions = [m["content"] for m in params["messages"] if m["role"] == "user"
                     and isinstance(m["content"], str) and not m["content"].startswith(REVISION)]
        question = questions[-1]
        with self._lock:
            self.requests.append({**params, "messages": list(params["messages"])})
            gate = self.hold.pop(question, None)
        if gate is not None:
            assert gate.wait(30), "the test never released this question"
        with self._lock:
            step = self.script[question].pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def settings_for(toolset):
    return load_settings(environ={"RAG_BUILD_DIR": str(toolset.release.path.parent.parent)}, dotenv={})


def make_chat(toolset, client, audit_dir, evaluated=True, **kwargs) -> Chat:
    base = Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 22))
    return Chat(settings_for(toolset), AuditLog(audit_dir), open_base=lambda settings, release_id: base,
                evaluated=lambda release_id: evaluated, **kwargs)


def finished(chat: Chat, user: str, conversation_id: str, index: int = 0, timeout: float = 20) -> dict:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        turn = chat.turn(user, conversation_id, index)
        if turn["state"] != "working":
            return turn
        time.sleep(0.02)
    raise AssertionError("the question never finished")


@contextmanager
def serving(chat: Chat, users=("ana", "ben")):
    """The real app on a real socket. Yields (port, auth)."""
    auth = Auth(users)
    ready, box = threading.Event(), {}

    def run():
        async def main():
            box["stop"], box["loop"] = asyncio.Event(), asyncio.get_running_loop()
            server, box["port"] = listen(chat, auth, 0)
            ready.set()
            await box["stop"].wait()
            server.stop()
            await server.close_all_connections()
        asyncio.run(main())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10), "the server never started"
    try:
        yield box["port"], auth
    finally:
        box["loop"].call_soon_threadsafe(box["stop"].set)
        thread.join(10)
        chat.close()


class Http:
    def __init__(self, port: int, cookie: str | None = None):
        self.port, self.cookie = port, cookie

    def request(self, method, path, body=None, headers=None, raw=None):
        sent = {"Host": f"127.0.0.1:{self.port}"}
        if method == "POST":
            sent["Content-Type"] = "application/json"
            raw = raw if raw is not None else json.dumps({} if body is None else body)
        if self.cookie:
            sent["Cookie"] = self.cookie
        sent.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            connection.request(method, path, body=raw, headers=sent)
            response = connection.getresponse()
            data = response.read()
            if "json" in (response.getheader("Content-Type") or ""):
                data = json.loads(data)
            return response.status, dict(response.getheaders()), data
        finally:
            connection.close()

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)

    def sign_in(self, code):
        status, headers, _ = self.post("/api/sign-in", {"code": code})
        assert status == 200
        self.cookie = headers["Set-Cookie"].split(";")[0]
        return headers["Set-Cookie"]

    def wait(self, conversation_id, index=0, timeout=20):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            status, _, turn = self.get(f"/api/conversations/{conversation_id}/questions/{index}")
            assert status == 200
            if turn["state"] != "working":
                return turn
            time.sleep(0.02)
        raise AssertionError("the question never finished")
