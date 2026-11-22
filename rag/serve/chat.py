"""Conversations: who owns them, which release they are pinned to, and running their questions.

A question runs on a worker thread. Until its answer has passed through `ask_audited` (checked,
revised once if needed, logged) the browser is given progress steps and nothing else, so nothing
unverified or unlogged is ever shown.
"""
from __future__ import annotations

import copy
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..agent.audit import AuditError, ask_audited
from ..agent.orchestrator import ContextLimitError, Session, open_session
from ..release import current_release_id
from .view import answer_view

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
MAX_TURNS = 40                        # per conversation
MAX_CONVERSATIONS = 20                # per user; the oldest is dropped
IDLE_S = 8 * 3600
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

STEPS = {
    "queued": "Waiting to start",
    "thinking": "Working out what to look up",
    "checking": "Checking numbers, verdicts and citations",
    "revising": "Correcting what the check found",
    "recording": "Saving to the audit log",
}
LOOKUPS = {
    "search_notes": "Searching the analyst's notes",
    "search_analysis": "Searching the analysis",
    "find_data": "Finding tables and charts",
    "list_claims": "Looking up the verdicts",
    "get_claim": "Reading the claim",
    "get_chapter": "Reading the chapter record",
    "describe_table": "Checking the table's columns",
    "query_table": "Querying the data",
    "get_figure": "Getting the chart",
}


class ChatError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Conversation:
    id: str
    user: str
    session: Session
    evaluated: bool
    used: float
    turns: list = field(default_factory=list)

    @property
    def working(self) -> bool:
        return any(t["state"] == "working" for t in self.turns)


class Chat:
    def __init__(self, settings, audit, open_base=open_session, evaluated=None, workers: int = 4,
                 clock=time.monotonic):
        self.settings, self.audit, self.open_base, self.clock = settings, audit, open_base, clock
        self._check_evaluated = evaluated or self._promotable
        self._lock = threading.Lock()                         # conversations and turns
        self._releases_lock = threading.Lock()                # opening a release
        self._bases: dict[str, Session] = {}
        self._evaluated: dict[str, bool] = {}
        self._conversations: dict[str, Conversation] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chat")

    def close(self) -> None:
        """Let questions already started finish: each is paid for, and must reach the audit log."""
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------ releases
    def _promotable(self, release_id: str) -> bool:
        from ..eval.run import promotable
        return promotable(self.settings, release_id)[0]

    def current(self) -> tuple[Session, bool]:
        """A base session on the release being served now, and whether it passed its evaluation."""
        release_id = current_release_id(self.settings)
        if not release_id:
            raise ChatError(503, "No release is being served. Ask the operator to promote one.")
        with self._releases_lock:
            if release_id not in self._bases:
                base = self.open_base(self.settings, release_id)     # re-hashes every file
                try:
                    passed = bool(self._check_evaluated(release_id))
                except Exception:
                    log.exception("could not read the evaluation of %s", release_id)
                    passed = False
                self._bases[release_id], self._evaluated[release_id] = base, passed
            return self._bases[release_id], self._evaluated[release_id]

    def status(self) -> dict:
        base, passed = self.current()
        return {"release_id": base.release_id, "evaluated": passed}

    # ------------------------------------------------------------ conversations
    def create(self, user: str) -> dict:
        base, passed = self.current()
        now = self.clock()
        conversation = Conversation(uuid.uuid4().hex, user, base.new_session(), passed, now)
        with self._lock:
            kept = {k: c for k, c in self._conversations.items() if c.working or now - c.used <= IDLE_S}
            mine = sorted((c for c in kept.values() if c.user == user and not c.working), key=lambda c: c.used)
            for old in mine[:max(0, len(mine) - MAX_CONVERSATIONS + 1)]:
                del kept[old.id]
            kept[conversation.id] = conversation
            self._conversations = kept
            return self._snapshot(conversation)

    def _owned(self, user: str, conversation_id: str) -> Conversation:
        c = self._conversations.get(conversation_id)
        if c is None or c.user != user:                       # someone else's looks like none at all
            raise ChatError(404, "That conversation has ended. Start a new one.")
        return c

    @staticmethod
    def _snapshot(c: Conversation) -> dict:
        return {"id": c.id, "release_id": c.session.release_id, "evaluated": c.evaluated,
                "turns": copy.deepcopy(c.turns)}

    def get(self, user: str, conversation_id: str) -> dict:
        with self._lock:
            return self._snapshot(self._owned(user, conversation_id))

    def turn(self, user: str, conversation_id: str, index: int) -> dict:
        with self._lock:
            turns = self._owned(user, conversation_id).turns
            if not 0 <= index < len(turns):
                raise ChatError(404, "That question isn't in this conversation.")
            return copy.deepcopy(turns[index])

    def figure(self, user: str, conversation_id: str, name: str) -> bytes:
        with self._lock:
            session = self._owned(user, conversation_id).session
        try:
            if name not in session.toolset.figures:
                raise KeyError(name)
            return session.toolset.release.file(f"figures/{name}.png").read_bytes()   # the pinned release's copy
        except (KeyError, OSError):
            raise ChatError(404, "That chart isn't in this release.") from None

    # ------------------------------------------------------------ questions
    def ask(self, user: str, conversation_id: str, question) -> dict:
        if not isinstance(question, str):
            raise ChatError(400, "Send the question as text.")
        question = CONTROL.sub("", question).strip()
        if not question:
            raise ChatError(400, "Type a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ChatError(400, f"Keep the question under {MAX_QUESTION_CHARS:,} characters.")
        try:
            question.encode("utf-8")
        except UnicodeEncodeError:
            raise ChatError(400, "The question contains characters that can't be read.") from None
        with self._lock:
            conversation = self._owned(user, conversation_id)
            if any(c.working for c in self._conversations.values() if c.user == user):
                raise ChatError(409, "Wait for your current answer before asking another question.")
            if len(conversation.turns) >= MAX_TURNS:
                raise ChatError(409, "This conversation is full. Start a new one.")
            turn = {"index": len(conversation.turns), "question": question, "state": "working",
                    "steps": [STEPS["queued"]], "answer": None, "error": None}
            conversation.turns.append(turn)
            conversation.used = self.clock()
            try:
                self._pool.submit(self._run, conversation, turn)
            except RuntimeError:                              # shutting down
                conversation.turns.pop()
                raise ChatError(503, "The server is stopping. Ask again when it restarts.") from None
            return copy.deepcopy(turn)

    def _run(self, conversation: Conversation, turn: dict) -> None:
        session = conversation.session

        def progress(stage: str, detail: str | None = None) -> None:
            step = LOOKUPS.get(detail, "Looking up the results") if stage == "tool" else STEPS.get(stage)
            if step:
                with self._lock:
                    if turn["steps"][-1] != step:
                        turn["steps"].append(step)

        session.on_event = progress
        answer = error = None
        try:
            checked, record = ask_audited(session, turn["question"], self.audit, user=conversation.user)
            answer = answer_view(checked, record, session)
        except AuditError:
            log.exception("answer withheld: the audit log could not be written")
            error = "The answer was withheld because it couldn't be saved to the audit log. Tell the operator."
        except ContextLimitError:
            error = "This conversation is too long to continue. Start a new one."
        except Exception:
            log.exception("question failed")
            error = "The question couldn't be answered because a service failed. Ask again in a moment."
        finally:
            session.on_event = None
            with self._lock:
                if answer is not None:
                    turn.update(state="done", answer=answer)
                else:
                    turn.update(state="error", error=error or "The question was interrupted. Ask it again.")
                conversation.used = self.clock()
