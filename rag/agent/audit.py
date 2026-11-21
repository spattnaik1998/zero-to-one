"""Append-only, tamper-evident audit log of every question the assistant answers (spec section 7).

One JSON record per question, in daily JSON Lines files (audit-YYYY-MM-DD.jsonl). Each record
carries a sequence number, the hash of the previous record and its own hash, so the files form
one hash chain; HEAD.json anchors the latest record. `AuditLog.verify` re-walks the chain and reports
an edited record (hash mismatch), a deleted, reordered or inserted one (sequence or link break),
and records cut from the end (HEAD ahead of the files).

A record holds what is needed to reconstruct why an answer was shown: the question; every
attempt (text, citations, and each tool call's name, input, the source ids it returned and a
SHA-256 of its result, not the result itself); both verifications; whether it was revised or
flagged; the release id and content digest; model and effort; tokens; timing; and the session id.
A question that fails with an exception is logged as an "error" record.

Releases are immutable, so a lookup can be replayed from its input and the release id;
`result_digest` of the replayed result must equal the recorded `result_sha256`.

Writes take a cross-process lock (an atomic lock file), so concurrent sessions cannot fork the
chain. If a record cannot be written, AuditError is raised and the caller must not show the
answer: an answer that is not on the record is not served.

The log can hold user questions; it lives under build/audit/ (git-ignored) by default and
should be kept under the client's retention policy (RAG_AUDIT_DIR).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
GENESIS = "0" * 64
LOCK_TIMEOUT_S = 10.0
STALE_LOCK_S = 60.0


class AuditError(RuntimeError):
    """The audit log could not be written or read; the answer must not be shown."""


def _clean(value):
    """JSON-safe copy: NaN/inf become null (they are not valid JSON), tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _canonical(record: dict) -> bytes:
    return json.dumps({k: v for k, v in record.items() if k != "hash"}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_hash(record: dict) -> str:
    return hashlib.sha256(_canonical(record)).hexdigest()


class _Lock:
    """Cross-process lock: an exclusively created file. A lock older than STALE_LOCK_S is assumed
    to belong to a crashed writer and is removed."""

    def __init__(self, path: Path, timeout: float = LOCK_TIMEOUT_S):
        self.path, self.timeout = path, timeout

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > STALE_LOCK_S:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise AuditError(f"audit log is locked ({self.path}); another writer is stuck")
                time.sleep(0.02)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


class AuditLog:
    def __init__(self, directory: Path, clock=None):
        self.dir = Path(directory)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------ writing
    def _head(self) -> dict:
        """The chain's last record. Normally HEAD.json; but a writer that crashed after appending its
        line and before moving HEAD leaves the files one record ahead, and continuing from the stale
        HEAD would fork the chain, so the newest file's last intact record wins when it is ahead."""
        f = self.dir / "HEAD.json"
        head = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"seq": 0, "hash": GENESIS}
        files = self.files()
        if files:
            last = None
            for line in files[-1].read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line) if line.strip() else None
                except json.JSONDecodeError:
                    continue
                if isinstance(r, dict) and r.get("hash") == record_hash(r):
                    last = r
            if last and last.get("seq", 0) > head["seq"]:
                head = {"seq": last["seq"], "hash": last["hash"], "file": files[-1].name}
        return head

    def append(self, record: dict) -> dict:
        """Add a record to the chain and return it with seq, prev_hash and hash filled in."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with _Lock(self.dir / ".lock"):
                head = self._head()
                now = self.clock()
                # the chain fields come last so a record's own keys can never override them
                full = _clean({**{k: v for k, v in record.items() if k != "hash"}, "schema": SCHEMA,
                               "record_id": uuid.uuid4().hex, "ts": now.isoformat(timespec="milliseconds"),
                               "seq": head["seq"] + 1, "prev_hash": head["hash"]})
                full["hash"] = record_hash(full)
                line = json.dumps(full, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                # never write to a file older than HEAD's (a clock that went backwards would break file order)
                path = self.dir / max(f"audit-{now:%Y-%m-%d}.jsonl", head.get("file", ""))
                # a crashed write can leave a partial last line: start on a fresh line so this record
                # stays intact and the partial one remains visible to verify()
                torn = path.is_file() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(("\n" if torn else "") + line)
                    f.flush()
                    os.fsync(f.fileno())
                tmp = self.dir / "HEAD.json.tmp"
                tmp.write_text(json.dumps({"seq": full["seq"], "hash": full["hash"], "record_id": full["record_id"],
                                           "file": path.name}) + "\n", encoding="utf-8")
                os.replace(tmp, self.dir / "HEAD.json")
                return full
        except AuditError:
            raise
        except (OSError, TypeError, ValueError) as e:
            raise AuditError(f"could not write the audit record: {type(e).__name__}: {e}") from e

    def record_answer(self, session, checked, duration_ms: int, user: str | None = None) -> dict:
        attempts = [a for a in (checked.first_answer, checked.answer) if a is not None]
        return self.append({
            "event": "answer", "user": user, "session_id": session.session_id,
            "release_id": session.release_id, "release_digest": session.toolset.release.manifest.get("content_digest"),
            "model": checked.answer.model, "effort": session.effort,
            "question": attempts[0].question, "answer": checked.answer.text,
            "stop_reason": checked.answer.stop_reason, "ok": checked.answer.ok,
            "flagged": checked.flagged, "revised": checked.revised, "revision_error": checked.revision_error,
            "verification": checked.verification.to_dict(), "first_verification": checked.first_verification.to_dict(),
            "attempts": [_attempt(a) for a in attempts],
            "cited_sources": sorted({c.source_id for c in checked.answer.citations}),
            "usage_total": checked.usage, "duration_ms": duration_ms,
        })

    def record_error(self, session, question: str, error: BaseException, duration_ms: int,
                     user: str | None = None) -> dict:
        return self.append({
            "event": "error", "user": user, "session_id": session.session_id, "release_id": session.release_id,
            "release_digest": session.toolset.release.manifest.get("content_digest"), "model": session.model,
            "effort": session.effort, "question": question, "error": f"{type(error).__name__}: {error}",
            "duration_ms": duration_ms,
        })

    # ------------------------------------------------------------ reading
    def files(self) -> list[Path]:
        return sorted(self.dir.glob("audit-*.jsonl")) if self.dir.is_dir() else []

    def records(self):
        for f in self.files():
            with open(f, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if line.strip():
                        try:
                            yield f.name, n, json.loads(line)
                        except json.JSONDecodeError:
                            yield f.name, n, None

    def find(self, record_id: str) -> dict | None:
        return next((r for _, _, r in self.records() if r and r.get("record_id") == record_id), None)

    def verify(self) -> list[str]:
        """Problems in the chain; empty when every record is intact and in order."""
        problems, expected_seq, prev = [], 1, GENESIS
        for name, n, r in self.records():
            where = f"{name}:{n}"
            if not isinstance(r, dict) or not isinstance(r.get("seq"), int):
                problems.append(f"{where}: " + ("not valid JSON" if r is None else "not an audit record"))
                continue
            if r.get("hash") != record_hash(r):
                problems.append(f"{where}: record {r.get('seq')} has been modified (hash mismatch)")
            if r.get("seq") != expected_seq:
                problems.append(f"{where}: expected record {expected_seq}, found {r.get('seq')} "
                                "(records deleted, inserted or reordered)")
            if r.get("prev_hash") != prev:
                problems.append(f"{where}: record {r.get('seq')} does not link to the record before it")
            expected_seq, prev = r["seq"] + 1, r.get("hash")
        head_file = self.dir / "HEAD.json"
        if head_file.is_file():
            try:
                head = json.loads(head_file.read_text(encoding="utf-8"))
                head["seq"], head["hash"]
            except (ValueError, KeyError, TypeError):
                return problems + ["HEAD.json is unreadable"]
            if head["seq"] != expected_seq - 1 or head["hash"] != prev:
                problems.append(f"HEAD points at record {head['seq']}, but the files end at record {expected_seq - 1} "
                                "(records removed from the end, or HEAD altered)")
        elif expected_seq > 1:
            problems.append("HEAD.json is missing")
        return problems


def result_digest(result) -> str | None:
    """SHA-256 of a tool result in canonical JSON: what a replay on the same release must reproduce."""
    if result is None:
        return None
    text = json.dumps(_clean(result), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_ids(value, out=None) -> list[str]:
    """Every source_id in a tool result, in order, without repeats."""
    out = [] if out is None else out
    if isinstance(value, dict):
        if isinstance(value.get("source_id"), str) and value["source_id"] not in out:
            out.append(value["source_id"])
        for v in value.values():
            _source_ids(v, out)
    elif isinstance(value, list):
        for v in value:
            _source_ids(v, out)
    return out


def _attempt(a) -> dict:
    return {"text": a.text, "stop_reason": a.stop_reason, "turns": a.turns, "budget_exhausted": a.budget_exhausted,
            "usage": a.usage, "model": a.model,
            "tool_calls": [{"name": t.name, "input": t.input, "ok": t.ok, "error": t.error, "ms": t.ms,
                            "source_ids": _source_ids(t.result), "result_sha256": result_digest(t.result)}
                           for t in a.tool_calls],
            # the cited text is tool-result content (up to a whole result block): fingerprint, don't copy
            "citations": [{"source_id": c.source_id, "title": c.title,
                           "cited_text_sha256": hashlib.sha256((c.cited_text or "").encode("utf-8")).hexdigest()}
                          for c in a.citations]}


def ask_audited(session, question: str, audit: AuditLog, user: str | None = None):
    """ask_checked, with the outcome written to the audit log before it is returned. If the question
    fails, an error record is written and the exception re-raised. If the record cannot be written,
    AuditError is raised and the answer must not be shown."""
    from .verifier import ask_checked
    start = time.perf_counter()
    try:
        checked = ask_checked(session, question)
    except Exception as e:
        audit.record_error(session, question, e, int((time.perf_counter() - start) * 1000), user)
        raise
    record = audit.record_answer(session, checked, int((time.perf_counter() - start) * 1000), user)
    return checked, record
