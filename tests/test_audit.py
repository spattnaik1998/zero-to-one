"""Audit log tests: the hash chain, tamper detection, concurrency, failure handling, and the
records written for real (scripted) agent answers."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from rag.agent.audit import GENESIS, AuditError, AuditLog, ask_audited, record_hash
from rag.agent.orchestrator import Session
from rag.config import ROOT


def lines(log):
    return [json.loads(x) for f in log.files() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def rewrite(path, records):
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def log(tmp_path):
    log = AuditLog(tmp_path / "audit")
    for i in range(3):
        log.append({"event": "answer", "question": f"q{i}", "answer": f"a{i}"})
    return log


# ---------------------------------------------------------------- the chain
def test_records_form_a_hash_chain(log):
    recs = lines(log)
    assert [r["seq"] for r in recs] == [1, 2, 3] and recs[0]["prev_hash"] == GENESIS
    assert all(b["prev_hash"] == a["hash"] for a, b in zip(recs, recs[1:]))
    assert all(r["hash"] == record_hash(r) for r in recs) and log.verify() == []
    head = json.loads((log.dir / "HEAD.json").read_text())
    assert head["seq"] == 3 and head["hash"] == recs[-1]["hash"]


def test_an_edited_record_is_detected(log):
    f = log.files()[0]
    recs = lines(log)
    recs[1]["answer"] = "a better answer"
    rewrite(f, recs)
    assert any("record 2 has been modified" in p for p in log.verify())


def test_a_deleted_record_is_detected(log):
    f = log.files()[0]
    recs = lines(log)
    rewrite(f, [recs[0], recs[2]])
    problems = log.verify()
    assert any("expected record 2, found 3" in p for p in problems) and any("does not link" in p for p in problems)


def test_reordered_records_are_detected(log):
    f = log.files()[0]
    recs = lines(log)
    rewrite(f, [recs[0], recs[2], recs[1]])
    assert log.verify()


def test_records_cut_from_the_end_are_detected(log):
    f = log.files()[0]
    rewrite(f, lines(log)[:2])
    assert any("HEAD points at record 3, but the files end at record 2" in p for p in log.verify())


def test_a_forged_record_with_a_valid_hash_is_detected(log):
    f = log.files()[0]
    recs = lines(log)
    forged = {**recs[1], "answer": "forged", "record_id": "x"}
    forged["hash"] = record_hash(forged)                       # a consistent hash, but the chain breaks after it
    rewrite(f, [recs[0], forged, recs[2]])
    assert any("record 3 does not link" in p for p in log.verify())


def test_a_missing_head_is_reported(log):
    (log.dir / "HEAD.json").unlink()
    assert log.verify() == ["HEAD.json is missing"]


def test_invalid_json_lines_are_reported(log):
    f = log.files()[0]
    f.write_text(f.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert any("not valid JSON" in p for p in log.verify())


# ---------------------------------------------------------------- files and time
def test_daily_files_continue_one_chain_and_a_clock_going_back_is_safe(tmp_path):
    days = iter([datetime(2026, 11, 20, 23, 59, tzinfo=timezone.utc), datetime(2026, 11, 21, 0, 1, tzinfo=timezone.utc),
                 datetime(2026, 11, 20, 23, 58, tzinfo=timezone.utc)])          # the clock goes back
    log = AuditLog(tmp_path / "audit", clock=lambda: next(days))
    for i in range(3):
        log.append({"event": "answer", "question": f"q{i}"})
    assert [f.name for f in log.files()] == ["audit-2026-11-20.jsonl", "audit-2026-11-21.jsonl"]
    assert len(log.files()[1].read_text().splitlines()) == 2 and log.verify() == []


# ---------------------------------------------------------------- concurrency and locking
WRITER = """
import sys
sys.path.insert(0, sys.argv[1])
from rag.agent.audit import AuditLog
log = AuditLog(sys.argv[2])
for i in range(int(sys.argv[3])):
    log.append({"event": "answer", "question": f"{sys.argv[4]}-{i}"})
"""


def test_concurrent_writers_keep_one_unbroken_chain(tmp_path):
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(ROOT), str(tmp_path / "audit"), "15", f"p{k}"])
             for k in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    log = AuditLog(tmp_path / "audit")
    recs = lines(log)
    assert len(recs) == 60 and sorted(r["seq"] for r in recs) == list(range(1, 61)) and log.verify() == []


def test_a_held_lock_times_out_and_a_stale_lock_is_cleared(tmp_path, monkeypatch):
    import rag.agent.audit as audit_mod
    log = AuditLog(tmp_path / "audit")
    log.dir.mkdir(parents=True)
    lock = log.dir / ".lock"
    lock.write_text("123")
    monkeypatch.setattr(audit_mod, "LOCK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(audit_mod._Lock.__init__, "__defaults__", (0.2,))
    with pytest.raises(AuditError, match="locked"):
        log.append({"event": "answer"})
    old = time.time() - audit_mod.STALE_LOCK_S - 5
    os.utime(lock, (old, old))                                # a crashed writer's lock
    assert log.append({"event": "answer"})["seq"] == 1 and not lock.exists()


# ---------------------------------------------------------------- failures
def test_an_unwritable_log_raises(tmp_path):
    (tmp_path / "audit").write_text("a file where the directory should be")
    with pytest.raises(AuditError, match="could not write"):
        AuditLog(tmp_path / "audit").append({"event": "answer"})


def test_values_are_made_json_safe_or_refused(tmp_path):
    log = AuditLog(tmp_path / "audit")
    r = log.append({"event": "answer", "value": float("nan"), "items": (1, 2)})
    assert r["value"] is None and r["items"] == [1, 2] and log.verify() == []
    with pytest.raises(AuditError):
        log.append({"event": "answer", "obj": object()})
    assert log.verify() == [] and len(lines(log)) == 1        # the failed record left no trace


# ---------------------------------------------------------------- records of real (scripted) answers
def t(s, cites=()):
    return NS(type="text", text=s, citations=[NS(source=c, title=c, cited_text=c) for c in cites] or None)


def resp(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=10, output_tokens=2, cache_read_input_tokens=5, cache_creation_input_tokens=0))


class FakeClaude:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        self.requests.append(params)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool(id, name, inp):
    return NS(type="tool_use", id=id, name=name, input=inp)


def session(toolset, client):
    return Session(toolset, client, "claude-opus-5-5", today=date(2026, 11, 21))


def test_a_revised_answer_is_recorded_with_both_attempts_and_their_evidence(toolset, tmp_path):
    client = FakeClaude(resp(tool("a", "get_claim", {"id": "H1"}), stop="tool_use"),
                        resp(t("H1 Partly holds, from 999 firms.", ["claim:H1"])),
                        resp(t("H1's verdict is Partly holds.", ["claim:H1"])))
    s = session(toolset, client)
    log = AuditLog(tmp_path / "audit")
    checked, record = ask_audited(s, "H1?", log, user="analyst@example")
    stored = log.find(record["record_id"])
    assert stored == record and log.verify() == []
    assert stored["event"] == "answer" and stored["question"] == "H1?" and stored["user"] == "analyst@example"
    assert stored["answer"] == "H1's verdict is Partly holds." and stored["revised"] and not stored["flagged"]
    assert stored["release_id"] == toolset.release.release_id and stored["release_digest"]
    assert stored["session_id"] == s.session_id and stored["model"] == "claude-opus-5-5" and stored["effort"] == "medium"
    first, second = stored["attempts"]
    assert first["text"] == "H1 Partly holds, from 999 firms." and second["tool_calls"] == []
    call = first["tool_calls"][0]
    assert call["name"] == "get_claim" and call["result"]["results"][0]["verdict"] == "Partly holds"   # the evidence
    assert stored["first_verification"]["passed"] is False and stored["verification"]["passed"] is True
    assert stored["cited_sources"] == ["claim:H1"] and stored["usage_total"]["input_tokens"] == 30
    assert stored["duration_ms"] >= 0


def test_a_failed_question_is_recorded_and_re_raised(toolset, tmp_path):
    class Overloaded(Exception):
        pass
    s = session(toolset, FakeClaude(Overloaded("529 overloaded")))
    log = AuditLog(tmp_path / "audit")
    with pytest.raises(Overloaded):
        ask_audited(s, "q", log)
    [r] = lines(log)
    assert r["event"] == "error" and "Overloaded: 529 overloaded" in r["error"] and r["question"] == "q"


def test_an_answer_that_cannot_be_recorded_is_not_returned(toolset, tmp_path):
    (tmp_path / "audit").write_text("blocked")
    with pytest.raises(AuditError):
        ask_audited(session(toolset, FakeClaude(resp(t("fine")))), "q", AuditLog(tmp_path / "audit"))


def test_records_from_one_session_share_its_id(toolset, tmp_path):
    s = session(toolset, FakeClaude(resp(t("first")), resp(t("second"))))
    log = AuditLog(tmp_path / "audit")
    ask_audited(s, "one", log)
    ask_audited(s, "two", log)
    assert {r["session_id"] for r in lines(log)} == {s.session_id} and log.verify() == []


# ---------------------------------------------------------------- the CLI
@pytest.fixture
def cli(toolset, monkeypatch):
    import rag.agent.orchestrator as orch
    from rag.__main__ import main

    def run(*argv, responses=()):
        monkeypatch.setattr(orch, "open_session", lambda *a, **k: session(toolset, FakeClaude(*responses)))
        return main(list(argv))
    return run


def test_cli_ask_records_the_answer_and_audit_reads_it(cli, capsys):
    assert cli("ask", "q", "--user", "u1", responses=[resp(t("Plain answer."))]) == 0
    out = capsys.readouterr().out
    record_id = out.split("audit record: ")[1].split()[0]
    assert cli("audit", "--verify") == 0 and "1 record(s)" in capsys.readouterr().out
    assert cli("audit", "--show", record_id) == 0
    assert json.loads(capsys.readouterr().out)["user"] == "u1"
    assert cli("audit", "--tail", "5") == 0 and "passed" in capsys.readouterr().out
    assert cli("audit", "--show", "nope") == 1


def test_cli_withholds_the_answer_when_the_log_cannot_be_written(cli, capsys, monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("RAG_AUDIT_DIR", str(blocked))
    assert cli("ask", "q", responses=[resp(t("A secret answer."))]) == 2
    captured = capsys.readouterr()
    assert "answer withheld" in captured.err and "A secret answer" not in captured.out


def test_cli_audit_verify_fails_on_tampering(cli, capsys):
    cli("ask", "q", responses=[resp(t("Plain answer."))])
    log = AuditLog(Path(os.environ["RAG_AUDIT_DIR"]))
    f = log.files()[0]
    f.write_text(f.read_text(encoding="utf-8").replace("Plain answer.", "Edited answer."), encoding="utf-8")
    capsys.readouterr()
    assert cli("audit") == 1 and "has been modified" in capsys.readouterr().out
