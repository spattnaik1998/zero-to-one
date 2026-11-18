import json
import shutil
import stat
import subprocess

import pytest

from rag.config import ROOT, load_settings
from rag.ingest.index import build_indexes
from rag.release import (ARTIFACTS, Check, GateReport, check_git_clean, check_index, check_provenance,
                         current_release_id, list_releases, open_release, publish, set_current, text_sha256)


NEWLINE, CRLF = bytes([10]), bytes([13, 10])
LF_JSON = b'{' + NEWLINE + b' "a": 1' + NEWLINE + b'}' + NEWLINE


def _settings(tmp_path, **env):
    return load_settings(environ={"RAG_BUILD_DIR": str(tmp_path / "build"), "OPENAI_EMBEDDING_MODEL": "hash-test",
                                  "RAG_SOURCES_DIR": str(tmp_path / "sources"), **env}, dotenv={})


# ---------------------------------------------------------------- provenance
def _output(root, summaries: dict, attest: dict | None = None, table: str = "firm,x\nApple Inc.,1\n"):
    (root / "output" / "tables").mkdir(parents=True)
    for name, data in summaries.items():
        (root / "output" / name).write_text(json.dumps(data))
    (root / "output" / "tables" / "h1_t.csv").write_text(table)
    if attest is not None:
        (root / "output" / "provenance.json").write_text(json.dumps({"summaries": attest}))


def test_stamped_live_summaries_pass(tmp_path):
    _output(tmp_path, {"h1_summary.json": {"a": 1, "_provenance": {"synthetic": False}}})
    assert check_provenance(tmp_path).ok


def test_a_synthetic_stamp_is_refused(tmp_path):
    _output(tmp_path, {"h1_summary.json": {"a": 1, "_provenance": {"synthetic": True}}})
    c = check_provenance(tmp_path)
    assert not c.ok and "synthetic run" in c.detail


def test_unstamped_summary_needs_a_matching_attestation(tmp_path):
    _output(tmp_path, {"h1_summary.json": {"a": 1}}, attest={})
    assert "no provenance stamp" in check_provenance(tmp_path).detail
    f = tmp_path / "output" / "h1_summary.json"
    (tmp_path / "output" / "provenance.json").write_text(json.dumps({"summaries": {"h1_summary.json": {"sha256": text_sha256(f)}}}))
    assert check_provenance(tmp_path).ok
    f.write_text(json.dumps({"a": 2}))                    # rewritten after attestation, e.g. by a synthetic run
    assert not check_provenance(tmp_path).ok


def test_attestation_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(LF_JSON)
    crlf.write_bytes(LF_JSON.replace(NEWLINE, CRLF))
    assert text_sha256(lf) == text_sha256(crlf)             # a Windows and a Linux checkout agree
    crlf.write_bytes(LF_JSON.replace(b"1", b"2").replace(NEWLINE, CRLF))
    assert text_sha256(lf) != text_sha256(crlf)             # content changes still count


def test_synthetic_firm_names_are_refused(tmp_path):
    _output(tmp_path, {"h1_summary.json": {"_provenance": {"synthetic": False}}}, table="firm,x\nFIRM 12,1\n")
    c = check_provenance(tmp_path)
    assert not c.ok and "FIRM <n>" in c.detail


def test_the_real_outputs_are_live():
    """Runs in CI: output/ and its attestation are committed."""
    c = check_provenance(ROOT)
    assert c.ok, c.detail


# ---------------------------------------------------------------- git
def test_uncommitted_release_inputs_are_refused_unless_allowed(tmp_path):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "README.md").write_text("x")
    run("add", ".")
    run("commit", "-q", "-m", "x")
    assert check_git_clean(tmp_path, allow_dirty=False).ok
    (tmp_path / "README.md").write_text("changed")
    assert not check_git_clean(tmp_path, allow_dirty=False).ok
    assert check_git_clean(tmp_path, allow_dirty=True).ok


# ---------------------------------------------------------------- index
def test_index_must_match_the_candidates_and_the_pinned_model(candidates, hash_embedder, tmp_path):
    build_indexes(candidates, candidates / "indexes.duckdb", hash_embedder)
    settings = _settings(tmp_path)
    assert check_index(candidates, settings).ok
    assert "settings pin" in check_index(candidates, _settings(tmp_path, OPENAI_EMBEDDING_MODEL="other")).detail
    notes = candidates / "notes_chunks.jsonl"
    notes.write_text(notes.read_text().replace("board of three", "board of five"))
    c = check_index(candidates, settings)
    assert not c.ok and "notes index is out of date" in c.detail


# ---------------------------------------------------------------- publish and open
@pytest.fixture
def built(tmp_path):
    settings = _settings(tmp_path)
    cands = settings.build_dir / "candidates"
    cands.mkdir(parents=True)
    for name in ARTIFACTS:
        (cands / name).write_text(f"contents of {name}" if name != "claims_registry.json" else
                                  json.dumps([{"id": "H1", "status": "final", "verdict": "Partly holds"}]))
    (tmp_path / "output" / "figures").mkdir(parents=True)
    (tmp_path / "output" / "figures" / "h1_fig1.png").write_bytes(b"png")
    return settings, GateReport([Check("all", True, "ok")])


def test_publish_is_immutable_versioned_and_idempotent(built, tmp_path):
    settings, report = built
    first = publish(settings, report, root=tmp_path)
    assert first["new"] and current_release_id(settings) == first["release_id"]
    rel = open_release(settings)
    assert rel.manifest["claims"] == [{"id": "H1", "status": "final", "verdict": "Partly holds"}]
    assert set(rel.manifest["files"]) == set(ARTIFACTS) | {"figures/h1_fig1.png"}
    assert not (rel.file("analysis.duckdb").stat().st_mode & stat.S_IWRITE)          # read-only
    assert publish(settings, report, root=tmp_path) == {**first, "new": False}         # same content, same release
    with pytest.raises(KeyError):
        rel.file("not_in_release.txt")


def test_a_failed_gate_publishes_nothing(built, tmp_path):
    settings, _ = built
    with pytest.raises(RuntimeError):
        publish(settings, GateReport([Check("x", False, "no")]), root=tmp_path)
    assert not (settings.build_dir / "releases").exists()


def test_tampering_is_detected(built, tmp_path):
    settings, report = built
    rel = open_release(settings, publish(settings, report, root=tmp_path)["release_id"])
    f = rel.file("indexes.duckdb")
    f.chmod(stat.S_IWRITE | stat.S_IREAD)
    f.write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        open_release(settings)
    f.unlink()
    with pytest.raises(ValueError, match="differ from its manifest"):
        open_release(settings)


def test_new_content_is_a_new_release_and_pin_rolls_back(built, tmp_path):
    settings, report = built
    first = publish(settings, report, root=tmp_path)["release_id"]
    (settings.build_dir / "candidates" / "figures.json").write_text("changed")
    second = publish(settings, report, root=tmp_path)["release_id"]
    assert second != first and current_release_id(settings) == second
    set_current(settings, first)
    assert current_release_id(settings) == first
    assert [r["release_id"] for r in list_releases(settings) if r["current"]] == [first]
    with pytest.raises(FileNotFoundError):
        set_current(settings, "r00000000-nope")


def test_no_release_yet(tmp_path):
    with pytest.raises(FileNotFoundError, match="no release published"):
        open_release(_settings(tmp_path))
