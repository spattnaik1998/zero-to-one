import hashlib
import json

from rag.config import load_settings
from rag.sources import describe, find_sources, write_manifest


def _settings(tmp_path):
    return load_settings(environ={"RAG_SOURCES_DIR": str(tmp_path / "sources"),
                                  "RAG_BUILD_DIR": str(tmp_path / "build"),
                                  "OPENAI_EMBEDDING_MODEL": "pinned-model"}, dotenv={})


def test_missing_folders_give_empty_lists(tmp_path):
    assert find_sources(tmp_path) == {"notes": []}


def test_manifest_records_hashes_and_models(tmp_path):
    notes = tmp_path / "sources" / "notes" / "notes.pdf"
    notes.parent.mkdir(parents=True)
    notes.write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "sources" / "notes" / "ignore.txt").write_text("not a pdf")
    (tmp_path / "sources" / "book").mkdir()
    (tmp_path / "sources" / "book" / "book.pdf").write_bytes(b"%PDF-1.4 not part of the corpus")

    s = _settings(tmp_path)
    [entry] = describe(s.sources_dir)
    assert entry.role == "notes" and entry.path == "notes/notes.pdf"
    assert entry.sha256 == hashlib.sha256(b"%PDF-1.4 fake").hexdigest()

    manifest = json.loads(write_manifest(s).read_text())
    assert manifest["models"]["embedding_model"] == "pinned-model"
    assert [x["path"] for x in manifest["sources"]] == ["notes/notes.pdf"]
    assert "book_licensed" not in manifest
