import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag.config import ROOT, load_settings
from rag.ingest.embed import EmbeddingCache, OpenAIEmbedder, normalise
from rag.ingest.index import _context, all_docs, build_indexes, content_hash
from rag.retrieval import HybridIndex, tokenize


def test_tokenize_keeps_money_percent_and_years():
    assert tokenize("CEO pay of $150,000, a 21% margin in 2014, and Google's") == \
        ["ceo", "pay", "$150,000", "21%", "margin", "2014", "google's"]


def test_context_drops_a_label_repeated_by_the_next():
    assert _context("H3 Durability", "H3c", "H3c · Are the giants durable?", None) == \
        "H3 Durability · H3c · Are the giants durable?"


def test_documents_carry_context_and_metadata(candidates):
    docs = {d.doc_id: d for d in all_docs(candidates)}
    assert docs["note-12-01"].embed_text.startswith("12 Man and Machine. Analyst's view:")
    assert docs["nb-h3-001"].embed_text.startswith("H3 Durability · H3c · Are the giants durable? (method):")
    assert docs["table:h3c_top20_2014"].text == ("Table h3c_top20_2014: 20 rows. Columns: firm (str; Apple Inc.); "
                                                 "rank_2025 (float64; 2.0 to 221.0)")
    assert docs["figure:h1_fig1_ccdf"].embed_text.startswith("H1 The power law. Figure h1_fig1_ccdf")
    assert docs["nb-h3-002"].meta["kind"] == "caveat" and docs["note-09-01"].meta["chapter"] == 9


def test_build_and_search(candidates, hash_embedder, tmp_path):
    db = tmp_path / "indexes.duckdb"
    emb = hash_embedder
    report = build_indexes(candidates, db, emb)
    assert report["counts"] == {"notes": 3, "analysis": 2, "catalog": 2} and report["dims"] == 64
    assert report["content_hash"] == content_hash(all_docs(candidates))

    notes = HybridIndex.load(db, "notes")
    q = "how much should a CEO be paid"
    hits = notes.search(q, emb.embed([q])[0], k=3)
    assert hits[0].doc_id == "note-09-01" and hits[0].bm25_rank == 1
    assert np.allclose(np.linalg.norm(notes.vectors, axis=1), 1.0)

    # filters apply before ranking; list-valued metadata matches any member
    assert [h.doc_id for h in notes.search(q, emb.embed([q])[0], where={"opinion": True})] == ["note-12-01"]
    analysis = HybridIndex.load(db, "analysis")
    q2 = "firms acquired or failed"
    assert [h.doc_id for h in analysis.search(q2, emb.embed([q2])[0], where={"kind": "caveat", "chapters": 5})] == ["nb-h3-002"]
    assert analysis.search(q2, emb.embed([q2])[0], where={"hypothesis": "H9"}) == []

    # keyword-only mode returns only documents with a keyword match
    q3 = "zebra giraffe"
    assert notes.search(q3, emb.embed([q3])[0], mode="bm25") == []
    assert len(notes.search(q3, emb.embed([q3])[0], mode="dense")) == 3

    with pytest.raises(KeyError):
        HybridIndex.load(db, "book")


def test_duplicate_document_ids_are_rejected(candidates):
    f = candidates / "notes_chunks.jsonl"
    f.write_text(f.read_text() + "\n" + f.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="duplicate"):
        all_docs(candidates)


def test_openai_embedder_batches_orders_counts_and_caches(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            self.embeddings = self

        def create(self, model, input):
            calls.append(list(input))
            data = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0, 0.0]) for i, t in enumerate(input)]
            return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(total_tokens=10 * len(input)))

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    cache = EmbeddingCache(tmp_path / "cache.duckdb")
    emb = OpenAIEmbedder("m", "key", cache, batch_size=2)
    texts = ["a", "bb", "ccc", "a"]                       # "a" twice: embedded once
    v = emb.embed(texts)
    assert calls == [["a", "bb"], ["ccc"]] and emb.api_calls == 2 and emb.tokens == 30
    assert v.shape == (4, 3) and np.allclose(v[0], v[3]) and not np.allclose(v[0], v[1])   # order kept
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)

    again = OpenAIEmbedder("m", "key", cache, batch_size=2)
    assert np.allclose(again.embed(texts), v) and again.api_calls == 0 and again.cache_hits == 4
    other_model = OpenAIEmbedder("other", "key", cache)
    other_model.embed(["a"])
    assert other_model.api_calls == 1                     # the cache is keyed on the model too
    cache.close()


def test_index_build_requires_the_registry_and_a_key(candidates, tmp_path):
    from rag.ingest.run import IndexBuildError, ingest_index
    build = candidates.parent
    (candidates / "claims_registry.json").unlink()
    settings = load_settings(environ={"RAG_BUILD_DIR": str(build), "OPENAI_API_KEY": "k",
                                      "OPENAI_EMBEDDING_MODEL": "m"}, dotenv={})
    with pytest.raises(IndexBuildError, match="claims_registry.json"):
        ingest_index(settings)                            # fails before any API call
    with pytest.raises(IndexBuildError, match="OPENAI_API_KEY"):
        ingest_index(load_settings(environ={"RAG_BUILD_DIR": str(build)}, dotenv={}))


def test_normalise_rejects_zero_vectors():
    with pytest.raises(ValueError):
        normalise(np.zeros((1, 3), np.float32))


INDEX_DB = ROOT / "build" / "candidates" / "indexes.duckdb"


@pytest.mark.skipif(not INDEX_DB.is_file() or not load_settings().openai_api_key,
                    reason="needs the built indexes and an OpenAI key (never in CI)")
def test_real_retrieval_smoke():
    from rag.eval.retrieval_smoke import run
    s = load_settings()
    cache = EmbeddingCache(s.build_dir / "cache" / "embeddings.duckdb")
    try:
        r = run(INDEX_DB, OpenAIEmbedder(s.embedding_model, s.openai_api_key, cache))
    finally:
        cache.close()
    assert r["totals"]["hybrid"][5] == r["n"]             # every question's answer in the top 5
    assert r["totals"]["hybrid"][1] >= 0.8 * r["n"]
