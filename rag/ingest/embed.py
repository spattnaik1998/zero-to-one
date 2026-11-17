"""Text embeddings with the pinned OpenAI model, cached on disk (spec section 6.2).

The cache is keyed on (model, text), so a rebuild sends only new or changed text to
OpenAI, and the same text always gets the same vector. Vectors are L2-normalised so
a dot product is the cosine similarity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import duckdb
import numpy as np


class Embedder(Protocol):
    model: str
    dims: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(v).all():
        raise ValueError("embedding contains a zero or non-finite vector")
    return (v / norms).astype(np.float32)


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """(model, text) -> vector, in a small DuckDB file under build/cache/."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute("CREATE TABLE IF NOT EXISTS vectors (key VARCHAR PRIMARY KEY, model VARCHAR, vector FLOAT[])")

    def get(self, model: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        keys = [_key(model, t) for t in texts]
        if not keys:
            return {}
        rows = self.con.execute("SELECT key, vector FROM vectors WHERE key IN (SELECT unnest(?))", [keys]).fetchall()
        return {k: np.asarray(v, dtype=np.float32) for k, v in rows}

    def put(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        self.con.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                             [(_key(model, t), model, v.tolist()) for t, v in zip(texts, vectors)])

    def close(self) -> None:
        self.con.close()


class OpenAIEmbedder:
    """Embeds with the pinned OpenAI model. Counts API calls and tokens for the build report."""

    def __init__(self, model: str, api_key: str, cache: EmbeddingCache | None = None, batch_size: int = 256):
        import openai                                   # only needed when embeddings are actually built
        self.model, self.cache, self.batch_size = model, cache, batch_size
        self.client = openai.OpenAI(api_key=api_key)
        self.dims = 0
        self.api_calls = self.tokens = self.cache_hits = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        cached = self.cache.get(self.model, texts) if self.cache else {}
        todo = [t for t in dict.fromkeys(texts) if _key(self.model, t) not in cached]
        self.cache_hits += len(texts) - sum(1 for t in texts if _key(self.model, t) not in cached)
        fresh: dict[str, np.ndarray] = {}
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            r = self.client.embeddings.create(model=self.model, input=batch)
            self.api_calls += 1
            self.tokens += r.usage.total_tokens
            vecs = normalise(np.array([d.embedding for d in sorted(r.data, key=lambda d: d.index)], dtype=np.float32))
            fresh.update({_key(self.model, t): v for t, v in zip(batch, vecs)})
            if self.cache:
                self.cache.put(self.model, batch, vecs)
        vectors = {**cached, **fresh}
        out = np.stack([vectors[_key(self.model, t)] for t in texts]) if texts else np.zeros((0, 0), np.float32)
        self.dims = out.shape[1] if len(out) else self.dims
        return normalise(out) if len(out) else out
