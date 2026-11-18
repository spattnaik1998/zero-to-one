"""Hybrid search over one index: BM25 keywords + dense vectors, fused by reciprocal rank.

Used by the build's checks now and by the agent's search tools later (spec section 6.2).
Filters on metadata (hypothesis, chapter, kind, voice, ...) are applied before scoring,
so a filtered search ranks only the documents that qualify.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60
_TOKEN = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[a-z]+(?:'[a-z]+)?")
STOPWORDS = frozenset("""a an and are as at be by did do does for from has have how in is it its of on or
that the their this to was were what when which who why will with""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers; keeps '$150,000', '21%', '2014' as single tokens."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    meta: dict
    score: float                # fused RRF score
    bm25_rank: int | None       # 1-based; None when the document has no keyword match
    dense_rank: int


def _matches(meta: dict, where: dict) -> bool:
    for k, want in where.items():
        have = meta.get(k)
        wants = want if isinstance(want, (list, tuple, set)) else [want]
        if isinstance(have, list):
            if not any(w in have for w in wants):
                return False
        elif have not in wants:
            return False
    return True


class HybridIndex:
    def __init__(self, name: str, ids: list[str], texts: list[str], embed_texts: list[str],
                 metas: list[dict], vectors: np.ndarray, model: str):
        self.name, self.ids, self.texts, self.metas, self.vectors, self.model = name, ids, texts, metas, vectors, model
        self.tokens = [tokenize(t) for t in embed_texts]
        self._bm25: dict[tuple[int, ...], BM25Okapi] = {}   # per filtered document set; the index never changes

    @classmethod
    def load(cls, db: Path, name: str) -> "HybridIndex":
        con = duckdb.connect(str(db), read_only=True)
        try:
            info = con.execute("SELECT model FROM index_info WHERE index_name = ?", [name]).fetchone()
            if info is None:
                raise KeyError(f"no index named {name!r} in {db}")
            rows = con.execute("SELECT doc_id, text, embed_text, meta, embedding FROM documents "
                               "WHERE index_name = ? ORDER BY doc_id", [name]).fetchall()
        finally:
            con.close()
        return cls(name, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                   [json.loads(r[3]) for r in rows], np.array([r[4] for r in rows], dtype=np.float32), info[0])

    def search(self, query: str, query_vector: np.ndarray, k: int = 5, where: dict | None = None,
               mode: str = "hybrid") -> list[Hit]:
        """mode: 'hybrid' (default), 'bm25' or 'dense' (the last two for evaluation)."""
        keep = [i for i, m in enumerate(self.metas) if not where or _matches(m, where)]
        if not keep:
            return []
        dense = self.vectors[keep] @ query_vector.astype(np.float32)
        dense_order = np.argsort(-dense, kind="stable")
        dense_rank = {keep[j]: r + 1 for r, j in enumerate(dense_order)}

        q = tokenize(query)
        bm25_rank: dict[int, int] = {}
        if q:
            key = tuple(keep)
            if key not in self._bm25:
                self._bm25[key] = BM25Okapi([self.tokens[i] for i in keep])
            scores = self._bm25[key].get_scores(q)
            matched = [j for j in np.argsort(-scores, kind="stable") if scores[j] > 0]
            bm25_rank = {keep[j]: r + 1 for r, j in enumerate(matched)}

        def fused(i):
            if mode == "dense":
                return 1 / dense_rank[i]
            if mode == "bm25":
                return 1 / bm25_rank[i] if i in bm25_rank else 0.0
            return 1 / (RRF_K + dense_rank[i]) + (1 / (RRF_K + bm25_rank[i]) if i in bm25_rank else 0.0)

        ranked = sorted(keep, key=lambda i: (-fused(i), dense_rank[i]))
        if mode == "bm25":
            ranked = [i for i in ranked if i in bm25_rank]
        return [Hit(self.ids[i], self.texts[i], self.metas[i], fused(i), bm25_rank.get(i), dense_rank[i])
                for i in ranked[:k]]
