"""Split text into retrieval chunks (spec section 6.1).

Chunks stay inside one group (e.g. one notebook cell or section), hold at most
`max_words` words (375 words is about 500 tokens at ~1.33 tokens per word), and
start with up to `overlap` (15%) of the previous chunk's closing sentences, so a
passage cut at a boundary is still found whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Hashable, Sequence

# Split on whitespace after . ! ? (optionally followed by a closing quote or bracket, which
# stays with its sentence) when the next sentence starts with a capital, digit or opener.
_SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+(?=[\"'“‘(A-Z0-9])")


@dataclass(frozen=True)
class Piece:
    """A paragraph, or part of one, with the metadata a chunk inherits."""
    text: str
    page_start: int
    page_end: int
    group: Hashable            # chunks never span two groups (e.g. chapter + section)
    meta: object = None        # passed through from the chunk's first piece


@dataclass(frozen=True)
class Chunk:
    text: str
    page_start: int
    page_end: int
    group: Hashable
    n_words: int
    overlap_words: int         # leading words repeated from the previous chunk
    meta: object = None


def words(text: str) -> int:
    return len(text.split())


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text.strip()) if s]


def _split_long(p: Piece, max_words: int) -> list[Piece]:
    """Break a paragraph longer than max_words at sentence boundaries."""
    out, buf = [], []
    for s in sentences(p.text):
        if buf and words(" ".join(buf + [s])) > max_words:
            out.append(" ".join(buf))
            buf = []
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return [Piece(t, p.page_start, p.page_end, p.group, p.meta) for t in out]


def _tail(text: str, budget: int) -> str:
    """Closing sentences of `text` totalling at most `budget` words."""
    kept: list[str] = []
    for s in reversed(sentences(text)):
        if words(" ".join([s] + kept)) > budget:
            break
        kept.insert(0, s)
    return " ".join(kept)


def chunk(pieces: Sequence[Piece], max_words: int = 375, overlap: float = 0.15) -> list[Chunk]:
    """Pack pieces into chunks of at most max_words (only a single sentence longer than
    that can exceed it). Short chunks occur only where a group ends."""
    budget = int(max_words * overlap)
    out: list[Chunk] = []
    buf: list[Piece] = []
    carry = ""

    def emit():
        nonlocal buf, carry
        body = " ".join(p.text for p in buf)
        text = f"{carry} {body}".strip() if carry else body
        out.append(Chunk(text, buf[0].page_start, buf[-1].page_end, buf[0].group,
                         words(text), words(carry), buf[0].meta))
        carry = _tail(body, budget)
        buf = []

    limit = max_words - budget          # a piece plus a full carry still fits
    for p in [q for p in pieces for q in (_split_long(p, limit) if words(p.text) > limit else [p])]:
        if buf and p.group != buf[0].group:
            emit()
            carry = ""
        elif buf and words(carry) + sum(words(b.text) for b in buf) + words(p.text) > max_words:
            emit()
        if not buf and carry and words(carry) + words(p.text) > max_words:
            carry = ""
        buf.append(p)
    if buf:
        emit()
    return out
