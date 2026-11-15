"""Checks against the real book and notes. They run only where sources/ holds them
(never in CI: the files are git-ignored), and guard against a new edition or a
revised notes file silently breaking the parsers."""
import re

import pytest

from rag.config import ROOT
from rag.ingest.book import book_chunks, check_structure, read_book
from rag.ingest.notes import read_notes

BOOK = sorted((ROOT / "sources" / "book").glob("*.pdf"))
NOTES = sorted((ROOT / "sources" / "notes").glob("*.pdf"))


@pytest.mark.skipif(len(BOOK) != 1, reason="book PDF not present in sources/book")
def test_real_book():
    parts, paras = read_book(BOOK[0])
    check_structure(parts)
    assert all(a.page_end < b.page_start for a, b in zip(parts, parts[1:]))
    first = {p.chapter: p.text for p in reversed(paras)}
    assert first[1].startswith("WHENEVER I INTERVIEW")
    chunks = book_chunks(paras)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(0 < c.n_words <= 375 for c in chunks)
    assert {c.chapter for c in chunks} == set(range(16))


@pytest.mark.skipif(len(NOTES) != 1, reason="notes PDF not present in sources/notes")
def test_real_notes():
    chunks, corrections = read_notes(NOTES[0])
    assert {c.chapter for c in chunks} == set(range(1, 16))
    assert [c.chapter for c in chunks if c.opinion] == [12]
    assert len(corrections) == 15
    text = " ".join(c.text for c in chunks)
    for fact in ("37 cents", "21%", "75%", "34%", "$150,000"):
        assert fact in text
    assert all(c.original and c.change for c in corrections)
    assert not re.search(r"page \d", text)
