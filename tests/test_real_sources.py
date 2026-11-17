"""Checks against the real notes in sources/notes/ (committed, so this runs in CI too).
They guard against a revised notes file silently breaking the parser."""
import re

import pytest

from rag.config import ROOT
from rag.ingest.notes import read_notes

NOTES = sorted((ROOT / "sources" / "notes").glob("*.pdf"))


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
