from rag.ingest.chunking import Piece, chunk, sentences


def _para(n, group="s1", page=1, tag="W"):
    """n words in sentences of ten; the tag is upper case so each sentence starts with a capital."""
    return Piece(" ".join(f"{tag}{i}." if i % 10 == 9 else f"{tag}{i}" for i in range(n)), page, page, group)


def test_sentences_keep_closing_quotes_and_are_lossless():
    text = "He said “no.” Then he left. (It rained.) Next one? Yes! 3 more."
    parts = sentences(text)
    assert parts[0] == "He said “no.”"
    assert " ".join(parts) == text


def test_chunks_respect_max_words_and_groups():
    pieces = [_para(200, "a", 1, "A"), _para(200, "a", 2, "B"), _para(50, "b", 3, "C")]
    out = chunk(pieces, max_words=300)
    assert all(c.n_words <= 300 for c in out)
    assert [c.group for c in out] == ["a", "a", "b"]
    assert out[-1].overlap_words == 0                   # no overlap across groups
    assert (out[1].page_start, out[1].page_end) == (2, 2)


def test_overlap_repeats_the_previous_chunk_tail_within_budget():
    out = chunk([_para(200, tag="A"), _para(200, tag="B")], max_words=300, overlap=0.15)
    second = out[1]
    assert 0 < second.overlap_words <= 45
    lead = " ".join(second.text.split()[:second.overlap_words])
    assert out[0].text.endswith(lead)


def test_long_paragraph_is_split_at_sentences_without_losing_text():
    long = _para(900, tag="X")
    out = chunk([long], max_words=300)
    assert len(out) >= 3 and all(c.n_words <= 300 for c in out)
    rebuilt = " ".join(" ".join(c.text.split()[c.overlap_words:]) for c in out)
    assert rebuilt == long.text


def test_meta_comes_from_the_first_piece():
    out = chunk([Piece("one.", 1, 1, "g", "first"), Piece("two.", 1, 1, "g", "second")])
    assert [c.meta for c in out] == ["first"]
