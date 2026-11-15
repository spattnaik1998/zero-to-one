import pytest

from rag.ingest.notes import _chapter_for, missing_chapters, read_notes


@pytest.fixture
def parsed(notes_pdf):
    return read_notes(notes_pdf)


def test_bullets_by_chapter_with_continuations_and_subitems(parsed):
    chunks, _ = parsed
    assert [(c.id, c.chapter, c.text) for c in chunks] == [
        ("note-01-01", 1, "First bullet starts here and continues on a second line."),
        ("note-01-02", 1, "Second bullet with a list:\n1) first item\n2) second item"),
        ("note-12-01", 12, "My view: computers complement people."),
        ("note-15-01", 15, "Four possible futures."),
    ]
    assert chunks[0].chapter_title == "1 The Challenge of the Future"
    assert {c.voice for c in chunks} == {"analyst"}


def test_only_my_view_is_opinion(parsed):
    chunks, _ = parsed
    assert [c.id for c in chunks if c.opinion] == ["note-12-01"]


def test_title_footer_and_table_stay_out_of_bullets(parsed):
    chunks, _ = parsed
    text = " ".join(c.text for c in chunks)
    for unwanted in ("Notes by chapter", "page 1", "Original note", "Right one line"):
        assert unwanted not in text


def test_corrections_rows_split_on_rules_across_pages(parsed):
    _, corrections = parsed
    assert [(c.id, c.page, c.original, c.change) for c in corrections] == [
        ("corr-01", 1, "Left cell line one left cell line two.", "Right one line."),
        ("corr-02", 1, "Short left.", "Right cell first, right cell second."),
        ("corr-03", 2, "Third original.", "Third change."),
    ]


def test_missing_chapters(parsed):
    chunks, _ = parsed
    assert missing_chapters(chunks) == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]


@pytest.mark.parametrize("heading", ["3 · Wrong Title", "Chapter three"])
def test_unrecognised_headings_are_rejected(heading):
    with pytest.raises(ValueError):
        _chapter_for(heading)
