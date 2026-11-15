import pytest

from rag.ingest.book import book_chunks, check_structure, read_book


@pytest.fixture
def parsed(book_pdf):
    return read_book(book_pdf)


def test_finds_parts_with_canonical_titles_and_pages(parsed):
    parts, _ = parsed
    assert [(p.number, p.title, p.page_start, p.page_end) for p in parts] == [
        (0, "Preface: Zero to One", 2, 3),
        (1, "1 The Challenge of the Future", 4, 4),
        (15, "Conclusion: Stagnation or Singularity?", 5, 5),
    ]


def test_incomplete_book_fails_structure_check(parsed):
    with pytest.raises(ValueError, match="missing"):
        check_structure(parsed[0])


def test_drop_cap_and_small_caps_join_the_first_sentence(parsed):
    _, paras = parsed
    first = {p.chapter: p.text for p in reversed(paras)}
    assert first[0] == ("EVERY MOMENT in this fixture happens once. This line sits beside the cap. "
                        "Back at the margin, the paragraph ends.")
    assert first[1].startswith("WHENEVER a fixture opens, it says so. beside the drop cap here.")
    assert first[15] == "IF THE FIXTURE ends, it ends here."


def test_paragraph_runs_across_a_page_break_with_hyphen_kept(parsed):
    _, paras = parsed
    p = next(p for p in paras if "dot-com" in p.text)
    assert p.text == "A new paragraph starts here and continues on the dot-com page that follows."
    assert (p.page_start, p.page_end) == (2, 3)


def test_sections_subsections_superscripts_and_small_caps(parsed):
    _, paras = parsed
    by_text = {p.text: p for p in paras}
    sup = by_text["The value grows to 10^30—a large number."]
    assert (sup.section, sup.subsection) == ("A FIXTURE SECTION", None)
    sub = by_text["Text under the subsection. SMALL CAPS AT THE MARGIN."]
    assert (sub.section, sub.subsection) == ("A FIXTURE SECTION", "A Fixture Subsection")


def test_front_matter_captions_and_back_matter_are_excluded(parsed):
    _, paras = parsed
    text = " ".join(p.text for p in paras)
    for unwanted in ("Copyright", "Figure caption", "Thanks to everyone", "Preface", "ZERO TO ONE"):
        assert unwanted not in text


def test_chunks_carry_chapter_metadata(parsed):
    _, paras = parsed
    chunks = book_chunks(paras)
    assert [c.id for c in chunks if c.chapter == 0][0] == "book-00-001"
    assert {c.voice for c in chunks} == {"thiel"}
    sub = next(c for c in chunks if "Text under the subsection" in c.text)
    assert (sub.chapter_title, sub.section) == ("Preface: Zero to One", "A FIXTURE SECTION")
    # A chunk records the subsection it starts in; this one starts before the subsection heading.
    assert sub.subsection is None and sub.text.startswith("The value grows to 10^30")
