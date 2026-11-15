"""Synthetic PDFs with the same typography as the book and the notes.

The real sources are copyrighted or private and never reach CI, so the parser
tests run on these fixtures. Text is placeholder; only the layout matters.
"""
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

H = letter[1]


def put(c, x, base, text, font="Times-Roman", size=15.0):
    """Draw text with its baseline `base` points from the top; return the x where it ends."""
    c.setFont(font, size)
    c.drawString(x, H - base, text)
    return x + c.stringWidth(text, font, size)


def drop_cap(c, letter_, base):
    return put(c, 77, base, letter_, size=45.0)


@pytest.fixture
def book_pdf(tmp_path) -> Path:
    path = tmp_path / "book.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    sp = 4

    # p1: front matter, ignored
    put(c, 77, 100, "Copyright fixture line.", size=11.2)
    c.showPage()

    # p2: preface with label, title, drop cap and small-caps opening
    put(c, 77, 125, "Preface", "Times-Italic", 21.2)
    put(c, 200, 185, "ZERO TO ONE", size=21.2)
    x = drop_cap(c, "E", 266) + 2
    x = put(c, x, 248, "VERY MOMENT", size=11.2) + sp
    put(c, x, 248, "in this fixture happens once.")
    put(c, 112, 266, "This line sits beside the cap.")
    put(c, 77, 284, "Back at the margin, the paragraph ends.")
    put(c, 92, 302, "A new paragraph starts here and continues on the dot-")
    c.showPage()

    # p3: continuation, section with a superscript, caption, subsection, small caps at the margin
    put(c, 77, 90, "com page that follows.")
    put(c, 200, 140, "A FIXTURE SECTION", size=21.2)
    x = put(c, 92, 180, "The value grows to 10")
    x = put(c, x, 174, "30", size=11.2)
    put(c, x, 180, "—a large number.")
    put(c, 300, 220, "Figure caption text", size=11.2)
    put(c, 77, 260, "A Fixture Subsection", "Times-Italic", 16.9)
    put(c, 92, 300, "Text under the subsection.")
    put(c, 77, 318, "SMALL CAPS AT THE MARGIN.", size=11.2)
    c.showPage()

    # p4: chapter 1
    put(c, 150, 121, "THE CHALLENGE OF THE FUTURE", size=21.2)
    x = drop_cap(c, "W", 194) + 2
    x = put(c, x, 176, "HENEVER", size=11.2) + sp
    put(c, x, 176, "a fixture opens, it says so.")
    put(c, 125, 194, "beside the drop cap here.")
    put(c, 77, 212, "And back to the margin.")
    c.showPage()

    # p5: conclusion
    put(c, 77, 125, "Conclusion", "Times-Italic", 21.2)
    put(c, 150, 185, "STAGNATION OR SINGULARITY?", size=21.2)
    x = drop_cap(c, "I", 266) + 2
    x = put(c, x, 248, "F THE FIXTURE", size=11.2) + sp
    put(c, x, 248, "ends, it ends here.")
    c.showPage()

    # p6: back matter, must stop the parser
    put(c, 77, 125, "Acknowledgments", "Times-Italic", 21.2)
    put(c, 77, 200, "Thanks to everyone.")
    c.showPage()
    c.save()
    return path


def _rule(c, top):
    c.setLineWidth(0.5)
    c.line(71, H - top, 540, H - top)


@pytest.fixture
def notes_pdf(tmp_path) -> Path:
    path = tmp_path / "notes.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    bullet = lambda base, text: (put(c, 71, base, "•", "Helvetica", 10), put(c, 83, base, text, size=9.8))  # noqa: E731

    put(c, 71, 88, "Zero to One", "Times-Bold", 20)
    put(c, 71, 104, "Notes by chapter.", "Times-Italic", 10)
    put(c, 71, 134, "1 · The Challenge of the Future", "Times-Bold", 12.5)
    bullet(150, "First bullet starts here and")
    put(c, 83, 163, "continues on a second line.", size=9.8)
    bullet(180, "Second bullet with a list:")
    put(c, 89, 193, "1) first item", size=9.8)
    put(c, 89, 206, "2) second item", size=9.8)
    put(c, 71, 236, "12 · Man and Machine", "Times-Bold", 12.5)
    bullet(252, "My view: computers complement people.")
    put(c, 71, 282, "Conclusion · Stagnation or Singularity?", "Times-Bold", 12.5)
    bullet(298, "Four possible futures.")
    put(c, 71, 330, "Changes from the original notes", "Times-Bold", 12.5)
    for top in (348, 366, 396, 424):
        _rule(c, top)
    put(c, 75, 360, "Original note", "Times-Bold", 8.6)
    put(c, 254, 360, "What changed", "Times-Bold", 8.6)
    put(c, 75, 378, "Left cell line one", size=8.6)
    put(c, 75, 389, "left cell line two.", size=8.6)
    put(c, 254, 378, "Right one line.", size=8.6)
    put(c, 75, 410, "Short left.", size=8.6)
    put(c, 254, 405, "Right cell first,", size=8.6)
    put(c, 254, 416, "right cell second.", size=8.6)
    put(c, 464, 760, "Zero to One · page 1", size=8.0)
    c.showPage()

    for top in (60, 78, 100):
        _rule(c, top)
    put(c, 75, 72, "Original note", "Times-Bold", 8.6)
    put(c, 254, 72, "What changed", "Times-Bold", 8.6)
    put(c, 75, 92, "Third original.", size=8.6)
    put(c, 254, 92, "Third change.", size=8.6)
    put(c, 464, 760, "Zero to One · page 2", size=8.0)
    c.showPage()
    c.save()
    return path
