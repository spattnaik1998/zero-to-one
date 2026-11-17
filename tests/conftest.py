"""A synthetic PDF with the same typography as the analyst's notes.

The parser tests run on it so they cover layout cases the real notes do not
contain. Text is placeholder; only the layout matters.
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
