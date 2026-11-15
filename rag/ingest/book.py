"""Book parser: PDF -> paragraphs -> chunks with chapter, section and page (spec section 6.1, step 1).

The layout rules come from the typography of the development copy (body 15pt,
headings 21.2pt, subsections 16.9pt italic, a 45pt drop cap opening each of the 16
parts). `check_structure` fails loudly if a different edition breaks them, so the
licensed copy gets the same checks before it is trusted.

Page numbers are PDF page numbers, not printed page numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .chapters import CHAPTERS, Chapter, match_title, normalize
from .chunking import Piece, chunk
from .pdf import Line, Word, group_lines, near, open_pdf, page_words

BODY, SMALL, HEADING, SUBSECTION, DROP_CAP_MIN = 15.0, 11.2, 21.2, 16.9, 40.0
INDENT = 8.0                                   # paragraph indent beyond the left margin
BACK_MATTER = {normalize(t) for t in ("Acknowledgments", "Acknowledgements", "Index", "About the Authors")}


@dataclass(frozen=True)
class Part:
    """A detected chapter, with the PDF pages it spans."""
    number: int
    title: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Paragraph:
    chapter: int
    section: str | None
    subsection: str | None
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class BookChunk:
    id: str
    chapter: int
    chapter_title: str
    section: str | None
    subsection: str | None
    page_start: int
    page_end: int
    n_words: int
    overlap_words: int
    text: str
    voice: str = "thiel"
    source: str = "book"


def _join(a: str, b: str) -> str:
    return a + b if a.endswith(("-", "—", "–")) else f"{a} {b}"


def _attach_superscripts(words: list[Word]) -> list[Word]:
    """Move small raised or lowered words that touch a body word onto that word's baseline.

    Exponents and stacked fractions sit off the baseline, so they would otherwise form
    lines of their own and be dropped. A raised word following its base is marked '^':
    10 + raised 30 reads '10^30', and raised 1, '/', lowered π read '1/π'.
    """
    body = [w for w in words if near(w.size, BODY)]
    overlaps = lambda w, b: w.top < b.bottom and w.bottom > b.top  # noqa: E731
    out = []
    for w in words:
        if w.size < BODY - 2 and len(w.text) <= 4:
            left = next((b for b in body if abs(w.x0 - b.x1) < 1.5 and overlaps(w, b)), None)
            right = next((b for b in body if abs(b.x0 - w.x1) < 1.5 and overlaps(w, b)), None)
            base = left or right
            if base and abs(w.bottom - base.bottom) > 2:
                raised = left is not None and w.bottom < base.bottom - 2 and not (right and right.text == "/")
                w = replace(w, text=("^" if raised else "") + w.text, bottom=base.bottom, size=base.size)
        out.append(w)
    return out


def read_book(path: Path) -> tuple[list[Part], list[Paragraph]]:
    """Walk the PDF and return the parts found and their paragraphs, in reading order."""
    parts: list[dict] = []
    paras: list[Paragraph] = []
    cur: dict | None = None          # paragraph being built
    section = subsection = None
    chapter: Chapter | None = None
    margin = None

    def close():
        nonlocal cur
        if cur:
            paras.append(Paragraph(chapter.number, section, subsection, cur["p0"], cur["p1"], cur["text"].strip()))
        cur = None

    with open_pdf(path) as pdf:
        last_page = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, 1):
            words = _attach_superscripts(page_words(page))
            caps = sorted((w for w in words if w.size >= DROP_CAP_MIN), key=lambda w: w.top)
            lines = group_lines([w for w in words if w.size < DROP_CAP_MIN], page_no)

            labels = [normalize(l.text) for l in lines if l.italic and near(l.size, HEADING)]
            if chapter and any(t in BACK_MATTER for t in labels):
                last_page = page_no - 1
                break

            cap: Word | None = caps[0] if caps else None
            if cap:
                title = " ".join(l.text for l in lines
                                 if l.bottom < cap.top and near(l.size, HEADING) and not l.italic)
                if chapter:
                    close()
                    parts[-1]["page_end"] = page_no - 1
                chapter = match_title(title)
                parts.append({"number": chapter.number, "title": chapter.display, "page_start": page_no})
                section = subsection = None
            if chapter is None:
                continue                                  # front matter

            body_x = [l.x0 for l in lines if near(l.size, BODY) and not (cap and _beside(l, cap))]
            margin = min(body_x) if body_x else margin
            pending_cap = cap.text if cap else ""
            last_kind = None
            for l in lines:
                if cap and l.bottom < cap.top:
                    continue                              # label and title above the drop cap
                kind = _kind(l, cap, margin)
                if kind == "heading":
                    close()
                    section = _join(section, l.text) if last_kind == "heading" and section else l.text
                    subsection = None
                elif kind == "subsection":
                    close()
                    subsection = _join(subsection, l.text) if last_kind == "subsection" and subsection else l.text
                elif kind == "body":
                    text = l.text
                    if pending_cap:
                        text, pending_cap = pending_cap + text, ""
                        close()
                    indented = margin is not None and l.x0 >= margin + INDENT and not (cap and _beside(l, cap))
                    if cur is None or indented:
                        close()
                        cur = {"p0": page_no, "p1": page_no, "text": text}
                    else:
                        cur["text"] = _join(cur["text"], text)
                        cur["p1"] = page_no
                last_kind = kind
        if chapter:
            close()
            parts[-1]["page_end"] = last_page
    return [Part(**p) for p in parts], paras


def _beside(line: Line, cap: Word) -> bool:
    """A line set to the right of the drop cap, which is indented but not a new paragraph."""
    return line.top < cap.bottom and line.x0 >= cap.x1 - 3


def _kind(line: Line, cap: Word | None, margin: float | None) -> str:
    if near(line.size, HEADING):
        return "label" if line.italic else "heading"
    if near(line.size, SUBSECTION) and line.italic:
        return "subsection"
    if near(line.size, BODY):
        return "body"
    if near(line.size, SMALL):
        if cap and _beside(line, cap):
            return "body"                             # small-caps opening words after the drop cap
        if margin is not None and abs(line.x0 - margin) < 2:
            return "body"                             # a run of small caps inside body text
    return "skip"                                     # captions, figure lettering, symbols


def check_structure(parts: list[Part]) -> None:
    """Raise if the parser did not find all 16 parts, in order."""
    found = [p.number for p in parts]
    expected = [c.number for c in CHAPTERS]
    if found != expected:
        missing = sorted(set(expected) - set(found))
        raise ValueError(f"book structure not recognised: found parts {found}; missing {missing}")


def book_chunks(paras: list[Paragraph], max_words: int = 375) -> list[BookChunk]:
    """Chunk within each chapter section; a chunk records the subsection it starts in."""
    titles = {c.number: c.display for c in CHAPTERS}
    pieces = [Piece(p.text, p.page_start, p.page_end, (p.chapter, p.section), p.subsection) for p in paras]
    out, seq = [], {}
    for c in chunk(pieces, max_words=max_words):
        ch, section = c.group
        seq[ch] = seq.get(ch, 0) + 1
        out.append(BookChunk(f"book-{ch:02d}-{seq[ch]:03d}", ch, titles[ch], section, c.meta,
                             c.page_start, c.page_end, c.n_words, c.overlap_words, c.text))
    return out
