"""Notes parser: the analyst's reading notes -> bullets by chapter + corrections table (spec section 6.1, step 2).

Layout of the notes PDF: chapter headings "N · Title" (bold, 12.5pt), bullets as a
glyph at the left edge with text indented beside it, a page footer, and a final
"Changes from the original notes" table whose rows are separated by horizontal rules.
Every bullet is the analyst's voice; a bullet starting "My view" is marked as opinion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .chapters import BY_NUMBER, CHAPTERS, normalize
from .pdf import Line, Word, group_lines, near, open_pdf, page_words

HEADING, BODY, TABLE, FOOTER = 12.5, 9.8, 8.6, 8.0
BULLETS = {"•", "(cid:127)", "▪", "●", "-"}
CHANGES = normalize("Changes from the original notes")
_HEADING = re.compile(r"^(\d{1,2}|Conclusion|Preface)\s+\S\s+(.+)$")
_SUBITEM = re.compile(r"^\d{1,2}\)\s")


@dataclass(frozen=True)
class NoteChunk:
    id: str
    chapter: int
    chapter_title: str
    page: int
    opinion: bool
    text: str
    voice: str = "analyst"
    source: str = "notes"


@dataclass(frozen=True)
class Correction:
    id: str
    page: int
    original: str
    change: str


def _chapter_for(heading: str):
    """Chapter number for a notes heading, or None for the corrections section."""
    if normalize(heading) == CHANGES:
        return None
    m = _HEADING.match(heading)
    if not m:
        raise ValueError(f"notes heading {heading!r} is not in the form 'N · Title'")
    key, title = m.groups()
    number = {"Preface": 0, "Conclusion": 15}.get(key, None)
    number = int(key) if number is None else number
    expected = BY_NUMBER.get(number)
    if expected is None or normalize(title) != normalize(expected.title):
        raise ValueError(f"notes heading {heading!r} does not match chapter {number} of the book")
    return number


def read_notes(path: Path) -> tuple[list[NoteChunk], list[Correction]]:
    bullets: list[dict] = []
    table_words: list[tuple[int, list[Word], list[float]]] = []
    chapter: int | None = None
    in_table = False

    with open_pdf(path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            words = page_words(page)
            if in_table or any(near(w.size, TABLE) for w in words):
                rules = sorted({round(l["top"], 1) for l in page.lines
                                if abs(l["top"] - l["bottom"]) < 1 and l["x1"] - l["x0"] > 100})
                table_words.append((page_no, [w for w in words if near(w.size, TABLE)], rules))
            for line in group_lines([w for w in words if not near(w.size, TABLE)], page_no):
                if near(line.size, HEADING) and line.bold:
                    chapter = _chapter_for(line.text)
                    in_table = chapter is None
                    continue
                if in_table or chapter is None or line.size < BODY - 1 or line.size > HEADING + 2:
                    continue                          # title block, footer, corrections section
                first = line.words[0].text
                if first in BULLETS:
                    text = Line(line.page, line.words[1:]).text if len(line.words) > 1 else ""
                    bullets.append({"chapter": chapter, "page": page_no, "text": text})
                elif bullets and bullets[-1]["chapter"] == chapter:
                    sep = "\n" if _SUBITEM.match(line.text) else " "
                    bullets[-1]["text"] = (bullets[-1]["text"] + sep + line.text).strip()

    chunks, seq = [], {}
    for b in bullets:
        seq[b["chapter"]] = seq.get(b["chapter"], 0) + 1
        text = b["text"].strip()
        chunks.append(NoteChunk(f"note-{b['chapter']:02d}-{seq[b['chapter']]:02d}", b["chapter"],
                                BY_NUMBER[b["chapter"]].display, b["page"],
                                text.lower().startswith("my view"), text))
    return chunks, _corrections(table_words)


def _corrections(pages: list[tuple[int, list[Word], list[float]]]) -> list[Correction]:
    """Rows between horizontal rules; left column = original note, right = what changed."""
    col_x = None
    for _, words, _ in pages:
        what = [w for w in words if w.text == "What" and "bold" in w.font.lower()]
        if what:
            col_x = what[0].x0 - 2
            break
    if col_x is None:
        return []

    out = []
    for page_no, words, rules in pages:
        for top, bottom in zip(rules, rules[1:]):
            row = [w for w in words if top <= (w.top + w.bottom) / 2 < bottom]
            if not row or any("bold" in w.font.lower() for w in row):
                continue                              # empty band or the header row
            cells = []
            for side in (lambda w: w.x0 < col_x, lambda w: w.x0 >= col_x):
                lines = group_lines([w for w in row if side(w)], page_no)
                cells.append(" ".join(l.text for l in lines).strip())
            out.append(Correction(f"corr-{len(out) + 1:02d}", page_no, *cells))
    return out


def chapters_covered(chunks: list[NoteChunk]) -> list[int]:
    return sorted({c.chapter for c in chunks})


def missing_chapters(chunks: list[NoteChunk]) -> list[int]:
    have = set(chapters_covered(chunks))
    return [c.number for c in CHAPTERS if c.number not in have]
