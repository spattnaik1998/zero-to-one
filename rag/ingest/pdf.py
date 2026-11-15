"""Line extraction shared by the PDF parsers.

pdfplumber gives words with position, size and font. Words are grouped into lines
by baseline (`bottom`), not by `top`: a line that mixes sizes, such as small caps
after a drop cap, shares a baseline but not a top.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# pdfminer logs a warning for every font missing a bounding box; the text is unaffected.
logging.getLogger("pdfminer").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    font: str


@dataclass(frozen=True)
class Line:
    page: int
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        """Words joined by spaces, except where they touch. pdfplumber splits a word wherever
        the font changes (small caps -> em dash -> italic), and those pieces have no gap."""
        out = self.words[0].text
        for prev, w in zip(self.words, self.words[1:]):
            out += ("" if w.x0 - prev.x1 < 1.0 else " ") + w.text
        return out

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    def _dominant(self, key):
        weight = Counter()
        for w in self.words:
            weight[key(w)] += len(w.text)
        return weight.most_common(1)[0][0]

    @property
    def size(self) -> float:
        """Font size carrying most of the line's characters."""
        return self._dominant(lambda w: round(w.size, 1))

    @property
    def font(self) -> str:
        return self._dominant(lambda w: w.font)

    @property
    def italic(self) -> bool:
        f = self.font.lower()
        return "ital" in f or "oblique" in f

    @property
    def bold(self) -> bool:
        return "bold" in self.font.lower()


def open_pdf(path: Path):
    return pdfplumber.open(str(path))


def page_words(page) -> list[Word]:
    return [Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"], w["size"], w["fontname"].split("+")[-1])
            for w in page.extract_words(extra_attrs=["size", "fontname"])]


def group_lines(words: list[Word], page_no: int, tol: float = 2.5) -> list[Line]:
    """Cluster words whose baselines are within `tol` points, then order each line left to right."""
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.bottom, w.x0)):
        if lines and w.bottom - lines[-1][0].bottom <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [Line(page_no, tuple(sorted(ws, key=lambda w: w.x0))) for ws in lines]


def near(a: float, b: float, tol: float = 0.4) -> bool:
    return abs(a - b) <= tol
