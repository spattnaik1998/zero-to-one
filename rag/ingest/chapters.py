"""The book's structure: preface (0), chapters 1-14, conclusion (15).

The notes parser and the notebook headings map onto this list, so a chapter number
means the same thing in the notes index, the chapter map and the claims registry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    label: str | None = None    # "Preface" / "Conclusion"; None for numbered chapters

    @property
    def display(self) -> str:
        return f"{self.label}: {self.title}" if self.label else f"{self.number} {self.title}"


CHAPTERS = (
    Chapter(0, "Zero to One", "Preface"),
    Chapter(1, "The Challenge of the Future"),
    Chapter(2, "Party Like It's 1999"),
    Chapter(3, "All Happy Companies Are Different"),
    Chapter(4, "The Ideology of Competition"),
    Chapter(5, "Last Mover Advantage"),
    Chapter(6, "You Are Not a Lottery Ticket"),
    Chapter(7, "Follow the Money"),
    Chapter(8, "Secrets"),
    Chapter(9, "Foundations"),
    Chapter(10, "The Mechanics of Mafia"),
    Chapter(11, "If You Build It, Will They Come?"),
    Chapter(12, "Man and Machine"),
    Chapter(13, "Seeing Green"),
    Chapter(14, "The Founder's Paradox"),
    Chapter(15, "Stagnation or Singularity?", "Conclusion"),
)
BY_NUMBER = {c.number: c for c in CHAPTERS}


def normalize(title: str) -> str:
    """Compare titles ignoring case, curly quotes, punctuation and spacing."""
    t = title.replace("’", "'").replace("‘", "'").casefold()
    return re.sub(r"[^a-z0-9]", "", t)
