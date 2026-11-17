"""Notebook splitter: notebooks/*.ipynb -> tagged chunks (spec section 6.1, step 4).

Every notebook opens with a claim cell: a heading naming the hypothesis and its
chapters, Thiel's words as a blockquote, a Claim | Test | Data table of
sub-claims, and a caveats section. The cells after it are methods (markdown and
code), figures and the verdict-inputs cell.

Each chunk carries a kind (claim / method / result / caveat) and a voice:
"thiel" for the blockquoted quotation, "evidence" for everything else. Cell
outputs are not indexed; the numbers live in the summaries and tables.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .chapters import BY_NUMBER
from .chunking import Piece, chunk, words

_TITLE = re.compile(r"^#\s*H(\d+)\s*\S\s*(.+)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_SUB = re.compile(r"\bH(\d)([a-d])\b")         # sub-claims H1a..H9d; extend if hypotheses reach H10
_CAVEATS = re.compile(r"^\*\*(?:[A-Z][a-z]+ )?[Cc]aveats?\.?\*\*\.?", re.M)
_SUBCLAIM_ROW = re.compile(r"^\|\s*\*\*(H\d[a-d])\.\*\*\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
_CHAPTER_TOKEN = re.compile(r"\d+\s*[–—-]\s*\d+|\d+|conclusion|preface", re.I)
MAX_WORDS = 375


@dataclass(frozen=True)
class ClaimCell:
    hypothesis: str                 # "H3"
    title: str                      # heading text without the chapter parenthesis
    chapters: tuple[int, ...]
    quote: str                      # Thiel's words, from the blockquote
    context: str                    # the rest of the claim cell before the caveats
    sub_claims: tuple[dict, ...]    # {"id", "claim", "test", "data"}
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class NotebookChunk:
    id: str
    hypothesis: str
    sub_claim: str | None
    kind: str                       # claim | method | result | caveat
    voice: str                      # thiel | evidence
    section: str | None
    cell: int
    cell_type: str
    chapters: tuple[int, ...]
    notebook: str
    n_words: int
    text: str
    source: str = "notebook"


def parse_chapters(text: str) -> tuple[int, ...]:
    """'*Zero to One*, ch. 3–4' -> (3, 4); 'ch. 1 and conclusion' -> (1, 15); '(2, 6, 13)' -> (2, 6, 13)."""
    if "ch." in text:
        text = text.split("ch.", 1)[1]
    out: list[int] = []
    for tok in _CHAPTER_TOKEN.findall(text):
        low = tok.lower()
        if low == "conclusion":
            out.append(15)
        elif low == "preface":
            out.append(0)
        elif re.search(r"[–—-]", tok):
            a, b = (int(x) for x in re.split(r"\s*[–—-]\s*", tok))
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    bad = [c for c in out if c not in BY_NUMBER]
    if bad:
        raise ValueError(f"chapter reference {text!r} names chapters outside the book: {bad}")
    return tuple(sorted(set(out)))


def _split_title(line: str) -> tuple[str, str, tuple[int, ...]]:
    m = _TITLE.match(line.strip())
    if not m:
        raise ValueError(f"claim cell must start with '# H<n> · Title (ch. …)', got {line!r}")
    number, rest = m.groups()
    paren = rest.rfind("(")
    if paren < 0 or not rest.rstrip().endswith(")"):
        raise ValueError(f"claim heading {line!r} does not end with its chapters in parentheses")
    return f"H{number}", rest[:paren].strip(), parse_chapters(rest[paren + 1:-1])


def parse_claim_cell(src: str) -> ClaimCell:
    lines = src.strip().splitlines()
    hypothesis, title, chapters = _split_title(lines[0])
    body = "\n".join(lines[1:])
    m = _CAVEATS.search(body)
    before, caveat_text = (body[:m.start()], body[m.end():]) if m else (body, "")

    # A bulleted caveat list ends at the first non-bullet line; what follows (e.g. a
    # "Before running" note) goes back to the context. Without bullets, the whole
    # paragraph after the marker is the caveat ("**Interpretation caveat.** ...").
    tail_lines = caveat_text.strip("\n").splitlines()
    bullets, after = [], []
    if any(re.match(r"^[*-]\s+", l) for l in tail_lines):
        for i, l in enumerate(tail_lines):
            if re.match(r"^[*-]\s+", l):
                bullets.append(re.sub(r"^[*-]\s+", "", l).strip())
            elif l.strip():
                after = tail_lines[i:]
                break
        caveats = tuple(bullets)
    else:
        caveats = (caveat_text.strip(),) if caveat_text.strip() else ()

    quote = " ".join(l.lstrip(">").strip() for l in before.splitlines() if l.startswith(">")).strip()
    rest = [l for l in before.splitlines() if not l.startswith(">")] + ([""] + after if after else [])
    sub_claims = tuple({"id": r.group(1), "claim": r.group(2).strip(), "test": r.group(3).strip(),
                        "data": r.group(4).strip()} for r in map(_SUBCLAIM_ROW.match, rest) if r)
    context = "\n".join(rest).strip()
    return ClaimCell(hypothesis, title, chapters, quote, context, sub_claims, caveats)


def read_cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _pieces(text: str, cell: int, group) -> list[str]:
    """Split a long markdown cell at sentences; a long code cell at blank lines."""
    if words(text) <= MAX_WORDS:
        return [text]
    if group == "code":
        out, buf = [], []
        for block in re.split(r"\n\s*\n", text):
            if buf and words("\n\n".join(buf + [block])) > MAX_WORDS:
                out.append("\n\n".join(buf))
                buf = []
            buf.append(block)
        return out + (["\n\n".join(buf)] if buf else [])
    return [c.text for c in chunk([Piece(text, cell, cell, group)], max_words=MAX_WORDS, overlap=0)]


def notebook_chunks(path: Path, root: Path) -> tuple[ClaimCell, list[NotebookChunk]]:
    cells = read_cells(path)
    claim = parse_claim_cell(_src(cells[0]))
    rel = path.relative_to(root).as_posix()
    out: list[NotebookChunk] = []

    def add(kind, voice, section, sub, cell, cell_type, text):
        for part in _pieces(text.strip(), cell, "code" if cell_type == "code" else cell):
            if part.strip():
                out.append(NotebookChunk(f"nb-{claim.hypothesis.lower()}-{len(out) + 1:03d}", claim.hypothesis,
                                         sub, kind, voice, section, cell, cell_type, claim.chapters, rel,
                                         words(part), part.strip()))

    if claim.quote:
        add("claim", "thiel", None, None, 0, "markdown", claim.quote)
    add("claim", "evidence", None, None, 0, "markdown", f"{claim.hypothesis} · {claim.title}\n\n{claim.context}")
    if claim.caveats:
        add("caveat", "evidence", None, None, 0, "markdown", "\n".join(f"- {c}" for c in claim.caveats))

    section = sub = None
    for i, cell in enumerate(cells[1:], 1):
        text = _src(cell)
        if not text.strip():
            continue
        kind = "method"
        if cell["cell_type"] == "markdown":
            first = text.strip().splitlines()[0]
            h = _HEADING.match(first)
            if h:
                heading = h.group(2).strip()
                if len(h.group(1)) <= 2:
                    section = heading
                    sub = None
                s = _SUB.search(heading)
                if s:
                    sub = f"H{s.group(1)}{s.group(2)}"
                if heading.lower().startswith(("figure", "verdict")):
                    kind = "result"
        add(kind, "evidence", section, sub, i, cell["cell_type"], text)
    return claim, out
