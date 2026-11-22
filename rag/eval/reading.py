"""How the evaluation reads an answer: which claim each verdict belongs to, what is said to be
untested, which chapters are named, and whether a request was refused (spec 10).

The scorer needs a stricter reading than the verifier. The verifier asks "is anything here wrong?"
and may let a verdict with no named claim through; the scorer asks "was this claim given this
verdict?", so a verdict earns credit only for the claims it is attached to:

  sentence   the group of claims just before it ("H4, H6 and H8 got Mostly holds"), or just after
             when the verdict leads ("Mostly holds: H4, H6, H8"; "Partly holds for H2")
  table      the claim in the verdict's column header, or in its row
  list       a verdict followed by a colon, or leading into a list, covers the items' claims
  carried    a sentence that names no claim takes the last one named ("H2 tests competition. Its
             verdict is Partly holds.")

Markdown emphasis and curly apostrophes are normalised first. Deterministic, like the verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REF = re.compile(
    r"\bH(?P<claim>\d{1,2})(?:[a-d]\b|\b)"
    r"|\b(?i:chapters?|ch\.)\s*(?P<chapters>\d{1,2}\b(?:(?:,?\s*(?:and|&)\s*|\s*[,–-]\s*|\s+to\s+)\d{1,2}\b)*)"
    r"|\b(?P<preface>[Pp]reface)\b")
# what may stand between two members of one group: "H4 (companies), H6 and H8"
GROUP_GAP = re.compile(r"(?:\s*\([^)]*\))?[\s,&/\"'()]*(?:and|plus)?[\s\"'()]*")
SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”)]))\s+(?=[A-Z\"“\[])|;\s+")
LIST_MARK = re.compile(r"^(?:[-+•]|\d{1,2}[.)])\s+")
HEADING = re.compile(r"^#{1,6}\s+")

# a verdict right after one of these, in the same clause, is refused or denied, not given
DENIED = re.compile(r"\bno verdict\b|\b(?:not|never|nor|neither|cannot|won't|refus\w*|declin\w*|rather than|"
                    r"instead of|asked (?:me )?to)\b|n't\b", re.I)
# words that are not denials though they contain one: a quoted title, "bits, not atoms"
NOT_DENIAL = re.compile(r"\"[^\"]*\"|“[^”]*”|,\s*not [a-z]+\b")
# 'saying it "Holds" would be making up a result': a verdict supposed, not given
SUPPOSED = re.compile(r"^[\"”')]*\s*would (?:be|mean)\b")
CLAUSE_RESET = re.compile(r";|,?\s+\b(?:but|while|whereas)\b", re.I)
UNTESTED = re.compile(r"\buntested\b|(?:\bnot|\bnever|n't)\s+(?:(?:been|yet|directly|empirically|get|be)\s+)*test(?:ed)?\b"
                      r"|\bno (?:hypothesis|hypotheses|claim|notebook)\b|\bnone of the hypotheses\b"
                      r"|(?:\bnot|n't)\s+test\b", re.I)
REFUSING = re.compile(r"\b(?:can't|cannot|won't|don't|doesn't|isn't|aren't|not|no|unable|decline|outside|only|"
                      r"read-only|there's no|you'll need|you will need)\b", re.I)


def normalise(text: str) -> str:
    return re.sub(r"\*+|`|__", "", text.replace("’", "'").replace("‘", "'"))


@dataclass(frozen=True)
class Ref:
    key: str                           # "H4" or "ch9"
    start: int
    end: int


def refs(text: str) -> list[Ref]:
    out = []
    for m in REF.finditer(text):
        if m.group("claim"):
            out.append(Ref(f"H{int(m.group('claim'))}", m.start(), m.end()))
        elif m.group("preface"):
            out.append(Ref("ch0", m.start(), m.end()))
        else:
            out += [Ref(f"ch{n}", m.start(), m.end()) for n in sorted(_expand(m.group("chapters")))]
    return out


def _expand(listing: str) -> set[int]:
    """'3, 5 and 8–10' -> {3, 5, 8, 9, 10}"""
    found = {int(n) for n in re.findall(r"\d{1,2}", listing)}
    for a, b in re.findall(r"(\d{1,2})\s*(?:–|-|\bto\b)\s*(\d{1,2})", listing):
        if int(a) < int(b) <= 15:
            found |= set(range(int(a), int(b) + 1))
    return found


def verdicts_in(text: str, verdicts: list[str]) -> list[tuple[str, int, int]]:
    """Verdict phrases, longest first. A multi-word verdict in any case; 'Holds' as written, or in
    lower case where it is called the verdict, and never as the start of 'Holds up' or 'Holds — with a ...'."""
    found: list[tuple[str, int, int]] = []
    for v in sorted(verdicts, key=len, reverse=True):
        if " " in v:
            patterns = [(re.compile(rf"(?<![\w-]){re.escape(v)}(?![\w-])", re.I), 0)]
        else:
            tail = r"(?![\w-]| up\b|[\s,—–-]+with a\b)"
            patterns = [(re.compile(rf"(?<![\w-]){v}{tail}"), 0),
                        (re.compile(rf"\bverdict(?: is| of|:)?\s+[\"'“]?({v}){tail}", re.I), 1)]
        for pattern, group in patterns:
            for m in pattern.finditer(text):
                s, e = m.span(group)
                if not any(s < b and a < e for _, a, b in found):
                    found.append((v, s, e))
    return sorted(found, key=lambda x: x[1])


def _groups(found: list[Ref], text: str) -> list[list[Ref]]:
    groups: list[list[Ref]] = []
    for r in found:
        if groups and (r.start < groups[-1][-1].end
                       or GROUP_GAP.fullmatch(text[groups[-1][-1].end:r.start])):
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _subjects(found: list[Ref]) -> list[Ref]:
    """Claims where any are named, else chapters: 'H3 (chapter 5) Holds' is about H3."""
    return [r for r in found if r.key.startswith("H")] or found


@dataclass(frozen=True)
class Given:
    verdict: str
    subjects: frozenset                # keys; empty when the answer names no subject for it


def _sentence_verdicts(text: str, verdicts: list[str], carried: frozenset) -> list[Given]:
    spans = verdicts_in(text, verdicts)
    if not spans:
        return []
    named = _subjects(refs(text))
    keys = [r.key for r in named]
    if len(spans) > 1 and len(keys) == len(spans) and re.search(r"\brespectively\b", text, re.I):
        return [Given(v, frozenset([k])) for (v, _, _), k in zip(spans, keys)]
    leads = not named or spans[0][1] < named[0].start
    out = []
    for i, (v, s, e) in enumerate(spans):
        before_from = spans[i - 1][2] if i else 0
        clause = text[before_from:s]
        resets = list(CLAUSE_RESET.finditer(clause))
        if resets:
            clause = clause[resets[-1].end():]
        if DENIED.search(NOT_DENIAL.sub(" ", clause)[-50:]) or SUPPOSED.search(text[e:]):
            continue
        if leads:
            until = spans[i + 1][1] if i + 1 < len(spans) else len(text)
            after = [r for r in named if e <= r.start < until]
            groups = _groups(after, text)
            subjects = groups[0] if groups else []
        else:
            groups = _groups([r for r in named if r.start < s], text)
            subjects = groups[-1] if groups else []
        out.append(Given(v, frozenset(r.key for r in subjects) or (carried if not named else frozenset())))
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _row_verdicts(cells: list[str], header: list[str], verdicts: list[str]) -> list[Given]:
    out = []
    named = [_subjects(refs(c)) for c in cells]
    for i, cell in enumerate(cells):
        for v, s, _ in verdicts_in(cell, verdicts):
            if DENIED.search(NOT_DENIAL.sub(" ", cell[:s])):
                continue
            by_column = _subjects(refs(header[i])) if i < len(header) else []
            before = next((n for n in named[:i] if n), [])
            after = next((n for n in named[i + 1:] if n), [])
            subjects = by_column or named[i] or before or after
            out.append(Given(v, frozenset(r.key for r in subjects)))
    return out


def given_verdicts(text: str, verdicts: list[str]) -> list[Given]:
    """Every verdict the answer gives, with the subjects it gives it to."""
    out: list[Given] = []
    header: list[str] = []
    carried: frozenset = frozenset()
    listed: str | None = None          # the verdict that leads into the list being read
    label = False                      # the last line was a heading or a label ("H2: competition")
    for raw in normalise(text).splitlines():
        line = raw.strip()
        if not line:
            if not label:
                carried = frozenset()  # a new paragraph names its own subject
            continue
        label = bool(HEADING.match(line)) or (len(line.split()) <= 12 and line[-1] not in ".!?")
        if line.startswith("|"):
            cells = _cells(line)
            if not header:
                header = cells
            out += _row_verdicts(cells, header, verdicts)
            listed = None
            continue
        header = []
        if HEADING.match(line):
            line, carried = HEADING.sub("", line), frozenset()
        item = LIST_MARK.match(line)
        if item and listed and not verdicts_in(line, verdicts):
            named_first = _groups(_subjects(refs(line)), line)
            if named_first and named_first[0][0].start <= item.end() + 2:
                out.append(Given(listed, frozenset(r.key for r in named_first[0])))
        elif not item:
            listed = None
        for sentence in SENTENCE.split(line):
            found = _sentence_verdicts(sentence, verdicts, carried)
            out += found
            named = _subjects(refs(sentence))
            if named:
                carried = frozenset(r.key for r in _groups(named, sentence)[0])
            if not item and len(found) == 1 and not named and line.endswith(":"):
                listed = found[0].verdict
    return out


def said_untested(text: str, keys: set[str], only_subject: bool) -> bool:
    """One of `keys` (a chapter and its placeholder claim, say) is said to be untested. With
    `only_subject` (the question is about nothing else), a sentence that names no other subject
    counts unless the last one named was a different one."""
    last: set[str] = set()
    list_is_untested = False
    for raw in normalise(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = _cells(line)
            if UNTESTED.search(line):
                named = {r.key for r in refs(line)}
                if re.fullmatch(r"\d{1,2}", cells[0]):
                    named.add(f"ch{int(cells[0])}")
                if named & keys:
                    return True
            continue
        item = LIST_MARK.match(line)
        if list_is_untested:
            # "- 8: Secrets", "8. Secrets" under "These chapters are not tested:"
            m = re.match(r"(?:[-+•]\s*)?(\d{1,2})\b", line)
            if (item or m) and (({r.key for r in refs(line)} | ({f"ch{int(m.group(1))}"} if m else set())) & keys):
                return True
            if not (item or m):
                list_is_untested = False
        for sentence in SENTENCE.split(line):
            named = {r.key for r in refs(sentence)}
            for clause in CLAUSE_RESET.split(sentence):
                if not UNTESTED.search(clause):
                    continue
                here = {r.key for r in refs(clause)}
                if here & keys:
                    return True
                if not here and named and named <= keys:
                    return True
                if not here and not named and (last & keys or (only_subject and not last)):
                    return True
                if clause.rstrip().endswith(":"):
                    list_is_untested = True
            if named:
                last = named
    return False


def chapters_named(text: str) -> set[int]:
    """Chapters named as chapters, or numbering the items of a list or table in an answer about chapters."""
    text = normalise(text)
    found = {int(r.key[2:]) for r in refs(text) if r.key.startswith("ch")}
    if re.search(r"\bchapters?\b", text, re.I):
        found |= {int(n) for n in re.findall(r"^[ \t]*(?:[-+•|][ \t]*)(\d{1,2})(?![.\d%x])\b", text, re.M)}
        found |= {int(n) for n in re.findall(r"^[ \t]*(\d{1,2})[.):][ \t]", text, re.M)}
    return found


def refuses(text: str, topic: str) -> bool:
    """A sentence that both refuses and names what was asked for: 'I can't fault Nvidia' is not a refusal."""
    for part in re.split(r"(?<=[.!?])\s+|\n|;", normalise(text)):
        if REFUSING.search(part) and re.search(topic, part, re.I):
            return True
    return False
