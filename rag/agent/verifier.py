"""Answer verifier: the answer contract, checked in code before an answer is shown (spec 6.4).

Deterministic rules, not a second model: the same answer always gets the same result, the
check costs nothing, and it cannot be argued with. Every rule compares the answer with the
tool results in the session so far (a follow-up may rely on earlier results).

  numbers    every number in the answer is supported by a tool result or the question, at its
             written precision. A number written with a unit word (thousand, million, bn, ...) is
             compared in absolute terms; one without is compared as written or as a percentage.
             Years, chapter numbers, claim ids, list markers and the publication date are not
             checked. Results of literal-only queries (no FROM) and echoed SQL are not evidence.
  verdicts   a verdict phrase must have come from a tool result, and is attached to the nearest
             claim or chapter named in its sentence: it must be that claim's verdict, and never
             belong to an untested or provisional claim or an untested chapter. A verdict that
             is denied ("no verdict, 'Holds' or anything else") is not a stated verdict.
  untested   an untested chapter that the question or answer is about must be called untested
             in a sentence that names it
  citations  every cited source_id was returned by a tool; a fact stated in an uncited paragraph
             must be stated in a cited paragraph too (also at higher precision); a long uncited
             paragraph is a warning
  voices     a paragraph citing the analyst's notes says they are the analyst's; one citing
             Thiel's quoted words names Thiel
  advice     no buy / sell / hold recommendation (a refusal that repeats the question is not one)

`ask_checked` runs a question, verifies, sends one revision request listing the failures, and
verifies again; an answer that still fails is returned flagged with its violations.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import asdict, dataclass, field

from ..units import column_unit

log = logging.getLogger(__name__)

LONG_PARAGRAPH_WORDS = 40
UNIT_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
               "trillion": 1e12, "tn": 1e12}
NUMBER = re.compile(
    r"(?<![\w.$/])(?:US)?(?P<sign>[-−])?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%)?(?:\s?(?P<unit>thousand|million|billion|trillion|bn|mn|tn|k|m)\b)?")
SENTENCE_END = re.compile(r"[.!?](?=\s+[\"'“‘*(\[A-Z0-9]|\s*$)|\n")
CHAPTER_MENTION = re.compile(r"\b(?:chapters?|ch\.)\s*(\d{1,2}(?:\s*(?:,|and|&|–|-|to)\s*\d{1,2})*)", re.I)
CLAIM_MENTION = re.compile(r"\bH(\d{1,2})[a-d]?\b")
UNTESTED_WORDS = re.compile(r"\b(?:untested|not (?:been )?tested|no hypothesis|isn't tested|hasn't been tested|"
                            r"no verdict|doesn't test|does not test|never tested|pending|provisional)\b", re.I)
DENIAL = re.compile(r"\b(?:no|not|never|nor|neither|without|isn't|wasn't|aren't|doesn't|didn't|cannot|can't)\b", re.I)
REFUSAL = re.compile(r"\b(?:not|never|can't|cannot|won't|don't|doesn't|whether|if)\b", re.I)
ANALYST_WORDS = re.compile(r"\b(?:analyst|notes?|reading|annotat)", re.I)
ADVICE = tuple(re.compile(p, re.I) for p in (
    r"\b(?:you|investors?)\s+(?:should|ought to|must|need to)\s+(?:buy|sell|short|hold|invest in|load up on)\b",
    r"\bI(?:\s+would|'d)?\s+(?:recommend|suggest|advise)\s+(?:buying|selling|holding|shorting|investing|that you (?:buy|sell))",
    r"\b(?:strong\s+)?(?:buy|sell)\s+(?:rating|recommendation|signal)\b",
    r"\bis\s+(?:a\s+)?(?:good|great|bad|strong)\s+(?:buy|investment)\b",
))


@dataclass
class Violation:
    rule: str
    detail: str
    severity: str = "error"            # "error" fails the answer; "warning" is reported only


@dataclass
class Verification:
    violations: list[Violation] = field(default_factory=list)
    numbers_checked: int = 0
    numbers_supported: int = 0

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, rule: str, detail: str, severity: str = "error") -> None:
        if not any(x.rule == rule and x.detail == detail for x in self.violations):
            self.violations.append(Violation(rule, detail, severity))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "numbers_checked": self.numbers_checked,
                "numbers_supported": self.numbers_supported, "violations": [asdict(v) for v in self.violations]}


# ---------------------------------------------------------------- numbers
@dataclass(frozen=True)
class Num:
    literal: str                       # as written, commas removed
    value: float                       # as written (no sign, no unit)
    decimals: int
    unit: float | None                 # multiplier of a unit word, if one follows
    percent: bool
    start: int


def _parse(m: re.Match) -> Num:
    literal = m.group("num").replace(",", "")
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    unit = m.group("unit")
    return Num(literal, float(literal), decimals, UNIT_SCALES.get(unit.lower()) if unit else None,
               bool(m.group("pct")), m.start("num"))


def _numbers_in(text: str) -> set[float]:
    return {_parse(m).value for m in NUMBER.finditer(text)}


def _checked_numbers(text: str):
    """Numbers the rule applies to: not a year, chapter number, claim id, list marker or the
    publication date (16 September 2014)."""
    for m in NUMBER.finditer(text):
        n = _parse(m)
        start = n.start
        plain_int = n.decimals == 0 and "," not in m.group("num") and not n.unit and not n.percent \
            and "$" not in m.group(0)
        if plain_int and 1900 <= n.value <= 2100:
            continue                                            # a year
        if plain_int and len(n.literal) == 2 and re.search(r"(?:19|20)\d\d\s?[–-]$", text[max(0, start - 6):start]):
            continue                                            # the end of a year range: 2011–14, 1948-73
        if plain_int and n.value == 16 and re.match(r"\s+September\b", text[m.end():]):
            continue                                            # the publication date
        if re.search(r"\b(?:chapters?|ch\.)\s*(?:\d{1,2}\s*(?:,|and|&|–|-|to)\s*)*$", text[max(0, start - 40):start],
                     re.I):
            continue                                            # chapter 9, chapters 9 and 10
        line_start = text.rfind("\n", 0, start) + 1
        if text[line_start:start].strip() == "" and re.match(r"\d+[.)]\s", text[start:]):
            continue                                            # a numbered-list marker
        yield n


@dataclass
class Evidence:
    numbers: set[float]                # every number as the tools returned it
    sources: dict[str, str]            # source_id -> voice
    text: str                          # every string in the tool results, for verdict provenance
    # numbers whose unit is unknown: a unit word in the answer may put them at any scale
    free: set[float] | None = None
    # numbers whose column or key names their scale (gain_bn), as absolute amounts
    absolute: set[float] = field(default_factory=set)

    def __post_init__(self):
        if self.free is None:
            self.free = set(self.numbers)

    @classmethod
    def from_calls(cls, calls, question: str = "") -> "Evidence":
        ev = cls(set(), {}, "", free=set())
        texts = []
        for c in calls:
            if not (c.ok and c.result):
                continue
            literal_only = c.name == "query_table" and not re.search(r"\bfrom\b", str(c.input.get("sql", "")), re.I)
            _walk(c.result, ev, texts, collect_numbers=not literal_only)
        q = _numbers_in(question)
        ev.numbers |= q
        ev.free |= q
        ev.text = "\n".join(texts)
        return ev

    def add(self, value: float, unit: dict | None = None) -> None:
        """A number from a tool result; `unit` from its column or key name (rag.units.column_unit)."""
        v = abs(float(value))
        self.numbers.add(v)
        if unit is None:
            self.free.add(v)
        elif unit["scale"]:
            self.absolute.add(v * unit["scale"])
        # a percentage has no scale: it supports "5.4%" or "5.4", never "5.4 million"

    def supports(self, n: Num) -> bool:
        """Is `n` supported at its written precision? With a unit word, compare absolute amounts:
        a number whose column names its scale counts only at that scale ("13.1" in op_income_bn is
        13.1 billion, not 13.1 million); one of unknown unit may be at any scale. Without a unit
        word, compare as written or as a percentage <-> fraction."""
        tol = 0.5 * 10 ** -n.decimals + 1e-9
        if n.unit:
            target, tol = n.value * n.unit, tol * n.unit
            candidates = itertools.chain((v * s for v in self.free for s in (1.0, 1e3, 1e6, 1e9, 1e12)),
                                         self.absolute)
        else:
            # a percentage may restate a fraction (0.7 -> 70%); a plain number may restate a percentage
            # as a fraction (70 -> 0.70); never the other way round, or 3,991 would "support" 40%
            target = n.value
            candidates = (v * s for v in self.numbers for s in ((1.0, 100.0) if n.percent else (1.0, 0.01)))
        return any(abs(c - target) <= max(tol, abs(target) * 1e-9) for c in candidates)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _walk(value, ev: Evidence, texts: list, collect_numbers: bool = True, key: str | None = None,
          unit: dict | None = None):
    """Collect sources, strings and numbers from a tool result. Numbers get the unit their column
    or key names, where the result says which column or key they belong to."""
    if isinstance(value, dict):
        if "source_id" in value:
            ev.sources[value["source_id"]] = value.get("voice", "evidence")
        if collect_numbers and isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list):
            _walk_rows(value, ev, texts)                       # a query result: units by column
            rest = {k: v for k, v in value.items() if k not in ("rows",)}
            for k, v in rest.items():
                _walk(v, ev, texts, False if k == "columns" else collect_numbers, k)
            return
        # a summary value ("summary:H2:airlines_op_margin_pct") or a schema card column ("gain_bn")
        own = None
        sid = value.get("source_id")
        if isinstance(sid, str) and sid.startswith("summary:"):
            own = column_unit(sid.rsplit(":", 1)[-1])
        elif isinstance(value.get("name"), str) and ("min" in value or "max" in value):
            own = column_unit(value["name"])
        for k, v in value.items():
            scoped = own if k in ("value", "min", "max", "examples") else None
            _walk(v, ev, texts, collect_numbers, k, scoped)
    elif isinstance(value, list):
        for v in value:
            _walk(v, ev, texts, collect_numbers, key, unit)
    elif isinstance(value, bool) or value is None:
        return
    elif _is_num(value):
        if collect_numbers:
            ev.add(value, unit)
    elif isinstance(value, str):
        if key in ("sql", "label"):                    # the model's own SQL is not evidence
            return
        texts.append(value)
        if collect_numbers:
            for v in _numbers_in(value):
                ev.add(v, unit)


def _walk_rows(result: dict, ev: Evidence, texts: list) -> None:
    """Query rows: a cell takes its column's unit; in a row_label/value table (h1b_wealth_creation)
    the `value` cell takes the unit its row's label names ("net_gain_neutral_tn"). Other cells in
    that row keep their own unit, or none."""
    columns = [str(c) for c in result["columns"]]
    units = [column_unit(c) for c in columns]
    label_col = columns.index("row_label") if "row_label" in columns else None
    for row in result["rows"]:
        if not isinstance(row, list):
            continue
        row_unit = column_unit(row[label_col]) if label_col is not None and label_col < len(row) \
            and isinstance(row[label_col], str) else None
        for i, cell in enumerate(row):
            u = units[i] if i < len(units) else None
            name = columns[i] if i < len(columns) else None
            _walk(cell, ev, texts, True, name, u if u is not None or name != "value" else row_unit)


# ---------------------------------------------------------------- text structure
def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence around [start, end): split at . ! ? followed by space and a capital, or a
    newline, so decimals, 'ch. 9' and 'U.S.' do not split it."""
    left = 0
    for m in SENTENCE_END.finditer(text, 0, start):
        left = m.end()
    m = SENTENCE_END.search(text, end)
    return left, m.start() if m else len(text)


def _sentence(text: str, start: int, end: int) -> str:
    a, b = _sentence_bounds(text, start, end)
    return text[a:b]


def _verdict_spans(text: str, verdicts: list[str]):
    """(verdict, start, end) occurrences, longest phrase first; multi-word verdicts also match in
    lower case ('mostly holds'), a one-word verdict ('Holds') only as written."""
    taken, found = [], []
    for v in sorted(verdicts, key=len, reverse=True):
        flags = re.I if " " in v else 0
        for m in re.finditer(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text, flags):
            if not any(m.start() < e and s < m.end() for s, e in taken):
                taken.append((m.start(), m.end()))
                found.append((v, m.start(), m.end()))
    return sorted(found, key=lambda x: x[1])


def _denied(text: str, start: int) -> bool:
    """A verdict preceded, within its sentence, by a denial: 'gave it no verdict, "Holds" or ...'."""
    a, _ = _sentence_bounds(text, start, start)
    return bool(DENIAL.search(text[max(a, start - 50):start]))


def _subject(text: str, start: int, end: int):
    """The claim id or chapter a verdict at [start, end) is attached to: the nearest one named before
    it in the sentence, else the nearest after it. Returns ("claim", "H3") / ("chapter", 9) / None."""
    a, b = _sentence_bounds(text, start, end)
    mentions = [("claim", f"H{m.group(1)}", m.start()) for m in CLAIM_MENTION.finditer(text, a, b)]
    for m in CHAPTER_MENTION.finditer(text, a, b):
        first = int(re.findall(r"\d{1,2}", m.group(1))[0])
        mentions.append(("chapter", first, m.start()))
    before = [x for x in mentions if x[2] < start]
    if before:
        kind, key, _ = max(before, key=lambda x: x[2])
        return kind, key
    after = [x for x in mentions if x[2] >= end]
    if after:
        kind, key, _ = min(after, key=lambda x: x[2])
        return kind, key
    return None


def _is_number(fact: str) -> bool:
    return bool(re.fullmatch(r"\d*\.?\d+", fact))


def _facts(text: str, verdicts: list[str]) -> set[str]:
    """Checkable facts a paragraph states: its numbers (as written, commas dropped) and the verdicts
    it asserts (not ones it denies)."""
    facts = {n.literal for n in _checked_numbers(text)}
    facts |= {v for v, s, e in _verdict_spans(text, verdicts) if not _denied(text, s)}
    return facts


def _paragraphs(segments: list[dict]):
    """(paragraph text, cited source_ids) using the citation positions in the segments."""
    full, spans, pos = "", [], 0
    for seg in segments:
        spans.append((pos, pos + len(seg["text"]), [s for s in seg["sources"] if s]))
        full += seg["text"]
        pos += len(seg["text"])
    for m in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", full):
        sources = sorted({s for a, b, ss in spans if a < m.end() and m.start() < b for s in ss})
        if m.group(0).strip():
            yield m.group(0).strip(), sources


# ---------------------------------------------------------------- the rules
def verify(answer, toolset, prior_calls=(), question: str | None = None) -> Verification:
    """Check one answer. `prior_calls`: tool calls whose results were already in the session's
    history (earlier questions, or the first attempt of a revision); `question`: the user's question."""
    v = Verification()
    if not answer.ok:
        v.add("incomplete", f"the answer did not complete (stop reason {answer.stop_reason})")
        return v
    text = answer.text
    calls = list(prior_calls) + list(answer.tool_calls)
    question = question or answer.question
    evidence = Evidence.from_calls(calls, question)
    claims = toolset.claims
    verdicts = sorted({c["verdict"] for c in claims.values() if c["verdict"]})

    # numbers
    for n in _checked_numbers(text):
        v.numbers_checked += 1
        if evidence.supports(n):
            v.numbers_supported += 1
        else:
            context = text[max(0, n.start - 50): n.start + len(n.literal) + 20].replace("\n", " ")
            v.add("numbers", f"{n.literal!r} is not supported by any tool result: ...{context}...")

    # verdicts
    lowered_evidence = evidence.text.lower()
    for verdict, s, e in _verdict_spans(text, verdicts):
        if _denied(text, s):
            continue                                    # "no verdict, 'Holds' or anything else"
        written = text[s:e]
        if verdict.lower() not in lowered_evidence:
            v.add("verdicts", f"verdict {written!r} did not come from a tool result in this session")
            continue
        subject = _subject(text, s, e)
        if subject is None:
            continue
        kind, key = subject
        if kind == "claim" and key in claims:
            c = claims[key]
            if c["status"] != "final":
                v.add("verdicts", f"verdict {written!r} given to {key}, which is {c['status'].replace('_', ' ')} "
                                  "and has no verdict")
            elif c["verdict"].lower() != verdict.lower():
                v.add("verdicts", f"{key}'s verdict is {c['verdict']!r}, not {written!r}")
        elif kind == "chapter" and key in toolset.chapters:
            ch = toolset.chapters[key]
            if not ch["tested"]:
                v.add("verdicts", f"verdict {written!r} given to chapter {key}, which no hypothesis tests")
            elif verdict.lower() not in {claims[h]["verdict"].lower() for h in ch["hypotheses"] if claims[h]["verdict"]}:
                v.add("verdicts", f"verdict {written!r} is not the verdict of any claim testing chapter {key}")

    # untested chapters the question or answer is about
    asked_about = {int(n) for m in CHAPTER_MENTION.finditer(question + "\n" + text)
                   for n in re.findall(r"\d{1,2}", m.group(1))}
    seen = set()
    for call in calls:
        if not (call.ok and call.name == "get_chapter"):
            continue
        item = call.result["results"][0]
        number = item["chapter"]
        if item["tested"] or number in seen:
            continue
        seen.add(number)
        names = [rf"\b(?:chapters?|ch\.)\s*(?:\d{{1,2}}\s*(?:,|and|&|–|-|to)\s*)*{number}\b"]
        names += [rf"\b{re.escape(h)}\b" for h in item["untested_placeholders"]]
        title = item["title"].split(" ", 1)[-1] if item["title"][:1].isdigit() else item["title"]
        names.append(re.escape(title))
        relevant = number in asked_about or any(re.search(p, question + "\n" + text, re.I) for p in names[1:])
        if not relevant:
            continue
        said = any(UNTESTED_WORDS.search(_sentence(text, m.start(), m.end()))
                   for p in names for m in re.finditer(p, text, re.I))
        if not said:
            v.add("untested", f"chapter {number} is untested, and no sentence about it says so")

    # citations and voices
    cited = {s for seg in answer.segments for s in seg["sources"] if s}
    for s in sorted(cited - set(evidence.sources)):
        v.add("citations", f"cites {s!r}, which no tool returned")
    paragraphs = list(_paragraphs(answer.segments))
    # facts stated in cited paragraphs: a summary may restate them without citing again, also at
    # lower precision ("5.4%" restating a cited "5.43%")
    cited_facts = {f for para, sources in paragraphs if sources for f in _facts(para, verdicts)}
    cited_numbers = Evidence({float(f) for f in cited_facts if _is_number(f)}, {}, "")
    for para, sources in paragraphs:
        if not sources:
            uncited = set()
            for n in _checked_numbers(para):
                if n.literal not in cited_facts and not cited_numbers.supports(n):
                    uncited.add(n.literal)
            uncited |= {f for f in _facts(para, verdicts) if not _is_number(f)} - cited_facts
            if uncited:
                shown = ", ".join(sorted(uncited)[:6])
                v.add("citations", f"a paragraph states facts no cited paragraph supports ({shown}): {para[:90]!r}")
            elif len(para.split()) >= LONG_PARAGRAPH_WORDS:
                v.add("citations", f"a long paragraph has no citation: {para[:90]!r}", "warning")
            continue
        voices = {evidence.sources.get(s) for s in sources}
        if "analyst" in voices and not ANALYST_WORDS.search(para):
            v.add("voices", f"cites the analyst's notes without saying so: {para[:90]!r}")
        if "thiel" in voices and "Thiel" not in para:
            v.add("voices", f"cites Thiel's words without naming him: {para[:90]!r}")

    # investment advice (a refusal that repeats the question's wording is not advice)
    for pattern in ADVICE:
        for m in pattern.finditer(text):
            a, _ = _sentence_bounds(text, m.start(), m.end())
            if not REFUSAL.search(text[max(a, m.start() - 40):m.start()]):
                v.add("advice", f"reads as investment advice: {m.group(0)!r}")
    return v


# ---------------------------------------------------------------- verify, revise once, flag
def revision_request(verification: Verification) -> str:
    problems = "\n".join(f"- [{x.rule}] {x.detail}" for x in verification.errors)
    return ("An automated check of your previous answer found these problems:\n"
            f"{problems}\n\n"
            "Rewrite your complete answer to the original question so that none of these problems remain. "
            "Support every number with a tool result (call tools again if needed) or remove it; give verdicts "
            "only as the tools state them; label the analyst's notes and Thiel's words; cite every paragraph "
            "that states a fact. Do not mention this check.")


@dataclass
class CheckedAnswer:
    answer: object                     # the final Answer
    verification: Verification
    first_verification: Verification
    revised: bool
    first_answer: object | None = None
    revision_error: str | None = None  # the revision could not run; the first answer is returned flagged

    @property
    def flagged(self) -> bool:
        return not self.verification.passed

    @property
    def usage(self) -> dict:
        """Tokens across both attempts."""
        total = dict(self.answer.usage)
        if self.first_answer is not None:
            total = {k: total.get(k, 0) + self.first_answer.usage.get(k, 0) for k in set(total) | set(self.first_answer.usage)}
        return total

    @property
    def all_tool_calls(self) -> list:
        return (list(self.first_answer.tool_calls) if self.first_answer is not None else []) + list(self.answer.tool_calls)

    def to_dict(self) -> dict:
        d = self.answer.to_dict()
        d.update(question=self.first_answer.question if self.first_answer else self.answer.question,
                 verification=self.verification.to_dict(), first_verification=self.first_verification.to_dict(),
                 revised=self.revised, flagged=self.flagged, revision_error=self.revision_error,
                 usage_total=self.usage,
                 first_answer=self.first_answer.to_dict() if self.first_answer is not None else None)
        return d


def ask_checked(session, question: str) -> CheckedAnswer:
    before = list(session.calls_history)                # results already in the history
    first = session.ask(question)
    v1 = verify(first, session.toolset, prior_calls=before)
    if v1.passed or not first.ok:
        return CheckedAnswer(first, v1, v1, revised=False)
    try:
        second = session.ask(revision_request(v1))
    except Exception as e:                              # context limit, API error: keep the first answer
        log.warning("revision failed: %s", e)
        return CheckedAnswer(first, v1, v1, revised=False, revision_error=f"{type(e).__name__}: {e}")
    v2 = verify(second, session.toolset, prior_calls=before + list(first.tool_calls), question=first.question)
    return CheckedAnswer(second, v2, v1, revised=True, first_answer=first)
