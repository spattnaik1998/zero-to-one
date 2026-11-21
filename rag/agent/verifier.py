"""Answer verifier: the answer contract, checked in code before an answer is shown (spec 6.4).

Deterministic rules, not a second model: the same answer always gets the same result, the
check costs nothing, and it cannot be argued with. Every rule compares the answer with the
tool results gathered for it (and, after a revision, for the first attempt too).

  numbers    every number in the answer is supported by a tool result or the question
             (allowing for rounding, percent <-> fraction and thousand/million/billion scaling);
             years, chapter numbers, claim ids and list markers are not checked
  verdicts   a verdict phrase must have come from a tool result in this answer, must match the
             claim it is attached to, and must not sit next to an untested or provisional claim
             or chapter unless the sentence says it is untested
  untested   when get_chapter reported an untested chapter, the answer must say so
  citations  every cited source_id was returned by a tool; a paragraph that states a number or a
             verdict cites something (a long uncited paragraph is a warning)
  voices     a paragraph citing the analyst's notes says they are the analyst's; one citing
             Thiel's quoted words names Thiel
  advice     no buy / sell / hold recommendation

`ask_checked` runs a question, verifies, sends one revision request listing the failures, and
verifies again; an answer that still fails is returned flagged with its violations.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

BOOK_CONSTANTS = (16.0, 2014.0)          # "16 September 2014", the publication date in the system prompt
SCALES = (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 1e12, 1e-12)
LONG_PARAGRAPH_WORDS = 40
NUMBER = re.compile(r"(?<![\w.$\-/])\$?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s?%)?")
CHAPTER_REF = re.compile(r"\b(?:chapters?|ch\.)\s*$", re.I)
CHAPTER_MENTION = re.compile(r"\b(?:chapters?|ch\.)\s*((?:\d{1,2})(?:\s*(?:,|and|&|–|-|to)\s*\d{1,2})*)", re.I)
CLAIM_MENTION = re.compile(r"\bH(\d{1,2})[a-d]?\b")
UNTESTED_WORDS = re.compile(r"\b(?:untested|not (?:been )?tested|no hypothesis|isn't tested|hasn't been tested|"
                            r"no verdict|not test|doesn't test|does not test|never tested|pending|provisional)\b", re.I)
ANALYST_WORDS = re.compile(r"\b(?:analyst|notes?|reading|annotat)", re.I)
ADVICE = tuple(re.compile(p, re.I) for p in (
    r"\b(?:you|investors?)\s+(?:should|ought to|must|need to)\s+(?:buy|sell|short|hold|invest in|load up on|avoid)\b",
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

    def to_dict(self) -> dict:
        return {"passed": self.passed, "numbers_checked": self.numbers_checked,
                "numbers_supported": self.numbers_supported, "violations": [asdict(v) for v in self.violations]}


# ---------------------------------------------------------------- evidence gathered for an answer
def _walk(value, out_numbers: set, out_sources: dict, out_text: list) -> None:
    if isinstance(value, dict):
        if "source_id" in value:
            out_sources[value["source_id"]] = value.get("voice", "evidence")
        for v in value.values():
            _walk(v, out_numbers, out_sources, out_text)
    elif isinstance(value, list):
        for v in value:
            _walk(v, out_numbers, out_sources, out_text)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        out_numbers.add(abs(float(value)))
    elif isinstance(value, str):
        out_text.append(value)
        out_numbers.update(_numbers_in(value))


def _numbers_in(text: str) -> set[float]:
    return {float(m.group(1).replace(",", "")) for m in NUMBER.finditer(text)}


@dataclass
class Evidence:
    numbers: set[float]
    sources: dict[str, str]            # source_id -> voice
    text: str                          # every string in the tool results, for verdict provenance

    @classmethod
    def from_calls(cls, calls, question: str = "") -> "Evidence":
        numbers, sources, texts = set(BOOK_CONSTANTS), {}, []
        for c in calls:
            if c.ok and c.result:
                _walk(c.result, numbers, sources, texts)
        numbers |= _numbers_in(question)
        return cls(numbers, sources, "\n".join(texts))

    def supports(self, literal: str) -> bool:
        """Is the number written as `literal` (e.g. '5.4', '3,991') supported at its precision?"""
        clean = literal.replace(",", "")
        value = float(clean)
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        for v in self.numbers:
            for s in SCALES:
                if abs(v * s - value) <= max(tol, abs(value) * 1e-9):
                    return True
        return False


# ---------------------------------------------------------------- the rules
def _checked_numbers(text: str):
    """(literal, start) for each number the rule applies to: not a year, chapter number,
    claim id or list marker."""
    for m in NUMBER.finditer(text):
        literal, start = m.group(1), m.start(1)
        has_pct, has_dollar = bool(m.group(2)), text[max(0, m.start() - 0):m.start(1)].endswith("$")
        plain_int = "." not in literal and "," not in literal
        if plain_int and 1900 <= int(literal) <= 2100 and not has_pct and not has_dollar:
            continue                                            # a year
        if CHAPTER_REF.search(text[max(0, start - 12):start]) or re.search(
                r"\b(?:chapters?|ch\.)\s*\d{1,2}(?:\s*(?:,|and|&|–|-|to)\s*\d{1,2})*\s*(?:,|and|&|–|-|to)\s*$",
                text[max(0, start - 40):start], re.I):
            continue                                            # chapter 9, chapters 9 and 10
        line_start = text.rfind("\n", 0, start) + 1
        if text[line_start:start].strip() == "" and re.match(r"\d+[.)]\s", text[start:]):
            continue                                            # a numbered-list marker
        yield literal, start


def _verdict_spans(text: str, verdicts: list[str]):
    """(verdict, start, end) occurrences, longest phrase first, case-sensitive, whole phrases."""
    taken, found = [], []
    for v in sorted(verdicts, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text):
            if not any(m.start() < e and s < m.end() for s, e in taken):
                taken.append((m.start(), m.end()))
                found.append((v, m.start(), m.end()))
    return sorted(found, key=lambda x: x[1])


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(c, 0, start) for c in ".!?\n")
    rights = [i for i in (text.find(c, end) for c in ".!?\n") if i != -1]
    return text[left + 1: min(rights) if rights else len(text)]


def _is_number(fact: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", fact))


def _facts(text: str, verdicts: list[str]) -> set[str]:
    """Checkable facts a paragraph states: its numbers (as written, commas dropped) and the verdicts
    it asserts (not ones it denies)."""
    facts = {lit.replace(",", "") for lit, _ in _checked_numbers(text)}
    for verdict, s, e in _verdict_spans(text, verdicts):
        if not UNTESTED_WORDS.search(_sentence(text, s, e)):
            facts.add(verdict)
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


def verify(answer, toolset, prior_calls=(), question: str | None = None) -> Verification:
    """Check one answer. `prior_calls` adds the tool results of an earlier attempt (a revision
    may rely on them); `question` is the user's original question."""
    v = Verification()
    if not answer.ok:
        v.violations.append(Violation("incomplete", f"the answer did not complete (stop reason {answer.stop_reason})"))
        return v
    text = answer.text
    evidence = Evidence.from_calls(list(prior_calls) + list(answer.tool_calls), question or answer.question)
    claims = toolset.claims
    verdicts = sorted({c["verdict"] for c in claims.values() if c["verdict"]})

    # numbers
    for literal, start in _checked_numbers(text):
        v.numbers_checked += 1
        if evidence.supports(literal):
            v.numbers_supported += 1
        else:
            context = text[max(0, start - 50): start + len(literal) + 20].replace("\n", " ")
            v.violations.append(Violation("numbers", f"{literal!r} is not supported by any tool result: ...{context}..."))

    # verdicts
    for verdict, s, e in _verdict_spans(text, verdicts):
        sentence = _sentence(text, s, e)
        if UNTESTED_WORDS.search(sentence):
            continue                       # a denial ("no verdict, 'Holds' or anything else") states no verdict
        if verdict not in evidence.text:
            v.violations.append(Violation("verdicts", f"verdict {verdict!r} did not come from a tool result in this answer"))
            continue
        ids = {f"H{n}" for n in CLAIM_MENTION.findall(sentence)}
        negated = bool(UNTESTED_WORDS.search(sentence))
        for cid in sorted(ids & set(claims)):
            c = claims[cid]
            if c["status"] != "final" and not negated:
                v.violations.append(Violation("verdicts", f"verdict {verdict!r} stated next to {cid}, which is "
                                                          f"{c['status']} and has no verdict"))
            elif c["status"] == "final" and len(ids & set(claims)) == 1 and c["verdict"] != verdict:
                v.violations.append(Violation("verdicts", f"{cid}'s verdict is {c['verdict']!r}, not {verdict!r}"))
        for m in CHAPTER_MENTION.finditer(sentence):
            for n in map(int, re.findall(r"\d{1,2}", m.group(1))):
                ch = toolset.chapters.get(n)
                if ch is not None and not ch["tested"] and not negated:
                    v.violations.append(Violation("verdicts", f"verdict {verdict!r} stated next to chapter {n}, "
                                                              "which no hypothesis tests"))

    # untested chapters
    for call in list(prior_calls) + list(answer.tool_calls):
        if call.ok and call.name == "get_chapter":
            item = call.result["results"][0]
            if not item["tested"] and not UNTESTED_WORDS.search(text):
                v.violations.append(Violation("untested", f"chapter {item['chapter']} is untested, and the answer "
                                                          "does not say so"))

    # citations and voices
    cited = {s for seg in answer.segments for s in seg["sources"] if s}
    for s in sorted(cited - set(evidence.sources)):
        v.violations.append(Violation("citations", f"cites {s!r}, which no tool returned"))
    paragraphs = list(_paragraphs(answer.segments))
    # facts stated in cited paragraphs: a summary may restate them without citing again, also at
    # lower precision ("5.4%" restating a cited "5.43%"), so numbers are matched like the numbers rule
    cited_facts = {f for para, sources in paragraphs if sources for f in _facts(para, verdicts)}
    cited_numbers = Evidence({float(f) for f in cited_facts if _is_number(f)}, {}, "")
    for para, sources in paragraphs:
        if not sources:
            uncited = {f for f in _facts(para, verdicts) - cited_facts
                       if not (_is_number(f) and cited_numbers.supports(f))}
            if uncited:
                shown = ", ".join(sorted(uncited)[:6])
                v.violations.append(Violation("citations", f"a paragraph states facts no cited paragraph supports "
                                                           f"({shown}): {para[:90]!r}"))
            elif len(para.split()) >= LONG_PARAGRAPH_WORDS:
                v.violations.append(Violation("citations", f"a long paragraph has no citation: {para[:90]!r}",
                                              "warning"))
            continue
        voices = {evidence.sources.get(s) for s in sources}
        if "analyst" in voices and not ANALYST_WORDS.search(para):
            v.violations.append(Violation("voices", f"cites the analyst's notes without saying so: {para[:90]!r}"))
        if "thiel" in voices and "Thiel" not in para:
            v.violations.append(Violation("voices", f"cites Thiel's words without naming him: {para[:90]!r}"))

    # investment advice
    for pattern in ADVICE:
        m = pattern.search(text)
        if m:
            v.violations.append(Violation("advice", f"reads as investment advice: {m.group(0)!r}"))
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

    @property
    def flagged(self) -> bool:
        return not self.verification.passed

    def to_dict(self) -> dict:
        d = self.answer.to_dict()
        d.update(question=self.first_answer.question if self.first_answer else self.answer.question,
                 verification=self.verification.to_dict(),
                 first_verification=self.first_verification.to_dict(),
                 revised=self.revised, flagged=self.flagged)
        return d


def ask_checked(session, question: str) -> CheckedAnswer:
    first = session.ask(question)
    v1 = verify(first, session.toolset)
    if v1.passed or not first.ok:
        return CheckedAnswer(first, v1, v1, revised=False)
    second = session.ask(revision_request(v1))
    v2 = verify(second, session.toolset, prior_calls=first.tool_calls, question=first.question)
    return CheckedAnswer(second, v2, v1, revised=True, first_answer=first)
