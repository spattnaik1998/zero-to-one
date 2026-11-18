"""Claims registry and chapter map (spec section 6.1, step 5).

The registry is the only source of verdicts the assistant may state. It joins:
  - each notebook's claim cell (title, chapters, Thiel's quote, sub-claims, caveats)
  - findings_report.py: VERDICTS, SECTIONS, H1_PROVISIONAL and the scorecard list
  - README.md's Findings table (claim wording, verdict, one-line summary)
  - output/: the summary JSON, tables and figures for each hypothesis
  - the analyst's notes, linked by chapter

`build_registry` returns the records plus a list of problems. Any problem means the
sources disagree, and the build must stop rather than publish one version of the truth.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .chapters import CHAPTERS
from .notebooks import ClaimCell, notebook_chunks, parse_chapters

FINAL, PROVISIONAL, RESULTS_IN, PENDING, UNTESTED = "final", "provisional", "results_in", "pending", "untested"
HAS_VERDICT = {FINAL}                     # the only status whose verdict the assistant may state
_README_ROW = re.compile(r"^\|\s*H(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")


@dataclass(frozen=True)
class Claim:
    id: str
    title: str
    claim: str                      # README wording
    chapters: tuple[int, ...]
    status: str
    verdict: str | None             # set only when status == "final"
    one_line: str | None
    thiel_quote: str
    sub_claims: tuple[dict, ...]
    caveats: tuple[str, ...]
    notebook: str | None
    summary: str | None
    summary_keys: tuple[str, ...]
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    note_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChapterEntry:
    number: int
    title: str
    hypotheses: tuple[str, ...]     # tested claims with results (final or provisional)
    untested: tuple[str, ...]       # placeholders such as H9
    tested: bool
    note_ids: tuple[str, ...]


def read_report(path: Path) -> dict:
    """VERDICTS, SECTIONS keys, H1_PROVISIONAL and the scorecard (n -> text), read without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {"verdicts": {}, "sections": set(), "h1_provisional": False, "scorecard": {}}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "VERDICTS":
                out["verdicts"] = ast.literal_eval(node.value)
            elif name == "SECTIONS" and isinstance(node.value, ast.Dict):
                out["sections"] = {ast.literal_eval(k) for k in node.value.keys}
            elif name == "H1_PROVISIONAL":
                out["h1_provisional"] = bool(ast.literal_eval(node.value))
        if isinstance(node, ast.For) and isinstance(node.iter, ast.List) and node.iter.elts \
                and all(isinstance(e, ast.Tuple) and len(e.elts) == 2 for e in node.iter.elts):
            try:
                pairs = [ast.literal_eval(e) for e in node.iter.elts]
            except ValueError:
                continue
            if all(isinstance(n, int) and isinstance(t, str) for n, t in pairs):
                out["scorecard"] = dict(pairs)
    return out


def read_readme(path: Path) -> dict[int, dict]:
    rows = {}
    in_findings = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_findings = line.strip().lower() == "## findings"
        m = _README_ROW.match(line) if in_findings else None
        if m:
            n, claim, verdict, one_line = m.groups()
            rows[int(n)] = {"claim": claim, "verdict": verdict, "one_line": one_line}
    return rows


def _chapters_of(text: str) -> tuple[int, ...] | None:
    m = re.search(r"\(([^()]*)\)\s*$", text)
    return parse_chapters(m.group(1)) if m else None


def _strip_chapters(text: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()


def build_registry(root: Path, notes: list | None = None) -> tuple[list[Claim], list[ChapterEntry], list[str]]:
    report = read_report(root / "findings_report.py")
    readme = read_readme(root / "README.md")
    notes = notes or []
    problems: list[str] = []

    cells: dict[int, tuple[ClaimCell, str]] = {}
    for nb in sorted((root / "notebooks").glob("*.ipynb")):
        claim, _ = notebook_chunks(nb, root)
        n, rel = int(claim.hypothesis[1:]), nb.relative_to(root).as_posix()
        if n in cells:
            problems.append(f"{claim.hypothesis}: claimed by both {cells[n][1]} and {rel}")
            continue
        cells[n] = (claim, rel)

    numbers = sorted(set(cells) | set(report["scorecard"]) | set(report["verdicts"]) | set(readme))
    claims: list[Claim] = []
    for n in numbers:
        cid = f"H{n}"
        cell, nb_path = cells.get(n, (None, None))
        row = readme.get(n)
        score = report["scorecard"].get(n)
        summary = root / "output" / f"h{n}_summary.json"

        if cell is None:
            if n in report["verdicts"] or row:
                problems.append(f"{cid}: has a verdict or README row but no notebook")
            status = UNTESTED
        elif not summary.is_file():
            status = PENDING
        elif n not in report["sections"]:
            status = RESULTS_IN
        else:
            status = PROVISIONAL if n == 1 and report["h1_provisional"] else FINAL

        chapters = cell.chapters if cell else (_chapters_of(score) if score else ())
        for label, text in (("README", row["claim"] if row else None), ("findings_report scorecard", score)):
            if cell and text is not None and _chapters_of(text) != cell.chapters:
                problems.append(f"{cid}: {label} gives chapters {_chapters_of(text)}, "
                                f"notebook heading gives {cell.chapters}")
        if cell and row is None:
            problems.append(f"{cid}: missing from the README Findings table")
        if cell and score is None:
            problems.append(f"{cid}: missing from the findings_report scorecard")
        verdict = report["verdicts"].get(n)
        if status in (FINAL, PROVISIONAL):
            if verdict is None:
                problems.append(f"{cid}: results are in but VERDICTS has no entry")
            elif row and row["verdict"] != verdict:
                problems.append(f"{cid}: README verdict {row['verdict']!r} != VERDICTS {verdict!r}")

        prefix = re.compile(rf"^h{n}[a-z]?_")
        tables = tuple(sorted(p.stem for p in (root / "output" / "tables").glob("*.csv") if prefix.match(p.stem)))
        figures = tuple(sorted(p.stem for p in (root / "output" / "figures").glob("*.png") if prefix.match(p.stem)))
        # top-level keys starting with "_" are metadata (e.g. _provenance), not evidence
        keys = tuple(k for k in json.loads(summary.read_text(encoding="utf-8")) if not k.startswith("_")) \
            if summary.is_file() else ()
        claims.append(Claim(
            id=cid,
            title=cell.title if cell else _strip_chapters(score or cid),
            claim=_strip_chapters(row["claim"]) if row else _strip_chapters(score or ""),
            chapters=chapters,
            status=status,
            verdict=verdict if status in HAS_VERDICT else None,
            # The one-line summary states a conclusion, so it follows the verdict rule.
            one_line=row["one_line"] if row and status in HAS_VERDICT else None,
            thiel_quote=cell.quote if cell else "",
            sub_claims=cell.sub_claims if cell else (),
            caveats=cell.caveats if cell else (),
            notebook=nb_path,
            summary=summary.relative_to(root).as_posix() if summary.is_file() else None,
            summary_keys=keys, tables=tables, figures=figures,
            note_ids=tuple(x.id for x in notes if x.chapter in chapters),
        ))

    chapter_map = []
    for c in CHAPTERS:
        with_results = tuple(k.id for k in claims if c.number in k.chapters and k.status in (FINAL, PROVISIONAL))
        placeholders = tuple(k.id for k in claims if c.number in k.chapters and k.status == UNTESTED)
        chapter_map.append(ChapterEntry(c.number, c.display, with_results, placeholders, bool(with_results),
                                        tuple(x.id for x in notes if x.chapter == c.number)))
    return claims, chapter_map, problems


def write_json(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
