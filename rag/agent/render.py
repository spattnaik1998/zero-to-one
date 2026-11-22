"""Tool results as Claude reads them: readable lines, each citable on its own.

Claude tends to quote what it cites. Sent as JSON, a result came back in answers as
`"status": "final"` or `[2025, 5.428724140694802, 13.134042]`. Here each fact (a query row, a
verdict, a caveat, a column) is its own line of plain text, so a quote reads as a sentence and a
citation points at that row alone.

Presentation only: the verifier and the audit fingerprints work on the structured result.
"""
from __future__ import annotations

from ..units import column_unit

# every column scaled in billions, trillions or thousands is an amount of money (millions are people)
DOLLAR_UNITS = ("billions", "trillions", "thousands")
UNIT_WORDS = {"percent": "%", "billions": " billion", "trillions": " trillion", "millions": " million",
              "thousands": " thousand"}
IDENTIFIERS = ("cik", "sic", "naics", "year")       # never written with thousands separators


def number(value, name: str = "") -> str:
    """A number at six significant digits, which is inside every rounding the verifier accepts."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "not available"
    if isinstance(value, (list, tuple)):
        return listed([number(v, name) for v in value])
    if isinstance(value, dict):
        return "; ".join(f"{k}: {number(v, str(k))}" for k, v in value.items())
    if not isinstance(value, (int, float)):
        return str(value)
    grouped = not any(i in name.lower() for i in IDENTIFIERS)
    if float(value).is_integer() and abs(value) < 1e18:
        n = int(value)
        return f"{n:,}" if grouped and abs(n) >= 10_000 else str(n)
    return f"{value:,.2f}" if abs(value) >= 100_000 else f"{value:.6g}"


def quantity(value, name: str) -> str:
    unit = column_unit(name)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not unit or not is_number:
        return number(value, name)
    shown = number(value, name)
    if unit["unit"] in DOLLAR_UNITS:
        shown = f"-${shown[1:]}" if shown.startswith("-") else f"${shown}"
    return shown + UNIT_WORDS[unit["unit"]]


def listed(items: list) -> str:
    items = [str(i) for i in items]
    return " and ".join([", ".join(items[:-1]), items[-1]]) if len(items) > 1 else "".join(items)


def _chapters(numbers: list) -> str:
    return f"chapter{'s' if len(numbers) != 1 else ''} {listed(numbers)}"


def _claim(c: dict) -> list[str]:
    cid = c["id"]
    lines = [f"{cid}, {c.get('title', '')}. The claim: {c.get('claim', '')}."]
    if c.get("chapters"):
        lines.append(f"{cid} draws on {_chapters(c['chapters'])}.")
    if c.get("status") == "final" and c.get("verdict"):
        lines.append(f'The analysis\'s verdict on {cid} is "{c["verdict"]}", and it is final.')
        if c.get("one_line"):
            lines.append(c["one_line"])
    elif c.get("status") == "untested":
        lines.append(f"{cid} has not been tested, so it has no verdict.")
    else:
        lines.append(f"{cid} is {str(c.get('status', 'not final')).replace('_', ' ')}, so it has no verdict.")
    for s in c.get("sub_claims") or []:
        lines.append(f"{s['id']}: {s.get('claim', '')} How it was tested: {s.get('test', '')} Data: {s.get('data', '')}")
    lines += [f"Caveat on {cid}: {x}" for x in c.get("caveats") or []]
    for key, what in (("tables", "Tables"), ("figures", "Figures"), ("summary_keys", "Summary value names")):
        if c.get(key):
            lines.append(f"{what} for {cid}: {', '.join(c[key])}.")
    return lines


def _chapter(c: dict) -> list[str]:
    n, title = c["chapter"], str(c.get("title") or "")
    title = title.split(" ", 1)[-1] if title[:1].isdigit() else title
    who = "The preface (chapter 0)" if n == 0 else f'Chapter {n}, "{title}",'
    if c.get("tested"):
        tested_by = listed([str(h).removeprefix("claim:") for h in c.get("hypotheses") or []])
        lines = [f"{who} is tested by {tested_by}."]
    else:
        lines = [f"{who} has not been tested: no hypothesis tests it, so it has no verdict."]
        lines += [f"{h} is the placeholder claim for chapter {n}; it has not been tested either."
                  for h in c.get("untested_placeholders") or []]
    if not c.get("notes"):
        lines.append(f"The analyst has no notes on chapter {n}.")
    return lines


def _query(q: dict) -> list[str]:
    columns = [str(c) for c in q["columns"]]
    if not q["rows"]:
        return [f"The query returned no rows (columns: {', '.join(columns)})."]
    lines = [f"Columns of this query: {', '.join(columns)}."]
    label = columns.index("row_label") if "row_label" in columns else None
    for row in q["rows"]:
        cells = []
        for column, value in zip(columns, row):
            # in a row_label/value table the label carries the unit ("net_gain_neutral_tn")
            named = str(row[label]) if column == "value" and label is not None else column
            cells.append(f"{column}: {quantity(value, named)}")
        lines.append("; ".join(cells) + ".")
    if q.get("truncated"):
        lines.append(f"Only the first {len(q['rows'])} rows are shown; narrow the query for the rest.")
    return lines


def _table(t: dict) -> list[str]:
    head = f"Table {t['name']} ({t.get('sub_claim') or t.get('hypothesis')}): {number(t.get('rows'))} rows"
    lines = [head + (f", one per {t['grain']}" if t.get("grain") else "") + "."]
    if t.get("section"):
        lines.append(f"Section: {t['section']}.")
    for c in t["columns"]:
        about = [str(c.get("type", ""))] + ([c["unit"]] if c.get("unit") else [])
        line = f"Column {c['name']} ({', '.join(about)})"
        if c.get("min") is not None and c.get("max") is not None:
            line += f": from {quantity(c['min'], c['name'])} to {quantity(c['max'], c['name'])}"
        elif c.get("examples"):
            line += f": for example {'; '.join(str(e) for e in c['examples'])}"
        lines.append(line + ".")
    return lines


def lines(item: dict) -> list[str]:
    """One result item as lines of text. Never empty: the API rejects empty text blocks."""
    if isinstance(item.get("text"), str):
        out = [item["text"]]                         # a passage is cited as itself
    elif isinstance(item.get("columns"), list) and isinstance(item.get("rows"), list):
        out = _query(item)
    elif isinstance(item.get("columns"), list) and "name" in item:
        out = _table(item)
    elif "status" in item and "id" in item:
        out = _claim(item)
    elif "chapter" in item and "tested" in item:
        out = _chapter(item)
    elif "path" in item and "value" in item:
        out = [f"Summary value {item['path']}: {quantity(item['value'], item['path'])}."]
    elif "caption" in item and "name" in item:
        out = [f"Figure {item['name']} ({item.get('hypothesis')}): {item['caption']}"]
    else:
        out = [f"{str(k).replace('_', ' ')}: {number(v, str(k))}." for k, v in item.items()
               if k not in ("source_id", "label", "voice") and v is not None]
    return [x for x in out if x.strip()] or ["(empty)"]
