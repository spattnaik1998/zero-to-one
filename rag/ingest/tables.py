"""Results tables, summary values and figures -> DuckDB (spec section 6.1, step 6).

This is how the assistant reaches the data the notebooks pulled through the API keys
(SEC, Census, FRED/BLS, OEWS). Numbers in answers come from here, never from memory.

The database holds:
  <table>            one table per output/tables/*.csv, same name and values
  summary_values     every leaf of every output/hN_summary.json, one row per value,
                     addressed by a path such as  airlines_vs_bigtech_margin.2025.bigtech_op_margin
  catalog_tables     a schema card per table: hypothesis, sub-claim, grain, columns with
                     types and ranges, and the notebook cell that wrote it
  catalog_figures    each figure with its caption and where the caption came from

Figures are identified by file name. The report and the notebooks number figures
differently ("Figure H1-1" is a different chart in each), so numbers are never IDs.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .notebooks import read_cells, _src

UNNAMED = "row_label"            # name given to a CSV's unnamed index column
_PREFIX = re.compile(r"^h(\d)([a-z])?_")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.M)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nulls: int
    distinct: int
    min: object = None
    max: object = None
    examples: tuple = ()


@dataclass(frozen=True)
class TableCard:
    name: str
    hypothesis: str
    sub_claim: str | None
    rows: int
    grain: str | None             # the column that identifies a row, if one does
    columns: tuple[Column, ...]
    renamed_index: bool           # the CSV's unnamed first column was renamed to row_label
    notebook: str | None
    cell: int | None
    section: str | None
    csv: str


@dataclass(frozen=True)
class FigureCard:
    name: str
    hypothesis: str
    caption: str | None
    caption_source: str           # report | notebook heading | none
    unresolved: tuple[str, ...]   # caption placeholders that could not be filled
    png: str
    pdf: str | None


# ---------------------------------------------------------------- summaries
def flatten(value, path: str = "") -> list[tuple[str, object]]:
    """Leaf values of nested JSON with their paths: a.b[0].c"""
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in flatten(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in flatten(v, f"{path}[{i}]")]
    return [(path, value)]


def summary_rows(output: Path) -> list[dict]:
    rows = []
    for f in sorted(output.glob("h*_summary.json")):
        hyp = "H" + re.match(r"h(\d+)_", f.name).group(1)
        for path, v in flatten(json.loads(f.read_text(encoding="utf-8"))):
            is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
            rows.append({
                "hypothesis": hyp, "path": path, "top_key": re.split(r"[.\[]", path, 1)[0],
                "value_type": "null" if v is None else type(v).__name__,
                "value_num": float(v) if is_num and math.isfinite(v) else (float(v) if isinstance(v, bool) else None),
                "value_text": None if v is None else (json.dumps(v) if isinstance(v, bool) else str(v)),
                "source": f.name,
            })
    return rows


# ---------------------------------------------------------------- provenance
def _writers(root: Path) -> dict[str, list[tuple[str, int, str]]]:
    """Code cells per notebook: (notebook path, cell index, source), plus the section heading in force."""
    out = {}
    for nb in sorted((root / "notebooks").glob("*.ipynb")):
        section, cells = None, []
        for i, c in enumerate(read_cells(nb)):
            src = _src(c)
            if c["cell_type"] == "markdown":
                m = _HEADING.search(src)
                section = m.group(1).strip() if m else section
            else:
                cells.append((i, src, section))
        out[nb.relative_to(root).as_posix()] = cells
    return out


def _name_pattern(name: str) -> re.Pattern:
    """The name as a string literal; a 4-digit year part may be an f-string placeholder."""
    parts = [r"(?:\d{4}|\{[^}]+\})" if re.fullmatch(r"\d{4}", p) else re.escape(p) for p in name.split("_")]
    return re.compile(r"""f?["']""" + "_".join(parts) + r"""["']""")


def find_writer(name: str, writers) -> tuple[str, int, str | None] | None:
    pat = _name_pattern(name)
    for nb, cells in writers.items():
        for i, src, section in cells:
            if pat.search(src):
                return nb, i, section
    return None


# ---------------------------------------------------------------- captions
def report_captions(report: Path, output: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Captions from fig(name, caption) calls in findings_report.py, read without importing it.
    f-string placeholders are filled from the summary when the variable is assigned from it."""
    tree = ast.parse(report.read_text(encoding="utf-8"))
    out = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        m = re.fullmatch(r"section_h(\d+)", fn.name)
        summary = {}
        if m and (output / f"h{m.group(1)}_summary.json").is_file():
            summary = json.loads((output / f"h{m.group(1)}_summary.json").read_text(encoding="utf-8"))
        assigns = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigns[t.id] = node.value
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                        and len(t.elts) == len(node.value.elts):          # E, L = s["a"], s["b"]
                    assigns.update({e.id: v for e, v in zip(t.elts, node.value.elts) if isinstance(e, ast.Name)})
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "fig" and len(call.args) >= 2
                    and isinstance(call.args[0], ast.Constant)):
                continue
            cap = call.args[1]
            if isinstance(cap, ast.Constant):
                out[call.args[0].value] = (cap.value, ())
            elif isinstance(cap, ast.JoinedStr):
                text, unresolved = "", []
                for part in cap.values:
                    if isinstance(part, ast.Constant):
                        text += part.value
                        continue
                    value = _resolve(part.value, assigns, summary)
                    if value is None:
                        unresolved.append(ast.unparse(part.value))
                        text += "{" + ast.unparse(part.value) + "}"
                    else:
                        spec = ast.unparse(part.format_spec)[2:-1] if part.format_spec else ""
                        text += format(value, spec)
                out[call.args[0].value] = (text, tuple(unresolved))
    return out


def _resolve(node, assigns: dict, summary: dict, depth: int = 0):
    """Value of `s["a"]["b"]`, `s.get("a", d)`, or a local name assigned from one of those."""
    if depth > 5:
        return None
    if isinstance(node, ast.Name) and node.id in assigns:
        return _resolve(assigns[node.id], assigns, summary, depth + 1)
    if isinstance(node, ast.Subscript):
        base = summary if isinstance(node.value, ast.Name) and node.value.id == "s" else \
            _resolve(node.value, assigns, summary, depth + 1)
        try:
            key = ast.literal_eval(node.slice)
            return base[key] if base is not None else None
        except (ValueError, KeyError, IndexError, TypeError):
            return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "s" and node.args:
        try:
            key = ast.literal_eval(node.args[0])
            default = ast.literal_eval(node.args[1]) if len(node.args) > 1 else None
        except ValueError:
            return None
        return summary.get(key, default)
    return None


def notebook_figure_headings(writers, root: Path) -> dict[str, str]:
    """Fallback caption: the markdown heading in force where save_fig wrote the figure."""
    out = {}
    for nb, cells in writers.items():
        for i, src, section in cells:
            for name in re.findall(r"save_fig\(\s*[^,]+,\s*[\"']([^\"']+)[\"']", src):
                if section:
                    out[name] = section
    return out


# ---------------------------------------------------------------- tables
def _column(name: str, s: pd.Series) -> Column:
    nonnull = s.dropna()
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    lo = hi = None
    if numeric and len(nonnull):
        lo, hi = (x.item() if hasattr(x, "item") else x for x in (nonnull.min(), nonnull.max()))
    examples = () if numeric else tuple(str(x) for x in nonnull.drop_duplicates().head(3))
    return Column(name, str(s.dtype), int(s.isna().sum()), int(nonnull.nunique()), lo, hi, examples)


def read_table(csv: Path) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(csv)
    renamed = df.columns[0].startswith("Unnamed: ")
    if renamed:
        df = df.rename(columns={df.columns[0]: UNNAMED})
    return df, renamed


def build_database(root: Path, db_path: Path) -> tuple[list[TableCard], list[FigureCard], int]:
    output = root / "output"
    writers = _writers(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    cards: list[TableCard] = []
    try:
        for csv in sorted((output / "tables").glob("*.csv")):
            name = csv.stem
            df, renamed = read_table(csv)
            con.register("_df", df)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _df')
            con.unregister("_df")
            m = _PREFIX.match(name)
            hyp = f"H{m.group(1)}" if m else "?"
            writer = find_writer(name, writers)
            grain = next((c for c in df.columns if df[c].notna().all() and df[c].is_unique), None)
            cards.append(TableCard(
                name, hyp, f"{hyp}{m.group(2)}" if m and m.group(2) else None, len(df), grain,
                tuple(_column(c, df[c]) for c in df.columns), renamed,
                writer[0] if writer else None, writer[1] if writer else None, writer[2] if writer else None,
                csv.relative_to(root).as_posix()))

        rows = summary_rows(output)
        con.register("_sv", pd.DataFrame(rows, columns=["hypothesis", "path", "top_key", "value_type",
                                                        "value_num", "value_text", "source"]))
        con.execute("CREATE TABLE summary_values AS SELECT * FROM _sv")
        con.unregister("_sv")

        captions = report_captions(root / "findings_report.py", output)
        headings = notebook_figure_headings(writers, root)
        figures = []
        for png in sorted((output / "figures").glob("*.png")):
            name = png.stem
            m = _PREFIX.match(name)
            if name in captions:
                caption, unresolved, src = *captions[name], "report"
            elif name in headings:
                caption, unresolved, src = headings[name], (), "notebook heading"
            else:
                caption, unresolved, src = None, (), "none"
            pdf = png.with_suffix(".pdf")
            figures.append(FigureCard(name, f"H{m.group(1)}" if m else "?", caption, src, unresolved,
                                      png.relative_to(root).as_posix(),
                                      pdf.relative_to(root).as_posix() if pdf.is_file() else None))

        for table, records in (("catalog_tables", cards), ("catalog_figures", figures)):
            frame = pd.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, (list, tuple, dict)) else v)
                                   for k, v in asdict(r).items()} for r in records])
            con.register("_cat", frame)
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _cat")
            con.unregister("_cat")
    finally:
        con.close()
    return cards, figures, len(rows)


def verify_database(root: Path, db_path: Path) -> list[str]:
    """Compare every table cell and every summary value in the database with the source files.
    Returns a list of differences (empty when the database is exact)."""
    problems = []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        for csv in sorted((root / "output" / "tables").glob("*.csv")):
            if csv.stem not in names:
                problems.append(f"{csv.stem}: missing from the database")
                continue
            want, _ = read_table(csv)
            got = con.execute(f'SELECT * FROM "{csv.stem}"').df()
            if list(got.columns) != list(want.columns) or len(got) != len(want):
                problems.append(f"{csv.stem}: shape {got.shape} != {want.shape}")
                continue
            for c in want.columns:
                a, b = want[c], got[c]
                if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
                    same = ((a.isna() & b.isna()) | (a.astype(float) == b.astype(float))).all()
                else:
                    same = (a.isna().to_numpy() == b.isna().to_numpy()).all() and \
                           (a.dropna().astype(str).to_numpy() == b.dropna().astype(str).to_numpy()).all()
                if not same:
                    problems.append(f"{csv.stem}.{c}: values differ from the CSV")

        stored = {(h, p): (n, t) for h, p, n, t in
                  con.execute("SELECT hypothesis, path, value_num, value_text FROM summary_values").fetchall()}
        expected = {(r["hypothesis"], r["path"]): (r["value_num"], r["value_text"]) for r in summary_rows(root / "output")}
        if len(stored) != len(expected):
            problems.append(f"summary_values: {len(stored)} rows, expected {len(expected)} (duplicate paths?)")
        for key, (num, text) in expected.items():
            if stored.get(key) != (num, text):
                problems.append(f"summary_values {key[0]} {key[1]}: {stored.get(key)} != {(num, text)}")
    finally:
        con.close()
    return problems
