"""The agent's tools, each reading one store of a pinned, verified release (spec section 6.3).

A `Toolset` is bound to one `Release` for its whole life, so every answer in a session
comes from the same corpus version. `call(name, input)` validates the input against the
tool's schema and its limits (strict tool schemas cannot express ranges) and returns a
JSON-able dict. Every result item carries `source_id` and `voice`, which the answer
verifier and citations use.

query_table is the only tool that takes free-form input. Its guards, all of which matter:
  - the statement must parse as exactly one SELECT;
  - an allow-list over DuckDB's parse tree: only this database's tables (or names the query
    defines in WITH), only harmless table functions, no functions that read settings, the
    environment or the system catalog (these would expose local paths);
  - a read-only connection with external access disabled and the configuration locked;
  - memory, thread and time limits, with DuckDB's temporary files kept outside the release
    (spill files inside it would make the release fail its integrity check);
  - a row cap, fetched without rewriting the query so its ORDER BY holds.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from decimal import Decimal
from pathlib import Path

import duckdb

from ..ingest.index import read_jsonl
from ..release import Release
from ..retrieval import HybridIndex
from ..units import column_unit

MAX_K = 10
DEFAULT_K = 5
MAX_ROWS = 200
DEFAULT_ROWS = 50
QUERY_SECONDS = 5.0
MEMORY_LIMIT = "512MB"
KINDS = ("claim", "method", "result", "caveat")
SAFE_TABLE_FUNCTIONS = frozenset({"range", "generate_series", "unnest"})
DENIED_FUNCTIONS = frozenset({"current_setting", "getenv", "current_database", "current_catalog", "current_schema",
                              "current_schemas", "current_user", "user", "session_user", "which_secret", "version",
                              "platform", "query", "query_table"})
DENIED_PREFIXES = ("duckdb_", "pragma_", "read_", "glob", "sniff_", "parquet_", "iceberg_", "delta_")


class ToolError(ValueError):
    """A bad tool call. The message is returned to the model as the tool result, so it says
    what to change."""


def _int(value, name: str, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise ToolError(f"{name} must be between {lo} and {hi}")
    return value


def _str(value, name: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string")
    return value.strip()


def _json_value(v):
    """A DuckDB value as plain JSON: numbers stay numbers (DECIMAL too), NaN/inf become null,
    lists and structs recurse, bytes become base64, anything else (dates, times) ISO text."""
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_value(x) for k, x in v.items()}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"base64": base64.b64encode(bytes(v)).decode("ascii")}
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def _typed(value_type: str, value_num, value_text):
    """A summary value with its original JSON type (summary_values stores ints and bools as floats)."""
    if value_type == "int":
        return int(value_num)
    if value_type == "float":
        return _json_value(value_num)
    if value_type == "bool":
        return value_text == "true"
    return value_text


def _walk(node, out: dict) -> None:
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "BASE_TABLE":
            out["tables"].append((node.get("catalog_name", ""), node.get("schema_name", ""), node.get("table_name", "")))
        elif kind == "TABLE_FUNCTION":
            out["table_functions"].append(node.get("function", {}).get("function_name", ""))
        if node.get("class") == "FUNCTION":
            out["functions"].append(node.get("function_name", ""))
        for entry in node.get("cte_map", {}).get("map", []) if isinstance(node.get("cte_map"), dict) else []:
            out["ctes"].add(entry.get("key", ""))
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out)


def check_sql(sql: str, allowed_tables: set[str]) -> None:
    """Raise ToolError unless `sql` is one SELECT over allowed tables and safe functions."""
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise ToolError(f"SQL does not parse: {str(e).splitlines()[0]}")
    if len(statements) != 1:
        raise ToolError("send exactly one SQL statement")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ToolError("only SELECT queries are allowed (WITH ... SELECT is fine)")
    with duckdb.connect() as con:                     # in-memory; used only to parse
        tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    if tree.get("error"):
        raise ToolError(f"SQL does not parse: {tree.get('error_message', 'unknown error')}")
    found = {"tables": [], "table_functions": [], "functions": [], "ctes": set()}
    _walk(tree, found)
    for catalog, schema, table in found["tables"]:
        if catalog or schema not in ("", "main") or (table not in allowed_tables and table not in found["ctes"]):
            name = ".".join(x for x in (catalog, schema, table) if x)
            raise ToolError(f"table {name!r} is not available; query the results tables, summary_values, "
                            "catalog_tables or catalog_figures")
    for fn in found["table_functions"]:
        if fn.lower() not in SAFE_TABLE_FUNCTIONS:
            raise ToolError(f"table function {fn!r} is not allowed (allowed: {sorted(SAFE_TABLE_FUNCTIONS)})")
    for fn in found["functions"]:
        low = fn.lower()
        if low in DENIED_FUNCTIONS or low.startswith(DENIED_PREFIXES):
            raise ToolError(f"function {fn!r} is not allowed")


class Toolset:
    def __init__(self, release: Release, embedder, scratch_dir: Path | None = None):
        model = release.manifest["models"]["embedding_model"]
        if embedder.model != model:
            raise ValueError(f"release {release.release_id} was embedded with {model}; "
                             f"the query embedder uses {embedder.model}")
        self.release, self.embedder = release, embedder
        # DuckDB's temporary files must live outside the release (build/releases/<id>/ -> build/cache/)
        self.scratch_dir = scratch_dir or release.path.parent.parent / "cache" / "duckdb_tmp"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.indexes = {name: HybridIndex.load(release.file("indexes.duckdb"), name)
                        for name in ("notes", "analysis", "catalog")}
        self.claims = {c["id"]: c for c in self._json("claims_registry.json")}
        self.chapters = {c["number"]: c for c in self._json("chapter_map.json")}
        self.cards = {c["name"]: c for c in self._json("schema_cards.json")}
        self.figures = {f["name"]: f for f in self._json("figures.json")}
        self.notes = {n["id"]: n for n in read_jsonl(release.file("notes_chunks.jsonl"))}
        self.db_path = release.file("analysis.duckdb")
        with self._connect() as con:                  # the release is immutable: read these once
            self.tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
            rows = con.execute("SELECT hypothesis, path, value_type, value_num, value_text FROM summary_values "
                               "WHERE path NOT LIKE '%.%' AND path NOT LIKE '%[%' ORDER BY hypothesis, path").fetchall()
        self.headline = {}
        for hyp, path, vtype, vnum, vtext in rows:
            self.headline.setdefault(hyp, []).append(
                {"source_id": f"summary:{hyp}:{path}", "voice": "evidence", "label": f"Summary value {hyp} {path}",
                 "path": path, "value": _typed(vtype, vnum, vtext)})

    def _json(self, name: str):
        return json.loads(self.release.file(name).read_text(encoding="utf-8"))

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, tool_input: dict) -> dict:
        definition = DEFINITIONS.get(name)
        if definition is None:
            raise ToolError(f"unknown tool {name!r}; tools are {sorted(DEFINITIONS)}")
        if not isinstance(tool_input, dict):
            raise ToolError("tool input must be an object")
        schema = definition["input_schema"]
        extra = set(tool_input) - set(schema["properties"])
        if extra:
            raise ToolError(f"unexpected argument(s) {sorted(extra)}; allowed: {sorted(schema['properties'])}")
        missing = [k for k in schema["required"] if k not in tool_input]
        if missing:
            raise ToolError(f"missing required argument(s) {missing}")
        result = getattr(self, f"tool_{name}")(**tool_input)
        return {"tool": name, "release_id": self.release.release_id, **result}

    def _hypothesis(self, value) -> str | None:
        if value is None:
            return None
        hyp = _str(value, "hypothesis").upper()
        if hyp not in self.claims:
            raise ToolError(f"hypothesis must be one of {sorted(self.claims)}")
        return hyp

    # ------------------------------------------------------------ search
    def _search(self, index: str, query, k, where: dict) -> list:
        query = _str(query, "query")
        k = _int(k, "k", DEFAULT_K, 1, MAX_K)
        vector = self.embedder.embed([query])[0]
        return self.indexes[index].search(query, vector, k=k, where={w: v for w, v in where.items() if v is not None})

    def tool_search_notes(self, query, chapter=None, opinion_only=None, k=None) -> dict:
        if chapter is not None:
            _int(chapter, "chapter", 0, 0, 15)
        if opinion_only is not None and not isinstance(opinion_only, bool):
            raise ToolError("opinion_only must be true or false")
        hits = self._search("notes", query, k, {"chapter": chapter, "opinion": True if opinion_only else None})
        return {"results": [self._note(h.doc_id, h.meta["opinion"], h.meta["chapter"], h.text,
                                       chapter_title=h.meta["chapter_title"], page=h.meta["page"]) for h in hits]}

    @staticmethod
    def _note(source_id: str, opinion: bool, chapter: int, text: str, **extra) -> dict:
        who = "Analyst's opinion" if opinion else "Analyst's note"
        return {"source_id": source_id, "voice": "analyst", "label": f"{who}, chapter {chapter}",
                "opinion": opinion, "chapter": chapter, **extra, "text": text}

    def tool_search_analysis(self, query, hypothesis=None, kind=None, k=None) -> dict:
        hypothesis = self._hypothesis(hypothesis)
        if kind is not None and kind not in KINDS:
            raise ToolError(f"kind must be one of {list(KINDS)}")
        hits = self._search("analysis", query, k, {"hypothesis": hypothesis, "kind": kind})
        return {"results": [{
            "source_id": h.doc_id, "voice": h.meta["voice"],
            "label": (f"Thiel, quoted in {h.meta['hypothesis']}" if h.meta["voice"] == "thiel" else
                      "Analysis " + " · ".join(x for x in (h.meta["hypothesis"], h.meta["section"]) if x)
                      + f" ({h.meta['kind']})"),
            "hypothesis": h.meta["hypothesis"],
            "sub_claim": h.meta["sub_claim"], "kind": h.meta["kind"], "section": h.meta["section"],
            "notebook": h.meta["notebook"], "cell": h.meta["cell"], "cell_type": h.meta["cell_type"],
            "text": h.text} for h in hits]}

    def tool_find_data(self, query, type=None, hypothesis=None, k=None) -> dict:
        if type is not None and type not in ("table", "figure"):
            raise ToolError("type must be 'table' or 'figure'")
        hits = self._search("catalog", query, k, {"type": type, "hypothesis": self._hypothesis(hypothesis)})
        return {"results": [{"source_id": h.doc_id, "voice": "evidence",
                             "label": f"{h.meta['type'].title()} {h.meta['name']} ({h.meta['hypothesis']})",
                             "type": h.meta["type"], "name": h.meta["name"], "hypothesis": h.meta["hypothesis"],
                             "text": h.text} for h in hits]}

    # ------------------------------------------------------------ registry
    def _claim_view(self, c: dict, full: bool) -> dict:
        view = {"source_id": f"claim:{c['id']}", "voice": "evidence",
                "label": f"Claim {c['id']}: {c['title']} (status {c['status']})",
                "id": c["id"], "title": c["title"],
                "claim": c["claim"], "chapters": c["chapters"], "status": c["status"],
                # verdicts and one-line findings exist only for final claims (the registry enforces it)
                "verdict": c["verdict"], "one_line": c["one_line"]}
        if full:
            view.update({k: c[k] for k in ("sub_claims", "caveats", "notebook", "tables", "figures",
                                           "note_ids", "summary_keys")})
            # Thiel's words are their own citable item in his voice, not part of the evidence record
            view["thiel_quote"] = ({"source_id": f"quote:{c['id']}", "voice": "thiel",
                                    "label": f"Thiel, quoted in {c['id']}", "text": c["thiel_quote"]}
                                   if c["thiel_quote"] else None)
            view["headline_values"] = self.headline.get(c["id"], [])
            if c["status"] != "final":
                view["note"] = ("no verdict: this claim is " + c["status"].replace("_", " ")
                                + "; do not state or imply one")
        return view

    def tool_list_claims(self) -> dict:
        return {"results": [self._claim_view(c, full=False) for c in self.claims.values()]}

    def tool_get_claim(self, id) -> dict:
        cid = _str(id, "id").upper()
        if cid not in self.claims:
            raise ToolError(f"no claim {cid}; claims are {sorted(self.claims)}")
        return {"results": [self._claim_view(self.claims[cid], full=True)]}

    def tool_get_chapter(self, chapter) -> dict:
        number = _int(chapter, "chapter", 0, 0, 15)
        e = self.chapters[number]
        return {"results": [{
            "source_id": f"chapter:{number}", "voice": "evidence",
            "label": f"Chapter {e['title']}" + ("" if e["tested"] else " (untested)"),
            "chapter": number, "title": e["title"], "tested": e["tested"],
            "hypotheses": [self._claim_view(self.claims[h], full=False) for h in e["hypotheses"]],
            "untested_placeholders": list(e["untested"]),
            "note": None if e["tested"] else "no hypothesis tests this chapter; say so instead of giving a verdict",
            "notes": [self._note(i, self.notes[i]["opinion"], number, self.notes[i]["text"]) for i in e["note_ids"]],
        }]}

    # ------------------------------------------------------------ data
    def tool_describe_table(self, name) -> dict:
        name = _str(name, "name")
        if name not in self.cards:
            raise ToolError(f"no table {name!r}; use find_data to look tables up")
        card = dict(self.cards[name])
        # each column's unit, from its name's suffix (_bn, _tn, _m, _k, _pct): the only record of its scale
        card["columns"] = [{**c, "unit": (column_unit(c["name"]) or {}).get("unit")} for c in card["columns"]]
        return {"results": [{"source_id": f"table:{name}", "voice": "evidence", "label": f"Table {name} (schema)",
                             **card}]}

    def tool_get_figure(self, name) -> dict:
        name = _str(name, "name")
        if name not in self.figures:
            raise ToolError(f"no figure {name!r}; use find_data with type 'figure' to look figures up")
        f = self.figures[name]
        return {"results": [{"source_id": f"figure:{name}", "voice": "evidence", "label": f"Figure {name}", "name": name,
                             "hypothesis": f["hypothesis"], "caption": f["caption"],
                             # a path inside the release, never a local filesystem path
                             "file": f"figures/{name}.png"}]}

    def _connect(self):
        return duckdb.connect(str(self.db_path), read_only=True, config={
            "enable_external_access": False, "memory_limit": MEMORY_LIMIT, "threads": 2,
            "temp_directory": str(self.scratch_dir), "lock_configuration": True})

    def tool_query_table(self, sql, max_rows=None) -> dict:
        sql = _str(sql, "sql").rstrip().rstrip(";").strip()
        cap = _int(max_rows, "max_rows", DEFAULT_ROWS, 1, MAX_ROWS)
        check_sql(sql, self.tables)
        with self._connect() as con:
            timer = threading.Timer(QUERY_SECONDS, con.interrupt)
            timer.start()
            try:
                cur = con.execute(sql)            # not wrapped: the query's own ORDER BY stays authoritative
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(cap + 1)
            except duckdb.InterruptException:
                raise ToolError(f"query stopped after {QUERY_SECONDS:.0f} s; narrow it down")
            except duckdb.Error as e:
                raise ToolError(f"query failed: {str(e).splitlines()[0]}")
            finally:
                timer.cancel()
        return {"results": [{
            "source_id": "query:" + hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12],
            "voice": "evidence", "label": f"Query result: {sql[:120]}", "sql": sql, "columns": columns,
            "rows": [[_json_value(v) for v in r] for r in rows[:cap]],
            "row_count": min(len(rows), cap), "truncated": len(rows) > cap,
            # units of the columns whose names carry one (gain_bn: billions); a renamed column has none
            "units": {c: u["unit"] for c in columns if (u := column_unit(c))},
        }]}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_K = {"type": "integer", "description": f"Number of results, 1-{MAX_K} (default {DEFAULT_K})."}
_HYP = {"type": "string", "description": "Hypothesis id, e.g. H3. Restricts the search to that claim."}

TOOL_DEFINITIONS = [
    {"name": "search_notes",
     "description": ("Search the analyst's reading notes: interpretations of each chapter, in the analyst's voice. "
                     "Results marked opinion are the analyst's own view. Do not use this to establish what the "
                     "data shows (use the analysis and data tools) or as Thiel's own words."),
     "input_schema": _schema({"query": {"type": "string", "description": "What to look for."},
                              "chapter": {"type": "integer", "description": "Chapter 0-15 (0 preface, 15 conclusion)."},
                              "opinion_only": {"type": "boolean", "description": "Only the analyst's own views."},
                              "k": _K}, ["query"])},
    {"name": "search_analysis",
     "description": ("Search the H1-H8 analysis notebooks: each claim, how it was tested (methods), caveats and "
                     "results. Results with voice 'thiel' are Thiel's quoted words. Use it to explain findings and "
                     "methods; for exact numbers use query_table or get_claim."),
     "input_schema": _schema({"query": {"type": "string", "description": "What to look for."},
                              "hypothesis": _HYP,
                              "kind": {"type": "string", "enum": list(KINDS), "description": "Only this kind of chunk."},
                              "k": _K}, ["query"])},
    {"name": "find_data",
     "description": ("Find which results table or figure holds the data for a question. Returns table and figure "
                     "names with a short description; follow with describe_table, query_table or get_figure."),
     "input_schema": _schema({"query": {"type": "string", "description": "The data you need, in words."},
                              "type": {"type": "string", "enum": ["table", "figure"]},
                              "hypothesis": _HYP, "k": _K}, ["query"])},
    {"name": "list_claims",
     "description": ("List every claim tested (H1-H8) or awaiting a test, with status and verdict. A verdict is "
                     "present only when the status is final; never state a verdict for any other claim."),
     "input_schema": _schema({}, [])},
    {"name": "get_claim",
     "description": ("Full record for one claim: verdict (final claims only), one-line finding, Thiel's quote, "
                     "sub-claims, caveats, tables, figures, linked notes, the summary's top-level keys and its "
                     "top-level scalar values (some claims have none; query summary_values for nested values). "
                     "The only source of verdicts."),
     "input_schema": _schema({"id": {"type": "string", "description": "Claim id, e.g. H2."}}, ["id"])},
    {"name": "get_chapter",
     "description": ("Everything about one chapter of the book: which claims test it (with verdicts if final), "
                     "untested placeholders, and the analyst's notes. Says when a chapter is untested."),
     "input_schema": _schema({"chapter": {"type": "integer",
                                          "description": "Chapter 0-15 (0 preface, 15 conclusion)."}}, ["chapter"])},
    {"name": "describe_table",
     "description": "Columns, types, ranges and provenance of one results table, before querying it.",
     "input_schema": _schema({"name": {"type": "string", "description": "Table name, e.g. h3c_top20_2014."}}, ["name"])},
    {"name": "query_table",
     "description": ("Run one read-only SELECT on the results database: the results tables (named like h2a_...), "
                     "summary_values (hypothesis, path, value_type, value_num, value_text: every number in the "
                     "summaries), catalog_tables and catalog_figures. Use it for every number you state. System "
                     "tables, settings and file functions are not available."),
     "input_schema": _schema({"sql": {"type": "string", "description": "One SELECT (WITH ... SELECT allowed)."},
                              "max_rows": {"type": "integer",
                                           "description": f"Rows to return, 1-{MAX_ROWS} (default {DEFAULT_ROWS})."}},
                             ["sql"])},
    {"name": "get_figure",
     "description": "One figure's caption and image file, by name (from find_data or get_claim).",
     "input_schema": _schema({"name": {"type": "string", "description": "Figure name, e.g. h1_fig1_ccdf."}}, ["name"])},
]
DEFINITIONS = {d["name"]: d for d in TOOL_DEFINITIONS}


def tool_definitions(strict: bool = True) -> list[dict]:
    """Tool definitions for the Messages API. `strict` guarantees schema-valid arguments; it can be
    turned off if a request feature in use is incompatible with it."""
    return [{**d, **({"strict": True} if strict else {})} for d in TOOL_DEFINITIONS]
