"""The agent's tools, each reading one store of a pinned, verified release (spec section 6.3).

A `Toolset` is bound to one `Release` for its whole life, so every answer in a session
comes from the same corpus version. `call(name, input)` validates the input (strict tool
schemas cannot express ranges, so limits are enforced here) and returns a JSON-able dict.
Every result item carries `source_id` and `voice`, which the answer verifier and citations use.

query_table is the only tool that takes free-form input. It runs on a read-only DuckDB
connection with file and network access disabled and the configuration locked, accepts one
SELECT statement, caps the rows returned, and stops queries that run too long.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from decimal import Decimal
from pathlib import Path

import duckdb

from ..release import Release
from ..retrieval import HybridIndex

MAX_K = 10
DEFAULT_K = 5
MAX_ROWS = 200
DEFAULT_ROWS = 50
QUERY_SECONDS = 5.0
KINDS = ("claim", "method", "result", "caveat")


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
    if isinstance(v, Decimal):                       # ROUND()/CAST give DECIMAL; return a number, not text
        v = float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)                                     # dates, decimals


class Toolset:
    def __init__(self, release: Release, embedder):
        model = release.manifest["models"]["embedding_model"]
        if embedder.model != model:
            raise ValueError(f"release {release.release_id} was embedded with {model}; "
                             f"the query embedder uses {embedder.model}")
        self.release, self.embedder = release, embedder
        self.indexes = {name: HybridIndex.load(release.file("indexes.duckdb"), name)
                        for name in ("notes", "analysis", "catalog")}
        self.claims = {c["id"]: c for c in self._json("claims_registry.json")}
        self.chapters = {c["number"]: c for c in self._json("chapter_map.json")}
        self.cards = {c["name"]: c for c in self._json("schema_cards.json")}
        self.figures = {f["name"]: f for f in self._json("figures.json")}
        self.notes = {n["id"]: n for n in map(json.loads, self.release.file("notes_chunks.jsonl")
                                               .read_text(encoding="utf-8").splitlines()) if n}
        self.db_path = release.file("analysis.duckdb")

    def _json(self, name: str):
        return json.loads(self.release.file(name).read_text(encoding="utf-8"))

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, tool_input: dict) -> dict:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None or name not in {d["name"] for d in TOOL_DEFINITIONS}:
            raise ToolError(f"unknown tool {name!r}")
        if not isinstance(tool_input, dict):
            raise ToolError("tool input must be an object")
        allowed = set(next(d for d in TOOL_DEFINITIONS if d["name"] == name)["input_schema"]["properties"])
        extra = set(tool_input) - allowed
        if extra:
            raise ToolError(f"unexpected argument(s) {sorted(extra)}; allowed: {sorted(allowed)}")
        result = handler(**tool_input)
        return {"tool": name, "release_id": self.release.release_id, **result}

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
        return {"results": [{
            "source_id": h.doc_id, "voice": "analyst", "opinion": h.meta["opinion"],
            "chapter": h.meta["chapter"], "chapter_title": h.meta["chapter_title"], "page": h.meta["page"],
            "text": h.text} for h in hits]}

    def tool_search_analysis(self, query, hypothesis=None, kind=None, k=None) -> dict:
        if hypothesis is not None and hypothesis not in self.claims:
            raise ToolError(f"hypothesis must be one of {sorted(self.claims)}")
        if kind is not None and kind not in KINDS:
            raise ToolError(f"kind must be one of {list(KINDS)}")
        hits = self._search("analysis", query, k, {"hypothesis": hypothesis, "kind": kind})
        return {"results": [{
            "source_id": h.doc_id, "voice": h.meta["voice"], "hypothesis": h.meta["hypothesis"],
            "sub_claim": h.meta["sub_claim"], "kind": h.meta["kind"], "section": h.meta["section"],
            "notebook": h.meta["notebook"], "cell": h.meta["cell"], "cell_type": h.meta["cell_type"],
            "text": h.text} for h in hits]}

    def tool_find_data(self, query, type=None, hypothesis=None, k=None) -> dict:
        if type is not None and type not in ("table", "figure"):
            raise ToolError("type must be 'table' or 'figure'")
        if hypothesis is not None and hypothesis not in self.claims:
            raise ToolError(f"hypothesis must be one of {sorted(self.claims)}")
        hits = self._search("catalog", query, k, {"type": type, "hypothesis": hypothesis})
        return {"results": [{"source_id": h.doc_id, "voice": "evidence", "type": h.meta["type"],
                             "name": h.meta["name"], "hypothesis": h.meta["hypothesis"], "text": h.text}
                            for h in hits]}

    # ------------------------------------------------------------ registry
    def _claim_view(self, c: dict, full: bool) -> dict:
        view = {"source_id": f"claim:{c['id']}", "voice": "evidence", "id": c["id"], "title": c["title"],
                "claim": c["claim"], "chapters": c["chapters"], "status": c["status"],
                # verdicts and one-line summaries exist only for final claims (the registry enforces it)
                "verdict": c["verdict"], "one_line": c["one_line"]}
        if full:
            view.update({k: c[k] for k in ("thiel_quote", "sub_claims", "caveats", "notebook", "tables", "figures")})
            view["note_ids"] = c["note_ids"]
            view["headline_values"] = self._headline_values(c["id"])
            if c["status"] != "final":
                view["note"] = ("no verdict: this claim is " + c["status"].replace("_", " ")
                                + "; do not state or imply one")
        return view

    def _headline_values(self, hypothesis: str) -> list[dict]:
        """Top-level scalar summary values (no nesting), cited by their summary_values path."""
        with self._connect() as con:
            rows = con.execute("SELECT path, value_type, value_num, value_text FROM summary_values "
                               "WHERE hypothesis = ? AND path NOT LIKE '%.%' AND path NOT LIKE '%[%' "
                               "ORDER BY path", [hypothesis]).fetchall()
        return [{"source_id": f"summary:{hypothesis}:{p}", "path": p,
                 "value": _json_value(vn) if t in ("int", "float") else vt} for p, t, vn, vt in rows]

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
            "source_id": f"chapter:{number}", "voice": "evidence", "chapter": number, "title": e["title"],
            "tested": e["tested"],
            "hypotheses": [self._claim_view(self.claims[h], full=False) for h in e["hypotheses"]],
            "untested_placeholders": list(e["untested"]),
            "note": None if e["tested"] else "no hypothesis tests this chapter; say so instead of giving a verdict",
            "notes": [{"source_id": i, "voice": "analyst", "opinion": self.notes[i]["opinion"],
                       "text": self.notes[i]["text"]} for i in e["note_ids"]],
        }]}

    # ------------------------------------------------------------ data
    def tool_describe_table(self, name) -> dict:
        name = _str(name, "name")
        if name not in self.cards:
            raise ToolError(f"no table {name!r}; use find_data to look tables up")
        c = self.cards[name]
        return {"results": [{"source_id": f"table:{name}", "voice": "evidence", **c}]}

    def tool_get_figure(self, name) -> dict:
        name = _str(name, "name")
        if name not in self.figures:
            raise ToolError(f"no figure {name!r}; use find_data with type 'figure' to look figures up")
        f = self.figures[name]
        return {"results": [{"source_id": f"figure:{name}", "voice": "evidence", "name": name,
                             "hypothesis": f["hypothesis"], "caption": f["caption"],
                             # a path inside the release, never a local filesystem path
                             "file": f"figures/{name}.png"}]}

    def _connect(self):
        return duckdb.connect(str(self.db_path), read_only=True,
                              config={"enable_external_access": False, "lock_configuration": True})

    def tool_query_table(self, sql, max_rows=None) -> dict:
        sql = _str(sql, "sql").rstrip().rstrip(";").strip()
        cap = _int(max_rows, "max_rows", DEFAULT_ROWS, 1, MAX_ROWS)
        try:
            statements = duckdb.extract_statements(sql)
        except duckdb.Error as e:
            raise ToolError(f"SQL does not parse: {str(e).splitlines()[0]}")
        if len(statements) != 1:
            raise ToolError("send exactly one SQL statement")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise ToolError("only SELECT queries are allowed (WITH ... SELECT is fine)")
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
        truncated = len(rows) > cap
        return {"results": [{
            "source_id": "query:" + hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12],
            "voice": "evidence", "sql": sql,
            "columns": columns, "rows": [[_json_value(v) for v in r] for r in rows[:cap]],
            "row_count": min(len(rows), cap), "truncated": truncated,
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
                     "sub-claims, caveats, tables, figures, linked notes and headline summary values. The only "
                     "source of verdicts."),
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
                     "summary_values (hypothesis, path, value_num, value_text: every number in the summaries), "
                     "catalog_tables and catalog_figures. Use it for every number you state."),
     "input_schema": _schema({"sql": {"type": "string", "description": "One SELECT (WITH ... SELECT allowed)."},
                              "max_rows": {"type": "integer",
                                           "description": f"Rows to return, 1-{MAX_ROWS} (default {DEFAULT_ROWS})."}},
                             ["sql"])},
    {"name": "get_figure",
     "description": "One figure's caption and image file, by name (from find_data or get_claim).",
     "input_schema": _schema({"name": {"type": "string", "description": "Figure name, e.g. h1_fig1_ccdf."}}, ["name"])},
]


def tool_definitions(strict: bool = True) -> list[dict]:
    """Tool definitions for the Messages API. `strict` guarantees schema-valid arguments; it can be
    turned off if a request feature in use is incompatible with it."""
    return [{**d, **({"strict": True} if strict else {})} for d in TOOL_DEFINITIONS]
