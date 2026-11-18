import json
from pathlib import Path

import duckdb
import pytest

from rag.config import load_settings
from rag.ingest.index import build_indexes
from rag.release import Check, GateReport, current_release_id, open_release, publish
from rag.tools import toolset as ts_mod
from rag.tools.toolset import TOOL_DEFINITIONS, ToolError, Toolset, tool_definitions

# ---------------------------------------------------------------- definitions
def test_definitions_fit_strict_tool_use():
    unsupported = {"minimum", "maximum", "minLength", "maxLength", "multipleOf"}
    for d in TOOL_DEFINITIONS:
        schema = d["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        assert not unsupported & {k for p in schema["properties"].values() for k in p}
        assert hasattr(Toolset, f"tool_{d['name']}")
    assert all(d["strict"] for d in tool_definitions())
    assert not any("strict" in d for d in tool_definitions(strict=False))
    assert "search_book" not in {d["name"] for d in TOOL_DEFINITIONS}


def test_release_and_embedder_must_match(toolset):
    class Other:
        model = "text-embedding-other"
    with pytest.raises(ValueError, match="embedded with hash-test"):
        Toolset(toolset.release, Other())


def test_every_result_is_pinned_and_labelled(toolset):
    r = toolset.call("search_notes", {"query": "CEO pay"})
    assert r["release_id"] == toolset.release.release_id and r["tool"] == "search_notes"
    assert all({"source_id", "voice"} <= set(x) for x in r["results"])


# ---------------------------------------------------------------- dispatch and validation
@pytest.mark.parametrize("name, inp, message", [
    ("search_book", {"query": "x"}, "unknown tool"),
    ("search_notes", {"query": "x", "top": 3}, "unexpected argument"),
    ("search_notes", {"query": ""}, "non-empty"),
    ("search_notes", {"query": "x", "k": 0}, "between 1 and 10"),
    ("search_notes", {"query": "x", "k": 11}, "between 1 and 10"),
    ("search_notes", {"query": "x", "k": True}, "integer"),
    ("search_notes", {"query": "x", "chapter": 16}, "between 0 and 15"),
    ("search_analysis", {"query": "x", "hypothesis": "H42"}, "hypothesis must be one of"),
    ("search_analysis", {"query": "x", "kind": "opinion"}, "kind must be one of"),
    ("find_data", {"query": "x", "type": "chart"}, "'table' or 'figure'"),
    ("get_claim", {"id": "H42"}, "no claim H42"),
    ("get_chapter", {"chapter": -1}, "between 0 and 15"),
    ("describe_table", {"name": "nope"}, "use find_data"),
    ("get_figure", {"name": "nope"}, "use find_data"),
])
def test_bad_calls_get_actionable_errors(toolset, name, inp, message):
    with pytest.raises(ToolError, match=message):
        toolset.call(name, inp)


def test_input_must_be_an_object(toolset):
    with pytest.raises(ToolError, match="object"):
        toolset.call("list_claims", ["x"])


# ---------------------------------------------------------------- search tools
def test_search_notes_filters_and_voice(toolset):
    hits = toolset.call("search_notes", {"query": "CEO pay board", "k": 2})["results"]
    assert hits[0]["source_id"] == "note-09-01" and hits[0]["voice"] == "analyst"
    assert len(hits) == 2
    only = toolset.call("search_notes", {"query": "computers", "opinion_only": True})["results"]
    assert [h["source_id"] for h in only] == ["note-12-01"] and only[0]["opinion"] is True
    assert toolset.call("search_notes", {"query": "airlines", "chapter": 9})["results"][0]["chapter"] == 9


def test_search_analysis_and_find_data(toolset):
    caveats = toolset.call("search_analysis", {"query": "firms acquired", "kind": "caveat"})["results"]
    assert [h["source_id"] for h in caveats] == ["nb-h3-002"] and caveats[0]["kind"] == "caveat"
    tables = toolset.call("find_data", {"query": "largest firms 2014 top 20", "type": "table"})["results"]
    assert tables[0]["name"] == "h3c_top20_2014" and tables[0]["type"] == "table"
    figs = toolset.call("find_data", {"query": "tail", "type": "figure"})["results"]
    assert [f["name"] for f in figs] == ["h1_fig1_ccdf"]


# ---------------------------------------------------------------- registry tools
def test_claims_carry_verdicts_only_when_final(toolset):
    listed = {c["id"]: c for c in toolset.call("list_claims", {})["results"]}
    assert listed["H1"]["verdict"] == "Partly holds" and listed["H9"]["verdict"] is None
    h9 = toolset.call("get_claim", {"id": "h9"})["results"][0]           # ids are case-insensitive
    assert h9["status"] == "untested" and "no verdict" in h9["note"] and h9["one_line"] is None
    h1 = toolset.call("get_claim", {"id": "H1"})["results"][0]
    assert h1["source_id"] == "claim:H1" and "note" not in h1
    assert h1["headline_values"] == [                  # ints stay ints, bools stay bools
        {"source_id": "summary:H1:alpha", "path": "alpha", "value": 1.8},
        {"source_id": "summary:H1:n_firms", "path": "n_firms", "value": 39363},
        {"source_id": "summary:H1:ok", "path": "ok", "value": True}]
    assert isinstance(h1["headline_values"][1]["value"], int) and h1["summary_keys"] == ["alpha"]
    assert toolset.call("get_claim", {"id": "H3"})["results"][0]["headline_values"] == []


def test_get_chapter_tested_and_untested(toolset):
    seven = toolset.call("get_chapter", {"chapter": 7})["results"][0]
    assert seven["tested"] and [h["id"] for h in seven["hypotheses"]] == ["H1"] and seven["note"] is None
    nine = toolset.call("get_chapter", {"chapter": 9})["results"][0]
    assert not nine["tested"] and nine["untested_placeholders"] == ["H9"] and "no hypothesis" in nine["note"]
    assert [n["source_id"] for n in nine["notes"]] == ["note-09-01"] and nine["notes"][0]["voice"] == "analyst"


def test_table_and_figure_details(toolset):
    card = toolset.call("describe_table", {"name": "h3c_top20_2014"})["results"][0]
    assert card["source_id"] == "table:h3c_top20_2014" and card["rows"] == 20
    fig = toolset.call("get_figure", {"name": "h1_fig1_ccdf"})["results"][0]
    assert fig["file"] == "figures/h1_fig1_ccdf.png"                    # inside the release, not a local path
    assert toolset.release.file(fig["file"]).read_bytes() == b"png"


# ---------------------------------------------------------------- query_table
def test_query_returns_rows_in_the_queries_order(toolset):
    r = toolset.call("query_table", {"sql": "SELECT firm, rank_2025 FROM h3c_top20_2014 ORDER BY rank_2025 DESC NULLS LAST;"})
    res = r["results"][0]
    assert res["columns"] == ["firm", "rank_2025"]
    assert [row[0] for row in res["rows"]] == ["GE", "Exxon", "Apple", "Odd"]   # NULL last, as written
    assert res["rows"][3][1] is None
    json.dumps(r)


def test_numbers_come_back_as_json_numbers(toolset):
    row = toolset.call("query_table", {"sql": "SELECT ROUND(2.345, 2) AS d, 'NaN'::DOUBLE AS n, DATE '2014-09-16' AS day"})
    d, n, day = row["results"][0]["rows"][0]
    assert d == 2.35 and isinstance(d, float)          # DECIMAL -> number, not the string "2.35"
    assert n is None and day == "2014-09-16"           # NaN -> null (valid JSON); dates -> ISO text
    json.dumps(row, allow_nan=False)


def test_query_row_cap_and_truncation(toolset):
    res = toolset.call("query_table", {"sql": "SELECT * FROM range(500)", "max_rows": 3})["results"][0]
    assert res["row_count"] == 3 and res["truncated"] is True
    assert toolset.call("query_table", {"sql": "SELECT 1 AS one"})["results"][0]["truncated"] is False
    with pytest.raises(ToolError, match="between 1 and 200"):
        toolset.call("query_table", {"sql": "SELECT 1", "max_rows": 201})


def test_query_source_ids_differ_per_query(toolset):
    a = toolset.call("query_table", {"sql": "SELECT 1"})["results"][0]["source_id"]
    b = toolset.call("query_table", {"sql": "SELECT 2"})["results"][0]["source_id"]
    assert a != b and a.startswith("query:")


@pytest.mark.parametrize("sql, message", [
    ("DELETE FROM h3c_top20_2014", "only SELECT"),
    ("CREATE TABLE x AS SELECT 1", "only SELECT"),
    ("PRAGMA version", "only SELECT"),
    ("SET enable_external_access = true", "only SELECT"),
    ("ATTACH 'x.duckdb'", "only SELECT"),
    ("COPY h3c_top20_2014 TO 'out.csv'", "only SELECT"),
    ("SELECT 1; SELECT 2", "exactly one"),
    ("SELECT 1; DELETE FROM h3c_top20_2014", "exactly one"),
    ("SELEC firm FROM h3c_top20_2014", "does not parse"),
    ("SELECT * FROM read_csv('README.md')", "table function 'read_csv'"),
    ("SELECT * FROM glob('*')", "table function 'glob'"),
    ("SELECT * FROM no_such_table", "not available"),
    ("SELECT firm FROM h3c_top20_2014 WHERE no_such_column = 1", "query failed"),
])
def test_query_refuses_anything_but_one_select(toolset, sql, message):
    with pytest.raises(ToolError, match=message):
        toolset.call("query_table", {"sql": sql})


def test_engine_still_blocks_files_without_the_allow_list(toolset):
    """Second layer: even if the allow-list missed something, the connection cannot touch files."""
    with toolset._connect() as con:
        for sql in ("SELECT * FROM read_csv('README.md')", "COPY (SELECT 1) TO 'x.csv'", "ATTACH 'x.duckdb'"):
            with pytest.raises(duckdb.Error):
                con.execute(sql)


def test_query_timeout(toolset, monkeypatch):
    monkeypatch.setattr(ts_mod, "QUERY_SECONDS", 0.3)
    with pytest.raises(ToolError, match="stopped after"):
        toolset.call("query_table", {"sql": "SELECT count(*) FROM range(10000000000) a, range(1000) b"})


def test_the_release_is_unchanged_after_all_calls(toolset):
    for sql in ("SELECT * FROM h3c_top20_2014", "SELECT * FROM summary_values"):
        toolset.call("query_table", {"sql": sql})
    open_release(load_settings(environ={"RAG_BUILD_DIR": str(toolset.release.path.parent.parent)}, dotenv={}),
                 toolset.release.release_id)                            # re-hashes every file


# ---------------------------------------------------------------- the real release
def _real():
    s = load_settings()
    return s if s.openai_api_key and current_release_id(s) else None


@pytest.mark.skipif(_real() is None, reason="needs a published release and an OpenAI key (never in CI)")
def test_real_release_tools():
    from rag.ingest.embed import EmbeddingCache, OpenAIEmbedder
    s = _real()
    cache = EmbeddingCache(s.build_dir / "cache" / "embeddings.duckdb")
    try:
        ts = Toolset(open_release(s), OpenAIEmbedder(s.embedding_model, s.openai_api_key, cache))
        assert ts.call("search_notes", {"query": "How much should a CEO pay himself?"})["results"][0]["source_id"] == "note-09-06"
        claims = {c["id"]: c for c in ts.call("list_claims", {})["results"]}
        assert claims["H9"]["verdict"] is None and all(claims[f"H{n}"]["verdict"] for n in range(1, 9))
        survivors = ts.call("query_table", {"sql": "SELECT count(*) FROM h3c_top20_2014 WHERE still_top20"})
        h3 = ts.call("query_table", {"sql": "SELECT value_num FROM summary_values "
                                            "WHERE hypothesis='H3' AND path='top20_2014_still'"})
        assert survivors["results"][0]["rows"][0][0] == h3["results"][0]["rows"][0][0] == 8
        assert [c["number"] for c in ts.chapters.values() if not c["tested"]] == [0, 8, 9, 10, 14]
        # NaN sorts above every number in DuckDB, so ORDER BY ... DESC would put it first; the real
        # data has none (missing CSV values load as NULL), and this keeps it that way
        with ts._connect() as con:
            cols = con.execute("SELECT table_name, column_name FROM information_schema.columns "
                               "WHERE data_type IN ('DOUBLE', 'FLOAT')").fetchall()
            assert not [c for c in cols if con.execute(
                f'SELECT count(*) FROM "{c[0]}" WHERE isnan("{c[1]}")').fetchone()[0]]
    finally:
        cache.close()


# ---------------------------------------------------------------- code review findings
@pytest.mark.parametrize("sql, message", [
    ("SELECT path FROM duckdb_databases()", "table function 'duckdb_databases'"),
    ("SELECT * FROM duckdb_settings()", "table function 'duckdb_settings'"),
    ("SELECT current_setting('temp_directory')", "function 'current_setting'"),
    ("SELECT getenv('USERNAME')", "function 'getenv'"),
    ("SELECT version()", "function 'version'"),
    ("SELECT * FROM query('SELECT 1')", "table function 'query'"),
    ("SELECT * FROM information_schema.tables", "table 'information_schema.tables'"),
    ("SELECT * FROM pg_catalog.pg_tables", "table 'pg_catalog.pg_tables'"),
    ("SELECT * FROM other_db.main.h3c_top20_2014", "not available"),
    ("SELECT * FROM h3c_top20_2014 WHERE firm IN (SELECT database_name FROM duckdb_databases())", "duckdb_databases"),
])
def test_system_catalog_settings_and_environment_are_not_reachable(toolset, sql, message):
    """These parse as SELECT and passed the first guards; they exposed local paths and settings."""
    with pytest.raises(ToolError, match=message):
        toolset.call("query_table", {"sql": sql})


@pytest.mark.parametrize("sql", [
    "WITH top AS (SELECT firm FROM h3c_top20_2014 WHERE still_top20) SELECT count(*) FROM top",
    "SELECT * FROM main.h3c_top20_2014",
    "SELECT r.range FROM range(3) r",
    "SELECT upper(firm), round(rank_2014 / 2.0, 1) FROM h3c_top20_2014",
    "SELECT * FROM summary_values s JOIN h3c_top20_2014 t ON s.path = t.firm",
])
def test_ordinary_analysis_queries_still_work(toolset, sql):
    assert toolset.call("query_table", {"sql": sql})["results"][0]["columns"]


def test_duckdb_temp_files_live_outside_the_release(toolset):
    with toolset._connect() as con:                    # not through query_table, which refuses current_setting
        temp = Path(con.execute("SELECT current_setting('temp_directory')").fetchone()[0]).resolve()
        limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    release = toolset.release.path.resolve()
    assert release not in temp.parents and temp != release
    assert temp == toolset.scratch_dir.resolve() and "MiB" in limit


def test_missing_required_arguments_are_tool_errors(toolset):
    for name in ("search_notes", "get_claim", "query_table", "get_chapter"):
        with pytest.raises(ToolError, match="missing required"):
            toolset.call(name, {})


def test_lists_structs_and_bytes_come_back_as_json(toolset):
    row = toolset.call("query_table", {"sql": "SELECT [1.5, 'NaN'::DOUBLE] AS l, {'a': 1, 'b': ROUND(2.345, 2)} AS s, "
                                              "'x'::BLOB AS b, TIME '12:30:00' AS t"})["results"][0]["rows"][0]
    assert row[0] == [1.5, None] and row[1] == {"a": 1, "b": 2.35}
    assert row[2] == {"base64": "eA=="} and row[3] == "12:30:00"


def test_hypothesis_ids_are_case_insensitive_everywhere(toolset):
    assert toolset.call("search_analysis", {"query": "firms", "hypothesis": "h3"})["results"]
    assert toolset.call("find_data", {"query": "firms", "hypothesis": "h3"})["results"][0]["hypothesis"] == "H3"


def test_blank_lines_in_candidate_files_are_skipped(tmp_path):
    from rag.ingest.index import read_jsonl
    f = tmp_path / "x.jsonl"
    f.write_text("\n".join(['{"id": 1}', "", "   ", '{"id": 2}', ""]))
    assert read_jsonl(f) == [{"id": 1}, {"id": 2}]


def test_bm25_is_built_once_per_filter_set(toolset):
    index = toolset.indexes["notes"]
    for _ in range(3):
        toolset.call("search_notes", {"query": "board"})
        toolset.call("search_notes", {"query": "board", "chapter": 9})
    assert len(index._bm25) == 2
