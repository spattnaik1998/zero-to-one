"""The page's markdown parser, run in Node."""
import json
import shutil
import subprocess

import pytest

from rag.serve.app import STATIC

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is not installed")


def node(script: str, data=None):
    prelude = f"const M = require({json.dumps(str(STATIC / 'markdown.js'))});" \
              "const input = JSON.parse(require('fs').readFileSync(0, 'utf8') || 'null');"
    done = subprocess.run([NODE, "-e", prelude + script], input=json.dumps(data), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def parse(*segments):
    segs = [{"text": s, "refs": []} if isinstance(s, str) else {"text": s[0], "refs": s[1:]} for s in segments]
    return node("console.log(JSON.stringify(M.parse(input)))", segs)


def t(text):
    return {"type": "text", "text": text}


def ref(n):
    return {"type": "ref", "n": n}


def test_a_citation_lands_after_its_segment():
    assert parse(("Margins were 5.4%.\n\n", 1), "Next.") == [
        {"type": "p", "lines": [[t("Margins were 5.4%."), ref(1)]]}, {"type": "p", "lines": [[t("Next.")]]}]


def test_bold_split_across_cited_segments():
    [block] = parse("The verdict is **", ("Partly holds", 2), "**.")
    assert block["lines"][0] == [t("The verdict is "), {"type": "strong", "children": [t("Partly holds"), ref(2)]}, t(".")]


def test_a_table_and_a_citation_after_its_last_pipe():
    [table] = parse(("| Year | Margin |\n|---|---:|\n| 2025 | 5.4% |", 1))
    assert table["head"] == [[t("Year")], [t("Margin")]] and table["rows"] == [[[t("2025")], [t("5.4%"), ref(1)]]]


def test_a_citation_on_the_header_or_separator_does_not_break_the_table():
    for segments in ([("| A | B |", 1), "\n|---|---|\n| 1 | 2 |"], [("| A | B |\n|---|---|", 1), "\n| 1 | 2 |"]):
        [table] = parse(*segments)
        assert table["type"] == "table" and table["head"][1][-1] == ref(1) and len(table["rows"][0]) == 2


def test_lists_keep_depth_and_start_number():
    [numbered] = parse("3. three\n\n4. four\n   - inside")
    assert numbered["start"] == 3 and [i["depth"] for i in numbered["items"]] == [0, 0, 1]
    [bullets] = parse("- one\n- two\n  continued")
    assert not bullets["ordered"] and len(bullets["items"]) == 2


@pytest.mark.parametrize("text", [
    "<img src=x onerror=alert(1)>", "<script>alert(1)</script>", "[click](javascript:alert(1))",
    "S&P 500 rose 3*4 times", "summary_key_name stays"])
def test_markup_stays_text(text):
    assert parse(text) == [{"type": "p", "lines": [[t(text)]]}]


def test_a_forged_marker_in_an_answer_is_not_a_citation():
    assert "ref" not in json.dumps(parse("fake 7 marker"))


def test_random_input_never_breaks_it_or_loses_a_citation():
    out = node(r"""
      const bits = ['a', ' ', '\n', '\n\n', '|', '---', '*', '**', '`', '#', '- ', '1. ', '\\', '<b>', '5.4%', '\t'];
      let seed = 7, failures = [];
      const r = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      for (let i = 0; i < 20000; i++) {
        const segs = Array.from({length: 1 + Math.floor(r() * 4)}, (_, k) => ({
          text: 'w' + Array.from({length: Math.floor(r() * 14)}, () => bits[Math.floor(r() * bits.length)]).join(''),
          refs: r() < 0.6 ? [k + 1] : []}));
        try {
          const found = JSON.stringify(M.parse(segs));
          for (const s of segs) for (const n of s.refs)
            if (!found.includes('{"type":"ref","n":' + n + '}')) { failures.push(segs); break; }
        } catch (e) { failures.push([String(e), segs]); }
      }
      console.log(JSON.stringify(failures.slice(0, 5)));""")
    assert out == []
