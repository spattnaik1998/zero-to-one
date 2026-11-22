// An answer's markdown as a tree of blocks and inline nodes. It never produces HTML: app.js builds
// elements and sets their text, so nothing in an answer or a tool result can become markup.
"use strict";

(function (root) {
  const OPEN = "", CLOSE = "";      // private-use characters around a citation number
  const MARKER = /(\d+)/g;
  const TRAILING_MARKERS = /(?:\s*\d+)+\s*$/;
  const SEPARATOR = /^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$/;
  const HEADING = /^(#{1,6})\s+(.*)$/;
  const ITEM = /^(\s*)([-*+•]|\d{1,3}[.)])\s+(.*)$/;

  function withMarkers(segments) {
    return segments.map((segment) => {
      const text = segment.text.replace(/[]/g, "");
      if (!segment.refs.length) return text;
      const body = text.replace(/\s+$/, "");
      const marks = segment.refs.map((n) => OPEN + n + CLOSE).join("");
      return body + marks + text.slice(body.length);
    }).join("").replace(/\r\n?/g, "\n");
  }

  const plain = (s) => s.replace(MARKER, "").trim();

  function cells(line) {
    let row = line.trim(), marks = "";
    const found = row.match(TRAILING_MARKERS);
    if (found) {                                 // a marker after the last pipe belongs in the last cell
      marks = found[0].trim();
      row = row.slice(0, found.index).trim();
    }
    row = row.replace(/^\|/, "").replace(/\|$/, "");
    const out = row.split(/(?<!\\)\|/).map((c) => c.trim());
    out[out.length - 1] += marks;
    return out;
  }

  function inline(text) {
    const out = [];
    const pattern = /(\d+)|\*\*(?=\S)(.+?)\*\*|`([^`\n]+)`|(?<![\w*])\*(?=\S)([^*\n]+?)\*(?![\w*])|\\([\\`*_$|#[\]~])/g;
    let at = 0;
    for (const m of text.matchAll(pattern)) {
      if (m.index > at) out.push({ type: "text", text: text.slice(at, m.index) });
      if (m[1]) out.push({ type: "ref", n: Number(m[1]) });
      else if (m[2]) out.push({ type: "strong", children: inline(m[2]) });
      else if (m[3]) {
        out.push({ type: "code", text: m[3].replace(MARKER, "") });
        for (const r of m[3].matchAll(MARKER)) out.push({ type: "ref", n: Number(r[1]) });
      } else if (m[4]) out.push({ type: "em", children: inline(m[4]) });
      else out.push({ type: "text", text: m[5] });
      at = m.index + m[0].length;
    }
    if (at < text.length) out.push({ type: "text", text: text.slice(at) });
    return out;
  }

  function parse(segments) {
    const lines = withMarkers(segments).split("\n");
    const blocks = [];
    let para = [], list = null;
    const endPara = () => {
      if (para.length) blocks.push({ type: "p", lines: para.map(inline) });
      para = [];
    };
    const end = () => {
      endPara();
      list = null;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i], bare = line.trim();
      if (!bare) {
        endPara();                               // a blank line doesn't restart a numbered list
        continue;
      }
      if (bare.startsWith("|") && i + 1 < lines.length && SEPARATOR.test(plain(lines[i + 1]))) {
        end();
        const head = cells(bare);
        const strays = [...lines[i + 1].matchAll(MARKER)].map((m) => m[0]).join("");
        head[head.length - 1] += strays;
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
        i--;
        blocks.push({ type: "table", head: head.map(inline), rows: rows.map((r) => r.map(inline)) });
        continue;
      }
      const heading = bare.match(HEADING);
      if (heading) {
        end();
        blocks.push({ type: "h", level: heading[1].length, children: inline(heading[2]) });
        continue;
      }
      const item = line.match(ITEM);
      if (item) {
        endPara();
        const ordered = /\d/.test(item[2]);
        const depth = Math.min(2, Math.floor(item[1].replace(/\t/g, "  ").length / 2));
        if (!list || (list.ordered !== ordered && depth === 0)) {
          list = { type: "list", ordered, start: ordered ? parseInt(item[2], 10) : 1, items: [] };
          blocks.push(list);
        }
        list.items.push({ depth, lines: [inline(item[3])] });
        continue;
      }
      if (list && /^\s/.test(line) && !para.length) {
        list.items[list.items.length - 1].lines.push(inline(bare));
        continue;
      }
      list = null;
      para.push(bare);
    }
    end();
    return blocks;
  }

  const api = { parse };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AnswerMarkdown = api;
})(globalThis);
