"""An answer as the page shows it. Built only from an answer that has been checked and logged.

Only what is shown goes to the browser: no tool inputs, no first draft, no local paths.
"""
from __future__ import annotations

from ..agent.verifier import Evidence

EXCERPTS, EXCERPT_CHARS = 4, 1500
CHECKS = {
    "numbers": "A number isn't backed by the results",
    "verdicts": "A verdict doesn't match the analysis",
    "untested": "An untested chapter isn't called untested",
    "citations": "A statement has no supporting citation",
    "voices": "The analyst's or Thiel's words aren't labelled",
    "format": "Raw data appears in the text",
    "advice": "Part of it reads as investment advice",
    "incomplete": "The answer didn't finish",
}


def answer_view(checked, record: dict, session) -> dict:
    answer = checked.answer
    voices = Evidence.from_calls(session.calls_history).sources
    sources: dict[str, dict] = {}
    for c in answer.citations:
        source = sources.setdefault(c.source_id, {
            "n": len(sources) + 1, "id": c.source_id, "title": c.title or c.source_id,
            "voice": voices.get(c.source_id, "unknown"), "excerpts": []})
        excerpt = (c.cited_text or "").strip()[:EXCERPT_CHARS]
        if excerpt and excerpt not in source["excerpts"] and len(source["excerpts"]) < EXCERPTS:
            source["excerpts"].append(excerpt)

    segments = [{"text": s["text"],
                 "refs": list(dict.fromkeys(sources[x]["n"] for x in s["sources"] if x in sources))}
                for s in answer.segments]
    if segments:
        segments[0]["text"] = segments[0]["text"].lstrip()
        segments[-1]["text"] = segments[-1]["text"].rstrip()

    known = session.toolset.figures
    names = [t.input.get("name") for t in checked.all_tool_calls if t.ok and t.name == "get_figure"]
    names += [s.removeprefix("figure:") for s in sources if s.startswith("figure:")]
    figures = [{"name": n, "caption": known[n]["caption"]} for n in dict.fromkeys(names) if n in known]

    notice = None
    if answer.refused:
        notice = "The model declined to answer this question. Try rewording it."
    elif not answer.ok:
        notice = "The answer was cut off before it finished, so don't rely on it. Try a narrower question."
    elif not answer.text.strip():
        notice = "The answer came back empty. Ask again."

    return {
        "segments": segments, "sources": list(sources.values()), "figures": figures, "notice": notice,
        "flagged": checked.flagged, "revised": checked.revised,
        "failed_checks": [{"check": CHECKS.get(x.rule, x.rule), "detail": x.detail}
                          for x in checked.verification.errors],
        "record_id": record["record_id"], "release_id": answer.release_id,
        "seconds": round(record["duration_ms"] / 1000, 1),
    }
