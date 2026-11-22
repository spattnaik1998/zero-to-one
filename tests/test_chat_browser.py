"""The chat in a real browser: real page, server, agent loop, tools, checker and audit log, with a
scripted model. Skipped where neither Chrome nor Edge is installed."""
import io
import threading
from types import SimpleNamespace as NS

import pytest

from browser import Browser, find_browser
from chat_support import ScriptedClaude, call, make_chat, reply, said, serving
from rag.agent.audit import AuditLog

BROWSER = find_browser()
pytestmark = pytest.mark.skipif(BROWSER is None, reason="no Chrome or Edge")
pytest.importorskip("websocket")

ATTACK = '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>'
QUESTION = "Does H1 hold? Show the chart."
DONE = "document.querySelectorAll('.turn')[{n}]?.dataset.state === 'done'"


@pytest.fixture
def page(toolset, tmp_path):
    from PIL import Image
    png = io.BytesIO()
    Image.new("RGB", (900, 420), "#2a78d6").save(png, "PNG")

    client = ScriptedClaude({
        QUESTION: [
            reply(call("a", "get_claim", id="H1"), call("b", "get_chapter", chapter=12),
                  call("c", "get_figure", name="h1_fig1_ccdf"), stop="tool_use"),
            reply(said("**The verdict on H1 is Partly holds.**", "claim:H1", title=f"Claim H1 {ATTACK}", excerpt=ATTACK),
                  said("\n\nThiel says it himself.", "quote:H1"),
                  said("\n\nThe analyst's note agrees.", "note-12-01"),
                  said(f"\n\n{ATTACK}\n\n| Claim | Standing |\n|---|---|\n| A very long cell " + "word " * 30
                       + "| kept |"))],
        "Make up a number.": [reply(said("Survival was 42%.")), reply(said("Survival was 43%."))],
    })
    chat = make_chat(toolset, client, tmp_path / "log", evaluated=False)
    chat.figure_bytes = png.getvalue()
    real_figure = chat.figure
    chat.figure = lambda *a: real_figure(*a) and chat.figure_bytes        # the fixture's chart isn't a real image
    with serving(chat) as (port, auth):
        browser = Browser(BROWSER, tmp_path / "profile")
        try:
            browser.resize(1280, 860)
            yield NS(browser=browser, client=client, url=f"http://127.0.0.1:{port}/", code=auth.codes["ana"],
                     audit=AuditLog(tmp_path / "log"))
        finally:
            browser.close()


def sign_in(page):
    page.browser.goto(f"{page.url}#code={page.code}")
    page.browser.wait("document.querySelector('#composer') && !document.querySelector('#composer').hidden")


def ask(page, question, n=0):
    page.browser.type("#question", question)
    page.browser.press_enter()
    page.browser.wait(DONE.format(n=n))


def test_signing_in(page):
    b = page.browser
    b.goto(page.url)
    b.wait("document.querySelector('#sign-in') && !document.querySelector('#sign-in').hidden")
    assert b.js("document.querySelector('#composer').hidden")
    sign_in(page)
    assert b.js("location.hash") == ""                       # the code doesn't stay in the address bar
    assert b.js("document.querySelector('#user').textContent") == "ana"
    assert not b.js("document.querySelector('#unevaluated').hidden")
    assert b.js("document.cookie") == ""                     # HttpOnly
    assert b.problems == []


def test_a_wrong_link_explains_itself(page):
    b = page.browser
    b.goto(f"{page.url}#code=wrong")
    b.wait("document.querySelector('#sign-in-message')?.textContent.includes(\"doesn't work\")")
    assert b.js("document.querySelector('#composer').hidden")


def test_progress_is_shown_and_the_draft_is_not(page):
    b = page.browser
    sign_in(page)
    gate = page.client.hold[QUESTION] = threading.Event()
    b.type("#question", QUESTION)
    b.press_enter()
    b.wait("document.querySelector('.progress li')?.parentElement.textContent.includes('Working out')")
    assert b.js("document.querySelector('#ask').disabled") and "Partly holds" not in b.js("document.body.innerText")
    gate.set()
    b.wait(DONE.format(n=0))
    assert not b.js("document.querySelector('#ask').disabled")


def test_an_answer_has_sources_by_voice_a_chart_and_no_injected_markup(page):
    b = page.browser
    sign_in(page)
    ask(page, QUESTION)
    assert b.js("document.querySelector('.answer strong').textContent").startswith("The verdict on H1")
    assert b.js("[...document.querySelectorAll('.answer .cite')].map(e => e.dataset.voice)") == \
        ["evidence", "thiel", "analyst"]
    assert b.js("[...document.querySelectorAll('.source .voice')].map(e => e.textContent)") == ["E", "T", "A"]
    colours = b.js("[...document.querySelectorAll('.source .voice')].map(e => getComputedStyle(e).backgroundColor)")
    assert len(set(colours)) == 3
    b.wait("document.querySelector('figure img')?.naturalWidth === 900")
    assert b.js("document.querySelectorAll('.answer table tbody tr').length") == 1
    b.js("document.querySelector('.answer .cite').click()")
    assert b.js("document.querySelector('.source details').open") and b.js(
        "document.querySelector('.source.selected') !== null")
    assert b.js("window.pwned === undefined && !document.querySelector('.answer img, .sources img, #record script')")
    assert "<img src=x" in b.js("document.querySelector('.answer').innerText")       # shown as text
    assert b.problems == []
    record = [r for _, _, r in page.audit.records()][-1]
    assert record["record_id"] in b.js("document.querySelector('.turn-margin .logged').textContent")


def test_a_failed_check_is_shown_above_the_answer(page):
    b = page.browser
    sign_in(page)
    ask(page, "Make up a number.")
    assert "failed the automatic check" in b.js("document.querySelector('.warning').textContent")
    assert "isn't backed" in b.js("document.querySelector('.warning li').textContent")
    assert b.js("""!!(document.querySelector('.warning').compareDocumentPosition(
                      document.querySelector('.answer')) & Node.DOCUMENT_POSITION_FOLLOWING)""")
    assert "42%" not in b.js("document.querySelector('.answer').textContent")


def test_reload_restores_and_a_new_conversation_and_sign_out_clear(page):
    b = page.browser
    sign_in(page)
    ask(page, QUESTION)
    b.js("window.beforeReload = true")
    b.send("Page.reload")
    b.wait("window.beforeReload === undefined && " + DONE.format(n=0))
    ask(page, "Make up a number.", n=1)
    b.js("document.querySelector('#new-conversation').click()")
    b.wait("document.querySelectorAll('.turn').length === 0 && !document.querySelector('#start').hidden")
    b.js("document.querySelector('#sign-out').click()")
    b.wait("!document.querySelector('#sign-in').hidden")
    left = b.js("document.body.textContent")
    assert "Partly holds" not in left, left
    assert b.problems == []


def test_shift_enter_makes_a_new_line_and_examples_do_not_send(page):
    b = page.browser
    sign_in(page)
    b.type("#question", "first")
    b.press_enter(shift=True)
    assert b.js("document.querySelector('#question').value") == "first\n"
    b.js("document.querySelector('.example').click()")
    assert b.js("document.querySelector('#question').value.length") > 10
    assert page.client.requests == [] and b.js("document.querySelectorAll('.turn').length") == 0


@pytest.mark.parametrize("width", [360, 768, 1280])
def test_nothing_overflows_sideways(page, width):
    b = page.browser
    b.resize(width, 800)
    sign_in(page)
    ask(page, QUESTION)
    b.wait("document.querySelector('figure img')?.complete")
    assert b.js("document.documentElement.scrollWidth") <= width
