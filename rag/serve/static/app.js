// The chat page. Everything from the server is set as text, never as HTML.
"use strict";

(function () {
  const POLL_MS = 700;
  const STORED = "ask-z21-conversation";
  const VOICES = {
    thiel: ["T", "Thiel's words"],
    analyst: ["A", "Analyst's notes"],
    evidence: ["E", "Evidence"],
    unknown: ["?", "Unknown source"],
  };

  const $ = (id) => document.getElementById(id);
  let me = null;
  let conversation = null;
  let busy = false;
  let epoch = 0;                          // raised on sign-out or a new conversation, to drop stale polls

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function api(path, body) {
    const options = { credentials: "same-origin", cache: "no-store" };
    if (body !== undefined) {
      options.method = "POST";
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (cause) {
      const error = new Error("Can't reach the server. Check that it's running.");
      error.status = 0;
      throw error;
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(data.error || "The request failed. Try again.");
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function store(id) {
    try {
      if (id) sessionStorage.setItem(STORED, id);
      else sessionStorage.removeItem(STORED);
    } catch (e) { /* storage is off: a reload starts a new conversation */ }
  }

  function stored() {
    try { return sessionStorage.getItem(STORED); } catch (e) { return null; }
  }

  function announce(text) {
    $("announcer").textContent = text;
  }

  // ---------------------------------------------------------------- screens
  function setBusy(value) {
    busy = value;
    $("ask").disabled = value;
    $("new-conversation").disabled = value;      // leaving mid-answer would lose an answer already paid for
  }

  function forget() {
    epoch += 1;
    conversation = null;
    store(null);
    setBusy(false);
    $("record").replaceChildren();
    $("form-error").textContent = "";
  }

  function showRelease(info) {
    $("unevaluated").hidden = info.evaluated;
    $("release").textContent = "Release " + info.release_id + ".";
  }

  function showSignIn(message) {
    forget();
    me = null;
    $("question").value = "";
    for (const id of ["account", "start", "composer", "unevaluated"]) $(id).hidden = true;
    $("sign-in").hidden = false;
    if (message) $("sign-in-message").textContent = message;
  }

  function showChat() {
    $("sign-in").hidden = true;
    $("account").hidden = false;
    $("composer").hidden = false;
    $("user").textContent = me.user;
    showRelease(conversation || me);
    $("start").hidden = Boolean(conversation && conversation.turns.length);
  }

  // ---------------------------------------------------------------- an answer
  function inlineNodes(parent, nodes, context) {
    for (const node of nodes) {
      if (node.type === "text") parent.append(node.text);
      else if (node.type === "code") parent.append(el("code", "", node.text));
      else if (node.type === "strong" || node.type === "em") {
        const child = el(node.type);
        inlineNodes(child, node.children, context);
        parent.append(child);
      } else if (node.type === "ref") {
        const source = context.sources.get(node.n);
        if (!source) continue;
        const cite = el("button", "cite", String(node.n));
        cite.type = "button";
        cite.dataset.voice = source.voice;
        cite.title = source.title;
        const voice = (VOICES[source.voice] || VOICES.unknown)[1];
        cite.setAttribute("aria-label", "Source " + node.n + ", " + voice + ": " + source.title);
        cite.addEventListener("click", () => context.open(node.n));
        parent.append(cite);
      }
    }
  }

  function linesInto(parent, lines, context) {
    lines.forEach((line, i) => {
      if (i) parent.append(el("br"));
      inlineNodes(parent, line, context);
    });
  }

  function answerBody(segments, context) {
    const box = el("div", "answer");
    for (const block of AnswerMarkdown.parse(segments)) {
      if (block.type === "p") {
        const p = el("p");
        linesInto(p, block.lines, context);
        box.append(p);
      } else if (block.type === "h") {
        const h = el("h3");
        inlineNodes(h, block.children, context);
        box.append(h);
      } else if (block.type === "list") {
        const list = el(block.ordered ? "ol" : "ul");
        if (block.ordered) list.start = block.start;
        for (const item of block.items) {
          const li = el("li", item.depth ? "depth-" + item.depth : "");
          linesInto(li, item.lines, context);
          list.append(li);
        }
        box.append(list);
      } else if (block.type === "table") {
        const scroll = el("div", "table-scroll");
        scroll.tabIndex = 0;                 // so a wide table can be scrolled by keyboard
        const table = el("table");
        const head = table.createTHead().insertRow();
        for (const cell of block.head) {
          const th = el("th");
          inlineNodes(th, cell, context);
          head.append(th);
        }
        const body = table.createTBody();
        for (const row of block.rows) {
          const tr = body.insertRow();
          for (const cell of row) {
            const td = tr.insertCell();
            const text = cell.filter((n) => n.type === "text").map((n) => n.text).join("").trim();
            if (/^[−+-]?\$?\d[\d,.]*\s*(%|x|×|bn|m|k)?$/.test(text)) td.className = "numeric";
            inlineNodes(td, cell, context);
          }
        }
        scroll.append(table);
        box.append(scroll);
      }
    }
    return box;
  }

  function warningBox(answer) {
    const box = el("div", "warning");
    box.setAttribute("role", "note");
    box.append(el("p", "", "This answer failed the automatic check. Confirm these points before you use it:"));
    const list = el("ul");
    for (const failed of answer.failed_checks) {
      const item = el("li", "", failed.check);
      item.append(el("span", "detail", failed.detail));
      list.append(item);
    }
    box.append(list);
    return box;
  }

  function sourceList(sources, entries) {
    const list = el("ol", "sources");
    list.setAttribute("aria-label", "Sources");
    for (const source of sources) {
      const [letter, name] = VOICES[source.voice] || VOICES.unknown;
      const item = el("li", "source");
      const details = el("details");
      const summary = el("summary");
      const mark = el("span", "voice", letter);
      mark.dataset.voice = source.voice;
      mark.title = name;
      summary.append(el("span", "number", String(source.n)), mark, el("span", "title", source.title));
      details.append(summary);
      for (const excerpt of source.excerpts) details.append(el("blockquote", "", excerpt));
      item.append(details);
      list.append(item);
      entries.set(source.n, { item, details, summary });
    }
    return list;
  }

  function renderTurn(article, turn, conversationId) {
    const body = el("div", "turn-body");
    const margin = el("aside", "turn-margin");
    body.append(el("h2", "question", turn.question));
    article.dataset.state = turn.state;

    if (turn.state === "working") {
      const progress = el("div", "progress");
      const steps = el("ol");
      for (const step of turn.steps) steps.append(el("li", "", step));
      progress.append(steps);
      body.append(progress);
    } else if (turn.state === "error") {
      const box = el("div", "warning");
      box.append(el("p", "", turn.error));
      body.append(box);
    } else {
      const answer = turn.answer;
      const entries = new Map();
      const context = {
        sources: new Map(answer.sources.map((s) => [s.n, s])),
        open(n) {
          for (const entry of entries.values()) entry.item.classList.remove("selected");
          const entry = entries.get(n);
          entry.item.classList.add("selected");
          entry.details.open = true;
          entry.item.scrollIntoView({ block: "nearest" });
          entry.summary.focus({ preventScroll: true });
        },
      };
      if (answer.notice) {
        const box = el("div", "warning");
        box.append(el("p", "", answer.notice));
        body.append(box);
      } else if (answer.flagged) {
        body.append(warningBox(answer));
      }
      body.append(answerBody(answer.segments, context));
      for (const figure of answer.figures) {
        const box = el("figure");
        const image = el("img");
        image.src = "/api/conversations/" + conversationId + "/figures/" + encodeURIComponent(figure.name) + ".png";
        image.alt = figure.caption;
        box.append(image, el("figcaption", "", figure.caption));
        body.append(box);
      }
      if (answer.sources.length) margin.append(sourceList(answer.sources, entries));
      else margin.append(el("p", "logged", "No sources cited."));
      const status = answer.flagged || answer.notice
        ? "Failed the check"
        : answer.revised ? "Passed the check after one correction" : "Passed the check";
      margin.append(el("p", "logged", status + " in " + answer.seconds + " seconds. Log record " + answer.record_id));
    }
    article.replaceChildren(body, margin);
  }

  function addTurn(turn) {
    const article = el("article", "turn");
    $("record").append(article);
    renderTurn(article, turn, conversation.id);
    return article;
  }

  // ---------------------------------------------------------------- asking
  async function follow(turn, article) {
    const started = epoch;
    const id = conversation.id;
    setBusy(true);
    let steps = turn.steps.length;
    let latest = turn;
    while (latest.state === "working") {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      if (epoch !== started) return;
      try {
        latest = await api("/api/conversations/" + id + "/questions/" + turn.index);
      } catch (error) {
        if (epoch !== started) return;
        if (error.status === 0 || error.status >= 500) {
          $("form-error").textContent = error.message;
          continue;
        }
        if (error.status === 401) return showSignIn(error.message);
        latest = { index: turn.index, question: turn.question, state: "error", error: error.message };
      }
      if (epoch !== started) return;
      $("form-error").textContent = "";
      if (latest.state !== "working" || latest.steps.length !== steps) {
        renderTurn(article, latest, id);
        if (latest.state === "working") {
          steps = latest.steps.length;
          announce(latest.steps[steps - 1]);
        }
      }
    }
    setBusy(false);
    announce(latest.state === "done" ? "The answer is ready." : latest.error);
    article.querySelector(".question").scrollIntoView({ block: "nearest" });
  }

  async function ask(event) {
    event.preventDefault();
    if (busy) return;
    const box = $("question");
    const question = box.value.trim();
    if (!question) {
      $("form-error").textContent = "Type a question first.";
      return;
    }
    $("form-error").textContent = "";
    setBusy(true);
    const started = epoch;
    try {
      if (!conversation) {
        const created = await api("/api/conversations", {});
        if (epoch !== started) return;
        conversation = created;
        store(conversation.id);
        showRelease(conversation);
      }
      const turn = await api("/api/conversations/" + conversation.id + "/questions", { question });
      if (epoch !== started) return;
      box.value = "";
      $("start").hidden = true;
      const article = addTurn(turn);
      article.scrollIntoView({ block: "start" });
      await follow(turn, article);
    } catch (error) {
      if (epoch !== started) return;
      setBusy(false);
      if (error.status === 401) return showSignIn(error.message);
      if (error.status === 404) {
        forget();
        showChat();
      }
      $("form-error").textContent = error.message;
    }
  }

  async function newConversation() {
    forget();
    try {
      me = await api("/api/me");             // a new conversation takes the release being served now
      showChat();
      $("question").focus();
    } catch (error) {
      showSignIn(error.message);
    }
  }

  async function signOut() {
    try { await api("/api/sign-out", {}); } catch (error) { /* already signed out */ }
    showSignIn("You're signed out. Open your sign-in link to come back.");
  }

  async function boot() {
    let message;
    const code = new URLSearchParams(location.hash.slice(1)).get("code");
    if (code) {
      history.replaceState(null, "", location.pathname);       // keep the code out of the history
      forget();
      try {
        await api("/api/sign-in", { code });
      } catch (error) {
        return showSignIn(error.message);
      }
    }
    try {
      me = await api("/api/me");
    } catch (error) {
      if (error.status !== 401) message = error.message;
      return showSignIn(message);
    }
    const id = stored();
    if (id && /^[0-9a-f]{32}$/.test(id)) {
      try {
        conversation = await api("/api/conversations/" + id);
      } catch (error) {
        store(null);
      }
    }
    showChat();
    if (conversation) {
      const articles = conversation.turns.map(addTurn);
      const last = conversation.turns[conversation.turns.length - 1];
      if (last && last.state === "working") await follow(last, articles[articles.length - 1]);
    }
    $("question").focus();
  }

  $("composer").addEventListener("submit", ask);
  $("new-conversation").addEventListener("click", newConversation);
  $("sign-out").addEventListener("click", signOut);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  for (const button of document.querySelectorAll(".example")) {
    button.addEventListener("click", () => {
      $("question").value = button.textContent;       // fills the box: asking costs money, so it doesn't send
      $("question").focus();
    });
  }
  window.addEventListener("hashchange", () => {
    if (location.hash.includes("code=")) boot();
  });
  boot();
})();
