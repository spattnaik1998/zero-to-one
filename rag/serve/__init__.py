"""The chat: API and web page for pilot users (spec sections 2 band C, 7 F6-F7).

Built:
  auth.py    named users, each with a sign-in code for this run of the server; the signed-in
             name is what the audit log records. Single sign-on replaces this in #10.
  chat.py    conversations, each owned by one user and pinned to one release; one question at a
             time per user; every answer through ask_audited (checked, then logged, then shown);
             progress as steps only, never draft text.
  view.py    an answer as the page shows it: cited segments, sources by voice, charts, and the
             checks it failed.
  app.py     the HTTP API and page (tornado), on this computer only: host and origin checks,
             same-site cookie, strict content security policy.
  static/    the page: plain HTML, CSS and JavaScript, no external assets, no HTML from answers.

`python -m rag serve --user NAME` starts it.
"""
