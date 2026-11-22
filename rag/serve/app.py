"""The chat's HTTP API and page.

    GET  /                                        the page
    POST /api/sign-in {code}                      sets the session cookie
    POST /api/sign-out
    GET  /api/me                                  user, release being served, whether it passed evaluation
    POST /api/conversations                       a conversation pinned to that release
    GET  /api/conversations/ID                    its questions and answers
    POST /api/conversations/ID/questions {question}   202: started
    GET  /api/conversations/ID/questions/N        steps while working, the answer once checked and logged
    GET  /api/conversations/ID/figures/NAME.png   a chart from that conversation's release

It listens on 127.0.0.1 only: there is no single sign-on or TLS until #10, so there is no option
to expose it. The Host check stops DNS rebinding; writes must be JSON and same-origin.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import tornado.httpserver
import tornado.netutil
import tornado.web

from .chat import ChatError

log = logging.getLogger(__name__)

ADDRESS = "127.0.0.1"
MAX_BODY = 32 * 1024
STATIC = Path(__file__).resolve().parent / "static"
ASSETS = {"index.html": "text/html", "app.css": "text/css", "app.js": "text/javascript",
          "markdown.js": "text/javascript"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


class Base(tornado.web.RequestHandler):
    needs_user = True

    def set_default_headers(self):
        self.set_header("Content-Security-Policy", CSP)
        self.set_header("X-Content-Type-Options", "nosniff")
        self.set_header("X-Frame-Options", "DENY")
        self.set_header("Referrer-Policy", "no-referrer")
        self.set_header("Cross-Origin-Resource-Policy", "same-origin")
        self.set_header("Cache-Control", "no-store")
        self.clear_header("Server")

    @property
    def chat(self):
        return self.settings["chat"]

    @property
    def auth(self):
        return self.settings["auth"]

    @property
    def cookie_name(self) -> str:
        return f"ask_z21_{self.settings['port']}"             # cookies ignore ports, so the name carries it

    def prepare(self):
        port = self.settings["port"]
        hosts = (f"{ADDRESS}:{port}", f"localhost:{port}")
        if self.request.host not in hosts:
            raise ChatError(403, "Open the chat at the address the server printed.")
        if self.request.method == "POST":
            origin = self.request.headers.get("Origin")
            if origin is not None and origin != f"http://{self.request.host}":
                raise ChatError(403, "Requests from other sites are not accepted.")
            if self.request.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                raise ChatError(415, "Send JSON.")
        self.user = self.auth.user(self.get_cookie(self.cookie_name))
        if self.needs_user and self.user is None:
            raise ChatError(401, "Sign in with the link the operator gave you.")

    def body(self) -> dict:
        try:
            value = json.loads(self.request.body or b"{}")
        except (ValueError, RecursionError):
            raise ChatError(400, "Send valid JSON.") from None
        if not isinstance(value, dict):
            raise ChatError(400, "Send a JSON object.")
        return value

    def send(self, value: dict, status: int = 200) -> None:
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.write(json.dumps(value, ensure_ascii=False, default=str))

    def log_exception(self, typ, value, tb):
        if not isinstance(value, ChatError):
            super().log_exception(typ, value, tb)

    def write_error(self, status_code: int, **kwargs):
        error = kwargs.get("exc_info", (None, None, None))[1]
        if isinstance(error, ChatError):
            self.send({"error": error.message}, error.status)
        else:                                                 # never a traceback or a path
            self.send({"error": "The server couldn't handle that request."}, status_code)
        self.finish()


class Asset(Base):
    needs_user = False

    def get(self, name: str = ""):
        name = name or "index.html"
        if name not in ASSETS:
            raise ChatError(404, "Not found.")
        self.set_header("Content-Type", f"{ASSETS[name]}; charset=utf-8")
        self.write((STATIC / name).read_bytes())


class Missing(Base):
    needs_user = False

    def prepare(self):
        raise ChatError(404, "Not found.")


class SignIn(Base):
    needs_user = False

    def post(self):
        signed_in = self.auth.sign_in(self.body().get("code"))
        if signed_in is None:
            raise ChatError(401, "That sign-in link doesn't work. It expires when the server restarts.")
        token, user = signed_in
        self.set_cookie(self.cookie_name, token, httponly=True, samesite="Strict", path="/")
        self.send({"user": user})


class SignOut(Base):
    def post(self):
        self.auth.sign_out(self.get_cookie(self.cookie_name))
        self.clear_cookie(self.cookie_name, httponly=True, samesite="Strict", path="/")
        self.send({})


class Me(Base):
    async def get(self):
        status = await asyncio.to_thread(self.chat.status)
        self.send({"user": self.user, **status})


class Conversations(Base):
    async def post(self):
        self.send(await asyncio.to_thread(self.chat.create, self.user), 201)


class OneConversation(Base):
    def get(self, conversation_id: str):
        self.send(self.chat.get(self.user, conversation_id))


class Questions(Base):
    def post(self, conversation_id: str):
        self.send(self.chat.ask(self.user, conversation_id, self.body().get("question")), 202)


class OneQuestion(Base):
    def get(self, conversation_id: str, index: str):
        self.send(self.chat.turn(self.user, conversation_id, int(index)))


class Figure(Base):
    def get(self, conversation_id: str, name: str):
        image = self.chat.figure(self.user, conversation_id, name)
        self.set_header("Content-Type", "image/png")
        self.write(image)


def make_app(chat, auth, port: int) -> tornado.web.Application:
    one = r"/api/conversations/([0-9a-f]{32})"
    return tornado.web.Application([
        (r"/", Asset),
        (r"/static/([a-z]+\.[a-z]+)", Asset),
        (r"/api/sign-in", SignIn),
        (r"/api/sign-out", SignOut),
        (r"/api/me", Me),
        (r"/api/conversations", Conversations),
        (one, OneConversation),
        (one + r"/questions", Questions),
        (one + r"/questions/([0-9]{1,3})", OneQuestion),
        (one + r"/figures/([a-z0-9_]+)\.png", Figure),
    ], chat=chat, auth=auth, port=port, default_handler_class=Missing, debug=False, serve_traceback=False)


def listen(chat, auth, port: int) -> tuple[tornado.httpserver.HTTPServer, int]:
    """Start listening on this computer. Returns the server and its port (port 0 picks a free one)."""
    sockets = tornado.netutil.bind_sockets(port, ADDRESS)
    port = sockets[0].getsockname()[1]
    server = tornado.httpserver.HTTPServer(make_app(chat, auth, port), max_body_size=MAX_BODY,
                                           max_buffer_size=4 * MAX_BODY)
    server.add_sockets(sockets)
    return server, port


def run(settings, users: list[str], port: int) -> int:
    """`python -m rag serve`."""
    from ..agent.audit import AuditLog
    from .auth import Auth
    from .chat import Chat

    chat = Chat(settings, AuditLog(settings.audit_dir))
    try:
        auth = Auth(users)
        status = chat.status()                                # fail now, not on the first question
    except ChatError as e:
        print(f"cannot start: {e.message}")
        chat.close()
        return 1
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as e:
        print(f"cannot start: {e}")
        chat.close()
        return 1

    async def serve_forever():
        server, actual = listen(chat, auth, port)
        print(f"Serving release {status['release_id']} on http://{ADDRESS}:{actual} (this computer only)")
        if not status["evaluated"]:
            print("NOT EVALUATED: this release has not passed the full evaluation with this code. "
                  "Internal testing only; don't give pilot users access.")
        for user, code in auth.codes.items():
            print(f"  sign-in link for {user}: http://{ADDRESS}:{actual}/#code={code}")
        print("Ctrl+C stops it.", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            server.stop()

    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        print("Stopping; waiting for questions in progress to be logged.")
    except OSError as e:
        print(f"cannot listen on port {port}: {e.strerror or e}")
        return 1
    finally:
        chat.close()
    return 0
