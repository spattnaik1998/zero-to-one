"""Who is asking (the name the audit log records).

Until single sign-on (#10): the operator names the users when starting the server, and each gets a
random sign-in code that works until it stops. A user can't choose or change their name. Only
hashes are held, and nothing is written to disk.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time

SIGNED_IN_S = 12 * 3600
USER_NAME = re.compile(r"[\w.@-]{1,64}")


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class Auth:
    def __init__(self, users, clock=time.time):
        self.clock = clock
        self._lock = threading.Lock()
        self._codes: dict[str, str] = {}                      # code hash -> user
        self._tokens: dict[str, tuple[str, float]] = {}       # token hash -> (user, expires)
        self.codes: dict[str, str] = {}                       # user -> code, for the operator to hand out
        for user in dict.fromkeys(users):
            if not USER_NAME.fullmatch(user):
                raise ValueError(f"{user!r} is not a user name (letters, digits and . _ @ -, up to 64)")
            self.codes[user] = secrets.token_urlsafe(24)
            self._codes[_hash(self.codes[user])] = user
        if not self.codes:
            raise ValueError("name at least one user")

    def sign_in(self, code) -> tuple[str, str] | None:
        """(session token, user), or None for a wrong code."""
        if not isinstance(code, str) or not 0 < len(code) <= 200:
            return None
        user = self._codes.get(_hash(code))
        if user is None:
            return None
        token, now = secrets.token_urlsafe(32), self.clock()
        with self._lock:
            self._tokens = {k: v for k, v in self._tokens.items() if v[1] > now}
            self._tokens[_hash(token)] = (user, now + SIGNED_IN_S)
        return token, user

    def user(self, token: str | None) -> str | None:
        if not token:
            return None
        with self._lock:
            found = self._tokens.get(_hash(token))
        return found[0] if found and found[1] > self.clock() else None

    def sign_out(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._tokens.pop(_hash(token), None)
