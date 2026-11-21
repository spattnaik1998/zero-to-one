"""Settings for the Ask Zero to One build.

Values come from the process environment first, then from the nearest `.env`
found by walking up from the repo root. That is the same lookup `thiel_common`
uses, so keep every key (SEC, Census, FRED, Anthropic, OpenAI) in that one file:
a second, closer `.env` hides the first from both modules.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = ("anthropic", "bedrock", "vertex", "foundry")
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def find_dotenv(start: Path = ROOT) -> Path | None:
    """Return the nearest `.env` at or above `start`, or None."""
    for d in [start, *start.parents]:
        f = d / ".env"
        if f.is_file():
            return f
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=value lines. Quotes are stripped; an unquoted value loses a trailing ` # comment`."""
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        env[key] = value
    return env


def read_dotenv(start: Path = ROOT) -> dict[str, str]:
    f = find_dotenv(start)
    return parse_dotenv(f.read_text(encoding="utf-8", errors="ignore")) if f else {}


@dataclass(frozen=True)
class Settings:
    claude_provider: str            # anthropic | bedrock | vertex | foundry
    claude_model: str
    anthropic_api_key: str | None   # needed only when claude_provider == "anthropic"
    openai_api_key: str | None
    embedding_model: str | None     # pinned: query vectors must match the released index
    sources_dir: Path
    build_dir: Path
    audit_dir: Path                 # append-only answer log; keep it under the client's retention policy


WATCHED = ("CLAUDE_PROVIDER", "CLAUDE_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL")


def env_conflicts(environ: Mapping[str, str] | None = None,
                  dotenv: Mapping[str, str] | None = None) -> list[str]:
    """Settings set in both the process environment and .env with different values.
    The environment wins, so a stale system variable silently hides a corrected .env."""
    environ = os.environ if environ is None else environ
    dotenv = read_dotenv() if dotenv is None else dotenv
    return [k for k in WATCHED
            if (environ.get(k) or "").strip() and (dotenv.get(k) or "").strip()
            and environ[k].strip() != dotenv[k].strip()]


def load_settings(environ: Mapping[str, str] | None = None,
                  dotenv: Mapping[str, str] | None = None) -> Settings:
    """Build Settings. Process environment overrides `.env`."""
    env = {**(read_dotenv() if dotenv is None else dotenv), **(os.environ if environ is None else environ)}
    get = lambda k: (env.get(k) or "").strip() or None  # noqa: E731

    provider = (get("CLAUDE_PROVIDER") or "anthropic").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"CLAUDE_PROVIDER={provider!r} is not one of {', '.join(PROVIDERS)}")

    return Settings(
        claude_provider=provider,
        claude_model=get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        openai_api_key=get("OPENAI_API_KEY"),
        embedding_model=get("OPENAI_EMBEDDING_MODEL"),
        sources_dir=ROOT / (get("RAG_SOURCES_DIR") or "sources"),
        build_dir=ROOT / (get("RAG_BUILD_DIR") or "build"),
        audit_dir=ROOT / (get("RAG_AUDIT_DIR") or str(Path(get("RAG_BUILD_DIR") or "build") / "audit")),
    )
