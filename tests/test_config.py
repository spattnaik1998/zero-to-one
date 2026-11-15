import pytest

from rag.config import DEFAULT_CLAUDE_MODEL, find_dotenv, load_settings, parse_dotenv


def test_parse_dotenv_handles_quotes_comments_and_export():
    env = parse_dotenv(
        "# comment\n"
        "export OPENAI_API_KEY=sk-test\n"
        'SEC_USER_AGENT="Jane Doe jane@example.com"\n'
        "CLAUDE_PROVIDER=bedrock   # trailing comment\n"
        "not a line\n"
    )
    assert env == {
        "OPENAI_API_KEY": "sk-test",
        "SEC_USER_AGENT": "Jane Doe jane@example.com",
        "CLAUDE_PROVIDER": "bedrock",
    }


def test_defaults():
    s = load_settings(environ={}, dotenv={})
    assert s.claude_provider == "anthropic"
    assert s.claude_model == DEFAULT_CLAUDE_MODEL
    assert s.embedding_model is None
    assert s.book_licensed is False


def test_environment_overrides_dotenv():
    s = load_settings(environ={"OPENAI_EMBEDDING_MODEL": "from-env"},
                      dotenv={"OPENAI_EMBEDDING_MODEL": "from-file", "RAG_BOOK_LICENSED": "true"})
    assert s.embedding_model == "from-env"
    assert s.book_licensed is True


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        load_settings(environ={"CLAUDE_PROVIDER": "somewhere"}, dotenv={})


def test_find_dotenv_walks_up(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"
