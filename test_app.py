from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import tiktoken
from pytest import MonkeyPatch

import app


def test_local_environment_is_loaded(monkeypatch: MonkeyPatch) -> None:
    loader = Mock(side_effect=lambda **kwargs: monkeypatch.setenv("OPENAI_API_KEY", "sk-test-local"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app, "load_dotenv", loader)
    monkeypatch.setattr(app.st, "secrets", {})
    factory = Mock()
    monkeypatch.setattr(app.openai, "OpenAI", factory)

    assert app.init_openai_client.__wrapped__() is factory.return_value
    loader.assert_called_once_with(
        dotenv_path=Path(app.__file__).resolve().parent / ".env", override=False,
    )
    factory.assert_called_once_with(api_key="sk-test-local")


def test_streamlit_key_takes_precedence(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app, "load_dotenv", Mock())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-environment")
    monkeypatch.setattr(app.st, "secrets", {"OPENAI_API_KEY": "sk-test-cloud"})
    factory = Mock()
    monkeypatch.setattr(app.openai, "OpenAI", factory)

    app.init_openai_client.__wrapped__()
    factory.assert_called_once_with(api_key="sk-test-cloud")


def test_short_rule_is_preserved() -> None:
    rule = "100. General\n" + "A player may take an action. " * 10
    assert app.build_rule_chunks(rule) == [rule.strip()]


def test_long_rule_is_split_without_losing_content() -> None:
    rule = "702. Abilities\n" + "702.123a {T}: Add {G}. " * 5000
    chunks = app.build_rule_chunks(rule)
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    assert len(chunks) > 1
    assert "".join(chunks) == rule.strip()
    assert all(len(encoding.encode(chunk)) <= 8191 for chunk in chunks)


def test_embedding_batches_and_vector_alignment(monkeypatch: MonkeyPatch) -> None:
    chunks = [f"Rule {i}" for i in range(65)]
    monkeypatch.setattr(app, "fetch_rules_text", lambda: ("rules", "url"))
    monkeypatch.setattr(app, "build_rule_chunks", lambda text: chunks)
    client = Mock()

    def embed(*, model: str, input: list[str]) -> SimpleNamespace:
        assert model == "text-embedding-3-small"
        assert len(input) <= 32
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float(int(text.split()[1]) + 1), 1.0])
            for i, text in reversed(list(enumerate(input)))
        ])

    client.embeddings.create.side_effect = embed
    monkeypatch.setattr(app.openai, "OpenAI", lambda **kwargs: client)
    result_chunks, vectors = app.build_embeddings.__wrapped__("hash", "test-key")
    assert result_chunks == chunks
    assert client.embeddings.create.call_count == 3
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6)
    np.testing.assert_allclose(vectors[:, 0] / vectors[:, 1], np.arange(1, 66), rtol=1e-6)


def test_empty_rules_do_not_call_openai(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app, "fetch_rules_text", lambda: ("   ", "url"))
    client_factory = Mock()
    monkeypatch.setattr(app.openai, "OpenAI", client_factory)
    assert app.build_embeddings.__wrapped__("empty", "test-key") == (None, None)
    client_factory.assert_not_called()
