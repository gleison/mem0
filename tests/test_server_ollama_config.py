import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")


def _reload_server(env: dict[str, str]):
    memory = MagicMock()
    with patch.dict(os.environ, env, clear=False), patch("mem0.Memory.from_config", return_value=memory) as factory:
        server_main = importlib.import_module("server.main")
        importlib.reload(server_main)
    return server_main, factory


def test_ollama_is_a_bundled_provider():
    server_main, _ = _reload_server({"AUTH_DISABLED": "true"})

    assert "ollama" in server_main.BUNDLED_LLM_PROVIDERS
    assert "ollama" in server_main.BUNDLED_EMBEDDER_PROVIDERS


def test_ollama_environment_builds_qwen_default_config():
    server_main, factory = _reload_server(
        {
            "AUTH_DISABLED": "true",
            "MEM0_DEFAULT_LLM_PROVIDER": "ollama",
            "MEM0_DEFAULT_LLM_MODEL": "qwen3:4b-instruct-2507-q8_0",
            "MEM0_DEFAULT_LLM_MAX_TOKENS": "0",
            "MEM0_DEFAULT_EMBEDDER_PROVIDER": "ollama",
            "MEM0_DEFAULT_EMBEDDER_MODEL": "qwen3-embedding:0.6b-q8_0",
            "MEM0_DEFAULT_EMBEDDING_DIMS": "1024",
            "OLLAMA_BASE_URL": "http://ollama:11434",
        }
    )

    config = factory.call_args.args[0]

    assert server_main.DEFAULT_CONFIG == config
    assert config["llm"] == {
        "provider": "ollama",
        "config": {
            "model": "qwen3:4b-instruct-2507-q8_0",
            "temperature": 0.2,
            "ollama_base_url": "http://ollama:11434",
            "max_tokens": 0,
        },
    }
    assert config["embedder"] == {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:0.6b-q8_0",
            "embedding_dims": 1024,
            "ollama_base_url": "http://ollama:11434",
        },
    }
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024
