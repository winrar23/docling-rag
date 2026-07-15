import pytest
from click.testing import CliRunner

_HERMETIC_DEFAULTS = {
    # embedding_model остаётся all-MiniLM-L6-v2 (не деф. deepvk/USER-bge-m3 — 2.3 ГБ):
    # CLI unit-тесты мокают Embedder, но герметичный дефолт не должен указывать на тяжёлую модель.
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_results": 5,
    "data_dir": "data",
    "log_file": "",  # заполняется per-test из tmp_path
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
    # порт 1 — заведомо несоединяемый: юнит-тест, случайно дошедший до реального
    # соединения с БД, падает быстро и громко вместо зависания/долгого таймаута.
    "database_url": "postgresql://test:test@127.0.0.1:1/test",
    "chunk_max_tokens": 512,
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def hermetic_config(tmp_path, monkeypatch):
    """CLI unit tests must never read the repo's live config.yaml or write repo logs/."""
    cfg = dict(_HERMETIC_DEFAULTS)
    cfg["log_file"] = str(tmp_path / "logs" / "search.log")
    monkeypatch.setattr("docling_rag.cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    return cfg
