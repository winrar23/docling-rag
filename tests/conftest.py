import pytest
from click.testing import CliRunner

_HERMETIC_DEFAULTS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_results": 5,
    "data_dir": "data",
    "log_file": "",  # заполняется per-test из tmp_path
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def hermetic_config(tmp_path, monkeypatch):
    """CLI unit tests must never read the repo's live config.yaml or write repo logs/."""
    cfg = dict(_HERMETIC_DEFAULTS)
    cfg["log_file"] = str(tmp_path / "logs" / "search.log")
    monkeypatch.setattr("cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    return cfg
