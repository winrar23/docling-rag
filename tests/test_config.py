import pytest

from docling_rag.cli.config_loader import load_config, ConfigError


def test_defaults_include_agent_keys(tmp_path):
    """load_config with no config file returns agent defaults."""
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["agent_enabled"] is False
    assert cfg["llm_base_url"] == "http://127.0.0.1:1234/v1"
    assert cfg["llm_api_key"] == "lm-studio"
    assert cfg["llm_model"] == "local-model"
    assert cfg["agent_top_k"] == 5


def test_defaults_preserve_existing_keys(tmp_path):
    """load_config with no config file preserves existing non-agent defaults."""
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["embedding_model"] == "deepvk/USER-bge-m3"
    assert cfg["top_k_results"] == 5
    assert cfg["chunk_max_tokens"] == 512


def test_user_config_overrides_agent_defaults(tmp_path):
    """User config.yaml overrides agent defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("agent_enabled: true\nllm_model: my-model\n")
    cfg = load_config(config_file)
    assert cfg["agent_enabled"] is True
    assert cfg["llm_model"] == "my-model"
    # Non-overridden defaults still present
    assert cfg["llm_base_url"] == "http://127.0.0.1:1234/v1"


def test_explicit_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_config(tmp_path / "nope.yaml", required=True)


def test_default_missing_config_falls_back(tmp_path):
    cfg = load_config(tmp_path / "config.yaml", required=False)
    assert cfg["top_k_results"] == 5


def test_non_dict_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="словар"):
        load_config(p, required=True)


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("key: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(p, required=True)


def test_unknown_keys_warn(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("top_k_result: 10\n")  # опечатка
    load_config(p, required=True)
    assert "top_k_result" in capsys.readouterr().err


def test_database_url_env_overrides_config(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("database_url: postgresql://cfg:cfg@cfg:5432/cfg\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "postgresql://env:env@env:5432/env")
    assert load_config(p, required=True)["database_url"] == "postgresql://env:env@env:5432/env"


def test_database_url_defaults_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert "127.0.0.1:5432/docling_rag" in load_config(tmp_path / "nope.yaml")["database_url"]


def test_defaults_include_llm_timeout():
    from docling_rag.cli.config_loader import _DEFAULTS
    assert _DEFAULTS["llm_timeout_sec"] == 120
