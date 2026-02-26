import pytest
from cli.config_loader import load_config


def test_defaults_include_agent_keys(tmp_path):
    """load_config with no config file returns agent defaults."""
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["agent_enabled"] is False
    assert cfg["llm_base_url"] == "http://127.0.0.1:1234/v1"
    assert cfg["llm_api_key"] == "lm-studio"
    assert cfg["llm_model"] == "local-model"
    assert cfg["agent_top_k"] == 5


def test_defaults_preserve_existing_keys(tmp_path):
    """load_config with no config file returns agent defaults."""
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["embedding_model"] == "all-MiniLM-L6-v2"
    assert cfg["top_k_results"] == 5
    assert cfg["data_dir"] == "data"
    assert cfg["log_file"] == "logs/search.log"


def test_user_config_overrides_agent_defaults(tmp_path):
    """User config.yaml overrides agent defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("agent_enabled: true\nllm_model: my-model\n")
    cfg = load_config(config_file)
    assert cfg["agent_enabled"] is True
    assert cfg["llm_model"] == "my-model"
    # Non-overridden defaults still present
    assert cfg["llm_base_url"] == "http://127.0.0.1:1234/v1"
