import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from click.testing import CliRunner

from cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.integration
def test_ask_end_to_end_with_mocked_llm(runner, tmp_path):
    """E2E: init → add .md → ask with mocked LLM response."""
    # 1. Init
    result = runner.invoke(main, ["init", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0

    # 2. Create test document
    doc = tmp_path / "test_doc.md"
    doc.write_text("# Architecture\n\nData Vault uses hubs, links, and satellites.\n")

    # 3. Add document (uses real Docling + embedder)
    result = runner.invoke(main, [
        "add", str(doc),
        "--data-dir", str(tmp_path),
        "--title", "Test Architecture",
        "--topic", "data vault",
    ])
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()

    # 4. Ask with enabled agent and mocked LLM response
    with (
        patch("cli.commands.load_config", return_value={
            "agent_enabled": True,
            "llm_base_url": "http://127.0.0.1:1234/v1",
            "llm_api_key": "lm-studio",
            "llm_model": "local-model",
            "agent_top_k": 5,
            "embedding_model": "all-MiniLM-L6-v2",
        }),
        patch("cli.commands._create_and_run_agent", return_value="Data Vault is a modeling methodology using hubs, links, and satellites."),
    ):
        result = runner.invoke(main, [
            "ask", "What is Data Vault?",
            "--data-dir", str(tmp_path),
        ])

    assert result.exit_code == 0
    assert "Data Vault" in result.output
