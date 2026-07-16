# tests/test_agent_integration.py
"""
E2E agent-пайплайн: init → add (реальные Docling + USER-bge-m3 + postgres) → ask.

Сам LLM-вызов замокан через _create_and_run_agent (семантика прежняя — LM Studio
для этого теста НЕ нужен). Хранилище — реальная тест-БД docling_rag_test через
фикстуру e2e_config (conftest.py), осознанно переопределяющую hermetic_config.
Требует: docker compose up -d postgres; .[agent] (pydantic-ai) установлен.
"""
import pytest
from unittest.mock import patch

from docling_rag.cli import main

pytestmark = pytest.mark.integration


def test_ask_end_to_end_with_mocked_llm(runner, e2e_config, tmp_path):
    """E2E: init → add .md → ask with mocked LLM response."""
    # 1. Init (идемпотентный DDL против тест-БД)
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0

    # 2. Create test document
    doc = tmp_path / "test_doc.md"
    doc.write_text("# Architecture\n\nData Vault uses hubs, links, and satellites.\n")

    # 3. Add document (uses real Docling + embedder + postgres)
    result = runner.invoke(main, [
        "add", str(doc),
        "--title", "Test Architecture",
        "--topic", "data vault",
    ])
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()

    # 4. Ask with enabled agent and mocked LLM response
    ask_cfg = dict(e2e_config, agent_enabled=True)
    with (
        patch("docling_rag.cli.commands.load_config", return_value=ask_cfg),
        patch(
            "docling_rag.cli.commands._create_and_run_agent",
            return_value="Data Vault is a modeling methodology using hubs, links, and satellites.",
        ),
    ):
        result = runner.invoke(main, ["ask", "What is Data Vault?"])

    assert result.exit_code == 0
    assert "Data Vault" in result.output
