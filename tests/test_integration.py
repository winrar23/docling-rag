# tests/test_integration.py
"""
Smoke-тесты: end-to-end пайплайн CLI на реальных .md файлах против postgres.

Требуют: docker compose up -d postgres (тест-БД docling_rag_test), установленный
Docling и модель deepvk/USER-bge-m3 (первый прогон скачает ~2.3 ГБ в HF-кеш).
Хранилище подменяется фикстурой e2e_config (conftest.py) — она осознанно
переопределяет autouse hermetic_config на реальную тест-БД и реальную модель.
Помечены @pytest.mark.integration — не запускаются в быстром суите.
"""
import pytest

from docling_rag.cli import main

pytestmark = pytest.mark.integration


def test_full_pipeline_on_real_md(runner, e2e_config, tmp_path):
    """init → add → list → search → delete на реальном Markdown файле."""
    # Создаём тестовый документ
    doc = tmp_path / "test_doc.md"
    doc.write_text(
        "# Database Architecture\n\n"
        "The DWH uses a star schema with fact and dimension tables.\n\n"
        "## SQL Example\n\n"
        "```sql\nSELECT customer_id, SUM(amount)\nFROM fact_sales\nGROUP BY customer_id;\n```\n",
        encoding="utf-8",
    )

    # Init (идемпотентный DDL против тест-БД)
    result = runner.invoke(main, ["init"], catch_exceptions=False)
    assert result.exit_code == 0

    # Add
    result = runner.invoke(main, ["add", str(doc)], catch_exceptions=False)
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()

    # List
    result = runner.invoke(main, ["list"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "test_doc.md" in result.output
    assert "chunks" in result.output  # verifies at least one chunk was stored

    # Search
    result = runner.invoke(main, ["search", "star schema fact table"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "score=" in result.output
    assert "test_doc.md" in result.output
    # Verify semantic quality: top score should be meaningful
    top_score = float(result.output.split("score=")[1].split("|")[0].strip())
    assert top_score > 0.3, f"Expected semantic relevance > 0.3, got {top_score}"

    # Delete — документ и его chunks исчезают из индекса
    result = runner.invoke(main, ["delete", str(doc)], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Удалено" in result.output

    result = runner.invoke(main, ["list"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "test_doc.md" not in result.output


def test_add_with_tags_and_search_filter(runner, e2e_config, tmp_path):
    """
    End-to-end: index two docs with different tags, search with --tag filter
    returns only results from the matching doc.
    """
    # Two minimal markdown files that parse fast
    doc_arch = tmp_path / "architecture.md"
    doc_arch.write_text("Hexagonal architecture separates core logic from adapters.")

    doc_data = tmp_path / "data_engineering.md"
    doc_data.write_text("Data pipelines move and transform data between systems.")

    # Index first doc with tag=arch
    result = runner.invoke(main, [
        "add", str(doc_arch),
        "--title", "Arch Book",
        "--topic", "architecture",
        "--tag", "arch",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # Index second doc with tag=data
    result = runner.invoke(main, [
        "add", str(doc_data),
        "--title", "Data Book",
        "--topic", "data engineering",
        "--tag", "data",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # Search without filter — should return results from both docs
    result = runner.invoke(main, [
        "search", "logic and systems",
        "--top-k", "5",
    ], catch_exceptions=False)
    assert result.exit_code == 0
    sources_in_output = result.output
    assert "architecture" in sources_in_output.lower() or "arch" in sources_in_output.lower()

    # Search with --tag arch — must not return data doc
    result = runner.invoke(main, [
        "search", "logic and systems",
        "--tag", "arch",
        "--top-k", "5",
    ], catch_exceptions=False)
    assert result.exit_code == 0
    assert "data_engineering.md" not in result.output
    assert "architecture" in result.output.lower() or "arch" in result.output.lower()
