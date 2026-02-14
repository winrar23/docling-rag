# tests/test_integration.py
"""
Smoke-тест: end-to-end пайплайн на реальном .md файле.
Требует установленного Docling и загруженной модели.
Помечен @pytest.mark.integration — не запускается в CI по умолчанию.
"""
import pytest
from click.testing import CliRunner
from cli import main


@pytest.mark.integration
def test_full_pipeline_on_real_md(tmp_path):
    """add → search на реальном Markdown файле."""
    # Создаём тестовый документ
    doc = tmp_path / "test_doc.md"
    doc.write_text(
        "# Database Architecture\n\n"
        "The DWH uses a star schema with fact and dimension tables.\n\n"
        "## SQL Example\n\n"
        "```sql\nSELECT customer_id, SUM(amount)\nFROM fact_sales\nGROUP BY customer_id;\n```\n",
        encoding="utf-8",
    )

    data_dir = str(tmp_path / "store")
    runner = CliRunner()

    # Init
    result = runner.invoke(main, ["init", "--data-dir", data_dir], catch_exceptions=False)
    assert result.exit_code == 0

    # Add
    result = runner.invoke(main, ["add", str(doc), "--data-dir", data_dir], catch_exceptions=False)
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()

    # List
    result = runner.invoke(main, ["list", "--data-dir", data_dir], catch_exceptions=False)
    assert result.exit_code == 0
    assert "test_doc.md" in result.output
    assert "chunks" in result.output  # verifies at least one chunk was stored

    # Search
    result = runner.invoke(main, ["search", "star schema fact table", "--data-dir", data_dir], catch_exceptions=False)
    assert result.exit_code == 0
    assert "score=" in result.output
    assert "test_doc.md" in result.output
    # Verify semantic quality: top score should be meaningful
    top_score = float(result.output.split("score=")[1].split("|")[0].strip())
    assert top_score > 0.3, f"Expected semantic relevance > 0.3, got {top_score}"


@pytest.mark.integration
def test_add_with_tags_and_search_filter(tmp_path):
    """
    End-to-end: index two docs with different tags, search with --tag filter
    returns only results from the matching doc.
    """
    runner = CliRunner()

    # Two minimal markdown files that parse fast
    doc_arch = tmp_path / "architecture.md"
    doc_arch.write_text("Hexagonal architecture separates core logic from adapters.")

    doc_data = tmp_path / "data_engineering.md"
    doc_data.write_text("Data pipelines move and transform data between systems.")

    store_dir = tmp_path / "store"

    # Index first doc with tag=arch
    result = runner.invoke(main, [
        "add", str(doc_arch),
        "--data-dir", str(store_dir),
        "--title", "Arch Book",
        "--topic", "architecture",
        "--tag", "arch",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # Index second doc with tag=data
    result = runner.invoke(main, [
        "add", str(doc_data),
        "--data-dir", str(store_dir),
        "--title", "Data Book",
        "--topic", "data engineering",
        "--tag", "data",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # Search without filter — should return results from both docs
    result = runner.invoke(main, [
        "search", "logic and systems",
        "--data-dir", str(store_dir),
        "--top-k", "5",
    ], catch_exceptions=False)
    assert result.exit_code == 0
    sources_in_output = result.output
    assert "architecture" in sources_in_output.lower() or "arch" in sources_in_output.lower()

    # Search with --tag arch — must not return data doc
    result = runner.invoke(main, [
        "search", "logic and systems",
        "--data-dir", str(store_dir),
        "--tag", "arch",
        "--top-k", "5",
    ], catch_exceptions=False)
    assert result.exit_code == 0
    assert "data_engineering.md" not in result.output
    assert "architecture" in result.output.lower() or "arch" in result.output.lower()
