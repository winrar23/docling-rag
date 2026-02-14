# tests/test_integration.py
"""
Smoke-тест: end-to-end пайплайн на реальном .md файле.
Требует установленного Docling и загруженной модели.
Пометить @pytest.mark.integration — не запускать в CI по умолчанию.
"""
import pytest
from pathlib import Path
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
    result = runner.invoke(main, ["init", "--data-dir", data_dir])
    assert result.exit_code == 0

    # Add
    result = runner.invoke(main, ["add", str(doc), "--data-dir", data_dir])
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()

    # Search
    result = runner.invoke(main, ["search", "star schema fact table", "--data-dir", data_dir])
    assert result.exit_code == 0
    assert "score=" in result.output
    assert "test_doc.md" in result.output
