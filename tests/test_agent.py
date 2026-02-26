import pytest
from unittest.mock import MagicMock


def test_format_search_results_with_results():
    """format_search_results formats chunks for LLM consumption."""
    from core.agent import format_search_results

    results = [
        ({"text": "Data Vault uses hubs and links.", "source_file": "dv.pdf",
          "page_number": 5, "element_type": "text", "headings": ["Ch1", "Hubs"]}, 0.92),
        ({"text": "Satellites store descriptive data.", "source_file": "dv.pdf",
          "page_number": 8, "element_type": "text", "headings": ["Ch2"]}, 0.85),
    ]
    output = format_search_results(results)

    assert "Data Vault uses hubs and links." in output
    assert "Satellites store descriptive data." in output
    assert "dv.pdf" in output
    assert "0.92" in output or "92" in output


def test_format_search_results_empty():
    """format_search_results returns informative message for empty results."""
    from core.agent import format_search_results

    output = format_search_results([])
    assert len(output) > 0  # Should return a message, not empty string


def test_build_doc_list_with_documents():
    """_build_doc_list formats registry entries for system prompt."""
    from core.agent import _build_doc_list

    mock_registry = MagicMock()
    mock_registry.load.return_value = {
        "books/arch.pdf": {"title": "Clean Architecture", "topic": "software", "tags": ["arch"], "added_at": "2026-01-01"},
        "notes/dv.md": {"title": None, "topic": "data vault", "tags": [], "added_at": "2026-01-02"},
    }
    output = _build_doc_list(mock_registry)

    assert "Clean Architecture" in output
    assert "software" in output
    assert "dv.md" in output  # fallback to filename when title is None


def test_build_doc_list_empty():
    """_build_doc_list handles empty registry."""
    from core.agent import _build_doc_list

    mock_registry = MagicMock()
    mock_registry.load.return_value = {}
    output = _build_doc_list(mock_registry)

    assert len(output) > 0  # Should return a message, not empty string


def test_agent_deps_dataclass():
    """AgentDeps can be created with required fields."""
    from core.agent import AgentDeps

    deps = AgentDeps(
        embedder=MagicMock(),
        storage=MagicMock(),
        registry=MagicMock(),
        top_k=5,
    )
    assert deps.top_k == 5
    assert deps.embedder is not None


def test_create_agent_returns_agent():
    """create_agent returns a pydantic-ai Agent instance."""
    from core.agent import create_agent

    agent = create_agent(
        model_name="test-model",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
    )
    # pydantic-ai Agent has run_sync method
    assert hasattr(agent, "run_sync")
