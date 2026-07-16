import numpy as np
import pytest
from unittest.mock import MagicMock

pytest.importorskip("pydantic_ai")
from pydantic_ai.models.test import TestModel

from docling_rag.core.agent import AgentDeps, create_agent, build_lmstudio_model
from docling_rag.core.chunker import Chunk
from tests.fakes import InMemoryRegistry, InMemoryStorage


class FakeEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0


def _seeded_deps() -> AgentDeps:
    storage = InMemoryStorage()
    registry = InMemoryRegistry()
    chunk = Chunk(text="Data Vault uses hubs and satellites.", source_file="dv.pdf",
                  chunk_id=0, page_number=42, element_type="text",
                  headings=["Ch 2"], context_text="ctx")
    storage.save([chunk], np.ones((1, 4), dtype=np.float32) / 2.0)
    registry.upsert("dv.pdf", title="DV Book", topic="dwh", tags=["arch"])
    return AgentDeps(embedder=FakeEmbedder(), storage=storage, registry=registry, top_k=3)


def test_format_search_results_with_results():
    """format_search_results formats chunks for LLM consumption."""
    from docling_rag.core.agent import format_search_results

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
    from docling_rag.core.agent import format_search_results

    output = format_search_results([])
    assert len(output) > 0  # Should return a message, not empty string


def test_build_doc_list_with_documents():
    """_build_doc_list formats registry entries for system prompt."""
    from docling_rag.core.agent import _build_doc_list

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
    from docling_rag.core.agent import _build_doc_list

    mock_registry = MagicMock()
    mock_registry.load.return_value = {}
    output = _build_doc_list(mock_registry)

    assert len(output) > 0  # Should return a message, not empty string


def test_agent_deps_dataclass():
    """AgentDeps can be created with required fields."""
    from docling_rag.core.agent import AgentDeps

    deps = AgentDeps(
        embedder=MagicMock(),
        storage=MagicMock(),
        registry=MagicMock(),
        top_k=5,
    )
    assert deps.top_k == 5
    assert deps.embedder is not None


def test_create_agent_returns_agent():
    """create_agent accepts any pydantic-ai Model (composable) and returns an Agent instance."""
    agent = create_agent(TestModel())
    # pydantic-ai Agent has run_sync method
    assert hasattr(agent, "run_sync")


def test_system_prompt_contains_tool_info():
    """SYSTEM_PROMPT explicitly lists available tools to prevent LLM hallucination."""
    from docling_rag.core.agent import SYSTEM_PROMPT

    assert "Available tools:" in SYSTEM_PROMPT
    assert "search_documents" in SYSTEM_PROMPT
    assert "query: str" in SYSTEM_PROMPT


def test_agent_tool_executes_real_search():
    """TestModel calls every tool once — search_documents must run against real storage."""
    agent = create_agent(TestModel())
    result = agent.run_sync("What is Data Vault?", deps=_seeded_deps())
    assert isinstance(result.output, str)
    messages = result.all_messages()
    tool_returns = [p.content for m in messages for p in m.parts
                    if getattr(p, "part_kind", "") == "tool-return"]
    assert any("dv.pdf" in str(c) and "p.42" in str(c) for c in tool_returns)


def test_dynamic_instructions_list_documents():
    agent = create_agent(TestModel())
    result = agent.run_sync("hi", deps=_seeded_deps())
    req = result.all_messages()[0]
    instructions = getattr(req, "instructions", "") or ""
    assert "DV Book" in instructions


def test_build_lmstudio_model_targets_base_url():
    model = build_lmstudio_model("m", "http://127.0.0.1:1234/v1", "key")
    assert type(model).__name__ == "OpenAIChatModel"
