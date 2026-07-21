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
    storage.append([chunk], np.ones((1, 4), dtype=np.float32) / 2.0)
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


def test_agent_deps_new_fields_default():
    """Обратная совместимость: старый 4-арговый конструктор работает, новые поля дефолтны."""
    deps = AgentDeps(embedder=MagicMock(), storage=MagicMock(), registry=MagicMock(), top_k=5)
    assert deps.search_log is None
    assert deps.sources == []


def test_agent_deps_sources_not_shared_between_instances():
    """default_factory: аккумулятор не разделяется между разными deps (классическая ловушка mutable default)."""
    a = AgentDeps(embedder=MagicMock(), storage=MagicMock(), registry=MagicMock(), top_k=5)
    b = AgentDeps(embedder=MagicMock(), storage=MagicMock(), registry=MagicMock(), top_k=5)
    a.sources.append("x")
    assert b.sources == []


def test_tool_accumulates_sources():
    """Tool складывает сырые (meta, score) в deps.sources — основа поля sources ответа /chat."""
    deps = _seeded_deps()
    create_agent(TestModel()).run_sync("What is Data Vault?", deps=deps)
    assert len(deps.sources) >= 1
    meta, score = deps.sources[0]
    assert meta["source_file"] == "dv.pdf"
    assert meta["page_number"] == 42
    assert isinstance(float(score), float)


def test_tool_logs_search_query_and_top_score():
    """Закрытие TODO п.5: агентский поиск пишется в search_log (query из tool-вызова, score первого результата)."""
    from tests.fakes import InMemorySearchLog
    deps = _seeded_deps()
    deps.search_log = InMemorySearchLog()
    create_agent(TestModel()).run_sync("What is Data Vault?", deps=deps)
    assert len(deps.search_log.entries) == 1
    query, top_score = deps.search_log.entries[0]
    assert isinstance(query, str) and query
    assert top_score == pytest.approx(float(deps.sources[0][1]))


def test_tool_does_not_log_empty_results():
    """Контракт как у CLI search: пустая выдача — записи в лог нет."""
    from tests.fakes import InMemorySearchLog

    class _EmptySearchStorage(InMemoryStorage):
        def search(self, query_embedding, top_k, allowed_sources=None):
            return []

    deps = _seeded_deps()
    storage = _EmptySearchStorage()
    chunk = Chunk(text="t", source_file="dv.pdf", chunk_id=0, page_number=1,
                  element_type="text", headings=[], context_text="t")
    storage.append([chunk], np.ones((1, 4), dtype=np.float32))
    deps.storage = storage
    deps.search_log = InMemorySearchLog()
    create_agent(TestModel()).run_sync("q", deps=deps)
    assert deps.search_log.entries == []
    assert deps.sources == []


def test_tool_log_failure_does_not_crash_run(capsys):
    """Отказ лога (БД лежит) не роняет run — предупреждение в stderr."""

    class _BoomLog:
        def log(self, query, top_score):
            raise RuntimeError("db down")

    deps = _seeded_deps()
    deps.search_log = _BoomLog()
    result = create_agent(TestModel()).run_sync("What is Data Vault?", deps=deps)
    assert isinstance(result.output, str)
    assert "лог поиска не записан" in capsys.readouterr().err


def test_static_instructions_survive_message_history():
    """КРИТИЧНО: при непустом message_history pydantic-ai НЕ отправляет system_prompt=,
    но отправляет instructions= — SYSTEM_PROMPT обязан доходить до модели в чате с историей."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    hist = [ModelRequest(parts=[UserPromptPart(content="прошлый вопрос")]),
            ModelResponse(parts=[TextPart(content="прошлый ответ")])]
    result = create_agent(TestModel()).run_sync("новый вопрос", deps=_seeded_deps(), message_history=hist)
    reqs = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
    instr = next((m.instructions for m in reqs if getattr(m, "instructions", None)), "") or ""
    assert "search_documents" in instr  # маркер SYSTEM_PROMPT
    assert "DV Book" in instr           # динамический список документов тоже на месте


def test_tool_accumulates_sources_across_multiple_calls():
    """Spec §6: sources накапливаются при нескольких tool-вызовах, дубли не схлопываются."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    calls = {"n": 0}

    def scripted(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={"query": "hubs"})])
        if calls["n"] == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={"query": "satellites"})])
        return ModelResponse(parts=[TextPart(content="готово")])

    deps = _seeded_deps()
    create_agent(FunctionModel(scripted)).run_sync("вопрос", deps=deps)
    assert len(deps.sources) == 2  # top_k=3, но в сеяном хранилище 1 chunk → по 1 результату на вызов
    assert deps.sources[0][0]["source_file"] == "dv.pdf"
    assert deps.sources[1][0]["source_file"] == "dv.pdf"  # дубль не схлопнут


def test_build_lmstudio_model_passes_timeout_to_http_client():
    """timeout_sec доезжает до httpx-клиента провайдера (отсечка зависшего LM Studio)."""
    from unittest.mock import patch as _patch
    with _patch("docling_rag.core.agent.OpenAIProvider") as MockProvider, \
         _patch("docling_rag.core.agent.OpenAIChatModel"):
        build_lmstudio_model("m", "http://u", "k", timeout_sec=7.0)
    http_client = MockProvider.call_args.kwargs["http_client"]
    assert http_client.timeout.read == 7.0
