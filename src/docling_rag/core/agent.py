from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from docling_rag.core.embedder import Embedder
from docling_rag.core.protocols import DocumentRegistryBackend, SearchLogBackend, StorageBackend
from docling_rag.core.search import run_search


@dataclass
class AgentDeps:
    embedder: Embedder
    storage: StorageBackend
    registry: DocumentRegistryBackend
    top_k: int
    # Заполняются механикой /chat и ask; дефолты сохраняют старый 4-арговый конструктор.
    search_log: SearchLogBackend | None = None
    sources: list = field(default_factory=list)  # (meta, score) из tool-вызовов за run


SYSTEM_PROMPT = (
    """

    You are a helpful assistant with access to a knowledge base that you can search when needed 
    You have access to a search tool that finds relevant chunks from indexed documents.

    Available tools:
    - search_documents(query: str): searches indexed documentation by semantic similarity.

    When to Search:
    - ONLY search when users explicitly ask for information that would be in the knowledge base
    - For greetings (hi, hello, hey) → Just respond conversationally, no search needed
    - For general questions about yourself → Answer directly, no search needed
    - For requests about specific topics or information → Use the search_documents tool

    Rules:
    1. Answer ONLY based on the search results. If no relevant information is found, say so.
    2. CRITICAL! Cite sources: mention the file name and page number for each fact.
    3. CRITICAL! Respond in the same language as the user's question.
    4. Be concise and precise.
    
    # Remember: Not every interaction requires a search. Use your judgment about when to search the knowledge base.

    """ 
)


def format_search_results(results: list[tuple[dict, float]]) -> str:
    """Format search results as text for LLM consumption."""
    if not results:
        return "No relevant documents found."

    parts = []
    for i, (meta, score) in enumerate(results, 1):
        source = Path(meta["source_file"]).name
        page = meta.get("page_number", "?")
        headings = meta.get("headings", [])
        heading_str = " > ".join(headings) if headings else ""
        text = meta["text"]

        header = f"[{i}] {source} (p.{page}, score={score:.2f})"
        if heading_str:
            header += f" [{heading_str}]"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def _build_doc_list(registry: DocumentRegistryBackend) -> str:
    """Format indexed documents list for dynamic system prompt."""
    doc_index = registry.load()
    if not doc_index:
        return "No documents are currently indexed."

    lines = ["Available documents:"]
    for source, entry in doc_index.items():
        title = entry.get("title") or Path(source).name
        topic = entry.get("topic") or ""
        topic_str = f" ({topic})" if topic else ""
        lines.append(f"- {title}{topic_str}")

    return "\n".join(lines)


def build_lmstudio_model(model_name: str, base_url: str, api_key: str,
                         timeout_sec: float = 120.0) -> OpenAIChatModel:
    """LM Studio speaks Chat Completions — keep the explicit OpenAIChatModel (v2: 'openai:' prefix means Responses API).

    timeout_sec — отсечка ожидания LLM (httpx), чтобы зависший LM Studio не держал запрос вечно.
    """
    provider = OpenAIProvider(base_url=base_url, api_key=api_key,
                              http_client=httpx.AsyncClient(timeout=timeout_sec))
    return OpenAIChatModel(model_name, provider=provider)


def create_agent(model) -> Agent:
    """Create pydantic-ai Agent with search tool for RAG. Accepts any Model (incl. TestModel)."""
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        output_type=str,
        # instructions, НЕ system_prompt: при непустом message_history pydantic-ai
        # не отправляет system_prompt, а instructions отправляет каждый run —
        # иначе чат с историей работал бы без RAG-правил.
        instructions=SYSTEM_PROMPT,
    )

    @agent.instructions
    def dynamic_instructions(ctx: RunContext[AgentDeps]) -> str:
        return _build_doc_list(ctx.deps.registry)

    @agent.tool
    async def search_documents(ctx: RunContext[AgentDeps], query: str) -> str:
        """Search indexed documentation by semantic similarity query. Returns relevant text chunks with source references."""
        results = run_search(
            query,
            ctx.deps.embedder,
            ctx.deps.storage,
            ctx.deps.top_k,
        )
        ctx.deps.sources.extend(results)
        if results and ctx.deps.search_log is not None:
            try:
                ctx.deps.search_log.log(query, float(results[0][1]))
            except Exception as e:  # отказ лога не роняет run — контракт как у CLI search
                print(f"предупреждение: лог поиска не записан: {e}", file=sys.stderr)
        return format_search_results(results)

    return agent
