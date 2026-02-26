from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.embedder import Embedder
from core.search import run_search
from storage.doc_registry import DocRegistry
from storage.file_storage import FileStorage


@dataclass
class AgentDeps:
    embedder: Embedder
    storage: FileStorage
    registry: DocRegistry
    top_k: int


SYSTEM_PROMPT = (
    "You are a technical documentation assistant. "
    "You have access to a search tool that finds relevant chunks from indexed documents. "
    "Rules:\n"
    "1. ALWAYS use the search_documents tool to find information before answering.\n"
    "2. Answer ONLY based on the search results. If no relevant information is found, say so.\n"
    "3. Cite sources: mention the file name and page number for each fact.\n"
    "4. Respond in the same language as the user's question.\n"
    "5. Be concise and precise."
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


def _build_doc_list(registry: DocRegistry) -> str:
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


def create_agent(model_name: str, base_url: str, api_key: str) -> Agent:
    """Create pydantic-ai Agent with search tool for RAG."""
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        output_type=str,
        system_prompt=SYSTEM_PROMPT,
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
        return format_search_results(results)

    return agent
