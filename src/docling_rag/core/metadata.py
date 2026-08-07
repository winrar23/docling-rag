"""Авто-извлечение метаданных документа (title/author/topic/tags) LLM'ом.

Один structured-вызов pydantic-ai по тексту первых чанков + словарь существующих
тем/тегов из registry как подсказка. Встроенные метаданные файла (PDF Info и т.п.)
НЕ используются — обычно мусор (решение спеки 2026-08-06).

Импорты pydantic_ai/core.agent — только lazy: модуль обязан импортироваться без .[agent].
"""
from __future__ import annotations

from typing import Callable, Sequence

from pydantic import BaseModel


class DocMeta(BaseModel):
    title: str | None = None
    author: str | None = None
    topic: str | None = None
    tags: list[str] = []


SNIPPET_MAX_CHARS = 8000

_INSTRUCTIONS = (
    "Ты извлекаешь библиографические метаданные книги из начала её текста.\n"
    "Верни:\n"
    "- title: название книги как в оригинале; null, если названия в тексте нет.\n"
    "- author: автор(ы) как в книге; несколько — одной строкой через запятую; "
    "null, если не указаны.\n"
    "- topic: одна короткая тема на русском, в нижнем регистре.\n"
    "- tags: 1-5 коротких тегов в нижнем регистре; язык — как у существующих тегов, "
    "если существующих нет — на русском.\n"
    "Если в списке существующих тем/тегов есть подходящие — используй ИХ; "
    "новые придумывай только когда ничего не подошло.\n"
    "Не выдумывай: бери только то, что подтверждается текстом."
)


def build_snippet(chunks: Sequence, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Первые чанки документа (титул/аннотация/оглавление) до лимита символов."""
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = chunk.text
        if total + len(text) > max_chars and parts:
            break
        parts.append(text[: max_chars - total])
        total += len(parts[-1])
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def _build_prompt(snippet: str, known_topics: Sequence[str], known_tags: Sequence[str]) -> str:
    topics = ", ".join(known_topics) if known_topics else "(пока нет)"
    tags = ", ".join(known_tags) if known_tags else "(пока нет)"
    return (
        f"Существующие темы: {topics}\n"
        f"Существующие теги: {tags}\n\n"
        f"Начало текста книги:\n---\n{snippet}\n---"
    )


_NULL_STRINGS = {"null", "none"}  # LLM пишет их строкой вместо JSON null (qwen3.6, 2026-08-07)


def _clean(meta: DocMeta) -> DocMeta:
    """Нормализация: strip, пустые/«null»-строки -> None, теги lowercase/уникальные, максимум 5."""
    def norm(value: str | None) -> str | None:
        value = (value or "").strip()
        if value.lower() in _NULL_STRINGS:
            return None
        return value or None

    tags: list[str] = []
    for tag in meta.tags:
        tag = tag.strip().lower()
        if tag and tag not in _NULL_STRINGS and tag not in tags:
            tags.append(tag)
    topic = norm(meta.topic)
    return DocMeta(title=norm(meta.title), author=norm(meta.author),
                   topic=topic.lower() if topic else None, tags=tags[:5])


def extract_metadata(snippet: str, known_topics: Sequence[str],
                     known_tags: Sequence[str], model) -> DocMeta:
    """Один structured-вызов LLM. Исключения не глотает — fail-soft у вызывающего."""
    from pydantic_ai import Agent  # lazy: extra .[agent]

    agent: Agent[None, DocMeta] = Agent(model, output_type=DocMeta,
                                        instructions=_INSTRUCTIONS)
    result = agent.run_sync(_build_prompt(snippet, known_topics, known_tags))
    return _clean(result.output)


def get_metadata_extractor(cfg: dict, registry) -> Callable[[Sequence], DocMeta] | None:
    """Фактори экстрактора по конфигу (паттерн get_embedder). None = шаг выключен.

    Closure импортирует agent-модуль лениво: ImportError/сетевые ошибки всплывают
    при ВЫЗОВЕ и попадают в fail-soft индексатора, не ломая CLI/worker на старте.
    """
    if not cfg.get("auto_metadata", True):
        return None

    def _extract(chunks: Sequence) -> DocMeta:
        from docling_rag.core.agent import build_lmstudio_model  # lazy: extra .[agent]

        model = build_lmstudio_model(
            cfg["llm_model"], cfg["llm_base_url"], cfg["llm_api_key"],
            timeout_sec=float(cfg.get("llm_timeout_sec", 120)),
        )
        docs = registry.load().values()
        topics = sorted({d["topic"] for d in docs if d.get("topic")})
        tags = sorted({t for d in docs for t in d.get("tags", [])})
        return extract_metadata(build_snippet(chunks), topics, tags, model)

    return _extract
