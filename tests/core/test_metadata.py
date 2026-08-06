"""Юниты core/metadata.py. pydantic-ai TestModel — без сети и без LM Studio."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")  # extra .[agent]; в dev-окружении установлен


def _chunk(text):
    return SimpleNamespace(text=text)


def test_metadata_module_does_not_import_pydantic_ai():
    """Контракт: модуль обязан импортироваться без extra .[agent] (lazy-импорты внутри функций).

    Проверка в отдельном процессе: другие тесты этого файла уже импортировали
    pydantic_ai в текущий sys.modules, а вычищать его оттуда руками нельзя —
    это ломает internal registry/singledispatch pydantic-ai и роняет соседние
    тесты (test_agent.py), запущенные в том же pytest-процессе после этого.
    """
    code = "import sys, docling_rag.core.metadata; assert 'pydantic_ai' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_build_snippet_respects_max_chars():
    from docling_rag.core.metadata import build_snippet

    chunks = [_chunk("a" * 5000), _chunk("b" * 5000), _chunk("c" * 5000)]
    snippet = build_snippet(chunks, max_chars=8000)
    assert len(snippet) <= 8001  # + разделитель
    assert snippet.startswith("a")
    assert "c" not in snippet  # третий чанк не влез


def test_extract_metadata_returns_structured_output():
    from pydantic_ai.models.test import TestModel

    from docling_rag.core.metadata import extract_metadata

    model = TestModel(custom_output_args={
        "title": "Чистая архитектура", "author": "Роберт Мартин",
        "topic": "проектирование", "tags": ["архитектура", "solid"],
    })
    meta = extract_metadata("ЧИСТАЯ АРХИТЕКТУРА...", ["проектирование"], ["solid"], model)
    assert meta.title == "Чистая архитектура"
    assert meta.author == "Роберт Мартин"
    assert meta.tags == ["архитектура", "solid"]


def test_build_prompt_contains_snippet_and_vocab():
    """Промпт (user message) обязан содержать фрагмент текста и словарь тем/тегов."""
    from docling_rag.core.metadata import _build_prompt

    prompt = _build_prompt("ТИТУЛЬНЫЙ ФРАГМЕНТ", ["базы данных"], ["postgres"])
    assert "ТИТУЛЬНЫЙ ФРАГМЕНТ" in prompt
    assert "базы данных" in prompt
    assert "postgres" in prompt


def test_clean_normalizes_output():
    from docling_rag.core.metadata import DocMeta, _clean

    meta = _clean(DocMeta(title="  Книга ", author="", topic=" Базы Данных ",
                          tags=[" Postgres", "postgres", "", "a", "b", "c", "d", "e"]))
    assert meta.title == "Книга"
    assert meta.author is None            # пустая строка -> None
    assert meta.topic == "базы данных"    # lowercase
    assert meta.tags == ["postgres", "a", "b", "c", "d"]  # lowercase, дедуп, максимум 5


def test_get_metadata_extractor_disabled_returns_none():
    from docling_rag.core.metadata import get_metadata_extractor

    assert get_metadata_extractor({"auto_metadata": False}, registry=None) is None


def test_get_metadata_extractor_enabled_returns_callable():
    from docling_rag.core.metadata import get_metadata_extractor

    cfg = {"auto_metadata": True, "llm_model": "m", "llm_base_url": "http://x",
           "llm_api_key": "k", "llm_timeout_sec": 5}
    extractor = get_metadata_extractor(cfg, registry=None)
    assert callable(extractor)
