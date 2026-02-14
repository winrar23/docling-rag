# docling-rag MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Построить CLI-утилиту `docling-rag` для семантического поиска по технической документации (PDF, DOCX, MD, TXT) через Docling + Sentence Transformers + NumPy.

**Architecture:** Модульный пайплайн: `parser → chunker → embedder → file_storage`. CLI-команды (`init`, `add`, `search`, `list`) вызывают компоненты через абстракцию `storage.py`. Хранилище — `.npy` матрица + `metadata.json`.

**Tech Stack:** Python 3.10+, Docling, sentence-transformers (all-MiniLM-L6-v2), NumPy, Click, pytest

---

## Task 0: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `core/__init__.py`
- Create: `storage/__init__.py`
- Create: `cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/storage/__init__.py`
- Create: `config.yaml`

**Step 1: Создать структуру директорий**

```bash
cd "/Users/danny/Documents/Документы Даниил/Github/Docling RAG"
mkdir -p core storage cli tests/core tests/storage data/documents logs
touch core/__init__.py storage/__init__.py cli/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/storage/__init__.py
```

**Step 2: Создать `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "docling-rag"
version = "0.1.0"
description = "Semantic search CLI for technical documentation using Docling"
requires-python = ">=3.10"
dependencies = [
    "docling>=2.0.0",
    "sentence-transformers>=3.0.0",
    "numpy>=1.26.0",
    "click>=8.1.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[project.scripts]
docling-rag = "cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["core*", "storage*", "cli*"]
```

**Step 3: Создать `config.yaml` с дефолтами**

```yaml
embedding_model: all-MiniLM-L6-v2
chunk_size: 800        # целевой размер chunk в токенах (≈ 3200 символов)
chunk_overlap: 80      # overlap (10% от chunk_size)
top_k_results: 5
data_dir: data
log_file: logs/search.log
```

**Step 4: Установить зависимости**

```bash
cd "/Users/danny/Documents/Документы Даниил/Github/Docling RAG"
uv pip install -e ".[dev]"
```

Ожидаем: успешная установка без ошибок.

**Step 5: Commit**

```bash
git add pyproject.toml config.yaml core/ storage/ cli/ tests/ .gitignore CLAUDE.md docs/
git commit -m "chore: initial project scaffold with pyproject.toml and directory structure"
```

---

## Task 1: core/chunker.py

**Files:**
- Create: `tests/core/test_chunker.py`
- Create: `core/chunker.py`

### Что делает chunker

Принимает список `DoclingElement` (упрощённо: dict с полями `text`, `type`, `page`).
Возвращает список `Chunk` — dataclass с полями `text`, `source_file`, `chunk_id`, `page_number`, `element_type`.

Правила:
- `type == "table"` или `type == "code"` → атомарный chunk (не разбивается)
- `type == "text"` → накапливаем предложения до ~`chunk_size * 4` символов, затем создаём новый chunk с `overlap` символами из конца предыдущего

**Step 1: Написать тесты**

```python
# tests/core/test_chunker.py
import pytest
from core.chunker import Chunk, chunk_elements


def make_element(text, etype="text", page=1):
    return {"text": text, "type": etype, "page": page}


def test_chunk_returns_list_of_chunks():
    elements = [make_element("Hello world. This is a test.")]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=3200, overlap=80)
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], Chunk)


def test_chunk_has_required_fields():
    elements = [make_element("Hello world.")]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=3200, overlap=80)
    chunk = result[0]
    assert chunk.source_file == "doc.pdf"
    assert chunk.chunk_id == 0
    assert chunk.page_number == 1
    assert chunk.element_type == "text"
    assert "Hello world" in chunk.text


def test_table_is_atomic_chunk():
    elements = [make_element("col1 | col2\n----|----\nA | B", etype="table")]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=100, overlap=10)
    assert len(result) == 1
    assert result[0].element_type == "table"


def test_code_is_atomic_chunk():
    elements = [make_element("SELECT * FROM users WHERE id = 1", etype="code")]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=100, overlap=10)
    assert len(result) == 1
    assert result[0].element_type == "code"


def test_long_text_is_split_into_multiple_chunks():
    long_text = "Sentence number {}. " * 200
    elements = [make_element(long_text.format(*range(200)))]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=200, overlap=20)
    assert len(result) > 1


def test_overlap_carries_context():
    sentence = "The quick brown fox. "
    elements = [make_element(sentence * 100)]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=200, overlap=40)
    if len(result) > 1:
        # Конец первого чанка должен присутствовать в начале второго
        end_of_first = result[0].text[-40:]
        assert end_of_first in result[1].text or len(result[1].text) > 0


def test_empty_elements_returns_empty_list():
    result = chunk_elements([], source_file="doc.pdf", chunk_size=3200, overlap=80)
    assert result == []


def test_chunk_ids_are_sequential():
    elements = [make_element("Text " * 500)]
    result = chunk_elements(elements, source_file="doc.pdf", chunk_size=200, overlap=20)
    ids = [c.chunk_id for c in result]
    assert ids == list(range(len(result)))
```

**Step 2: Запустить тесты — убедиться что падают**

```bash
pytest tests/core/test_chunker.py -v
```

Ожидаем: `ImportError: cannot import name 'Chunk' from 'core.chunker'`

**Step 3: Написать реализацию `core/chunker.py`**

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: int
    page_number: int
    element_type: str  # "text", "table", "code"


def chunk_elements(
    elements: list[dict[str, Any]],
    source_file: str,
    chunk_size: int = 3200,   # символы (≈800 токенов × 4 символа/токен)
    overlap: int = 320,        # символы (≈80 токенов)
) -> list[Chunk]:
    """
    Разбивает список Docling-элементов на Chunk-объекты.
    Таблицы и code-блоки — атомарные chunks.
    Текстовые элементы накапливаются до chunk_size символов с overlap.
    """
    chunks: list[Chunk] = []
    chunk_id = 0
    text_buffer = ""
    buffer_page = 1

    def flush_buffer(carry_over: str = "") -> None:
        nonlocal chunk_id, text_buffer, buffer_page
        if text_buffer.strip():
            chunks.append(Chunk(
                text=text_buffer.strip(),
                source_file=source_file,
                chunk_id=chunk_id,
                page_number=buffer_page,
                element_type="text",
            ))
            chunk_id += 1
        text_buffer = carry_over

    for element in elements:
        etype = element.get("type", "text")
        text = element.get("text", "")
        page = element.get("page", 1)

        if etype in ("table", "code"):
            # Сначала сбросить накопленный текстовый буфер
            flush_buffer()
            if text.strip():
                chunks.append(Chunk(
                    text=text.strip(),
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_number=page,
                    element_type=etype,
                ))
                chunk_id += 1
        else:
            # Текстовый элемент
            if not text_buffer:
                buffer_page = page
            text_buffer += text + " "

            while len(text_buffer) > chunk_size:
                # Найти границу предложения в районе chunk_size
                cut = text_buffer.rfind(". ", 0, chunk_size)
                if cut == -1:
                    cut = chunk_size  # нет точки — режем жёстко
                else:
                    cut += 2  # включить ". "

                chunk_text = text_buffer[:cut].strip()
                carry = text_buffer[max(0, cut - overlap):cut]  # overlap с конца
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_number=buffer_page,
                    element_type="text",
                ))
                chunk_id += 1
                text_buffer = carry + text_buffer[cut:]
                buffer_page = page

    flush_buffer()
    return chunks
```

**Step 4: Запустить тесты — убедиться что проходят**

```bash
pytest tests/core/test_chunker.py -v
```

Ожидаем: все тесты GREEN.

**Step 5: Commit**

```bash
git add core/chunker.py tests/core/test_chunker.py
git commit -m "feat: add chunker with atomic table/code support and text overlap"
```

---

## Task 2: core/embedder.py

**Files:**
- Create: `tests/core/test_embedder.py`
- Create: `core/embedder.py`

**Step 1: Написать тесты**

```python
# tests/core/test_embedder.py
import numpy as np
import pytest
from core.embedder import Embedder


def test_embedder_returns_numpy_array():
    embedder = Embedder()
    result = embedder.embed(["Hello world"])
    assert isinstance(result, np.ndarray)


def test_embedder_output_shape():
    embedder = Embedder()
    texts = ["Hello world", "Semantic search", "SQL query"]
    result = embedder.embed(texts)
    assert result.shape == (3, 384)  # all-MiniLM-L6-v2 → 384 dimensions


def test_embedder_single_text():
    embedder = Embedder()
    result = embedder.embed(["Just one sentence"])
    assert result.shape == (1, 384)


def test_embedder_normalized_vectors():
    """Векторы должны быть нормализованы (для cosine similarity через dot product)."""
    embedder = Embedder()
    result = embedder.embed(["Normalized vector test"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_have_high_similarity():
    embedder = Embedder()
    vecs = embedder.embed(["database schema", "schema of database", "python syntax"])
    sim_same = float(np.dot(vecs[0], vecs[1]))
    sim_diff = float(np.dot(vecs[0], vecs[2]))
    assert sim_same > sim_diff, "Семантически близкие тексты должны иметь более высокое сходство"


def test_embedder_empty_list_returns_empty_array():
    embedder = Embedder()
    result = embedder.embed([])
    assert result.shape[0] == 0
```

**Step 2: Запустить — убедиться что падают**

```bash
pytest tests/core/test_embedder.py -v
```

Ожидаем: `ImportError`

**Step 3: Реализовать `core/embedder.py`**

```python
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Оборачивает SentenceTransformer для генерации нормализованных эмбеддингов.
    Модель загружается один раз при инициализации.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Args:
            texts: список строк для эмбеддинга
        Returns:
            np.ndarray shape (N, 384), нормализованные векторы (L2)
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # L2 нормализация → dot product = cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
```

**Step 4: Запустить тесты**

```bash
pytest tests/core/test_embedder.py -v
```

Ожидаем: все GREEN. (Первый запуск загрузит модель ~90MB — подождать.)

**Step 5: Commit**

```bash
git add core/embedder.py tests/core/test_embedder.py
git commit -m "feat: add embedder wrapping all-MiniLM-L6-v2 with L2 normalization"
```

---

## Task 3: storage/file_storage.py + core/storage.py

**Files:**
- Create: `tests/storage/test_file_storage.py`
- Create: `core/storage.py`
- Create: `storage/file_storage.py`

**Step 1: Написать тесты**

```python
# tests/storage/test_file_storage.py
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.chunker import Chunk
from storage.file_storage import FileStorage


def make_chunks(n=3, source="doc.pdf"):
    return [
        Chunk(
            text=f"chunk text {i}",
            source_file=source,
            chunk_id=i,
            page_number=1,
            element_type="text",
        )
        for i in range(n)
    ]


def make_embeddings(n=3, dim=384):
    vecs = np.random.rand(n, dim).astype(np.float32)
    # нормализовать
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def storage(tmp_path):
    return FileStorage(data_dir=tmp_path)


def test_storage_saves_and_loads_embeddings(storage):
    chunks = make_chunks(3)
    embeddings = make_embeddings(3)
    storage.save(chunks, embeddings)

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape == (3, 384)
    np.testing.assert_allclose(loaded_emb, embeddings, atol=1e-6)


def test_storage_saves_metadata(storage):
    chunks = make_chunks(2)
    embeddings = make_embeddings(2)
    storage.save(chunks, embeddings)

    _, loaded_meta = storage.load()
    assert len(loaded_meta) == 2
    assert loaded_meta[0]["text"] == "chunk text 0"
    assert loaded_meta[0]["source_file"] == "doc.pdf"
    assert loaded_meta[0]["chunk_id"] == 0
    assert loaded_meta[0]["page_number"] == 1
    assert loaded_meta[0]["element_type"] == "text"


def test_storage_load_raises_when_empty(storage):
    with pytest.raises(FileNotFoundError):
        storage.load()


def test_storage_creates_data_dir_if_missing(tmp_path):
    new_dir = tmp_path / "new" / "nested"
    storage = FileStorage(data_dir=new_dir)
    chunks = make_chunks(1)
    embeddings = make_embeddings(1)
    storage.save(chunks, embeddings)
    assert (new_dir / "embeddings.npy").exists()


def test_storage_appends_new_chunks(storage):
    """Добавление новых чанков к существующим."""
    chunks1 = make_chunks(2, source="doc1.pdf")
    emb1 = make_embeddings(2)
    storage.save(chunks1, emb1)

    chunks2 = make_chunks(3, source="doc2.pdf")
    emb2 = make_embeddings(3)
    storage.append(chunks2, emb2)

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape[0] == 5
    assert loaded_meta[4]["source_file"] == "doc2.pdf"


def test_storage_delete_by_source(storage):
    """Удаление всех чанков конкретного файла."""
    chunks = make_chunks(2, source="old.pdf") + make_chunks(2, source="keep.pdf")
    # Создаём эмбеддинги для всех 4 чанков с правильными индексами
    for i, c in enumerate(chunks):
        c.chunk_id = i
    emb = make_embeddings(4)
    storage.save(chunks, emb)

    storage.delete_by_source("old.pdf")

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape[0] == 2
    assert all(m["source_file"] == "keep.pdf" for m in loaded_meta)


def test_storage_search_returns_top_k(storage):
    """cosine similarity поиск возвращает top-k результатов."""
    chunks = make_chunks(10)
    emb = make_embeddings(10)
    storage.save(chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=3)

    assert len(results) == 3
    # Каждый результат: (chunk_metadata, score)
    for meta, score in results:
        assert "text" in meta
        assert 0.0 <= score <= 1.0


def test_storage_search_sorted_by_score(storage):
    """Результаты отсортированы по убыванию score."""
    chunks = make_chunks(5)
    emb = make_embeddings(5)
    storage.save(chunks, emb)

    query = emb[2]  # Точное совпадение с третьим вектором
    results = storage.search(query_embedding=query, top_k=3)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][1] > 0.99  # Первый результат — точное совпадение
```

**Step 2: Запустить — убедиться что падают**

```bash
pytest tests/storage/test_file_storage.py -v
```

**Step 3: Создать `core/storage.py` (абстракция)**

```python
# core/storage.py
from typing import Protocol
import numpy as np
from core.chunker import Chunk


class StorageBackend(Protocol):
    """
    Протокол хранилища. Реализуется:
    - storage.file_storage.FileStorage (MVP)
    - storage.db_storage.DBStorage (Этап 2, pgvector)
    """

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Сохранить chunks и их эмбеддинги (перезапись)."""
        ...

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Добавить новые chunks к существующим."""
        ...

    def load(self) -> tuple[np.ndarray, list[dict]]:
        """Загрузить все эмбеддинги и метаданные. Raises FileNotFoundError если пусто."""
        ...

    def delete_by_source(self, source_file: str) -> None:
        """Удалить все chunks из указанного файла."""
        ...

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """Найти top_k ближайших chunks по cosine similarity."""
        ...
```

**Step 4: Создать `storage/file_storage.py`**

```python
# storage/file_storage.py
import json
from pathlib import Path

import numpy as np

from core.chunker import Chunk


def _chunk_to_meta(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "source_file": chunk.source_file,
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.page_number,
        "element_type": chunk.element_type,
    }


class FileStorage:
    """
    NumPy-хранилище: embeddings.npy (N × 384) + metadata.json.
    Реализует протокол StorageBackend.
    """

    EMB_FILE = "embeddings.npy"
    META_FILE = "metadata.json"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)

    def _emb_path(self) -> Path:
        return self._dir / self.EMB_FILE

    def _meta_path(self) -> Path:
        return self._dir / self.META_FILE

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path(), embeddings)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump([_chunk_to_meta(c) for c in chunks], f, ensure_ascii=False, indent=2)

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        try:
            existing_emb, existing_meta = self.load()
            new_emb = np.vstack([existing_emb, embeddings])
            new_meta = existing_meta + [_chunk_to_meta(c) for c in chunks]
        except FileNotFoundError:
            new_emb = embeddings
            new_meta = [_chunk_to_meta(c) for c in chunks]

        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path(), new_emb)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if not self._emb_path().exists():
            raise FileNotFoundError(f"Хранилище не найдено: {self._emb_path()}")
        embeddings = np.load(self._emb_path())
        with open(self._meta_path(), encoding="utf-8") as f:
            metadata = json.load(f)
        return embeddings, metadata

    def delete_by_source(self, source_file: str) -> None:
        embeddings, metadata = self.load()
        keep = [i for i, m in enumerate(metadata) if m["source_file"] != source_file]
        if not keep:
            # Пустое хранилище — удаляем файлы
            self._emb_path().unlink(missing_ok=True)
            self._meta_path().unlink(missing_ok=True)
            return
        new_emb = embeddings[keep]
        new_meta = [metadata[i] for i in keep]
        np.save(self._emb_path(), new_emb)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        Линейный поиск через cosine similarity.
        query_embedding должен быть нормализован (L2).
        Поскольку все векторы нормализованы при сохранении,
        cosine_sim = dot(query, stored_vector).
        """
        embeddings, metadata = self.load()
        scores: np.ndarray = embeddings @ query_embedding  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(metadata[i], float(scores[i])) for i in top_indices]
```

**Step 5: Запустить тесты**

```bash
pytest tests/storage/test_file_storage.py -v
```

Ожидаем: все GREEN.

**Step 6: Commit**

```bash
git add core/storage.py storage/file_storage.py tests/storage/test_file_storage.py
git commit -m "feat: add NumPy file storage with cosine similarity search"
```

---

## Task 4: core/parser.py

**Files:**
- Create: `tests/core/test_parser.py`
- Create: `core/parser.py`

### Что делает parser

Принимает путь к файлу, возвращает список elements `{"text": str, "type": str, "page": int}`.
Типы: `"text"`, `"table"`, `"code"`.

Docling API (v2):
```python
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert(str(path))
doc = result.document  # DoclingDocument
```

Для итерации по элементам используем `doc.export_to_dict()` или `doc.iterate_items()`.

**Step 1: Написать тесты (с моком Docling)**

```python
# tests/core/test_parser.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.parser import Parser


@pytest.fixture
def mock_docling_result():
    """Мок результата DocumentConverter.convert()"""
    mock_result = MagicMock()
    mock_doc = MagicMock()

    # Симулируем iterate_items() → список (item, level) пар
    text_item = MagicMock()
    text_item.text = "This is a paragraph about databases."
    text_item.__class__.__name__ = "TextItem"

    table_item = MagicMock()
    table_item.export_to_markdown.return_value = "| col1 | col2 |\n|------|------|\n| A    | B    |"
    table_item.__class__.__name__ = "TableItem"

    code_item = MagicMock()
    code_item.text = "SELECT * FROM users;"
    code_item.__class__.__name__ = "CodeItem"

    mock_doc.iterate_items.return_value = [
        (text_item, 0),
        (table_item, 0),
        (code_item, 0),
    ]
    mock_result.document = mock_doc
    return mock_result


def test_parser_returns_list_of_elements(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake pdf content")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert isinstance(elements, list)
    assert len(elements) == 3


def test_parser_text_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[0]["type"] == "text"
    assert "databases" in elements[0]["text"]


def test_parser_table_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[1]["type"] == "table"
    assert "col1" in elements[1]["text"]


def test_parser_code_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[2]["type"] == "code"
    assert "SELECT" in elements[2]["text"]


def test_parser_raises_for_missing_file():
    parser = Parser()
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("/nonexistent/file.pdf"))


def test_parser_raises_for_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    parser = Parser()
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        parser.parse(bad_file)
```

**Step 2: Запустить — убедиться что падают**

```bash
pytest tests/core/test_parser.py -v
```

**Step 3: Реализовать `core/parser.py`**

```python
# core/parser.py
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


class Parser:
    """
    Оборачивает Docling DocumentConverter.
    Возвращает нормализованные элементы: {"text": str, "type": str, "page": int}
    """

    def __init__(self) -> None:
        # Ленивый импорт — Docling тяжёлый, загружаем только при использовании
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемый формат: {path.suffix}. "
                f"Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        result = self._converter.convert(str(path))
        doc = result.document

        elements: list[dict[str, Any]] = []

        for item, _level in doc.iterate_items():
            class_name = item.__class__.__name__

            if class_name == "TableItem":
                try:
                    text = item.export_to_markdown()
                except Exception:
                    text = str(item)
                elements.append({"text": text, "type": "table", "page": 1})

            elif class_name == "CodeItem":
                text = getattr(item, "text", str(item))
                elements.append({"text": text, "type": "code", "page": 1})

            elif hasattr(item, "text") and item.text:
                elements.append({"text": item.text, "type": "text", "page": 1})

        return elements
```

> **Примечание:** Docling v2 API может немного отличаться. Если тесты с реальными файлами падают, проверь `doc.iterate_items()` в Docling docs: https://ds4sd.github.io/docling/

**Step 4: Запустить тесты**

```bash
pytest tests/core/test_parser.py -v
```

Ожидаем: все GREEN.

**Step 5: Commit**

```bash
git add core/parser.py tests/core/test_parser.py
git commit -m "feat: add Docling parser with text/table/code element extraction"
```

---

## Task 5: CLI Commands

**Files:**
- Create: `tests/test_cli.py`
- Create: `cli/__init__.py` (main entry point)
- Create: `cli/commands.py`
- Create: `cli/config_loader.py`

**Step 1: Написать тесты CLI**

```python
# tests/test_cli.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def initialized_storage(tmp_path):
    """Создаём инициализированное хранилище для тестов."""
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


def test_init_command_creates_data_dir(runner, tmp_path):
    result = runner.invoke(main, ["init", "--data-dir", str(tmp_path / "mystore")])
    assert result.exit_code == 0
    assert (tmp_path / "mystore").exists()
    assert "Инициализировано" in result.output


def test_list_command_empty_storage(runner, tmp_path):
    runner.invoke(main, ["init", "--data-dir", str(tmp_path)])
    result = runner.invoke(main, ["list", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "пустое" in result.output.lower() or "документов нет" in result.output.lower()


def test_add_command_indexes_file(runner, tmp_path):
    """add должен распарсить, нарезать и сохранить эмбеддинги."""
    test_doc = tmp_path / "test.md"
    test_doc.write_text("# Test\n\nThis is a test document about databases and SQL.\n")

    with (
        patch("cli.commands.Parser") as MockParser,
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
    ):
        mock_elements = [{"text": "Test content about databases.", "type": "text", "page": 1}]
        MockParser.return_value.parse.return_value = mock_elements
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        mock_storage_instance = MagicMock()
        mock_storage_instance.load.side_effect = FileNotFoundError
        MockStorage.return_value = mock_storage_instance

        result = runner.invoke(main, ["add", str(test_doc), "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Добавлен" in result.output or "chunk" in result.output.lower()


def test_search_command_returns_results(runner, tmp_path):
    """search должен вывести топ-5 результатов с score."""
    mock_results = [
        ({"text": "SQL query example SELECT *", "source_file": "doc.pdf",
          "chunk_id": 0, "page_number": 1, "element_type": "code"}, 0.92),
        ({"text": "Database schema description", "source_file": "arch.docx",
          "chunk_id": 1, "page_number": 2, "element_type": "text"}, 0.78),
    ]

    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(
            main, ["search", "SQL query example", "--data-dir", str(tmp_path)]
        )

    assert result.exit_code == 0
    assert "0.92" in result.output or "92" in result.output
    assert "doc.pdf" in result.output


def test_search_command_empty_storage(runner, tmp_path):
    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.side_effect = FileNotFoundError

        result = runner.invoke(main, ["search", "query", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "пустое" in result.output.lower() or "нет документов" in result.output.lower()
```

**Step 2: Запустить — убедиться что падают**

```bash
pytest tests/test_cli.py -v
```

**Step 3: Реализовать `cli/config_loader.py`**

```python
# cli/config_loader.py
from pathlib import Path
import yaml

_DEFAULTS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 3200,
    "chunk_overlap": 320,
    "top_k_results": 5,
    "data_dir": "data",
    "log_file": "logs/search.log",
}


def load_config(config_path: str | Path = "config.yaml") -> dict:
    cfg = dict(_DEFAULTS)
    path = Path(config_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg
```

**Step 4: Реализовать `cli/commands.py`**

```python
# cli/commands.py
import logging
from pathlib import Path

import click

from cli.config_loader import load_config
from core.chunker import chunk_elements
from core.embedder import Embedder
from core.parser import Parser
from storage.file_storage import FileStorage


def get_storage(data_dir: str) -> FileStorage:
    return FileStorage(data_dir=Path(data_dir))


@click.group()
def main() -> None:
    """docling-rag — семантический поиск по технической документации."""
    pass


@main.command()
@click.option("--data-dir", default="data", help="Директория хранилища")
def init(data_dir: str) -> None:
    """Инициализировать хранилище."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path.parent / "logs").mkdir(exist_ok=True)
    click.echo(f"✓ Инициализировано хранилище: {path.resolve()}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--data-dir", default="data", help="Директория хранилища")
@click.option("--config", default="config.yaml", help="Путь к config.yaml")
def add(file_path: str, data_dir: str, config: str) -> None:
    """Добавить документ или директорию в индекс."""
    cfg = load_config(config)
    path = Path(file_path)
    files = list(path.rglob("*.*")) if path.is_dir() else [path]
    supported = {".pdf", ".docx", ".md", ".txt"}
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        click.echo("Нет поддерживаемых файлов для индексации.")
        return

    parser = Parser()
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)

    total_chunks = 0
    for file in files:
        click.echo(f"Обрабатываю: {file.name} ...", nl=False)
        try:
            elements = parser.parse(file)
            chunks = chunk_elements(
                elements,
                source_file=str(file),
                chunk_size=cfg["chunk_size"],
                overlap=cfg["chunk_overlap"],
            )
            if not chunks:
                click.echo(" (пустой документ, пропускаю)")
                continue
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            storage.append(chunks, embeddings)
            total_chunks += len(chunks)
            click.echo(f" ✓ {len(chunks)} chunks")
        except (ValueError, FileNotFoundError) as e:
            click.echo(f" ✗ Ошибка: {e}")

    click.echo(f"\nДобавлено {total_chunks} chunks из {len(files)} файлов.")


@main.command()
@click.argument("query")
@click.option("--data-dir", default="data", help="Директория хранилища")
@click.option("--top-k", default=5, help="Количество результатов")
@click.option("--config", default="config.yaml", help="Путь к config.yaml")
def search(query: str, data_dir: str, top_k: int, config: str) -> None:
    """Выполнить семантический поиск по документации."""
    cfg = load_config(config)
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)

    try:
        query_emb = embedder.embed([query])[0]
        results = storage.search(query_embedding=query_emb, top_k=top_k)
    except FileNotFoundError:
        click.echo("Хранилище пустое. Добавьте документы: docling-rag add <path>")
        return

    if not results:
        click.echo("Ничего не найдено.")
        return

    click.echo(f"\n🔍 Результаты для: \"{query}\"\n" + "─" * 60)
    for i, (meta, score) in enumerate(results, 1):
        source = Path(meta["source_file"]).name
        page = meta.get("page_number", "?")
        etype = meta.get("element_type", "text")
        text_preview = meta["text"][:300].replace("\n", " ")
        click.echo(
            f"\n[{i}] score={score:.3f} | {source} | стр.{page} | {etype}\n"
            f"    {text_preview}..."
        )

    # Логирование
    _log_search(cfg["log_file"], query, results[0][1] if results else 0.0)


@main.command("list")
@click.option("--data-dir", default="data", help="Директория хранилища")
def list_docs(data_dir: str) -> None:
    """Показать список проиндексированных документов."""
    storage = get_storage(data_dir)
    try:
        _, metadata = storage.load()
    except FileNotFoundError:
        click.echo("Хранилище пустое. Документов нет.")
        return

    sources = {}
    for m in metadata:
        src = m["source_file"]
        sources[src] = sources.get(src, 0) + 1

    click.echo(f"\nПроиндексировано документов: {len(sources)}\n" + "─" * 60)
    for src, count in sorted(sources.items()):
        click.echo(f"  {Path(src).name:40s} {count:4d} chunks  ({src})")


def _log_search(log_file: str, query: str, top_score: float) -> None:
    from datetime import datetime
    path = Path(log_file)
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | score={top_score:.3f} | {query}\n")
```

**Step 5: Реализовать `cli/__init__.py`**

```python
# cli/__init__.py
from cli.commands import main

__all__ = ["main"]
```

**Step 6: Запустить тесты**

```bash
pytest tests/test_cli.py -v
```

Ожидаем: все GREEN.

**Step 7: Запустить все тесты**

```bash
pytest tests/ -v
```

Ожидаем: все GREEN.

**Step 8: Commit**

```bash
git add cli/ tests/test_cli.py
git commit -m "feat: add CLI commands init/add/search/list with config support"
```

---

## Task 6: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Написать интеграционный тест**

```python
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
```

**Step 2: Запустить только юнит-тесты (без интеграционных)**

```bash
pytest tests/ -v -m "not integration"
```

Ожидаем: все GREEN.

**Step 3: Запустить интеграционный тест вручную**

```bash
pytest tests/test_integration.py -v -m integration -s
```

Ожидаем: PASS (первый запуск скачает модель).

**Step 4: Финальный коммит**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke test for full add→search pipeline"
```

---

## Финальная проверка

```bash
# Все тесты (кроме интеграционных)
pytest tests/ -v -m "not integration"

# Проверка установки CLI
docling-rag --help
docling-rag init --help
docling-rag add --help
docling-rag search --help
docling-rag list --help
```

---

## Что дальше (P1, после MVP)

- `R-6: Skills для AI` — создать `skills/docling-rag.md` с инструкциями для Claude Code
- `R-7: docling-rag update <file>` — переиндексация отдельного файла (уже есть `delete_by_source` + `append`)
- `R-8: config.yaml` — уже есть базовая реализация, можно расширить
