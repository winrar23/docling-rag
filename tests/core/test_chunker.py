from docling_rag.core.chunker import Chunk, _get_chunker


def test_chunk_has_headings_field():
    chunk = Chunk(
        text="Some text",
        source_file="doc.pdf",
        chunk_id=0,
        page_number=1,
        element_type="text",
        headings=["Chapter 1", "Section 1.1"],
        context_text="Chapter 1\nSection 1.1\nSome text",
    )
    assert chunk.headings == ["Chapter 1", "Section 1.1"]
    assert chunk.context_text == "Chapter 1\nSection 1.1\nSome text"


def test_chunk_headings_default_empty():
    chunk = Chunk(
        text="Text",
        source_file="doc.pdf",
        chunk_id=0,
        page_number=1,
        element_type="text",
    )
    assert chunk.headings == []
    assert chunk.context_text == ""

from unittest.mock import MagicMock, patch
from docling_rag.core.chunker import chunk_document


def _make_mock_doc_chunk(text, headings=None, label_value="text", page_no=1):
    """Helper: create a mock DocChunk matching docling_core's API."""
    chunk = MagicMock()
    chunk.text = text
    chunk.meta.headings = headings or []

    doc_item = MagicMock()
    doc_item.label.value = label_value
    doc_item.prov = [MagicMock(page_no=page_no)]
    chunk.meta.doc_items = [doc_item]

    return chunk


def test_chunk_document_returns_chunks_with_headings():
    from docling_rag.core.chunker import _get_chunker
    mock_doc = MagicMock()
    mock_chunks = [
        _make_mock_doc_chunk("Intro text", headings=["Chapter 1"], label_value="text", page_no=1),
        _make_mock_doc_chunk("col|val\n---|---\na|b", headings=["Chapter 1", "Data"], label_value="table", page_no=2),
    ]

    with patch("docling_rag.core.chunker.HybridChunker") as MockHybrid, \
         patch("docling_rag.core.chunker.HuggingFaceTokenizer") as MockTok:
        _get_chunker.cache_clear()
        MockTok.from_pretrained.return_value = MagicMock()
        instance = MockHybrid.return_value
        instance.chunk.return_value = iter(mock_chunks)
        instance.contextualize.side_effect = lambda c: "\n".join((c.meta.headings or []) + [c.text])

        result = chunk_document(mock_doc, source_file="report.pdf", embedding_model="all-MiniLM-L6-v2")

    assert len(result) == 2
    assert result[0].text == "Intro text"
    assert result[0].headings == ["Chapter 1"]
    assert result[0].element_type == "text"
    assert result[0].page_number == 1
    assert result[0].source_file == "report.pdf"
    assert result[0].chunk_id == 0
    assert "Chapter 1" in result[0].context_text

    assert result[1].element_type == "table"
    assert result[1].page_number == 2
    assert result[1].chunk_id == 1


def test_chunk_document_empty_doc_returns_empty():
    from docling_rag.core.chunker import _get_chunker
    mock_doc = MagicMock()

    with patch("docling_rag.core.chunker.HybridChunker") as MockHybrid, \
         patch("docling_rag.core.chunker.HuggingFaceTokenizer") as MockTok:
        _get_chunker.cache_clear()
        MockTok.from_pretrained.return_value = MagicMock()
        MockHybrid.return_value.chunk.return_value = iter([])

        result = chunk_document(mock_doc, source_file="empty.pdf")

    assert result == []


def test_chunk_document_no_prov_defaults_page_1():
    """If doc_item has no prov, page_number defaults to 1."""
    from docling_rag.core.chunker import _get_chunker
    mock_doc = MagicMock()
    chunk = _make_mock_doc_chunk("text", label_value="text")
    chunk.meta.doc_items[0].prov = []  # no provenance

    with patch("docling_rag.core.chunker.HybridChunker") as MockHybrid, \
         patch("docling_rag.core.chunker.HuggingFaceTokenizer") as MockTok:
        _get_chunker.cache_clear()
        MockTok.from_pretrained.return_value = MagicMock()
        MockHybrid.return_value.chunk.return_value = iter([chunk])
        MockHybrid.return_value.contextualize.return_value = "text"

        result = chunk_document(mock_doc, source_file="doc.pdf")

    assert result[0].page_number == 1


def test_chunk_document_code_element_type():
    from docling_rag.core.chunker import _get_chunker
    mock_doc = MagicMock()
    chunk = _make_mock_doc_chunk("print('hello')", label_value="code", page_no=3)

    with patch("docling_rag.core.chunker.HybridChunker") as MockHybrid, \
         patch("docling_rag.core.chunker.HuggingFaceTokenizer") as MockTok:
        _get_chunker.cache_clear()
        MockTok.from_pretrained.return_value = MagicMock()
        MockHybrid.return_value.chunk.return_value = iter([chunk])
        MockHybrid.return_value.contextualize.return_value = "print('hello')"

        result = chunk_document(mock_doc, source_file="doc.pdf")

    assert result[0].element_type == "code"
    assert result[0].page_number == 3


def test_chunker_is_cached():
    from docling_rag.core.chunker import _get_chunker
    with patch("docling_rag.core.chunker.HuggingFaceTokenizer") as MockTok, \
         patch("docling_rag.core.chunker.HybridChunker") as MockHybrid:
        _get_chunker.cache_clear()
        a = _get_chunker("model-x", 512)
        b = _get_chunker("model-x", 512)
    assert a is b
    MockTok.from_pretrained.assert_called_once()


class TestChunkerModelResolution:
    # NOTE: deviation from the brief's literal test body — also patch HybridChunker.
    # Installed docling-core's HybridChunker is a pydantic model (tokenizer: BaseTokenizer,
    # arbitrary_types_allowed=True) with a model_validator that runs a real isinstance check.
    # Passing an unspec'd MagicMock as tokenizer (from patching only
    # HuggingFaceTokenizer.from_pretrained) raises pydantic_core.ValidationError instead of
    # passing through. Patching HybridChunker too (same pattern as the pre-existing
    # test_chunker_is_cached above) avoids the real validator while still exercising the
    # exact assertions the brief cares about: model-id resolution and max_tokens propagation.
    def test_org_qualified_model_name_used_as_is(self):
        _get_chunker.cache_clear()
        with patch("docling_rag.core.chunker.HybridChunker"), \
             patch("docling_rag.core.chunker.HuggingFaceTokenizer.from_pretrained") as m:
            _get_chunker("deepvk/USER-bge-m3", 512)
        m.assert_called_once_with("deepvk/USER-bge-m3", max_tokens=512)

    def test_bare_model_name_gets_sentence_transformers_prefix(self):
        _get_chunker.cache_clear()
        with patch("docling_rag.core.chunker.HybridChunker"), \
             patch("docling_rag.core.chunker.HuggingFaceTokenizer.from_pretrained") as m:
            _get_chunker("all-MiniLM-L6-v2", 256)
        m.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", max_tokens=256)

    def test_cache_keyed_by_model_and_max_tokens(self):
        # side_effect=fresh MagicMock() per call: the default mocked-class .return_value
        # is a single fixed object regardless of call args, which would make this identity
        # check pass trivially. A fresh instance per call means `a is c` only holds because
        # _get_chunker's lru_cache returns a cached result (real cache-identity behavior).
        _get_chunker.cache_clear()
        with patch("docling_rag.core.chunker.HybridChunker", side_effect=lambda **kw: MagicMock()), \
             patch("docling_rag.core.chunker.HuggingFaceTokenizer.from_pretrained"):
            a = _get_chunker("all-MiniLM-L6-v2", 256)
            b = _get_chunker("all-MiniLM-L6-v2", 512)
            c = _get_chunker("all-MiniLM-L6-v2", 256)
        assert a is c and a is not b
