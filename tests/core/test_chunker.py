from core.chunker import Chunk


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
from core.chunker import chunk_document


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
    mock_doc = MagicMock()
    mock_chunks = [
        _make_mock_doc_chunk("Intro text", headings=["Chapter 1"], label_value="text", page_no=1),
        _make_mock_doc_chunk("col|val\n---|---\na|b", headings=["Chapter 1", "Data"], label_value="table", page_no=2),
    ]

    with patch("core.chunker.HybridChunker") as MockHybrid:
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
    mock_doc = MagicMock()

    with patch("core.chunker.HybridChunker") as MockHybrid:
        MockHybrid.return_value.chunk.return_value = iter([])

        result = chunk_document(mock_doc, source_file="empty.pdf")

    assert result == []


def test_chunk_document_no_prov_defaults_page_1():
    """If doc_item has no prov, page_number defaults to 1."""
    mock_doc = MagicMock()
    chunk = _make_mock_doc_chunk("text", label_value="text")
    chunk.meta.doc_items[0].prov = []  # no provenance

    with patch("core.chunker.HybridChunker") as MockHybrid:
        MockHybrid.return_value.chunk.return_value = iter([chunk])
        MockHybrid.return_value.contextualize.return_value = "text"

        result = chunk_document(mock_doc, source_file="doc.pdf")

    assert result[0].page_number == 1


def test_chunk_document_code_element_type():
    mock_doc = MagicMock()
    chunk = _make_mock_doc_chunk("print('hello')", label_value="code", page_no=3)

    with patch("core.chunker.HybridChunker") as MockHybrid:
        MockHybrid.return_value.chunk.return_value = iter([chunk])
        MockHybrid.return_value.contextualize.return_value = "print('hello')"

        result = chunk_document(mock_doc, source_file="doc.pdf")

    assert result[0].element_type == "code"
    assert result[0].page_number == 3
