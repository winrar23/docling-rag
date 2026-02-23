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
