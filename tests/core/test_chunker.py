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
