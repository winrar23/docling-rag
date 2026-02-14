import json
import pytest
from pathlib import Path
from storage.doc_registry import DocRegistry


@pytest.fixture
def registry(tmp_path):
    return DocRegistry(data_dir=tmp_path)


def test_load_returns_empty_when_no_file(registry):
    assert registry.load() == {}


def test_upsert_creates_entry(registry):
    registry.upsert("docs/book.pdf", title="Clean Code", topic="software", tags=["refactoring"])
    data = registry.load()
    assert "docs/book.pdf" in data
    entry = data["docs/book.pdf"]
    assert entry["title"] == "Clean Code"
    assert entry["topic"] == "software"
    assert entry["tags"] == ["refactoring"]
    assert "added_at" in entry


def test_upsert_preserves_added_at_on_second_call(registry):
    registry.upsert("a.pdf", title="First", topic=None, tags=[])
    first_added_at = registry.load()["a.pdf"]["added_at"]

    registry.upsert("a.pdf", title="Updated", topic="new", tags=["x"])
    second = registry.load()["a.pdf"]

    assert second["added_at"] == first_added_at  # preserved
    assert second["title"] == "Updated"           # updated


def test_upsert_none_title_and_topic(registry):
    registry.upsert("b.pdf", title=None, topic=None, tags=[])
    entry = registry.load()["b.pdf"]
    assert entry["title"] is None
    assert entry["topic"] is None
    assert entry["tags"] == []


def test_upsert_multiple_docs(registry):
    registry.upsert("a.pdf", title="A", topic="t1", tags=["x"])
    registry.upsert("b.pdf", title="B", topic="t2", tags=["y"])
    data = registry.load()
    assert len(data) == 2
    assert data["a.pdf"]["title"] == "A"
    assert data["b.pdf"]["title"] == "B"


def test_get_returns_entry(registry):
    registry.upsert("c.pdf", title="C", topic="t", tags=["a"])
    entry = registry.get("c.pdf")
    assert entry is not None
    assert entry["title"] == "C"


def test_get_returns_none_for_missing(registry):
    assert registry.get("missing.pdf") is None


def test_delete_removes_entry(registry):
    registry.upsert("d.pdf", title="D", topic=None, tags=[])
    registry.upsert("e.pdf", title="E", topic=None, tags=[])
    registry.delete("d.pdf")
    data = registry.load()
    assert "d.pdf" not in data
    assert "e.pdf" in data


def test_delete_nonexistent_is_noop(registry):
    registry.upsert("f.pdf", title="F", topic=None, tags=[])
    registry.delete("nonexistent.pdf")  # must not raise
    assert "f.pdf" in registry.load()


def test_delete_on_empty_is_noop(registry):
    registry.delete("anything.pdf")  # must not raise
