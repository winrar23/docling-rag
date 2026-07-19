"""InMemoryRegistry: id в записях + get_by_id — синхронно с DocumentRegistryBackend."""
from tests.fakes import InMemoryRegistry


def test_entries_have_unique_ids():
    reg = InMemoryRegistry()
    reg.upsert("/a.pdf", "A", None, [])
    reg.upsert("/b.pdf", "B", None, [])
    entries = reg.load()
    ids = {e["id"] for e in entries.values()}
    assert len(ids) == 2 and all(ids)


def test_get_by_id_roundtrip():
    reg = InMemoryRegistry()
    reg.upsert("/a.pdf", "A", None, ["x"])
    doc_id = reg.load()["/a.pdf"]["id"]
    source, entry = reg.get_by_id(doc_id)
    assert source == "/a.pdf" and entry["title"] == "A"


def test_get_by_id_unknown_and_malformed_return_none():
    reg = InMemoryRegistry()
    assert reg.get_by_id("00000000-0000-0000-0000-000000000000") is None
    assert reg.get_by_id("not-a-uuid") is None


def test_upsert_preserves_id():
    reg = InMemoryRegistry()
    reg.upsert("/a.pdf", "A", None, [])
    id1 = reg.load()["/a.pdf"]["id"]
    reg.upsert("/a.pdf", "A2", None, [])
    assert reg.load()["/a.pdf"]["id"] == id1
