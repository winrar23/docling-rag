import pytest

pytest.importorskip("fastapi")

from docling_rag.api import app as app_module  # noqa: E402


def _clear_cache():
    getattr(app_module.get_settings, "cache_clear", lambda: None)()


def test_get_settings_reads_config_once_per_process(monkeypatch):
    calls = {"n": 0}

    def fake_load_config(*a, **kw):
        calls["n"] += 1
        return {"database_url": "postgresql://x"}

    monkeypatch.setattr(app_module, "load_config", fake_load_config)
    _clear_cache()
    try:
        assert app_module.get_settings()["database_url"] == "postgresql://x"
        app_module.get_settings()
        assert calls["n"] == 1  # конфиг читается один раз, не на каждый запрос
    finally:
        _clear_cache()
