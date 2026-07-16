import pytest
from click.testing import CliRunner

_HERMETIC_DEFAULTS = {
    # embedding_model остаётся all-MiniLM-L6-v2 (не деф. deepvk/USER-bge-m3 — 2.3 ГБ):
    # CLI unit-тесты мокают Embedder, но герметичный дефолт не должен указывать на тяжёлую модель.
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_results": 5,
    "log_file": "",  # заполняется per-test из tmp_path
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
    # порт 1 — заведомо несоединяемый: юнит-тест, случайно дошедший до реального
    # соединения с БД, падает быстро и громко вместо зависания/долгого таймаута.
    "database_url": "postgresql://test:test@127.0.0.1:1/test",
    "chunk_max_tokens": 512,
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def hermetic_config(tmp_path, monkeypatch):
    """CLI unit tests must never read the repo's live config.yaml or write repo logs/."""
    cfg = dict(_HERMETIC_DEFAULTS)
    cfg["log_file"] = str(tmp_path / "logs" / "search.log")
    monkeypatch.setattr("docling_rag.cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    return cfg


@pytest.fixture
def fake_backends(monkeypatch):
    from tests.fakes import InMemoryRegistry, InMemoryStorage
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", lambda dsn: storage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: registry)
    return storage, registry


# --- e2e (integration) ---
# Реюз фикстур postgres-тест-БД (docling_rag_test): db_url создаёт БД и схему,
# clean_db делает TRUNCATE documents CASCADE перед тестом.
from tests.storage.test_db_backends import clean_db, db_url  # noqa: E402, F401


@pytest.fixture
def e2e_config(hermetic_config, clean_db, tmp_path, monkeypatch):
    """Осознанное переопределение autouse-фикстуры hermetic_config для e2e-тестов.

    hermetic_config патчит load_config на герметичные дефолты с заведомо
    несоединяемой БД (порт 1) и лёгкой моделью; e2e-тесты вместо этого работают
    с реальной тест-БД docling_rag_test и реальной моделью deepvk/USER-bge-m3.
    Явная зависимость от hermetic_config гарантирует порядок: ре-патч
    load_config здесь применяется ПОСЛЕ герметичного и потому выигрывает;
    function-scoped monkeypatch откатывает оба патча в обратном порядке.
    """
    import psycopg

    cfg = dict(hermetic_config)
    cfg.update(
        embedding_model="deepvk/USER-bge-m3",  # e2e проверяет реальную модель (vector(1024))
        database_url=clean_db,  # docling_rag_test: схема готова, documents обнулена
    )
    monkeypatch.setattr("docling_rag.cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    yield cfg
    with psycopg.connect(clean_db) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.commit()
