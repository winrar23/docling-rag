import pytest
from click.testing import CliRunner

_HERMETIC_DEFAULTS = {
    # embedding_model остаётся all-MiniLM-L6-v2 (не деф. deepvk/USER-bge-m3 — 2.3 ГБ):
    # CLI unit-тесты мокают Embedder, но герметичный дефолт не должен указывать на тяжёлую модель.
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_results": 5,
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
    "llm_timeout_sec": 120,
    # порт 1 — заведомо несоединяемый: юнит-тест, случайно дошедший до реального
    # соединения с БД, падает быстро и громко вместо зависания/долгого таймаута.
    "database_url": "postgresql://test:test@127.0.0.1:1/test",
    "chunk_max_tokens": 512,
    "embed_url": None,
    # юниты не зовут LLM — шаг metadata выключен герметично
    "auto_metadata": False,
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def hermetic_config(monkeypatch):
    """CLI unit tests must never read the repo's live config.yaml."""
    cfg = dict(_HERMETIC_DEFAULTS)
    monkeypatch.setattr("docling_rag.cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    return cfg


@pytest.fixture(autouse=True)
def hermetic_search_log(monkeypatch):
    """Лог поиска пишется в БД — юниты не должны открывать соединение.

    autouse: `search` логирует на каждом успешном запросе, поэтому иначе КАЖДЫЙ
    search-тест ходил бы в несоединяемый DSN и печатал предупреждение, маскируя
    настоящие сбои. Тесты, проверяющие отказ лога, патчат DBSearchLog сами.
    """
    from tests.fakes import InMemorySearchLog
    log = InMemorySearchLog()
    monkeypatch.setattr("docling_rag.cli.commands.DBSearchLog", lambda dsn: log)
    return log


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
def e2e_config(hermetic_config, hermetic_search_log, clean_db, tmp_path, monkeypatch):
    """Осознанное переопределение autouse-фикстур hermetic_* для e2e-тестов.

    hermetic_config патчит load_config на герметичные дефолты с заведомо
    несоединяемой БД (порт 1) и лёгкой моделью; e2e-тесты вместо этого работают
    с реальной тест-БД docling_rag_test и реальной моделью deepvk/USER-bge-m3.
    hermetic_search_log подменяет лог поиска in-memory фейком — e2e возвращает
    настоящий DBSearchLog, иначе сквозной путь «search пишет в БД» остался бы
    непокрытым (ровно там уже пряталась регрессия с логом в /tmp).
    Явная зависимость от обеих фикстур гарантирует порядок: ре-патчи здесь
    применяются ПОСЛЕ герметичных и потому выигрывают; function-scoped
    monkeypatch откатывает всё в обратном порядке.
    """
    import psycopg

    from docling_rag.storage.db_search_log import DBSearchLog

    cfg = dict(hermetic_config)
    cfg.update(
        embedding_model="deepvk/USER-bge-m3",  # e2e проверяет реальную модель (vector(1024))
        database_url=clean_db,  # docling_rag_test: схема готова, documents обнулена
    )
    monkeypatch.setattr("docling_rag.cli.commands.load_config", lambda *_a, **_kw: dict(cfg))
    monkeypatch.setattr("docling_rag.cli.commands.DBSearchLog", DBSearchLog)
    yield cfg
    with psycopg.connect(clean_db) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.commit()
