# core/errors.py
from typing import Iterator


class StorageError(Exception):
    """Storage is corrupted or inconsistent."""


# NB: намеренно НЕ наследуют StorageError — иначе существующие `except StorageError`
# (CLI search/list/ask: «Хранилище повреждено») перехватят инфраструктурные ошибки.
class StorageUnavailableError(Exception):
    """Storage backend is unreachable (e.g. postgres down)."""


class StorageSchemaMissingError(Exception):
    """Storage schema is not initialized (run init)."""


class UnsupportedFormatError(Exception):
    """File format is not supported by the parser."""


class LLMUnavailableError(Exception):
    """LLM endpoint is unreachable."""


class EmbedServiceUnavailableError(Exception):
    """Embed-сервис недоступен (connect/timeout/5xx). НЕ наследует StorageError."""


def cause_chain(e: BaseException) -> Iterator[BaseException]:
    """Yield e и всю цепочку __cause__/__context__ (с защитой от циклов).

    httpx/openai/psycopg заворачивают исходную ошибку в несколько слоёв —
    потребители ищут в цепочке конкретные типы через isinstance
    (cli: _is_connection_error; api: классификация connect/timeout в /chat).
    """
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
