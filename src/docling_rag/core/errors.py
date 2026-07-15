# core/errors.py
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
