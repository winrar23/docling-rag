# core/errors.py
class StorageError(Exception):
    """Storage is corrupted or inconsistent."""


class UnsupportedFormatError(Exception):
    """File format is not supported by the parser."""


class LLMUnavailableError(Exception):
    """LLM endpoint is unreachable."""
