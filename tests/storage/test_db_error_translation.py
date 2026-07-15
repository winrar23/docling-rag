"""Юнит-тесты транслятора psycopg -> доменные исключения (без postgres, герметично)."""
import psycopg
import pytest

from docling_rag.core.errors import StorageError, StorageSchemaMissingError, StorageUnavailableError
from docling_rag.storage.db_storage import _translate_db_errors


def test_operational_error_becomes_storage_unavailable():
    with pytest.raises(StorageUnavailableError) as exc_info:
        with _translate_db_errors():
            raise psycopg.OperationalError("connection refused")
    assert "connection refused" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, psycopg.OperationalError)


def test_undefined_table_becomes_schema_missing():
    with pytest.raises(StorageSchemaMissingError):
        with _translate_db_errors():
            raise psycopg.errors.UndefinedTable('relation "chunks" does not exist')


def test_programming_error_mentioning_vector_becomes_schema_missing():
    """register_vector/DDL без расширения vector -> схема не инициализирована."""
    with pytest.raises(StorageSchemaMissingError):
        with _translate_db_errors():
            raise psycopg.ProgrammingError('type "vector" does not exist')


def test_other_programming_error_passes_through():
    with pytest.raises(psycopg.ProgrammingError):
        with _translate_db_errors():
            raise psycopg.ProgrammingError("syntax error at or near SELEKT")


def test_unrelated_exceptions_pass_through():
    """Транслятор не должен глотать/переименовывать неожиданные ошибки."""
    with pytest.raises(FileNotFoundError):
        with _translate_db_errors():
            raise FileNotFoundError("Storage is empty")
    with pytest.raises(ValueError):
        with _translate_db_errors():
            raise ValueError("top_k must be positive")


def test_domain_errors_do_not_inherit_storage_error():
    """Иначе CLI `except StorageError` («Хранилище повреждено») перехватит их первым."""
    assert not issubclass(StorageUnavailableError, StorageError)
    assert not issubclass(StorageSchemaMissingError, StorageError)
