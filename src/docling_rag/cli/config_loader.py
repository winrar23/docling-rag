# cli/config_loader.py
import os
import sys
from pathlib import Path

import yaml

_DEFAULTS = {
    "embedding_model": "deepvk/USER-bge-m3",
    "top_k_results": 5,
    "data_dir": "data",
    "log_file": "logs/search.log",
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
    "database_url": "postgresql://docling:docling@127.0.0.1:5432/docling_rag",
    "chunk_max_tokens": 512,
}


class ConfigError(Exception):
    """Invalid or missing configuration."""


def load_config(config_path: str | Path = "config.yaml", *, required: bool = False) -> dict:
    cfg = dict(_DEFAULTS)
    path = Path(config_path)
    if not path.exists():
        if required:
            raise ConfigError(f"Конфиг не найден: {path}")
        return cfg
    try:
        with open(path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Невалидный YAML в {path}: {e}") from e
    if not isinstance(user_cfg, dict):
        raise ConfigError(f"Конфиг {path} должен быть YAML-словарём, получен {type(user_cfg).__name__}")
    unknown = set(user_cfg) - set(_DEFAULTS)
    if unknown:
        print(f"Предупреждение: неизвестные ключи конфига: {', '.join(sorted(unknown))}", file=sys.stderr)
    cfg.update(user_cfg)

    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        cfg["database_url"] = env_url

    return cfg
