# cli/config_loader.py
from pathlib import Path
import yaml

_DEFAULTS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_results": 5,
    "data_dir": "data",
    "log_file": "logs/search.log",
    "agent_enabled": False,
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_api_key": "lm-studio",
    "llm_model": "local-model",
    "agent_top_k": 5,
}


def load_config(config_path: str | Path = "config.yaml") -> dict:
    cfg = dict(_DEFAULTS)
    path = Path(config_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg
