from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    path = Path(__file__).parent / config_path
    with open(path, "r") as f:
        return yaml.safe_load(f)