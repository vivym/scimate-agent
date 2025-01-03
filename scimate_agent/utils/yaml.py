from pathlib import Path
from typing import Any


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error reading YAML file {path}: {e}")


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    import yaml

    try:
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception as e:
        raise ValueError(f"Error writing YAML file {path}: {e}")
