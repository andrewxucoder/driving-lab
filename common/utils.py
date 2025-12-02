"""General-purpose helpers."""

from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_placeholder(path: str | Path) -> Any:
    """Stub for YAML loading to be implemented later."""
    raise NotImplementedError("YAML loading is not yet implemented.")
