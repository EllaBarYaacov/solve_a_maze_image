"""Shared vision–language backend modules (training + inference)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent


def load_backend_module(backend_name: str) -> Any:
    """
    Load ``vlm_models/<backend_name>.py`` and return the module.

    The module should implement backend-specific hooks such as
    ``build_model_collator(cfg)`` (training), ``load_for_inference(cfg)`` (generative
    inference), and optionally ``generate_from_config(cfg)`` (e.g. detection backends
    that bypass ``generate``).
    """
    path = _DIR / f"{backend_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"VLM backend not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"vlm_models_{backend_name}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
