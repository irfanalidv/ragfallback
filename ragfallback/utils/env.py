"""Load local environment files (optional ``python-dotenv``)."""

from __future__ import annotations

import os
from typing import Optional, Tuple


def load_env(dotenv_path: Optional[str] = None) -> bool:
    """
    Load ``.env`` into ``os.environ`` if ``python-dotenv`` is installed.

    Searches upward from the current working directory for ``.env`` when ``dotenv_path``
    is omitted (``find_dotenv``).

    Returns:
        True if a file was loaded, False if dotenv is missing or no file found.
    """
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        return False
    path = dotenv_path or find_dotenv(usecwd=True)
    if not path:
        return False
    load_dotenv(path, override=False)
    return True


def mistral_config_from_env() -> Tuple[Optional[str], str]:
    """Return ``(api_key, model_name)`` from the environment (after optional ``load_env``)."""
    key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_TOKEN")
    model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    return key, model
