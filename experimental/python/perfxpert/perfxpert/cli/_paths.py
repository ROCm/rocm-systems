"""Shared user path helpers for CLI config files."""

from __future__ import annotations

import os
from pathlib import Path


def user_home() -> Path:
    """Return the explicit HOME override when set, else the platform home."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return Path(home).expanduser()
    return Path.home()


def xdg_config_home() -> Path:
    """Return XDG_CONFIG_HOME, or HOME/.config when XDG is unset."""
    override = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return user_home() / ".config"
