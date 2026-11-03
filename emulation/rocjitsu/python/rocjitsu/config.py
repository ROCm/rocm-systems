# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installed JSON presets, exposed under their existing filename stems."""

from pathlib import Path


def _directory():
    from ._launch import launcher

    prefix = Path(launcher()).parent.parent
    directory = prefix / "share/rocjitsu/configs"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Native Rocjitsu configs are missing from {directory}; "
            "install the native runtime and its configs into this Python environment."
        )
    return directory


def names():
    """Return the JSON filename stems provided by the native installation."""
    return tuple(sorted(path.stem for path in _directory().glob("*.json")))


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        path = _directory() / (name + ".json")
    except FileNotFoundError as error:
        raise AttributeError(str(error)) from error
    if not path.is_file():
        raise AttributeError(f"No installed Rocjitsu preset named {name!r}")
    return path


def __dir__():
    try:
        return ["names", *names()]
    except FileNotFoundError:
        return ["names"]


def _resolve(value):
    if isinstance(value, str) and "/" not in value and not Path(value).exists():
        filename = value if value.endswith(".json") else value + ".json"
        path = _directory() / filename
        if path.is_file():
            return path
    return value
