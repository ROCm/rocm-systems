"""Setuptools shim for legacy setuptools validation.

PerfXpert no longer builds or packages a generated opencode binary during
Python package builds. The default `perfxpert-code` path resolves only patched
artifacts built explicitly from the pinned opencode submodule.

Project metadata lives in ``pyproject.toml``. This file remains only to keep a
clear error for old setuptools versions that otherwise produce broken wheel
metadata.
"""

from __future__ import annotations

import sys

from setuptools import setup


# -------------------------------------------------------------------
# pip 22 / setuptools <61 bootstrap guard (Phase 8 Blocker 4).
#
# Stock ``rocm/dev-ubuntu-22.04:latest`` ships pip 22.0.2 + setuptools
# 59.x. Even though ``pyproject.toml`` declares
# ``[build-system] requires = ["setuptools>=61"]``, pre-PEP-517 pip
# invokes ``setup.py`` directly and the old setuptools writes
# ``UNKNOWN`` as the wheel metadata name — pip then rejects the wheel
# with the misleading error ``filename has 'perfxpert', but metadata
# has 'unknown'``. Refuse with a CLEAR, actionable message instead.
# -------------------------------------------------------------------
def _guard_old_setuptools() -> None:
    try:
        import setuptools  # noqa: F401 — we just need __version__
    except ImportError:
        return
    ver_str = getattr(setuptools, "__version__", "") or ""
    try:
        # Parse the first dotted segment as major; forgiving of pre-release
        # suffixes ("61.0.0rc1" → 61).
        major = int(ver_str.split(".")[0].split("-")[0])
    except (ValueError, IndexError):
        return
    if major < 61:
        msg = (
            "\n\n"
            "[perfxpert/setup.py] setuptools {v} is too old (requires >= 61).\n"
            "  Old setuptools writes 'UNKNOWN' into the wheel metadata and\n"
            "  pip rejects the build with 'filename has \"perfxpert\", but\n"
            "  metadata has \"unknown\"'. Fix:\n\n"
            "      pip install -U pip setuptools wheel\n\n"
            "  Then re-run your pip install command.\n"
        ).format(v=ver_str or "<unknown>")
        print(msg, file=sys.stderr)
        raise SystemExit(1)


_guard_old_setuptools()

setup()
