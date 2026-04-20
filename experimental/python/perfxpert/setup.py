"""Setuptools shim — runs `scripts/build-bundled-opencode.sh` automatically
during `pip install` (wheel + editable) so the patched opencode binary is
bundled into ``perfxpert/_bundled/opencode`` as part of install.

Project metadata lives in ``pyproject.toml``. This file only exists to
inject a pre-build hook. Deleting it would leave users needing to run
the build script manually.

Behavior:
* Pre-build (``build_py``), pre-editable-install (``develop`` /
  ``editable_wheel``): invoke the build script if the bundled binary
  is missing OR older than the patch series.
* If ``bun`` is not on PATH: emit a clear warning and SKIP the build —
  install still succeeds so users can ``pip install`` for development
  tasks that don't touch the bundled opencode path (tests, analyze,
  mcp server). A runtime ``perfxpert-code`` invocation will then fail
  with an actionable message pointing at the build script.
* Opt-out via ``PERFXPERT_SKIP_BUNDLED_BUILD=1`` — useful in
  tightly-sandboxed CI where network/bun isn't available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

try:  # PEP 660 — available on setuptools 64+
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except ImportError:  # older setuptools
    _editable_wheel = None  # type: ignore[assignment]


_HERE = Path(__file__).resolve().parent
_BUILD_SCRIPT = _HERE / "scripts" / "build-bundled-opencode.sh"
_BUNDLE_PATH = _HERE / "perfxpert" / "_bundled" / "opencode"
_PATCHES_DIR = _HERE / ".patches"
_SKIP_ENV = "PERFXPERT_SKIP_BUNDLED_BUILD"


def _opencode_build_needed() -> tuple[bool, str]:
    """Return (should_build, reason). ``should_build=False`` stops here."""
    if os.environ.get(_SKIP_ENV, "").strip() in {"1", "true", "yes"}:
        return False, f"{_SKIP_ENV}=1 — skipping bundled opencode build"
    if not _BUILD_SCRIPT.is_file():
        return False, f"build script missing ({_BUILD_SCRIPT}); nothing to do"
    if shutil.which("bun") is None:
        return False, (
            "bun not found on PATH; skipping bundled opencode build. "
            "Install bun (curl -fsSL https://bun.sh/install | bash) and run "
            f"`bash {_BUILD_SCRIPT.relative_to(_HERE)}` to populate "
            f"{_BUNDLE_PATH.relative_to(_HERE)}."
        )
    # Rebuild when the binary is missing OR older than the newest patch.
    if not _BUNDLE_PATH.is_file():
        return True, "bundled opencode binary missing — building"
    binary_mtime = _BUNDLE_PATH.stat().st_mtime
    if _PATCHES_DIR.is_dir():
        for patch in _PATCHES_DIR.glob("*.patch"):
            if patch.stat().st_mtime > binary_mtime:
                return True, f"patch {patch.name} is newer than binary — rebuilding"
    return False, "bundled opencode binary up-to-date"


def _run_opencode_build() -> None:
    should_build, reason = _opencode_build_needed()
    print(f"[perfxpert/setup.py] opencode build: {reason}", file=sys.stderr)
    if not should_build:
        return
    result = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)],
        cwd=str(_HERE),
        check=False,
    )
    if result.returncode != 0:
        # Don't fail the install — leave a clear breadcrumb.
        print(
            f"[perfxpert/setup.py] WARNING: build-bundled-opencode.sh exited "
            f"{result.returncode}. `perfxpert-code` won't launch until you run "
            f"`bash {_BUILD_SCRIPT.relative_to(_HERE)}` successfully.",
            file=sys.stderr,
        )


class _BuildPyWithOpencode(_build_py):
    def run(self) -> None:
        _run_opencode_build()
        super().run()


class _DevelopWithOpencode(_develop):
    def run(self) -> None:
        _run_opencode_build()
        super().run()


_cmdclass: dict = {
    "build_py": _BuildPyWithOpencode,
    "develop": _DevelopWithOpencode,
}

if _editable_wheel is not None:
    class _EditableWheelWithOpencode(_editable_wheel):
        def run(self) -> None:
            _run_opencode_build()
            super().run()

    _cmdclass["editable_wheel"] = _EditableWheelWithOpencode


setup(cmdclass=_cmdclass)
