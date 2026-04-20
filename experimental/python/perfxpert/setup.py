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
* If ``bun`` is not on PATH: auto-download the `bun` binary into a
  vendored cache directory (``~/.cache/perfxpert/bun/bin/bun``) and
  put it on PATH for the build step only. Supports Linux x64,
  Linux arm64, macOS x64, macOS arm64. Install still succeeds with
  a WARNING on unsupported platforms (Windows, musl libc, offline) —
  only ``perfxpert-code`` is affected; library + analyze + MCP paths
  all work regardless.
* Opt-out via ``PERFXPERT_SKIP_BUNDLED_BUILD=1`` — useful in
  tightly-sandboxed CI where network isn't available.
* Opt-out of auto-bun via ``PERFXPERT_SKIP_BUN_DOWNLOAD=1`` — fall back
  to the warn-and-skip path if you prefer to manage bun yourself.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
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
_SKIP_BUN_ENV = "PERFXPERT_SKIP_BUN_DOWNLOAD"


# ---------------------------------------------------------------------------
# Bun auto-download (fallback when `bun` isn't on PATH).
# ---------------------------------------------------------------------------

_BUN_CACHE_DIR = Path.home() / ".cache" / "perfxpert" / "bun"
_BUN_BIN = _BUN_CACHE_DIR / "bin" / "bun"


def _detect_bun_asset_name() -> str | None:
    """Return the bun release-asset filename for the current platform.

    Bun ships pre-built zip archives named e.g.
    ``bun-linux-x64.zip``. Returns ``None`` for platforms bun doesn't
    ship a binary for (Windows, musl, etc.).
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch_map = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    arch = arch_map.get(machine)
    if arch is None:
        return None
    if system == "linux":
        # bun doesn't ship a musl-libc build under the same zip name.
        # Detect glibc by looking for libc.so.6; bail if musl-only.
        if not any(
            Path(d).glob("libc.so.6")
            for d in ("/lib64", "/lib", "/lib/x86_64-linux-gnu", "/lib/aarch64-linux-gnu")
        ):
            # Could still work but safer to bail and let the user install.
            return None
        return f"bun-linux-{arch}.zip"
    if system == "darwin":
        return f"bun-darwin-{arch}.zip"
    return None


def _auto_install_bun() -> Path | None:
    """Download + extract bun into ``~/.cache/perfxpert/bun/bin/bun``.

    Returns the absolute path to the extracted ``bun`` binary on
    success, ``None`` on failure (network error, unsupported platform,
    opt-out via ``PERFXPERT_SKIP_BUN_DOWNLOAD=1``).
    """
    if os.environ.get(_SKIP_BUN_ENV, "").strip() in {"1", "true", "yes"}:
        print(
            f"[perfxpert/setup.py] {_SKIP_BUN_ENV}=1 — "
            "not auto-downloading bun; skipping bundled opencode build.",
            file=sys.stderr,
        )
        return None
    if _BUN_BIN.is_file() and os.access(_BUN_BIN, os.X_OK):
        return _BUN_BIN

    asset = _detect_bun_asset_name()
    if asset is None:
        print(
            "[perfxpert/setup.py] no prebuilt bun binary for this platform "
            f"(system={platform.system()}, machine={platform.machine()}). "
            "Install bun manually from https://bun.sh and rerun pip install, "
            "or set PERFXPERT_OPENCODE_PATH at runtime.",
            file=sys.stderr,
        )
        return None

    url = f"https://github.com/oven-sh/bun/releases/latest/download/{asset}"
    print(
        f"[perfxpert/setup.py] bun not on PATH — downloading {asset} "
        f"(~40 MB) → {_BUN_CACHE_DIR}",
        file=sys.stderr,
    )
    _BUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_BUN_CACHE_DIR / "bin").mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as zf:
            # bun zip contains `<asset-stem>/bun`
            member = next((n for n in zf.namelist() if n.endswith("/bun")), None)
            if member is None:
                print(
                    "[perfxpert/setup.py] bun zip layout unrecognised — "
                    "skipping bundled opencode build.",
                    file=sys.stderr,
                )
                return None
            with zf.open(member) as src, _BUN_BIN.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp_path.unlink(missing_ok=True)
        _BUN_BIN.chmod(0o755)
    except (urllib.error.URLError, zipfile.BadZipFile, OSError) as exc:
        print(
            f"[perfxpert/setup.py] WARNING: failed to download bun: {exc}. "
            "Install bun manually from https://bun.sh; the library + analyze + "
            "MCP paths still work without it.",
            file=sys.stderr,
        )
        return None

    print(
        f"[perfxpert/setup.py] bun installed at {_BUN_BIN}; "
        "continuing with bundled opencode build.",
        file=sys.stderr,
    )
    return _BUN_BIN


def _ensure_bun_on_path() -> str | None:
    """Return a PATH env string that includes an available bun binary.

    Returns ``None`` if bun is neither on PATH nor auto-installable.
    """
    existing = shutil.which("bun")
    if existing:
        return os.environ.get("PATH", "")
    installed = _auto_install_bun()
    if installed is None:
        return None
    # Prepend our cache `bin/` to PATH for the subprocess only.
    current_path = os.environ.get("PATH", "")
    return f"{installed.parent}{os.pathsep}{current_path}"


# ---------------------------------------------------------------------------
# Build-hook entry points.
# ---------------------------------------------------------------------------


def _opencode_build_needed() -> tuple[bool, str]:
    """Return (should_build, reason). ``should_build=False`` stops here."""
    if os.environ.get(_SKIP_ENV, "").strip() in {"1", "true", "yes"}:
        return False, f"{_SKIP_ENV}=1 — skipping bundled opencode build"
    if not _BUILD_SCRIPT.is_file():
        return False, f"build script missing ({_BUILD_SCRIPT}); nothing to do"
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
    build_path = _ensure_bun_on_path()
    if build_path is None:
        print(
            "[perfxpert/setup.py] WARNING: bun not available — "
            "bundled opencode binary NOT built. Library + analyze + MCP "
            "paths still work; `perfxpert-code` will prompt to install "
            "bun at first launch.",
            file=sys.stderr,
        )
        return
    env = os.environ.copy()
    env["PATH"] = build_path
    result = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)],
        cwd=str(_HERE),
        env=env,
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
