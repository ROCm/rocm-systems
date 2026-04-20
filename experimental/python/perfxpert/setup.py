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

try:  # PEP 660 — available on setuptools 64+
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except ImportError:  # older setuptools
    _editable_wheel = None  # type: ignore[assignment]


_HERE = Path(__file__).resolve().parent
_BUILD_SCRIPT = _HERE / "scripts" / "build-bundled-opencode.sh"
_BUNDLE_PATH = _HERE / "perfxpert" / "_bundled" / "opencode"
_PATCHES_DIR = _HERE / ".patches"
_OPENCODE_DIR = _HERE / "opencode"
_OPENCODE_URL = "https://github.com/sst/opencode.git"
_OPENCODE_TAG = "v1.4.11"
_SKIP_ENV = "PERFXPERT_SKIP_BUNDLED_BUILD"
_SKIP_BUN_ENV = "PERFXPERT_SKIP_BUN_DOWNLOAD"
_SKIP_OPENCODE_FETCH_ENV = "PERFXPERT_SKIP_OPENCODE_FETCH"


# ---------------------------------------------------------------------------
# Bun auto-download (fallback when `bun` isn't on PATH).
# ---------------------------------------------------------------------------

_BUN_CACHE_DIR = Path.home() / ".cache" / "perfxpert" / "bun"


def _bun_bin_path(binary_name: str = "bun") -> Path:
    """Return the cache-dir path where the downloaded bun binary lives."""
    return _BUN_CACHE_DIR / "bin" / binary_name


def _is_musl_libc() -> bool:
    """Return True if running against musl libc (Alpine etc.) rather than glibc."""
    # Probe `ldd --version` — glibc identifies itself, musl exits 1 with
    # a different banner. Either the environ marker `ldd` provides, or
    # presence of ld-musl-*.so.1 in the usual lib dirs, is enough.
    for d in (
        "/lib",
        "/lib64",
        "/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
    ):
        if any(Path(d).glob("ld-musl-*.so.*")):
            return True
    try:
        out = subprocess.run(
            ["ldd", "--version"], capture_output=True, text=True, timeout=2
        )
        # musl prints "musl libc" in stdout OR stderr on most distros.
        combined = (out.stdout + out.stderr).lower()
        if "musl" in combined:
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def _detect_bun_asset_name() -> tuple[str | None, str]:
    """Return (asset-filename, binary-name) for the current platform.

    Bun ships prebuilt zips: ``bun-{os}-{arch}[-musl].zip``. The archive
    contains either ``bun`` (unix) or ``bun.exe`` (Windows). Returns
    ``(None, "")`` for platforms bun doesn't ship a binary for.
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
        return None, ""
    if system == "linux":
        # Pick the musl variant on Alpine / distroless-musl images so
        # the download actually runs. glibc hosts get the default zip.
        suffix = "-musl" if _is_musl_libc() else ""
        return f"bun-linux-{arch}{suffix}.zip", "bun"
    if system == "darwin":
        return f"bun-darwin-{arch}.zip", "bun"
    if system == "windows":
        # bun ships native Windows binaries as of mid-2024. The zip
        # contains `bun.exe` instead of `bun`. Build may still fail
        # if the underlying opencode toolchain lacks Windows support,
        # but at least bun itself is installable.
        return f"bun-windows-{arch}.zip", "bun.exe"
    return None, ""


def _auto_install_bun() -> Path | None:
    """Download + extract bun into ``~/.cache/perfxpert/bun/bin/``.

    Returns the absolute path to the extracted bun binary on success
    (``bun`` on unix, ``bun.exe`` on Windows), ``None`` on failure
    (network error, unsupported platform, opt-out via
    ``PERFXPERT_SKIP_BUN_DOWNLOAD=1``).
    """
    if os.environ.get(_SKIP_BUN_ENV, "").strip() in {"1", "true", "yes"}:
        print(
            f"[perfxpert/setup.py] {_SKIP_BUN_ENV}=1 — "
            "not auto-downloading bun; skipping bundled opencode build.",
            file=sys.stderr,
        )
        return None

    asset, binary_name = _detect_bun_asset_name()
    if asset is None:
        print(
            "[perfxpert/setup.py] no prebuilt bun binary for this platform "
            f"(system={platform.system()}, machine={platform.machine()}). "
            "Install bun manually from https://bun.sh and rerun pip install, "
            "or set PERFXPERT_OPENCODE_PATH at runtime.",
            file=sys.stderr,
        )
        return None

    bun_bin = _bun_bin_path(binary_name)
    if bun_bin.is_file() and (os.access(bun_bin, os.X_OK) or platform.system() == "Windows"):
        return bun_bin

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
            # bun zip contains `<asset-stem>/bun` (or `bun.exe` on Windows)
            member = next(
                (n for n in zf.namelist() if n.endswith(f"/{binary_name}")),
                None,
            )
            if member is None:
                print(
                    "[perfxpert/setup.py] bun zip layout unrecognised — "
                    "skipping bundled opencode build.",
                    file=sys.stderr,
                )
                return None
            with zf.open(member) as src, bun_bin.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp_path.unlink(missing_ok=True)
        if platform.system() != "Windows":
            bun_bin.chmod(0o755)
    except (urllib.error.URLError, zipfile.BadZipFile, OSError) as exc:
        print(
            f"[perfxpert/setup.py] WARNING: failed to download bun: {exc}. "
            "Install bun manually from https://bun.sh; the library + analyze + "
            "MCP paths still work without it.",
            file=sys.stderr,
        )
        return None

    print(
        f"[perfxpert/setup.py] bun installed at {bun_bin}; "
        "continuing with bundled opencode build.",
        file=sys.stderr,
    )
    return bun_bin


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
# opencode source checkout (scoped submodule init / fallback direct clone).
# ---------------------------------------------------------------------------


def _opencode_dir_is_populated() -> bool:
    """Return True if the vendored opencode/ source tree is populated.

    ``package.json`` is the load-bearing file the build script checks
    (``build-bundled-opencode.sh`` step 1); if it's present, opencode is
    good to go regardless of whether the directory came from a git
    submodule, a tarball extraction, or a direct clone.
    """
    return (_OPENCODE_DIR / "package.json").is_file()


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 300) -> bool:
    """Run a git command silently. Return True on exit-0, False otherwise.

    Never raises — callers treat a False return as "try the next
    fallback" so a transient git failure never aborts `pip install`.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(
            f"[perfxpert/setup.py] git {' '.join(args)} failed: {exc}",
            file=sys.stderr,
        )
        return False
    if result.returncode != 0:
        print(
            f"[perfxpert/setup.py] git {' '.join(args)} exited "
            f"{result.returncode}: {result.stderr.strip()[:400]}",
            file=sys.stderr,
        )
        return False
    return True


def _ensure_opencode_checkout() -> bool:
    """Populate ``opencode/`` if empty, using the cheapest strategy that works.

    The build hook requires the opencode source tree. Normally it's
    present because either (a) the user ran ``git submodule update
    --init`` on the rocm-systems checkout, or (b) pip's built-in
    recursive submodule init populated it during the VCS install.

    When a user invokes ``scripts/pip-install-from-git.sh`` (or sets
    ``GIT_CONFIG_COUNT``/``submodule.active`` env vars manually to skip
    pip's expensive all-submodule init), opencode is intentionally left
    un-initialised — scoped down to just the opencode path. This
    function handles that scoped case plus the pathological case where
    opencode didn't get populated at all.

    Strategy, in order (stops at the first one that succeeds):

    1. ``opencode/package.json`` already exists — nothing to do.
    2. ``_HERE`` sits inside a git work-tree and
       ``experimental/python/perfxpert/opencode`` is a registered
       submodule of that tree — run a scoped
       ``git submodule update --init --depth 1 -- <path>`` so only
       opencode gets initialised.
    3. Direct shallow clone of ``sst/opencode`` at the pinned tag into
       ``opencode/`` — the hard fallback for wheel-extracted trees /
       sdist installs where no enclosing .git exists.

    Opt-out via ``PERFXPERT_SKIP_OPENCODE_FETCH=1`` for air-gap CI.
    """
    if _opencode_dir_is_populated():
        return True

    if os.environ.get(_SKIP_OPENCODE_FETCH_ENV, "").strip() in {"1", "true", "yes"}:
        print(
            f"[perfxpert/setup.py] {_SKIP_OPENCODE_FETCH_ENV}=1 — "
            "not fetching opencode source; bundled build will be skipped.",
            file=sys.stderr,
        )
        return False

    # Strategy 2: scoped submodule init in the enclosing git work-tree.
    # We look upward from _HERE for the repo root. If opencode is
    # registered as a submodule relative to that root, init only that
    # one path — cheap, honors the pinned SHA recorded in the tree.
    repo_root = None
    for parent in (_HERE, *_HERE.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is not None:
        rel_opencode = str(_OPENCODE_DIR.relative_to(repo_root))
        gitmodules = repo_root / ".gitmodules"
        if gitmodules.is_file():
            # Confirm the submodule is actually registered at the
            # expected path — ``git config -f .gitmodules --get-regexp``
            # is the portable query. Exit-0 = registered, non-zero =
            # not registered (fall through to direct clone).
            probed = subprocess.run(
                [
                    "git",
                    "config",
                    "-f",
                    str(gitmodules),
                    "--get-regexp",
                    rf"^submodule\..*\.path$",
                    rf"^{rel_opencode}$",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probed.returncode == 0 and rel_opencode in probed.stdout:
                print(
                    f"[perfxpert/setup.py] opencode/ is empty — running scoped "
                    f"`git submodule update --init --depth 1 -- {rel_opencode}`",
                    file=sys.stderr,
                )
                if _run_git(
                    [
                        "submodule",
                        "update",
                        "--init",
                        "--depth",
                        "1",
                        "--",
                        rel_opencode,
                    ],
                    cwd=repo_root,
                ):
                    if _opencode_dir_is_populated():
                        return True
                # Fall through to direct clone on failure.

    # Strategy 3: direct shallow clone at the pinned tag.
    print(
        f"[perfxpert/setup.py] opencode/ is empty — cloning {_OPENCODE_URL} "
        f"@ {_OPENCODE_TAG} (shallow) into {_OPENCODE_DIR}",
        file=sys.stderr,
    )
    # git clone refuses to populate a non-empty directory. Remove the
    # empty dir first (it's a git submodule placeholder — no tracked
    # files inside, just a .git pointer file at most).
    if _OPENCODE_DIR.exists():
        try:
            shutil.rmtree(_OPENCODE_DIR)
        except OSError as exc:
            print(
                f"[perfxpert/setup.py] could not remove empty opencode/ "
                f"placeholder: {exc} — bundled build will be skipped.",
                file=sys.stderr,
            )
            return False
    if _run_git(
        [
            "clone",
            "--depth",
            "1",
            "--branch",
            _OPENCODE_TAG,
            _OPENCODE_URL,
            str(_OPENCODE_DIR),
        ]
    ):
        if _opencode_dir_is_populated():
            return True

    print(
        "[perfxpert/setup.py] WARNING: opencode source checkout failed. "
        "Library + analyze + MCP paths still work; `perfxpert-code` will "
        "exit with a helpful error until opencode is checked out.",
        file=sys.stderr,
    )
    return False


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
    # Belt-and-suspenders: the opencode source tree is normally populated
    # by the enclosing git submodule init, but when the user invokes
    # `scripts/pip-install-from-git.sh` (or sets `submodule.active`
    # themselves) to skip pip's slow all-submodule init, the source can
    # be missing here. Check + fetch on-demand before the build script
    # tries to find package.json.
    if not _ensure_opencode_checkout():
        print(
            "[perfxpert/setup.py] WARNING: opencode source tree unavailable — "
            "bundled opencode binary NOT built. Library + analyze + MCP "
            "paths still work; `perfxpert-code` will exit with a helpful "
            "error at first launch.",
            file=sys.stderr,
        )
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
