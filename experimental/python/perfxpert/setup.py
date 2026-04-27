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
* If ``bun`` is not on PATH: bootstrap bun into the user's home directory
  and add it to PATH for this build. If the OS-level prerequisites needed
  for that bootstrap are missing, fail with distro-specific package-manager
  guidance instead of silently producing a broken ``perfxpert-code``.
* Opt-out via ``PERFXPERT_SKIP_BUNDLED_BUILD=1`` — useful in
  tightly-sandboxed CI where network isn't available and the interactive
  ``perfxpert-code`` TUI is intentionally not being built.
* The setup hook will initialize the repo-pinned opencode submodule
  when possible. If the source tree is missing that submodule, it fails
  with an actionable message rather than cloning a mutable upstream tag
  during install.
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
_SKIP_ENV = "PERFXPERT_SKIP_BUNDLED_BUILD"
_SKIP_OPENCODE_FETCH_ENV = "PERFXPERT_SKIP_OPENCODE_FETCH"
_DEFAULT_BUN_INSTALL_URL = "https://bun.sh/install"


def _package_manager_prereq_hint() -> str:
    return (
        "Install the OS prerequisites first:\n"
        "  Ubuntu 22.04 / 24.04:\n"
        "    apt install -y curl git unzip python3-venv python3-pip\n"
        "  RHEL 9:\n"
        "    command -v curl >/dev/null || dnf install -y curl\n"
        "    dnf install -y git unzip python3.11 python3.11-pip\n"
        "  RHEL 10:\n"
        "    command -v curl >/dev/null || dnf install -y curl\n"
        "    dnf install -y git unzip python3 python3-pip\n"
        "  SLES 15:\n"
        "    zypper install -y curl git unzip python311 python311-pip\n"
    )


def _missing_tools(tools: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for tool in tools:
        if shutil.which(tool) is None:
            missing.append(tool)
    return missing


def _missing_build_prereqs() -> list[str]:
    return _missing_tools(("bash", "git"))


def _missing_bun_bootstrap_prereqs() -> list[str]:
    return _missing_tools(("bash", "curl", "unzip"))


def _missing_os_prereqs() -> list[str]:
    return _missing_tools(("bash", "curl", "git", "unzip"))


def _fail_build(message: str) -> None:
    print(
        f"[perfxpert/setup.py] ERROR: {message}\n\n{_package_manager_prereq_hint()}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _run_bun_install_script(env: dict[str, str]) -> int:
    url = os.environ.get("PERFXPERT_BUN_INSTALL_URL", _DEFAULT_BUN_INSTALL_URL)
    missing = _missing_bun_bootstrap_prereqs()
    if missing:
        _fail_build(
            "missing OS prerequisites for bun bootstrap: "
            + ", ".join(missing)
            + "."
        )
    curl = shutil.which("curl")
    bash = shutil.which("bash")
    assert curl is not None
    assert bash is not None

    curl_proc = subprocess.Popen(
        [curl, "-fsSL", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=False,
    )
    assert curl_proc.stdout is not None
    bash_proc = subprocess.run(
        [bash],
        stdin=curl_proc.stdout,
        env=env,
        check=False,
    )
    curl_proc.stdout.close()
    _, curl_stderr = curl_proc.communicate()
    if curl_proc.returncode != 0:
        sys.stderr.write(curl_stderr.decode("utf-8", errors="replace"))
        return curl_proc.returncode
    return bash_proc.returncode


def _ensure_bun_on_path() -> str | None:
    """Return a PATH env string that includes an available bun binary.

    Bootstraps bun into ``$BUN_INSTALL`` when needed so ``pip install`` can
    produce the bundled patched ``perfxpert-code`` binary end to end.
    """
    existing = shutil.which("bun")
    if existing:
        return os.environ.get("PATH", "")
    missing = _missing_bun_bootstrap_prereqs()
    if missing:
        _fail_build(
            "missing OS prerequisites for bun bootstrap: "
            + ", ".join(missing)
            + "."
        )

    bun_install = os.environ.get("BUN_INSTALL") or str(Path.home() / ".bun")
    env = os.environ.copy()
    env["BUN_INSTALL"] = bun_install

    print(
        "[perfxpert/setup.py] bun not on PATH — bootstrapping bun so the "
        "bundled patched opencode binary is built during pip install.",
        file=sys.stderr,
    )
    rc = _run_bun_install_script(env)
    if rc != 0:
        _fail_build(f"bun bootstrap failed with exit code {rc}.")

    bun_path = str(Path(bun_install) / "bin")
    path = os.environ.get("PATH", "")
    if bun_path not in path.split(os.pathsep):
        path = bun_path + os.pathsep + path if path else bun_path
    if shutil.which("bun", path=path) is None:
        _fail_build(f"bun bootstrap finished but bun is unavailable under {bun_path}.")
    return path


# ---------------------------------------------------------------------------
# opencode source checkout (scoped submodule init / no-network fallback).
# ---------------------------------------------------------------------------


def _opencode_dir_is_populated() -> bool:
    """Return True if the vendored opencode/ source tree is populated.

    ``package.json`` is the load-bearing file the build script checks
    (``build-bundled-opencode.sh`` step 1), but the patch/build pipeline
    also needs git metadata because ``apply-opencode-patches.sh`` uses
    ``git apply`` + ``git checkout`` for sequencing and rollback. Accept
    only the repo-pinned git checkout here; tarball/direct-clone layouts
    without ``.git`` are intentionally rejected.
    """
    return (_OPENCODE_DIR / "package.json").is_file() and (_OPENCODE_DIR / ".git").exists()


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
    3. Refuse to clone from the network during install. If the source
       tree does not contain the repo-pinned submodule, warn and skip.

    Opt-out via ``PERFXPERT_SKIP_OPENCODE_FETCH=1`` for air-gap CI.
    """
    if _opencode_dir_is_populated():
        return True

    if os.environ.get(_SKIP_OPENCODE_FETCH_ENV, "").strip() in {"1", "true", "yes"}:
        print(
            f"[perfxpert/setup.py] {_SKIP_OPENCODE_FETCH_ENV}=1 — "
            "not fetching opencode source.",
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
            # not registered (fall through to the warning path).
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

    rel_hint = (
        str(_OPENCODE_DIR.relative_to(repo_root))
        if repo_root is not None
        else "experimental/python/perfxpert/opencode"
    )
    print(
        "[perfxpert/setup.py] WARNING: opencode/ is empty and setup.py will "
        "not clone from the network during install. Initialize the pinned "
        f"submodule first (`git submodule update --init --depth 1 -- {rel_hint}`) "
        "or use scripts/pip-install-from-git.sh so pip scopes submodule init "
        "to the PerfXpert opencode submodule.",
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
        return True, f"build script missing ({_BUILD_SCRIPT})"
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
    if not _BUILD_SCRIPT.is_file():
        _fail_build(
            f"required bundled opencode build script is missing: {_BUILD_SCRIPT}. "
            f"Set {_SKIP_ENV}=1 only if this build intentionally excludes perfxpert-code."
        )
    missing = _missing_build_prereqs()
    if missing:
        _fail_build(
            "missing OS prerequisites for bundled perfxpert-code build: "
            + ", ".join(missing)
            + "."
        )
    # Belt-and-suspenders: the opencode source tree is normally populated
    # by the enclosing git submodule init, but when the user invokes
    # `scripts/pip-install-from-git.sh` (or sets `submodule.active`
    # themselves) to skip pip's slow all-submodule init, the source can
    # be missing here. Check + fetch on-demand before the build script
    # tries to find package.json.
    if not _ensure_opencode_checkout():
        _fail_build(
            "opencode source tree unavailable. The default perfxpert-code "
            "path requires the repo-pinned opencode submodule."
        )
    build_path = _ensure_bun_on_path()
    if build_path is None:
        _fail_build("bun is unavailable after bootstrap.")
    env = os.environ.copy()
    env["PATH"] = build_path
    result = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)],
        cwd=str(_HERE),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _fail_build(
            f"build-bundled-opencode.sh exited {result.returncode}; "
            "bundled perfxpert-code was not produced."
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
