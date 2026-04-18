"""opencode_launcher — `perfxpert-code` entry point.

Launches the bundled opencode binary with the AMD-themed config directory.
Respects PERFXPERT_OPENCODE_PATH override.
Prints an AMD banner before handing control over.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Iterable

from perfxpert.tools._tooldep import require_tool


__all__ = ["main", "resolve_opencode_binary", "resolve_config_dir", "print_banner"]


_BRANDING_NAME = "AMD ROCm PerfXpert"
_BRANDING_VERSION = "0.2.0"


def _perfxpert_version() -> str:
    try:
        import perfxpert
        return getattr(perfxpert, "__version__", _BRANDING_VERSION)
    except ImportError:
        return _BRANDING_VERSION


def resolve_opencode_binary() -> Path:
    """Locate the opencode binary.

    Priority:
    1. $PERFXPERT_OPENCODE_PATH (user override)
    2. perfxpert/_bundled/opencode (per-platform wheel)
    3. `which opencode` on PATH
    4. Use require_tool with install hint
    """
    override = os.environ.get("PERFXPERT_OPENCODE_PATH")
    if override:
        p = Path(override)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        print(
            f"perfxpert-code: WARNING — PERFXPERT_OPENCODE_PATH={override} "
            "is missing or not executable; falling back to bundled",
            file=sys.stderr,
        )

    # Bundled binary
    try:
        with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode") as p:
            if p.is_file():
                return p
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    # PATH fallback with install helper
    try:
        require_tool("opencode", allow_install=True)
        on_path = shutil.which("opencode")
        if on_path:
            return Path(on_path)
    except Exception:
        pass

    raise FileNotFoundError(
        "opencode binary not found. Set PERFXPERT_OPENCODE_PATH, "
        "install a platform wheel that bundles opencode, or add opencode to PATH."
    )


def resolve_config_dir() -> Path:
    """Return the bundled AMD-themed opencode config directory."""
    try:
        with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode_config") as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        raise FileNotFoundError(
            "perfxpert/_bundled/opencode_config not found. "
            "Reinstall perfxpert."
        ) from e


def print_banner(stream=sys.stderr) -> None:
    """Print the AMD PerfXpert banner."""
    version = _perfxpert_version()
    banner = f"""
\033[38;5;196m╔══════════════════════════════════════════════════════════╗
║                                                          ║
║           AMD ROCm PerfXpert — opencode edition          ║
║                     version {version:<10s}                   ║
║                                                          ║
║   Interactive GPU performance analysis and optimization  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝\033[0m
"""
    stream.write(banner)


def _handle_version_flag(argv: Iterable[str]) -> bool:
    """If --version / -V appears, print AMD-branded version and exit."""
    for a in argv:
        if a in {"--version", "-V"}:
            version = _perfxpert_version()
            print(f"{_BRANDING_NAME} {version} (opencode wrapper)")
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point for `perfxpert-code`."""
    from perfxpert.runtime import recursion_guard

    if argv is None:
        argv = sys.argv[1:]

    if _handle_version_flag(argv):
        return 0

    try:
        recursion_guard.ensure_not_recursive("opencode")
        binary = resolve_opencode_binary()
        config_dir = resolve_config_dir()
    except recursion_guard.RecursionGuardViolation as e:
        print(f"\033[31mperfxpert-code: {e}\033[0m", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"\033[31mperfxpert-code: {e}\033[0m", file=sys.stderr)
        return 1

    # Banner to stderr so it doesn't pollute piped stdout
    if os.environ.get("PERFXPERT_CODE_NO_BANNER", "").strip().lower() not in {"1", "true", "yes"}:
        print_banner()

    # opencode 1.4.x discovers `opencode.json` from cwd (no `--config` flag).
    # Copy the bundled config into a stable per-user dir and `cd` there before exec,
    # so MCP wiring + AGENTS.md instructions apply without polluting the user's cwd.
    cmd = [str(binary), *argv]

    # Pass through most of the user env; opencode needs LLM API keys and rocprofv3 envs.
    # We do NOT use the EXECUTION-tool env whitelist here because opencode is the
    # user's interactive session and they explicitly consent to it.
    try:
        cache_root = _runtime_cache_root()
        with tempfile.TemporaryDirectory(prefix="opencode-", dir=str(cache_root)) as tmp_dir:
            runtime_cfg_dir = _prepare_runtime_config_dir(config_dir, Path(tmp_dir))
            try:
                with recursion_guard.opencode_session():
                    env = dict(os.environ)
                    proc = subprocess.run(
                        cmd,
                        env=env,
                        cwd=str(runtime_cfg_dir),
                        check=False,
                    )
            finally:
                _make_runtime_config_cleanup_safe(runtime_cfg_dir)
    except KeyboardInterrupt:
        return 130
    except OSError as e:
        print(f"\033[31mperfxpert-code: {e}\033[0m", file=sys.stderr)
        return 1
    return proc.returncode


def _runtime_cache_root() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        cache_base = Path(xdg_cache_home)
    else:
        home = os.environ.get("HOME")
        if home:
            cache_base = Path(home) / ".cache"
        else:
            try:
                cache_base = Path.home() / ".cache"
            except (RuntimeError, OSError):
                cache_base = Path(tempfile.gettempdir()) / f"perfxpert-{os.getuid()}"
    cache_root = cache_base / "perfxpert"
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_root.chmod(0o700)
    return cache_root


def _prepare_runtime_config_dir(src_config_dir: Path, runtime_dir: Path | None = None) -> Path:
    """Stage a read-only copy of the bundled config where opencode will pick it up.

    opencode 1.4.x has no `--config <path>` flag; it auto-discovers `opencode.json`
    from the current directory. We create a dedicated runtime dir so we can point
    opencode at our bundled config without forcing the user to run from a specific
    directory or clobber their own `opencode.json`. The staged copy is created in a
    private cache root and made read-only before launch so the runtime config is not
    left behind as a mutable user-writable directory.
    """
    if runtime_dir is None:
        runtime_dir = Path(
            tempfile.mkdtemp(prefix="opencode-", dir=str(_runtime_cache_root()))
        )
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_dir.chmod(0o700)
    for f in src_config_dir.iterdir():
        if not f.is_file():
            continue
        target = runtime_dir / f.name
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            target.chmod(0o600)
            target.unlink()
        shutil.copy2(f, target)
        target.chmod(0o400)
    runtime_dir.chmod(0o500)
    return runtime_dir


def _make_runtime_config_cleanup_safe(runtime_dir: Path) -> None:
    if not runtime_dir.exists():
        return
    runtime_dir.chmod(0o700)
    for child in runtime_dir.iterdir():
        if child.is_file():
            child.chmod(0o600)
