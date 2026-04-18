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
        if p.is_file():
            return p
        print(f"perfxpert-code: WARNING — PERFXPERT_OPENCODE_PATH={override} not found; "
              "falling back to bundled", file=sys.stderr)

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
    if argv is None:
        argv = sys.argv[1:]

    if _handle_version_flag(argv):
        return 0

    try:
        binary = resolve_opencode_binary()
        config_dir = resolve_config_dir()
    except FileNotFoundError as e:
        print(f"\033[31mperfxpert-code: {e}\033[0m", file=sys.stderr)
        return 1

    # Banner to stderr so it doesn't pollute piped stdout
    if os.environ.get("PERFXPERT_CODE_NO_BANNER", "").strip().lower() not in {"1", "true", "yes"}:
        print_banner()

    # opencode 1.4.x discovers `opencode.json` from cwd (no `--config` flag).
    # Copy the bundled config into a stable per-user dir and `cd` there before exec,
    # so MCP wiring + AGENTS.md instructions apply without polluting the user's cwd.
    runtime_cfg_dir = _prepare_runtime_config_dir(config_dir)

    cmd = [str(binary), *argv]

    # Pass through most of the user env; opencode needs LLM API keys and rocprofv3 envs.
    # We do NOT use the EXECUTION-tool env whitelist here because opencode is the
    # user's interactive session and they explicitly consent to it.
    env = dict(os.environ)
    # Recursion guard marker (spec §5.8 / R10)
    env["PERFXPERT_IN_OPENCODE_SESSION"] = "1"

    try:
        proc = subprocess.run(cmd, env=env, cwd=str(runtime_cfg_dir), check=False)
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def _prepare_runtime_config_dir(src_config_dir: Path) -> Path:
    """Stage a read-only copy of the bundled config where opencode will pick it up.

    opencode 1.4.x has no `--config <path>` flag; it auto-discovers `opencode.json`
    from the current directory. We create a dedicated runtime dir so we can point
    opencode at our bundled config without forcing the user to run from a specific
    directory or clobber their own `opencode.json`.
    """
    import shutil
    runtime_dir = Path.home() / ".cache" / "perfxpert" / "opencode"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for f in src_config_dir.iterdir():
        target = runtime_dir / f.name
        if not target.exists() or target.read_bytes() != f.read_bytes():
            shutil.copy2(f, target)
    return runtime_dir
