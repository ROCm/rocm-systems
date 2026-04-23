"""External-tool dependency verifier + install-offer helper (spec §5.8 R21).

Every agent or tool that invokes an external binary, imports an optional
Python package, or dlopens a shared library MUST route the availability
check through this module. Failures produce copy-pasteable install
instructions; with PERFXPERT_ALLOW_INSTALL=1 set, the helper offers to
run the install command itself.

Supported dependency kinds:
- "binary"     — probed via shutil.which
- "pylib"      — probed via importlib.util.find_spec
- "shared_lib" — probed by file presence under common ROCm paths
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


class ExternalToolMissing(RuntimeError):
    """Raised when a required external tool is not available.

    Attributes:
        name:         Registry name of the missing tool.
        install_hint: Copy-pasteable command string or URL for the user.
        install_cmd:  Optional shlex-style list the helper can run on behalf
                      of the user (if PERFXPERT_ALLOW_INSTALL=1). None means
                      "describe only; do not offer to install".
    """

    def __init__(self, name: str, install_hint: str, install_cmd: Optional[List[str]] = None) -> None:
        self.name = name
        self.install_hint = install_hint
        self.install_cmd = install_cmd
        super().__init__(f"{name!r} not available — {install_hint}")


@dataclass(frozen=True)
class _Dep:
    kind: str  # "binary" | "pylib" | "shared_lib"
    install_hint: str
    install_cmd: Optional[List[str]] = None
    shared_lib_filename: Optional[str] = None
    smoke_test: Optional[List[str]] = None  # args to run after binary is found; non-zero → unavailable


# Canonical registry of every external dependency perfxpert touches.
_TOOL_REGISTRY: Dict[str, _Dep] = {
    "opencode": _Dep(
        kind="binary",
        install_hint="curl -fsSL https://opencode.ai/install | bash",
        install_cmd=["bash", "-c", "curl -fsSL https://opencode.ai/install | bash"],
        smoke_test=["--version"],
    ),
    "rocprofv3": _Dep(
        kind="binary",
        install_hint=(
            "rocprofv3 ships with ROCm. Install ROCm 6.3+ from "
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ "
            "then ensure /opt/rocm/bin is on PATH."
        ),
    ),
    "amdclang++": _Dep(
        kind="binary",
        install_hint=(
            "amdclang++ ships with ROCm. Install ROCm then set "
            "PERFXPERT_CXX=/opt/rocm/bin/amdclang++ or add it to PATH."
        ),
    ),
    "rocprof-trace-decoder": _Dep(
        kind="shared_lib",
        shared_lib_filename="librocprof-trace-decoder.so",
        install_hint=(
            "ATT requires the ROCprof Trace Decoder library. Download "
            "rocprof-trace-decoder-manylinux-2.28-X.Y.Z-Linux.sh from the "
            "AMD ROCm releases and install under /opt/rocm."
        ),
    ),
    "mcp": _Dep(
        kind="pylib",
        install_hint="pip install 'mcp>=1.0'",
        install_cmd=[sys.executable, "-m", "pip", "install", "mcp>=1.0"],
    ),
    "pexpect": _Dep(
        kind="pylib",
        install_hint="pip install 'pexpect>=4.8'",
        install_cmd=[sys.executable, "-m", "pip", "install", "pexpect>=4.8"],
    ),
    "anthropic": _Dep(
        kind="pylib",
        install_hint="pip install anthropic",
        install_cmd=[sys.executable, "-m", "pip", "install", "anthropic"],
    ),
    "openai": _Dep(
        kind="pylib",
        install_hint="pip install openai",
        install_cmd=[sys.executable, "-m", "pip", "install", "openai"],
    ),
}


def _shared_lib_search_paths(name: str) -> List[str]:
    """Return ROCm-aware search paths for shared libraries."""
    paths = []
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    for sub in ("lib", "lib64", "lib/rocprofiler-sdk"):
        paths.append(os.path.join(rocm, sub))

    # Check for tool-specific environment variable (e.g., PERFXPERT_ROCPROF_TRACE_DECODER_PATH)
    env_var = f"PERFXPERT_{name.upper().replace('-', '_')}_PATH"
    if env_var in os.environ:
        env_path = os.environ[env_var]
        # If it points to a directory, add it; if it's a file path, add its parent
        if os.path.isfile(env_path):
            paths.append(os.path.dirname(env_path))
        elif os.path.isdir(env_path):
            paths.append(env_path)

    # Check user home directory installation paths
    home = os.path.expanduser("~")
    for sub in ("lib", "lib64"):
        paths.append(os.path.join(home, ".local/opt", name, "opt/rocm", sub))

    paths.extend(os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    return [p for p in paths if p]


def check_tool_available(name: str) -> Tuple[bool, str]:
    """Return (is_available, detail_message) for the named dependency."""
    dep = _TOOL_REGISTRY.get(name)
    if dep is None:
        return False, f"unknown tool {name!r} (not in _TOOL_REGISTRY)"

    if dep.kind == "binary":
        path = shutil.which(name)
        if not path:
            return False, f"{name} not found on PATH. Install: {dep.install_hint}"
        if dep.smoke_test is not None:
            cmd = [path] + list(dep.smoke_test)
            try:
                result = subprocess.run(
                    cmd,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    return (
                        False,
                        f"binary at {path} but smoke-test {dep.smoke_test!r} " f"exited {result.returncode}",
                    )
            except OSError as e:
                return False, f"binary at {path} but smoke-test failed: {e}"
            except subprocess.TimeoutExpired:
                return False, f"binary at {path} but smoke-test timed out after 2 s"
        return True, f"{name} at {path}"

    if dep.kind == "pylib":
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return True, f"{name} importable"
        return False, f"{name} not importable. pip install: {dep.install_hint}"

    if dep.kind == "shared_lib":
        filename = dep.shared_lib_filename or f"lib{name}.so"
        for p in _shared_lib_search_paths(name):
            candidate = Path(p) / filename
            if candidate.is_file():
                return True, f"{name} at {candidate}"
        return False, f"{filename} not found under ROCm paths. {dep.install_hint}"

    return False, f"unknown kind {dep.kind!r} for {name}"


def offer_install(name: str) -> bool:
    """Offer to run the install command for `name`. Returns True on success.

    Prints the command, prompts y/N on stdin. If declined OR the dep has no
    install_cmd, returns False without running anything.
    """
    dep = _TOOL_REGISTRY.get(name)
    if dep is None or dep.install_cmd is None:
        print(
            f"No install command registered for {name}. Install manually: "
            f"{dep.install_hint if dep else 'unknown tool'}"
        )
        return False

    print(f"\nperfxpert would like to install {name} via:")
    print(f"  {' '.join(dep.install_cmd)}")
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Install declined.")
        return False

    try:
        result = subprocess.run(dep.install_cmd, check=False)
    except OSError as e:
        print(f"Install failed: {e}")
        return False
    ok = result.returncode == 0
    if ok:
        print(f"{name} installed successfully.")
    else:
        print(f"Install exited with code {result.returncode}.")
    return ok


def require_tool(name: str, *, allow_install: bool = False) -> str:
    """Assert that `name` is available; raise ExternalToolMissing otherwise.

    If `allow_install=True` AND `PERFXPERT_ALLOW_INSTALL=1` AND the registry
    has an install_cmd, prompts the user to install. Only raises after a
    declined / failed install (or if allow_install is False).

    Returns the detail string from check_tool_available on success.
    """
    ok, detail = check_tool_available(name)
    if ok:
        return detail

    dep = _TOOL_REGISTRY.get(name)
    user_opted_in = os.environ.get("PERFXPERT_ALLOW_INSTALL", "").strip() in ("1", "true", "yes")
    if allow_install and user_opted_in and dep and dep.install_cmd:
        installed = offer_install(name)
        if installed:
            ok, detail = check_tool_available(name)
            if ok:
                return detail

    raise ExternalToolMissing(
        name=name,
        install_hint=dep.install_hint if dep else "(unknown tool)",
        install_cmd=dep.install_cmd if dep else None,
    )


__all__ = [
    "ExternalToolMissing",
    "check_tool_available",
    "require_tool",
    "offer_install",
]
