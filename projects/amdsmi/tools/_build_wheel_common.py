"""
_build_wheel_common.py
======================

Shared helpers used by both ``tools/build_wheel_debian.py`` and
``tools/build_wheel_rpm.py``. Anything OS-specific (interpreter discovery,
prerequisite installation, OS detection) lives in the per-format
drivers. Edit this file to change behavior shared across both.

Python 3.6-safe (uses ``universal_newlines``, not ``text``; no f-strings
in critical paths).
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def run(cmd, cwd=None, env=None, check=True):
    """Execute *cmd* and stream output to the console."""
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, check=check, cwd=cwd, env=merged_env)


def abort(message, code=1):
    log.error(message)
    sys.exit(code)


def find_project_dir(start):
    """Walk up from *start* looking for the first directory with CMakeLists.txt."""
    candidate = Path(start)
    for _ in range(10):
        if (candidate / "CMakeLists.txt").exists():
            return candidate
        candidate = candidate.parent
    abort("Could not auto-detect project root.  Pass --project-dir explicitly.")


def best_effort_pip_upgrade(py, packages):
    """Try to upgrade *packages* via pip; log a warning on failure."""
    result = subprocess.run(
        [py, "-m", "pip", "install", "--upgrade"] + list(packages),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        log.warning(
            "pip upgrade failed with %s (packages: %s)\nstdout: %s\nstderr: %s",
            py,
            ", ".join(packages),
            result.stdout.strip(),
            result.stderr.strip(),
        )
    return result.returncode == 0


def mark_safe_git_dir(path):
    """Register *path* as a safe git directory (avoids dubious-ownership errors)."""
    git_bin = shutil.which("git")
    if not git_bin or not Path(path).exists():
        return True
    result = subprocess.run(
        [git_bin, "config", "--global", "--add", "safe.directory", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        log.warning(
            "git safe.directory add failed for %s (rc=%s): %s",
            path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def write_temp_git_config(config_path, safe_paths):
    """Write a temporary gitconfig that marks *safe_paths* as safe directories."""
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for p in safe_paths:
            lines.append("[safe]\n\tdirectory = %s\n" % p)
        Path(config_path).write_text("\n".join(lines))
        log.info("Temporary git config for safe.directory: %s", config_path)
        return {"GIT_CONFIG_GLOBAL": str(config_path)}
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to create temporary git config: %s", exc)
        return {}
