# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sparse-clone rocprofiler-systems and install the pytest venv."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import die, log, run
from .constants import PROJECT_SUBDIR, ROCM_SYSTEMS_REPO


def _git_rev(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _update_submodules(repo_dir: Path, env: dict) -> None:
    """Init/update the project's git submodules (needed for --build-from-source)."""
    log("Updating git submodules for projects/rocprofiler-systems (recursive)...")
    # Path-limited first (faster; avoids other monorepo projects), then fall back.
    rc = run(
        [
            "git",
            "-C",
            str(repo_dir),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--",
            PROJECT_SUBDIR,
        ],
        env=env,
        check=False,
    )
    if rc != 0:
        run(
            ["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"],
            env=env,
        )


def sync_source(
    workdir: Path,
    branch: str,
    env: dict,
    no_fetch: bool = False,
    submodules: bool = False,
    force: bool = False,
) -> tuple[Path, bool]:
    """Sparse-clone (or update) rocm-systems.

    Returns ``(source_dir, source_changed)`` where ``source_changed`` is True when
    a fresh clone happened or the branch HEAD moved since the last run.

    When ``no_fetch`` is set (offline mode), an existing checkout is reused as-is
    with no network access; if none exists the run aborts with guidance. When
    ``submodules`` is set (source builds), submodules are initialized (online) or
    verified present (offline). When ``force`` is set, a successful fetch is
    always treated as a source change so examples/tests are rebuilt/refreshed.
    """
    repo_dir = workdir / "rocm-systems"
    src_dir = repo_dir / PROJECT_SUBDIR

    if no_fetch:
        if not (src_dir / "CMakeLists.txt").is_file():
            die(
                "offline mode (--offline) but no source checkout at "
                f"{src_dir}.\n"
                "       Run once with --prepare-only on a networked node (e.g. the "
                "login node) first."
            )
        rev = _git_rev(repo_dir)
        log(f"Offline: reusing existing checkout at {rev[:10]} (no fetch).")
        if submodules and not (src_dir / "external").is_dir():
            die(
                "offline --build-from-source but submodules are not present at "
                f"{src_dir / 'external'}.\n"
                "       Run --prepare-only --build-from-source on a networked node "
                "first."
            )
        return src_dir, False

    if (src_dir / "CMakeLists.txt").is_file():
        old_rev = _git_rev(repo_dir)
        log(f"Existing checkout at {old_rev[:10]}; fetching latest '{branch}'...")
        run(["git", "-C", str(repo_dir), "fetch", "--prune", "origin", branch], env=env)
        run(["git", "-C", str(repo_dir), "checkout", branch], env=env, check=False)
        run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"], env=env)
        new_rev = _git_rev(repo_dir)
        changed = old_rev != new_rev or force
        if changed and force and old_rev == new_rev:
            log(
                f"--force-sync: source still at {new_rev[:10]}, "
                "refreshing staged build/test artifacts anyway."
            )
        elif changed:
            log(f"Source updated: {old_rev[:10]} -> {new_rev[:10]}")
        else:
            log(f"Source already up to date at {new_rev[:10]}")
        if submodules:
            _update_submodules(repo_dir, env)
        return src_dir, changed

    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            branch,
            ROCM_SYSTEMS_REPO,
            str(repo_dir),
        ],
        env=env,
    )
    run(["git", "-C", str(repo_dir), "sparse-checkout", "set", PROJECT_SUBDIR], env=env)

    if not (src_dir / "CMakeLists.txt").is_file():
        die(f"sparse checkout did not produce expected source at {src_dir}")

    if submodules:
        _update_submodules(repo_dir, env)

    log(f"Checked out {branch} @ {_git_rev(repo_dir)[:10]}")
    return src_dir, True


def make_venv(workdir: Path, src_dir: Path, skip_install: bool = False) -> Path:
    """Create a venv and install requirements.txt. Returns the venv python path.

    When ``skip_install`` is set (offline mode), an existing venv is reused without
    any pip network access; if none exists the run aborts with guidance.
    """
    venv_dir = workdir / "venv"
    py = venv_dir / "bin" / "python"

    if skip_install:
        if not py.exists():
            die(
                "offline mode (--offline) but no venv at "
                f"{venv_dir}.\n"
                "       Run once with --prepare-only on a networked node (e.g. the "
                "login node) first."
            )
        log(f"Offline: reusing existing venv at {venv_dir} (no pip install).")
        return py

    if not py.exists():
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"], quiet=True)

    req = src_dir / "requirements.txt"
    if req.is_file():
        run([str(py), "-m", "pip", "install", "-r", str(req)])
    else:
        log(f"WARNING: {req} not found; installing pytest directly")
        run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-subtests",
                "PyYAML",
                "numpy",
            ]
        )
    # extra plugins used by the runner: real per-test timeouts + flaky retries
    run(
        [str(py), "-m", "pip", "install", "pytest-timeout", "pytest-rerunfailures"],
        check=False,
    )
    return py


def capture_pip_freeze(venv_py: Path, out_path: Path, facts: dict) -> None:
    """Record the exact resolved venv package versions (reproducibility manifest)."""
    try:
        r = subprocess.run(
            [str(venv_py), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out_path.write_text(r.stdout)
        facts["pip_freeze_log"] = str(out_path)
        log(f"pip freeze -> {out_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not capture pip freeze: {exc}")


def install_system_deps() -> None:
    """Best-effort install of build/runtime system packages (needs sudo/root)."""
    apt = shutil.which("apt-get")
    if not apt:
        log("apt-get not found; skipping system-dependency installation.")
        return
    prefix = [] if os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
    if not prefix and os.geteuid() != 0:
        log("Not root and sudo unavailable; skipping system-dependency installation.")
        return
    pkgs = ["build-essential", "cmake", "libopenmpi-dev", "git"]
    run(prefix + [apt, "update"], check=False)
    run(prefix + [apt, "install", "-y", *pkgs], check=False)
