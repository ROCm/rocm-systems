#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared utilities for DME integration CI helpers."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stderr)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, logging the command for CI visibility.

    Fail-fast by default (``check=True``). Use ``capture=True`` to capture
    stdout/stderr as text; otherwise output streams to the parent process
    so GitHub Actions log grouping works correctly.
    """
    logger = logging.getLogger("dme.run")
    logger.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=capture, env=env
    )


def gh_error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def gh_warning(message: str) -> None:
    print(f"::warning::{message}", flush=True)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
