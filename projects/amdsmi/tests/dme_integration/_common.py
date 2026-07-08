#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices. All rights reserved.
"""Shared utilities for DME integration CI helpers."""

import logging
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


def gh_group(title: str) -> None:
    """Emit a GitHub Actions log group marker."""
    print(f"::group::{title}", flush=True)


def gh_endgroup() -> None:
    print("::endgroup::", flush=True)


def gh_error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def gh_warning(message: str) -> None:
    print(f"::warning::{message}", flush=True)
