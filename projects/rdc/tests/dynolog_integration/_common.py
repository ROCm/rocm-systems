#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared utilities for the dynolog <-> RDC integration CI helpers."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stderr)


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, logging the command for CI visibility.

    Fails fast on a non-zero exit. Output streams to the parent process so
    GitHub Actions log grouping works correctly.
    """
    logger = logging.getLogger("dynolog.run")
    logger.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True)


def gh_error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def gh_warning(message: str) -> None:
    print(f"::warning::{message}", flush=True)
