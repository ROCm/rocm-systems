# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small command lookups shared by pytest modules."""

from __future__ import annotations

import functools
import os
import subprocess

_LS_COMMAND = "ls"
_COREUTILS_COMMAND = "coreutils"
_COREUTILS_PATH = "/usr/bin/coreutils"
# Only the Red Hat multi-call binary accepts this option.
_COREUTILS_LS_ARGS = ["--coreutils-prog=ls"]


@functools.lru_cache(maxsize=None)
def get_ls_command() -> tuple[str, list[str]]:
    """Return the ``ls`` command name and any extra arguments.

    On Red Hat, ``ls`` is an applet of the ``coreutils`` multi-call binary.
    Invoking that binary by a resolved path does not select the ``ls`` applet,
    so pass ``--coreutils-prog=ls`` instead. On Ubuntu 26.04, ``/usr/bin/coreutils``
    also ships but rejects that option, so probe it instead of assuming from the
    file's existence. Otherwise return the name ``ls`` and let the process
    search ``PATH``. A shell alias is not visible to ``exec``.
    """
    if os.path.exists(_COREUTILS_PATH):
        try:
            probe = subprocess.run(
                [_COREUTILS_PATH, *_COREUTILS_LS_ARGS, "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return _LS_COMMAND, []
        if probe.returncode == 0:
            return _COREUTILS_COMMAND, list(_COREUTILS_LS_ARGS)
    return _LS_COMMAND, []
