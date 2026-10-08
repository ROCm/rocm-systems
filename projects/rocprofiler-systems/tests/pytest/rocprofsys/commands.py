# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small command lookups shared by pytest modules."""

from __future__ import annotations

import os


def get_ls_command() -> tuple[str, list[str]]:
    """Return the ``ls`` command name and any extra arguments.

    On Red Hat, ``ls`` is an applet of the ``coreutils`` multi-call binary.
    Invoking that binary by a resolved path does not select the ``ls`` applet,
    so pass ``--coreutils-prog=ls`` instead. Elsewhere, return the name ``ls``
    and let the process search ``PATH``. A shell alias is not visible to
    ``exec``.
    """
    if os.path.exists("/usr/bin/coreutils"):
        return "coreutils", ["--coreutils-prog=ls"]
    return "ls", []
