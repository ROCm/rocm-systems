# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import fnmatch


def fnmatch_glob_matches(pattern: str, target: str) -> bool:
    """Case-sensitive ``fnmatch.fnmatchcase`` with ``"all"`` aliased to ``"*"``
    and surrounding ``/`` stripped."""
    glob = pattern.strip()
    if glob == "all":
        glob = "*"
    glob = glob.strip("/")
    if not glob or not target:
        return False
    return fnmatch.fnmatchcase(target, glob)
