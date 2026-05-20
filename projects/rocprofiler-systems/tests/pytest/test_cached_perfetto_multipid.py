# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cached-Perfetto multi-pid coverage via fork-example.

Deferred: fork-example in sys_run mode produces only the parent pid's
.proto file in the output directory. The cached-perfetto post-processor
needs further investigation to understand how forked children's cache
files are discovered and merged into the parent's finalize pass. This
file is a placeholder for the test that will land after that
investigation completes.
"""

from __future__ import annotations
import pytest

pytestmark = [
    pytest.mark.fork,
    pytest.mark.skip(
        reason="deferred: fork-example children's cache files are not visible "
        "to the parent's cached-perfetto post-processor in sys_run mode; "
        "needs investigation of discovery::find_cache_files semantics"
    ),
]


def test_multipid_placeholder() -> None:
    """Placeholder so pytest discovery sees the file and reports it skipped."""
