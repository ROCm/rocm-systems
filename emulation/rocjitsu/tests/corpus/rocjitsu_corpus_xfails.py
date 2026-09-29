# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Temporary non-strict xfails, loaded only by the RocJITsu corpus runner."""

import pytest

ISSUE = 'https://github.com/ROCm/rocm-systems/issues/12482'
XFAILS = {
    'cts.gfx1250.memory_isa.memory_isa_gfx1250_block_test': (
        f'gfx1250 block transfers ignore sparse M0 masks; {ISSUE}'
    ),
    'cts.gfx1250.memory_isa.memory_isa_gfx1250_flat_aperture_test': (
        f'gfx1250 flat-aperture test times out; {ISSUE}'
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        callspec = getattr(item, 'callspec', None)
        case = callspec.params.get('corpus_case') if callspec else None
        reason = XFAILS.get(getattr(case, 'id', None))
        if reason:
            item.add_marker(pytest.mark.xfail(strict=False, reason=reason))
