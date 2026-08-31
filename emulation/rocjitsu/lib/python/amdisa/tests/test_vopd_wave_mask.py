# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOPD `v_dual_cndmask_b32` must read its lane mask at the wave's width.

The emitted body used `read_scalar64` unconditionally. In wave32 the lane mask
is 32 bits, and a 64-bit read of it goes through `resolve_src_scalar64`, which
rejects any selector that cannot begin a register pair. `VCC_HI` (107) is such a
selector, so the model threw `std::logic_error` and the process aborted --
`_topk_topp_kernel` on gfx1250 did exactly that in all eight of its shapes.

`read_wave_mask_scalar` picks the width from `wf_size()` and is what the
non-VOPD cndmask has always used. These tests pin the VOPD path to it. The
first checks the generator template, the second checks the checked-in generated
files, because a template fix that is never regenerated changes nothing.
"""

from pathlib import Path

import pytest

_ROCJITSU = Path(__file__).resolve().parents[3]
_GENERATED = _ROCJITSU / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
_TEMPLATE = Path(__file__).resolve().parents[1] / 'codegen/_generator.py'


def _cndmask_line(text: str) -> str:
    """The one line that resolves the cndmask condition."""
    lines = [ln for ln in text.splitlines() if 'uint64_t condition' in ln]
    assert len(lines) == 1, f'expected one condition line, found {len(lines)}'
    return lines[0]


def _vopd_exec_files():
    files = sorted(_GENERATED.glob('*/vopd_exec.cpp'))
    assert files, f'no generated vopd_exec.cpp under {_GENERATED}'
    return files


class TestVopdCndmaskWaveMask:
    def test_generator_template_reads_the_mask_at_wave_width(self):
        line = _cndmask_line(_TEMPLATE.read_text())
        assert 'read_wave_mask_scalar' in line
        assert 'read_scalar64' not in line

    @pytest.mark.parametrize('path', _vopd_exec_files(), ids=lambda p: p.parent.name)
    def test_generated_file_reads_the_mask_at_wave_width(self, path):
        line = _cndmask_line(path.read_text())
        assert 'read_wave_mask_scalar' in line
        assert 'read_scalar64' not in line

    @pytest.mark.parametrize('path', _vopd_exec_files(), ids=lambda p: p.parent.name)
    def test_generated_file_includes_the_helper(self, path):
        assert (
            '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"' in path.read_text()
        )
