# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOPD CNDMASK reads its condition at the wave width.

Wave32 permits single-SGPR selectors such as VCC_HI; a pair read would reject
those selectors. Check both the active generator and its checked-in output.
"""

from pathlib import Path
import re

import pytest

_ROCJITSU = Path(__file__).resolve().parents[3]
_GENERATED = _ROCJITSU / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
_TEMPLATE = Path(__file__).resolve().parents[1] / 'codegen/execute/rj_arith_vopd.py'


def _cndmask_condition(text: str) -> str:
    """The complete assignment expression that resolves the cndmask condition."""
    # Resolve the source at the wave width before passing its lane condition.
    assignments = re.findall(r'uint(?:32|64)_t condition\s*=([^;]+);', text)
    assert len(assignments) == 1, f'expected one condition, found {len(assignments)}'
    return assignments[0]


def _vopd_exec_files():
    files = sorted(_GENERATED.glob('*/vopd_exec.cpp'))
    assert files, f'no generated vopd_exec.cpp under {_GENERATED}'
    return files


class TestVopdCndmaskWaveMask:
    def test_generator_template_reads_the_mask_at_wave_width(self):
        line = _cndmask_condition(_TEMPLATE.read_text())
        assert 'read_wave_mask_scalar' in line
        assert 'read_scalar64' not in line

    @pytest.mark.parametrize('path', _vopd_exec_files(), ids=lambda p: p.parent.name)
    def test_generated_file_reads_the_mask_at_wave_width(self, path):
        text = path.read_text()
        line = _cndmask_condition(text)
        assert 'read_wave_mask_scalar' in line
        assert 'read_scalar64' not in line
        if path.parent.name in ('rdna3', 'rdna4'):
            assert 'const uint32_t condition' in text
            assert 'rj_arith_v_dual_cndmask_b32_w32' in text

    @pytest.mark.parametrize('path', _vopd_exec_files(), ids=lambda p: p.parent.name)
    def test_generated_file_includes_the_helper(self, path):
        assert (
            '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"' in path.read_text()
        )
