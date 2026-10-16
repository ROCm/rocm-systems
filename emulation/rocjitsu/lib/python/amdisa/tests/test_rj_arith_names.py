# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep rj-arith public names and emitted wave suffixes consistent with BuiltinsAMDGPU."""

from rj_arith_headers import public_headers

import re

import pytest

from amdisa.codegen.execute.rj_arith_names import wave_api


@pytest.mark.parametrize(
    'stem,width,expected',
    [
        ('rj_arith_v_wmma_f32_16x16x16_f16', 32, 'rj_arith_v_wmma_f32_16x16x16_f16_w32'),
        ('rj_arith_v_wmma_f32_16x16x16_f16', 64, 'rj_arith_v_wmma_f32_16x16x16_f16_w64'),
        (
            'rj_arith_v_wmma_f32_16x16x16_f16_gfx12',
            32,
            'rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12',
        ),
        (
            'rj_arith_v_wmma_f32_16x16x16_f16_gfx12',
            64,
            'rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12',
        ),
        ('rj_arith_ds_permute_b32_gfx12', 64, 'rj_arith_ds_permute_b32_w64_gfx12'),
        ('rj_arith_v_dual_mul_f32_gfx12', 32, 'rj_arith_v_dual_mul_f32_w32_gfx12'),
    ],
)
def test_wave_suffix_precedes_architecture(stem, width, expected):
    assert wave_api(stem, width) == expected


def test_all_public_instruction_names_follow_the_convention():
    header = public_headers()
    names = re.findall(r'^int (rj_arith_\w+)\(', header, re.M)
    assert names
    assert len(names) == len(set(names))
    for name in names:
        assert '_rdna' not in name and '_wave' not in name, name
        if name.startswith(('rj_arith_s_', 'rj_arith_v_s_')):
            assert not re.search(r'_w(?:32|64)(?:_|$)', name), name
        else:
            assert re.search(r'_w(?:32|64)(?:_gfx[0-9]+)?$', name), name
