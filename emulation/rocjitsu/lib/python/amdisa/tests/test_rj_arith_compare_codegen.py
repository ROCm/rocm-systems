# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path


import re


import pytest


@pytest.mark.parametrize(
    'arch,kind,count',
    [
        ('rdna3', 'integer', 144),
        ('rdna4', 'integer', 120),
        ('rdna3', 'float', 160),
        ('rdna4', 'float', 140),
        ('rdna3', 'class', 10),
        ('rdna4', 'class', 10),
    ],
)
def test_generated_comparisons_cannot_bypass_rj_arith(arch, kind, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for encoding in ('vopc', 'vop3'):
        source = (root / f'{encoding}_exec.cpp').read_text()
        checked = 0
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_compare<' in body:
                family = 'float'
            elif 'arith::execute_compare<' in body:
                family = (
                    'class'
                    if re.search(r'rj_arith_v_cmpx?_class_f', body)
                    else 'integer'
                )
            else:
                continue
            if family != kind:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            if '::execute_modifier_impl(' in match[0]:
                if 'dpp_new_result' in body:
                    assert 'if (wf.instruction_execution_failed())' in body
                if kind == 'float':
                    assert (
                        'exec_mask &= dpp_row_bank_mask_ & dpp_source_write_mask_;'
                        in body
                    )
            checked += 1
        assert checked == count


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_every_vopc_handler_routes_through_rj_arith(arch):
    source = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
        / 'vopc_exec.cpp'
    ).read_text()
    assert 'amdgpu::execute_v_cmp' not in source
    assert 'ROCJITSU_TRY_SIMD' not in source
