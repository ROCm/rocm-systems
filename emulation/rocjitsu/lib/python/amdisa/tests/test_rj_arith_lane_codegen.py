# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


from pathlib import Path


import pytest


from amdisa.codegen.execute.rj_arith_lane import LANE_NAMES, gen_lane


def context(arch, name, dst, src):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name='',
        is_vop3=False,
        has_abs=False,
    )


@pytest.mark.parametrize('name', sorted(LANE_NAMES))
def test_lane_family_uses_rj_arith_without_optional_state(name):
    count = (
        3
        if name
        in (
            'V_PERMLANE16_B32',
            'V_PERMLANEX16_B32',
            'V_PERMLANE_BCAST_B32',
            'V_PERMLANE_UP_B32',
            'V_PERMLANE_DOWN_B32',
            'V_PERMLANE_XOR_B32',
            'V_PERM_PK16_B4_U4',
            'V_PERM_PK16_B6_U4',
            'V_PERM_PK16_B8_U4',
        )
        else (
            2
            if '_VAR_' in name
            or name in ('V_READLANE_B32', 'V_WRITELANE_B32', 'V_PERMLANE_IDX_GEN_B32')
            else 1
        )
    )
    sources = ['a', 'b', 'c'][:count]
    for arch in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'):
        destinations = (
            ['d', 'a'] if name.startswith('V_PERMLANE') and '_SWAP_' in name else ['d']
        )
        body = gen_lane(context(arch, name, destinations, sources))
        if name == 'V_PERMLANE_IDX_GEN_B32' and arch != 'cdna5':
            assert body is None
            continue
        assert 'amdgpu::arith::execute_' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
    assert gen_lane(context('cdna2', name, ['d'], sources)) is None
    assert gen_lane(context('rdna4', name, [], sources)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_lane_families_use_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    body = '\n'.join(
        (root / f'{encoding}_exec.cpp').read_text() for encoding in ('vop1', 'vop3')
    )
    for helper, count in [
        ('execute_read_lane<true>', 2),
        ('execute_read_lane<false>', 1),
        ('execute_write_lane', 1),
        ('execute_swap<true>', 1),
        ('execute_swap<false>', 2),
        ('execute_permlane64', 1),
        ('execute_permlane16<', 2 if arch == 'rdna3' else 4),
    ]:
        assert body.count('amdgpu::arith::' + helper) == count


@pytest.mark.parametrize('arch,count', [('cdna4', 4), ('cdna5', 2)])
def test_generated_row_swaps_use_two_destination_bridge(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    body = '\n'.join(p.read_text() for p in root.glob('*exec*.cpp'))
    assert body.count('amdgpu::arith::execute_permlane_swap<') == count


def test_gfx1250_group_permutations_use_arithmetic_bridge():
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5'
    )
    body = (root / 'vop3_exec_data.cpp').read_text()
    for kind in ('Bcast', 'Up', 'Down', 'Xor'):
        assert f'execute_group_permutation<amdgpu::ValuPermutation::{kind}>' in body


def test_gfx1250_packed_permutations_use_arithmetic_bridge():
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5'
    )
    body = (root / 'vop3_exec_data.cpp').read_text()
    for bits in (4, 6, 8):
        assert f'execute_packed_permutation<{bits}>' in body
