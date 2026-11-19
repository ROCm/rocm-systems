# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for vector-special execute code generation."""

from amdisa.codegen.execute.vector_special import (
    gen_vector_movrel,
    gen_vector_swaprel,
)
from types import SimpleNamespace

import re

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import Cdna5Profile, Rdna3Profile, Rdna4Profile


def test_relative_vgpr_ops_use_unsigned_packed_m0_fields():
    move = gen_vector_movrel(
        ['vdst'],
        ['src0'],
        'srcdst2',
        True,
        supports_dpp=True,
        supports_dpp8=True,
        supports_sdwa=True,
    )
    swap = gen_vector_swaprel(['vdst'], ['src0'], True)
    full_width = gen_vector_movrel(['vdst'], ['src0'], 'src', True)

    for cpp in (move, swap):
        assert 'wf.m0() & 0x3ffu' in cpp
        assert '(wf.m0() >> 16) & 0x3ffu' in cpp
        assert 'resolved_vgpr_offset(wf' in cpp
    assert 'amdgpu::arith::execute_move32' in move
    assert 'DppPlan rel_dpp_plan' in move
    assert 'apply_dpp(\n        rel_src, rel_dpp_plan' in move
    assert 'apply_dpp8(rel_src' in move
    assert 'stage_source(rel_src' in move
    assert 'rel_staged_src_binding(rel_src' in move
    assert 'amdgpu::arith::execute_swap<false>' in swap
    assert 'std::optional<uint32_t>' in move
    assert 'Operand rel_src' in move
    assert 'Operand rel_dst' in move
    assert 'rel_src_offset > 255u' not in swap
    assert 'rel_dst_offset > 255u' not in swap
    assert 'rel_src_valid ? rel_src_index : 0u' in move
    assert 'if (!rel_dst_valid) return;' in move
    assert 'amdgpu::relative_vgpr_index(rel_src_base, wf.m0(),' in full_width
    assert 'static_cast<int32_t>(wf.m0())' not in full_width


@pytest.mark.parametrize(
    ('generator', 'args'),
    [
        (gen_vector_movrel, (['vdst'], ['src0'], 'unknown')),
    ],
)
def test_unsupported_vector_special_operations_fail_during_generation(generator, args):
    with pytest.raises(ValueError):
        generator(*args)


def test_permlane_is_not_shared_across_profiles():
    assert 'vector_permlane16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_permlanex16' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_arch_local_execute_bodies_are_not_shared():
    assert 'scalar_getreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg_imm' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_movrel' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_swaprel' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_perm_pk16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_qsad' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_gfx1250_true16_e32_dst_reg_uses_physical_vgpr():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='cdna5', profile=Cdna5Profile())
    fmac_f16 = Instruction(
        'V_FMAC_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )
    add_f32 = Instruction(
        'V_ADD_F32',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )

    assert codegen._e32_true16_dst_reg_expr(fmac_f16, 'ENC_VOP2') == (
        '(inst_.vdst & 0x7fu)'
    )
    assert codegen._e32_true16_dst_reg_expr(add_f32, 'ENC_VOP2') == 'inst_.vdst'


def test_rdna4_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='rdna4', profile=Rdna4Profile())
    add_f16 = Instruction(
        'V_ADD_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )
    cndmask_b16 = Instruction(
        'V_CNDMASK_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('src2', 64, 'OPR_SREG', True, False, False, False, 3),
        ],
    )

    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert codegen._e32_true16_dst_reg_expr(add_f16, 'ENC_VOP2') == (
        '(inst_.vdst & 0x7fu)'
    )


def test_rdna3_vop3_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='rdna3', profile=Rdna3Profile())
    cndmask_b16 = Instruction(
        'V_CNDMASK_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('src2', 64, 'OPR_SREG', True, False, False, False, 3),
        ],
    )
    or_b16 = Instruction(
        'V_OR_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
        ],
    )
    add_f16 = Instruction(
        'V_ADD_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )

    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
