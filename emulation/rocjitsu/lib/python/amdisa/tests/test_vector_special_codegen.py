# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for vector-special execute code generation."""

from types import SimpleNamespace

from amdisa.codegen.execute.vector_special import gen_vector_permlane
from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Instruction, Operand


def test_permlane_uses_opsel_fi_and_bound_ctrl_bits():
    cpp = gen_vector_permlane(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'imm',
        cross=False,
        op_sel_expr='inst_.opsel',
    )

    assert 'bool fi = (inst_.opsel & 0x1u) != 0;' in cpp
    assert 'bool bound_ctrl = (inst_.opsel & 0x2u) != 0;' in cpp


def test_permlane_imm_selectors_are_four_bits_per_lane():
    cpp = gen_vector_permlane(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'imm',
        cross=False,
    )

    assert 'uint32_t sel = (sel_word >> ((sub & 7u) * 4u)) & 0xF;' in cpp
    assert 'sub * 2' not in cpp

    assert 'uint64_t exec = wf.exec();' in cpp
    assert '!(exec & (1ULL << lane))' in cpp
    assert 'bool src_active = (exec & (1ULL << src_lane)) != 0;' in cpp
    assert '!src_active && !fi' in cpp
    assert 'bound_ctrl' in cpp


def test_permlanex16_fetches_from_other_half_row():
    cpp = gen_vector_permlane(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'var',
        cross=True,
    )

    assert 'uint32_t row_base = lane & ~0x1Fu;' in cpp
    assert 'uint32_t half = (lane ^ 0x10u) & 0x10u;' in cpp
    assert 'uint32_t src_lane = row_base | half | sel;' in cpp
    assert 'sel ^ 0x10' not in cpp


def test_permlane_is_not_shared_across_profiles():
    assert 'vector_permlane16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_permlanex16' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_arch_local_execute_bodies_are_not_shared():
    assert 'scalar_getreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg_imm' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_movrel' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_gfx1250_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=SimpleNamespace(vop3p_opsel_fields=('opsel', 'opsel_hi')),
    )
    mov_b16 = Instruction(
        'V_MOV_B16',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
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
    not_b16 = Instruction(
        'V_NOT_B16',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
        ],
    )

    assert codegen._requires_arch_local_execute(mov_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(not_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
    assert not codegen._can_force_shared_simd_probe(mov_b16, 'ENC_VOP1')


def test_gfx1250_non_true16_simd_probe_can_still_be_shared():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=SimpleNamespace(vop3p_opsel_fields=('opsel', 'opsel_hi')),
    )
    mov_b32 = Instruction(
        'V_MOV_B32',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 1),
        ],
    )

    assert not codegen._requires_arch_local_execute(mov_b32, 'ENC_VOP1')
    assert codegen._can_force_shared_simd_probe(mov_b32, 'ENC_VOP1')


def test_gfx1250_true16_e32_dst_reg_uses_physical_vgpr():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='gfx1250')
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
