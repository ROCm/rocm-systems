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

    assert 'uint32_t sel = (sel_word >> (sub * 4)) & 0xF;' in cpp
    assert 'sub * 2' not in cpp


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
    codegen.isa_spec = SimpleNamespace(arch_name='gfx1250')
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

    assert codegen._requires_arch_local_execute(mov_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
