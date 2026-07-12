# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for vector-special execute code generation."""

from types import SimpleNamespace

from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.codegen.execute.vector_special import (
    gen_vector_bitop3,
    gen_vector_cvt_pk,
    gen_vector_div_fixup,
    gen_vector_mad_32_16,
    gen_vector_permlane,
)
from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import Gfx1250Profile, Rdna3Profile, Rdna4Profile


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


def test_vop3_f16_simd_probes_split_true16_from_generic():
    assert simd_probe_line('v_add_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_BINARY_F16'
    )
    assert simd_probe_line('v_add_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16'
    )
    assert simd_probe_line('v_rcp_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16'
    )
    assert simd_probe_line('v_rcp_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16'
    )
    assert simd_probe_line('v_fma_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16'
    )
    assert simd_probe_line('v_fma_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16'
    )
    assert simd_probe_line('v_div_fixup_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16'
    )
    assert simd_probe_line('v_div_fixup_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16'
    )
    assert simd_probe_line('v_fmac_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP16'
    )
    assert simd_probe_line('v_fmac_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_FMAC_VOP3_TRUE16_FP16'
    )
    assert simd_probe_line('v_cmp_class_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_CLASS_B32'
    )
    assert simd_probe_line('v_cmp_class_f16_vop3', true16_vop3=True).startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_CLASS_TRUE16_B32'
    )
    assert simd_probe_line('v_cvt_f32_f16_vop3') == (
        '  ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3();'
    )
    assert simd_probe_line('v_cvt_f32_f16_vop3', true16_vop3=True) == (
        '  ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3_TRUE16();'
    )


def test_true16_vop3_simd_probe_leaves_unsupported_b16_scalar():
    assert simd_probe_line('v_not_b16_vop3', true16_vop3=True) is None
    assert simd_probe_line('v_cndmask_b16_vop3', true16_vop3=True) is None


def test_gfx1250_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
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

    assert codegen._requires_arch_local_execute(mov_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(not_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert not codegen._can_force_shared_simd_probe(mov_b16, 'ENC_VOP1')


def test_gfx1250_non_true16_simd_probe_can_still_be_shared():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
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
    codegen.isa_spec = SimpleNamespace(arch_name='gfx1250', profile=Gfx1250Profile())
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


def test_gfx1250_bitop3_b16_uses_true16_helpers():
    body = gen_vector_bitop3(
        ['vdst'], ['src0', 'src1', 'src2'], 'b16', true16_opsel='inst_.opsel'
    )

    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in body
    assert 'read_vop3_true16_src(src2, wf, lane, inst_.opsel, 2)' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, inst_.opsel, result, true)' in body


def test_vop3_mad_32_16_uses_true16_sources_for_src0_src1_only():
    body = gen_vector_mad_32_16(
        ['vdst'], ['src0', 'src1', 'src2'], 'u32', is_vop3=True, opsel='inst_.opsel'
    )

    assert 'uint32_t opsel = inst_.opsel;' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in body
    assert 'read_vop3_true16_src(src2' not in body
    assert 'uint32_t s2 = src2.read_lane(wf, lane);' in body


def test_vop3_div_fixup_f16_uses_true16_sources_and_destination():
    body = gen_vector_div_fixup(['vdst'], ['src0', 'src1', 'src2'], 'f16', is_vop3=True)

    assert 'uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
    assert 'util::f16_to_f32(static_cast<uint16_t>(' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in body
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in body
    assert 'uint32_t result_bits = util::f32_to_f16(result);' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result_bits, true)' in body
    assert 'std::bit_cast<float>(src0.read_lane' not in body


def test_vop3_pack_and_pknorm_f16_use_true16_source_halves():
    pack = gen_vector_cvt_pk(
        ['vdst'],
        ['src0', 'src1'],
        'vector_pack_b32_f16',
        None,
        opsel='inst_.opsel',
        dtype='f16',
        is_vop3=True,
    )
    pknorm = gen_vector_cvt_pk(
        ['vdst'],
        ['src0', 'src1'],
        'vector_cvt_pknorm',
        'i16',
        opsel='inst_.opsel',
        dtype='f16',
        is_vop3=True,
    )

    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in pack
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in pack
    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in pknorm
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in pknorm
    assert 'float s0 = std::bit_cast<float>' not in pknorm
    assert 'auto cvt_i16 = [](float f) -> int16_t {' in pknorm


def test_true16_special_vop3_simd_routes_use_true16_glue():
    assert simd_probe_line('v_mad_u32_u16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_SRC01'
    )
    assert simd_probe_line('v_mad_i32_i16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_SRC01'
    )
    assert simd_probe_line('v_pack_b32_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC'
    )
    assert simd_probe_line('v_cvt_pk_norm_i16_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC'
    )
    assert simd_probe_line('v_div_fixup_f16_vop3').startswith(
        '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16'
    )
