# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for vector-special execute code generation."""

from types import SimpleNamespace

import hashlib
import re

import pytest

from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.codegen.execute.vector_special import (
    _TRIG_PREOP_CHUNK_BITS,
    _TRIG_PREOP_TWO_OVER_PI_CHUNKS,
    _TRIG_PREOP_VALID_BITS,
    gen_vector_bitop3,
    gen_vector_cvt_pk,
    gen_vector_div_fixup,
    gen_vector_mad_32_16,
    gen_vector_movrel,
    gen_vector_mullit,
    gen_vector_perm_pk16,
    gen_vector_permlane,
    gen_vector_qsad,
    gen_vector_sat_pack,
    gen_vector_swaprel,
    gen_vector_trig_preop,
)
from amdisa.codegen.config import CodegenConfig
from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import Cdna5Profile, Rdna3Profile, Rdna4Profile


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
    assert 'read_lane(rel_src, lane)' in move
    assert 'DppPlan rel_dpp_plan' in move
    assert 'apply_dpp(\n        rel_src, rel_dpp_plan' in move
    assert 'apply_dpp8(rel_src' in move
    assert 'stage_source(rel_src' in move
    assert 'rel_staged_src_binding(rel_src' in move
    assert 'write_lane(rel_src, lane, dst_value)' in swap
    assert 'std::optional<uint32_t>' in move
    assert 'Operand rel_src' in move
    assert 'Operand rel_dst' in move
    assert 'rel_src_offset > 255u' not in swap
    assert 'rel_dst_offset > 255u' not in swap
    assert 'rel_src_valid ? rel_src_index : 0u' in move
    assert 'if (!rel_dst_valid) return;' in move
    assert 'wf.m0() <= 1023u' in full_width
    assert 'static_cast<int32_t>(wf.m0())' not in full_width


def test_saturating_pack_codegen_covers_all_narrowing_modes():
    u8 = gen_vector_sat_pack(['vdst'], ['src0'], 'u8_i16')
    i4 = gen_vector_sat_pack(['vdst'], ['src0'], 'i4_i8')
    u4 = gen_vector_sat_pack(['vdst'], ['src0'], 'u4_u8')

    assert 'std::clamp(lo, 0, 255)' in u8
    assert 'std::clamp(value, -8, 7)' in i4
    assert 'std::min((raw >> (i * 8)) & 0xffu, 15u)' in u4

    with pytest.raises(ValueError, match='unsupported saturating-pack operation'):
        gen_vector_sat_pack(['vdst'], ['src0'], 'unknown')


def test_perm_and_qsad_codegen_cover_multiword_results():
    perm = gen_vector_perm_pk16(['vdst'], ['src0', 'src1', 'src2'], 'b6', True)
    qsad = gen_vector_qsad(['vdst'], ['src0', 'src1', 'src2'], 'sad_pk_u16', True)
    mqsad = gen_vector_qsad(['vdst'], ['src0', 'src1', 'src2'], 'msad_u32', True)

    assert 'uint32_t packed[3]' in perm
    assert 'source_bit = index * 6u' in perm
    assert 'Operand dst_word = vdst;' in perm
    assert 'dst_word.encoding_value_' in perm
    assert 'read_lane64(src1, lane)' in perm
    assert '(window + byte) * 8' in qsad
    assert 'write_lane64(vdst, lane, packed_result)' in qsad
    assert 'value & 0xffffu' in qsad
    assert 'if (b != 0)' in mqsad
    assert 'accum[window] = amdgpu::RegisterAccess(wf).read_lane' in mqsad
    assert 'if (std::optional<uint64_t> constant = src2.const_value())' in mqsad
    assert 'accum[window] + sum' in mqsad


@pytest.mark.parametrize(
    ('generator', 'args'),
    [
        (gen_vector_movrel, (['vdst'], ['src0'], 'unknown')),
        (gen_vector_perm_pk16, (['vdst'], ['src0', 'src1', 'src2'], 'unknown', True)),
        (gen_vector_qsad, (['vdst'], ['src0', 'src1', 'src2'], 'unknown', True)),
    ],
)
def test_unsupported_vector_special_operations_fail_during_generation(generator, args):
    with pytest.raises(ValueError):
        generator(*args)


def test_mullit_and_trig_preop_preserve_special_fp_rules():
    mullit = gen_vector_mullit(
        ['vdst'], ['src0', 'src1', 'src2'], is_vop3=True, has_abs=True
    )
    trig = gen_vector_trig_preop(['vdst'], ['src0', 'src1'], is_vop3=True, has_abs=True)

    assert 's1 == -std::numeric_limits<float>::max()' in mullit
    assert 's1 == -std::numeric_limits<float>::infinity()' in mullit
    assert 's2 <= 0.0f || std::isnan(s2)' in mullit
    assert '(s0 == 0.0f || s1 == 0.0f) ? 0.0f' in mullit
    assert 'amdgpu::clamp_floating_result(result, wf)' in mullit
    assert '0xA2F983u' in trig
    assert 'kTwoOverPiChunks' in trig
    assert 'kChunkBits = 24u' in trig
    assert 'kValidBits = 1201u' in trig
    assert 'selector * 53u' in trig
    assert 'if (bit >= kValidBits) return 0' in trig
    assert 'exponent > 1077u' in trig
    assert 'exponent >= 1968u' in trig
    assert 'scale_u53_f64_rtz(segment, scale)' in trig
    assert 'amdgpu::fp_mode::effective_omod' in trig
    assert 'amdgpu::fp_mode::finish_f64' in trig
    assert re.search(r'finish_f64\([^;]*effective_omod', trig, flags=re.DOTALL)
    assert not re.search(r'finish_f64\([^;]*inst_\.omod', trig, flags=re.DOTALL)


def test_trig_preop_two_over_pi_table_integrity():
    table_bytes = b''.join(
        chunk.to_bytes(3, byteorder='big') for chunk in _TRIG_PREOP_TWO_OVER_PI_CHUNKS
    )

    assert _TRIG_PREOP_CHUNK_BITS == 24
    assert _TRIG_PREOP_VALID_BITS == 1201
    assert len(_TRIG_PREOP_TWO_OVER_PI_CHUNKS) == 51
    assert (
        hashlib.sha256(table_bytes).hexdigest()
        == '6945fb5ae75f6a3e8dec95553821e5886324e9e96181eb7944d828f508eee5b9'
    )


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
    assert 'vector_swaprel' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_perm_pk16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_qsad' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_vop3_f16_simd_probes_split_true16_from_generic():
    add_generic = simd_probe_line('v_add_f16_vop3')
    add_true16 = simd_probe_line('v_add_f16_vop3', true16_vop3=True)
    rcp_generic = simd_probe_line('v_rcp_f16_vop3')
    rcp_true16 = simd_probe_line('v_rcp_f16_vop3', true16_vop3=True)

    assert 'if (wf.fp16_ovfl())' in add_generic
    assert 'util::f32_to_f16_ovfl_simd' in add_generic
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_F16' in add_generic
    assert 'if (wf.fp16_ovfl())' in add_true16
    assert 'util::f32_to_f16_ovfl_simd' in add_true16
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16' in add_true16
    assert 'if (!wf.fp16_ovfl())' not in rcp_generic
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16' in rcp_generic
    assert 'if (!wf.fp16_ovfl())' not in rcp_true16
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16' in rcp_true16
    fma_generic = simd_probe_line('v_fma_f16_vop3')
    fma_true16 = simd_probe_line('v_fma_f16_vop3', true16_vop3=True)
    assert fma_generic == '  ROCJITSU_TRY_SIMD_FMA_VOP3_FP16();'
    assert fma_true16 == '  ROCJITSU_TRY_SIMD_FMA_VOP3_TRUE16_FP16();'
    div_fixup = simd_probe_line('v_div_fixup_f16_vop3')
    assert 'if (!wf.fp16_ovfl())' not in div_fixup
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16' in div_fixup
    div_fixup_true16 = simd_probe_line('v_div_fixup_f16_vop3', true16_vop3=True)
    assert 'if (!wf.fp16_ovfl())' not in div_fixup_true16
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16' in div_fixup_true16
    fmac_generic = simd_probe_line('v_fmac_f16_vop3')
    fmac_true16 = simd_probe_line('v_fmac_f16_vop3', true16_vop3=True)
    assert fmac_generic == '  ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_FP16();'
    assert fmac_true16 == '  ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_TRUE16_FP16();'
    assert simd_probe_line('v_fmac_f16_vop2') == '  ROCJITSU_TRY_SIMD_VOP2_FMAC_F16();'
    assert simd_probe_line('v_fmamk_f16_vop2') == (
        '  ROCJITSU_TRY_SIMD_VOP2_FMA_F16_MULTIPLY_LITERAL('
        'amdgpu::RegisterAccess(wf).read_scalar(inst.simm32));'
    )
    assert simd_probe_line('v_fmaak_f16_vop2') == (
        '  ROCJITSU_TRY_SIMD_VOP2_FMA_F16_ADD_LITERAL('
        'amdgpu::RegisterAccess(wf).read_scalar(inst.simm32));'
    )
    assert simd_probe_line('v_fmac_f64_vop2') == '  ROCJITSU_TRY_SIMD_VOP2_FMA_F64();'
    assert simd_probe_line('v_fma_f64_vop3') == '  ROCJITSU_TRY_SIMD_FMA_VOP3_FP64();'
    assert (
        simd_probe_line('v_fmac_f64_vop3')
        == '  ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_FP64();'
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
    codegen.config = CodegenConfig()
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
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
    codegen.config = CodegenConfig()
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
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
    assert 'uint32_t s2 = amdgpu::RegisterAccess(wf).read_lane(src2, lane);' in body


def test_vop3_div_fixup_f16_uses_true16_sources_and_destination():
    body = gen_vector_div_fixup(['vdst'], ['src0', 'src1', 'src2'], 'f16', is_vop3=True)

    assert 'uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
    assert 'util::f16_to_f32(static_cast<uint16_t>(' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in body
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in body
    assert (
        'uint32_t result_bits = util::f32_to_f16_mode(result, wf.fp16_ovfl());' in body
    )
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
    assert 'util::round_to_nearest_even(std::clamp(f * 32767.0f' in pknorm


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
    div_fixup = simd_probe_line('v_div_fixup_f16_vop3')
    assert 'if (!wf.fp16_ovfl())' not in div_fixup
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16' in div_fixup
