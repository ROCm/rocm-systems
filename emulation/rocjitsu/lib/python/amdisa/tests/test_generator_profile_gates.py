# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator
from amdisa.codegen.execute.vector_special import (
    gen_vector_div_scale,
    gen_vector_movrel,
)
from amdisa.codegen.execute.vector_cmp import (
    gen_vector_cmp,
    gen_vector_cmp_class,
)
from amdisa.isa_profile import Gfx1250Profile, Rdna4Profile


def test_simm64_literals_require_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32'])
    assert not codegen._supports_simm64_literal_operands()

    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32', 'OPR_SIMM64'])
    assert codegen._supports_simm64_literal_operands()


def test_movrel_uses_two_arg_resolver_without_vgpr_msb_indexing():
    src_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=False
    )
    dst_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'dst', uses_vgpr_msb_indexing=False
    )

    assert 'Isa::resolved_vgpr_offset(src0.opr_type_, src0.encoding_value_)' in src_body
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' not in src_body
    assert 'Isa::resolved_vgpr_offset(vdst.opr_type_, vdst.encoding_value_)' in dst_body
    assert 'Isa::resolved_vgpr_offset(wf, vdst.opr_type_' not in dst_body


def test_movrel_uses_wavefront_resolver_with_vgpr_msb_indexing():
    body = gen_vector_movrel(['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=True)
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' in body
    assert 'src0.vgpr_msb_role()' in body


def test_div_scale_writes_explicit_sdst_mask():
    body = gen_vector_div_scale(
        ['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'f32', is_vop3=True
    )

    assert 'if (wf.wf_size() <= 32)' in body
    assert 'sdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in body
    assert 'sdst.write_scalar64(wf, vcc)' in body
    assert 'wf.set_vcc(vcc)' not in body


def test_vector_cmp_writes_explicit_sdst_mask():
    body = gen_vector_cmp(['sdst'], ['src0', 'src1'], 't', 'u32', is_vop3=True)

    assert 'if (wf.wf_size() <= 32)' in body
    assert 'sdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in body
    assert 'sdst.write_scalar64(wf, vcc)' in body
    assert 'wf.set_vcc(vcc)' not in body


def test_vector_cmp_omits_redundant_mask_clears():
    body = gen_vector_cmp(['sdst'], ['src0', 'src1'], 'eq', 'u32', is_vop3=True)

    assert 'uint64_t vcc = 0;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_vector_cmp_class_writes_explicit_sdst_mask():
    body = gen_vector_cmp_class(
        ['sdst'], ['src0', 'src1'], 'f32', is_cmpx=False, is_vop3=True
    )

    assert 'if (wf.wf_size() <= 32)' in body
    assert 'sdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in body
    assert 'sdst.write_scalar64(wf, vcc)' in body
    assert 'wf.set_vcc(vcc)' not in body


def test_vector_cmp_class_omits_redundant_mask_clears():
    body = gen_vector_cmp_class(
        ['sdst'], ['src0', 'src1'], 'f32', is_cmpx=False, is_vop3=True
    )

    assert 'uint64_t vcc = 0;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_div_scale_uses_signed_tiny_exponent_threshold():
    body = gen_vector_div_scale(
        ['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'f32', is_vop3=True
    )

    assert 'exp2 <= -23' in body
    assert 'exp2 <= 23' not in body


def test_gfx1250_profile_enables_generator_backed_quirks():
    profile = Gfx1250Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert profile.use_hwreg_helpers
    assert profile.hwreg_wave_sched_mode_id == 26
    assert profile.generate_scaled_wmma_vop3px2
    assert profile.smem_address_uses_access_size


def test_rdna4_profile_keeps_gfx1250_quirks_disabled():
    profile = Rdna4Profile()

    assert not profile.uses_packed_16bit_e32_source_selectors
    assert not profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert not profile.use_hwreg_helpers
    assert profile.hwreg_wave_sched_mode_id is None
    assert not profile.generate_scaled_wmma_vop3px2
    assert not profile.smem_address_uses_access_size


def test_packed_16bit_source_gate_is_limited_to_e32_16bit_sources():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Gfx1250Profile())

    packed_src = SimpleNamespace(is_input=True, size=16, operand_type='OPR_SRC')
    wide_src = SimpleNamespace(is_input=True, size=32, operand_type='OPR_SRC')
    dst = SimpleNamespace(is_input=False, size=16, operand_type='OPR_VGPR')

    assert codegen._operand_uses_packed_16bit_source('ENC_VOP1', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOP2', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOPC', packed_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP3', packed_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', wide_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', dst)


def test_gfx1250_helper_blocks_emit_hwreg_and_scaled_wmma_hooks():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    hwreg = codegen._emit_hwreg_helpers()
    assert 'HW_REG_WAVE_SCHED_MODE = 26' in hwreg
    assert 'wf.wave_sched_mode_raw()' in hwreg
    assert 'wf.set_wave_sched_mode_raw' in hwreg
    assert '[[maybe_unused]] bool read_hwreg' in hwreg
    assert '[[maybe_unused]] bool write_hwreg' in hwreg

    assert codegen._supports_gfx1250_scaled_wmma_vop3px2()
    assert 'VWmmaScaleF3216x16x128F8f6f4Vop3px2' in (
        codegen._emit_gfx1250_scaled_wmma_vop3px2_class()
    )
    assert 'isWmmaScaleF32F8f6f4Vop3px2' in (
        codegen._emit_gfx1250_scaled_wmma_vop3px2_decoder_helpers()
    )


def test_gfx1250_vopd_template_uses_dx9_zero_and_fma(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )
    codegen.out_path = str(tmp_path)

    codegen.gen_vopd()
    cpp = (tmp_path / 'gfx1250' / 'vopd.cpp').read_text()

    assert 'case 3:\n              case 7:' not in cpp
    assert 'if (lhs == 0.0f || rhs == 0.0f)' in cpp
    fma_start = cpp.index('case kVopdFmaF32: {')
    fma_case = cpp[fma_start : cpp.index('case kVopdSubNcU32:', fma_start)]
    assert 'std::fma(std::bit_cast<float>(src0),' in fma_case
    assert 'std::bit_cast<float>(src1),' in fma_case
    assert 'std::bit_cast<float>(src2))' in fma_case


def test_bf16_mad_mix_half_updates_read_destination_operand():
    codegen = object.__new__(CodeGenerator)
    codegen.semantics = SimpleNamespace(
        instructions={
            'V_FMA_MIX_F32_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mix_f32_bf16'
            ),
            'V_FMA_MIXLO_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mixlo_bf16'
            ),
            'V_FMA_MIXHI_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mixhi_bf16'
            ),
        }
    )

    assert not codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIX_F32_BF16'))
    assert codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIXLO_BF16'))
    assert codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIXHI_BF16'))


def test_gfx1250_ds_atomic_routes_data_through_vgpr_resolver():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    expr = codegen._vgpr_base_expr('data0', role='Src1')
    assert 'resolved_vgpr_offset' in expr
    assert 'VgprMsbRole::Src1' in expr

    expr_cmp = codegen._vgpr_base_expr('data1', role='Src2')
    assert 'resolved_vgpr_offset' in expr_cmp
    assert 'VgprMsbRole::Src2' in expr_cmp


def test_rdna4_ds_atomic_uses_raw_encoding():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
    )

    expr = codegen._vgpr_base_expr('data0', role='Src1')
    assert 'resolved_vgpr_offset' not in expr
    assert 'inst_.data0' in expr


def test_ev124_125_arch_gating_in_generated_operand():
    import pathlib

    gen_root = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
    )

    rdna4_op = (gen_root / 'rdna4' / 'operand.cpp').read_text()
    assert 'if (ev == 124)\n    return 0u; // NULL' in rdna4_op
    assert 'if (ev == 125)\n    return wf.m0()' in rdna4_op

    cdna4_op = (gen_root / 'cdna4' / 'operand.cpp').read_text()
    assert 'if (ev == 124)\n    return wf.m0()' in cdna4_op
    assert 'ev == 125' not in cdna4_op.split('can_resolve_src_scalar')[1].split('}')[0]
