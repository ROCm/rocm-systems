# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator
from amdisa.codegen.execute.vector_special import (
    gen_vector_div_scale,
    gen_vector_movrel,
)
from amdisa.codegen.execute.matrix import gen_mfma
from amdisa.codegen.execute.vector_cmp import (
    gen_vector_add_co,
    gen_vector_cmp,
    gen_vector_cmp_class,
    gen_vector_cmpx,
)
from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import Gfx1250Profile, Rdna4Profile
from amdisa.semantics import InstructionSemantics


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


def test_vop3_cmp_writes_explicit_mask_width_for_wave_size():
    body = gen_vector_cmp(['vdst'], ['src0', 'src1'], 'eq', 'f32', is_vop3=True)

    assert 'if (wf.wf_size() <= 32)' in body
    assert 'vdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in body
    assert 'vdst.write_scalar64(wf, vcc)' in body
    assert 'wf.set_vcc(vcc)' not in body


def test_vop3_add_co_writes_explicit_sdst_mask_width_for_wave_size():
    body = gen_vector_add_co(['vdst', 'sdst'], ['src0', 'src1'], 'add', 'u32')

    assert 'if (wf.wf_size() <= 32)' in body
    assert 'sdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in body
    assert 'sdst.write_scalar64(wf, vcc)' in body
    assert 'wf.set_vcc(vcc)' not in body


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


def test_true16_vop3_integer_ops_do_not_use_whole_dword_simd_probe():
    assert simd_probe_line('v_mul_lo_u16_vop3') is None
    assert simd_probe_line('v_mad_u16_vop3') is None
    assert simd_probe_line('v_mad_i16_vop3') is None
    assert simd_probe_line('v_cmp_lt_i16_vop3') is None
    assert simd_probe_line('v_cmp_eq_u16_vop3') is None
    assert simd_probe_line('v_or_b16_vop3') is None
    assert simd_probe_line('v_add_nc_u32_vop3') is not None


def test_true16_vop3_cmp_uses_selected_source_halves():
    body = gen_vector_cmp(['vdst'], ['src0', 'src1'], 'lt', 'i16', is_vop3=True)

    assert 'uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
    assert body.index('uint32_t opsel = amdgpu::vop3_opsel(inst_);') < body.index(
        'for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)'
    )
    assert 'if (opsel & (1u << 0)) s0_raw >>= 16;' in body
    assert 'if (opsel & (1u << 1)) s1_raw >>= 16;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_true16_vop3_cmpx_hoists_opsel():
    opsel_line = 'uint32_t opsel = amdgpu::vop3_opsel(inst_);'
    loop_line = 'for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)'

    for dtype in ('i16', 'u16'):
        body = gen_vector_cmpx(['src0', 'src1'], 'lt', dtype, is_vop3=True)

        assert body.count(opsel_line) == 1
        assert body.index(opsel_line) < body.index(loop_line)
        assert 'if (opsel & (1u << 0)) s0_raw >>= 16;' in body
        assert 'if (opsel & (1u << 1)) s1_raw >>= 16;' in body


def test_gfx1250_wmma_f32_passes_c_modifier_to_accumulator_helper():
    inst = Instruction('V_WMMA_F32_16X16X32_F16', 'ENC_VOP3P', 0, [])
    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'gfx1250')

    assert 'amdgpu::exec_wmma_f32_16x16x32_f16(' in body
    assert 'amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi)' in body


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


def test_ds_swizzle_generator_distinguishes_ds_and_vds_sources():
    def body_for(enc_name: str) -> str:
        codegen = object.__new__(CodeGenerator)
        codegen.isa_spec = SimpleNamespace(
            arch_name='gfx1250',
            profile=Gfx1250Profile(),
            inst_encodings=[],
            encoding_map={},
        )
        src_name = 'addr' if enc_name == 'ENC_VDS' else 'data0'
        inst = Instruction(
            'DS_SWIZZLE_B32',
            enc_name,
            0,
            [
                Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
                Operand(src_name, 32, 'OPR_VGPR', True, False, False, False, 1),
            ],
        )
        sem = InstructionSemantics('DS_SWIZZLE_B32', 'ds_swizzle')
        return codegen._gen_execute_body(inst, sem, enc_name)

    vds_body = body_for('ENC_VDS')
    ds_body = body_for('ENC_DS')

    assert 'src_data[i] = cu.read_vgpr(vb + inst_.addr, i);' in vds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.data0, i);' not in vds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.data0, i);' in ds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.addr, i);' not in ds_body
    assert '2u * (lane & 0x3u)' in vds_body
    assert '2u * (lane & 0x3u)' in ds_body


def test_packed_16bit_source_gate_is_limited_to_e32_16bit_sources():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Gfx1250Profile())

    packed_src = SimpleNamespace(
        is_input=True, is_output=False, size=16, operand_type='OPR_SRC'
    )
    packed_vgpr_src = SimpleNamespace(
        is_input=True, is_output=False, size=16, operand_type='OPR_VGPR'
    )
    wide_src = SimpleNamespace(
        is_input=True, is_output=False, size=32, operand_type='OPR_SRC'
    )
    dst = SimpleNamespace(
        name='vdst', is_input=False, is_output=True, size=16, operand_type='OPR_VGPR'
    )

    assert codegen._operand_uses_packed_16bit_source('ENC_VOP1', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOP2', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOPC', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOPC', packed_vgpr_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP3', packed_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', wide_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', dst)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOP2', dst, reads_dst=True)


def test_gfx1250_generated_operand_merges_packed_16bit_destinations():
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
        / 'gfx1250'
    )
    operand_cpp = (gen_root / 'operand.cpp').read_text()
    operand_h = (gen_root / 'operand.h').read_text()

    assert 'if (ev >= 0 && ev <= 127)' in operand_cpp
    assert 'packed->shift ? 0x0000ffffu : 0xffff0000u' in operand_cpp
    assert 'amdgpu::apply_gpr_idx(wf, off, false)' in operand_cpp
    assert 'void Operand::write_lane_chunk' in operand_cpp
    assert 'void write_lane_chunk(amdgpu::Wavefront &wf' in operand_h


def test_gfx1250_generated_vop2_uses_packed_16bit_vsrc1():
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
        / 'gfx1250'
    )
    vop2_cpp = (gen_root / 'vop2.cpp').read_text()

    assert 'vsrc1(16, OperandType::OPR_VGPR,' in vop2_cpp
    assert (
        'static_cast<unsigned short>(reinterpret_cast<const OpEncoding *>(inst)->vsrc1), true)'
        in vop2_cpp
    )


def test_gfx1250_generated_vop2_fmac_f16_reads_packed_vdst():
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
        / 'gfx1250'
    )
    vop2_cpp = (gen_root / 'vop2.cpp').read_text()

    start = vop2_cpp.index('VFmacF16Vop2::VFmacF16Vop2')
    end = vop2_cpp.index('void VFmacF16Vop2::execute_impl', start)
    ctor = vop2_cpp[start:end]

    assert 'vdst(16, OperandType::OPR_VGPR,' in ctor
    compact_ctor = ''.join(ctor.split())
    assert (
        'vdst(16,OperandType::OPR_VGPR,static_cast<unsignedshort>('
        'reinterpret_cast<constOpEncoding*>(inst)->vdst),true)' in compact_ctor
    )


def test_gfx1250_generated_vop3_mad_u16_uses_true16_helpers():
    import pathlib

    execute_shared = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'shared'
        / 'execute_shared.h'
    ).read_text()

    start = execute_shared.index('inline void execute_v_mad_u16_vop3')
    end = execute_shared.index('inline void execute_v_mad_u32_u16_vop3', start)
    body = execute_shared[start:end]

    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT' not in body
    assert 'uint32_t opsel = vop3_opsel(inst.inst_);' in body
    assert 'read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(inst.src1, wf, lane, opsel, 1)' in body
    assert 'read_vop3_true16_src(inst.src2, wf, lane, opsel, 2)' in body
    assert 'write_vop3_true16_dst(inst.vdst, wf, lane, opsel, result)' in body


def test_gfx1250_generated_vop3_lshrrev_b16_uses_true16_helpers():
    import pathlib

    execute_shared = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'shared'
        / 'execute_shared.h'
    ).read_text()

    start = execute_shared.index('inline void execute_v_lshrrev_b16_vop3')
    end = execute_shared.index('inline void execute_v_lshrrev_b32_vop2', start)
    body = execute_shared[start:end]

    assert 'uint32_t opsel = vop3_opsel(inst.inst_);' in body
    assert 'read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(inst.src1, wf, lane, opsel, 1)' in body
    assert 'write_vop3_true16_dst(inst.vdst, wf, lane, opsel, result)' in body
    assert 'inst.vdst.write_lane' not in body


def test_gfx1250_generated_fp8_vop3_shared_byte_select_uses_inst_member():
    import pathlib

    execute_shared = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'shared'
        / 'execute_shared.h'
    ).read_text()

    start = execute_shared.index('inline void execute_v_cvt_f32_fp8_vop3')
    end = execute_shared.index('inline void execute_v_cvt_f32_i32_vop1', start)
    body = execute_shared[start:end]

    assert 'amdgpu::vop3_opsel(inst.inst_)' in body
    assert 'amdgpu::vop3_fp8_decode_e5m3(inst)' in body
    assert 'util::fp8_e5m3_to_f32' in body
    assert 'util::fp8_e4m3_to_f32' in body
    assert 'amdgpu::vop3_opsel(inst_)' not in body
    assert 'amdgpu::vop3_fp8_decode_e5m3(inst_)' not in body


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
    assert 'VWmmaScaleF32Vop3px2' in (codegen._emit_gfx1250_scaled_wmma_vop3px2_class())
    assert 'isWmmaScaleF32Vop3px2' in (
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


def test_gfx1250_flat_cmpswap_payload_width_is_independent_of_element_width():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    b32 = SimpleNamespace(
        name='GLOBAL_ATOMIC_CMPSWAP_B32',
        operation='cmpswap',
        elem_size=4,
        num_elems=2,
    )
    b32_body = codegen._gen_flat_atomic([], [], b32)
    assert 'd->is_load = amdgpu::gfx12_atomic_returns(inst_.th);' in b32_body
    assert 'd->elem_size = 4;' in b32_body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in b32_body
    assert 'data_base + 1' in b32_body

    b64 = SimpleNamespace(
        name='GLOBAL_ATOMIC_CMPSWAP_B64',
        operation='cmpswap',
        elem_size=8,
        num_elems=4,
    )
    b64_body = codegen._gen_flat_atomic([], [], b64)
    assert 'd->elem_size = 8;' in b64_body
    assert 'd->store_data.resize(wf.wf_size() * 16);' in b64_body
    assert 'data_base + 3' in b64_body


def test_gfx1250_flat_u64_atomic_payload_width_uses_two_dwords():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    u64 = SimpleNamespace(
        name='GLOBAL_ATOMIC_ADD_U64',
        operation='add',
        elem_size=8,
        num_elems=2,
    )
    body = codegen._gen_flat_atomic([], [], u64)
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in body
    assert 'data_base + 1' in body


def test_gfx1250_buffer_cmpswap_payload_width_is_independent_of_element_width():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    b64 = SimpleNamespace(
        name='BUFFER_ATOMIC_CMPSWAP_B64',
        operation='cmpswap',
        elem_size=8,
        num_elems=4,
    )
    body = codegen._gen_buffer_atomic([], [], b64)
    assert 'd->is_load = amdgpu::gfx12_atomic_returns(inst_.th);' in body
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 16);' in body
    assert 'data_base + 3' in body


def test_gfx1250_buffer_u64_atomic_payload_width_uses_two_dwords():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    u64 = SimpleNamespace(
        name='BUFFER_ATOMIC_ADD_U64',
        operation='add',
        elem_size=8,
        num_elems=2,
    )
    body = codegen._gen_buffer_atomic([], [], u64)
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in body
    assert 'data_base + 1' in body


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
