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
from amdisa.isa_profile import (
    Gfx1250Profile,
    Rdna3_5Profile,
    Rdna3Profile,
    Rdna4Profile,
)
from amdisa.parser import Parser
from amdisa.semantics import InstructionSemantics


def test_simm64_literals_require_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32'])
    assert not codegen._supports_simm64_literal_operands()

    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32', 'OPR_SIMM64'])
    assert codegen._supports_simm64_literal_operands()


def test_vop_dpp8_support_is_detected_from_machine_inst_structs():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1Inst'),
            SimpleNamespace(fmt_enc_name='Vop2VopDpp16'),
        ]
    )
    assert not codegen._supports_vop_dpp8()

    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1Inst'),
            SimpleNamespace(fmt_enc_name='Vop1VopDpp8'),
        ]
    )
    assert codegen._supports_vop_dpp8()


def test_vop3_sdst_dpp_support_is_detected_from_machine_inst_structs():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1VopDpp16', enc_name='VOP1_VOP_DPP16'),
            SimpleNamespace(
                fmt_enc_name='Vop3SdstEncVopDpp16', enc_name='VOP3_SDST_ENC_VOP_DPP16'
            ),
            SimpleNamespace(
                fmt_enc_name='Vop3SdstEncVopDpp8', enc_name='VOP3_SDST_ENC_VOP_DPP8'
            ),
        ]
    )

    assert codegen._vop_dpp_struct_names('VOP3_SDST_ENC') == (
        'Vop3SdstEncVopDpp16MachineInst',
        'Vop3SdstEncVopDpp8MachineInst',
    )
    assert codegen._supports_vop_dpp_encoding('VOP3_SDST_ENC')


def test_rdna4_parser_injects_s_waitcnt_compat():
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[6]
    spec = Parser(
        str(
            repo_root
            / 'shared'
            / 'machine-readable-isa'
            / 'isa'
            / 'amdgpu_isa_rdna4.xml'
        ),
        Rdna4Profile(),
    ).parse()

    sopp = spec.encoding_map['ENC_SOPP']
    assert any(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts)

    dt_ptr = sopp.primary_dt_ptrs[9]
    dte = spec.primary_decode_table[dt_ptr]
    assert dte.sub_decode_funcs[9] == 'decodeSWaitcntSopp'


def _fake_sopp_parser(arch_name: str, *, sub_decode_funcs: list[str | None] | None):
    parser = object.__new__(Parser)
    sopp = SimpleNamespace(
        insts=[
            Instruction('S_WAIT_ALU', 'ENC_SOPP', 8, []),
            Instruction('S_WAIT_IDLE', 'ENC_SOPP', 10, []),
        ],
        primary_dt_ptrs=[-1] * 10,
    )
    sopp.primary_dt_ptrs[9] = 0
    dte = SimpleNamespace(sub_decode_funcs=sub_decode_funcs, decode_func=None)
    parser.isa_spec = SimpleNamespace(
        arch_name=arch_name,
        encoding_map={'ENC_SOPP': sopp},
        primary_decode_table=[dte],
    )
    return parser, sopp, dte


def test_gfx1250_parser_injects_s_waitcnt_compat_once_in_opcode_order():
    parser, sopp, dte = _fake_sopp_parser('gfx1250', sub_decode_funcs=[None] * 16)

    parser._inject_compat_insts()
    parser._inject_compat_insts()

    assert [(inst.name, inst.opcode) for inst in sopp.insts] == [
        ('S_WAIT_ALU', 8),
        ('S_WAITCNT', 9),
        ('S_WAIT_IDLE', 10),
    ]
    assert (
        sum(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts) == 1
    )
    assert dte.sub_decode_funcs[9] == 'decodeSWaitcntSopp'


def test_s_waitcnt_compat_injection_can_patch_direct_decode_entry():
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=None)

    parser._inject_compat_insts()

    assert any(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts)
    assert dte.decode_func == 'decodeSWaitcntSopp'


def test_s_waitcnt_compat_injection_skips_untargeted_arch():
    parser, sopp, dte = _fake_sopp_parser('rdna3', sub_decode_funcs=[None] * 16)

    parser._inject_compat_insts()

    assert [(inst.name, inst.opcode) for inst in sopp.insts] == [
        ('S_WAIT_ALU', 8),
        ('S_WAIT_IDLE', 10),
    ]
    assert dte.sub_decode_funcs[9] is None


def test_rdna4_s_waitcnt_compat_uses_gfx11_layout():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    inst = Instruction(
        'S_WAITCNT',
        'ENC_SOPP',
        9,
        [Operand('simm16', 16, 'OPR_WAITCNT', True, False, False, True, 1)],
    )
    sem = InstructionSemantics('S_WAITCNT', 'waitcnt')

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOPP')

    assert 'uint8_t exp = imm & 0x7;' in body
    assert 'uint8_t lgkm = (imm >> 4) & 0x3F;' in body
    assert 'uint8_t vm = (imm >> 10) & 0x3F;' in body


def test_rdna4_s_waitcnt_compat_formats_with_gfx11_layout():
    decode = Rdna4Profile().waitcnt_decode

    assert 'uint32_t expcnt = encoding_value_ & 0x7;' in decode
    assert 'uint32_t lgkmcnt = (encoding_value_ >> 4) & 0x3F;' in decode
    assert 'uint32_t vmcnt = (encoding_value_ >> 10) & 0x3F;' in decode


def test_readlane_family_uses_source_vgpr_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_VGPR', 'OPR_SRC_VGPR'])
    sem = InstructionSemantics('V_READFIRSTLANE_B32', 'vector_readfirstlane')
    src0 = Operand('src0', 32, 'OPR_VGPR', True, False, False, False, 0)
    vdst = Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0)

    assert codegen._constructor_operand_type(sem, src0) == 'OPR_SRC_VGPR'
    assert codegen._constructor_operand_type(sem, vdst) == 'OPR_VGPR'


def test_readlane_family_decodes_lane_selector_as_scalar_value():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna3',
        profile=Rdna3Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        Operand('vdst', 32, 'OPR_SREG', False, True, False, False, 0),
        Operand('src0', 32, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 32, 'OPR_SSRC_LANESEL', True, False, False, False, 2),
    ]
    inst = Instruction('V_READLANE_B32', 'ENC_VOP3', 0, operands)
    sem = InstructionSemantics('V_READLANE_B32', 'vector_readlane')

    body = codegen._gen_execute_body(inst, sem, 'ENC_VOP3')

    assert 'uint32_t lane = src1.read_scalar(wf);' in body
    assert 'src1.encoding_value_' not in body


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


def test_rdna_wmma_uses_arch_specific_wave32_operand_layout():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]
    inst = Instruction('V_WMMA_F32_16X16X16_F16', 'ENC_VOP3P', 0, operands)

    for arch in ('rdna3', 'rdna3_5'):
        body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], arch)

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert (
            'amdgpu::exec_gfx11_wmma_f32(cu, wf.wf_size(), 16, 16, 16, 16, dst,' in body
        )
        assert 'amdgpu::exec_f32(cu, 16, 16, 16' not in body

    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')
    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'amdgpu::exec_wmma_f32(cu, 16, 16, 16, 16, dst,' in body
    assert 'amdgpu::exec_f32(cu, 16, 16, 16' not in body


def test_rdna_wmma_i32_iu8_uses_arch_specific_wave32_operand_layout():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 128, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]
    inst = Instruction('V_WMMA_I32_16X16X16_IU8', 'ENC_VOP3P', 0, operands)

    for arch in ('rdna3', 'rdna3_5'):
        body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], arch)

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert 'auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i8' in body
        assert 'auto extract_b = (inst_.neg & 0x2u) ? amdgpu::extract_i8' in body
        assert (
            'amdgpu::exec_gfx11_wmma_i32(cu, wf.wf_size(), 16, 16, 16, 8, dst,' in body
        )
        assert 'amdgpu::exec_i32_mixed(cu, 16, 16, 16' not in body

    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')
    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i8' in body
    assert 'auto extract_b = (inst_.neg & 0x2u) ? amdgpu::extract_i8' in body
    assert 'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 8, dst,' in body
    assert 'amdgpu::exec_i32_mixed(cu, 16, 16, 16' not in body


def test_rdna_wmma_f16_bf16_use_arch_specific_wave32_dispatch():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]

    for dtype in ('F16', 'BF16'):
        inst = Instruction(
            f'V_WMMA_{dtype}_16X16X16_{dtype}',
            'ENC_VOP3P',
            0,
            operands,
        )
        lower = dtype.lower()

        for arch in ('rdna3', 'rdna3_5'):
            body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], arch)

            assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
            assert (
                f'amdgpu::exec_gfx11_wmma_{lower}(cu, wf.wf_size(), 16, 16, 16, 16, dst,'
                in body
            )
            assert '(inst_.op_sel >> 2) & 0x1u' in body
            assert f'amdgpu::exec_wmma_{lower}(cu, 16, 16, 16' not in body

        body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')
        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert f'amdgpu::exec_wmma_{lower}(cu, 16, 16, 16, 16, dst,' in body
        assert 'wf.wf_size());' in body
        assert 'exec_gfx11_wmma' not in body


def test_gfx1250_wmma_i32_iu4_emits_executor():
    inst = Instruction('V_WMMA_I32_16X16X16_IU4', 'ENC_VOP3P', 0, [])
    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'gfx1250')

    assert 'auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i4' in body
    assert 'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 4, dst, src0_base,' in body


def test_div_scale_uses_signed_tiny_exponent_threshold():
    body = gen_vector_div_scale(
        ['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'f32', is_vop3=True
    )

    assert 'exp2 <= -23' in body
    assert 'exp2 <= 23' not in body


def test_gfx1250_profile_enables_generator_backed_quirks():
    profile = Gfx1250Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert profile.use_hwreg_helpers
    assert profile.hwreg_wave_sched_mode_id == 26
    assert profile.generate_scaled_wmma_vop3px2
    assert profile.smem_address_uses_access_size


def test_rdna3_profile_enables_gfx11_vop3_true16_only():
    profile = Rdna3Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert not profile.vbuffer_store_data_uses_dst_vgpr_msb_role


def test_rdna4_profile_enables_gfx12_true16_but_keeps_gfx1250_quirks_disabled():
    profile = Rdna4Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert not profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert not profile.use_hwreg_helpers
    assert profile.hwreg_wave_sched_mode_id is None
    assert not profile.generate_scaled_wmma_vop3px2
    assert not profile.smem_address_uses_access_size


def test_ds_swizzle_generator_uses_addr_source_for_ds_and_vds():
    def body_for(enc_name: str) -> str:
        codegen = object.__new__(CodeGenerator)
        codegen.isa_spec = SimpleNamespace(
            arch_name='gfx1250',
            profile=Gfx1250Profile(),
            inst_encodings=[],
            encoding_map={},
        )
        inst = Instruction(
            'DS_SWIZZLE_B32',
            enc_name,
            0,
            [
                Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
                Operand('addr', 32, 'OPR_VGPR', True, False, False, False, 1),
            ],
        )
        sem = InstructionSemantics('DS_SWIZZLE_B32', 'ds_swizzle')
        return codegen._gen_execute_body(inst, sem, enc_name)

    vds_body = body_for('ENC_VDS')
    ds_body = body_for('ENC_DS')

    assert 'src_data[i] = cu.read_vgpr(vb + inst_.addr, i);' in vds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.data0, i);' not in vds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.addr, i);' in ds_body
    assert 'src_data[i] = cu.read_vgpr(vb + inst_.data0, i);' not in ds_body
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


def test_gfx1250_generated_vop3_add_f16_applies_dpp():
    import pathlib

    arch_root = (
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
    encodings_h = (arch_root / 'encodings.h').read_text()
    vop3_alu = '\n'.join(
        path.read_text() for path in sorted(arch_root.glob('vop3_alu*.cpp'))
    )

    vop3_base = encodings_h[
        encodings_h.index('class Vop3') : encodings_h.index('class Vop3p')
    ]
    assert 'uint32_t dpp_ctrl_ = 0;' in vop3_base

    start = vop3_alu.index('VAddF16Vop3::VAddF16Vop3')
    end = vop3_alu.index('void VAddF16Vop3::execute_impl', start)
    ctor = vop3_alu[start:end]
    assert 'Vop3VopDpp16MachineInst' in ctor
    assert 'dpp_ctrl_ = dp->dpp_ctrl;' in ctor

    start = vop3_alu.index('void VAddF16Vop3::execute_impl')
    end = vop3_alu.index('VAddNcU16Vop3::VAddNcU16Vop3', start)
    body = vop3_alu[start:end]
    assert 'amdgpu::dpp::apply_dpp(src_operands_[0], dpp_ctrl_' in body
    assert 'if (dpp_src0_)' in body
    assert 'src0.set_delegate(dpp_src0_.get());' in body
    assert 'src0.clear_delegate();' in body


def test_generated_vop3_dot2_true16_uses_true16_helpers():
    import pathlib

    vop3 = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'rdna4'
        / 'vop3.cpp'
    ).read_text()

    start = vop3.index('void VDot2F16F16Vop3::execute_impl')
    end = vop3.index('VDot2Bf16Bf16Vop3::VDot2Bf16Bf16Vop3', start)
    body = vop3[start:end]

    assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in body
    assert 'uint32_t raw0 = src0.read_lane(wf, lane);' in body
    assert 'uint32_t raw1 = src1.read_lane(wf, lane);' in body
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in body
    assert 'util::f16_to_f32' in body
    assert 'util::f32_to_f16' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result_bits)' in body
    assert 'throw util::UnimplementedInst' not in body


def test_generated_rdna4_vop3_cvt_f32_f16_applies_true16_source_modifiers():
    import pathlib

    vop3 = (
        pathlib.Path(__file__).resolve().parents[4]
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'rdna4'
        / 'vop3.cpp'
    ).read_text()

    start = vop3.index('void VCvtF32F16Vop3::execute_impl')
    end = vop3.index('VCvtU16F16Vop3::VCvtU16F16Vop3', start)
    body = vop3[start:end]

    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'float src = util::f16_to_f32(static_cast<uint16_t>(raw));' in body
    assert 'if (inst_.abs & (1u << 0))' in body
    assert 'src = std::fabs(src);' in body
    assert 'if (inst_.neg & (1u << 0))' in body
    assert 'vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(src));' in body


def test_generated_rdna3_dot2acc_uses_dot2c_simd_probe():
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

    start = execute_shared.index('inline void execute_v_dot2acc_f32_f16_vop2')
    end = execute_shared.index('inline void execute_v_dot2c_f32_f16_vop2', start)
    body = execute_shared[start:end]

    assert 'ROCJITSU_TRY_SIMD_DOTC_F16(false);' in body
    assert 'uint32_t acc = inst.vdst.read_lane(wf, lane);' in body
    assert 'float facc = std::bit_cast<float>(acc);' in body
    assert 'facc += a0 * b0 + a1 * b1;' in body
    assert 'throw util::UnimplementedInst' not in body


def test_rdna4_swmmac_uses_src2_as_sparse_index_vgpr():
    inst = Instruction(
        'V_SWMMAC_F32_16X16X32_FP8_FP8',
        'ENC_VOP3P',
        0,
        [
            Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
            Operand('src0', 64, 'OPR_SRC_VGPR', True, False, False, False, 1),
            Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
            Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
        ],
    )

    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')

    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'uint32_t const_acc = amdgpu::ACC_FROM_VGPR;' in body
    assert 'uint32_t s2 = dst;' in body
    assert 'uint32_t index_base = amdgpu::src_base(vb, src2.encoding_value_);' in body
    assert 'uint32_t index_key = inst_.opsel & 0x1u;' in body
    assert (
        'amdgpu::exec_swmmac_f32(cu, 16, 16, 32, 8, dst, '
        'amdgpu::src_base(vb, src0.encoding_value_), '
        'amdgpu::src_base(vb, src1.encoding_value_), s2, index_base, 16, '
        'index_key, amdgpu::extract_fp8, amdgpu::extract_fp8, const_acc, wf.wf_size());'
    ) in body
    assert 'resolve_acc' not in body


def test_rdna4_f16_bf16_swmmac_dispatch_wiring_is_generated():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
        Operand('src0', 64, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
    ]

    cases = [
        ('F16', 'FP8_FP8', 'exec_swmmac_f16', 'extract_fp8', 'extract_fp8'),
        ('BF16', 'BF8_BF8', 'exec_swmmac_bf16', 'extract_bf8', 'extract_bf8'),
    ]
    for result_type, input_type, exec_fn, extract_a, extract_b in cases:
        inst = Instruction(
            f'V_SWMMAC_{result_type}_16X16X32_{input_type}',
            'ENC_VOP3P',
            0,
            operands,
        )

        body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert 'uint32_t const_acc = amdgpu::ACC_FROM_VGPR;' in body
        assert 'uint32_t s2 = dst;' in body
        assert (
            'uint32_t index_base = amdgpu::src_base(vb, src2.encoding_value_);' in body
        )
        assert 'uint32_t index_key = inst_.opsel & 0x1u;' in body
        assert (
            f'amdgpu::{exec_fn}(cu, 16, 16, 32, 8, dst, '
            'amdgpu::src_base(vb, src0.encoding_value_), '
            'amdgpu::src_base(vb, src1.encoding_value_), s2, index_base, 16, '
            f'index_key, amdgpu::{extract_a}, amdgpu::{extract_b}, const_acc, wf.wf_size());'
        ) in body


def test_rdna4_swmmac_uses_32_index_entries_for_wide_8bit_k():
    inst = Instruction(
        'V_SWMMAC_F32_16X16X128_FP8_FP8',
        'ENC_VOP3P',
        0,
        [
            Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
            Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
            Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
            Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
        ],
    )

    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'rdna4')

    assert 'index_base, 32, index_key, amdgpu::extract_fp8, amdgpu::extract_fp8' in body


def test_gfx1250_generated_fp8_vop3_shared_byte_select_uses_inst_member():
    import pathlib

    source_root = pathlib.Path(__file__).resolve().parents[4]
    execute_shared = (
        source_root
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

    gfx1250_vop3_cvt = (
        source_root
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
        / 'gfx1250'
        / 'vop3_cvt.cpp'
    ).read_text()
    start = gfx1250_vop3_cvt.index('void VCvtF16Fp8Vop3::execute_impl')
    end = gfx1250_vop3_cvt.index('VCvtF16Bf8Vop3::VCvtF16Bf8Vop3', start)
    body = gfx1250_vop3_cvt[start:end]
    body_words = ' '.join(body.split())

    assert '>> (((amdgpu::vop3_opsel(inst_) & 0x2u) >> 1) * 8u)' in body_words
    assert '((amdgpu::vop3_opsel(inst_) & 0x1u) << 1)' not in body


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
        operand_types={'OPR_SRC_SIMPLE'},
    )
    codegen.out_path = str(tmp_path)

    codegen.gen_vopd()
    cpp = (tmp_path / 'gfx1250' / 'vopd.cpp').read_text()

    assert '(word0 >> 24) == 0xCF' in cpp
    assert '[[maybe_unused]] bool vopd3' not in cpp
    assert 'vopd3 ? OperandType::OPR_SRC_SIMPLE : OperandType::OPR_SRC' in cpp
    assert 'case 3:\n              case 7:' not in cpp
    assert 'if (lhs == 0.0f || rhs == 0.0f)' in cpp
    src_neg_start = cpp.index('bool Vopd::uses_src_neg_modifier')
    src_neg_body = cpp[
        src_neg_start : cpp.index('uint32_t Vopd::apply_neg', src_neg_start)
    ]
    assert 'case kVopdCndmaskB32:' in src_neg_body
    assert 'if (uses_src_neg_modifier(slot.op))' in cpp
    execute_start = cpp.index('uint32_t Vopd::execute_slot')
    fma_start = cpp.index('case kVopdFmaF32', execute_start)
    fma_case = cpp[fma_start : cpp.index('case kVopdSubNcU32:', fma_start)]
    assert 'std::fma(std::bit_cast<float>(src0),' in fma_case
    assert 'std::bit_cast<float>(src1),' in fma_case
    assert 'std::bit_cast<float>(src2))' in fma_case
    assert 'constexpr uint16_t kVopdFmaF64 = 32;' in cpp
    assert 'constexpr uint16_t kVopdAddF64 = 33;' in cpp
    assert 'constexpr uint16_t kVopdMulF64 = 34;' in cpp
    assert 'constexpr uint16_t kVopdMaxNumF64 = 35;' in cpp
    assert 'constexpr uint16_t kVopdMinNumF64 = 36;' in cpp
    assert 'kVopdFmacF64' not in cpp


def test_rdna4_profile_enables_generated_vopd():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        operand_types={'OPR_SRC'},
    )

    assert codegen._supports_generated_vopd()


def test_rdna4_vopd_template_uses_available_src_operand_type(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        operand_types={'OPR_SRC'},
    )
    codegen.out_path = str(tmp_path)

    codegen.gen_vopd()
    cpp = (tmp_path / 'rdna4' / 'vopd.cpp').read_text()

    assert 'OperandType::OPR_SRC_SIMPLE' not in cpp
    assert 'vopd3 ? OperandType::OPR_SRC : OperandType::OPR_SRC' not in cpp
    assert 'return Operand(bits, OperandType::OPR_SRC, encoded);' in cpp
    assert '(word0 >> 24) == 0xCF' not in cpp
    assert 'Format::Vopd3' not in cpp
    assert 'kVopdAddF64' not in cpp
    assert 'execute_slot64' not in cpp
    assert 'constexpr uint16_t kVopdDot2AccF32F16 = 12;' in cpp
    assert 'constexpr uint16_t kVopdDot2AccF32Bf16 = 13;' in cpp
    assert 'constexpr uint16_t kVopdAndB32 = 18;' in cpp
    assert 'constexpr uint16_t kVopdBitop2B32' not in cpp
    assert 'constexpr uint16_t kVopdFmaF32' not in cpp
    assert 'v_dual_dot2acc_f32_f16' in cpp
    assert 'v_dual_max_num_f32' in cpp
    assert 'v_dual_max_f32' not in cpp


def test_rdna3_and_rdna35_vopd_generation_matches_common_profile(tmp_path):
    generated = {}
    for arch_name, profile in (
        ('rdna3', Rdna3Profile()),
        ('rdna3_5', Rdna3_5Profile()),
    ):
        codegen = object.__new__(CodeGenerator)
        codegen.isa_spec = SimpleNamespace(
            arch_name=arch_name,
            profile=profile,
            operand_types={'OPR_SRC'},
        )
        codegen.out_path = str(tmp_path)

        codegen.gen_vopd()
        generated[arch_name] = (tmp_path / arch_name / 'vopd.cpp').read_text()

    rdna3_cpp = generated['rdna3']
    assert 'constexpr uint16_t kVopdDot2AccF32F16 = 12;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdDot2AccF32Bf16 = 13;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdAndB32 = 18;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdBitop2B32' not in rdna3_cpp
    assert 'constexpr uint16_t kVopdFmaF32' not in rdna3_cpp
    assert 'v_dual_max_f32' in rdna3_cpp
    assert 'v_dual_max_num_f32' not in rdna3_cpp

    normalized = {
        arch: cpp.replace('rdna3_5', 'rdnaX').replace('rdna3', 'rdnaX')
        for arch, cpp in generated.items()
    }
    assert normalized['rdna3'] == normalized['rdna3_5']


def test_gfx1250_vopd_uses_plain_src_when_simple_src_operand_is_absent(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
        operand_types={'OPR_SRC'},
    )
    codegen.out_path = str(tmp_path)

    codegen.gen_vopd()
    cpp = (tmp_path / 'gfx1250' / 'vopd.cpp').read_text()

    assert '[[maybe_unused]] bool vopd3' in cpp
    assert 'OperandType::OPR_SRC_SIMPLE' not in cpp
    assert 'return Operand(bits, OperandType::OPR_SRC, encoded);' in cpp


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


def test_gfx1250_cluster_load_generators_force_request_l1_bypass():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1250',
        profile=Gfx1250Profile(),
    )

    cluster = SimpleNamespace(
        name='CLUSTER_LOAD_B32',
        elem_size=4,
        num_elems=1,
        sign_extend=False,
        d16_hi=False,
        d16_lo=False,
    )
    ordinary = SimpleNamespace(
        name='GLOBAL_LOAD_B32',
        elem_size=4,
        num_elems=1,
        sign_extend=False,
        d16_hi=False,
        d16_lo=False,
    )
    cluster_body = codegen._gen_flat_load([], [], cluster)
    ordinary_body = codegen._gen_flat_load([], [], ordinary)

    assert 'd->request_force_l1_bypass = true;' in cluster_body
    assert 'd->request_force_l1_bypass = true;' not in ordinary_body

    cluster_async = SimpleNamespace(
        name='CLUSTER_LOAD_ASYNC_TO_LDS_B32',
        elem_size=4,
        num_elems=1,
    )
    global_async = SimpleNamespace(
        name='GLOBAL_LOAD_ASYNC_TO_LDS_B32',
        elem_size=4,
        num_elems=1,
    )
    cluster_async_body = codegen._gen_global_load_async_to_lds([], [], cluster_async)
    global_async_body = codegen._gen_global_load_async_to_lds([], [], global_async)

    assert 'd->request_force_l1_bypass = true;' in cluster_async_body
    assert 'd->request_force_l1_bypass = true;' not in global_async_body


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

    rdna3_op = (gen_root / 'rdna3' / 'operand.cpp').read_text()
    assert 'if (ev == 124)\n    return 0u; // NULL' in rdna3_op
    assert 'if (ev == 125)\n    return wf.m0()' in rdna3_op

    rdna3_5_op = (gen_root / 'rdna3_5' / 'operand.cpp').read_text()
    assert 'if (ev == 124)\n    return 0u; // NULL' in rdna3_5_op
    assert 'if (ev == 125)\n    return wf.m0()' in rdna3_5_op

    cdna4_op = (gen_root / 'cdna4' / 'operand.cpp').read_text()
    assert 'if (ev == 124)\n    return wf.m0()' in cdna4_op
    assert 'ev == 125' not in cdna4_op.split('can_resolve_src_scalar')[1].split('}')[0]
