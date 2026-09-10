# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ISA dimension properties on IsaProfile subclasses."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import (
    CodeGenerator,
    _ImplOutputs,
    _SourceImplUnit,
)
from amdisa.codegen.config import CodegenConfig
from amdisa.__main__ import (
    _apply_codegen_identity,
    _codegen_config,
    _detect_profile,
    _parse_isa_arg,
)
from amdisa.gpuisa import InstEncoding, Instruction, MicrocodeField
from amdisa.isa_properties_codegen import emit_isa_properties
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    CdnaProfile,
    Cdna5Profile,
    DppOpcodeRule,
    DppCtrlDialect,
    MatrixLayout,
    MemoryCoherencyModel,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
    SwmmacLayout,
    WaveStateLayout,
)


@pytest.mark.parametrize(
    (
        'profile',
        'renders_halves',
        'opsel_field',
        'vop3p_hi_high_field',
        'dpp_dialect',
    ),
    [
        (CdnaProfile(), False, 'op_sel', 'op_sel_hi_2', DppCtrlDialect.GFX9),
        (Rdna4Profile(), True, 'opsel', 'opsel_hi_2', DppCtrlDialect.GFX10_PLUS),
        (Cdna5Profile(), True, 'opsel', 'opsel_hi_2', DppCtrlDialect.GFX10_PLUS),
    ],
)
def test_vop3_disassembly_profile(
    profile, renders_halves, opsel_field, vop3p_hi_high_field, dpp_dialect
):
    assert profile.uses_true16_vop3_opsel
    assert profile.renders_true16_vop3_operands is renders_halves
    assert profile.vop3_opsel_field == opsel_field
    assert profile.vop3p_opsel_hi_high_field == vop3p_hi_high_field
    assert profile.dpp_ctrl_dialect == dpp_dialect


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), False),
        (Rdna1Profile(), True),
        (Rdna3Profile(), True),
        (Rdna4Profile(), True),
        (Cdna5Profile(), False),
    ],
)
def test_supports_wgp_mode(profile, expected):
    assert profile.supports_wgp_mode is expected


@pytest.mark.parametrize(
    ('profile', 'uses_ttmp', 'uses_cluster_ttmp'),
    [
        (CdnaProfile(), False, False),
        (Rdna4Profile(), True, False),
        (Cdna5Profile(), True, True),
    ],
)
def test_ttmp_workgroup_id_properties(profile, uses_ttmp, uses_cluster_ttmp):
    assert profile.uses_ttmp_workgroup_ids is uses_ttmp
    assert profile.uses_cluster_ttmp_workgroup_ids is uses_cluster_ttmp


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), WaveStateLayout.LEGACY),
        (Rdna3Profile(), WaveStateLayout.LEGACY),
        (Rdna4Profile(), WaveStateLayout.GFX12),
        (Cdna5Profile(), WaveStateLayout.GFX12_5),
    ],
)
def test_wave_state_layout(profile, expected):
    assert profile.wave_state_layout is expected


@pytest.mark.parametrize(
    ('profile', 'granule', 'bits'),
    [
        (CdnaProfile(), 1024, 13),
        (Rdna2Profile(), 1024, 13),
        (Rdna3Profile(), 256, 15),
        (Rdna4Profile(), 256, 18),
        (Cdna5Profile(), 256, 18),
    ],
)
def test_compute_tmpring_wavesize_properties(profile, granule, bits):
    assert profile.compute_tmpring_wavesize_granule == granule
    assert profile.compute_tmpring_wavesize_bits == bits


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), True),
        (Rdna1Profile(), False),
        (Rdna3Profile(), False),
        (Rdna4Profile(), False),
        (Cdna5Profile(), False),
    ],
)
def test_descriptor_sgpr_count_encoded(profile, expected):
    assert profile.descriptor_sgpr_count_encoded is expected


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), 256),
        (Rdna4Profile(), 256),
        (Cdna5Profile(), 1024),
    ],
)
def test_max_addressable_vgprs_per_wf(profile, expected):
    assert profile.max_addressable_vgprs_per_wf == expected


@pytest.mark.parametrize(
    ('profile', 'expected_wave32', 'expected_wave64'),
    [
        (Cdna1Profile(), 0, 4),
        (Cdna2Profile(), 0, 8),
        (CdnaProfile(), 0, 8),
        (Rdna1Profile(), 8, 4),
        (Rdna2Profile(), 8, 4),
        (Rdna3Profile(), 8, 4),
        (Rdna3_5Profile(), 8, 4),
        (Rdna4Profile(), 8, 4),
        (Cdna5Profile(), 16, 0),
    ],
)
def test_descriptor_vgpr_count_granule(profile, expected_wave32, expected_wave64):
    assert profile.descriptor_vgpr_count_granule_wave32 == expected_wave32
    assert profile.descriptor_vgpr_count_granule_wave64 == expected_wave64


@pytest.mark.parametrize(
    'profile',
    [
        CdnaProfile(),
        Cdna1Profile(),
        Cdna2Profile(),
        Rdna1Profile(),
        Rdna2Profile(),
        Rdna3Profile(),
        Rdna3_5Profile(),
        Rdna4Profile(),
        Cdna5Profile(),
    ],
)
def test_amdgpu_profiles_split_execution_sources(profile):
    assert profile.split_execution_sources


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (Cdna1Profile(), False),
        (Cdna2Profile(), False),
        (CdnaProfile(), False),
        (Rdna1Profile(), False),
        (Rdna2Profile(), False),
        (Rdna3Profile(), False),
        (Rdna3_5Profile(), False),
        (Rdna4Profile(), False),
        (Cdna5Profile(), True),
    ],
)
def test_uses_vgpr_msb_indexing(profile, expected):
    assert profile.uses_vgpr_msb_indexing is expected


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), False),
        (Cdna1Profile(), False),
        (Cdna2Profile(), False),
        (Rdna1Profile(), False),
        (Rdna3Profile(), True),
        (Rdna3_5Profile(), True),
        (Rdna4Profile(), True),
        (Cdna5Profile(), True),
    ],
)
def test_dpp_inactive_source_policy(profile, expected):
    assert profile.dpp_bound_ctrl_applies_to_inactive_sources is expected


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), False),
        (Cdna1Profile(), False),
        (Cdna2Profile(), False),
        (Rdna1Profile(), False),
        (Rdna3Profile(), True),
        (Rdna3_5Profile(), True),
        (Rdna4Profile(), True),
        (Cdna5Profile(), True),
    ],
)
def test_dpp_suppressed_compare_policy(profile, expected):
    assert profile.dpp_suppressed_compare_lanes_zero is expected


@pytest.mark.parametrize(
    ('profile', 'src0_size_bits', 'expected'),
    [
        (CdnaProfile(), 32, DppOpcodeRule.ALLOW),
        (CdnaProfile(), 64, DppOpcodeRule.ROW_SELECT_ONLY),
        (Cdna1Profile(), 64, DppOpcodeRule.ALLOW),
        (Cdna2Profile(), 64, DppOpcodeRule.ALLOW),
    ],
)
def test_cdna_64bit_input_dpp_rule(profile, src0_size_bits, expected):
    assert (
        profile.dpp_opcode_rule('ENC_VOP1', 'V_MOV_B64', src0_size_bits=src0_size_bits)
        is expected
    )


@pytest.mark.parametrize('opcode', ['V_READFIRSTLANE_B32', 'V_CLREXCP', 'V_SWAP_B32'])
def test_cdna3_and_4_lane_special_opcodes_do_not_accept_dpp(opcode):
    assert (
        CdnaProfile().dpp_opcode_rule('ENC_VOP1', opcode, src0_size_bits=32)
        is DppOpcodeRule.FORBID
    )


@pytest.mark.parametrize(
    'profile',
    [
        Cdna1Profile(),
        Cdna2Profile(),
        CdnaProfile(),
        Rdna1Profile(),
        Rdna2Profile(),
    ],
)
@pytest.mark.parametrize(
    'opcode',
    [
        'V_READFIRSTLANE_B32',
        'V_CVT_I32_F64',
        'V_CVT_F64_I32',
        'V_CVT_F32_F64',
        'V_CVT_F64_F32',
        'V_CVT_U32_F64',
        'V_CVT_F64_U32',
        'V_TRUNC_F64',
        'V_CEIL_F64',
        'V_RNDNE_F64',
        'V_FLOOR_F64',
        'V_RCP_F64',
        'V_RSQ_F64',
        'V_SQRT_F64',
        'V_FREXP_EXP_I32_F64',
        'V_FREXP_MANT_F64',
        'V_FRACT_F64',
        'V_CLREXCP',
        'V_SWAP_B32',
        'V_CMP_CLASS_F64',
        'V_CMPX_CLASS_F64',
        'V_CMP_EQ_F64',
        'V_CMPX_LT_I64',
        'V_CMP_GE_U64',
        'V_CMPX_NE_U64',
    ],
)
def test_legacy_dpp_prohibition_tables(profile, opcode):
    assert (
        profile.dpp_opcode_rule('ENC_VOP1', opcode, src0_size_bits=64)
        is DppOpcodeRule.FORBID
    )


@pytest.mark.parametrize(
    'profile',
    [
        CdnaProfile(),
        Cdna1Profile(),
        Cdna2Profile(),
        Rdna1Profile(),
        Rdna2Profile(),
        Rdna3Profile(),
        Rdna3_5Profile(),
        Rdna4Profile(),
    ],
)
def test_public_snapshot_literal_field_normalizes_to_simm32(profile):
    assert profile.field_renames('VOP2_INST_LITERAL')['literal'] == 'simm32'
    assert (
        profile.normalize_operand_field_name('VOP2_INST_LITERAL', 'literal') == 'simm32'
    )


@pytest.mark.parametrize(
    'profile',
    [Cdna1Profile(), Cdna2Profile(), CdnaProfile(), Rdna1Profile(), Rdna2Profile()],
)
@pytest.mark.parametrize('opcode', ['V_MOV_B64', 'V_CMP_EQ_U32', 'V_SWAPREL_B32'])
def test_legacy_dpp_prohibition_tables_do_not_match_nearby_legal_opcodes(
    profile, opcode
):
    src0_size_bits = 64 if opcode == 'V_MOV_B64' else 32
    assert (
        profile.dpp_opcode_rule('ENC_VOP1', opcode, src0_size_bits=src0_size_bits)
        is not DppOpcodeRule.FORBID
    )


@pytest.mark.parametrize(
    ('profile', 'wave', 'row_bcast', 'row_xmask'),
    [
        (CdnaProfile(), True, True, False),
        (Rdna1Profile(), True, True, False),
        (Rdna3Profile(), False, False, True),
        (Rdna4Profile(), False, False, True),
        (Cdna5Profile(), False, False, True),
    ],
)
def test_dpp_control_range_capabilities(profile, wave, row_bcast, row_xmask):
    assert profile.dpp_supports_wave_controls is wave
    assert profile.dpp_supports_row_broadcast_controls is row_bcast
    assert profile.dpp_supports_row_xmask is row_xmask


@pytest.mark.parametrize(
    ('profile', 'encoding', 'opcode', 'expected'),
    [
        (CdnaProfile(), 'ENC_VOP3P', 'V_PK_ADD_U16', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP1', 'V_CVT_F64_I32', DppOpcodeRule.FORBID),
        (Rdna3Profile(), 'ENC_VOP1', 'V_MOV_B32', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP1', 'V_NOP', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP1', 'V_SWAPREL_B32', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP1', 'V_SWAP_B32', DppOpcodeRule.FORBID),
        (Rdna3Profile(), 'ENC_VOP2', 'V_FMAMK_F32', DppOpcodeRule.FORBID),
        (Rdna3Profile(), 'ENC_VOP2', 'V_ADD_F32', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP3P', 'V_PK_ADD_U16', DppOpcodeRule.FORBID),
        (Rdna3Profile(), 'ENC_VOP3P', 'V_WMMA_F32_16X16X16_F16', DppOpcodeRule.FORBID),
        (Rdna3Profile(), 'ENC_VOP3P', 'V_FMA_MIX_F32', DppOpcodeRule.ALLOW),
        (Rdna3Profile(), 'ENC_VOP3P', 'V_DOT2_F32_F16', DppOpcodeRule.ALLOW),
        (Rdna4Profile(), 'ENC_VOP3P', 'V_DOT2_F32_F16', DppOpcodeRule.FORBID),
        (Rdna4Profile(), 'ENC_VOP3', 'V_MUL_LO_U32', DppOpcodeRule.FORBID),
        (Rdna4Profile(), 'ENC_VOP1', 'V_NOP', DppOpcodeRule.FORBID),
        (Rdna4Profile(), 'ENC_VOP1', 'V_PIPEFLUSH', DppOpcodeRule.ALLOW),
        (Rdna4Profile(), 'VOP3_SDST_ENC', 'V_ADD_CO_CI_U32', DppOpcodeRule.ALLOW),
        (Rdna4Profile(), 'ENC_VOPC', 'V_CMP_EQ_F64', DppOpcodeRule.FORBID),
        (Rdna4Profile(), 'ENC_VOPC', 'V_CMP_EQ_U32', DppOpcodeRule.ALLOW),
        (
            Cdna5Profile(),
            'ENC_VOP3',
            'V_ADD_F64',
            DppOpcodeRule.ROW_SELECT_ONLY,
        ),
        (
            Cdna5Profile(),
            'ENC_VOP3',
            'V_MUL_LO_U32',
            DppOpcodeRule.ROW_SELECT_ONLY,
        ),
        (Cdna5Profile(), 'ENC_VOP3', 'V_TRIG_PREOP_F64', DppOpcodeRule.FORBID),
        (
            Cdna5Profile(),
            'ENC_VOP3',
            'V_LDEXP_F64',
            DppOpcodeRule.ROW_SELECT_ONLY,
        ),
        (Cdna5Profile(), 'ENC_VOP3', 'V_CVT_SCALE_F32_F16', DppOpcodeRule.FORBID),
        (Cdna5Profile(), 'ENC_VOP3P', 'V_FMA_MIX_F32', DppOpcodeRule.ALLOW),
        (Cdna5Profile(), 'ENC_VOP3P', 'V_DOT4_F32_FP8', DppOpcodeRule.FORBID),
    ],
)
def test_dpp_opcode_rules(profile, encoding, opcode, expected):
    assert profile.dpp_opcode_rule(encoding, opcode) is expected


@pytest.mark.parametrize(
    'profile',
    [
        CdnaProfile(),
        Cdna1Profile(),
        Cdna2Profile(),
        Rdna1Profile(),
        Rdna2Profile(),
        Rdna3Profile(),
        Rdna3_5Profile(),
        Rdna4Profile(),
    ],
)
def test_public_snapshot_profiles_support_schema_1_1_1(profile):
    assert '1.1.1' in profile.supported_versions


def test_non_split_generation_leaves_exec_named_sources_untouched(tmp_path):
    class NonSplitRdna4Profile(Rdna4Profile):
        @property
        def split_execution_sources(self):
            return False

    arch_dir = tmp_path / 'test'
    arch_dir.mkdir()
    exec_named_source = arch_dir / 'sopp_exec.cpp'
    exec_named_source.write_text('user-owned source')

    generator = object.__new__(CodeGenerator)
    generator.out_path = str(tmp_path)
    generator.isa_spec = SimpleNamespace(
        arch_name='test',
        generated_dir_name='test',
        cpp_namespace='test',
        profile=NonSplitRdna4Profile(),
    )
    generator._write_inst_impl_files(
        'ENC_SOPP',
        'sopp',
        [],
        _ImplOutputs(model=['model implementation']),
    )

    assert exec_named_source.read_text() == 'user-owned source'


@pytest.mark.parametrize(
    ('profile', 'enc_name', 'expected'),
    [
        (CdnaProfile(), 'ENC_FLAT', '0x7F'),
        (Rdna3Profile(), 'ENC_FLAT', '0x7F'),
        (Rdna4Profile(), 'ENC_VFLAT', 'OPR_SREG_NULL'),
        (Rdna4Profile(), 'ENC_VGLOBAL', 'OPR_SREG_NULL'),
        (Cdna5Profile(), 'ENC_VFLAT', 'OPR_SREG_NULL'),
        (Cdna5Profile(), 'ENC_VGLOBAL', 'OPR_SREG_NULL'),
    ],
)
def test_saddr_null_selector_is_encoding_specific(profile, enc_name, expected):
    assert profile.saddr_null_selector_expr(enc_name) == expected


def test_saddr_null_selector_rejects_unrelated_encodings():
    assert Rdna3Profile().saddr_null_selector_expr('ENC_VOP3') is None
    assert Rdna4Profile().saddr_null_selector_expr('ENC_VSCRATCH') is None


def test_isa_properties_codegen_uses_profile_values(tmp_path):
    specs = [
        ('cdna3', SimpleNamespace(profile=CdnaProfile()), None),
        ('rdna4', SimpleNamespace(profile=Rdna4Profile()), None),
        ('cdna5', SimpleNamespace(profile=Cdna5Profile()), None),
    ]

    output = emit_isa_properties(str(tmp_path), specs).read_text()

    assert 'uint32_t max_addressable_vgprs_per_wf = 0;' in output
    assert 'bool mode_has_gpr_idx_en = false;' in output
    assert 'enum class WaveStateLayout : uint8_t {' in output
    assert 'uint32_t compute_tmpring_wavesize_granule = 0;' in output
    assert 'uint32_t compute_tmpring_wavesize_bits = 0;' in output
    assert 'uint32_t wave_size = 0;' in output
    assert 'uint32_t wave_size_max = 0;' in output
    assert 'uint32_t descriptor_vgpr_count_granule_wave32 = 0;' in output
    assert 'uint32_t descriptor_vgpr_count_granule_wave64 = 0;' in output
    assert 'MAX_SUPPORTED_ADDRESSABLE_VGPRS_PER_WF = 1024;' in output
    assert (
        'case ROCJITSU_CODE_ARCH_CDNA3:\n'
        '    return {\n'
        '        .supports_wgp_mode = false,\n'
        '        .mode_has_gpr_idx_en = true,\n'
        '        .descriptor_sgpr_count_encoded = true,\n'
        '        .uses_ttmp_workgroup_ids = false,\n'
        '        .uses_cluster_ttmp_workgroup_ids = false,\n'
        '        .wave_state_layout = WaveStateLayout::Legacy,\n'
        '        .compute_tmpring_wavesize_granule = 1024,\n'
        '        .compute_tmpring_wavesize_bits = 13,\n'
        '        .wave_size = 64,\n'
        '        .wave_size_max = 64,\n'
        '        .max_addressable_vgprs_per_wf = 256,\n'
        '        .descriptor_vgpr_count_granule_wave32 = 0,\n'
        '        .descriptor_vgpr_count_granule_wave64 = 8,\n'
        '    };'
    ) in output
    assert (
        'case ROCJITSU_CODE_ARCH_RDNA4:\n'
        '    return {\n'
        '        .supports_wgp_mode = true,\n'
        '        .mode_has_gpr_idx_en = false,\n'
        '        .descriptor_sgpr_count_encoded = false,\n'
        '        .uses_ttmp_workgroup_ids = true,\n'
        '        .uses_cluster_ttmp_workgroup_ids = false,\n'
        '        .wave_state_layout = WaveStateLayout::Gfx12,\n'
        '        .compute_tmpring_wavesize_granule = 256,\n'
        '        .compute_tmpring_wavesize_bits = 18,\n'
        '        .wave_size = 32,\n'
        '        .wave_size_max = 64,\n'
        '        .max_addressable_vgprs_per_wf = 256,\n'
        '        .descriptor_vgpr_count_granule_wave32 = 8,\n'
        '        .descriptor_vgpr_count_granule_wave64 = 4,\n'
        '    };'
    ) in output
    assert (
        'case ROCJITSU_CODE_ARCH_CDNA5:\n'
        '    return {\n'
        '        .supports_wgp_mode = false,\n'
        '        .mode_has_gpr_idx_en = false,\n'
        '        .descriptor_sgpr_count_encoded = false,\n'
        '        .uses_ttmp_workgroup_ids = true,\n'
        '        .uses_cluster_ttmp_workgroup_ids = true,\n'
        '        .wave_state_layout = WaveStateLayout::Gfx12_5,\n'
        '        .compute_tmpring_wavesize_granule = 256,\n'
        '        .compute_tmpring_wavesize_bits = 18,\n'
        '        .wave_size = 32,\n'
        '        .wave_size_max = 32,\n'
        '        .max_addressable_vgprs_per_wf = 1024,\n'
        '        .descriptor_vgpr_count_granule_wave32 = 16,\n'
        '        .descriptor_vgpr_count_granule_wave64 = 0,\n'
        '    };'
    ) in output


def test_isa_properties_codegen_uses_source_arch_for_custom_identity(tmp_path):
    specs = [
        (
            'gfx1250',
            SimpleNamespace(arch_name='cdna5', profile=Cdna5Profile()),
            None,
        )
    ]

    output = emit_isa_properties(str(tmp_path), specs).read_text()

    assert 'case ROCJITSU_CODE_ARCH_CDNA5:' in output
    assert '.max_addressable_vgprs_per_wf = 1024,' in output


def test_checked_in_isa_properties_matches_all_profiles(tmp_path):
    profiles = [
        ('cdna1', Cdna1Profile()),
        ('cdna2', Cdna2Profile()),
        ('cdna3', CdnaProfile()),
        ('cdna4', Cdna4Profile()),
        ('rdna1', Rdna1Profile()),
        ('rdna2', Rdna2Profile()),
        ('rdna3', Rdna3Profile()),
        ('rdna3_5', Rdna3_5Profile()),
        ('rdna4', Rdna4Profile()),
        ('cdna5', Cdna5Profile()),
    ]
    specs = [
        (name, SimpleNamespace(profile=profile), None) for name, profile in profiles
    ]

    generated = emit_isa_properties(str(tmp_path), specs).read_text()
    checked_in = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h'
    ).read_text()

    assert generated == checked_in


def test_public_literal_type_preserves_legacy_extension_identity():
    assert (
        Cdna1Profile().normalize_operand_type('ENC_VOP2', 'literal', 'OPR_SIMM16')
        == 'OPR_SIMM32'
    )
    assert (
        Cdna5Profile().normalize_operand_type('ENC_VOP2', 'literal', 'OPR_SIMM16')
        == 'OPR_SIMM16'
    )


def test_selector_case_normalization_is_schema_specific():
    assert Cdna1Profile().lowercase_operand_selector_names is True
    assert Rdna4Profile().lowercase_operand_selector_names is True
    assert Cdna5Profile().lowercase_operand_selector_names is False


def test_gfx1250_operand_execution_backend_uses_separate_source(tmp_path):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='cdna5',
            generated_dir_name='cdna5',
            cpp_namespace='cdna5',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Cdna5Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_h = (tmp_path / 'cdna5' / 'operand.h').read_text()
    operand_cpp = (tmp_path / 'cdna5' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'cdna5' / 'operand_exec.cpp').read_text()

    assert 'namespace cdna5 {' in operand_h
    assert 'rocjitsu/isa/arch/amdgpu/cdna5/isa.h' in operand_h
    assert 'rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h' in operand_h
    assert 'ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_OPERAND_H_' in operand_h
    assert 'class Operand : public IsaOperand<Isa>' in operand_h
    assert 'ROCJITSU_ISA_MODEL_ONLY' not in operand_h
    assert ': IsaOperand<Isa>(size_bits, opr_type, encoding_value)' in operand_cpp
    assert 'ROCJITSU_ISA_MODEL_ONLY' not in operand_cpp
    assert 'uint32_t Operand::read_scalar' in operand_cpp
    assert 'current_isa_operand_backend()' in operand_cpp
    assert (
        'execution_backend_ ? execution_backend_->read_scalar : nullptr' in operand_cpp
    )
    assert 'uint32_t Operand::read_scalar_exec' not in operand_cpp
    assert 'rocjitsu/vm/amdgpu/compute_unit.h' not in operand_cpp
    assert 'uint32_t Operand::read_scalar_exec' in operand_exec_cpp
    assert 'const void *Operand::full_execution_backend()' in operand_exec_cpp
    assert 'bool Operand::full_execution_backend_complete()' in operand_exec_cpp
    assert 'Operand::simd_vgpr_base_mut_impl' in operand_cpp
    assert 'Operand::simd_vgpr_base_mut_exec' in operand_exec_cpp
    assert '&Operand::simd_vgpr_base_mut_exec' in operand_exec_cpp
    assert 'backend.simd_vgpr_base_mut != nullptr' in operand_exec_cpp
    assert 'backend.simd_notify_read64_mut != nullptr' in operand_exec_cpp
    assert 'vgpr_msb_role() == amdgpu::VgprMsbRole::None' in operand_exec_cpp
    assert 'apply_gpr_idx(wf, *off, amdgpu::VgprMsbRole::Dst)' not in operand_exec_cpp
    assert 'execution_backend_registered_' not in operand_exec_cpp
    assert 'rocjitsu/vm/amdgpu/compute_unit.h' in operand_exec_cpp
    assert 'RegisterAccess(wf.cu())' not in operand_exec_cpp
    assert 'RegisterAccess(wf)' in operand_exec_cpp
    assert 'owns_vgpr_range(wf, reg, 1)' in operand_exec_cpp


def test_gfx1250_instruction_execution_backend_is_dense_and_scoped(tmp_path):
    generator = object.__new__(CodeGenerator)
    generator.out_path = str(tmp_path)
    generator.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        generated_dir_name='cdna5',
        cpp_namespace='cdna5',
        profile=Cdna5Profile(),
    )
    generator.config = CodegenConfig()
    generator._split_execution_classes = ['FirstInstruction', 'SecondInstruction']

    generator.gen_execution_backend()
    backend_h = (tmp_path / 'cdna5' / 'execution_backend.h').read_text()
    backend_cpp = (tmp_path / 'cdna5' / 'execution_backend_exec.cpp').read_text()

    assert 'const IsaExecutionBackend &execution_backend();' in backend_h
    assert 'enum class InstructionExecutionId : size_t {' in backend_h
    assert 'FirstInstruction,' in backend_h
    assert 'SecondInstruction,' in backend_h
    assert 'Count,' in backend_h
    assert 'static_cast<size_t>(InstructionExecutionId::Count)' in backend_cpp
    assert (
        'std::array<Instruction::ExecuteFn, kInstructionCallbackCount>' in backend_cpp
    )
    assert '&execute_with_backend<FirstInstruction>' in backend_cpp
    assert '&execute_with_backend<SecondInstruction>' in backend_cpp
    assert 'execute_impl may construct temporary operands' in backend_cpp
    assert 'ScopedIsaExecutionBackend scope(&execution_backend());' in backend_cpp
    assert '.operand_backend = Operand::full_execution_backend()' in backend_cpp
    assert 'execute_registered_' not in backend_cpp


def test_rdna4_operand_execution_backend_is_split_from_model_source(tmp_path):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='rdna4',
            generated_dir_name='rdna4',
            cpp_namespace='rdna4',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Rdna4Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_h = (tmp_path / 'rdna4' / 'operand.h').read_text()
    operand_cpp = (tmp_path / 'rdna4' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'rdna4' / 'operand_exec.cpp').read_text()

    assert 'class Operand : public IsaOperand<Isa>' in operand_h
    assert 'uint32_t Operand::read_scalar' in operand_cpp
    assert 'uint32_t Operand::read_scalar_exec' in operand_exec_cpp
    assert 'rocjitsu/vm/amdgpu/wavefront.h' not in operand_cpp
    assert 'rocjitsu/vm/amdgpu/wavefront.h' in operand_exec_cpp


def test_cdna1_split_operand_emits_simd_dispatch_methods(tmp_path):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='cdna1',
            generated_dir_name='cdna1',
            cpp_namespace='cdna1',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Cdna1Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_h = (tmp_path / 'cdna1' / 'operand.h').read_text()
    operand_cpp = (tmp_path / 'cdna1' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'cdna1' / 'operand_exec.cpp').read_text()

    assert 'bool simd_capable() const override;' in operand_h
    assert 'void read_lane_chunk(' in operand_h
    assert 'bool Operand::simd_capable() const' in operand_cpp
    assert 'bool Operand::simd_capable_exec() const' in operand_exec_cpp
    assert 'if (!reads_value())' in operand_exec_cpp
    assert 'if (!is_writable())' in operand_exec_cpp
    assert 'void Operand::read_lane_chunk_exec(' in operand_exec_cpp
    assert 'assert(lane_base <= wf.wf_size());' in operand_exec_cpp
    assert 'assert(count <= wf.wf_size() - lane_base);' in operand_exec_cpp
    assert 'write_lane_exec(wf, lane_base + i, vals[i]);' in (operand_exec_cpp)
    assert 'amdgpu::OperandExecutionAccess::raw_compute_unit' in operand_exec_cpp


class TestCdnaProfile:
    """CdnaProfile holds capabilities shared by CDNA3 and CDNA4."""

    def setup_method(self):
        self.p = CdnaProfile()

    def test_wave_size(self):
        assert self.p.wave_size == 64

    def test_wave_size_max_equals_wave_size(self):
        assert self.p.wave_size_max == 64

    def test_has_mfma(self):
        assert self.p.has_mfma is True

    def test_has_acc_vgpr(self):
        assert self.p.has_acc_vgpr is True

    def test_acc_vgpr_encoding_base(self):
        assert self.p.acc_vgpr_encoding_base == 768

    def test_max_acc_vgprs(self):
        assert self.p.max_acc_vgprs == 256

    def test_flat_scratch_mechanism(self):
        assert self.p.flat_scratch_mechanism == 'hwreg'

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX940_SC0_SC1_NT

    def test_has_wmma_false(self):
        assert self.p.has_wmma is False

    def test_swmmac_layout_none(self):
        assert self.p.swmmac_layout is SwmmacLayout.NONE

    def test_has_vopd_false(self):
        assert self.p.has_vopd is False

    def test_waitcnt_family_default(self):
        assert self.p.waitcnt_family == 'gfx9'

    def test_waitcnt_lgkmcnt_mask(self):
        assert self.p.waitcnt_lgkmcnt_mask == '0x0F'

    def test_d16_loads_zero_unselected_half(self):
        assert self.p.d16_loads_zero_unselected_half is True

    def test_field_renames_flat(self):
        renames = self.p.field_renames('ENC_FLAT')
        assert renames.get('sve') == 'lds'

    def test_field_renames_vop3p(self):
        renames = self.p.field_renames('ENC_VOP3P')
        assert renames.get('pad_14') == 'op_sel_hi_2'

    def test_field_renames_literal(self):
        assert self.p.field_renames('ENC_VOP2') == {'literal': 'simm32'}

    def test_literal_operand_uses_normalized_field_name(self):
        assert (
            self.p.normalize_operand_field_name('VOP2_INST_LITERAL', 'literal')
            == 'simm32'
        )

    def test_compound_mfma_is_not_a_family_default(self):
        assert self.p.mfma_scale_vop3px2_specs == ()
        assert self.p.inst_size_overrides == {}
        assert self.p.vop3px2_prefix_opcode is None


class TestCdna4Profile:
    def setup_method(self):
        self.p = Cdna4Profile()

    def test_compound_mfma_is_explicitly_enabled(self):
        assert tuple(spec.opcode for spec in self.p.mfma_scale_vop3px2_specs) == (
            45,
            46,
        )
        assert set(self.p.inst_size_overrides.values()) == {16}
        assert self.p.vop3px2_prefix_opcode == 0x2C


class TestCdna1Profile:
    """Cdna1Profile: no AccVGPR, GFX9_GLC coherency, sgpr_pair scratch."""

    def setup_method(self):
        self.p = Cdna1Profile()

    def test_has_acc_vgpr_false(self):
        assert self.p.has_acc_vgpr is False

    def test_acc_vgpr_encoding_base_zero(self):
        assert self.p.acc_vgpr_encoding_base == 0

    def test_max_acc_vgprs_zero(self):
        assert self.p.max_acc_vgprs == 0

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX9_GLC

    def test_flat_scratch_mechanism(self):
        assert self.p.flat_scratch_mechanism == 'sgpr_pair'

    def test_has_mfma_inherited(self):
        assert self.p.has_mfma is True

    def test_wave_size_inherited(self):
        assert self.p.wave_size == 64

    def test_d16_loads_preserve_unselected_half(self):
        assert self.p.d16_loads_zero_unselected_half is False


class TestCdna2Profile:
    """Cdna2Profile: AccVGPR base 512, GFX9_GLC coherency, sgpr_pair scratch."""

    def setup_method(self):
        self.p = Cdna2Profile()

    def test_has_acc_vgpr_true(self):
        assert self.p.has_acc_vgpr is True

    def test_acc_vgpr_encoding_base(self):
        assert self.p.acc_vgpr_encoding_base == 512

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX9_GLC

    def test_flat_scratch_mechanism(self):
        assert self.p.flat_scratch_mechanism == 'sgpr_pair'

    def test_d16_loads_zero_unselected_half(self):
        assert self.p.d16_loads_zero_unselected_half is True


class TestRdna1Profile:
    def setup_method(self):
        self.p = Rdna1Profile()

    def test_wave_size(self):
        assert self.p.wave_size == 32

    def test_wave_size_max(self):
        assert self.p.wave_size_max == 64

    def test_waitcnt_family(self):
        assert self.p.waitcnt_family == 'gfx10'

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC

    def test_waitcnt_lgkmcnt_mask(self):
        # RDNA1 uses a 6-bit LGKMCNT field (mask 0x3F)
        assert self.p.waitcnt_lgkmcnt_mask == '0x3F'

    def test_has_mfma_false(self):
        assert self.p.has_mfma is False


class TestRdna3Profile:
    def setup_method(self):
        self.p = Rdna3Profile()

    def test_wave_size(self):
        assert self.p.wave_size == 32

    def test_wave_size_max(self):
        assert self.p.wave_size_max == 64

    def test_waitcnt_family(self):
        assert self.p.waitcnt_family == 'gfx11'

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX11_SC0_SC1_TH

    def test_has_wmma(self):
        assert self.p.has_wmma is True

    def test_swmmac_layout_none(self):
        assert self.p.swmmac_layout is SwmmacLayout.NONE

    def test_has_vopd(self):
        assert self.p.has_vopd is True

    def test_has_vopd3_false(self):
        assert self.p.has_vopd3 is False

    def test_vopd_slot_opcodes(self):
        assert self.p.vopd_x_slot_opcodes == frozenset(range(14))
        assert self.p.vopd_y_slot_opcodes == frozenset((*range(14), 16, 17, 18))

    def test_sop1_base_condition_imports_as_default(self):
        cond = 'Nothas_lit_0_Nothas_lit_1'
        assert self.p.normalize_encoding_condition('ENC_SOP1', cond) == 'default'
        assert self.p.skip_inst_encoding('ENC_SOP1', cond) is False

    def test_operand_read64_zero_extends_simm32_literal(self, tmp_path):
        generator = CodeGenerator(
            SimpleNamespace(
                arch_name='rdna3',
                generated_dir_name='rdna3',
                cpp_namespace='rdna3',
                opnd_selectors=[],
                operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
                profile=Rdna3Profile(),
            ),
            str(tmp_path),
        )

        generator.gen_operand()
        operand_cpp = (tmp_path / 'rdna3' / 'operand.cpp').read_text()
        operand_exec_cpp = (tmp_path / 'rdna3' / 'operand_exec.cpp').read_text()

        assert (
            'if (opr_type == OperandType::OPR_SIMM32)\n'
            '    return static_cast<uint64_t>(static_cast<uint32_t>(ev));'
        ) in operand_exec_cpp
        assert 'return read_immediate64(opr_type_, ev);' in operand_exec_cpp
        assert (
            'return read_immediate64(opr_type_, encoding_value_);' in operand_exec_cpp
        )
        assert 'read_immediate64' not in operand_cpp


class TestRdna4Profile:
    def setup_method(self):
        self.p = Rdna4Profile()

    def test_wave_size(self):
        assert self.p.wave_size == 32

    def test_waitcnt_family(self):
        assert self.p.waitcnt_family == 'gfx12'

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX12_SCOPE_TH

    def test_has_wmma(self):
        assert self.p.has_wmma is True

    def test_swmmac_layout(self):
        assert self.p.swmmac_layout is SwmmacLayout.RUNTIME_WAVE

    def test_has_vopd(self):
        assert self.p.has_vopd is True

    def test_has_vopd3_false(self):
        assert self.p.has_vopd3 is False

    def test_sop1_base_condition_imports_as_default(self):
        cond = 'Nothas_lit_0_Nothas_lit_1'
        assert self.p.normalize_encoding_condition('ENC_SOP1', cond) == 'default'
        assert self.p.skip_inst_encoding('ENC_SOP1', cond) is False


class TestCdna5Profile:
    def setup_method(self):
        self.p = Cdna5Profile()

    def test_supported_versions(self):
        assert self.p.supported_versions == ['1.2.0']

    def test_wave_size_max(self):
        assert self.p.wave_size_max == 32

    def test_swmmac_layout(self):
        assert self.p.swmmac_layout is SwmmacLayout.FIXED_WAVE

    def test_matrix_layout(self):
        assert self.p.matrix_layout is MatrixLayout.WMMA_SPLIT_K

    def test_has_vopd3(self):
        assert self.p.has_vopd3 is True

    def test_vopd_slot_opcodes(self):
        assert self.p.vopd_x_slot_opcodes == frozenset(range(12))
        assert self.p.vopd_y_slot_opcodes == frozenset(
            (*range(12), 16, 17, *range(20, 25))
        )
        assert self.p.vopd3_x_slot_opcodes == frozenset(
            (0, *range(3, 12), 16, 17, *range(19, 23), *range(32, 37))
        )
        assert self.p.vopd3_y_slot_opcodes == frozenset(
            (0, *range(3, 12), *range(16, 25))
        )

    def test_generated_identities(self):
        assert self.p.generated_arch_name == 'cdna5'
        assert self.p.generated_dir_name == 'cdna5'
        assert self.p.cpp_namespace == 'cdna5'

    def test_field_renames_literal(self):
        assert self.p.field_renames('ENC_SOP1').get('literal') == 'simm32'

    def test_literal_operand_keeps_gfx1250_identity(self):
        assert (
            self.p.normalize_operand_field_name('VOP2_INST_LITERAL', 'literal')
            == 'literal'
        )

    def test_sop1_base_condition_imports_as_default(self):
        cond = '!has_lit64_0&!has_lit64_1&!has_lit_0&!has_lit_1'
        assert self.p.normalize_encoding_condition('ENC_SOP1', cond) == 'default'
        assert self.p.skip_inst_encoding('ENC_SOP1', cond) is False

    def test_sop1_literal_conditions_stay_skipped(self):
        assert self.p.skip_inst_encoding('SOP1_INST_LITERAL', 'has_lit_0') is True

    def test_compound_literal_parent_encoding(self):
        assert (
            self.p.derive_parent_enc_name('VOP3_SDST_ENC_INST_LITERAL')
            == 'VOP3_SDST_ENC'
        )

    def test_waitcnt_family(self):
        assert self.p.waitcnt_family == 'gfx12'

    def test_coherency_model(self):
        assert self.p.coherency_model == MemoryCoherencyModel.GFX12_SCOPE_TH

    def test_vgpr_msb_indexing(self):
        assert self.p.uses_vgpr_msb_indexing is True

    def test_d16_loads_zero_unselected_half(self):
        assert self.p.d16_loads_zero_unselected_half is True

    def test_source_split_limits_leave_precommit_margin(self):
        limits = self.p.source_split_max_bytes
        assert limits['ENC_VOP3'] <= 450 * 1024
        assert limits['ENC_VOPC'] <= 450 * 1024

    def test_source_split_file_stems_are_logical(self):
        assert self.p.source_split_file_stem('ENC_VOPC', 'V_CMP_LT_F32', None) == 'cmp'
        assert (
            self.p.source_split_file_stem('ENC_VOPC', 'V_CMPX_CLASS_F64', None)
            == 'cmpx'
        )
        assert (
            self.p.source_split_file_stem('ENC_VOP3', 'V_CVT_PK_FP8_F32', None) == 'cvt'
        )
        assert (
            self.p.source_split_file_stem(
                'ENC_VOP3',
                'V_MAD_CO_U64_U32',
                SimpleNamespace(data_type='u64'),
            )
            == 'alu'
        )
        assert (
            self.p.source_split_file_stem(
                'ENC_VOP3',
                'V_S_EXP_F32',
                SimpleNamespace(semantic_class='pseudo_scalar_unary'),
            )
            == 'alu'
        )

    def test_generated_source_split_file_matcher_is_scoped(self):
        units = [_SourceImplUnit('alu', ['impl']), _SourceImplUnit('cmpx', ['impl'])]

        assert CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_part1.cpp', units
        )
        assert CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_alu.cpp', units
        )
        assert CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_alu_2.cpp', units
        )
        assert CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_cmpx.cpp', units
        )
        assert not CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_helper.cpp', units
        )
        assert not CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_alu_helper.cpp', units
        )

        mixed_units = [
            _SourceImplUnit('alu', ['impl']),
            _SourceImplUnit(None, ['impl']),
        ]
        assert CodeGenerator._is_generated_source_split_file(
            'vop3', 'vop3_misc.cpp', mixed_units
        )

    def test_empty_execution_output_removes_stale_files(self, tmp_path):
        arch_name = self.p.generated_arch_name
        arch_dir = tmp_path / self.p.generated_dir_name
        arch_dir.mkdir()
        stale_files = [
            arch_dir / 'sopp_exec.cpp',
            arch_dir / 'sopp_exec_part1.cpp',
            arch_dir / 'sopp_exec_alu.cpp',
        ]
        for path in stale_files:
            path.write_text('stale')
        unrelated_file = arch_dir / 'sopp_exec_helper.cpp'
        unrelated_file.write_text('keep')

        gen = object.__new__(CodeGenerator)
        gen.out_path = str(tmp_path)
        gen.isa_spec = SimpleNamespace(
            arch_name=arch_name,
            generated_dir_name=self.p.generated_dir_name,
            cpp_namespace=self.p.cpp_namespace,
            profile=self.p,
        )
        gen._write_inst_impl_files(
            'ENC_SOPP',
            'sopp',
            [],
            _ImplOutputs(),
            _ImplOutputs(model=[_SourceImplUnit('alu', [])]),
        )

        assert all(not path.exists() for path in stale_files)
        assert unrelated_file.exists()

    def test_logical_source_chunks_put_extra_impls_in_support_chunk(self):
        gen = object.__new__(CodeGenerator)
        inst_impl = 'instruction'
        chunks = gen._build_logical_source_chunks(
            'vop3',
            ['support', inst_impl],
            [_SourceImplUnit('alu', [inst_impl])],
            max_bytes=1024,
            chunk_overhead=0,
        )

        assert chunks == [
            ('vop3_support', ['support']),
            ('vop3_alu', [inst_impl]),
        ]

    def test_logical_source_chunks_size_support_impls(self):
        gen = object.__new__(CodeGenerator)
        chunks = gen._build_logical_source_chunks(
            'vop3',
            ['support-one', 'support-two', 'instruction'],
            [_SourceImplUnit('alu', ['instruction'])],
            max_bytes=len('support-one\n\nsupport-two\n\n') - 1,
            chunk_overhead=0,
        )

        assert chunks[:2] == [
            ('vop3_support', ['support-one']),
            ('vop3_support_2', ['support-two']),
        ]

    def test_logical_source_chunks_reject_duplicate_filenames(self):
        gen = object.__new__(CodeGenerator)
        with pytest.raises(
            AssertionError, match='duplicate generated source file name'
        ):
            gen._build_logical_source_chunks(
                'vop3',
                ['first', 'second', 'third'],
                [
                    _SourceImplUnit('alu', ['first']),
                    _SourceImplUnit('alu', ['second']),
                    _SourceImplUnit('alu_2', ['third']),
                ],
                max_bytes=len('first\n\nsecond\n\n') - 1,
                chunk_overhead=0,
            )

    def test_detect_profile_uses_public_cdna5_architecture(self, tmp_path):
        xml = tmp_path / 'amdgpu_isa_cdna5.xml'
        xml.write_text(
            '<Spec><ISA><Architecture><ArchitectureName>AMD CDNA 5</ArchitectureName>'
            '</Architecture></ISA></Spec>'
        )
        assert _detect_profile(str(xml)) == 'cdna5'

    def test_parse_single_isa_arg_with_profile(self):
        assert _parse_isa_arg('cdna5:/tmp/isa.xml') == (
            'cdna5',
            '/tmp/isa.xml',
            'cdna5',
        )

    def test_parse_single_isa_arg_without_profile(self, tmp_path):
        xml = tmp_path / 'amdgpu_isa_cdna5.xml'
        xml.write_text(
            '<Spec><ISA><Architecture><ArchitectureName>AMD CDNA 5</ArchitectureName>'
            '</Architecture></ISA></Spec>'
        )
        assert _parse_isa_arg(str(xml)) == (None, str(xml), 'cdna5')

    @pytest.mark.parametrize('name', ['rdna3.5', 'rdna3_5'])
    def test_parse_single_isa_arg_accepts_rdna3_5_aliases(self, name):
        assert _parse_isa_arg(f'{name}:/tmp/isa.xml') == (
            name,
            '/tmp/isa.xml',
            'rdna3.5',
        )

    @pytest.mark.parametrize(
        'name',
        [
            '',
            '1250gfx',
            'gfx-1250',
            'gfx@1250',
            'nested/gfx1250',
            r'nested\gfx1250',
            '/tmp/escaped',
        ],
    )
    def test_parse_single_isa_arg_rejects_invalid_codegen_identity(
        self, name, tmp_path
    ):
        xml = tmp_path / 'amdgpu_isa_gfx1250.xml'
        xml.write_text('<Spec />')

        with pytest.raises(ValueError, match='invalid ISA name'):
            _parse_isa_arg(f'{name}:{xml}')

    def test_explicit_name_controls_codegen_identity(self):
        spec = SimpleNamespace(
            arch_name='cdna5',
            generated_dir_name='cdna5',
            cpp_namespace='cdna5',
        )

        _apply_codegen_identity(spec, 'gfx1250')

        assert spec.arch_name == 'cdna5'
        assert spec.generated_dir_name == 'gfx1250'
        assert spec.cpp_namespace == 'gfx1250'

    def test_codegen_include_paths_keep_handwritten_base_independent(self):
        config = _codegen_config(
            'lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/custom/generated',
            include_root='lib/rocjitsu/src',
        )

        assert config.include_base == 'rocjitsu/isa/arch/amdgpu'
        assert (
            config.generated_include_base == 'rocjitsu/isa/arch/amdgpu/custom/generated'
        )
        assert (
            config.shared_generated_include_base
            == 'rocjitsu/isa/arch/amdgpu/custom/generated'
        )

    def test_codegen_include_paths_allow_independent_handwritten_base(self):
        config = CodegenConfig.for_output(
            '/tmp/generated',
            handwritten_include_base='custom/amdgpu',
        )

        assert config.include_base == 'custom/amdgpu'
        assert config.generated_include_base == '/tmp/generated'

    def test_regeneration_helper_uses_stable_generated_include_prefix(self):
        rocjitsu = Path(__file__).resolve().parents[4]
        source_root = rocjitsu / 'lib' / 'rocjitsu' / 'src'
        isa_output = source_root / 'rocjitsu' / 'isa' / 'arch' / 'amdgpu' / 'generated'

        helper = (rocjitsu / 'scripts' / 'generate-amdisa.sh').read_text()
        assert '--include-root "$rocjitsu/lib/rocjitsu/src"' in helper

        config = _codegen_config(str(isa_output), include_root=str(source_root))
        assert config.generated_include_base == 'rocjitsu/isa/arch/amdgpu/generated'

    def test_explicit_codegen_prefix_remains_independent_of_output(self):
        config = CodegenConfig(generated_include_base='stable/generated')

        assert config.generated_include('rdna4', 'vop3.h') == (
            'stable/generated/rdna4/vop3.h'
        )

    def test_test_encoding_uses_primary_decode_key(self):
        generator = object.__new__(CodeGenerator)
        generator.isa_spec = SimpleNamespace(profile=SimpleNamespace(max_enc_bits=9))
        enc = InstEncoding(
            'ENC_SMEM',
            order=0,
            bit_cnt=64,
            enc_field_bit_cnt=6,
            op_field_bit_cnt=8,
            ucode_fields=[
                MicrocodeField('op', 8, 16),
                MicrocodeField('encoding', 6, 26),
            ],
            enc_conds=[],
        )
        enc.primary_dt_ptrs = [-1] * 256
        enc.primary_dt_ptrs[1] = 488
        inst = Instruction('S_LOAD_I8', 'ENC_SMEM', 1, [])

        assert generator._sample_test_encoding_words(enc, inst) == (
            0xF4010000,
            0x00000000,
        )

    def test_operand_read_lane64_preserves_literal64(self, tmp_path):
        generator = CodeGenerator(
            SimpleNamespace(
                arch_name='cdna5',
                generated_dir_name='cdna5',
                cpp_namespace='cdna5',
                opnd_selectors=[],
                operand_types=['OPR_SIMM32', 'OPR_SIMM64', 'OPR_VGPR'],
                profile=Cdna5Profile(),
            ),
            str(tmp_path),
        )

        generator.gen_operand()
        operand_cpp = (tmp_path / 'cdna5' / 'operand_exec.cpp').read_text()
        read_lane64 = operand_cpp[
            operand_cpp.index('uint64_t Operand::read_lane64') : operand_cpp.index(
                'void Operand::write_lane64'
            )
        ]

        assert (
            'if (has_literal64_)\n'
            '    return literal64_value_;\n'
            '  if (is_immediate_type(opr_type_))'
        ) in read_lane64


class TestMemoryCoherencyModelEnum:
    def test_all_five_values_exist(self):
        assert MemoryCoherencyModel.GFX9_GLC is not None
        assert MemoryCoherencyModel.GFX940_SC0_SC1_NT is not None
        assert MemoryCoherencyModel.GFX10_GLC_DLC_SLC is not None
        assert MemoryCoherencyModel.GFX11_SC0_SC1_TH is not None
        assert MemoryCoherencyModel.GFX12_SCOPE_TH is not None

    def test_values_are_distinct(self):
        vals = [m.value for m in MemoryCoherencyModel]
        assert len(vals) == len(set(vals))
