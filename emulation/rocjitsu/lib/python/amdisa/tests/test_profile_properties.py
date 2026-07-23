# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ISA dimension properties on IsaProfile subclasses."""

from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import (
    CodeGenerator,
    _ImplOutputs,
    _SourceImplUnit,
)
from amdisa.__main__ import _detect_profile
from amdisa.gpuisa import InstEncoding, Instruction, MicrocodeField
from amdisa.isa_properties_codegen import emit_isa_properties
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    CdnaProfile,
    Gfx1250Profile,
    MemoryCoherencyModel,
    Rdna1Profile,
    Rdna3Profile,
    Rdna4Profile,
)


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), False),
        (Rdna1Profile(), True),
        (Rdna3Profile(), True),
        (Rdna4Profile(), True),
        (Gfx1250Profile(), False),
    ],
)
def test_supports_wgp_mode(profile, expected):
    assert profile.supports_wgp_mode is expected


@pytest.mark.parametrize(
    ('profile', 'uses_ttmp', 'uses_cluster_ttmp'),
    [
        (CdnaProfile(), False, False),
        (Rdna4Profile(), True, False),
        (Gfx1250Profile(), True, True),
    ],
)
def test_ttmp_workgroup_id_properties(profile, uses_ttmp, uses_cluster_ttmp):
    assert profile.uses_ttmp_workgroup_ids is uses_ttmp
    assert profile.uses_cluster_ttmp_workgroup_ids is uses_cluster_ttmp


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), True),
        (Rdna1Profile(), False),
        (Rdna3Profile(), False),
        (Rdna4Profile(), False),
        (Gfx1250Profile(), False),
    ],
)
def test_descriptor_sgpr_count_encoded(profile, expected):
    assert profile.descriptor_sgpr_count_encoded is expected


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (CdnaProfile(), 256),
        (Rdna4Profile(), 256),
        (Gfx1250Profile(), 1024),
    ],
)
def test_max_addressable_vgprs_per_wf(profile, expected):
    assert profile.max_addressable_vgprs_per_wf == expected


def test_only_gfx1250_splits_execution_sources():
    assert Gfx1250Profile().split_execution_sources
    assert not Rdna4Profile().split_execution_sources


def test_non_split_generation_leaves_exec_named_sources_untouched(tmp_path):
    arch_dir = tmp_path / 'rdna4'
    arch_dir.mkdir()
    exec_named_source = arch_dir / 'sopp_exec.cpp'
    exec_named_source.write_text('user-owned source')

    generator = object.__new__(CodeGenerator)
    generator.out_path = str(tmp_path)
    generator.isa_spec = SimpleNamespace(arch_name='rdna4', profile=Rdna4Profile())
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
        (Gfx1250Profile(), 'ENC_VFLAT', 'OPR_SREG_NULL'),
        (Gfx1250Profile(), 'ENC_VGLOBAL', 'OPR_SREG_NULL'),
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
        ('gfx1250', SimpleNamespace(profile=Gfx1250Profile()), None),
    ]

    output = emit_isa_properties(str(tmp_path), specs).read_text()

    assert 'uint32_t max_addressable_vgprs_per_wf = 0;' in output
    assert 'MAX_SUPPORTED_ADDRESSABLE_VGPRS_PER_WF = 1024;' in output
    assert (
        'case ROCJITSU_CODE_ARCH_CDNA3:\n'
        '    return {\n'
        '        .supports_wgp_mode = false,\n'
        '        .descriptor_sgpr_count_encoded = true,\n'
        '        .uses_ttmp_workgroup_ids = false,\n'
        '        .uses_cluster_ttmp_workgroup_ids = false,\n'
        '        .max_addressable_vgprs_per_wf = 256,\n'
        '    };'
    ) in output
    assert (
        'case ROCJITSU_CODE_ARCH_RDNA4:\n'
        '    return {\n'
        '        .supports_wgp_mode = true,\n'
        '        .descriptor_sgpr_count_encoded = false,\n'
        '        .uses_ttmp_workgroup_ids = true,\n'
        '        .uses_cluster_ttmp_workgroup_ids = false,\n'
        '        .max_addressable_vgprs_per_wf = 256,\n'
        '    };'
    ) in output
    assert (
        'case ROCJITSU_CODE_ARCH_GFX1250:\n'
        '    return {\n'
        '        .supports_wgp_mode = false,\n'
        '        .descriptor_sgpr_count_encoded = false,\n'
        '        .uses_ttmp_workgroup_ids = true,\n'
        '        .uses_cluster_ttmp_workgroup_ids = true,\n'
        '        .max_addressable_vgprs_per_wf = 1024,\n'
        '    };'
    ) in output


@pytest.mark.parametrize(
    ('arch', 'profile', 'raw_exec'),
    [
        ('cdna3', CdnaProfile(), False),
        ('rdna4', Rdna4Profile(), True),
    ],
)
def test_operand_exec_register_access_is_wave32_gated(
    tmp_path, arch, profile, raw_exec
):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name=arch,
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=profile,
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_cpp = (tmp_path / arch / 'operand.cpp').read_text()

    assert 'return static_cast<uint32_t>(wf.exec());' in operand_cpp
    assert 'wf.set_exec((wf.exec() & 0xFFFFFFFF00000000ULL) | val);' in operand_cpp
    if raw_exec:
        assert 'return static_cast<uint32_t>(wf.exec_raw() >> 32);' in operand_cpp
        assert 'wf.set_exec_raw((wf.exec_raw() & 0x00000000FFFFFFFFULL)' in operand_cpp
    else:
        assert 'exec_raw' not in operand_cpp
        assert 'set_exec_raw' not in operand_cpp


def test_gfx1250_operand_execution_backend_uses_separate_source(tmp_path):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='gfx1250',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Gfx1250Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_h = (tmp_path / 'gfx1250' / 'operand.h').read_text()
    operand_cpp = (tmp_path / 'gfx1250' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'gfx1250' / 'operand_exec.cpp').read_text()

    assert 'class Operand : public IsaOperand<Isa>' in operand_h
    assert 'ROCJITSU_ISA_MODEL_ONLY' not in operand_h
    assert ': IsaOperand<Isa>(size_bits, opr_type, encoding_value)' in operand_cpp
    assert 'ROCJITSU_ISA_MODEL_ONLY' not in operand_cpp
    assert 'uint32_t Operand::read_scalar' in operand_cpp
    assert 'void Operand::require_execution_backend()' in operand_cpp
    assert '!backend.simd_notify_read64_mut' in operand_cpp
    assert 'uint32_t Operand::read_scalar_exec' not in operand_cpp
    assert 'rocjitsu/vm/amdgpu/compute_unit.h' not in operand_cpp
    assert 'uint32_t Operand::read_scalar_exec' in operand_exec_cpp
    assert 'Operand::execution_backend_registered_' in operand_exec_cpp
    assert 'rocjitsu/vm/amdgpu/compute_unit.h' in operand_exec_cpp


def test_rdna4_operand_execution_backend_stays_in_common_source(tmp_path):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='rdna4',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Rdna4Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_h = (tmp_path / 'rdna4' / 'operand.h').read_text()
    operand_cpp = (tmp_path / 'rdna4' / 'operand.cpp').read_text()

    assert 'class Operand : public AmdgpuIsaOperand<Isa>' in operand_h
    assert 'uint32_t Operand::read_scalar' in operand_cpp
    assert not (tmp_path / 'rdna4' / 'operand_exec.cpp').exists()


class TestCdnaProfile:
    """CdnaProfile represents CDNA3 and CDNA4."""

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

    def test_has_vopd_false(self):
        assert self.p.has_vopd is False

    def test_waitcnt_family_default(self):
        assert self.p.waitcnt_family == 'gfx9'

    def test_waitcnt_lgkmcnt_mask(self):
        assert self.p.waitcnt_lgkmcnt_mask == '0x0F'

    def test_field_renames_flat(self):
        renames = self.p.field_renames('ENC_FLAT')
        assert renames.get('sve') == 'lds'

    def test_field_renames_vop3p(self):
        renames = self.p.field_renames('ENC_VOP3P')
        assert renames.get('pad_14') == 'op_sel_hi_2'

    def test_field_renames_other_enc_empty(self):
        assert self.p.field_renames('ENC_VOP2') == {}


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

    def test_has_vopd(self):
        assert self.p.has_vopd is True

    def test_has_vopd3_false(self):
        assert self.p.has_vopd3 is False

    def test_operand_read64_zero_extends_simm32_literal(self, tmp_path):
        generator = CodeGenerator(
            SimpleNamespace(
                arch_name='rdna3',
                opnd_selectors=[],
                operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
                profile=Rdna3Profile(),
            ),
            str(tmp_path),
        )

        generator.gen_operand()
        operand_cpp = (tmp_path / 'rdna3' / 'operand.cpp').read_text()

        assert (
            'if (opr_type == OperandType::OPR_SIMM32)\n'
            '    return static_cast<uint64_t>(static_cast<uint32_t>(ev));'
        ) in operand_cpp
        assert 'return read_immediate64(opr_type_, ev);' in operand_cpp
        assert 'return read_immediate64(opr_type_, encoding_value_);' in operand_cpp


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

    def test_has_vopd(self):
        assert self.p.has_vopd is True

    def test_has_vopd3_false(self):
        assert self.p.has_vopd3 is False


class TestGfx1250Profile:
    def setup_method(self):
        self.p = Gfx1250Profile()

    def test_supported_versions(self):
        assert self.p.supported_versions == ['1.2.0']

    def test_wave_size_max(self):
        assert self.p.wave_size_max == 32

    def test_has_vopd3(self):
        assert self.p.has_vopd3 is True

    def test_generated_arch_name(self):
        assert self.p.generated_arch_name == 'gfx1250'

    def test_field_renames_literal(self):
        assert self.p.field_renames('ENC_SOP1').get('literal') == 'simm32'

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
        arch_dir = tmp_path / arch_name
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
        gen.isa_spec = SimpleNamespace(arch_name=arch_name, profile=self.p)
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

    def test_hwreg_ids(self):
        assert self.p.hwreg_mode_id == 1
        assert self.p.hwreg_status_id == 2
        assert self.p.hwreg_ib_sts2_id == 28

    def test_detect_profile_uses_filename_override(self, tmp_path):
        xml = tmp_path / 'amdgpu_isa_gfx1250.xml'
        xml.write_text('<Spec />')
        assert _detect_profile(str(xml)) == 'gfx1250'

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
                arch_name='gfx1250',
                opnd_selectors=[],
                operand_types=['OPR_SIMM32', 'OPR_SIMM64', 'OPR_VGPR'],
                profile=Gfx1250Profile(),
            ),
            str(tmp_path),
        )

        generator.gen_operand()
        operand_cpp = (tmp_path / 'gfx1250' / 'operand_exec.cpp').read_text()
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
