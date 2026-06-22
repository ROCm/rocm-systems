# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ISA dimension properties on IsaProfile subclasses."""

from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import CodeGenerator, _SourceImplUnit
from amdisa.__main__ import _detect_profile
from amdisa.gpuisa import InstEncoding, Instruction, MicrocodeField
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
            ('vop3_support_1', ['support-one']),
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
                    _SourceImplUnit('alu_1', ['third']),
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
        operand_cpp = (tmp_path / 'gfx1250' / 'operand.cpp').read_text()
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
