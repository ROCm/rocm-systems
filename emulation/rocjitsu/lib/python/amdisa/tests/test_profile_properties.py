# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ISA dimension properties on IsaProfile subclasses."""

import pytest

from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    CdnaProfile,
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
