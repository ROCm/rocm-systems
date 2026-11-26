# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from amdisa.codegen.execute import conversion, fp8_formats
from amdisa.codegen.execute.vector_special import gen_vector_cvt_pk
from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.semantics import (
    INTEGER_SOURCE_CONVERSIONS,
    TO_FLOAT_CONVERSIONS,
    derive_semantics,
)


@pytest.mark.parametrize(
    'arch, policy, shared_narrowing, shared_widening',
    [
        ('cdna3', fp8_formats.Fp8Policy.FNUZ, False, False),
        ('cdna4', fp8_formats.Fp8Policy.OCP_UTIL, False, False),
        ('cdna5', fp8_formats.Fp8Policy.OCP_SHARED_NARROWING, True, False),
        ('rdna4', fp8_formats.Fp8Policy.OCP_SHARED, True, True),
        ('rdna3', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        ('cdna1', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        ('', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        (None, fp8_formats.DEFAULT_FP8_POLICY, False, False),
    ],
)
def test_fp8_policy_selects_narrowing_and_widening(
    arch, policy, shared_narrowing, shared_widening
):
    assert fp8_formats.fp8_policy(arch) is policy
    assert fp8_formats.uses_shared_fp8_narrowing(arch) == shared_narrowing
    assert fp8_formats.uses_shared_fp8_widening(arch) == shared_widening
    assert fp8_formats.uses_fnuz_fp8(arch) == (arch == 'cdna3')


@pytest.mark.parametrize('fmt, helper', [('fp8', 'fp8_e4m3'), ('bf8', 'bf8_e5m2')])
def test_cdna5_uses_shared_narrowing_for_non_scaled_ocp_conversions(fmt, helper):
    packed = gen_vector_cvt_pk(
        ['dst'],
        ['a', 'b'],
        'vector_cvt_pk',
        f'{fmt}_f32',
        arch_name='cdna5',
        is_vop3=True,
        fp8_format_select='inst_.clamp',
    )
    assert (
        f'amdgpu::conversion_to_fp8<amdgpu::conversion::{fmt.title()}>(wf, 0u, inst_.neg)'
        in packed
    )
    assert ('util::f32_to_fp8_e5m3_rne_mode' in packed) == (fmt == 'fp8')
    widened = gen_vector_cvt_pk(
        ['dst'], ['a'], 'vector_cvt_pk', f'f32_{fmt}', arch_name='cdna5'
    )
    assert f'util::{helper}_to_f32' in widened


"""Tests for legacy FP8 format-policy selection."""
