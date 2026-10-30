# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate the RDNA-only rj-arith selection and public API bindings."""

from rj_arith_headers import public_headers

from types import SimpleNamespace
import re

import pytest

from amdisa.codegen.execute.rj_arith_matrix import gen_dense_wmma


def context(name, arch):
    return SimpleNamespace(
        arch_name=arch,
        profile=SimpleNamespace(uses_rj_arith_matrix=arch in ('rdna3', 'rdna4')),
        inst=SimpleNamespace(name=name),
        dst_ops=['vdst'],
        src_ops=['src0', 'src1', 'src2'],
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize(
    'result,factors',
    [
        ('F32', 'F16'),
        ('F32', 'BF16'),
        ('F16', 'F16'),
        ('BF16', 'BF16'),
        ('I32', 'IU8'),
        ('I32', 'IU4'),
    ],
)
def test_shared_dense_forms_bind_both_public_wave_widths(arch, result, factors):
    name = f'V_WMMA_{result}_16X16X16_{factors}'
    body = gen_dense_wmma(context(name, arch))
    suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    assert f'lanes == 32 ? {api}_w32{suffix} : {api}_w64{suffix}' in body
    header = public_headers()
    declarations = header
    for function in (api + '_w32' + suffix, api + '_w64' + suffix):
        assert re.search(r'\bint ' + function + r'\(', declarations)
    assert 'exec_mask' not in body
    assert 'arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['rdna3_5', 'cdna4', 'cdna5'])
def test_other_architectures_keep_their_existing_generators(arch):
    assert gen_dense_wmma(context('V_WMMA_F32_16X16X16_F16', arch)) is None


def test_packed_rdna3_preserves_half_selection_and_inline_replication():
    body = gen_dense_wmma(context('V_WMMA_F16_16X16X16_F16', 'rdna3'))
    assert 'RJ_ARITH_WMMA_HIGH_C_D' in body
    assert 'high ? 0xc : 0x3' in body
    assert 'inline_word<false>' in body
    assert '8, 256 / lanes' in body


def test_rdna4_packed_outputs_use_adjacent_rows():
    body = gen_dense_wmma(context('V_WMMA_BF16_16X16X16_BF16', 'rdna4'))
    assert 'HIGH_C_D' not in body
    assert '128 / lanes' in body
    assert 'inline_word<true>' in body


def test_wave64_iu4_retains_its_single_partial_factor_register():
    body = gen_dense_wmma(context('V_WMMA_I32_16X16X16_IU4', 'rdna4'))
    assert '(32 + lanes - 1) / lanes, 256 / lanes' in body
    assert 'std::nullopt, 0xf, 32);' in body


@pytest.mark.parametrize(
    'arch,name',
    [
        ('rdna2', 'V_WMMA_F32_16X16X16_F16'),
        ('rdna3_5', 'V_WMMA_F32_16X16X16_F16'),
        ('cdna5', 'V_WMMA_F32_16X16X16_F16'),
        ('rdna3', 'V_WMMA_F32_16X16X16_FP8_FP8'),
        ('rdna3', 'V_WMMA_I32_16X16X32_IU4'),
        ('rdna4', 'V_WMMA_F16_16X16X16_BF16'),
        ('rdna4', 'V_WMMA_F32_16X16X32_F16'),
        ('rdna4', 'V_MFMA_F32_16X16X16_F16'),
    ],
)
def test_unsupported_architectures_and_formats_keep_existing_dispatch(arch, name):
    assert gen_dense_wmma(context(name, arch)) is None
