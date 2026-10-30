# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sparse matrix bindings and metadata-selector normalization."""

from rj_arith_headers import public_headers

from types import SimpleNamespace
import re

import pytest

from amdisa.codegen.execute.rj_arith_sparse_matrix import gen_sparse_wmma


def context(result, depth, factors, arch='rdna4'):
    return SimpleNamespace(
        arch_name=arch,
        profile=SimpleNamespace(uses_rj_arith_matrix=arch in ('rdna3', 'rdna4')),
        inst=SimpleNamespace(name=f'V_SWMMAC_{result}_16X16X{depth}_{factors}'),
        dst_ops=['vdst'],
        src_ops=['src0', 'src1', 'src2'],
    )


@pytest.mark.parametrize(
    'result,depth,factors',
    [
        ('F32', 32, 'F16'),
        ('F32', 32, 'BF16'),
        ('F16', 32, 'F16'),
        ('BF16', 32, 'BF16'),
        ('F32', 32, 'FP8_FP8'),
        ('F32', 32, 'FP8_BF8'),
        ('F32', 32, 'BF8_FP8'),
        ('F32', 32, 'BF8_BF8'),
        ('I32', 32, 'IU8'),
        ('I32', 32, 'IU4'),
        ('I32', 64, 'IU4'),
    ],
)
def test_sparse_forms_bind_both_wave_widths_and_preserve_operands(
    result, depth, factors
):
    ctx = context(result, depth, factors)
    body = gen_sparse_wmma(ctx)
    api = 'rj_arith_' + ctx.inst.name.lower()
    assert f'lanes == 32 ? {api}_w32 : {api}_w64' in body
    header = public_headers()
    for function in (api + '_w32', api + '_w64'):
        assert re.search(r'\bint ' + function + r'\(', header)
    assert 'flags, modifiers, dst, src0_base, src1_base, index_base,' in body
    assert 'resolved_vgpr_offset' not in body
    assert 'exec_mask' not in body
    assert 'const_acc' not in body
    assert 'arithmetic_flags()' in body


def test_sparse_index_aliases_are_normalized_before_rj_arith_validation():
    assert 'inst_.opsel & (lanes == 32 ? 0u : 1u)' in gen_sparse_wmma(
        context('I32', 64, 'IU4')
    )
    assert 'inst_.opsel & (1u)' in gen_sparse_wmma(context('I32', 32, 'IU4'))
    assert 'inst_.opsel & (lanes == 32 ? 1u : 3u)' in gen_sparse_wmma(
        context('F32', 32, 'F16')
    )


def test_other_architectures_and_unmodeled_shapes_keep_existing_dispatch():
    for arch in ('rdna3', 'rdna3_5', 'cdna5'):
        assert gen_sparse_wmma(context('F32', 32, 'F16', arch)) is None
    assert gen_sparse_wmma(context('F32', 64, 'F16')) is None
    assert gen_sparse_wmma(context('I32', 64, 'IU8')) is None


def test_sparse_half_types_forward_tied_accumulator_modifiers():
    for result, factors in [
        ('F32', 'F16'),
        ('F32', 'BF16'),
        ('F16', 'F16'),
        ('BF16', 'BF16'),
    ]:
        body = gen_sparse_wmma(context(result, 32, factors))
        assert '(inst_.neg & 7u) | (uint64_t(inst_.neg_hi & 7u) << 3)' in body
    body = gen_sparse_wmma(context('F32', 32, 'FP8_FP8'))
    assert 'inst_.neg' not in body
