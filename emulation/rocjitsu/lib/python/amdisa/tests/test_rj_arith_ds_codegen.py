# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


from pathlib import Path


import pytest


from amdisa.codegen.execute.rj_arith_ds import DS_ARCHITECTURES, DS_NAMES, gen_ds


def context(arch, name, dst, src):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name='',
        is_vop3=False,
        has_abs=False,
    )


@pytest.mark.parametrize('arch', DS_ARCHITECTURES)
@pytest.mark.parametrize('name', sorted(DS_NAMES))
def test_ds_routing_preserves_offsets_and_selects_wave64_architecture(arch, name):
    sources = ['addr'] if name == 'DS_SWIZZLE_B32' else ['addr', 'data0']
    body = gen_ds(context(arch, name, ['vdst'], sources))
    if arch not in ('rdna4', 'cdna5') and name == 'DS_BPERMUTE_FI_B32':
        assert body is None
        return
    assert 'inst_.offset0 | (inst_.offset1 << 8)' in body
    assert 'amdgpu::arith::execute_ds_' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status' not in body
    assert 'VectorMemState' not in body
    if name != 'DS_SWIZZLE_B32':
        suffix = (
            '_w64_gfx12'
            if arch not in ('rdna1', 'rdna2', 'rdna3', 'rdna3_5') and '_FI_' not in name
            else '_w64'
        )
        assert 'rj_arith_' + name.lower() + suffix in body
    assert gen_ds(context('unknown', name, ['vdst'], sources)) is None


@pytest.mark.parametrize('arch', DS_ARCHITECTURES)
def test_generated_ds_register_operations_use_rj_arith_but_memory_stays_local(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    encoding = 'vds' if arch in ('rdna4', 'cdna5') else 'ds'
    body = (root / arch / f'{encoding}_exec.cpp').read_text()
    assert body.count('amdgpu::arith::execute_ds_swizzle(') == 1
    assert body.count('amdgpu::arith::execute_ds_permute<') == (
        3 if arch in ('rdna4', 'cdna5') else 2
    )
    assert 'amdgpu::execute_ds_swizzle_b32_' not in body
    assert 'amdgpu::execute_ds_permute_b32_' not in body
    assert 'amdgpu::execute_ds_bpermute_b32_' not in body
    assert 'amdgpu::VectorMemState' in body
