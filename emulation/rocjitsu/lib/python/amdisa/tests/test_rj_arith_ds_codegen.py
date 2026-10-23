# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import pytest

from amdisa.codegen.execute.rj_arith_ds import DS_NAMES, gen_ds


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(DS_NAMES))
def test_ds_routing_preserves_offsets_and_selects_wave64_architecture(arch, name):
    sources = ['addr'] if name == 'DS_SWIZZLE_B32' else ['addr', 'data0']
    body = gen_ds(arch, name, ['vdst'], sources)
    if arch == 'rdna3' and name == 'DS_BPERMUTE_FI_B32':
        assert body is None
        return
    assert 'inst_.offset0 | (inst_.offset1 << 8)' in body
    assert 'amdgpu::arith::execute_ds_' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status' not in body
    assert 'VectorMemState' not in body
    if name != 'DS_SWIZZLE_B32':
        suffix = '_w64_gfx12' if arch == 'rdna4' and '_FI_' not in name else '_w64'
        assert 'rj_arith_' + name.lower() + suffix in body
    assert gen_ds('cdna5', name, ['vdst'], sources) is None
