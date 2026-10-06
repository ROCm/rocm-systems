# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.isa_profile import Cdna4Profile


@pytest.mark.parametrize(
    ('name', 'size', 'elements'),
    [
        ('DS_WRITE_B32', 4, 1),
        ('DS_WRITE_B64', 8, 1),
        ('DS_WRITE_B96', 4, 3),
        ('DS_WRITE_B128', 4, 4),
    ],
)
def test_ds_dword_store_snapshots_observed_registers(name, size, elements):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='cdna4', profile=Cdna4Profile())
    codegen._current_inst_fields = {'acc'}
    semantics = SimpleNamespace(
        name=name, elem_size=size, num_elems=elements, d16_hi=False
    )
    body = codegen._gen_ds_write([], [], semantics)
    assert f'read_vgpr_region(data_base, {size * elements // 4}, exec)' in body
    assert 'data.copy_dwords_lane_major(d->store_data, exec);' in body
    assert '(inst_.acc ? 256u : 0u)' in body
    assert body.index('ds_calculate_addresses(') < body.index('read_vgpr_region(')
    assert body.index('if (data.valid())') < body.index('copy_dwords_lane_major(')
    snapshot = body[body.index('if (data.valid())') : body.index('for (uint32_t lane')]
    assert 'set_data(std::move(d));\n      return;' in snapshot
    # The scalar reads remain available for a partially backed register range.
    assert 'RegisterAccess(cu).read_vgpr(' in body


@pytest.mark.parametrize(
    ('name', 'size'), [('DS_WRITE_B8_D16_HI', 1), ('DS_WRITE_B16_D16_HI', 2)]
)
def test_ds_subword_store_retains_selected_byte_observations(name, size):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='cdna4', profile=Cdna4Profile())
    semantics = SimpleNamespace(name=name, elem_size=size, num_elems=1, d16_hi=True)
    body = codegen._gen_ds_write([], [], semantics)
    assert 'copy_dwords_lane_major' not in body
    assert f'read_vgpr(data_base, lane, {((1 << size) - 1) << 2:#x})' in body
    assert 'val0 >>= 16;' in body
