# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace
from types import SimpleNamespace

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    CdnaProfile,
)
from amdisa.semantics import InstructionSemantics, derive_semantics


def _codegen(arch_name, profile):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name=arch_name, profile=profile)
    return codegen


@pytest.mark.parametrize(
    ('arch_name', 'profile'),
    [
        ('cdna1', Cdna1Profile()),
        ('cdna2', Cdna2Profile()),
        ('cdna3', CdnaProfile()),
        ('cdna4', Cdna4Profile()),
    ],
)
@pytest.mark.parametrize('name', ['S_ATOMIC_DEC', 's_atomic_dec'])
def test_smem_atomic_accepts_derived_global_decrement(arch_name, profile, name):
    sem = derive_semantics(name, 'ENC_SMEM')
    assert sem is not None
    assert sem.semantic_class == 'smem_atomic'

    body = _codegen(arch_name, profile)._gen_smem_atomic(sem)

    assert 'd->atomic_op = amdgpu::AtomicOp::DEC;' in body
    assert 'd->elem_size = 4;' in body
    assert 'd->num_dwords = 1;' in body
    assert 'smem_calculate_address(inst_, wf);' in body


@pytest.mark.parametrize(
    ('name', 'operation', 'elem_size', 'num_elems', 'op_enum'),
    [
        ('S_ATOMIC_DEC', 'dec', 4, 1, 'DEC'),
        ('S_ATOMIC_DEC_X2', 'dec', 8, 2, 'DEC'),
        ('S_ATOMIC_CMPSWAP', 'cmpswap', 4, 2, 'CMPSWAP'),
        ('S_ATOMIC_CMPSWAP_X2', 'cmpswap', 8, 4, 'CMPSWAP'),
    ],
)
def test_smem_atomic_payload_and_return_widths_follow_semantics(
    name, operation, elem_size, num_elems, op_enum
):
    sem = InstructionSemantics(
        name,
        'smem_atomic',
        operation=operation,
        elem_size=elem_size,
        num_elems=num_elems,
    )

    body = _codegen('cdna4', Cdna4Profile())._gen_smem_atomic(sem)

    return_dwords = elem_size // 4
    assert f'd->elem_size = {elem_size};' in body
    assert f'resolve_scalar_register_range(wf, inst_.sdata, {num_elems}u);' in body
    assert f'd->num_dwords = {return_dwords};' in body
    assert f'd->atomic_op = amdgpu::AtomicOp::{op_enum};' in body
    last = num_elems - 1
    assert (
        f'd->store_data[{last}] = amdgpu::read_scalar_register(wf, *data_register, {last});'
        in body
    )
    assert f'd->store_data[{num_elems}]' not in body
    narrowed = f'd->dst_register.width = {return_dwords};'
    assert (narrowed in body) == (return_dwords != num_elems)


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        pytest.param(
            {'name': 'S_BUFFER_ATOMIC_DEC'},
            'supports only global S_ATOMIC_',
            id='buffer-mnemonic',
        ),
        pytest.param(
            {'operation': 'unknown'},
            'unsupported smem_atomic operation',
            id='unknown-operation',
        ),
        pytest.param(
            {'operation': None},
            'unsupported smem_atomic operation',
            id='missing-operation',
        ),
    ],
)
def test_smem_atomic_rejects_unsupported_contract(changes, message):
    sem = InstructionSemantics(
        'S_ATOMIC_DEC', 'smem_atomic', operation='dec', elem_size=4, num_elems=1
    )
    # Exercise the emitter directly: derivation currently filters these inputs.
    sem = replace(sem, **changes)

    with pytest.raises(ValueError, match=message):
        _codegen('cdna4', Cdna4Profile())._gen_smem_atomic(sem)
