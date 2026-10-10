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
    'changes',
    [
        pytest.param({'name': 'S_BUFFER_ATOMIC_DEC'}, id='buffer-mnemonic'),
        pytest.param({'name': 'S_ATOMIC_DEC_X2'}, id='x2-mnemonic'),
        pytest.param({'operation': 'inc'}, id='other-operation'),
        pytest.param({'operation': 'unknown'}, id='unknown-operation'),
        pytest.param({'operation': None}, id='missing-operation'),
        pytest.param({'elem_size': 8}, id='element-width'),
        pytest.param({'num_elems': 2}, id='register-count'),
        pytest.param({'elem_size': 8, 'num_elems': 2}, id='64-bit-payload'),
    ],
)
def test_smem_atomic_rejects_unsupported_contract(changes):
    sem = InstructionSemantics(
        'S_ATOMIC_DEC', 'smem_atomic', operation='dec', elem_size=4, num_elems=1
    )
    # Exercise the emitter directly: derivation currently filters these inputs.
    sem = replace(sem, **changes)

    with pytest.raises(ValueError, match='supports only 32-bit S_ATOMIC_DEC'):
        _codegen('cdna4', Cdna4Profile())._gen_smem_atomic(sem)
