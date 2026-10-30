# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


def context(arch, name, dst, src):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name="",
        is_vop3=False,
        has_abs=False,
    )


import re


import pytest


from amdisa.codegen.execute.rj_arith_scalar import (
    BITSET,
    COMPARISONS,
    FLOAT_COMPARISONS,
    FLOAT_ARITHMETIC,
    FLOAT_ROUNDING,
    IMMEDIATE,
    SCALAR,
    gen_scalar,
)


@pytest.mark.parametrize('name', sorted(IMMEDIATE))
def test_immediate_arithmetic_reads_old_destination_before_commit(name):
    body = gen_scalar(context('rdna3', name, ['dst'], ['imm']))
    assert body.index('read_scalar(dst)') < body.index('status != RJ_ARITH_SUCCESS')
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
        'write_scalar(dst, result)'
    )
    assert 'uint16_t(registers.read_scalar(imm))' in body
    assert ('wf.write_scc(scc)' in body) == IMMEDIATE[name]
    assert 'wf.read_scc' not in body


@pytest.mark.parametrize('name,bits', sorted(BITSET.items()))
def test_bitset_reads_full_destination_and_word_index_without_scc(name, bits):
    suffix = '64' if bits == 64 else ''
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(context(arch, name, ['dst'], ['index']))
        assert f'uint{bits}_t result = registers.read_scalar{suffix}(dst)' in body
        assert 'uint32_t bit = registers.read_scalar(index)' in body
        assert f'rj_arith_{name.lower()}(' in body
        assert '&result, bit)' in body
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
        assert 'scc' not in body
    assert gen_scalar(context('cdna5', name, ['dst'], ['index'])) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_scalar_selection_bit_updates_and_masks_route_to_rj_arith(arch):
    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    text = '\n'.join(
        (generated / f'{encoding}_exec.cpp').read_text()
        for encoding in ('sop1', 'sop2')
    )
    names = set(BITSET)
    names.update(
        f'S_{op}_B{bits}' for op in ('CSELECT', 'QUADMASK', 'WQM') for bits in (32, 64)
    )
    if arch == 'rdna4':
        names.add('S_MUL_U64')
    for name in names:
        assert text.count(f'rj_arith_{name.lower()}(') == 1, name
        assert f'amdgpu::execute_{name.lower()}_' not in text, name


def test_generated_rdna4_scalar_float_comparisons_and_rounding_route_to_rj_arith():
    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/rdna4'
    )
    text = '\n'.join(
        (generated / f'{encoding}_exec.cpp').read_text()
        for encoding in ('sop1', 'sopc')
    )
    for name in FLOAT_COMPARISONS | FLOAT_ROUNDING:
        assert text.count(f'rj_arith_{name.lower()}(') == 1, name
        assert f'amdgpu::execute_{name.lower()}_' not in text, name


def test_generated_rdna4_scalar_float_arithmetic_routes_to_rj_arith():
    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/rdna4'
    )
    text = '\n'.join(
        (generated / f'{encoding}_exec.cpp').read_text()
        for encoding in ('sop1', 'sop2')
    )
    for name in FLOAT_ARITHMETIC:
        assert text.count(f'rj_arith_{name.lower()}(') == 1, name
        assert f'amdgpu::execute_{name.lower()}_' not in text, name


@pytest.mark.parametrize('name', ['S_ROUND_MODE', 'S_DENORM_MODE'])
def test_guest_mode_updates_are_functional_state_independent_of_exec(name):
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(context(arch, name, [], ['simm16']))
        assert f'rj_arith_{name.lower()}(' in body
        assert 'uint16_t(simm16.encoding_value()), &mode' in body
        assert body.index('wf.mode_raw()') < body.index('status != RJ_ARITH_SUCCESS')
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'wf.set_mode_raw(mode)'
        )
        assert 'arithmetic_reports_fp_status' not in body
        assert 'wf.exec()' not in body
    assert gen_scalar(context('cdna5', name, [], ['simm16'])) is None
