# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from rj_arith_headers import public_headers

from pathlib import Path
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


@pytest.mark.parametrize('name', sorted(SCALAR))
def test_scalar_bindings_match_public_api_widths_and_scc(name):
    shape = SCALAR[name]
    header = public_headers()
    declaration = re.search(r'\bint rj_arith_' + name.lower() + r'\(([^;]+)\);', header)
    assert declaration is not None
    types = [t.strip() for t in re.findall(r'uint\d+_t\s*\*?', declaration[1])]
    expected = ['uint64_t', 'uint64_t', f'uint{shape.result_bits}_t *']
    expected += [f'uint{bits}_t' for bits in shape.input_bits]
    if name in ('S_ADDC_U32', 'S_SUBB_U32') or name.startswith('S_CSELECT_'):
        expected.append('uint32_t')
    if shape.writes_scc:
        expected.append('uint32_t *')
    if name in ('S_ADD_CO_CI_U32', 'S_SUB_CO_CI_U32'):
        expected.append('uint32_t')
    assert types == expected
    sources = [f'src{i}' for i in range(len(shape.input_bits))]
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(arch, name, ['dst'], sources)
        assert 'exec' not in body.replace('instruction_execution', '')
        assert ('wf.write_scc(scc)' in body) == shape.writes_scc
        assert ('wf.read_scc()' in body) == shape.reads_scc
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
        for i, bits in enumerate(shape.input_bits):
            suffix = '64' if bits == 64 else ''
            assert f'read_scalar{suffix}(src{i})' in body
    assert gen_scalar('cdna4', name, ['dst'], sources) is None


@pytest.mark.parametrize('name', sorted(COMPARISONS))
def test_comparisons_only_write_scc_and_read_encoded_destination(name):
    immediate = name.startswith('S_CMPK_')
    dst, src = (['sdst'], ['imm']) if immediate else ([], ['a', 'b'])
    header = public_headers()
    declaration = re.search(r'\bint rj_arith_' + name.lower() + r'\(([^;]+)\);', header)
    assert declaration is not None
    types = [t.strip() for t in re.findall(r'uint\d+_t\s*\*?', declaration[1])]
    assert types == ['uint64_t', 'uint64_t', 'uint32_t *'] + [
        f'uint{bits}_t' for bits in COMPARISONS[name]
    ]
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(arch, name, dst, src)
        assert 'registers.write_scalar' not in body
        assert 'wf.read_scc' not in body
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index('wf.write_scc(scc)')
        if immediate:
            assert 'read_scalar(sdst)' in body
            assert 'uint16_t(registers.read_scalar(imm))' in body
    assert gen_scalar('cdna4', name, dst, src) is None


@pytest.mark.parametrize('name', sorted(IMMEDIATE))
def test_immediate_arithmetic_reads_old_destination_before_commit(name):
    body = gen_scalar('rdna3', name, ['dst'], ['imm'])
    assert body.index('read_scalar(dst)') < body.index('status != RJ_ARITH_SUCCESS')
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index('write_scalar(dst, result)')
    assert 'uint16_t(registers.read_scalar(imm))' in body
    assert ('wf.write_scc(scc)' in body) == IMMEDIATE[name]
    assert 'wf.read_scc' not in body


@pytest.mark.parametrize('name,bits', sorted(BITSET.items()))
def test_bitset_reads_full_destination_and_word_index_without_scc(name, bits):
    suffix = '64' if bits == 64 else ''
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(arch, name, ['dst'], ['index'])
        assert f'uint{bits}_t result = registers.read_scalar{suffix}(dst)' in body
        assert 'uint32_t bit = registers.read_scalar(index)' in body
        assert f'rj_arith_{name.lower()}(' in body
        assert '&result, bit)' in body
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
        assert 'scc' not in body
    assert gen_scalar('cdna5', name, ['dst'], ['index']) is None


@pytest.mark.parametrize('bits', [32, 64])
def test_select_observes_only_selected_source(bits):
    body = gen_scalar('rdna4', f'S_CSELECT_B{bits}', ['dst'], ['a', 'b'])
    suffix = '64' if bits == 64 else ''
    assert f'input_scc ? registers.read_scalar{suffix}(a) : 0' in body
    assert f'!input_scc ? registers.read_scalar{suffix}(b) : 0' in body
    assert '&result, input0, input1, input_scc)' in body


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


@pytest.mark.parametrize(
    'name,bits', sorted((FLOAT_COMPARISONS | FLOAT_ROUNDING).items())
)
def test_scalar_float_bindings_preserve_results_and_optional_state(name, bits):
    comparison = name in FLOAT_COMPARISONS
    dst = [] if comparison else ['dst']
    src = ['a', 'b'] if comparison else ['a']
    body = gen_scalar('rdna4', name, dst, src)
    denorm = 'fp_denorm_mode_f32' if bits == 32 else 'fp_denorm_mode_f16_f64'
    assert f'wf.{denorm}() & 1u' in body
    assert 'RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status()' in body
    assert ', &causes)' in body
    assert 'RJ_ARITH_SEMANTICS_' not in body
    assert 'wf.exec()' not in body
    commit = (
        'wf.write_scc(result)' if comparison else 'registers.write_scalar(dst, result)'
    )
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(commit)
    assert body.index(commit) < body.index('wf.set_trapsts')
    assert ('wf.write_scc' in body) == comparison
    assert ('registers.write_scalar' in body) != comparison
    for arch in ('rdna3', 'cdna5'):
        assert gen_scalar(arch, name, dst, src) is None
    header = public_headers()
    declaration = re.search(r'\bint rj_arith_' + name.lower() + r'\(([^;]+)\);', header)
    assert declaration is not None
    types = [t.strip() for t in re.findall(r'uint\d+_t\s*\*?', declaration[1])]
    assert types == ['uint64_t', 'uint64_t', 'uint32_t *'] + ['uint32_t'] * len(src) + [
        'uint32_t *'
    ]


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


@pytest.mark.parametrize('name', sorted(FLOAT_ARITHMETIC))
def test_scalar_float_arithmetic_api_shape_and_policy(name):
    operation = FLOAT_ARITHMETIC[name]
    src = [f'source{i}' for i in range(operation.inputs)]
    body = gen_scalar('rdna4', name, ['dst'], src)
    assert 'arithmetic_flags()' in body
    assert ('registers.read_scalar(dst)' in body) == operation.accumulator
    assert ('wf.fp16_ovfl()' in body) == (name == 'S_CVT_F16_F32')
    assert ('wf.fp_round_mode_f32()' in body) == (
        name.startswith('S_FMA') and name.endswith('F32')
    )
    assert ('RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body) == bool(operation.input_fp_bits)
    assert ('RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body) == bool(operation.output_fp_bits)
    assert ', &causes)' in body
    assert 'RJ_ARITH_SEMANTICS_' not in body
    assert 'wf.exec()' not in body and 'wf.write_scc' not in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
        'registers.write_scalar(dst, result)'
    )
    for arch in ('rdna3', 'cdna5'):
        assert gen_scalar(arch, name, ['dst'], src) is None
    header = public_headers()
    declaration = re.search(r'\bint rj_arith_' + name.lower() + r'\(([^;]+)\);', header)
    assert declaration is not None
    types = [t.strip() for t in re.findall(r'uint\d+_t\s*\*?', declaration[1])]
    assert types == ['uint64_t', 'uint64_t', 'uint32_t *'] + [
        'uint32_t'
    ] * operation.inputs + ['uint32_t *']


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
        body = gen_scalar(arch, name, [], ['simm16'])
        assert f'rj_arith_{name.lower()}(' in body
        assert 'uint16_t(simm16.encoding_value()), &mode' in body
        assert body.index('wf.mode_raw()') < body.index('status != RJ_ARITH_SUCCESS')
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index('wf.set_mode_raw(mode)')
        assert 'arithmetic_reports_fp_status' not in body
        assert 'wf.exec()' not in body
    assert gen_scalar('cdna5', name, [], ['simm16']) is None
