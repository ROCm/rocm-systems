from amdisa.codegen.execute.rj_arith_scalar import GFX1250_CONVERSION_NAMES
from amdisa.codegen.execute.rj_arith_scalar import GFX1250_FMA_NAMES
from amdisa.codegen.execute.rj_arith_scalar import GFX1250_BINARY_NAMES

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


import pytest


from amdisa.codegen.execute.rj_arith_scalar import (
    GFX1250_BINARY_NAMES,
    GFX1250_FMA_NAMES,
    GFX1250_CONVERSION_NAMES,
    BITSET,
    COMPARISONS,
    FLOAT_COMPARISONS,
    FLOAT_ARITHMETIC,
    FLOAT_ROUNDING,
    IMMEDIATE,
    ROUNDED_RESULTS,
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
    for arch in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'):
        body = gen_scalar(context(arch, name, ['dst'], ['index']))
        assert f'uint{bits}_t result = registers.read_scalar{suffix}(dst)' in body
        assert 'uint32_t bit = registers.read_scalar(index)' in body
        assert f'rj_arith_{name.lower()}(' in body
        assert '&result, bit)' in body
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
        assert 'scc' not in body
    assert gen_scalar(context('rdna3_5', name, ['dst'], ['index'])) is None


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


@pytest.mark.parametrize('name', sorted(FLOAT_ARITHMETIC))
def test_scalar_float_rounding_reads_destination_mode_field(name):
    # gfx1201: FP32 results round by MODE[1:0]; FP16 results, including
    # S_CVT_F16_F32, by MODE[3:2]. Other forms do not round to a float.
    operation = FLOAT_ARITHMETIC[name]
    src = [f'src{i}' for i in range(operation.inputs)]
    body = gen_scalar(context('rdna4', name, ['dst'], src))
    f32 = 'flags |= uint64_t(wf.fp_round_mode_f32()) << 24;' in body
    f16 = 'flags |= uint64_t(wf.fp_round_mode_f16_f64()) << 24;' in body
    assert (f32, f16) == (
        name in ROUNDED_RESULTS and operation.output_fp_bits == 32,
        name in ROUNDED_RESULTS and operation.output_fp_bits == 16,
    )
    if name in ROUNDED_RESULTS:
        assert body.index('<< 24;') < body.index('status != RJ_ARITH_SUCCESS')


@pytest.mark.parametrize('name', ['S_ROUND_MODE', 'S_DENORM_MODE'])
def test_guest_mode_updates_are_functional_state_independent_of_exec(name):
    for arch in ('rdna3', 'rdna4', 'cdna5'):
        body = gen_scalar(context(arch, name, [], ['simm16']))
        assert f'rj_arith_{name.lower()}(' in body
        assert 'uint16_t(simm16.encoding_value()), &mode' in body
        assert body.index('wf.mode_raw()') < body.index('status != RJ_ARITH_SUCCESS')
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'wf.set_mode_raw(mode)'
        )
        assert 'arithmetic_reports_fp_status' not in body
        assert 'wf.exec()' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(SCALAR))
def test_cdna_integer_shapes_match_existing_bindings(arch, name):
    shape = SCALAR[name]
    source = [f'input{i}' for i in range(len(shape.input_bits))]
    assert gen_scalar(context(arch, name, ['dst'], source)) == gen_scalar(
        context('rdna3', name, ['dst'], source)
    )


@pytest.mark.parametrize('name', sorted(FLOAT_COMPARISONS))
def test_gfx1250_comparison_uses_architecture_policy(name):
    body = gen_scalar(context('cdna5', name, [], ['src0', 'src1']))
    assert f'rj_arith_{name.lower()}_gfx1250(' in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index('wf.write_scc(result)')
    assert 'wf.exec()' not in body
    assert 'raise_alu_causes' not in body and 'set_trapsts' not in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_scalar(context(arch, name, [], ['src0', 'src1'])) is None


@pytest.mark.parametrize('name', sorted(FLOAT_ROUNDING))
def test_gfx1250_rounding_uses_architecture_policy(name):
    body = gen_scalar(context('cdna5', name, ['dst'], ['src']))
    assert f'rj_arith_{name.lower()}_gfx1250(' in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
        'registers.write_scalar(dst, result)'
    )
    assert 'wf.exec()' not in body and 'set_trapsts' not in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_scalar(context(arch, name, ['dst'], ['src'])) is None


@pytest.mark.parametrize('name', sorted(GFX1250_BINARY_NAMES))
def test_gfx1250_binary_uses_architecture_policy(name):
    body = gen_scalar(context('cdna5', name, ['dst'], ['a', 'b']))
    assert f'rj_arith_{name.lower()}_gfx1250(' in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
        'registers.write_scalar(dst, result)'
    )
    assert 'wf.exec()' not in body and 'set_trapsts' not in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_scalar(context(arch, name, ['dst'], ['a', 'b'])) is None


@pytest.mark.parametrize('name', sorted(GFX1250_FMA_NAMES))
def test_gfx1250_fma_uses_architecture_policy(name):
    inputs = ['a', 'b'] if name.startswith('S_FMAC_') else ['a', 'b', 'c']
    body = gen_scalar(context('cdna5', name, ['dst'], inputs))
    assert f'rj_arith_{name.lower()}_gfx1250(' in body
    assert ('fp_round_mode_f32()' in body) == name.endswith('F32')
    assert ('result = registers.read_scalar(dst)' in body) == name.startswith('S_FMAC_')
    assert 'set_trapsts' not in body and 'wf.exec()' not in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_scalar(context(arch, name, ['dst'], inputs)) is None


@pytest.mark.parametrize('name', sorted(GFX1250_CONVERSION_NAMES))
def test_gfx1250_conversion_uses_architecture_policy(name):
    inputs = ['a', 'b'] if name.startswith('S_CVT_PK_') else ['a']
    body = gen_scalar(context('cdna5', name, ['dst'], inputs))
    assert f'rj_arith_{name.lower()}_gfx1250(' in body
    assert 'set_trapsts' not in body and 'wf.exec()' not in body
    if name == 'S_CVT_HI_F32_F16':
        assert 'read_scalar_f16_carrier(a, wf, true)' in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_scalar(context(arch, name, ['dst'], inputs)) is None


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_register_moves_use_arith_without_migrating_control_flow(arch):
    from amdisa.codegen.execute.rj_arith_scalar import MOVES, RELATIVE_MOVES

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    text = '\n'.join(
        (generated / f'{encoding}_exec.cpp').read_text()
        for encoding in ('sop1', 'sopk')
    )
    for name in MOVES:
        assert f'rj_arith_{name.lower()}(' in text, name
    for name in RELATIVE_MOVES:
        body = gen_scalar(context(arch, name, ['dst'], ['src']))
        assert 'relative_scalar_selector' in body
        assert 'rj_arith_s_mov_b' in body
    for name in ('S_BRANCH', 'S_WAITCNT', 'S_BARRIER', 'S_SETPC_B64'):
        assert gen_scalar(context(arch, name, ['dst'], ['src'])) is None


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_exec_updates_stage_both_inputs_before_outputs(arch):
    from amdisa.codegen.execute.rj_arith_scalar import EXEC_UPDATES

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    text = (generated / 'sop1_exec.cpp').read_text()
    names = {name for name in EXEC_UPDATES if name.endswith('B64')}
    if arch == 'cdna5':
        names = {
            name
            for name in EXEC_UPDATES
            if '_NOT' in name or 'N1_' not in name and 'N2_' not in name
        }
    else:
        names = {name for name in names if '_NOT' not in name}
    for name in names:
        assert f'rj_arith_{name.lower()}(' in text, name
        body = gen_scalar(context(arch, name, ['dst'], ['src']))
        assert body.index('registers.read_scalar') < body.index('const int status')
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
        assert body.index('registers.write_scalar') < body.index('wf.write_scc')


@pytest.mark.parametrize(
    'arch,name',
    [('cdna3', 'S_GETPC_B64'), ('cdna4', 'S_GETPC_B64'), ('cdna5', 'S_GET_PC_I64')],
)
def test_cdna_getpc_uses_architecture_width_without_changing_pc(arch, name):
    body = gen_scalar(context(arch, name, ['dst'], []))
    assert f'rj_arith_{name.lower()}(' in body
    assert '&result, wf.pc)' in body
    assert 'wf.pc =' not in body and 'wf.exec' not in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index('write_scalar64')
    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    assert f'rj_arith_{name.lower()}(' in (generated / 'sop1_exec.cpp').read_text()


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
@pytest.mark.parametrize('bits', [32, 64])
def test_cdna_first_zero_is_independent_of_other_scalar_migration(arch, bits):
    body = gen_scalar(context(arch, f'S_FF0_I32_B{bits}', ['dst'], ['src']))
    assert f'rj_arith_s_ff0_i32_b{bits}(' in body
    assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
        'write_scalar(dst, result)'
    )
    assert 'scc' not in body
    assert gen_scalar(context(arch, 'S_FF1_I32_B32', ['dst'], ['src'])) is not None
