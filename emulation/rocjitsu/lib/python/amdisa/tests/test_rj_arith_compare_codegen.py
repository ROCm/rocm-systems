# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


from rj_arith_headers import public_headers

from pathlib import Path
import re

import pytest

from amdisa.codegen.execute.rj_arith_compare import (
    INTEGER_COMPARISONS,
    FLOAT_COMPARISONS,
    CLASS_COMPARISONS,
    gen_integer_compare,
    gen_float_compare,
    gen_class_compare,
)


def context(arch, name, dst, src, *, is_vop3=False, result_writer=None):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name='',
        is_vop3=is_vop3,
        has_abs=is_vop3,
        result_writer=result_writer,
    )


@pytest.mark.parametrize('name', sorted(INTEGER_COMPARISONS))
def test_integer_compare_signatures_and_result_commit(name):
    bits, constant = INTEGER_COMPARISONS[name]
    header = public_headers()
    for suffix, width in (('_w32', 32), ('_w64', 64)):
        api = 'rj_arith_' + name.lower() + suffix
        declaration = re.search(r'\bint ' + api + r'\(([^;]+)\);', header)
        assert declaration is not None
        parameters = declaration[1].split(',')
        assert len(parameters) == 6
        assert parameters[1].strip().startswith(f'uint{width}_t ')
        assert parameters[3].strip().startswith(f'uint{width}_t *')
    for arch in ('rdna3', 'rdna4'):
        body = gen_integer_compare(
            context(
                arch,
                name,
                ['dst'],
                ['a', 'b'],
                is_vop3=True,
                result_writer='commit_result',
            )
        )
        assert f'execute_compare<{bits}>' in body
        assert f'a, b, {str(constant).lower()}, commit_result' in body
        assert 'wf.set_exec' not in body and 'wf.set_vcc' not in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == (bits == 16)
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == (bits == 16)
        implicit = gen_integer_compare(
            context(arch, name, [], ['a', 'b'], is_vop3=False)
        )
        assert ('wf.set_exec(result)' in implicit) == name.startswith('V_CMPX_')
        assert ('wf.set_vcc_mask(result)' in implicit) == name.startswith('V_CMP_')
        assert 'RJ_ARITH_ALU_HIGH' not in implicit
    assert (
        gen_integer_compare(context('cdna2', name, [], ['a', 'b'], is_vop3=False))
        is None
    )


@pytest.mark.parametrize(
    'arch,kind,count',
    [
        ('rdna3', 'integer', 144),
        ('rdna4', 'integer', 120),
        ('rdna3', 'float', 160),
        ('rdna4', 'float', 140),
        ('rdna3', 'class', 10),
        ('rdna4', 'class', 10),
    ],
)
def test_generated_comparisons_cannot_bypass_rj_arith(arch, kind, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for encoding in ('vopc', 'vop3'):
        source = (root / f'{encoding}_exec.cpp').read_text()
        checked = 0
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_compare<' in body:
                family = 'float'
            elif 'arith::execute_compare<' in body:
                family = (
                    'class'
                    if re.search(r'rj_arith_v_cmpx?_class_f', body)
                    else 'integer'
                )
            else:
                continue
            if family != kind:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'ROCJITSU_TRY_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            if '::execute_modifier_impl(' in match[0]:
                if 'dpp_new_result' in body:
                    assert 'if (wf.instruction_execution_failed())' in body
                if kind == 'float':
                    assert (
                        'exec_mask &= dpp_row_bank_mask_ & dpp_source_write_mask_;'
                        in body
                    )
            checked += 1
        assert checked == count


@pytest.mark.parametrize('name', sorted(FLOAT_COMPARISONS))
def test_float_comparisons_preserve_policy_modifiers_and_state_signature(name):
    bits, constant = FLOAT_COMPARISONS[name]
    header = public_headers()
    for arch in ('rdna3', 'rdna4'):
        body = gen_float_compare(
            context(
                arch,
                name,
                ['dst'],
                ['a', 'b'],
                is_vop3=True,
                result_writer='commit_result',
            )
        )
        if (arch == 'rdna4' and constant) or '_TRU_' in name:
            assert body is None
            continue
        api = 'rj_arith_' + name.lower().replace('_tru_', '_t_')
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        for suffix, width in (('_w32', 32), ('_w64', 64)):
            declaration = re.search(
                r'\bint ' + api + suffix + arch_suffix + r'\(([^;]+)\);', header
            )
            assert declaration is not None
            parameters = declaration[1].split(',')
            assert len(parameters) == 7
            assert parameters[1].strip().startswith(f'uint{width}_t ')
            assert parameters[3].strip().startswith(f'uint{width}_t *')
            assert parameters[-1].strip() == (
                'uint32_t *excp_flag_user' if arch == 'rdna4' else 'uint32_t *trapsts'
            )
        assert f'execute_float_compare<{bits}>' in body
        assert 'arithmetic_flags()' in body
        assert 'commit_result' in body
        for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B', 'CLAMP'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == (bits == 16)
        assert f'a, b, {str(constant).lower()}' in body
        implicit = gen_float_compare(context(arch, name, [], ['a', 'b'], is_vop3=False))
        assert 'inst_.abs' not in implicit
        assert ('wf.set_exec(result)' in implicit) == name.startswith('V_CMPX_')
    assert (
        gen_float_compare(context('cdna2', name, [], ['a', 'b'], is_vop3=False)) is None
    )


@pytest.mark.parametrize('name', sorted(CLASS_COMPARISONS))
def test_class_comparisons_use_one_word_class_masks_without_optional_state(name):
    bits = CLASS_COMPARISONS[name]
    header = public_headers()
    for suffix in ('_w32', '_w64'):
        declaration = re.search(
            r'\bint rj_arith_' + name.lower() + suffix + r'\(([^;]+)\);', header
        )
        assert declaration is not None
        assert len(declaration[1].split(',')) == 6
    for arch in ('rdna3', 'rdna4'):
        body = gen_class_compare(
            context(
                arch,
                name,
                ['dst'],
                ['a', 'b'],
                is_vop3=True,
                result_writer='commit_result',
            )
        )
        suffix = '_gfx12' if arch == 'rdna4' else ''
        for wave in (32, 64):
            assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body

        assert f'execute_compare<{bits}, {min(bits, 32)}>' in body
        assert f'rj_arith_{name.lower()}_w64' in body
        assert 'RJ_ARITH_ALU_ABS_A' in body and 'RJ_ARITH_ALU_NEG_A' in body
        assert 'RJ_ARITH_ALU_ABS_B' not in body and 'RJ_ARITH_ALU_NEG_B' not in body
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == (bits == 16)
        assert 'arithmetic_reports_fp_status' not in body
        assert 'false, commit_result' in body
    assert (
        gen_class_compare(context('cdna2', name, [], ['a', 'b'], is_vop3=False)) is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_every_vopc_handler_routes_through_rj_arith(arch):
    source = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
        / 'vopc_exec.cpp'
    ).read_text()
    assert 'amdgpu::execute_v_cmp' not in source
    assert 'ROCJITSU_TRY_SIMD' not in source


@pytest.mark.parametrize('arch', ('cdna3', 'cdna4', 'cdna5'))
@pytest.mark.parametrize('cmpx', (False, True))
@pytest.mark.parametrize('explicit_destination', (False, True))
def test_cdna_integer_compare_architectural_destinations(
    arch, cmpx, explicit_destination
):
    name = 'V_CMPX_EQ_U32' if cmpx else 'V_CMP_EQ_U32'
    dst = ['dst'] if explicit_destination else []
    body = gen_integer_compare(context(arch, name, dst, ['a', 'b']))
    assert 'execute_compare<32>' in body
    assert ('wf.set_exec(result)' in body) == cmpx
    writes_scalar = not cmpx or arch in ('cdna3', 'cdna4')
    assert ('write_wave_mask_scalar(dst' in body) == (
        writes_scalar and explicit_destination
    )
    assert ('wf.set_vcc_mask(result)' in body) == (
        writes_scalar and not explicit_destination
    )
    custom = gen_integer_compare(
        context(arch, name, dst, ['a', 'b'], result_writer='commit_result')
    )
    assert 'commit_result' in custom
    assert 'wf.set_exec' not in custom


@pytest.mark.parametrize('arch', ('cdna3', 'cdna4', 'cdna5'))
@pytest.mark.parametrize(
    'generator,name',
    ((gen_float_compare, 'V_CMPX_EQ_F32'), (gen_class_compare, 'V_CMPX_CLASS_F32')),
)
def test_cdna_float_compare_reuses_numeric_model_without_gfx12_status(
    arch, generator, name
):
    body = generator(
        context(
            arch, name, ['dst'], ['a', 'b'], is_vop3=True, result_writer='commit_result'
        )
    )
    assert 'rj_arith_' + name.lower() + '_w64,' in body
    assert '_gfx12' not in body
    assert ('write_wave_mask_scalar(dst' in body) == (arch in ('cdna3', 'cdna4'))
    assert 'commit_result' in body
    assert 'RJ_ARITH_ALU_ABS_A' in body


@pytest.mark.parametrize('arch,count', (('cdna3', 396), ('cdna4', 396), ('cdna5', 324)))
def test_all_cdna_comparison_encodings_use_arithmetic_library(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for path in root.glob('*exec*.cpp'):
        source = path.read_text()
        for match in re.finditer(r'void (VCmp\w+)::execute_impl\([^)]*\)\s*\{', source):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            assert 'amdgpu::arith::execute_' in source[match.end() : end], match[1]
            checked += 1
    assert checked == count


@pytest.mark.parametrize('prefix', ['V_CMP', 'V_CMPX'])
def test_tru_alias_is_cdna_only(prefix):
    for arch in ['rdna2', 'cdna3', 'cdna4', 'cdna5']:
        body = gen_float_compare(
            context(arch, prefix + '_TRU_F32', ['d'], ['a', 'b'], is_vop3=True)
        )
        assert (body is None) == (arch == 'rdna2')
