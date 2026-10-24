# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from rj_arith_headers import public_headers

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
            arch, name, ['dst'], ['a', 'b'], True, 'commit_result'
        )
        assert f'execute_compare<{bits}>' in body
        assert f'a, b, {str(constant).lower()}, commit_result' in body
        assert 'wf.set_exec' not in body and 'wf.set_vcc' not in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == (bits == 16)
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == (bits == 16)
        implicit = gen_integer_compare(arch, name, [], ['a', 'b'], False)
        assert ('wf.set_exec(result)' in implicit) == name.startswith('V_CMPX_')
        assert ('wf.set_vcc_mask(result)' in implicit) == name.startswith('V_CMP_')
        assert 'RJ_ARITH_ALU_HIGH' not in implicit
    assert gen_integer_compare('cdna4', name, [], ['a', 'b'], False) is None


@pytest.mark.parametrize('name', sorted(FLOAT_COMPARISONS))
def test_float_comparisons_preserve_policy_modifiers_and_state_signature(name):
    bits, constant = FLOAT_COMPARISONS[name]
    header = public_headers()
    for arch in ('rdna3', 'rdna4'):
        body = gen_float_compare(arch, name, ['dst'], ['a', 'b'], True, 'commit_result')
        if arch == 'rdna4' and constant:
            assert body is None
            continue
        api = 'rj_arith_' + name.lower()
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
        assert 'arithmetic_reports_fp_status(), commit_result' in body
        for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B', 'CLAMP'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == (bits == 16)
        assert f'a, b, {str(constant).lower()}' in body
        implicit = gen_float_compare(arch, name, [], ['a', 'b'], False)
        assert 'inst_.abs' not in implicit
        assert ('wf.set_exec(result)' in implicit) == name.startswith('V_CMPX_')
    assert gen_float_compare('cdna4', name, [], ['a', 'b'], False) is None


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
        body = gen_class_compare(arch, name, ['dst'], ['a', 'b'], True, 'commit_result')
        assert f'execute_compare<{bits}, 32>' in body
        assert f'rj_arith_{name.lower()}_w64' in body
        assert 'RJ_ARITH_ALU_ABS_A' in body and 'RJ_ARITH_ALU_NEG_A' in body
        assert 'RJ_ARITH_ALU_ABS_B' not in body and 'RJ_ARITH_ALU_NEG_B' not in body
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == (bits == 16)
        assert 'arithmetic_reports_fp_status' not in body
        assert 'false, commit_result' in body
    assert gen_class_compare('cdna4', name, [], ['a', 'b'], False) is None
