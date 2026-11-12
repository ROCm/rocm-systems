# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/minmax.h calls for legacy and IEEE 754-2019 min/max instructions.

Scalar bodies and SIMD functors use the same helpers on raw encodings and share
the input-flush policy (input_policy.py) with floating-point comparisons. VOP3 forms use
shared/floating_operation.h to apply ABS/NEG before the operation and
OMOD/CLAMP afterward.
"""

import re

from amdisa.codegen.execute import input_policy

_NS = 'amdgpu::minmax'

# Semantic form -> (C++ operation type, source count).
FORMS: dict[str, tuple[str, int]] = {
    'minimum': ('Minimum', 2),
    'maximum': ('Maximum', 2),
    'min_num': ('MinNum', 2),
    'max_num': ('MaxNum', 2),
    'minimum3': ('Minimum3', 3),
    'maximum3': ('Maximum3', 3),
    'minimummaximum': ('MinimumMaximum', 3),
    'maximumminimum': ('MaximumMinimum', 3),
    'min3_num': ('Min3Num', 3),
    'max3_num': ('Max3Num', 3),
    'minmax_num': ('MinMaxNum', 3),
    'maxmin_num': ('MaxMinNum', 3),
    'med3_num': ('Med3Num', 3),
    'legacy_min': ('LegacyMin', 2),
    'legacy_max': ('LegacyMax', 2),
    'legacy_min3': ('LegacyMin3', 3),
    'legacy_max3': ('LegacyMax3', 3),
    'legacy_med3': ('LegacyMed3', 3),
    'legacy_minmax': ('LegacyMinMax', 3),
    'legacy_maxmin': ('LegacyMaxMin', 3),
}

# Source and result format seen by the helpers.
FORMATS = input_policy.FORMATS
LEGACY_FORMS = frozenset({'min', 'max', 'min3', 'max3', 'med3', 'minmax', 'maxmin'})
LEGACY_POLICY = 'amdgpu::fp_mode::legacy_minmax_policy(wf.cu().arch(), wf.ieee_mode())'

_TEMPLATE = re.compile(
    r'v_(?P<form>\w+)_(?P<dtype>f16|f32|f64)_(?P<encoding>vop2|vop3)'
)


def _format(dtype: str) -> str:
    return f'amdgpu::fp_format::{FORMATS[dtype]}'


def _call(
    dtype: str,
    form: str,
    sources: list[str],
    policy: str,
    legacy_policy: str = LEGACY_POLICY,
) -> str:
    """Emit input flushing and selection, without instruction modifiers."""
    op, count = FORMS[form]
    assert len(sources) == count, (form, sources)
    if form.startswith('legacy_'):
        operation = operation_expr(dtype, form, policy, legacy_policy)
        return f'{operation}({", ".join(sources)})'
    args = [policy, *sources]
    return f'{_NS}::evaluate<{_format(dtype)}, {_NS}::{op}>({", ".join(args)})'


def operation_expr(
    dtype: str, form: str, policy: str, legacy_policy: str = LEGACY_POLICY
) -> str:
    """Capture the input-flush policy in an operation free of instruction modifiers."""
    op, _ = FORMS[form]
    if form.startswith('legacy_'):
        return f'{_NS}::LegacyOperation<{_format(dtype)}, {_NS}::{op}>{{{policy}, {legacy_policy}}}'
    return f'{_NS}::Operation<{_format(dtype)}, {_NS}::{op}>{{{policy}}}'


def minmax_expr(
    dtype: str,
    form: str,
    reads: list[str],
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Emit raw-bit selection, using the common wrapper when modifiers are present."""
    if modifiers is None and output_policy is None:
        return _call(dtype, form, reads, input_policy.NAME)
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    operation = operation_expr(dtype, form, input_policy.NAME)
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation,
        *reads,
    ]
    return f'amdgpu::floating_operation::apply<{_format(dtype)}>({", ".join(args)})'


def simd_functor(dtype: str, form: str) -> str:
    """Emit VOP2 input flushing and selection on raw lanes."""
    params = ['a', 'b', 'c'][: FORMS[form][1]]
    captures = [f'{input_policy.NAME} = {input_policy.policy_expr(dtype)}']
    if form.startswith('legacy_'):
        captures.append(f'legacy_policy = {LEGACY_POLICY}')
    call = _call(dtype, form, params, input_policy.NAME, 'legacy_policy')
    args = ', '.join(f'auto {p}' for p in params)
    return f'[{", ".join(captures)}]({args}) {{ return {call}; }}'


def simd_probe(template_name: str, true16_vop3: bool = False) -> str | None:
    """Emit the min/max SIMD fast-path call, or None to use the regular dispatch.

    None also covers VOP2 F16, whose register-half handling is provided by
    SIMD_VOP2_BINARY. All min/max functors receive raw unsigned encodings.
    """
    match = _TEMPLATE.fullmatch(template_name)
    if match is None:
        return None
    form, dtype, encoding = match['form'], match['dtype'], match['encoding']
    if form in LEGACY_FORMS:
        form = f'legacy_{form}'
    if form not in FORMS:
        return None
    source_count = FORMS[form][1]
    if encoding == 'vop2':
        if dtype == 'f16':
            return None
        functor = simd_functor(dtype, form)
        if dtype == 'f64':
            return f'  ROCJITSU_TRY_SIMD_VOP2_BINARY_RAW_FP64({functor});'
        return f'  ROCJITSU_TRY_SIMD_VOP2_BINARY(uint32_t, {functor});'
    shape = 'BINARY' if source_count == 2 else 'TERNARY'
    if dtype == 'f64':
        if source_count != 2:
            return None
        macro = 'VOP3_BINARY_RAW_FP64'
    elif dtype == 'f16' and true16_vop3:
        macro = f'VOP3_{shape}_TRUE16_RAW_FP16'
    else:
        # Without true16 half selection, F16 uses the same unsigned
        # 32-bit loads/stores as F32; the functor interprets the low 16 bits.
        macro = f'VOP3_{shape}_RAW_FP'
    operation = operation_expr(dtype, form, input_policy.policy_expr(dtype))
    return f'  ROCJITSU_TRY_SIMD_{macro}({_format(dtype)}, {operation});'
