# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/minmax.h calls for IEEE 754-2019 min/max instructions.

Scalar bodies and SIMD functors use the same helpers on raw encodings and share
the input-flush policy with floating-point comparisons. VOP3 forms use
shared/floating_operation.h to apply ABS/NEG before the operation and
OMOD/CLAMP afterward.
"""

import re

from amdisa.codegen.execute import float_compare

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
}

# Source and result format seen by the helpers.
FORMATS: dict[str, str] = {
    'f16': 'F16',
    'f32': 'F32',
    'f64': 'F64',
}

_TEMPLATE = re.compile(
    r'v_(?P<form>\w+)_(?P<dtype>f16|f32|f64)_(?P<encoding>vop2|vop3)'
)


def _format(dtype: str) -> str:
    return f'amdgpu::comparison::{FORMATS[dtype]}'


def _call(
    dtype: str,
    form: str,
    sources: list[str],
    policy: str,
) -> str:
    """Emit input flushing and selection, without instruction modifiers."""
    op, count = FORMS[form]
    assert len(sources) == count, (form, sources)
    args = [policy, *sources]
    return f'{_NS}::evaluate<{_format(dtype)}, {_NS}::{op}>({", ".join(args)})'


def operation_expr(dtype: str, form: str, policy: str) -> str:
    """Capture the input-flush policy in an operation free of instruction modifiers."""
    op, _ = FORMS[form]
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
        return _call(dtype, form, reads, float_compare.POLICY)
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    operation = operation_expr(dtype, form, float_compare.POLICY)
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
    captures = [f'{float_compare.POLICY} = {float_compare.policy_expr(dtype)}']
    call = _call(dtype, form, params, float_compare.POLICY)
    args = ', '.join(f'auto {p}' for p in params)
    return f'[{", ".join(captures)}]({args}) {{ return {call}; }}'


def simd_probe(template_name: str, true16_vop3: bool = False) -> str | None:
    """Emit the min/max SIMD fast-path call, or None to use the regular dispatch.

    None also covers VOP2 F16, whose register-half handling is provided by
    SIMD_VOP2_BINARY. All min/max functors receive raw unsigned encodings.
    """
    match = _TEMPLATE.fullmatch(template_name)
    if match is None or match['form'] not in FORMS:
        return None
    form, dtype, encoding = match['form'], match['dtype'], match['encoding']
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
    elif dtype == 'f16' and source_count == 2:
        macro = 'VOP3_BINARY_RAW_FP16'
    else:
        # Without true16 half selection, ternary F16 uses the same unsigned
        # 32-bit loads/stores as F32; the functor interprets the low 16 bits.
        macro = f'VOP3_{shape}_RAW_FP'
    operation = operation_expr(dtype, form, float_compare.policy_expr(dtype))
    return f'  ROCJITSU_TRY_SIMD_{macro}({_format(dtype)}, {operation});'
