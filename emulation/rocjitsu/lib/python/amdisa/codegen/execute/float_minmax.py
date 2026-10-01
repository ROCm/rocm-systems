# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/minmax.h calls for IEEE 754-2019 min/max instructions.

Scalar bodies and SIMD functors use the same helper and share the input-flush
policy with floating-point comparisons.
"""

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

# Source format seen by the helper. Some F16 SIMD paths widen their inputs to F32.
FORMATS: dict[str, str] = {
    'f16': 'F16',
    'f32': 'F32',
    'f64': 'F64',
    'widened_f16': 'WidenedF16',
}


def _call(
    fmt: str,
    form: str,
    sources: list[str],
    policy: str,
    modifiers: tuple[str, str] | None = None,
) -> str:
    op, count = FORMS[form]
    assert len(sources) == count, (form, sources)
    template = f'amdgpu::comparison::{FORMATS[fmt]}, {_NS}::{op}'
    args = [policy, *sources]
    if modifiers:
        args.insert(0, f'{_NS}::Modifiers{{{", ".join(modifiers)}}}')
    return f'{_NS}::evaluate<{template}>({", ".join(args)})'


def minmax_expr(
    dtype: str, form: str, reads: list[str], modifiers: tuple[str, str] | None = None
) -> str:
    """Emit a scalar selection returning raw bits; the caller declares the policy."""
    return _call(dtype, form, reads, float_compare.POLICY, modifiers)


def simd_functor(dtype: str, form: str, fmt: str | None = None) -> str:
    """Emit a SIMD lambda that captures the instruction's input-flush policy.

    ``dtype`` chooses the MODE field. ``fmt`` overrides the input representation,
    e.g. ``widened_f16`` when the VOP3 helper passes F16 values as F32.
    """
    params = ['a', 'b', 'c'][: FORMS[form][1]]
    policy = float_compare.POLICY
    call = _call(fmt or dtype, form, params, policy)
    capture = f'{policy} = {float_compare.policy_expr(dtype)}'
    args = ', '.join(f'auto {p}' for p in params)
    return f'[{capture}]({args}) {{ return {call}; }}'
