# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Floating VOPC relations evaluated by shared/comparison.h.

Scalar, SIMD and arch-local compare bodies all emit the same
``comparison::evaluate`` call on raw source encodings. Source modifiers and
MODE input flushing use shared/source_modifier.h and shared/input_denormal.h;
comparison.h owns the relation and NaN ordering.
"""

from amdisa.semantics import FLOAT_COMPARE_RELATIONS, is_float_relation

__all__ = ['is_float_relation']

_NS = 'amdgpu::comparison'

# VOPC relation mnemonic -> comparison.h relation. F and T/TRU read no source
# and keep their constant bodies.
RELATIONS: dict[str, str] = {
    'lt': 'Lt',
    'eq': 'Eq',
    'le': 'Le',
    'gt': 'Gt',
    'lg': 'Lg',
    'ge': 'Ge',
    'o': 'O',
    'u': 'U',
    'nge': 'Nge',
    'nlg': 'Nlg',
    'ngt': 'Ngt',
    'nle': 'Nle',
    'neq': 'Neq',
    'nlt': 'Nlt',
}

assert RELATIONS.keys() == FLOAT_COMPARE_RELATIONS

FORMATS: dict[str, str] = {'f16': 'F16', 'f32': 'F32', 'f64': 'F64'}

# Name of the per-instruction policy the scalar bodies declare before the lane loop.
POLICY = 'compare_policy'


def lane_type(dtype: str) -> str:
    """Raw-encoding lane type: F16 occupies the low half of a 32-bit lane."""
    return 'uint64_t' if dtype == 'f64' else 'uint32_t'


def policy_expr(dtype: str) -> str:
    mode = 'f32' if dtype == 'f32' else 'f16_f64'
    return f'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_{mode}())'


def policy_decl(dtype: str, indent: str = '  ') -> str:
    return f'{indent}const auto {POLICY} = {policy_expr(dtype)};'


def evaluate_expr(
    dtype: str,
    op: str,
    a: str,
    b: str,
    policy: str = POLICY,
    modifiers: tuple[str, str] | None = None,
) -> str:
    """Return the relation on raw encodings ``a`` and ``b``.

    ``modifiers`` is the (ABS, NEG) field pair of a VOP3 form; VOPC forms
    receive sources that already carry any DPP modifiers.
    """
    args = [a, b, *(modifiers or ()), policy]
    return (
        f'{_NS}::evaluate<amdgpu::fp_format::{FORMATS[dtype]}, {_NS}::{RELATIONS[op]}>'
        f'({", ".join(args)})'
    )


def simd_functor(dtype: str, op: str, modifiers: tuple[str, str] | None = None) -> str:
    """Return a SIMD compare functor; the policy and modifiers are captured once."""
    captures = [f'{POLICY} = {policy_expr(dtype)}']
    if modifiers is not None:
        captures += [f'abs_mods = {modifiers[0]}', f'neg_mods = {modifiers[1]}']
    call = evaluate_expr(
        dtype, op, 'a', 'b', modifiers=('abs_mods', 'neg_mods') if modifiers else None
    )
    return f'[{", ".join(captures)}](auto a, auto b) {{ return {call}; }}'
