# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/transcendental_f64.h calls for V_RCP_F64, V_RSQ_F64 and V_SQRT_F64.

Scalar bodies and SIMD functors evaluate raw encodings with the same helper,
which applies MODE input and output flushing. VOP3 forms wrap the operation in
shared/floating_operation.h: ABS/NEG before it, then OMOD/CLAMP with the
TRANS-unit output policy.
"""

import re

_NS = 'amdgpu::transcendental_f64'

# Semantic call name -> transcendental_f64::Kind.
CALLS: dict[str, str] = {'rcp_f64': 'RCP', 'rsq_f64': 'RSQ', 'sqrt_f64': 'SQRT'}

POLICY = 'transcendental_policy'
POLICY_EXPR = 'amdgpu::transcendental_f64_policy(wf)'

_TEMPLATE = re.compile(r'v_(?P<op>rcp|rsq|sqrt)_f64_(?P<encoding>vop1|vop3)')


def policy_decl(indent: str = '  ') -> str:
    """Declare the MODE policy once, before the lane loop."""
    return f'{indent}const auto {POLICY} = {POLICY_EXPR};'


def operation_expr(kind: str, policy: str = POLICY) -> str:
    """Capture the MODE policy in an operation free of instruction modifiers."""
    return f'{_NS}::Operation<{_NS}::Kind::{kind}>{{{policy}}}'


def evaluate_expr(
    kind: str,
    source: str,
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Return the result encoding, using the common wrapper for VOP3 modifiers."""
    if modifiers is None and output_policy is None:
        return f'{_NS}::evaluate<{_NS}::Kind::{kind}>({source}, {POLICY})'
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation_expr(kind),
        source,
    ]
    return (
        f'amdgpu::floating_operation::apply<amdgpu::fp_format::F64>({", ".join(args)})'
    )


def simd_probe(template_name: str) -> str | None:
    """Emit the SIMD fast-path call on raw 64-bit lanes, or None for other kernels."""
    match = _TEMPLATE.fullmatch(template_name)
    if match is None:
        return None
    operation = operation_expr(match['op'].upper(), POLICY_EXPR)
    if match['encoding'] == 'vop1':
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(uint64_t, {operation});'
    return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_TRANSCENDENTAL_FP64({operation});'
