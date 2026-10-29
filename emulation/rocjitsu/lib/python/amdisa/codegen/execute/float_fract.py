# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/fract.h calls for V_FRACT_F16/F32/F64 on raw encodings.

Scalar bodies and VOP1 SIMD functors use the same helper with the same MODE
policy, resolved once before the lane loop. VOP3 forms use
shared/floating_operation.h to apply ABS/NEG before the operation and
OMOD/CLAMP afterward, and run scalar only: VOP3 FRACT has no SIMD path.
"""

from amdisa.codegen.execute import input_policy

# Name of the policy the scalar body declares before the lane loop.
POLICY = 'fract_policy'


def policy_expr(dtype: str) -> str:
    """Return the MODE rounding and denormal policy of ``dtype``'s fields."""
    mode = input_policy.mode_field(dtype)
    return (
        f'amdgpu::fract::Policy::make(wf.fp_denorm_mode_{mode}(), '
        f'wf.fp_round_mode_{mode}())'
    )


def policy_decl(dtype: str, indent: str = '  ') -> str:
    return f'{indent}const auto {POLICY} = {policy_expr(dtype)};'


def operation_expr(dtype: str, policy: str = POLICY) -> str:
    """Capture the policy in an operation free of instruction modifiers."""
    return f'amdgpu::fract::Operation<{input_policy.fp_format(dtype)}>{{{policy}}}'


def fract_expr(
    dtype: str,
    source: str,
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Emit the result bits from a raw source register read.

    ``modifiers`` holds the (ABS, NEG) fields; ``output_policy`` names a
    declared output_modifier::Policy.
    """
    fmt = input_policy.fp_format(dtype)
    if modifiers is None and output_policy is None:
        return f'amdgpu::fract::evaluate<{fmt}>({source}, {POLICY})'
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation_expr(dtype),
        source,
    ]
    return f'amdgpu::floating_operation::apply<{fmt}>({", ".join(args)})'


def simd_functor(dtype: str) -> str:
    """Return the VOP1 SIMD functor on raw lanes; VOP3 FRACT has no SIMD path."""
    return operation_expr(dtype, policy_expr(dtype))
