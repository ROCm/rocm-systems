# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/ldexp.h calls for V_LDEXP_F16 on raw encodings.

Scalar bodies and SIMD functors use the same helper with the same MODE policy,
resolved once before the lane loop. VOP3 forms use shared/floating_operation.h
to apply ABS/NEG to the floating source and OMOD/CLAMP to the result; the
exponent source takes no modifiers.
"""

_FORMAT = 'amdgpu::fp_format::F16'

# Name of the policy the scalar body declares before the lane loop.
POLICY = 'ldexp_policy'


def policy_expr() -> str:
    """Return the F16 MODE rounding, denormal and FP16_OVFL policy."""
    return (
        'amdgpu::ldexp::Policy::make(wf.fp_denorm_mode_f16_f64(), '
        'wf.fp_round_mode_f16_f64(), wf.fp16_ovfl())'
    )


def policy_decl(indent: str = '  ') -> str:
    return f'{indent}const auto {POLICY} = {policy_expr()};'


def operation_expr(policy: str = POLICY) -> str:
    """Capture the policy in an operation free of instruction modifiers."""
    return f'amdgpu::ldexp::Operation<{_FORMAT}>{{{policy}}}'


def ldexp_expr(
    value: str,
    exponent: str,
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Emit the scaled half bits from raw value and exponent register reads.

    ``modifiers`` holds the value's (ABS, NEG) fields; ``output_policy`` names a
    declared output_modifier::Policy.
    """
    if modifiers is None and output_policy is None:
        return f'amdgpu::ldexp::evaluate<{_FORMAT}>({value}, {exponent}, {POLICY})'
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation_expr(),
        value,
        exponent,
    ]
    return f'amdgpu::floating_operation::apply<{_FORMAT}>({", ".join(args)})'


def simd_functor() -> str:
    """Return the VOP2 SIMD functor on raw lanes."""
    return operation_expr(policy_expr())


def simd_probe(template_name: str, true16_vop3: bool = False) -> str | None:
    """Emit the VOP3 SIMD probe, or None for the regular dispatch.

    The VOP2 form uses its SIMD_VOP2_BINARY entry, so true16 e32 bodies select
    their register halves through the same functor.
    """
    if template_name != 'v_ldexp_f16_vop3':
        return None
    macro = 'VOP3_LDEXP_TRUE16_RAW_FP16' if true16_vop3 else 'VOP3_LDEXP_RAW_FP16'
    return f'  ROCJITSU_TRY_SIMD_{macro}({simd_functor()});'
