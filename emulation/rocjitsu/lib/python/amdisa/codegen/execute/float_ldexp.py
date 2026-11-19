# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/ldexp.h calls for the remaining legacy LDEXP formats.

Scalar bodies and SIMD functors use the same helper with the same MODE policy,
resolved once before the lane loop. VOP3 forms use shared/floating_operation.h
to apply ABS/NEG to the floating source and OMOD/CLAMP to the result; the
exponent source takes no modifiers.
"""

from amdisa.codegen.execute import input_policy

# Exponent operand width other than the format's: V_LDEXP_F64 takes an I32.
_EXPONENT_WIDTHS: dict[str, int] = {'f64': 32}

# Name of the policy the scalar body declares before the lane loop.
POLICY = 'ldexp_policy'


def policy_expr(dtype: str, omod: str) -> str:
    """Return the MODE policy; F16 saturates under FP16_OVFL.

    ``omod`` is a C++ condition that holds when effective OMOD is enabled: a
    tiny result then flushes before OMOD, which cannot recover tininess from
    the rounded bits. Every caller must decide it.
    """
    mode = input_policy.mode_field(dtype)
    saturate = 'wf.fp16_ovfl()' if dtype == 'f16' else 'false'
    return (
        f'amdgpu::ldexp::Policy::make(wf.fp_denorm_mode_{mode}(), '
        f'wf.fp_round_mode_{mode}(), {saturate}, {omod})'
    )


def policy_decl(dtype: str, omod: str, indent: str = '  ') -> str:
    return f'{indent}const auto {POLICY} = {policy_expr(dtype, omod)};'


def _template_args(dtype: str) -> str:
    """Format and, for F64, the exponent width that differs from the format's."""
    fmt = input_policy.fp_format(dtype)
    width = _EXPONENT_WIDTHS.get(dtype)
    return f'{fmt}, {width}' if width else fmt


def operation_expr(dtype: str, policy: str = POLICY) -> str:
    """Capture the policy in an operation; it applies no instruction modifiers,
    but the policy's flush_tiny, set by effective OMOD or disabled output
    denormals, decides whether a tiny result flushes."""
    return f'amdgpu::ldexp::Operation<{_template_args(dtype)}>{{{policy}}}'


def ldexp_expr(
    dtype: str,
    value: str,
    exponent: str,
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Emit the scaled bits from raw value and exponent register reads.

    ``modifiers`` holds the value's (ABS, NEG) fields; ``output_policy`` names a
    declared output_modifier::Policy. An F64 exponent read is widened to the
    value's 64-bit lane; the operation reads only its low 32 bits.
    """
    if dtype == 'f64':
        exponent = f'static_cast<uint64_t>({exponent})'
    if modifiers is None and output_policy is None:
        return f'amdgpu::ldexp::evaluate<{_template_args(dtype)}>({value}, {exponent}, {POLICY})'
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation_expr(dtype),
        value,
        exponent,
    ]
    return f'amdgpu::floating_operation::apply<{input_policy.fp_format(dtype)}>({", ".join(args)})'


def simd_functor(dtype: str) -> str:
    """Return the VOP2 SIMD functor on raw lanes.

    VOP2 has no OMOD of its own, and SDWA OMOD keeps the scalar body, so
    effective OMOD is always off here.
    """
    return operation_expr(dtype, policy_expr(dtype, 'false'))


def simd_probe(template_name: str, true16_vop3: bool = False) -> str | None:
    """All LDEXP forms route through rj-arith."""
    return None
