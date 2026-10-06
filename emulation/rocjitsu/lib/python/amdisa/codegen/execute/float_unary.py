# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/fract.h and frexp.h calls for unary operations on raw encodings.

Scalar bodies and SIMD functors use the same helpers with the same MODE policy,
resolved once before the lane loop. VOP3 forms use shared/floating_operation.h
to apply ABS/NEG before the operation and OMOD/CLAMP afterward.
"""

import re
from dataclasses import dataclass

# Format of the source and floating result, by data type.
FORMATS: dict[str, str] = {'f16': 'F16', 'f32': 'F32', 'f64': 'F64'}


@dataclass(frozen=True)
class Form:
    """One operation: its C++ evaluation, functor and MODE policy."""

    evaluate: str
    operation: str
    policy: str
    # Name of the policy the scalar body declares before the lane loop.
    policy_name: str
    # Whether the result is a floating encoding that takes output modifiers;
    # otherwise it is an integer that ABS/NEG cannot change.
    float_result: bool = True


_INPUT_POLICY = 'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_{mode}())'

FORMS: dict[str, Form] = {
    'fract': Form(
        evaluate='amdgpu::fract::evaluate',
        operation='amdgpu::fract::Operation',
        policy='amdgpu::fract::Policy::make(wf.fp_denorm_mode_{mode}(), wf.fp_round_mode_{mode}())',
        policy_name='fract_policy',
    ),
    'frexp_mant': Form(
        evaluate='amdgpu::frexp::mantissa',
        operation='amdgpu::frexp::Mantissa',
        policy=_INPUT_POLICY,
        policy_name='frexp_policy',
    ),
    'frexp_exp': Form(
        evaluate='amdgpu::frexp::exponent',
        operation='amdgpu::frexp::Exponent',
        policy=_INPUT_POLICY,
        policy_name='frexp_policy',
        float_result=False,
    ),
}

# VOP1/VOP3 template names of the supported forms, e.g. v_frexp_exp_i32_f64_vop3.
_TEMPLATE = re.compile(
    r'v_(?P<form>fract|frexp_mant|frexp_exp)(?:_i16|_i32)?'
    r'_(?P<dtype>f16|f32|f64)_(?P<encoding>vop1|vop3)'
)


def _format(dtype: str) -> str:
    return f'amdgpu::fp_format::{FORMATS[dtype]}'


def policy_expr(form: str, dtype: str) -> str:
    """Return the MODE policy of ``dtype``'s fields for ``form``."""
    mode = 'f32' if dtype == 'f32' else 'f16_f64'
    return FORMS[form].policy.format(mode=mode)


def policy_decl(form: str, dtype: str, indent: str = '  ') -> str:
    """Declare the scalar policy once, before the lane loop."""
    return f'{indent}const auto {FORMS[form].policy_name} = {policy_expr(form, dtype)};'


def operation_expr(form: str, dtype: str, policy: str) -> str:
    """Capture the policy in an operation free of instruction modifiers."""
    return f'{FORMS[form].operation}<{_format(dtype)}>{{{policy}}}'


def unary_expr(
    form: str,
    dtype: str,
    read: str,
    modifiers: tuple[str, str] | None = None,
    output_policy: str | None = None,
) -> str:
    """Emit the operation on raw source bits, with modifiers when present.

    ``modifiers`` holds the (ABS, NEG) fields; ``output_policy`` names a
    declared output_modifier::Policy. Integer results ignore both, and an F64
    source's exponent is narrowed to its I32 destination.
    """
    policy = FORMS[form].policy_name
    evaluate = f'{FORMS[form].evaluate}<{_format(dtype)}>({read}, {policy})'
    if not FORMS[form].float_result:
        return f'static_cast<uint32_t>({evaluate})' if dtype == 'f64' else evaluate
    if modifiers is None and output_policy is None:
        return evaluate
    source_fields = ', '.join(modifiers or ('0u', '0u'))
    args = [
        f'amdgpu::floating_operation::SourceModifiers{{{source_fields}}}',
        output_policy or 'amdgpu::output_modifier::Policy{}',
        operation_expr(form, dtype, policy),
        read,
    ]
    return f'amdgpu::floating_operation::apply<{_format(dtype)}>({", ".join(args)})'


def vop1_functor(form: str, dtype: str) -> str:
    """Return the VOP1 SIMD functor on raw lanes, free of modifiers."""
    return operation_expr(form, dtype, policy_expr(form, dtype))


def simd_probe(template_name: str, true16_vop3: bool = False) -> str | None:
    """Emit the SIMD probe for VOP3 and F64 forms, or None for the regular dispatch.

    F16/F32 VOP1 forms use SIMD_VOP1_UNARY entries built by vop1_functor, so
    true16 e32 bodies select their register halves through the same functor.
    """
    match = _TEMPLATE.fullmatch(template_name)
    if match is None:
        return None
    form, dtype, encoding = match['form'], match['dtype'], match['encoding']
    functor = vop1_functor(form, dtype)
    if not FORMS[form].float_result:
        # The exponent ignores ABS/NEG and takes no output modifiers.
        if dtype == 'f64':
            narrow = (
                f'[operation = {functor}](auto s) {{ return util::stdx::static_simd_cast<'
                'util::narrow32<uint32_t>>(operation(std::bit_cast<util::native<uint64_t>>(s))); }'
            )
            return f'  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(uint32_t, {narrow});'
        if encoding == 'vop1':
            return None
        if dtype == 'f16':
            macro = 'VOP3_UNARY_TRUE16_B16' if true16_vop3 else 'VOP3_UNARY_B16'
            return f'  ROCJITSU_TRY_SIMD_{macro}({functor});'
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, {functor});'
    if encoding == 'vop1':
        if dtype != 'f64':
            return None
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(uint64_t, {functor});'
    if dtype == 'f64':
        macro = 'VOP3_UNARY_RAW_FP64'
    elif dtype == 'f16':
        macro = 'VOP3_UNARY_TRUE16_RAW_FP16' if true16_vop3 else 'VOP3_UNARY_RAW_FP16'
    else:
        macro = 'VOP3_UNARY_RAW_FP'
    return f'  ROCJITSU_TRY_SIMD_{macro}({_format(dtype)}, {functor});'
