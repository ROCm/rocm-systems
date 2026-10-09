# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOP3 source and destination modifier helpers for execute body generation.

These are pure functions that emit C++ lines for VOP3 input modifiers
(abs, neg) and output modifiers (omod, clamp). They take explicit
parameters rather than accessing transient instance state.
"""

from __future__ import annotations

from amdisa.codegen.execute import input_policy


def vop3_src_mod(
    varname: str, src_idx: int, has_abs: bool, indent: str = '    '
) -> list[str]:
    """Apply shared ABS/NEG to a decoded float or double source.

    An encoding without ABS passes a zero field; NEG always uses its source bit.
    """
    abs_field = 'inst_.abs' if has_abs else '0u'
    return [
        f'{indent}{varname} = amdgpu::source_modifier::apply_to_float('
        f'{varname}, {src_idx}, {abs_field}, inst_.neg);'
    ]


def vop3_dst_mod(
    varname: str, indent: str = '    ', *, omod_result_type: str = 'f32'
) -> list[str]:
    """Modify an F32 intermediate using the selected destination's OMOD policy.

    ``omod_result_type`` selects only the architecture/MODE policy. The emitted
    value remains F32 and is finalized in that format; callers producing a
    narrower destination must also finalize after the architectural narrowing.
    """
    if omod_result_type == 'f32':
        omod_expr = (
            'amdgpu::fp_mode::effective_omod(wf.cu().arch(), '
            'wf.fp_denorm_mode_f32(), wf.ieee_mode(), inst_.omod)'
        )
    elif omod_result_type == 'f16':
        omod_expr = (
            'amdgpu::fp_mode::effective_f16_omod(wf.cu().arch(), '
            'wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), false, inst_.omod)'
        )
    else:
        raise ValueError(
            f'unsupported VOP3 OMOD result policy type: {omod_result_type}'
        )
    return [
        f'{indent}const uint32_t effective_omod = {omod_expr};',
        f'{indent}if (effective_omod == 1) {varname} *= 2.0f;',
        f'{indent}else if (effective_omod == 2) {varname} *= 4.0f;',
        f'{indent}else if (effective_omod == 3) {varname} *= 0.5f;',
        f'{indent}if (inst_.clamp) {varname} = amdgpu::clamp_floating_result({varname}, wf);',
        f'{indent}{varname} = amdgpu::fp_mode::finalize_omod_f32({varname}, effective_omod);',
    ]


def vop3_dst_mod_f64(varname: str, indent: str = '    ') -> list[str]:
    """Generate MODE-aware VOP3 output modifier lines for a double result."""
    return [
        f'{indent}const uint32_t effective_omod = amdgpu::fp_mode::effective_omod('
        'wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), inst_.omod);',
        f'{indent}if (effective_omod == 1) {varname} *= 2.0;',
        f'{indent}else if (effective_omod == 2) {varname} *= 4.0;',
        f'{indent}else if (effective_omod == 3) {varname} *= 0.5;',
        f'{indent}if (inst_.clamp) {varname} = amdgpu::clamp_floating_result({varname}, wf);',
        f'{indent}{varname} = amdgpu::fp_mode::finalize_omod_f64({varname}, effective_omod);',
    ]


# Scalar expressions use inst_; shared-body generation qualifies it later.
OUTPUT_POLICY = 'output_policy'
OUTPUT_MODIFIERS = ('inst_.omod', 'inst_.clamp')


def output_policy_expr(dtype: str, fields: tuple[str, str] = OUTPUT_MODIFIERS) -> str:
    """Emit the effective OMOD/CLAMP policy using dtype's MODE fields.

    ``fields`` holds the (OMOD, CLAMP) expressions from the instruction.
    """
    omod, clamp = fields
    fmt = f'amdgpu::fp_format::{input_policy.FORMATS[dtype]}'
    return f'amdgpu::output_modifier_policy<{fmt}>(wf, {omod}, {clamp})'


def output_policy_decl(
    dtype: str, fields: tuple[str, str] = OUTPUT_MODIFIERS, indent: str = '  '
) -> str:
    """Declare the scalar output policy once, before the lane loop."""
    return f'{indent}const auto {OUTPUT_POLICY} = {output_policy_expr(dtype, fields)};'


def apply_output(dtype: str, bits: str) -> str:
    """Apply OMOD then CLAMP to an already rounded destination encoding."""
    fmt = f'amdgpu::fp_format::{input_policy.FORMATS[dtype]}'
    return f'amdgpu::output_modifier::apply<{fmt}>({bits}, {OUTPUT_POLICY})'
