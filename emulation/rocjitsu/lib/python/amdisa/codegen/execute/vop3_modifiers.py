# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOP3 source and destination modifier helpers for execute body generation.

These are pure functions that emit C++ lines for VOP3 input modifiers
(abs, neg) and output modifiers (omod, clamp). They take explicit
parameters rather than accessing transient instance state.
"""

from __future__ import annotations


def vop3_src_mod(
    varname: str, src_idx: int, has_abs: bool, indent: str = '    '
) -> list[str]:
    """Generate VOP3 input modifier lines (abs then neg) for a floating-point src.

    Works for both float and double temporaries. The generated C++ uses
    ``std::fabs`` and unary negation which are type-generic.

    Args:
        varname: C++ variable name to modify in-place.
        src_idx: Source operand index (0, 1, or 2) for the modifier bitmask.
        has_abs: Whether the encoding format has an ``abs`` field.
        indent: Indentation prefix for each generated line.
    """
    lines = []
    if has_abs:
        lines.append(
            f'{indent}if (inst_.abs & (1u << {src_idx})) {varname} = std::fabs({varname});'
        )
    lines.append(f'{indent}if (inst_.neg & (1u << {src_idx})) {varname} = -{varname};')
    return lines


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
