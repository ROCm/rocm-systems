# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOP3 source and destination modifier helpers for execute body generation.

These are pure functions that emit C++ lines for VOP3 input modifiers
(abs, neg) and output modifiers (omod, clamp). They take explicit
parameters rather than accessing transient instance state.
"""

from __future__ import annotations


def vop3_src_mod(varname: str, src_idx: int, has_abs: bool,
                 indent: str = '    ') -> list[str]:
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
            f'{indent}if (inst_.abs & (1u << {src_idx})) {varname} = std::fabs({varname});')
    lines.append(
        f'{indent}if (inst_.neg & (1u << {src_idx})) {varname} = -{varname};')
    return lines


def vop3_dst_mod(varname: str, indent: str = '    ') -> list[str]:
    """Generate VOP3 output modifier lines (omod then clamp) for a float result."""
    return [
        f'{indent}if (inst_.omod == 1) {varname} *= 2.0f;',
        f'{indent}else if (inst_.omod == 2) {varname} *= 4.0f;',
        f'{indent}else if (inst_.omod == 3) {varname} *= 0.5f;',
        f'{indent}if (inst_.clamp) {varname} = std::clamp({varname}, 0.0f, 1.0f);',
    ]


def vop3_dst_mod_f64(varname: str, indent: str = '    ') -> list[str]:
    """Generate VOP3 output modifier lines for a double result."""
    return [
        f'{indent}if (inst_.omod == 1) {varname} *= 2.0;',
        f'{indent}else if (inst_.omod == 2) {varname} *= 4.0;',
        f'{indent}else if (inst_.omod == 3) {varname} *= 0.5;',
        f'{indent}if (inst_.clamp) {varname} = std::clamp({varname}, 0.0, 1.0);',
    ]
