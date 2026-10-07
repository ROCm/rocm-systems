# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit the shared/input_denormal.h policy used by floating-point stages.

Comparisons and min/max read the same MODE input-denormal field for each
format, so they declare the policy once per instruction under one name.
"""

# Floating formats with an input policy -> shared/fp_format.h layout.
FORMATS: dict[str, str] = {'f16': 'F16', 'f32': 'F32', 'f64': 'F64'}

# Name of the per-instruction policy declared before the body.
NAME = 'input_policy'


def policy_expr(dtype: str) -> str:
    """Return the input policy for ``dtype`` from the wave's MODE."""
    mode = 'f32' if dtype == 'f32' else 'f16_f64'
    return f'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_{mode}())'


def policy_decl(dtype: str, indent: str = '  ') -> str:
    """Declare the input policy for ``dtype`` once per instruction."""
    return f'{indent}const auto {NAME} = {policy_expr(dtype)};'
