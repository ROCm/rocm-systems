# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared cube opcode mapping and helper calls for execute emitters."""

from __future__ import annotations

CUBE_OPERATIONS = {'cubeid': 'ID', 'cubesc': 'SC', 'cubetc': 'TC', 'cubema': 'MA'}


def cube_expression(op: str, x: str, y: str, z: str) -> str:
    """Emit cube selection with the wave's rounding and NaN policy."""
    return (
        f'::rocjitsu::amdgpu::cube::execute<::rocjitsu::amdgpu::cube::Operation::{CUBE_OPERATIONS[op]}>'
        f'({x}, {y}, {z}, wf.fp_round_mode_f32(), wf.cu().arch(), wf.ieee_mode())'
    )


def cube_omod(expression: str, inst: str = 'inst_') -> str:
    """Emit cube scaling; callers disable generic OMOD and apply CLAMP separately."""
    return (
        f'::rocjitsu::amdgpu::cube::apply_omod({expression}, wf.fp_round_mode_f32(), '
        f'wf.cu().arch(), wf.ieee_mode(), {inst}.omod)'
    )
