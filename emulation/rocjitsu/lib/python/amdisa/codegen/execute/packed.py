# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed math execute body generators.

Free functions that emit C++ execute_impl bodies for packed 16-bit
instructions (V_PK_ADD_F16, etc.), packed F32 instructions,
and V_PK_MOV_B32.

The opsel_exprs parameter carries the C++ expressions for op_sel and
op_sel_hi fields, derived from the ISA profile at call time.
"""

from __future__ import annotations


def gen_pk_binop_u64(dst: list[str], src: list[str], op: str) -> str:
    """Forward packed U64 source negation and saturation to rj-arith."""
    if op not in ('add', 'sub'):
        raise ValueError(f'unsupported packed U64 binary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_u64<false>(',
            f'      wf, rj_arith_v_pk_{op}_nc_u64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]});',
        ]
    )


def gen_pk_binop_f64(dst: list[str], src: list[str], op: str) -> str:
    """Forward both FP64 elements and their independent signs to rj-arith."""
    if op not in ('add', 'mul', 'min_num', 'max_num'):
        raise ValueError(f'unsupported packed F64 binary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_f64<false>(',
            f'      wf, rj_arith_v_pk_{op}_f64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]});',
        ]
    )


def gen_pk_ternary_f64(dst: list[str], src: list[str], op: str) -> str:
    """Forward packed FP64 FMA through the carrier-aware guest-FP adapter."""
    if op != 'fma':
        raise ValueError(f'unsupported packed F64 ternary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_f64<true>(',
            f'      wf, rj_arith_v_pk_fma_f64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]}, &{src[2]});',
        ]
    )


def gen_pk_lshl_add_u64(dst: list[str], src: list[str]) -> str:
    """Forward shift-add through the adapter that validates active counts before stores."""
    return '\n'.join(
        [
            '  amdgpu::arith::execute_packed_u64<true>(',
            f'      wf, rj_arith_v_pk_lshl_add_u64_w32, 0, {dst[0]}, {src[0]}, {src[1]}, &{src[2]});',
        ]
    )
