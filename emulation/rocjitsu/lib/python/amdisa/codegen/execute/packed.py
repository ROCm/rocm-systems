# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit paired-64-bit VM bindings.

FP64 arithmetic forwards to rj-arith; U64 still retains legacy pair reads and
commits until its migration. Relative modifiers remain instruction-specific.
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


# FP8 DOT4 semantic operation -> (src0 is FP8, src1 is FP8), as for FP8 WMMA.
