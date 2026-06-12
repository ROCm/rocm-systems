# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Register layout functions for MFMA/WMMA matrix instructions.

Ported verbatim from mma_exec.h — GFX9 layout (CDNA, RDNA3/3.5/4) and
WMMA v3 layout (GFX1250).
"""

from __future__ import annotations

from typing import NamedTuple

WMMA_WAVE32 = 32


class InputLoc(NamedTuple):
    vgpr_offset: int
    lane: int
    sub_element: int
    bit_offset: int
    data_bits: int


class OutputLoc(NamedTuple):
    reg: int
    lane: int


class PackedOutputLoc(NamedTuple):
    reg: int
    lane: int
    sub_element: int


# ---------------------------------------------------------------------------
# GFX9 layout (CDNA1-4, RDNA3/3.5/4)
# ---------------------------------------------------------------------------


def input_loc(
    dim: int, K: int, B: int, i: int, k: int, b: int, data_bits: int
) -> InputLoc:
    lanes_per_block = 64 // (dim * B)
    elems_per_group = K // lanes_per_block

    if elems_per_group > 16 and data_bits >= 8:
        chunk = k // 16
        g = chunk % lanes_per_block
        local = (chunk // lanes_per_block) * 16 + (k % 16)
        lane = b * dim + g * dim * B + i
    else:
        local = k % elems_per_group
        lane = b * dim + (k // elems_per_group) * dim * B + i

    if data_bits == 64:
        return InputLoc(local * 2, lane, 0, 0, data_bits)
    if data_bits == 32:
        return InputLoc(local, lane, 0, 0, data_bits)
    bit = local * data_bits
    bit_in_word = bit % 32
    sub_element = (bit_in_word // data_bits) if (32 % data_bits == 0) else 0
    return InputLoc(bit // 32, lane, sub_element, bit_in_word, data_bits)


def output_loc_32(M: int, N: int, i: int, j: int, b: int) -> OutputLoc:
    multirows = 64 // N
    mn_div_4 = (M * N) // 4
    blocks_per_reg = (64 + mn_div_4 - 1) // mn_div_4

    reg = b * ((M * N) // 64) + (i // (4 * multirows)) * 4 + (i % 4)
    lane = (b % blocks_per_reg) * N + ((i // 4) % multirows) * blocks_per_reg * N + j
    return OutputLoc(reg, lane)


def output_loc_16(M: int, N: int, i: int, j: int, b: int) -> PackedOutputLoc:
    """Map f32 output position to packed 16-bit position using GFX9 layout.

    Two consecutive f32 register positions pack into one 32-bit VGPR as two
    16-bit sub-elements: reg//2 holds the VGPR offset, reg%2 the sub-element.
    """
    f32 = output_loc_32(M, N, i, j, b)
    return PackedOutputLoc(f32.reg // 2, f32.lane, f32.reg % 2)


# ---------------------------------------------------------------------------
# GFX1250 WMMA v3 layout
# ---------------------------------------------------------------------------


def wmma_input_loc(dim: int, K: int, i: int, k: int, data_bits: int) -> InputLoc:
    lanes_per_group = WMMA_WAVE32 // dim
    elems_per_group = K // lanes_per_group

    local = k % elems_per_group
    lane = (k // elems_per_group) * dim + i

    if dim == 16 and K >= 32:
        block_elems = elems_per_group // 2
        if data_bits == 4 and K == 128:
            block_elems = 16
        if block_elems != 0:
            lane_group = (k // block_elems) % lanes_per_group
            reg_group = k // (block_elems * lanes_per_group)
            local = reg_group * block_elems + (k % block_elems)
            lane = lane_group * dim + i

    if data_bits == 64:
        return InputLoc(local * 2, lane, 0, 0, data_bits)
    if data_bits == 32:
        return InputLoc(local, lane, 0, 0, data_bits)
    bit = local * data_bits
    bit_in_word = bit % 32
    sub_element = (bit_in_word // data_bits) if (32 % data_bits == 0) else 0
    return InputLoc(bit // 32, lane, sub_element, bit_in_word, data_bits)


def wmma_output_loc_32(M: int, N: int, row: int, col: int) -> OutputLoc:
    elems_per_lane = (M * N) // WMMA_WAVE32
    lane = (row // elems_per_lane) * N + col
    reg = row % elems_per_lane
    return OutputLoc(reg, lane)


def wmma_output_loc_16(M: int, N: int, row: int, col: int) -> PackedOutputLoc:
    elems_per_lane = (M * N) // WMMA_WAVE32
    lane = (row // elems_per_lane) * N + col
    elem = row % elems_per_lane
    return PackedOutputLoc(elem // 2, lane, elem % 2)


# ---------------------------------------------------------------------------
# Lane permutation for cbsz/abid and blgp
# ---------------------------------------------------------------------------


def permute_a_lane(lane: int, cbsz: int, abid: int) -> int:
    if cbsz == 0:
        return lane
    S = 64 >> cbsz
    return (lane % S) + S * abid


def permute_b_lane(lane: int, blgp: int) -> int:
    if blgp == 0:
        return lane
    if blgp == 1:
        return lane % 32
    if blgp == 2:
        return lane % 32 + 32
    if blgp == 3:
        return (lane + 16) % 64
    if blgp == 4:
        return lane % 16
    if blgp == 5:
        return lane % 16 + 16
    if blgp == 6:
        return lane % 16 + 32
    if blgp == 7:
        return lane % 16 + 48
    return lane
