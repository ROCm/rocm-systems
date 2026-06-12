# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Golden compute and VGPR array builder for MFMA/WMMA CTS tests.

Uses numpy matmul as an independent arithmetic reference and ported
layout functions for VGPR placement.
"""

from __future__ import annotations

import re
import struct
from typing import NamedTuple

import numpy as np

from amdisa.codegen.cts.mma_layout import (
    input_loc,
    output_loc_16,
    output_loc_32,
    permute_a_lane,
    permute_b_lane,
    wmma_input_loc,
    wmma_output_loc_16,
    wmma_output_loc_32,
)

ACC_VGPR_OFFSET = 256


class MfmaParams(NamedTuple):
    M: int
    N: int
    K: int
    B: int
    a_bits: int
    b_bits: int
    out_type: str
    layout: str
    has_cbsz: bool


# ---------------------------------------------------------------------------
# Mnemonic → instruction parameters
# ---------------------------------------------------------------------------

_MFMA_RE = re.compile(
    r"v_mfma_([a-z0-9]+)_(\d+)x(\d+)x(\d+)" r"(?:_(\d+)b)?" r"_?([a-z][a-z0-9_]*)"
)

_WMMA_RE = re.compile(r"v_wmma_([a-z0-9]+)_(\d+)x(\d+)x(\d+)" r"_([a-z0-9_]+)")

_INPUT_BITS = {
    "f32": 32,
    "f16": 16,
    "bf16": 16,
    "bf16_1k": 16,
    "i8": 8,
    "iu8": 8,
    "iu4": 4,
    "fp8_fp8": 8,
    "fp8_bf8": 8,
    "bf8_fp8": 8,
    "bf8_bf8": 8,
    "f8f6f4": 8,
}


def _cdna_default_B(M: int, N: int, K: int, a_bits: int) -> int:
    """Derive block count for CDNA MFMA when not explicit in name."""
    if a_bits == 32:
        if K == 1:
            return {32: 2, 16: 4, 4: 16}[M]
        return 1
    if a_bits in (16, 8):
        small_K = {32: (4 if a_bits == 16 else 4), 16: (4,), 4: (4,)}
        if M == 4:
            return 16
        if M == 16 and K <= 4:
            return 4
        if M == 32 and K <= 4:
            return 2
        return 1
    return 1


_CDNA_B_LOOKUP: dict[tuple[int, int, int, int], int] = {
    # (M, N, K, a_bits) → B
    # f32 inputs
    (32, 32, 1, 32): 2,
    (16, 16, 1, 32): 4,
    (4, 4, 1, 32): 16,
    (32, 32, 2, 32): 1,
    (16, 16, 4, 32): 1,
    # f16/bf16 inputs (16-bit)
    (32, 32, 2, 16): 2,
    (16, 16, 2, 16): 4,
    (4, 4, 2, 16): 16,
    (32, 32, 4, 16): 2,
    (16, 16, 4, 16): 4,
    (4, 4, 4, 16): 16,
    (32, 32, 8, 16): 1,
    (16, 16, 16, 16): 1,
    (32, 32, 16, 16): 1,
    (16, 16, 32, 16): 1,
    # i8 inputs (8-bit)
    (32, 32, 4, 8): 2,
    (16, 16, 4, 8): 4,
    (4, 4, 4, 8): 16,
    (32, 32, 8, 8): 1,
    (16, 16, 16, 8): 1,
    (32, 32, 16, 8): 1,
    (16, 16, 32, 8): 1,
    (32, 32, 32, 8): 1,
    (16, 16, 64, 8): 1,
    # fp8/bf8 (8-bit)
    (16, 16, 32, 8): 1,
    (32, 32, 16, 8): 1,
    # f8f6f4 unscaled (fp8 mode, 8-bit)
    (16, 16, 128, 8): 1,
    (32, 32, 64, 8): 1,
}

_CDNA_BF16_NONK_B: dict[tuple[int, int, int], int] = {
    # bf16 (non-1k) has fewer blocks than f16/bf16_1k at the same K
    (32, 32, 4): 1,
}

_RDNA_WMMA_B: dict[tuple[int, int, int], int] = {
    # (M, N, K) → B for RDNA3/3.5/4 WMMA
    (16, 16, 16): 2,
    (16, 16, 32): 2,
}


def parse_mfma_params(
    mnemonic: str,
    isa_group: str,
    isa_name: str = "",
) -> MfmaParams | None:
    """Parse instruction mnemonic into execution parameters.

    Args:
        mnemonic: e.g. "v_mfma_f32_32x32x4_2b_f16"
        isa_group: "cdna" | "rdna" | "gfx1250"
        isa_name: specific ISA (e.g. "rdna4") for B-value disambiguation.

    Returns:
        MfmaParams or None if unparseable/unsupported.
    """
    mn = mnemonic.lower()

    if mn.startswith("v_mfma_"):
        m = _MFMA_RE.match(mn)
        if not m:
            return None
        out_type = m.group(1)
        M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
        explicit_B = int(m.group(5)) if m.group(5) else None
        in_suffix = m.group(6)

        a_bits = _INPUT_BITS.get(in_suffix)
        b_bits = a_bits
        if in_suffix in ("fp8_bf8", "bf8_fp8"):
            a_bits = 8
            b_bits = 8

        if a_bits is None:
            return None

        if explicit_B is not None:
            B = explicit_B
        else:
            B = _CDNA_B_LOOKUP.get((M, N, K, a_bits), _cdna_default_B(M, N, K, a_bits))
            if in_suffix == "bf16":
                B = _CDNA_BF16_NONK_B.get((M, N, K), B)

        return MfmaParams(
            M=M,
            N=N,
            K=K,
            B=B,
            a_bits=a_bits,
            b_bits=b_bits,
            out_type=out_type,
            layout="gfx9",
            has_cbsz=True,
        )

    if mn.startswith("v_wmma_"):
        m = _WMMA_RE.match(mn)
        if not m:
            return None
        out_type = m.group(1)
        M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
        in_suffix = m.group(5)

        a_bits = _INPUT_BITS.get(in_suffix)
        b_bits = a_bits
        if in_suffix in ("fp8_bf8", "bf8_fp8"):
            a_bits = 8
            b_bits = 8
        if a_bits is None:
            return None

        if isa_group == "gfx1250":
            return MfmaParams(
                M=M,
                N=N,
                K=K,
                B=1,
                a_bits=a_bits,
                b_bits=b_bits,
                out_type=out_type,
                layout="wmma_v3",
                has_cbsz=False,
            )

        B = _RDNA_WMMA_B.get((M, N, K), 2)
        if isa_name == "rdna4" and out_type in ("f16", "bf16"):
            B = 1
        return MfmaParams(
            M=M,
            N=N,
            K=K,
            B=B,
            a_bits=a_bits,
            b_bits=b_bits,
            out_type=out_type,
            layout="gfx9",
            has_cbsz=False,
        )

    return None


# ---------------------------------------------------------------------------
# Float format conversion
# ---------------------------------------------------------------------------


def _f16_to_float(bits: int) -> float:
    """IEEE 754 half-precision to float."""
    return float(np.frombuffer(struct.pack("<H", bits & 0xFFFF), dtype=np.float16)[0])


def _float_to_f16(val: float) -> int:
    """Float to IEEE 754 half-precision bits."""
    return int(np.float16(val).view(np.uint16))


def _bf16_to_float(bits: int) -> float:
    """BFloat16 to float: bits are upper 16 bits of f32."""
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16(val: float) -> int:
    """Float to BFloat16 (truncation, not rounding)."""
    fbits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
    return (fbits >> 16) & 0xFFFF


def _fp8_e4m3_to_float(bits: int) -> float:
    """OCP FP8 E4M3 (fp8) to float."""
    b = bits & 0xFF
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        val = (mant / 8.0) * (2.0**-6)
    else:
        val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return -val if sign else val


def _float_to_fp8_e4m3(val: float) -> int:
    """Float to OCP FP8 E4M3 (simple truncation)."""
    if val != val:
        return 0x7F
    sign = 0
    if val < 0:
        sign = 1
        val = -val
    if val == 0:
        return sign << 7
    if val >= 448.0:
        return (sign << 7) | 0x7E
    exp = 0
    mant_f = val
    if mant_f < 2**-6 / 8:
        return sign << 7
    if mant_f < 2**-6:
        mant = int(mant_f / (2**-6) * 8)
        return (sign << 7) | (mant & 0x7)
    import math

    exp = int(math.floor(math.log2(mant_f))) + 7
    exp = max(1, min(exp, 14))
    mant = int((mant_f / (2 ** (exp - 7)) - 1.0) * 8)
    mant = max(0, min(mant, 7))
    return (sign << 7) | (exp << 3) | mant


def _bf8_e5m2_to_float(bits: int) -> float:
    """OCP BF8 E5M2 (bf8) to float."""
    b = bits & 0xFF
    sign = (b >> 7) & 1
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 0x1F:
        if mant != 0:
            return float("nan")
        return float("-inf") if sign else float("inf")
    if exp == 0:
        val = (mant / 4.0) * (2.0**-14)
    else:
        val = (1.0 + mant / 4.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def _float_to_bf8_e5m2(val: float) -> int:
    """Float to OCP BF8 E5M2 (simple truncation)."""
    if val != val:
        return 0x7F
    sign = 0
    if val < 0:
        sign = 1
        val = -val
    if val == 0:
        return sign << 7
    if val >= 57344.0:
        return (sign << 7) | 0x7C
    import math

    if val < 2**-14 / 4:
        return sign << 7
    if val < 2**-14:
        mant = int(val / (2**-14) * 4)
        return (sign << 7) | (mant & 0x3)
    exp = int(math.floor(math.log2(val))) + 15
    exp = max(1, min(exp, 30))
    mant = int((val / (2 ** (exp - 15)) - 1.0) * 4)
    mant = max(0, min(mant, 3))
    return (sign << 7) | (exp << 2) | mant


def _float_to_f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def _f32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _i32_to_bits(val: int) -> int:
    return val & 0xFFFFFFFF


_TO_FLOAT = {
    "f32": _f32_bits_to_float,
    "f16": _f16_to_float,
    "bf16": _bf16_to_float,
    "bf16_1k": _bf16_to_float,
    "fp8": _fp8_e4m3_to_float,
    "bf8": _bf8_e5m2_to_float,
    "fp8_fp8": _fp8_e4m3_to_float,
    "bf8_bf8": _bf8_e5m2_to_float,
    "fp8_bf8": None,
    "bf8_fp8": None,
    "i8": None,
}

_FROM_FLOAT = {
    "f32": _float_to_f32_bits,
    "f16": _float_to_f16,
    "bf16": _float_to_bf16,
    "bf16_1k": _float_to_bf16,
    "fp8": _float_to_fp8_e4m3,
    "bf8": _float_to_bf8_e5m2,
}


def _input_a_type(in_suffix: str) -> str:
    if "_" in in_suffix:
        return in_suffix.split("_")[0]
    return in_suffix


def _input_b_type(in_suffix: str) -> str:
    if "_" in in_suffix:
        return in_suffix.split("_")[1]
    return in_suffix


# ---------------------------------------------------------------------------
# Matrix value generators
# ---------------------------------------------------------------------------


def make_a_values(M: int, K: int, in_type: str) -> np.ndarray:
    """Generate distinct per-element A[M][K] values.

    Values 1-5 are exact in all target float formats.
    """
    vals = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        for k in range(K):
            vals[i, k] = (i * K + k) % 5 + 1
    return vals


def make_b_values(K: int, N: int, in_type: str) -> np.ndarray:
    """Generate distinct per-element B[K][N] values (1-7)."""
    vals = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        for j in range(N):
            vals[k, j] = (k * N + j) % 7 + 1
    return vals


def make_c_values(M: int, N: int) -> np.ndarray:
    """Generate accumulator C[M][N] values (0-2)."""
    vals = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            vals[i, j] = (i * N + j) % 3
    return vals


# ---------------------------------------------------------------------------
# Golden compute (numpy matmul)
# ---------------------------------------------------------------------------


def compute_golden_f32(
    M: int,
    N: int,
    K: int,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """D = A @ B + C in float32."""
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    D = (a @ b).astype(np.float32)
    if C is not None:
        D = (D + C.astype(np.float32)).astype(np.float32)
    return D


def compute_golden_i32(
    M: int,
    N: int,
    K: int,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """D = A @ B + C in int32."""
    a = A.astype(np.int32)
    b = B.astype(np.int32)
    D = a @ b
    if C is not None:
        D = D + C.astype(np.int32)
    return D


# ---------------------------------------------------------------------------
# VGPR array builder
# ---------------------------------------------------------------------------


def _pack_subelement(word: int, value: int, bit_offset: int, data_bits: int) -> int:
    """Pack a sub-element value into a 32-bit word at given bit offset."""
    mask = (1 << data_bits) - 1
    word &= ~(mask << bit_offset)
    word |= (value & mask) << bit_offset
    return word


def _convert_input_value(val: float, fmt: str) -> int:
    """Convert float to packed bits in the target input format."""
    conv = _FROM_FLOAT.get(fmt)
    if conv is None:
        return int(val) & 0xFF if fmt in ("i8", "iu8") else int(val) & 0xF
    return conv(val)


def _effective_wave_size(layout: str, B: int) -> int:
    """Return the effective wave size for VGPR array dimensions."""
    if layout == "wmma_v3":
        return 32
    return 64


def _physicalize_vgprs(
    vgprs: list[list[int]],
    phys_wf: int,
) -> list[list[int]]:
    """Convert virtual wide VGPR arrays to physical register layout.

    On wave32 with B=2, the GFX9 layout produces 64 virtual lanes per VGPR.
    Physical VGPR = logical_vgpr * stride + virtual_lane // phys_wf,
    physical lane = virtual_lane % phys_wf, where stride = eff_wf // phys_wf.
    Each logical VGPR maps to `stride` physical VGPRs.
    """
    eff_wf = len(vgprs[0]) if vgprs else 0
    if eff_wf <= phys_wf:
        return vgprs

    stride = eff_wf // phys_wf
    num_logical = len(vgprs)
    num_physical = num_logical * stride
    phys: list[list[int]] = [[0] * phys_wf for _ in range(num_physical)]

    for v, row in enumerate(vgprs):
        for lane, val in enumerate(row):
            if val != 0:
                preg = v * stride + lane // phys_wf
                plane = lane % phys_wf
                phys[preg][plane] = val

    return phys


def build_src_vgprs(
    p: MfmaParams,
    dim: str,
    matrix: np.ndarray,
    in_type: str,
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
) -> list[list[int]]:
    """Build VGPR data for an input matrix (A or B).

    Args:
        p: MfmaParams for the instruction.
        dim: "A" or "B" — determines which dimension is the row.
        matrix: Logical matrix values (float or int).
        in_type: Format name ("f16", "bf16", "fp8", "bf8", "i8", "f32", ...).
        cbsz, abid, blgp: Lane permutation modifiers.

    Returns:
        List of VGPRs, each a list of uint32 values indexed by lane.
    """
    eff_wf = _effective_wave_size(p.layout, p.B)
    data_bits = _INPUT_BITS.get(in_type, p.a_bits)

    if dim == "A":
        row_dim, K = p.M, p.K
    else:
        row_dim, K = p.N, p.K

    max_vgpr = 0
    placements: list[tuple[int, int, int, int, int]] = []

    for b in range(p.B):
        for i in range(row_dim):
            for k in range(K):
                if p.layout == "gfx9":
                    loc = input_loc(row_dim, K, p.B, i, k, b, data_bits)
                else:
                    loc = wmma_input_loc(row_dim, K, i, k, data_bits)
                lane = loc.lane
                if dim == "A" and cbsz != 0:
                    lane = permute_a_lane(lane, cbsz, abid)
                elif dim == "B" and blgp != 0:
                    lane = permute_b_lane(lane, blgp)
                val_bits = _convert_input_value(
                    float(matrix[i, k]) if dim == "A" else float(matrix[k, i]),
                    in_type,
                )
                if loc.vgpr_offset > max_vgpr:
                    max_vgpr = loc.vgpr_offset
                placements.append(
                    (loc.vgpr_offset, lane, val_bits, loc.bit_offset, data_bits)
                )

    num_vgprs = max_vgpr + 1
    vgprs: list[list[int]] = [[0] * eff_wf for _ in range(num_vgprs)]

    for vgpr_off, lane, val_bits, bit_off, dbits in placements:
        if lane < eff_wf:
            vgprs[vgpr_off][lane] = _pack_subelement(
                vgprs[vgpr_off][lane],
                val_bits,
                bit_off,
                dbits,
            )

    return vgprs


def build_dst_vgprs_f32(
    p: MfmaParams,
    D: np.ndarray,
) -> list[list[int]]:
    """Build expected VGPR data for f32/i32 output.

    Args:
        p: MfmaParams.
        D: Golden output matrix D[M][N] (float32 or int32).

    Returns:
        List of VGPRs, each a list of uint32 values indexed by lane.
    """
    eff_wf = _effective_wave_size(p.layout, p.B)
    max_vgpr = 0
    placements: list[tuple[int, int, int]] = []

    for b in range(p.B):
        for i in range(p.M):
            for j in range(p.N):
                if p.layout == "gfx9":
                    loc = output_loc_32(p.M, p.N, i, j, b)
                else:
                    loc = wmma_output_loc_32(p.M, p.N, i, j)
                if p.out_type in ("f32",):
                    val_bits = _float_to_f32_bits(float(D[i, j]))
                else:
                    val_bits = _i32_to_bits(int(D[i, j]))
                if loc.reg > max_vgpr:
                    max_vgpr = loc.reg
                placements.append((loc.reg, loc.lane, val_bits))

    num_vgprs = max_vgpr + 1
    vgprs: list[list[int]] = [[0] * eff_wf for _ in range(num_vgprs)]

    for reg, lane, val_bits in placements:
        if lane < eff_wf:
            vgprs[reg][lane] = val_bits

    return vgprs


def build_dst_vgprs_f16(
    p: MfmaParams,
    D: np.ndarray,
    out_fmt: str,
) -> list[list[int]]:
    """Build expected VGPR data for f16/bf16 packed output (WMMA only).

    Args:
        p: MfmaParams.
        D: Golden output matrix D[M][N].
        out_fmt: "f16" or "bf16".

    Returns:
        List of VGPRs, each a list of uint32 values indexed by lane.
    """
    eff_wf = _effective_wave_size(p.layout, p.B)
    max_vgpr = 0
    placements: list[tuple[int, int, int, int]] = []

    to_bits = _float_to_f16 if out_fmt == "f16" else _float_to_bf16

    for b in range(p.B):
        for i in range(p.M):
            for j in range(p.N):
                if p.layout == "gfx9":
                    loc = output_loc_16(p.M, p.N, i, j, b)
                else:
                    loc = wmma_output_loc_16(p.M, p.N, i, j)
                val_bits = to_bits(float(D[i, j]))
                if loc.reg > max_vgpr:
                    max_vgpr = loc.reg
                placements.append((loc.reg, loc.lane, val_bits, loc.sub_element * 16))

    num_vgprs = max_vgpr + 1
    vgprs: list[list[int]] = [[0] * eff_wf for _ in range(num_vgprs)]

    for reg, lane, val_bits, bit_off in placements:
        if lane < eff_wf:
            vgprs[reg][lane] = _pack_subelement(
                vgprs[reg][lane],
                val_bits,
                bit_off,
                16,
            )

    return vgprs


def build_acc_vgprs(
    p: MfmaParams,
    C: np.ndarray | None,
) -> list[list[int]] | None:
    """Build accumulator VGPR data (pre-filled src2).

    Returns None if C is None (zero accumulator test).
    """
    if C is None:
        return None

    if p.out_type in ("f16", "bf16"):
        return build_dst_vgprs_f16(p, C, p.out_type)
    return build_dst_vgprs_f32(p, C)


# ---------------------------------------------------------------------------
# Input suffix parsing for mixed types
# ---------------------------------------------------------------------------


def in_suffix_from_mnemonic(mnemonic: str) -> str:
    """Extract the input type suffix from an MFMA/WMMA mnemonic."""
    mn = mnemonic.lower()
    m = _MFMA_RE.match(mn)
    if m:
        return m.group(6)
    m = _WMMA_RE.match(mn)
    if m:
        return m.group(5)
    return ""


def a_format(in_suffix: str) -> str:
    """Get the A-matrix element format from input suffix."""
    parts = in_suffix.split("_")
    if len(parts) >= 2 and parts[0] in ("fp8", "bf8"):
        return parts[0]
    return in_suffix


def b_format(in_suffix: str) -> str:
    """Get the B-matrix element format from input suffix."""
    parts = in_suffix.split("_")
    if len(parts) >= 2 and parts[0] in ("fp8", "bf8"):
        return parts[1]
    return in_suffix


# ---------------------------------------------------------------------------
# Complete test case generator
# ---------------------------------------------------------------------------


class MfmaTestData(NamedTuple):
    mnemonic: str
    src0_vgprs: list[list[int]]
    src1_vgprs: list[list[int]]
    acc_vgprs: list[list[int]] | None
    expected_vgprs: list[list[int]]
    eff_wave_size: int
    params: MfmaParams
    config_label: str


def generate_test_case(
    mnemonic: str,
    isa_group: str,
    has_acc: bool = False,
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
    config_label: str = "basic",
    isa_name: str = "",
) -> MfmaTestData | None:
    """Generate a complete test case for one MFMA/WMMA instruction.

    Args:
        mnemonic: Instruction mnemonic.
        isa_group: "cdna" | "rdna" | "gfx1250".
        has_acc: If True, pre-fill accumulator with non-zero values.
        cbsz, abid, blgp: Lane permutation modifiers (CDNA only).
        config_label: Label for the test configuration.
        isa_name: Specific ISA name for B-value disambiguation.

    Returns:
        MfmaTestData or None if unsupported.
    """
    p = parse_mfma_params(mnemonic, isa_group, isa_name=isa_name)
    if p is None:
        return None

    in_suffix = in_suffix_from_mnemonic(mnemonic)
    a_fmt = a_format(in_suffix)
    b_fmt = b_format(in_suffix)
    is_int = p.out_type == "i32"

    A = make_a_values(p.M, p.K, a_fmt)
    B = make_b_values(p.K, p.N, b_fmt)

    if is_int:
        A = np.clip(A, -128, 127) if "i8" in in_suffix else np.clip(A, -8, 7)

    C = make_c_values(p.M, p.N) if has_acc else None

    if is_int:
        D = compute_golden_i32(p.M, p.N, p.K, A, B, C)
    else:
        D = compute_golden_f32(p.M, p.N, p.K, A, B, C)

    src0 = build_src_vgprs(p, "A", A, a_fmt, cbsz, abid, blgp)
    src1 = build_src_vgprs(p, "B", B, b_fmt, cbsz, abid, blgp)
    acc = build_acc_vgprs(p, C) if has_acc else None

    if p.out_type in ("f16", "bf16"):
        expected = build_dst_vgprs_f16(p, D, p.out_type)
    else:
        expected = build_dst_vgprs_f32(p, D)

    eff_wf = _effective_wave_size(p.layout, p.B)
    phys_wf = 32 if isa_group in ("rdna", "gfx1250") else 64

    if eff_wf > phys_wf:
        src0 = _physicalize_vgprs(src0, phys_wf)
        src1 = _physicalize_vgprs(src1, phys_wf)
        if acc is not None:
            acc = _physicalize_vgprs(acc, phys_wf)
        expected = _physicalize_vgprs(expected, phys_wf)
        eff_wf = phys_wf

    return MfmaTestData(
        mnemonic=mnemonic,
        src0_vgprs=src0,
        src1_vgprs=src1,
        acc_vgprs=acc,
        expected_vgprs=expected,
        eff_wave_size=eff_wf,
        params=p,
        config_label=config_label,
    )
