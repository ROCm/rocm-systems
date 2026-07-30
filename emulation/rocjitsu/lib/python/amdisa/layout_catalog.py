# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix instruction lane layout catalog with auto-derived XOR masks.

Encodes the register-to-lane mapping for MFMA (CDNA) and WMMA (RDNA)
matrix instructions. The key data for each shape is ``row_group_lanes``:
for each group of ``dst_vgprs`` matrix rows, the base lane index where
that group starts. Cross-ISA layout conversion reduces to a uniform XOR
on lane indices when the only difference is row-group permutation.

Formulas (verified against AMD Matrix Calculator and hardware):

MFMA f32_MxNxK output D[i][j]:
    lane = N * (i // dst_vgprs) + j
    register = v[i % dst_vgprs]

WMMA f32_MxNxK output D[i][j]:
    lane = (N*2) * ((i // dst_vgprs) % 2) + N * (i // (2*dst_vgprs)) + j
    register = v[i % dst_vgprs]
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixLayoutDescriptor:
    """Compact description of a matrix instruction's register-to-lane mapping.

    ``row_group_lanes`` gives the base lane for each group of ``dst_vgprs``
    consecutive rows. Column index is always ``lane % n`` (low lane bits).
    """

    mnemonic_pattern: str
    kind: str
    m: int
    n: int
    k: int
    wave_size: int
    src_elem_bits: int
    dst_elem_bits: int
    src_vgprs: int
    dst_vgprs: int
    row_group_lanes: tuple[int, ...]

    @property
    def num_row_groups(self) -> int:
        return self.m // self.dst_vgprs

    def lane_for_row(self, row: int) -> int:
        """Base lane for a given row (column offset not included)."""
        group = row // self.dst_vgprs
        return self.row_group_lanes[group]


def _mfma_row_group_lanes(m: int, n: int, dst_vgprs: int) -> tuple[int, ...]:
    """Compute MFMA row_group_lanes: lane = N * (group_index)."""
    num_groups = m // dst_vgprs
    return tuple(n * g for g in range(num_groups))


def _wmma_row_group_lanes(m: int, n: int, dst_vgprs: int) -> tuple[int, ...]:
    """Compute WMMA row_group_lanes: groups 1,2 swapped vs MFMA."""
    num_groups = m // dst_vgprs
    return tuple((n * 2) * ((g % 2)) + n * (g // 2) for g in range(num_groups))


def _make_mfma(
    pattern: str,
    m: int,
    n: int,
    k: int,
    src_bits: int,
    dst_bits: int,
    src_vgprs: int,
    dst_vgprs: int,
) -> MatrixLayoutDescriptor:
    return MatrixLayoutDescriptor(
        mnemonic_pattern=pattern,
        kind='mfma',
        m=m,
        n=n,
        k=k,
        wave_size=64,
        src_elem_bits=src_bits,
        dst_elem_bits=dst_bits,
        src_vgprs=src_vgprs,
        dst_vgprs=dst_vgprs,
        row_group_lanes=_mfma_row_group_lanes(m, n, dst_vgprs),
    )


def _make_wmma(
    pattern: str,
    m: int,
    n: int,
    k: int,
    src_bits: int,
    dst_bits: int,
    src_vgprs: int,
    dst_vgprs: int,
) -> MatrixLayoutDescriptor:
    return MatrixLayoutDescriptor(
        mnemonic_pattern=pattern,
        kind='wmma',
        m=m,
        n=n,
        k=k,
        wave_size=64,
        src_elem_bits=src_bits,
        dst_elem_bits=dst_bits,
        src_vgprs=src_vgprs,
        dst_vgprs=dst_vgprs,
        row_group_lanes=_wmma_row_group_lanes(m, n, dst_vgprs),
    )


# =========================================================================
# MFMA layouts (CDNA1-4)
# =========================================================================

MFMA_F32_16X16X16_F16 = _make_mfma(
    'V_MFMA_F32_16X16X16_*',
    16,
    16,
    16,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)

MFMA_F32_32X32X8_F16 = _make_mfma(
    'V_MFMA_F32_32X32X8_*',
    32,
    32,
    8,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=16,
)

MFMA_F32_16X16X32_F16 = _make_mfma(
    'V_MFMA_F32_16X16X32_*',
    16,
    16,
    32,
    src_bits=16,
    dst_bits=32,
    src_vgprs=4,
    dst_vgprs=4,
)

MFMA_F32_32X32X16_F16 = _make_mfma(
    'V_MFMA_F32_32X32X16_*',
    32,
    32,
    16,
    src_bits=16,
    dst_bits=32,
    src_vgprs=4,
    dst_vgprs=16,
)

MFMA_I32_16X16X32_I8 = _make_mfma(
    'V_MFMA_I32_16X16X32_*',
    16,
    16,
    32,
    src_bits=8,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)

MFMA_I32_32X32X16_I8 = _make_mfma(
    'V_MFMA_I32_32X32X16_*',
    32,
    32,
    16,
    src_bits=8,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=16,
)

MFMA_F32_16X16X16_BF16 = _make_mfma(
    'V_MFMA_F32_16X16X16_BF16',
    16,
    16,
    16,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)

MFMA_F32_32X32X8_BF16 = _make_mfma(
    'V_MFMA_F32_32X32X8_BF16',
    32,
    32,
    8,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=16,
)

MFMA_F64_16X16X4_F64 = _make_mfma(
    'V_MFMA_F64_16X16X4_F64',
    16,
    16,
    4,
    src_bits=64,
    dst_bits=64,
    src_vgprs=2,
    dst_vgprs=4,
)

# =========================================================================
# WMMA layouts (RDNA3+)
# =========================================================================

WMMA_F32_16X16X16_F16 = _make_wmma(
    'V_WMMA_F32_16X16X16_*',
    16,
    16,
    16,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)

WMMA_F32_16X16X32_F16 = _make_wmma(
    'V_WMMA_F32_16X16X32_*',
    16,
    16,
    32,
    src_bits=16,
    dst_bits=32,
    src_vgprs=4,
    dst_vgprs=4,
)

WMMA_I32_16X16X32_I8 = _make_wmma(
    'V_WMMA_I32_16X16X32_*',
    16,
    16,
    32,
    src_bits=8,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)

WMMA_F32_16X16X16_BF16 = _make_wmma(
    'V_WMMA_F32_16X16X16_BF16',
    16,
    16,
    16,
    src_bits=16,
    dst_bits=32,
    src_vgprs=2,
    dst_vgprs=4,
)


# =========================================================================
# Layout catalog
# =========================================================================

LAYOUT_CATALOG: dict[str, MatrixLayoutDescriptor] = {
    desc.mnemonic_pattern: desc
    for desc in [
        MFMA_F32_16X16X16_F16,
        MFMA_F32_32X32X8_F16,
        MFMA_F32_16X16X32_F16,
        MFMA_F32_32X32X16_F16,
        MFMA_I32_16X16X32_I8,
        MFMA_I32_32X32X16_I8,
        MFMA_F32_16X16X16_BF16,
        MFMA_F32_32X32X8_BF16,
        MFMA_F64_16X16X4_F64,
        WMMA_F32_16X16X16_F16,
        WMMA_F32_16X16X32_F16,
        WMMA_I32_16X16X32_I8,
        WMMA_F32_16X16X16_BF16,
    ]
}


def compute_xor_mask(
    guest: MatrixLayoutDescriptor,
    host: MatrixLayoutDescriptor,
) -> tuple[int, int, int] | None:
    """Compute the uniform XOR byte mask for layout conversion.

    Args:
        guest: Source ISA layout (e.g., CDNA4 MFMA).
        host: Target ISA layout (e.g., RDNA4 WMMA).

    Returns:
        ``(xor_byte_mask, range_start_lane, range_end_lane)`` or None if
        layouts are identical (no permutation needed).

    Raises:
        AssertionError: If the XOR is non-uniform across row groups.
    """
    assert guest.m == host.m and guest.n == host.n
    assert guest.wave_size == host.wave_size

    xor_lane = 0
    range_start = guest.wave_size
    range_end = 0

    for g_base, h_base in zip(guest.row_group_lanes, host.row_group_lanes):
        if g_base != h_base:
            xor_val = g_base ^ h_base
            if xor_lane == 0:
                xor_lane = xor_val
            assert xor_val == xor_lane, (
                f'Non-uniform XOR: {xor_val} vs {xor_lane} '
                f'(guest={guest.mnemonic_pattern}, host={host.mnemonic_pattern})'
            )
            range_start = min(range_start, min(g_base, h_base))
            range_end = max(range_end, max(g_base, h_base) + guest.n)

    if xor_lane == 0:
        return None

    bytes_per_lane = guest.dst_elem_bits // 8
    return (xor_lane * bytes_per_lane, range_start, range_end)


def find_descriptor(mnemonic: str) -> MatrixLayoutDescriptor | None:
    """Look up a layout descriptor by instruction mnemonic.

    Tries exact match first, then pattern matching with wildcards.
    """
    if mnemonic in LAYOUT_CATALOG:
        return LAYOUT_CATALOG[mnemonic]
    for pattern, desc in LAYOUT_CATALOG.items():
        if pattern.endswith('*') and mnemonic.startswith(pattern[:-1]):
            return desc
    return None


_ELEM_BITS: dict[str, int] = {
    'F32': 32,
    'I32': 32,
    'F64': 64,
    'F16': 16,
    'BF16': 16,
    'I16': 16,
    'U16': 16,
    'I8': 8,
    'U8': 8,
    'IU8': 8,
    'FP8': 8,
    'BF8': 8,
    'I4': 4,
    'U4': 4,
    'IU4': 4,
}

_MFMA_RE = re.compile(
    r'V_(?:S?MFMA[C]?)_' r'(F32|F64|I32|F16|BF16)_' r'(\d+)X(\d+)X(\d+)_' r'(\w+)$'
)

_WMMA_RE = re.compile(
    r'V_(?:S?WMMA[C]?)_'
    r'(F32|F16|BF16|I32|FP8|BF8)_'
    r'(\d+)X(\d+)X(\d+)_?'
    r'(\w+)?$'
)


def derive_layout_descriptor(mnemonic: str) -> MatrixLayoutDescriptor | None:
    """Auto-derive a MatrixLayoutDescriptor from an instruction mnemonic.

    Parses M, N, K, and element types from the mnemonic, then applies
    the standard MFMA or WMMA row-group formula. Returns None for
    unrecognized mnemonics.
    """
    m_mfma = _MFMA_RE.match(mnemonic)
    m_wmma = _WMMA_RE.match(mnemonic) if not m_mfma else None

    if m_mfma:
        dst_type, m_str, n_str, k_str, src_type = m_mfma.groups()
        kind = 'mfma'
    elif m_wmma:
        dst_type, m_str, n_str, k_str, src_type = m_wmma.groups()
        kind = 'wmma'
    else:
        return None

    m, n, k = int(m_str), int(n_str), int(k_str)

    dst_bits = _ELEM_BITS.get(dst_type, 32)
    src_base = (src_type or dst_type).split('_')[0]
    src_bits = _ELEM_BITS.get(src_base, 16)

    wave_size = 64
    dst_vgprs = m * n * (dst_bits // 32) // wave_size
    if dst_vgprs == 0:
        dst_vgprs = 1

    elems_per_dword = 32 // src_bits if src_bits <= 32 else 1
    src_elems_per_lane = k // (wave_size // (m if kind == 'mfma' else n))
    src_vgprs = max(1, (src_elems_per_lane + elems_per_dword - 1) // elems_per_dword)

    if kind == 'mfma':
        row_groups = _mfma_row_group_lanes(m, n, dst_vgprs)
    else:
        row_groups = _wmma_row_group_lanes(m, n, dst_vgprs)

    return MatrixLayoutDescriptor(
        mnemonic_pattern=mnemonic,
        kind=kind,
        m=m,
        n=n,
        k=k,
        wave_size=wave_size,
        src_elem_bits=src_bits,
        dst_elem_bits=dst_bits,
        src_vgprs=src_vgprs,
        dst_vgprs=dst_vgprs,
        row_group_lanes=row_groups,
    )
