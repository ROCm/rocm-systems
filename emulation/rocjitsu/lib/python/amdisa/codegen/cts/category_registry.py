# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Maps semantic_class values to CTS test categories."""

from __future__ import annotations

SCALAR_ALU_CLASSES = frozenset(
    {
        'scalar_unary',
        'scalar_binop',
        'scalar_mov',
        'scalar_cselect',
        'scalar_saveexec',
        'scalar_bfe',
    }
)

SCALAR_CMP_CLASSES = frozenset(
    {
        'scalar_cmp',
        'scalar_cmpk',
        'scalar_bitcmp',
    }
)

VECTOR_ALU_UNARY_CLASSES = frozenset(
    {
        'vector_unary',
        'vector_mov',
        'vector_cvt',
    }
)

VECTOR_ALU_BINARY_CLASSES = frozenset(
    {
        'vector_binop',
        'vector_cndmask',
        'vector_fmamk',
        'vector_fmaak',
    }
)

VECTOR_ALU_TERNARY_CLASSES = frozenset(
    {
        'vector_ternary',
        'vector_add_co',
    }
)

VECTOR_CMP_CLASSES = frozenset(
    {
        'vector_cmp',
        'vector_cmpx',
        'vector_cmp_class',
    }
)

VECTOR_SPECIAL_CLASSES = frozenset(
    {
        'vector_div_fixup',
        'vector_div_scale',
        'vector_div_fmas',
        'vector_mad_64_32',
        'vector_dot',
        'vector_permlane16',
        'vector_permlanex16',
        'vector_permlane64',
        'vector_permlane16_swap',
        'vector_permlane32_swap',
        'vector_readlane',
        'vector_writelane',
        'vector_bitop3',
        'vector_cvt_pk',
    }
)

DOT_PRODUCT_CLASSES = frozenset(
    {
        'dot4_i32_i8',
        'dot4_u32_u8',
        'dot8_i32_i4',
        'dot8_u32_u4',
        'dot2_i32_i16',
        'dot2_u32_u16',
    }
)

PACKED_CLASSES = frozenset(
    {
        'pk_binop',
        'pk_ternary',
        'mad_mix',
        'pk_dot',
    }
)

MATRIX_CLASSES = frozenset(
    {
        'mfma',
        'accvgpr_read',
        'accvgpr_write',
    }
)

CTS_CATEGORY_MAP: dict[str, str] = {}
for _cls in DOT_PRODUCT_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'dot_product'
for _cls in SCALAR_ALU_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'scalar_alu'
for _cls in SCALAR_CMP_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'scalar_cmp'
for _cls in VECTOR_ALU_UNARY_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'vector_alu_unary'
for _cls in VECTOR_ALU_BINARY_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'vector_alu_binary'
for _cls in VECTOR_ALU_TERNARY_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'vector_alu_ternary'
for _cls in VECTOR_CMP_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'vector_cmp'
for _cls in VECTOR_SPECIAL_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'vector_special'
for _cls in PACKED_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'packed'
for _cls in MATRIX_CLASSES:
    CTS_CATEGORY_MAP[_cls] = 'matrix'


def cts_category(semantic_class: str) -> str | None:
    """Return the CTS category for a semantic class, or None if not testable."""
    return CTS_CATEGORY_MAP.get(semantic_class)
