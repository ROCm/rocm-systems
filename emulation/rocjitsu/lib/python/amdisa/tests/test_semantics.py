# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for mnemonic-derived instruction semantics."""

from amdisa.legalization import canonical_mnemonic
from amdisa.semantics import derive_semantics


def test_mtbuf_d16_format_order_has_same_semantics() -> None:
    cdna = derive_semantics("tbuffer_load_format_d16_xy", "ENC_MTBUF")
    rdna = derive_semantics("tbuffer_load_d16_format_xy", "ENC_MTBUF")

    assert cdna is not None
    assert rdna is not None
    assert cdna.semantic_class == rdna.semantic_class == "tbuffer_load"
    assert cdna.elem_size == rdna.elem_size == 2
    assert cdna.num_elems == rdna.num_elems == 2
    assert cdna.d16_lo is True
    assert rdna.d16_lo is True


def test_mtbuf_d16_format_rename_is_canonicalized() -> None:
    assert (
        canonical_mnemonic("TBUFFER_STORE_FORMAT_D16_XYZW")
        == "TBUFFER_STORE_D16_FORMAT_XYZW"
    )
