# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for DBT encoding translator code generation helpers."""

from amdisa.encoding_translator_codegen import (
    EncodingTranslation,
    _classify_fields,
    _emit_encode_fn,
    _struct_name,
)
from amdisa.gpuisa import InstEncoding, MicrocodeField


def _field(name: str, bits: int = 1, offset: int = 0) -> MicrocodeField:
    return MicrocodeField(name=name, bit_cnt=bits, bit_offset=offset)


def _encoding(name: str, fields: list[str]) -> InstEncoding:
    return InstEncoding(
        name=name,
        order=0,
        bit_cnt=64,
        enc_field_bit_cnt=6,
        op_field_bit_cnt=8,
        ucode_fields=[_field(field) for field in fields],
        enc_conds=[],
    )


def test_gfx940_to_gfx11_vector_coherency_uses_remap() -> None:
    src = _encoding("ENC_MUBUF", ["encoding", "op", "sc0", "sc1", "nt"])
    dst = _encoding("ENC_MUBUF", ["encoding", "op", "glc", "dlc", "slc"])
    mappings = _classify_fields(src, dst, "ENC_MUBUF")
    assert not any(m.kind == "insert" and m.dst_name in {"glc", "dlc", "slc"} for m in mappings)

    trans = EncodingTranslation(
        src_enc_name="ENC_MUBUF",
        dst_enc_name="ENC_MUBUF",
        src_struct=_struct_name("ENC_MUBUF"),
        dst_struct=_struct_name("ENC_MUBUF"),
        src_bit_cnt=64,
        dst_bit_cnt=64,
        mappings=mappings,
        has_gfx11_coherency_remap=True,
    )
    body = "\n".join(_emit_encode_fn(trans, "dst", "rdna3"))

    assert "remap_gfx940_to_gfx11" in body
    assert "dst.glc = coh.glc;" in body
    assert "dst.dlc = coh.dlc;" in body
    assert "dst.slc = coh.slc;" in body
    assert "dst.glc = 0;" not in body


def test_gfx9_to_gfx11_smem_glc_uses_remap() -> None:
    src = _encoding("ENC_SMEM", ["encoding", "op", "glc"])
    dst = _encoding("ENC_SMEM", ["encoding", "op", "glc", "dlc"])
    mappings = _classify_fields(src, dst, "ENC_SMEM")
    assert not any(m.kind == "insert" and m.dst_name in {"glc", "dlc"} for m in mappings)

    trans = EncodingTranslation(
        src_enc_name="ENC_SMEM",
        dst_enc_name="ENC_SMEM",
        src_struct=_struct_name("ENC_SMEM"),
        dst_struct=_struct_name("ENC_SMEM"),
        src_bit_cnt=64,
        dst_bit_cnt=64,
        mappings=mappings,
        has_gfx11_glc_remap=True,
    )
    body = "\n".join(_emit_encode_fn(trans, "dst", "rdna3"))

    assert "remap_gfx9_to_gfx11" in body
    assert "dst.glc = coh.glc;" in body
    assert "dst.dlc = coh.dlc;" in body
    assert "dst.glc = 0;" not in body
