# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build valid instruction encodings with specified operand registers.

Extends the sample encoding logic from ``_generator.py`` to allow setting
arbitrary operand register indices and modifier fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec


def _set_field(
    word0: int, word1: int, field_name: str, value: int, enc: InstEncoding
) -> tuple[int, int]:
    """Set a microcode field in a 64-bit encoding word pair."""
    for f in enc.ucode_fields:
        if f.name == field_name:
            mask = (1 << f.bit_cnt) - 1
            value &= mask
            offset = f.bit_offset
            # Clear the field
            clear_mask = ~(mask << offset)
            combined = ((word1 << 32) | word0) & (clear_mask & ((1 << 64) - 1))
            # Set the field
            combined |= value << offset
            return combined & 0xFFFF_FFFF, (combined >> 32) & 0xFFFF_FFFF
    return word0, word1


def build_encoding(
    isa_spec: IsaSpec,
    enc: InstEncoding,
    inst: Instruction,
    operand_regs: dict[str, int] | None = None,
    modifiers: dict[str, int] | None = None,
) -> tuple[int, int] | None:
    """Build a valid encoding for an instruction with specific operand registers.

    Args:
        isa_spec: The ISA specification.
        enc: The encoding format containing this instruction.
        inst: The instruction to encode.
        operand_regs: Optional dict mapping field names (e.g., 'ssrc0', 'sdst')
                      to register indices. Unspecified fields default to 0.
        modifiers: Optional dict mapping modifier field names ('neg', 'abs',
                   'omod', 'clamp') to their values.

    Returns:
        (word0, word1) or None if the instruction can't be encoded.
    """
    op_field = next((f for f in enc.ucode_fields if f.name == 'op'), None)
    has_encoding_field = any(f.name == 'encoding' for f in enc.ucode_fields)
    ptrs = enc.primary_dt_ptrs
    if not op_field or not has_encoding_field or not ptrs:
        return None
    if inst.opcode >= len(ptrs) or ptrs[inst.opcode] == -1:
        return None

    enc_val = ptrs[inst.opcode]
    word = (enc_val << (32 - isa_spec.profile.max_enc_bits)) | (
        inst.opcode << op_field.bit_offset
    )
    w0 = word & 0xFFFF_FFFF
    w1 = (word >> 32) & 0xFFFF_FFFF

    if operand_regs:
        for field_name, reg_idx in operand_regs.items():
            w0, w1 = _set_field(w0, w1, field_name, reg_idx, enc)

    if modifiers:
        for field_name, value in modifiers.items():
            w0, w1 = _set_field(w0, w1, field_name, value, enc)

    return w0, w1


# Convenience constants for common SGPR register assignments.
# CTS uses a fixed register convention to avoid aliasing:
#   s0 = dst, s2 = src0, s4 = src1 (or s[2:3] for 64-bit)
CTS_SCALAR_REGS_UNARY = {'sdst': 0, 'ssrc0': 2}
CTS_SCALAR_REGS_BINARY = {'sdst': 0, 'ssrc0': 2, 'ssrc1': 4}

# VOP1: src0 encodes VGPR N as 256+N.  vdst is a raw VGPR index.
#   v2 = dst, v0 = src0
CTS_VOP1_REGS = {'vdst': 2, 'src0': 256 + 0}

# VOP2: src0 encodes VGPR N as 256+N.  vsrc1 and vdst are raw VGPR indices.
#   v4 = dst, v0 = src0, v2 = src1
CTS_VOP2_REGS = {'vdst': 4, 'src0': 256 + 0, 'vsrc1': 2}

# SOPC: no destination — result is SCC only.
#   s0 = src0, s2 = src1
CTS_SOPC_REGS = {'ssrc0': 0, 'ssrc1': 2}

# VOPC: no destination — result is VCC only.
#   v0 = src0, v2 = src1
CTS_VOPC_REGS = {'src0': 256 + 0, 'vsrc1': 2}

# VOP3 ternary: three 9-bit source operands + vdst.
#   v6 = dst, v0 = src0, v2 = src1, v4 = src2
CTS_VOP3_TERNARY_REGS = {
    'vdst': 6,
    'src0': 256 + 0,
    'src1': 256 + 2,
    'src2': 256 + 4,
}

# VOP3 unary (promoted VOP1): one 9-bit source + vdst.
#   v4 = dst, v0 = src0
CTS_VOP3_UNARY_REGS = {
    'vdst': 4,
    'src0': 256 + 0,
}

# VOP3 binary (promoted VOP2): two 9-bit sources + vdst.
#   v4 = dst, v0 = src0, v2 = src1
CTS_VOP3_BINARY_REGS = {
    'vdst': 4,
    'src0': 256 + 0,
    'src1': 256 + 2,
}

# VOP3P: three 9-bit sources + vdst (dot products, packed ops).
#   v6 = dst, v0 = src0, v2 = src1, v4 = src2
CTS_VOP3P_REGS = {
    'vdst': 6,
    'src0': 256 + 0,
    'src1': 256 + 2,
    'src2': 256 + 4,
}

# VOP3/VOP3_SDST_ENC for mad_u64_u32/mad_i64_i32: 64-bit dst + 64-bit src2.
#   v8:v9 = dst, v0 = src0, v2 = src1, v4:v5 = src2
CTS_VOP3_MAD64_REGS = {
    'vdst': 8,
    'src0': 256 + 0,
    'src1': 256 + 2,
    'src2': 256 + 4,
}

# DS (LDS): raw VGPR indices (not 256+N).
#   v6 = dst, v0 = addr, v2 = data0, v4 = data1
CTS_DS_REGS = {
    'vdst': 6,
    'addr': 0,
    'data0': 2,
    'data1': 4,
}

# MFMA (VOP3P_MFMA): src0/src1 are 9-bit (256+N for VGPR N), vdst is 8-bit.
#   v0 = src0 base, v16 = src1 base, v32 = dst/acc base
CTS_MFMA_REGS = {
    'vdst': 32,
    'src0': 256 + 0,
    'src1': 256 + 16,
    'src2': 256 + 32,
}

# WMMA (ENC_VOP3P): same encoding as VOP3P.
#   v0 = src0 base, v16 = src1 base, v32 = dst/acc base
CTS_WMMA_REGS = {
    'vdst': 32,
    'src0': 256 + 0,
    'src1': 256 + 16,
    'src2': 256 + 32,
}
