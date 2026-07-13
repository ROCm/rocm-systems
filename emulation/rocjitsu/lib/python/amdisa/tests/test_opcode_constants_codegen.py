# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for generated AMDGPU opcode and encoding constants."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import InstEncoding, Instruction


def _encoding(
    name: str,
    opcode_bits: int,
    decode_values: list[int],
) -> InstEncoding:
    """Build the smallest encoding needed by the constant generators."""
    enc = InstEncoding(name, 0, 32, 0, opcode_bits, [], [])
    enc.primary_dt_ptrs = decode_values
    return enc


def _generator(tmp_path: Path, encodings: list[InstEncoding]) -> CodeGenerator:
    """Build a generator backed by a compact synthetic ISA specification."""
    max_decode_value = max(
        (value for enc in encodings for value in (enc.primary_dt_ptrs or [])),
        default=0,
    )
    spec = SimpleNamespace(
        arch_name='testisa',
        inst_encodings=encodings,
        primary_decode_table=[None] * (max_decode_value + 2),
        encoding_map={enc.enc_name: enc for enc in encodings},
        profile=SimpleNamespace(derive_parent_enc_name=lambda name: name),
    )
    return CodeGenerator(spec, str(tmp_path))


def test_opcode_name_fragments_preserve_packed_isa_abbreviations():
    assert CodeGenerator._opcode_name_fragment('getpc') == 'GetPc'
    assert CodeGenerator._opcode_name_fragment('saveexec') == 'SaveExec'
    assert CodeGenerator._opcode_const_base_name('s_getpc_b64') == 'kSGetPcB64'
    assert CodeGenerator._encoding_group_name('kVop1', 0, False) == 'kVop1'
    assert CodeGenerator._encoding_group_name('kVop1', 1, False) == 'kVop1Hi1'
    assert CodeGenerator._encoding_group_name('kVop3', 1, True) == 'kVop3OpHi1'


def test_opcode_header_qualifies_ambiguous_mnemonics_and_masks_raw_opcodes(tmp_path):
    vop1 = _encoding('ENC_VOP1', 8, [10])
    vop3 = _encoding('ENC_VOP3', 10, [20])
    sop1 = _encoding('ENC_SOP1', 8, [30])
    vop1.insts = [Instruction('v_mov_b32', vop1.enc_name, 0x101, [])]
    vop3.insts = [Instruction('v_mov_b32', vop3.enc_name, 0x141, [])]
    sop1.insts = [Instruction('s_getpc_b64', sop1.enc_name, 0x11C, [])]

    generator = _generator(tmp_path, [vop1, vop3, sop1])
    generator.gen_opcode_constants()
    output = (tmp_path / 'testisa' / 'opcodes.h').read_text()

    assert 'inline constexpr uint16_t kVMovB32Vop1 = 1;' in output
    assert 'inline constexpr uint16_t kVMovB32Vop3 = 321;' in output
    assert 'inline constexpr uint16_t kVMovB32 =' not in output
    assert 'inline constexpr uint16_t kSGetPcB64Sop1 = 28;' in output
    assert 'inline constexpr uint16_t kSGetPcB64 = 28;' in output


def test_opcode_header_rejects_qualified_name_collisions(tmp_path):
    first = _encoding('ENC_SOP1', 8, [10])
    second = _encoding('ENC_SOP1', 8, [11])
    first.insts = [Instruction('s_mov_b32', first.enc_name, 0, [])]
    second.insts = [Instruction('s_mov_b32', second.enc_name, 1, [])]

    with pytest.raises(ValueError, match='opcode constant collision'):
        _generator(tmp_path, [first, second]).gen_opcode_constants()


def test_encoding_constants_expand_decode_ranges_without_redundant_op_hi_zero(tmp_path):
    enc = _encoding('ENC_VOP3', 10, [100, 102])
    enc.insts = [Instruction('v_add_f32', enc.enc_name, 1, [])]
    generator = _generator(tmp_path, [enc])
    generator.isa_spec.primary_decode_table[100] = SimpleNamespace(num_dupe_entries=2)

    output = str(generator._encoding_constants_block())

    assert 'inline constexpr uint16_t kVop3 = 100;' in output
    assert 'kVop3OpHi0' not in output
    assert 'inline constexpr uint16_t kVop3OpHi1 = 101;' in output
    assert 'inline constexpr uint16_t kVop3OpHi2 = 102;' in output
    assert 'not necessarily the narrower MachineInst::encoding bitfield' in output


def test_encoding_constants_reject_base_name_collisions(tmp_path):
    first = _encoding('ENC_DS', 8, [100])
    second = _encoding('ENC_DS', 8, [101])
    first.insts = [Instruction('ds_read_b32', first.enc_name, 1, [])]
    second.insts = [Instruction('ds_write_b32', second.enc_name, 2, [])]

    with pytest.raises(ValueError, match='encoding constant collision'):
        _generator(tmp_path, [first, second])._encoding_constants_block()


def test_checked_in_rdna4_headers_pin_representative_generated_constants():
    project_root = Path(__file__).resolve().parents[4]
    isa_dir = project_root / 'lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4'

    opcodes = (isa_dir / 'opcodes.h').read_text()
    encodings = (isa_dir / 'encodings.h').read_text()

    assert 'inline constexpr uint16_t kSMovB32Sop1 = 0;' in opcodes
    assert 'inline constexpr uint16_t kVop3 = 424;' in encodings
    assert 'inline constexpr uint16_t kVop3OpHi1 = 425;' in encodings
    assert 'kVop3OpHi0' not in encodings
