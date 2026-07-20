# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for generated cross-ISA target encoding."""

from amdisa.encoding_translator_codegen import (
    EncodingTranslation,
    FieldMapping,
    _emit_encode_fn,
)


def _sop1_translation(*, has_builder: bool) -> EncodingTranslation:
    """Return a minimal SOP1 recipe for encoder-emission tests."""
    return EncodingTranslation(
        src_enc_name='ENC_SOP1',
        dst_enc_name='ENC_SOP1',
        src_struct='Sop1MachineInst',
        dst_struct='Sop1MachineInst',
        src_bit_cnt=32,
        dst_bit_cnt=32,
        mappings=[FieldMapping('copy', 'ssrc0', 'ssrc0', 8, 8)],
        dst_enc_field_val=0x17D,
        dst_has_builder=has_builder,
    )


def test_encode_function_uses_generated_builder_for_concrete_target_format():
    output = '\n'.join(
        _emit_encode_fn(_sop1_translation(has_builder=True), 'rocjitsu::dst', 'dst')
    )

    assert 'rocjitsu::dst::Sop1BuilderFields dst{};' in output
    assert 'rocjitsu::dst::build_sop1(dst_op, dst)' in output
    assert 'dst.encoding' not in output
    assert 'std::bit_cast<uint32_t>(dst)' not in output


def test_encode_function_keeps_machine_inst_for_decoder_only_target_format():
    output = '\n'.join(
        _emit_encode_fn(_sop1_translation(has_builder=False), 'rocjitsu::dst', 'dst')
    )

    assert 'rocjitsu::dst::Sop1MachineInst dst{};' in output
    assert 'dst.encoding = 0x17D;' in output
    assert 'std::bit_cast<uint32_t>(dst)' in output
    assert 'build_sop1' not in output
