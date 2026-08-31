# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.gpuisa import Instruction
from amdisa.isa_profile import Cdna4Profile, CdnaProfile, Rdna4Profile
from amdisa.parser import Parser


def _mrisa_dir() -> Path:
    default = (
        Path(__file__).resolve().parents[6] / 'shared' / 'machine-readable-isa' / 'isa'
    )
    return Path(os.environ.get('MRISA_PATH', default))


def _find_instruction(spec, encoding_name: str, instruction_name: str):
    encoding = spec.encoding_map[encoding_name]
    return next(inst for inst in encoding.insts if inst.name == instruction_name)


def test_cdna4_manual_opcode_rule_filters_xml_dpp_availability():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), Cdna4Profile()).parse()
    inst = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F64_I32')

    # The XML exposes the conditional extension, but CDNA4 section 12.16.1
    # explicitly prohibits DPP for this opcode.
    assert 'VOP1_VOP_DPP' in inst.available_encodings
    assert not CodeGenerator(spec, '')._instruction_supports_dpp(inst, 'ENC_VOP1')


def test_rdna4_distinguishes_vop1_instructions_with_and_without_dpp():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    wide = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F64_I32')
    narrow = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F32_I32')

    assert 'VOP1_VOP_DPP16' not in wide.available_encodings
    assert 'VOP1_VOP_DPP8' not in wide.available_encodings
    assert 'VOP1_VOP_DPP16' in narrow.available_encodings
    assert 'VOP1_VOP_DPP8' in narrow.available_encodings

    generator = CodeGenerator(spec, '')
    assert not generator._instruction_supports_dpp(wide, 'ENC_VOP1')
    assert not generator._instruction_supports_dpp8(wide, 'ENC_VOP1')
    assert generator._instruction_supports_dpp(narrow, 'ENC_VOP1')
    assert generator._instruction_supports_dpp8(narrow, 'ENC_VOP1')


def test_rdna4_dpp_predicate_filters_opcodes_without_alternates():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    encoding = spec.encoding_map['ENC_VOP3']
    generator = CodeGenerator(spec, '')

    opcodes = generator._encoded_dpp_opcodes(encoding, 'dpp8')
    helper = generator._opcode_predicate_helper_impl(
        encoding,
        'has_encoded_dpp8',
        list(opcodes),
        'amdgpu::dpp::is_src_dpp8(inst_.src0)',
    )

    assert 384 not in opcodes  # V_NOP
    assert 385 in opcodes  # V_MOV_B32
    assert 'amdgpu::dpp::is_src_dpp8(inst_.src0)' in helper
    assert 'switch (inst_.op)' not in helper
    assert 'inst_.op >=' in helper


def test_opcode_set_condition_omits_unsigned_domain_boundaries():
    assert CodeGenerator._opcode_set_condition([0, 1, 5]) == (
        'inst_.op <= 1 || inst_.op == 5'
    )
    assert CodeGenerator._opcode_set_condition(list(range(248, 256)), 255) == (
        'inst_.op >= 248'
    )
    assert CodeGenerator._opcode_set_condition(list(range(4)), 3) == 'true'


def test_cdna4_distinguishes_vop1_instructions_with_and_without_sdwa():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), Cdna4Profile()).parse()
    supported = _find_instruction(spec, 'ENC_VOP1', 'V_MOV_B32')
    unsupported = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F64_I32')

    assert 'VOP1_VOP_SDWA' in supported.available_encodings
    assert 'VOP1_VOP_SDWA' not in unsupported.available_encodings

    generator = CodeGenerator(spec, '')
    assert generator._instruction_supports_sdwa(supported, 'ENC_VOP1')
    assert not generator._instruction_supports_sdwa(unsupported, 'ENC_VOP1')


def test_cdna_profile_restores_manual_sdwa_opcode_missing_from_xml():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna3.xml'), CdnaProfile()).parse()
    inst = _find_instruction(spec, 'ENC_VOP2', 'V_PK_FMAC_F16')

    assert 'VOP2_VOP_SDWA' not in inst.available_encodings
    assert CodeGenerator(spec, '')._instruction_supports_sdwa(inst, 'ENC_VOP2')


def test_modifier_availability_requires_explicit_encoding_provenance():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), Cdna4Profile()).parse()
    generator = CodeGenerator(spec, '')
    unknown = Instruction('V_SYNTHETIC', 'ENC_VOP1', 0, [])
    known_absent = Instruction(
        'V_SYNTHETIC',
        'ENC_VOP1',
        0,
        [],
        available_encodings=frozenset(),
    )

    with pytest.raises(ValueError, match='V_SYNTHETIC.*encoding provenance'):
        generator._instruction_supports_dpp(unknown, 'ENC_VOP1')
    assert not generator._instruction_supports_dpp(known_absent, 'ENC_VOP1')
