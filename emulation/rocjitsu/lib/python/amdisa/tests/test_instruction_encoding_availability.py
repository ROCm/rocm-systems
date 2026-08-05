# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.gpuisa import Instruction
from amdisa.isa_profile import CdnaProfile, Rdna4Profile
from amdisa.parser import Parser


def _mrisa_dir() -> Path:
    return (
        Path(__file__).resolve().parents[6] / 'shared' / 'machine-readable-isa' / 'isa'
    )


def _find_instruction(spec, encoding_name: str, instruction_name: str):
    encoding = spec.encoding_map[encoding_name]
    return next(inst for inst in encoding.insts if inst.name == instruction_name)


def test_cdna4_preserves_conditional_dpp_encoding_availability():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), CdnaProfile()).parse()
    inst = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F64_I32')

    assert 'VOP1_VOP_DPP' in inst.available_encodings
    assert CodeGenerator(spec, '')._instruction_supports_dpp(inst, 'ENC_VOP1')


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


def test_cdna4_distinguishes_vop1_instructions_with_and_without_sdwa():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), CdnaProfile()).parse()
    supported = _find_instruction(spec, 'ENC_VOP1', 'V_MOV_B32')
    unsupported = _find_instruction(spec, 'ENC_VOP1', 'V_CVT_F64_I32')

    assert 'VOP1_VOP_SDWA' in supported.available_encodings
    assert 'VOP1_VOP_SDWA' not in unsupported.available_encodings

    generator = CodeGenerator(spec, '')
    assert generator._instruction_supports_sdwa(supported, 'ENC_VOP1')
    assert not generator._instruction_supports_sdwa(unsupported, 'ENC_VOP1')


def test_modifier_availability_requires_explicit_encoding_provenance():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_cdna4.xml'), CdnaProfile()).parse()
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
