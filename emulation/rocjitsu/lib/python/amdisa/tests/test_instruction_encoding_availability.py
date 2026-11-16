# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from copy import deepcopy
import os
from pathlib import Path

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.gpuisa import Instruction
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    Cdna5Profile,
    CdnaProfile,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
)
from amdisa.parser import Parser
from amdisa.semantics import derive_semantics


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


@pytest.mark.parametrize(
    'arch,profile,encoding,name',
    [
        ('rdna1', Rdna1Profile(), 'ENC_VOP3', 'V_LSHLREV_B64'),
        ('rdna2', Rdna2Profile(), 'ENC_VOP3', 'V_LSHLREV_B64'),
        ('rdna3', Rdna3Profile(), 'ENC_MIMG', 'IMAGE_GATHER4H'),
        ('rdna3_5', Rdna3_5Profile(), 'ENC_MIMG', 'IMAGE_GATHER4H'),
    ],
)
def test_subdecoder_keeps_opcodes_across_primary_prefixes(
    arch, profile, encoding, name
):
    spec = Parser(str(_mrisa_dir() / f'amdgpu_isa_{arch}.xml'), profile).parse()
    inst = _find_instruction(spec, encoding, name)
    enc = spec.encoding_map[encoding]
    # The generator emits one named table for every primary prefix of an encoding.
    # Every such table must route the documented instruction, including opcodes
    # whose prefix was first encountered without another instruction beside it.
    for index in set(enc.primary_dt_ptrs) - {-1}:
        entry = spec.primary_decode_table[index]
        assert entry.sub_decode_funcs[inst.opcode] == f'decode{inst.fmt_name}'


@pytest.mark.parametrize(
    'arch,profile',
    [
        ('cdna1', Cdna1Profile()),
        ('cdna2', Cdna2Profile()),
        ('cdna3', CdnaProfile()),
        ('cdna4', Cdna4Profile()),
        ('cdna5', Cdna5Profile()),
        ('rdna1', Rdna1Profile()),
        ('rdna2', Rdna2Profile()),
        ('rdna3', Rdna3Profile()),
        ('rdna3_5', Rdna3_5Profile()),
        ('rdna4', Rdna4Profile()),
    ],
)
def test_dpp8_admission_keeps_staged_sources_within_one_dword(arch, profile):
    spec = Parser(str(_mrisa_dir() / f'amdgpu_isa_{arch}.xml'), profile).parse()
    generator = CodeGenerator(spec, '')
    admitted = 0
    for enc in spec.encoding_map.values():
        if profile.is_alt_encoding(enc.enc_name):
            continue
        _, dpp8_struct = generator._vop_dpp_struct_names(enc.enc_name)
        if dpp8_struct is None:
            continue
        for inst in enc.insts:
            if not generator._instruction_supports_dpp8(inst, enc.enc_name):
                continue
            admitted += 1
    if arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4'):
        assert admitted == 0
    else:
        assert admitted > 0


def test_dpp8_admission_rejects_wide_encoded_source():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    generator = CodeGenerator(spec, '')
    inst = deepcopy(_find_instruction(spec, 'ENC_VOP1', 'V_MOV_B32'))
    assert generator._instruction_supports_dpp8(inst, 'ENC_VOP1')
    generator._encoded_source_operands(inst)[0].size = 64

    with pytest.raises(
        AssertionError, match='DPP8 requires a source no wider than 32 bits'
    ):
        generator._instruction_supports_dpp8(inst, 'ENC_VOP1')


def test_dpp8_movrel_rejects_wide_semantic_source(monkeypatch):
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    generator = CodeGenerator(spec, '')
    inst = _find_instruction(spec, 'ENC_VOP1', 'V_MOVRELS_B32')
    assert generator._instruction_supports_dpp8(inst, 'ENC_VOP1')
    sem = derive_semantics(inst.name, inst.enc_name)
    sources, destinations = generator._execute_operand_roles(inst, sem)
    sources = deepcopy(sources)
    sources[0].size = 64
    # Keep encoded admission valid while changing the source MOVREL emits.
    monkeypatch.setattr(
        generator, '_execute_operand_roles', lambda *_: (sources, destinations)
    )

    with pytest.raises(
        AssertionError, match='DPP8 requires a source no wider than 32 bits'
    ):
        generator._gen_execute_body(inst, sem, 'ENC_VOP1')


@pytest.mark.parametrize(
    'name',
    [
        'V_DOT2_F32_F16',
        'V_DOT2_F32_BF16',
        'V_DOT4_F32_FP8_FP8',
        'V_DOT4_F32_FP8_BF8',
        'V_DOT4_F32_BF8_FP8',
        'V_DOT4_F32_BF8_BF8',
    ],
)
def test_rdna4_floating_dot_dpp_availability(name):
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    inst = _find_instruction(spec, 'ENC_VOP3P', name)
    generator = CodeGenerator(spec, '')
    if name.startswith('V_DOT4_'):
        assert 'VOP3P_VOP_DPP16' not in inst.available_encodings
        assert 'VOP3P_VOP_DPP8' not in inst.available_encodings
    assert generator._instruction_supports_dpp(inst, 'ENC_VOP3P')
    assert generator._instruction_supports_dpp8(inst, 'ENC_VOP3P')


@pytest.mark.parametrize(
    'name',
    [
        'V_DOT4_I32_IU8',
        'V_DOT4_U32_U8',
        'V_DOT8_U32_U4',
        'V_PK_ADD_F16',
        'V_PK_ADD_U16',
    ],
)
def test_rdna4_unsupported_packed_dpp_remains_forbidden(name):
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    inst = _find_instruction(spec, 'ENC_VOP3P', name)
    generator = CodeGenerator(spec, '')
    assert not generator._instruction_supports_dpp(inst, 'ENC_VOP3P')
    assert not generator._instruction_supports_dpp8(inst, 'ENC_VOP3P')
