# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import pytest


def _isa_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'shared/machine-readable-isa/isa'
        if candidate.is_dir():
            return candidate
    raise AssertionError('cannot locate shared machine-readable ISA directory')


@lru_cache(maxsize=1)
def _verifier():
    path = _isa_dir() / 'verify_amdgpu_isa_cdna5_gfx1251_delta.py'
    spec = importlib.util.spec_from_file_location('gfx1251_delta_provenance', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    isa_dir = _isa_dir()
    for name in (
        'amdgpu_isa_cdna5.xml',
        'amdgpu_isa_cdna5_gfx1251_delta.xml',
        'amdgpu_isa_cdna5_gfx1251_provenance.json',
    ):
        (tmp_path / name).write_bytes((isa_dir / name).read_bytes())
    return tmp_path / 'amdgpu_isa_cdna5_gfx1251_provenance.json', tmp_path


def _mutate_delta(tmp_path: Path, transform) -> tuple[Path, Path]:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    contents = delta_path.read_text(encoding='utf-8')
    delta_path.write_text(transform(contents), encoding='utf-8')
    return manifest_path, isa_dir


def test_production_delta_matches_public_provenance_manifest() -> None:
    isa_dir = _isa_dir()
    manifest_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_provenance.json'
    _verifier().validate(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['llvm_commit'] == '3bcd9a803184e2d3657b9d5cc2a1773e9ce0f116'
    assert len(manifest['instructions']) == 9
    assert all(
        {'tablegen_definition', 'tablegen_profile', 'tablegen_opcode', 'primary_mc'}
        <= record.keys()
        for record in manifest['instructions']
    )


def test_provenance_rejects_unrecorded_alias(tmp_path: Path) -> None:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    tree = ET.parse(delta_path)
    instruction = tree.getroot().find('./InstructionAdditions/Instruction')
    assert instruction is not None
    aliases = ET.Element('AliasedInstructionNames')
    ET.SubElement(aliases, 'InstructionName').text = 'UNRECORDED_ALIAS'
    instruction.insert(2, aliases)
    tree.write(delta_path, encoding='utf-8')

    with pytest.raises(
        _verifier().ProvenanceError, match='aliases are not provenance-approved'
    ):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_identifier_not_derived_from_mc_bytes(
    tmp_path: Path,
) -> None:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    tree = ET.parse(delta_path)
    identifier = tree.getroot().find(
        './EncodingIdentifierAdditions/EncodingIdentifierAddition/EncodingIdentifier'
    )
    assert identifier is not None and identifier.text is not None
    identifier.text = identifier.text[:-1] + (
        '1' if identifier.text[-1] == '0' else '0'
    )
    tree.write(delta_path, encoding='utf-8')

    with pytest.raises(
        _verifier().ProvenanceError, match='masked by the public base XML'
    ):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


@pytest.mark.parametrize(
    'container_name', ['InstructionAdditions', 'EncodingIdentifierAdditions']
)
def test_provenance_rejects_unrecorded_container_attributes(
    tmp_path: Path, container_name: str
) -> None:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    tree = ET.parse(delta_path)
    container = tree.getroot().find(container_name)
    assert container is not None
    container.set('Replace', 'true')
    tree.write(delta_path, encoding='utf-8')

    with pytest.raises(
        _verifier().ProvenanceError, match=f'{container_name} attributes'
    ):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_unrecorded_description(tmp_path: Path) -> None:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    tree = ET.parse(delta_path)
    description = tree.getroot().find('./InstructionAdditions/Instruction/Description')
    assert description is not None
    description.text = 'Unrecorded description'
    tree.write(delta_path, encoding='utf-8')

    with pytest.raises(
        _verifier().ProvenanceError, match='repository-authored provenance template'
    ):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_unrecorded_leaf_attribute(tmp_path: Path) -> None:
    manifest_path, isa_dir = _copy_inputs(tmp_path)
    delta_path = isa_dir / 'amdgpu_isa_cdna5_gfx1251_delta.xml'
    tree = ET.parse(delta_path)
    field = tree.getroot().find(
        './InstructionAdditions/Instruction/InstructionEncodings/InstructionEncoding/'
        'Operands/Operand/FieldName'
    )
    assert field is not None
    field.set('Source', 'unrecorded')
    tree.write(delta_path, encoding='utf-8')

    with pytest.raises(_verifier().ProvenanceError, match='field attributes'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_comment_before_root(tmp_path: Path) -> None:
    manifest_path, isa_dir = _mutate_delta(
        tmp_path, lambda contents: '<!-- unrecorded -->\n' + contents
    )

    with pytest.raises(_verifier().ProvenanceError, match='XML comment'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_comment_inside_root(tmp_path: Path) -> None:
    def add_comment(contents: str) -> str:
        root_offset = contents.index('<IsaAdditions')
        opening_end = contents.index('>', root_offset) + 1
        return (
            contents[:opening_end] + '\n  <!-- unrecorded -->' + contents[opening_end:]
        )

    manifest_path, isa_dir = _mutate_delta(tmp_path, add_comment)

    with pytest.raises(_verifier().ProvenanceError, match='XML comment'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_processing_instruction(tmp_path: Path) -> None:
    manifest_path, isa_dir = _mutate_delta(
        tmp_path, lambda contents: '<?unrecorded data?>\n' + contents
    )

    with pytest.raises(_verifier().ProvenanceError, match='processing instruction'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_non_whitespace_tail(tmp_path: Path) -> None:
    manifest_path, isa_dir = _mutate_delta(
        tmp_path,
        lambda contents: contents.replace(
            '</EncodingIdentifierAddition>',
            '</EncodingIdentifierAddition>unrecorded',
            1,
        ),
    )

    with pytest.raises(_verifier().ProvenanceError, match='non-whitespace tail'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


def test_provenance_rejects_mixed_container_text(tmp_path: Path) -> None:
    manifest_path, isa_dir = _mutate_delta(
        tmp_path,
        lambda contents: contents.replace(
            '<EncodingIdentifierAdditions>',
            '<EncodingIdentifierAdditions>unrecorded',
            1,
        ),
    )

    with pytest.raises(_verifier().ProvenanceError, match='mixed container text'):
        _verifier().validate(manifest_path, isa_dir=isa_dir)


@pytest.mark.parametrize(
    ('field', 'line_number'),
    [
        ('assembly_line', 0),
        ('assembly_line', 3),
        ('check_line', 0),
        ('check_line', 3),
    ],
)
def test_mc_provenance_rejects_out_of_range_line_numbers(
    tmp_path: Path, field: str, line_number: int
) -> None:
    source = tmp_path / 'mc.s'
    source.write_text(
        'v_test_instruction\n// GFX1251: encoding: [0x01]\n', encoding='utf-8'
    )
    record = {
        'path': source.name,
        'assembly_line': 1,
        'check_line': 2,
        'assembly': 'v_test_instruction',
        'test': 'GFX1251',
        'bytes': ['0x01'],
    }
    record[field] = line_number

    line_kind = 'assembly' if field == 'assembly_line' else 'check'
    with pytest.raises(
        _verifier().ProvenanceError,
        match=rf'{line_kind} line {line_number} is out of range',
    ):
        _verifier()._verify_mc_record(tmp_path, record, 'test MC vector')
