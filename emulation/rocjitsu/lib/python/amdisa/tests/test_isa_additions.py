# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from copy import deepcopy
import re
import subprocess
import sys
import xml.etree.ElementTree as elem_tree
from pathlib import Path

import pytest

from amdisa.__main__ import _group_isa_additions_args
from amdisa.isa_additions import (
    ADDITION_SOURCE_ATTR,
    IsaAdditionError,
    apply_isa_additions,
)
from amdisa.isa_profile import Cdna1Profile, Cdna5Profile, Rdna4Profile
from amdisa.parser import Parser

_BASE_XML = '''\
<Spec>
  <Document><SchemaVersion>1.2.0</SchemaVersion></Document>
  <ISA>
    <Architecture>
      <ArchitectureName>AMD TEST 1</ArchitectureName>
      <ArchitectureId>1</ArchitectureId>
    </Architecture>
    <Encodings>
      <Encoding Order="0">
        <EncodingName>ENC_TEST</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask Radix="2">11110000111000000000000000000000</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier Radix="2">10110000000000000000000000000000</EncodingIdentifier>
          <EncodingIdentifier Radix="2">10100000001000000000000000000000</EncodingIdentifier>
          <EncodingIdentifier Radix="2">10100000010000000000000000000000</EncodingIdentifier>
          <EncodingIdentifier Radix="2">10100000010000000000000000000001</EncodingIdentifier>
        </EncodingIdentifiers>
        <EncodingConditions>
          <EncodingCondition>
            <ConditionName>default</ConditionName>
            <ConditionId>0</ConditionId>
          </EncodingCondition>
          <EncodingCondition>
            <ConditionName>alternate</ConditionName>
            <ConditionId>1</ConditionId>
          </EncodingCondition>
        </EncodingConditions>
        <MicrocodeFormat>
          <BitMap>
            <Field>
              <FieldName>ENCODING</FieldName>
              <BitLayout><Range Order="0"><BitCount>4</BitCount><BitOffset>28</BitOffset></Range></BitLayout>
            </Field>
            <Field>
              <FieldName>OP</FieldName>
              <BitLayout><Range Order="0"><BitCount>3</BitCount><BitOffset>21</BitOffset></Range></BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
    </Encodings>
    <Instructions>
      <Instruction>
        <InstructionFlags>
          <IsBranch>false</IsBranch>
          <IsConditionalBranch>false</IsConditionalBranch>
          <IsIndirectBranch>false</IsIndirectBranch>
          <IsProgramTerminator>false</IsProgramTerminator>
          <IsImmediatelyExecuted>false</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>BASE_INST</InstructionName>
        <Description>Base test instruction.</Description>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_TEST</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">1</Opcode>
            <Operands>
              <Operand Input="true" Output="false" IsImplicit="false"
                       IsBinaryMicrocodeRequired="true" Order="1">
                <DataFormatName>FMT_NUM_B32</DataFormatName>
                <OperandType>OPR_TEST</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
        <FunctionalGroup>
          <Name>TEST</Name>
          <FunctionalSubgroups><Subgroup>TEST</Subgroup></FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
    </Instructions>
    <OperandTypes>
      <OperandType><OperandTypeName>OPR_TEST</OperandTypeName></OperandType>
    </OperandTypes>
  </ISA>
</Spec>
'''

_PROFILE = Cdna5Profile()


def _identifier_text(
    opcode: int,
    *,
    alternate_layout: bool = False,
    alternate_encoding_value: bool = False,
) -> str:
    encoding = '1011' if alternate_encoding_value else '1010'
    return f'{encoding}0000{opcode:03b}{"0" * 20}{int(alternate_layout)}'


def _identifier_addition(
    opcode: int,
    *,
    encoding: str = 'ENC_TEST',
    declared_opcode: int | str | None = None,
    radix: int | str = 2,
    text: str | None = None,
) -> str:
    if declared_opcode is None:
        declared_opcode = opcode
    if text is None:
        text = _identifier_text(opcode)
    return f'''\
<EncodingIdentifierAddition>
  <EncodingName>{encoding}</EncodingName>
  <Opcode>{declared_opcode}</Opcode>
  <EncodingIdentifier Radix="{radix}">{text}</EncodingIdentifier>
</EncodingIdentifierAddition>
'''


def _instruction(
    name: str = 'ADDITION_INST',
    encoding: str = 'ENC_TEST',
    opcode: int | str = 2,
    operand_type: str = 'OPR_TEST',
) -> str:
    return f'''\
<Instruction>
  <InstructionFlags>
    <IsBranch>false</IsBranch>
    <IsConditionalBranch>false</IsConditionalBranch>
    <IsIndirectBranch>false</IsIndirectBranch>
    <IsProgramTerminator>false</IsProgramTerminator>
    <IsImmediatelyExecuted>false</IsImmediatelyExecuted>
  </InstructionFlags>
  <InstructionName>{name}</InstructionName>
  <Description>Added test instruction.</Description>
  <InstructionEncodings>
    <InstructionEncoding>
      <EncodingName>{encoding}</EncodingName>
      <EncodingCondition>default</EncodingCondition>
      <Opcode Radix="10">{opcode}</Opcode>
      <Operands>
        <Operand Input="true" Output="false" IsImplicit="false"
                 IsBinaryMicrocodeRequired="true" Order="1">
          <DataFormatName>FMT_NUM_B32</DataFormatName>
          <OperandType>{operand_type}</OperandType>
          <OperandSize>32</OperandSize>
        </Operand>
      </Operands>
    </InstructionEncoding>
  </InstructionEncodings>
  <FunctionalGroup>
    <Name>TEST</Name>
    <FunctionalSubgroups><Subgroup>TEST</Subgroup></FunctionalSubgroups>
  </FunctionalGroup>
</Instruction>
'''


def _write_additions(
    tmp_path: Path,
    instructions: str = '',
    *,
    identifier: str = 'test-additions',
    architecture: str = 'AMD TEST 1',
    schema: str = '1.2.0',
    filename: str = 'additions.xml',
    extra_root: str = '',
    extra_attributes: str = '',
    identifier_additions: str | None = None,
    instructions_attributes: str = '',
    identifier_container_attributes: str = '',
) -> Path:
    identifiers = ''
    if identifier_additions is not None:
        identifiers = f'''\
  <EncodingIdentifierAdditions{identifier_container_attributes}>
    {identifier_additions}
  </EncodingIdentifierAdditions>'''
    path = tmp_path / filename
    path.write_text(f'''\
<IsaAdditions Id="{identifier}" BaseArchitecture="{architecture}"
                     BaseSchemaVersion="{schema}"{extra_attributes}>
  {identifiers}
  <InstructionAdditions{instructions_attributes}>
    {instructions}
  </InstructionAdditions>
  {extra_root}
</IsaAdditions>
''')
    return path


def _base_root() -> elem_tree.Element:
    return elem_tree.fromstring(_BASE_XML)


def test_empty_additions_document_does_not_mutate_base_xml(tmp_path):
    root = _base_root()
    before = elem_tree.tostring(root)
    addition = _write_additions(tmp_path)

    provenance = apply_isa_additions(root, [str(addition)], _PROFILE)

    assert elem_tree.tostring(root) == before
    assert [(item.identifier, item.path) for item in provenance] == [
        ('test-additions', str(addition))
    ]


def test_addition_is_appended_with_provenance(tmp_path):
    root = _base_root()
    addition = _write_additions(tmp_path, _instruction())

    apply_isa_additions(root, [str(addition)], _PROFILE)

    instructions = root.findall('./ISA/Instructions/Instruction')
    assert [node.findtext('InstructionName') for node in instructions] == [
        'BASE_INST',
        'ADDITION_INST',
    ]
    assert instructions[0].get(ADDITION_SOURCE_ATTR) is None
    assert instructions[1].get(ADDITION_SOURCE_ATTR) == 'test-additions'


def test_instruction_using_existing_populated_identifier_succeeds(tmp_path):
    root = _base_root()
    addition = _write_additions(tmp_path, _instruction(opcode=2))

    apply_isa_additions(root, [str(addition)], _PROFILE)

    assert (
        root.findall('./ISA/Instructions/Instruction')[-1].findtext('InstructionName')
        == 'ADDITION_INST'
    )


def test_instruction_and_primary_identifier_are_merged_atomically(tmp_path):
    root = _base_root()
    addition = _write_additions(
        tmp_path,
        _instruction(opcode=3),
        identifier_additions=_identifier_addition(3),
    )

    apply_isa_additions(root, [str(addition)], _PROFILE)

    identifiers = root.findall(
        './ISA/Encodings/Encoding/EncodingIdentifiers/EncodingIdentifier'
    )
    assert identifiers[-1].text == _identifier_text(3)
    assert identifiers[-1].get(ADDITION_SOURCE_ATTR) == 'test-additions'
    assert (
        root.findall('./ISA/Instructions/Instruction')[-1].get(ADDITION_SOURCE_ATTR)
        == 'test-additions'
    )


def test_multiple_additions_preserve_command_line_and_document_order(tmp_path):
    first = _write_additions(
        tmp_path,
        _instruction('ADDITION_A', opcode=2) + _instruction('ADDITION_B', opcode=3),
        identifier='first',
        filename='first.xml',
        identifier_additions=_identifier_addition(3),
    )
    second = _write_additions(
        tmp_path,
        _instruction('ADDITION_C', opcode=4),
        identifier='second',
        filename='second.xml',
        identifier_additions=_identifier_addition(4),
    )
    left = _base_root()
    right = _base_root()

    left_provenance = apply_isa_additions(left, [str(first), str(second)], _PROFILE)
    right_provenance = apply_isa_additions(right, [str(first), str(second)], _PROFILE)

    assert elem_tree.tostring(left) == elem_tree.tostring(right)
    assert [item.identifier for item in left_provenance] == ['first', 'second']
    assert left_provenance == right_provenance
    assert [
        node.findtext('InstructionName')
        for node in left.findall('./ISA/Instructions/Instruction')
    ] == ['BASE_INST', 'ADDITION_A', 'ADDITION_B', 'ADDITION_C']
    assert [
        node.text
        for node in left.findall(
            './ISA/Encodings/Encoding/EncodingIdentifiers/EncodingIdentifier'
        )[-2:]
    ] == [_identifier_text(3), _identifier_text(4)]


def test_all_additions_validate_before_base_is_modified(tmp_path):
    valid = _write_additions(
        tmp_path,
        _instruction('VALID', opcode=2),
        identifier='valid',
        filename='valid.xml',
    )
    invalid = _write_additions(
        tmp_path,
        _instruction('INVALID', encoding='ENC_MISSING', opcode=3),
        identifier='invalid',
        filename='invalid.xml',
    )
    root = _base_root()
    before = elem_tree.tostring(root)

    with pytest.raises(IsaAdditionError, match='unknown encoding'):
        apply_isa_additions(root, [str(valid), str(invalid)], _PROFILE)

    assert elem_tree.tostring(root) == before


def test_bad_later_identifier_addition_does_not_partially_mutate_base(tmp_path):
    first = _write_additions(
        tmp_path,
        _instruction('VALID', opcode=3),
        identifier='first',
        filename='first.xml',
        identifier_additions=_identifier_addition(3),
    )
    second = _write_additions(
        tmp_path,
        identifier='second',
        filename='second.xml',
        identifier_additions=_identifier_addition(4),
    )
    root = _base_root()
    before = elem_tree.tostring(root)

    with pytest.raises(IsaAdditionError, match=r'opcode 4 is unowned'):
        apply_isa_additions(root, [str(first), str(second)], _PROFILE)

    assert elem_tree.tostring(root) == before


@pytest.mark.parametrize(
    ('keyword', 'value', 'message'),
    [
        ('architecture', 'AMD OTHER 1', 'base architecture'),
        ('schema', '1.1.1', 'base schema'),
        ('identifier', 'bad id', 'invalid additions document Id'),
    ],
)
def test_additions_metadata_must_match_base(tmp_path, keyword, value, message):
    addition = _write_additions(tmp_path, _instruction(), **{keyword: value})

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_duplicate_additions_document_id_is_rejected(tmp_path):
    first = _write_additions(tmp_path, identifier='same', filename='first.xml')
    second = _write_additions(tmp_path, identifier='same', filename='second.xml')

    with pytest.raises(IsaAdditionError, match='duplicate additions document Id'):
        apply_isa_additions(_base_root(), [str(first), str(second)], _PROFILE)


def test_duplicate_instruction_across_additions_is_rejected(tmp_path):
    first = _write_additions(
        tmp_path,
        _instruction('SAME_NAME', opcode=2),
        identifier='first',
        filename='first.xml',
    )
    second = _write_additions(
        tmp_path,
        _instruction('SAME_NAME', opcode=3),
        identifier='second',
        filename='second.xml',
    )

    with pytest.raises(IsaAdditionError, match='already exists'):
        apply_isa_additions(_base_root(), [str(first), str(second)], _PROFILE)


def test_instruction_name_collision_after_codegen_normalization_is_rejected(tmp_path):
    addition = _write_additions(tmp_path, _instruction('base_inst', opcode=2))

    with pytest.raises(
        IsaAdditionError,
        match=r"base_inst.*/ENC_TEST.*symbol 'BaseInstTest'.*BASE_INST",
    ):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_invalid_instruction_name_is_rejected_before_merge(tmp_path):
    root = _base_root()
    before = elem_tree.tostring(root)
    addition = _write_additions(tmp_path, _instruction('BAD-NAME', opcode=2))

    with pytest.raises(IsaAdditionError, match=r"invalid instruction name 'BAD-NAME'"):
        apply_isa_additions(root, [str(addition)], _PROFILE)

    assert elem_tree.tostring(root) == before


def test_opcode_constant_name_collision_after_normalization_is_rejected(tmp_path):
    root = _base_root()
    root.find('./ISA/Instructions/Instruction/InstructionName').text = 'S_GETPC_B64'
    addition = _write_additions(tmp_path, _instruction('S_GET_PC_B64', opcode=2))

    with pytest.raises(
        IsaAdditionError,
        match=r"S_GET_PC_B64.*/ENC_TEST.*opcode constant symbol "
        r"'kSGetPcB64Test'.*S_GETPC_B64",
    ):
        apply_isa_additions(root, [str(addition)], _PROFILE)


@pytest.mark.parametrize(
    ('instruction', 'message'),
    [
        (
            _instruction().replace('<Instruction>', '<Instruction Unexpected="1">'),
            'unknown attributes: Unexpected',
        ),
        (
            _instruction().replace(
                '</Instruction>', '<UnexpectedNode /></Instruction>'
            ),
            r'unknown elements: <UnexpectedNode>',
        ),
        (
            _instruction().replace(' Input="true"', ''),
            'missing required attributes: Input',
        ),
        (
            _instruction().replace('<OperandSize>32</OperandSize>', ''),
            r'expected exactly 1 <OperandSize>',
        ),
    ],
)
def test_incomplete_or_unknown_instruction_structure_is_rejected_before_merge(
    tmp_path, instruction, message
):
    root = _base_root()
    before = elem_tree.tostring(root)
    addition = _write_additions(tmp_path, instruction)

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(root, [str(addition)], _PROFILE)

    assert elem_tree.tostring(root) == before


def test_same_instruction_may_repeat_slot_when_only_one_condition_is_active(tmp_path):
    instructions = _instruction().replace(
        '  </InstructionEncodings>',
        '''\
      <InstructionEncoding>
        <EncodingName>ENC_TEST</EncodingName>
        <EncodingCondition>alternate</EncodingCondition>
        <Opcode Radix="10">2</Opcode>
        <Operands>
          <Operand Input="true" Output="false" IsImplicit="false"
                   IsBinaryMicrocodeRequired="true" Order="1">
            <DataFormatName>FMT_NUM_B32</DataFormatName>
            <OperandType>OPR_TEST</OperandType>
            <OperandSize>32</OperandSize>
          </Operand>
        </Operands>
      </InstructionEncoding>
  </InstructionEncodings>''',
    )
    addition = _write_additions(tmp_path, instructions)
    root = _base_root()

    apply_isa_additions(root, [str(addition)], _PROFILE)

    added = root.findall('./ISA/Instructions/Instruction')[-1]
    assert len(added.findall('./InstructionEncodings/InstructionEncoding')) == 2


def test_instruction_requires_an_active_encoding_condition(tmp_path):
    root = _base_root()
    before = elem_tree.tostring(root)
    instruction = _instruction().replace(
        '<EncodingCondition>default</EncodingCondition>',
        '<EncodingCondition>alternate</EncodingCondition>',
    )
    addition = _write_additions(tmp_path, instruction)

    with pytest.raises(IsaAdditionError, match=r'no encoding form active'):
        apply_isa_additions(root, [str(addition)], _PROFILE)

    assert elem_tree.tostring(root) == before


def test_unknown_encoding_condition_is_rejected_before_merge(tmp_path):
    root = _base_root()
    before = elem_tree.tostring(root)
    instruction = _instruction().replace(
        '<EncodingCondition>default</EncodingCondition>',
        '<EncodingCondition>defualt</EncodingCondition>',
    )
    addition = _write_additions(tmp_path, instruction)

    with pytest.raises(IsaAdditionError, match=r"unknown encoding condition 'defualt'"):
        apply_isa_additions(root, [str(addition)], _PROFILE)

    assert elem_tree.tostring(root) == before


def test_encoding_condition_id_must_match_declared_condition(tmp_path):
    instruction = _instruction().replace(
        '<EncodingCondition>default</EncodingCondition>',
        '<EncodingCondition Id="99">default</EncodingCondition>',
    )
    addition = _write_additions(tmp_path, instruction)

    with pytest.raises(
        IsaAdditionError, match=r"condition 'default' has Id '99', expected 0"
    ):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_same_instruction_cannot_repeat_slot_for_same_encoding_condition(tmp_path):
    instruction = elem_tree.fromstring(_instruction())
    encodings = instruction.find('InstructionEncodings')
    encodings.append(deepcopy(encodings[0]))
    addition = _write_additions(
        tmp_path, elem_tree.tostring(instruction, encoding='unicode')
    )

    with pytest.raises(IsaAdditionError, match='repeats opcode.*condition'):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


@pytest.mark.parametrize(
    ('instructions', 'message'),
    [
        (_instruction('BASE_INST', opcode=2), 'already exists'),
        (_instruction('ADDITION_INST', opcode=1), 'already owned'),
        (_instruction(encoding='ENC_MISSING'), 'unknown encoding'),
        (_instruction(operand_type='OPR_MISSING'), 'unknown operand type'),
        (_instruction(opcode='not-decimal'), 'invalid decimal opcode'),
        (_instruction(opcode=-1), 'negative opcode'),
        (_instruction(opcode=8), 'outside the encoding range'),
    ],
)
def test_conflicting_or_invalid_additions_are_rejected(tmp_path, instructions, message):
    addition = _write_additions(tmp_path, instructions)

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


@pytest.mark.parametrize(
    ('identifier_additions', 'instructions', 'message'),
    [
        (
            _identifier_addition(3, encoding='ENC_MISSING'),
            _instruction(opcode=3),
            'unknown encoding',
        ),
        (
            _identifier_addition(3, text=f'{_identifier_text(3)[:-1]}x'),
            _instruction(opcode=3),
            'malformed radix-2 identifier',
        ),
        (
            _identifier_addition(3, text=_identifier_text(3)[:-1]),
            _instruction(opcode=3),
            'identifier width 31.*encoding width 32',
        ),
        (
            _identifier_addition(3, radix=16),
            _instruction(opcode=3),
            'identifier radix 16.*encoding radix 2',
        ),
        (
            _identifier_addition(3, text=f'0{_identifier_text(3)[1:]}'),
            _instruction(opcode=3),
            'incompatible with the base identifier mask/layout',
        ),
        (
            _identifier_addition(4, declared_opcode=3),
            _instruction(opcode=3),
            'decodes opcode 4, not declared opcode 3',
        ),
        (_identifier_addition(3), '', 'opcode 3 is unowned'),
        (_identifier_addition(1), '', 'duplicate identifier'),
    ],
)
def test_invalid_identifier_additions_are_rejected(
    tmp_path, identifier_additions, instructions, message
):
    addition = _write_additions(
        tmp_path,
        instructions,
        identifier_additions=identifier_additions,
    )

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


@pytest.mark.parametrize(
    ('fragment', 'message'),
    [
        (
            '<EncodingIdentifierAddition><Opcode>3</Opcode>'
            f'<EncodingIdentifier Radix="2">{_identifier_text(3)}</EncodingIdentifier>'
            '</EncodingIdentifierAddition>',
            'exactly one <EncodingName>',
        ),
        (
            '<EncodingIdentifierAddition><EncodingName>ENC_TEST</EncodingName>'
            f'<EncodingIdentifier Radix="2">{_identifier_text(3)}</EncodingIdentifier>'
            '</EncodingIdentifierAddition>',
            'exactly one <Opcode>',
        ),
        (
            '<EncodingIdentifierAddition><EncodingName>ENC_TEST</EncodingName>'
            '<Opcode>3</Opcode></EncodingIdentifierAddition>',
            'exactly one <EncodingIdentifier>',
        ),
        (
            _identifier_addition(3).replace('Radix="2"', 'Radix="2" Replace="true"'),
            'unknown attributes: Replace',
        ),
    ],
)
def test_missing_or_unknown_identifier_structure_is_rejected(
    tmp_path, fragment, message
):
    addition = _write_additions(
        tmp_path,
        _instruction(opcode=3),
        identifier_additions=fragment,
    )

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_identifier_collision_against_earlier_addition_is_rejected(tmp_path):
    first = _write_additions(
        tmp_path,
        _instruction(opcode=3),
        identifier='first',
        filename='first.xml',
        identifier_additions=_identifier_addition(3),
    )
    second = _write_additions(
        tmp_path,
        identifier='second',
        filename='second.xml',
        identifier_additions=_identifier_addition(
            3, text=_identifier_text(3, alternate_layout=True)
        ),
    )

    with pytest.raises(IsaAdditionError, match=r'collides.*opcode 3'):
        apply_isa_additions(_base_root(), [str(first), str(second)], _PROFILE)


def test_identifier_opcode_collision_with_different_encoding_value_is_rejected(
    tmp_path,
):
    first = _write_additions(
        tmp_path,
        _instruction(opcode=3),
        identifier='first',
        filename='first.xml',
        identifier_additions=_identifier_addition(3),
    )
    second = _write_additions(
        tmp_path,
        identifier='second',
        filename='second.xml',
        identifier_additions=_identifier_addition(
            3,
            text=_identifier_text(3, alternate_encoding_value=True),
        ),
    )

    with pytest.raises(IsaAdditionError, match=r'collides.*opcode 3'):
        apply_isa_additions(_base_root(), [str(first), str(second)], _PROFILE)


def test_identifier_collision_against_base_is_rejected(tmp_path):
    addition = _write_additions(
        tmp_path,
        identifier_additions=_identifier_addition(
            1, text=_identifier_text(1, alternate_layout=True)
        ),
    )

    with pytest.raises(IsaAdditionError, match=r'collides.*opcode 1'):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_duplicate_identifier_against_earlier_addition_is_rejected(tmp_path):
    first = _write_additions(
        tmp_path,
        _instruction(opcode=3),
        identifier='first',
        filename='first.xml',
        identifier_additions=_identifier_addition(3),
    )
    second = _write_additions(
        tmp_path,
        identifier='second',
        filename='second.xml',
        identifier_additions=_identifier_addition(3),
    )

    with pytest.raises(IsaAdditionError, match='duplicate identifier'):
        apply_isa_additions(_base_root(), [str(first), str(second)], _PROFILE)


def test_empty_identifier_additions_group_is_rejected(tmp_path):
    addition = _write_additions(tmp_path, identifier_additions='')

    with pytest.raises(IsaAdditionError, match='must not be empty'):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_unknown_additions_structure_is_rejected(tmp_path):
    addition = _write_additions(tmp_path, extra_root='<Encodings />')

    with pytest.raises(IsaAdditionError, match='unknown additions root elements'):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_unknown_additions_attribute_is_rejected(tmp_path):
    addition = _write_additions(tmp_path, extra_attributes=' Replace="true"')

    with pytest.raises(IsaAdditionError, match='unknown additions root attributes'):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


@pytest.mark.parametrize(
    ('container', 'kwargs'),
    [
        ('InstructionAdditions', {'instructions_attributes': ' Replace="true"'}),
        (
            'EncodingIdentifierAdditions',
            {
                'identifier_additions': _identifier_addition(3),
                'identifier_container_attributes': ' Replace="true"',
            },
        ),
    ],
)
def test_additions_container_attributes_are_rejected(tmp_path, container, kwargs):
    addition = _write_additions(tmp_path, **kwargs)

    with pytest.raises(
        IsaAdditionError, match=rf'<{container}> must not have attributes: Replace'
    ):
        apply_isa_additions(_base_root(), [str(addition)], _PROFILE)


def test_cli_additions_arguments_are_grouped_in_order():
    assert _group_isa_additions_args(
        ['cdna5:first.xml', 'rdna4:other.xml', 'cdna5:second.xml']
    ) == {
        'cdna5': ['first.xml', 'second.xml'],
        'rdna4': ['other.xml'],
    }


@pytest.mark.parametrize('entry', ['missing-colon', ':additions.xml', 'cdna5:'])
def test_cli_additions_argument_requires_name_and_path(entry):
    with pytest.raises(ValueError, match='name'):
        _group_isa_additions_args([entry])


def _write_encoding_clone_additions(
    tmp_path: Path,
    base_xml: Path,
    *,
    encoding_name: str,
    instruction_name: str,
    opcode: int,
) -> Path:
    base_root = elem_tree.parse(base_xml).getroot()
    original = next(
        instruction
        for instruction in base_root.findall('./ISA/Instructions/Instruction')
        if any(
            encoding.findtext('EncodingName') == encoding_name
            for encoding in instruction.findall(
                './InstructionEncodings/InstructionEncoding'
            )
        )
    )
    addition = deepcopy(original)
    addition.find('InstructionName').text = instruction_name
    instruction_encodings = addition.find('InstructionEncodings')
    for encoding in list(instruction_encodings):
        if encoding.findtext('EncodingName') != encoding_name:
            instruction_encodings.remove(encoding)
        else:
            encoding.find('Opcode').text = str(opcode)

    addition_root = elem_tree.Element(
        'IsaAdditions',
        {
            'Id': 'compat-slot-test',
            'BaseArchitecture': base_root.findtext(
                './ISA/Architecture/ArchitectureName'
            ),
            'BaseSchemaVersion': base_root.findtext('./Document/SchemaVersion'),
        },
    )
    instructions = elem_tree.SubElement(addition_root, 'InstructionAdditions')
    instructions.append(addition)
    path = tmp_path / 'compat-slot-test.xml'
    elem_tree.ElementTree(addition_root).write(path, encoding='unicode')
    return path


@pytest.mark.parametrize(
    ('base_filename', 'profile', 'encoding_name', 'opcode', 'owner'),
    [
        (
            'amdgpu_isa_cdna5.xml',
            Cdna5Profile(),
            'ENC_VOP1',
            103,
            'V_PERMLANE64_B32',
        ),
        ('amdgpu_isa_rdna4.xml', Rdna4Profile(), 'ENC_SOPP', 9, 'S_WAITCNT'),
    ],
)
def test_compatibility_instruction_slots_are_reserved_before_merge(
    tmp_path, base_filename, profile, encoding_name, opcode, owner
):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / base_filename
    )
    addition = _write_encoding_clone_additions(
        tmp_path,
        base_xml,
        encoding_name=encoding_name,
        instruction_name='COLLIDING_ADDITION',
        opcode=opcode,
    )
    root = elem_tree.parse(base_xml).getroot()
    before = elem_tree.tostring(root)

    with pytest.raises(
        IsaAdditionError,
        match=rf'opcode {opcode}.*already owned by instruction {owner!r}',
    ):
        apply_isa_additions(root, [str(addition)], profile)

    assert elem_tree.tostring(root) == before


def test_instruction_in_fully_skipped_encoding_is_rejected_before_merge(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_encoding_clone_additions(
        tmp_path,
        base_xml,
        encoding_name='ENC_VOP3PX2',
        instruction_name='SKIPPED_ADDITION',
        opcode=52,
    )
    root = elem_tree.parse(base_xml).getroot()
    before = elem_tree.tostring(root)

    with pytest.raises(IsaAdditionError, match=r'no encoding form active'):
        apply_isa_additions(root, [str(addition)], Cdna5Profile())

    assert elem_tree.tostring(root) == before


def _cdna1_mimg_identifier(opcode: int) -> str:
    value = list('00000000000000000000000000000000' '11110000000000000000000000000000')
    value[39:46] = f'{opcode:07b}'
    return ''.join(value)


def _write_cdna1_mimg_opm_additions(tmp_path: Path, base_xml: Path) -> Path:
    base_root = elem_tree.parse(base_xml).getroot()
    original = next(
        instruction
        for instruction in base_root.findall('./ISA/Instructions/Instruction')
        if any(
            encoding.findtext('EncodingName') == 'ENC_MIMG'
            for encoding in instruction.findall(
                './InstructionEncodings/InstructionEncoding'
            )
        )
    )
    addition = deepcopy(original)
    addition.find('InstructionName').text = 'IMAGE_ADDITION_TEST'
    instruction_encodings = addition.find('InstructionEncodings')
    for encoding in list(instruction_encodings):
        if encoding.findtext('EncodingName') != 'ENC_MIMG':
            instruction_encodings.remove(encoding)
        else:
            encoding.find('Opcode').text = '134'

    addition_root = elem_tree.Element(
        'IsaAdditions',
        {
            'Id': 'cdna1-mimg-opm-test',
            'BaseArchitecture': 'AMD CDNA 1',
            'BaseSchemaVersion': '1.1.1',
        },
    )
    identifier_additions = elem_tree.SubElement(
        addition_root, 'EncodingIdentifierAdditions'
    )
    identifier_addition = elem_tree.SubElement(
        identifier_additions, 'EncodingIdentifierAddition'
    )
    elem_tree.SubElement(identifier_addition, 'EncodingName').text = 'ENC_MIMG'
    elem_tree.SubElement(identifier_addition, 'Opcode').text = '6'
    identifier = elem_tree.SubElement(
        identifier_addition, 'EncodingIdentifier', {'Radix': '2'}
    )
    identifier.text = _cdna1_mimg_identifier(6)
    instructions = elem_tree.SubElement(addition_root, 'InstructionAdditions')
    instructions.append(addition)
    path = tmp_path / 'cdna1-mimg-opm-test.xml'
    elem_tree.ElementTree(addition_root).write(path, encoding='unicode')
    return path


def test_identifier_ownership_uses_expanded_opm_opcode(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna1.xml'
    )
    addition = _write_cdna1_mimg_opm_additions(tmp_path, base_xml)

    spec = Parser(str(base_xml), Cdna1Profile(), [str(addition)]).parse()

    added = next(
        instruction
        for instruction in spec.encoding_map['ENC_MIMG'].insts
        if instruction.name == 'IMAGE_ADDITION_TEST'
    )
    assert added.opcode == 134
    assert spec.encoding_map['ENC_MIMG'].primary_dt_ptrs[134] != -1


def _cdna5_sop1_identifier(opcode: int, *, implied_literal: bool = False) -> str:
    if implied_literal:
        value = list(
            '00000000000000000000000000000000' '10111110100000000000000000000000'
        )
        value[48:56] = f'{opcode:08b}'
    else:
        value = list('10111110100000000000000000000000')
        value[16:24] = f'{opcode:08b}'
    return ''.join(value)


_CDNA5_SOP1_OPCODE7_BYTES = bytes.fromhex('00 07 80 be')
_CDNA5_SOP1_LITERAL_OPCODE7_BYTES = bytes.fromhex('00 07 80 be 00 00 00 00')


def _cdna5_primary_table_index(encoded: bytes) -> int:
    return int.from_bytes(encoded[:4], byteorder='little') >> 23


def _assert_sop1_opcode7_decode_mapping(spec) -> None:
    table_index = _cdna5_primary_table_index(_CDNA5_SOP1_OPCODE7_BYTES)
    assert table_index == 381
    assert _cdna5_sop1_identifier(7) == (
        f'{int.from_bytes(_CDNA5_SOP1_OPCODE7_BYTES, byteorder="little"):032b}'
    )
    assert spec.encoding_map['ENC_SOP1'].primary_dt_ptrs[7] == table_index

    entry = spec.primary_decode_table[table_index]
    assert entry.decode_func == 'subDecodeSop1'
    assert entry.sub_decode_table == 'sub_decode_sop1'
    assert entry.sub_decode_funcs[7] == 'decodeSAdditionTestB32Sop1'


def _write_cdna5_clone_additions(
    tmp_path: Path,
    base_xml: Path,
    *,
    opcode: int = 7,
    instruction_name: str = 'S_ADDITION_TEST_B32',
    add_identifiers: bool = False,
    include_implied_literal: bool = False,
) -> Path:
    base_root = elem_tree.parse(base_xml).getroot()
    original = next(
        instruction
        for instruction in base_root.findall('./ISA/Instructions/Instruction')
        if instruction.findtext('InstructionName') == 'S_MOV_B32'
    )
    addition = deepcopy(original)
    addition.find('InstructionName').text = instruction_name
    retained_encodings = {'ENC_SOP1'}
    if include_implied_literal:
        retained_encodings.add('SOP1_INST_LITERAL')
    instruction_encodings = addition.find('InstructionEncodings')
    for encoding in list(instruction_encodings):
        if encoding.findtext('EncodingName') not in retained_encodings:
            instruction_encodings.remove(encoding)
    for opcode_node in addition.findall(
        './InstructionEncodings/InstructionEncoding/Opcode'
    ):
        opcode_node.text = str(opcode)

    addition_root = elem_tree.Element(
        'IsaAdditions',
        {
            'Id': 'cdna5-test',
            'BaseArchitecture': 'AMD CDNA 5',
            'BaseSchemaVersion': '1.2.0',
        },
    )
    if add_identifiers:
        identifier_additions = elem_tree.SubElement(
            addition_root, 'EncodingIdentifierAdditions'
        )
        for encoding_name, implied_literal in (
            ('ENC_SOP1', False),
            ('SOP1_INST_LITERAL', True),
        ):
            if implied_literal and not include_implied_literal:
                continue
            identifier_addition = elem_tree.SubElement(
                identifier_additions, 'EncodingIdentifierAddition'
            )
            elem_tree.SubElement(identifier_addition, 'EncodingName').text = (
                encoding_name
            )
            elem_tree.SubElement(identifier_addition, 'Opcode').text = str(opcode)
            identifier = elem_tree.SubElement(
                identifier_addition, 'EncodingIdentifier', {'Radix': '2'}
            )
            identifier.text = _cdna5_sop1_identifier(
                opcode, implied_literal=implied_literal
            )
    instructions = elem_tree.SubElement(addition_root, 'InstructionAdditions')
    instructions.append(addition)
    path = tmp_path / 'cdna5-test.xml'
    elem_tree.ElementTree(addition_root).write(path, encoding='unicode')
    return path


def test_cdna5_distinct_active_conditions_cannot_generate_the_same_class(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml)
    tree = elem_tree.parse(addition)
    encodings = tree.find('./InstructionAdditions/Instruction/InstructionEncodings')
    primary = next(
        encoding
        for encoding in encodings
        if encoding.findtext('EncodingName') == 'ENC_SOP1'
    )
    duplicate = deepcopy(primary)
    condition = duplicate.find('EncodingCondition')
    condition.text = 'default'
    condition.set('Id', '0')
    encodings.append(duplicate)
    tree.write(addition, encoding='unicode')
    root = elem_tree.parse(base_xml).getroot()
    before = elem_tree.tostring(root)

    with pytest.raises(
        IsaAdditionError,
        match=r"multiple active forms.*symbol 'SAdditionTestB32Sop1'",
    ):
        apply_isa_additions(root, [str(addition)], Cdna5Profile())

    assert elem_tree.tostring(root) == before


def test_cdna5_unknown_operand_field_is_rejected_before_merge(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml)
    tree = elem_tree.parse(addition)
    primary = next(
        encoding
        for encoding in tree.findall(
            './InstructionAdditions/Instruction/InstructionEncodings/'
            'InstructionEncoding'
        )
        if encoding.findtext('EncodingName') == 'ENC_SOP1'
    )
    destination = next(
        field
        for field in primary.findall('./Operands/Operand/FieldName')
        if field.text == 'SDST'
    )
    destination.text = 'BOGUS'
    tree.write(addition, encoding='unicode')
    root = elem_tree.parse(base_xml).getroot()
    before = elem_tree.tostring(root)

    with pytest.raises(IsaAdditionError, match=r"unknown operand field 'BOGUS'"):
        apply_isa_additions(root, [str(addition)], Cdna5Profile())

    assert elem_tree.tostring(root) == before


@pytest.mark.parametrize(
    ('field_name', 'message'),
    [
        ('literal64', r"unknown operand field 'literal64'"),
        (
            'LITERAL',
            r"implied-literal operand field 'LITERAL'.*type or role not established",
        ),
    ],
)
def test_cdna5_implied_literal_field_requires_an_established_contract(
    tmp_path, field_name, message
):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(
        tmp_path, base_xml, include_implied_literal=True
    )
    tree = elem_tree.parse(addition)
    implied_literal = next(
        encoding
        for encoding in tree.findall(
            './InstructionAdditions/Instruction/InstructionEncodings/'
            'InstructionEncoding'
        )
        if encoding.findtext('EncodingName') == 'SOP1_INST_LITERAL'
    )
    destination = next(
        operand
        for operand in implied_literal.findall('./Operands/Operand')
        if operand.findtext('FieldName') == 'SDST'
    )
    destination.find('FieldName').text = field_name
    tree.write(addition, encoding='unicode')
    root = elem_tree.parse(base_xml).getroot()
    before = elem_tree.tostring(root)

    with pytest.raises(IsaAdditionError, match=message):
        apply_isa_additions(root, [str(addition)], Cdna5Profile())

    assert elem_tree.tostring(root) == before


def test_cdna5_implied_literal_contract_allows_a_new_value_shape(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(
        tmp_path,
        base_xml,
        add_identifiers=True,
        include_implied_literal=True,
    )
    tree = elem_tree.parse(addition)
    literal_operands = [
        operand
        for encoding in tree.findall(
            './InstructionAdditions/Instruction/InstructionEncodings/'
            'InstructionEncoding'
        )
        if encoding.findtext('EncodingName') == 'SOP1_INST_LITERAL'
        for operand in encoding.findall('./Operands/Operand')
        if operand.findtext('FieldName') == 'LITERAL'
    ]
    assert literal_operands
    for operand in literal_operands:
        operand.find('OperandSize').text = '128'
        operand.find('DataFormatName').text = 'FMT_NUM_PK2_F64'
    tree.write(addition, encoding='unicode')

    spec = Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()

    added = next(
        instruction
        for instruction in spec.encoding_map['ENC_SOP1'].insts
        if instruction.name == 'S_ADDITION_TEST_B32'
    )
    assert added.source_addition == spec.applied_additions[0]


def test_cdna5_normalized_instruction_name_collision_is_rejected(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(
        tmp_path,
        base_xml,
        instruction_name='s_mov_b32',
    )

    with pytest.raises(
        IsaAdditionError,
        match=r"s_mov_b32.*/ENC_SOP1.*symbol 'SMovB32Sop1'.*S_MOV_B32",
    ):
        apply_isa_additions(
            elem_tree.parse(base_xml).getroot(), [str(addition)], Cdna5Profile()
        )


def test_parser_rejects_unknown_base_addition_provenance(tmp_path):
    source_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    tree = elem_tree.parse(source_xml)
    instruction = tree.find('./ISA/Instructions/Instruction')
    instruction_name = instruction.findtext('InstructionName')
    instruction.set(ADDITION_SOURCE_ATTR, 'unknown-addition')
    base_xml = tmp_path / 'base-with-unknown-provenance.xml'
    tree.write(base_xml, encoding='unicode')

    with pytest.raises(
        IsaAdditionError,
        match=rf'{instruction_name}.*{ADDITION_SOURCE_ATTR}.*unknown-addition',
    ):
        Parser(str(base_xml), Cdna5Profile()).parse()


def test_parser_retains_additions_provenance_on_added_instructions(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml, add_identifiers=True)

    spec = Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()

    assert [(item.identifier, item.path) for item in spec.applied_additions] == [
        ('cdna5-test', str(addition))
    ]
    added = [
        instruction
        for encoding in spec.inst_encodings
        for instruction in encoding.insts
        if instruction.name == 'S_ADDITION_TEST_B32'
    ]
    assert added
    assert all(
        instruction.source_addition == spec.applied_additions[0]
        for instruction in added
    )
    _assert_sop1_opcode7_decode_mapping(spec)


def test_primary_and_implied_literal_identifiers_build_decode_pointers(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(
        tmp_path,
        base_xml,
        add_identifiers=True,
        include_implied_literal=True,
    )

    spec = Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()

    _assert_sop1_opcode7_decode_mapping(spec)
    literal_table_index = _cdna5_primary_table_index(_CDNA5_SOP1_LITERAL_OPCODE7_BYTES)
    assert literal_table_index == 381
    assert _cdna5_sop1_identifier(7, implied_literal=True) == (
        f'{int.from_bytes(_CDNA5_SOP1_LITERAL_OPCODE7_BYTES, byteorder="little"):064b}'
    )
    assert (
        spec.encoding_map['SOP1_INST_LITERAL'].primary_dt_ptrs[7] == literal_table_index
    )
    literal_entry = spec.primary_decode_table[literal_table_index]
    assert literal_entry.decode_func == 'subDecodeSop1'
    assert literal_entry.sub_decode_table == 'sub_decode_sop1'
    assert literal_entry.sub_decode_funcs[7] == 'decodeSAdditionTestB32Sop1'


def test_implied_literal_identifier_requires_matching_parent_owner(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(
        tmp_path,
        base_xml,
        add_identifiers=True,
        include_implied_literal=True,
    )
    tree = elem_tree.parse(addition)
    for parent_path in (
        './EncodingIdentifierAdditions',
        './InstructionAdditions/Instruction/InstructionEncodings',
    ):
        parent = tree.find(parent_path)
        for child in list(parent):
            if child.findtext('EncodingName') == 'ENC_SOP1':
                parent.remove(child)
    tree.write(addition, encoding='unicode')

    with pytest.raises(
        IsaAdditionError, match=r'implied-literal identifier.*parent encoding.*owner'
    ):
        Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()


def test_parser_rejects_added_instruction_without_decode_identifier(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml)

    with pytest.raises(
        IsaAdditionError, match=r'S_ADDITION_TEST_B32.*opcode 7.*unreachable'
    ):
        Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()


def test_later_parse_without_additions_is_not_affected(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml, add_identifiers=True)

    with_addition = Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()
    without_addition = Parser(str(base_xml), Cdna5Profile()).parse()

    assert with_addition.encoding_map['ENC_SOP1'].primary_dt_ptrs[7] != -1
    assert without_addition.encoding_map['ENC_SOP1'].primary_dt_ptrs[7] == -1
    assert all(
        instruction.name != 'S_ADDITION_TEST_B32'
        for encoding in without_addition.inst_encodings
        for instruction in encoding.insts
    )


def test_parser_rejects_added_opcode_outside_base_encoding_range(tmp_path):
    base_xml = (
        Path(__file__).resolve().parents[6]
        / 'shared'
        / 'machine-readable-isa'
        / 'isa'
        / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml)
    tree = elem_tree.parse(addition)
    for opcode in tree.findall(
        './InstructionAdditions/Instruction/InstructionEncodings/'
        'InstructionEncoding/Opcode'
    ):
        opcode.text = '999999'
    tree.write(addition, encoding='unicode')

    with pytest.raises(
        IsaAdditionError,
        match=r'S_ADDITION_TEST_B32.*/ENC_SOP1 opcode 999999.*outside the encoding range',
    ):
        Parser(str(base_xml), Cdna5Profile(), [str(addition)]).parse()


def _generated_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*')
        if path.is_file()
    }


def test_empty_additions_document_is_byte_identical_through_cli(tmp_path):
    repo_root = Path(__file__).resolve().parents[6]
    base_xml = (
        repo_root / 'shared' / 'machine-readable-isa' / 'isa' / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_additions(
        tmp_path,
        identifier='empty-cdna5',
        architecture='AMD CDNA 5',
        filename='empty-cdna5.xml',
    )
    baseline_output = tmp_path / 'baseline'
    additions_output = tmp_path / 'with-additions'
    baseline_output.mkdir()
    additions_output.mkdir()

    subprocess.run(
        [
            sys.executable,
            '-m',
            'amdisa',
            f'cdna5:{base_xml}',
            '--isa-output',
            str(baseline_output / 'isa'),
            '--include-root',
            str(baseline_output),
            '--dbt-output',
            str(baseline_output / 'dbt'),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            '-m',
            'amdisa',
            f'cdna5:{base_xml}',
            '--isa-additions',
            f'cdna5:{addition}',
            '--isa-output',
            str(additions_output / 'isa'),
            '--include-root',
            str(additions_output),
            '--dbt-output',
            str(additions_output / 'dbt'),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert _generated_files(additions_output) == _generated_files(baseline_output)


def test_nonempty_additions_document_generates_decoder_dispatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[6]
    base_xml = (
        repo_root / 'shared' / 'machine-readable-isa' / 'isa' / 'amdgpu_isa_cdna5.xml'
    )
    addition = _write_cdna5_clone_additions(tmp_path, base_xml, add_identifiers=True)
    output = tmp_path / 'generated'
    output.mkdir()

    subprocess.run(
        [
            sys.executable,
            '-m',
            'amdisa',
            f'cdna5:{base_xml}',
            '--isa-additions',
            f'cdna5:{addition}',
            '--isa-output',
            str(output / 'isa'),
            '--dbt-output',
            str(output / 'dbt'),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    decoder = (output / 'isa' / 'cdna5' / 'decoder.cpp').read_text()
    match = re.search(
        r'DecoderImpl::sub_decode_sop1 = \{\s*(.*?)\s*\};', decoder, re.DOTALL
    )
    assert match is not None
    entries = [entry.strip() for entry in match.group(1).split(',')]
    assert entries[7] == '&detail::decodeSAdditionTestB32Sop1'
