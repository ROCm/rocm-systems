# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import xml.etree.ElementTree as elem_tree

from amdisa.isa_profile import Rdna1Profile
from amdisa.parser import Parser, _collapse_register_ranges


def _encoding(field: str) -> elem_tree.Element:
    return elem_tree.fromstring(f'''\
<Encoding Order="0">
  <EncodingName>ENC_TEST</EncodingName>
  <BitCount>32</BitCount>
  <MicrocodeFormat>
    <BitMap>
      <Field IsConditional="false">
        <FieldName>ENCODING</FieldName>
        <BitLayout RangeCount="1">
          <Range Order="0"><BitCount>6</BitCount><BitOffset>26</BitOffset></Range>
        </BitLayout>
      </Field>
      {field}
    </BitMap>
  </MicrocodeFormat>
</Encoding>
''')


def _parser() -> Parser:
    parser = object.__new__(Parser)
    parser.profile = Rdna1Profile()
    return parser


def test_partitioned_opcode_retains_opm_decode_field():
    encoding = _encoding('''\
<Field IsConditional="false">
  <FieldName>OP</FieldName>
  <BitLayout RangeCount="2">
    <Range Order="0"><BitCount>7</BitCount><BitOffset>18</BitOffset></Range>
    <Range Order="1"><BitCount>1</BitCount><BitOffset>0</BitOffset></Range>
  </BitLayout>
</Field>''')

    fields, encoding_bits, opcode_bits, opm_bits = _parser().parse_ucode_bitmap(
        encoding, 32
    )

    by_name = {field.name: field for field in fields}
    assert (encoding_bits, opcode_bits, opm_bits) == (6, 7, 1)
    assert (by_name['op'].bit_cnt, by_name['op'].bit_offset) == (7, 18)
    assert (by_name['opm'].bit_cnt, by_name['opm'].bit_offset) == (1, 0)


def test_partitioned_non_opcode_field_uses_logical_bit_suffix():
    encoding = _encoding('''\
<Field IsConditional="false">
  <FieldName>OP_SEL_HI</FieldName>
  <BitLayout RangeCount="2">
    <Range Order="0"><BitCount>2</BitCount><BitOffset>59</BitOffset></Range>
    <Range Order="1"><BitCount>1</BitCount><BitOffset>14</BitOffset></Range>
  </BitLayout>
</Field>
<Field IsConditional="false">
  <FieldName>OP</FieldName>
  <BitLayout RangeCount="1">
    <Range Order="0"><BitCount>7</BitCount><BitOffset>18</BitOffset></Range>
  </BitLayout>
</Field>''')

    fields, _, _, _ = _parser().parse_ucode_bitmap(encoding, 64)

    by_name = {field.name: field for field in fields}
    assert (by_name['op_sel_hi'].bit_cnt, by_name['op_sel_hi'].bit_offset) == (2, 59)
    assert (by_name['op_sel_hi_2'].bit_cnt, by_name['op_sel_hi_2'].bit_offset) == (
        1,
        14,
    )


def test_uppercase_predefined_names_keep_register_ranges_and_disassembly_case():
    pairs = list(elem_tree.fromstring('''\
<PredefinedValues>
  <PredefinedValue><Name>TTMP0</Name><Value>108</Value></PredefinedValue>
  <PredefinedValue><Name>TTMP1</Name><Value>109</Value></PredefinedValue>
  <PredefinedValue><Name>NULL</Name><Value>124</Value></PredefinedValue>
</PredefinedValues>
'''))

    values, patterns = _collapse_register_ranges(pairs, 'OPR_SREG', {}, True)

    assert values == [
        ('OPR_SREG_TTMP_MIN', '108'),
        ('OPR_SREG_TTMP_MAX', '109'),
        ('OPR_SREG_NULL', '124'),
    ]
    assert patterns[-1].operand_name == 'null'
