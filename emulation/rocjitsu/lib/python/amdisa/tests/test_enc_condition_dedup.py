# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for parse_encoding_conditions de-duplication (P2 workaround).

The CDNA3 XML has three identical ``default`` EncodingCondition blocks in
ENC_FLAT.  parse_encoding_conditions must emit only the first occurrence.
These tests bypass Parser.__init__ (which requires a real XML file) and
patch parse_condition with a controlled stub.
"""

import xml.etree.ElementTree as ET

from amdisa.parser import Parser
from amdisa import xml_schema as xs


def _make_parser() -> Parser:
    """Create a Parser instance without calling __init__ (no XML file needed)."""
    return object.__new__(Parser)


def _make_conds_node(names: list[str]) -> ET.Element:
    """Build a minimal EncodingConditions XML element with the given names."""
    root = ET.Element(xs.ENCODING_CONDS)
    for name in names:
        cond = ET.SubElement(root, xs.ENCODING_COND)
        name_el = ET.SubElement(cond, xs.COND_NAME)
        name_el.text = name
    return root


def _stub_parse_condition(cond_node: ET.Element, enc_name: str) -> tuple[str, str]:
    """Minimal parse_condition replacement: return (name, 'expr')."""
    name_node = cond_node.find(xs.COND_NAME)
    assert name_node is not None
    name = name_node.text or ''
    # Mirror the real logic: rename 'default' to 'default_encoding'
    if name == 'default':
        name = 'default_encoding'
    return (name, f'expr_for_{name}')


class TestEncConditionDedup:
    def setup_method(self):
        self.parser = _make_parser()
        # Patch parse_condition on the instance
        self.parser.parse_condition = _stub_parse_condition  # type: ignore[method-assign]

    def test_unique_names_kept_in_order(self):
        """All unique names are preserved in insertion order."""
        conds = _make_conds_node(['alpha', 'beta', 'gamma'])
        result = self.parser.parse_encoding_conditions(conds, 'ENC_TEST')
        assert [r[0] for r in result] == ['alpha', 'beta', 'gamma']

    def test_duplicate_dropped_first_wins(self):
        """A duplicate name is silently dropped; the first occurrence wins."""
        conds = _make_conds_node(['alpha', 'alpha', 'beta'])
        result = self.parser.parse_encoding_conditions(conds, 'ENC_TEST')
        assert [r[0] for r in result] == ['alpha', 'beta']
        assert len(result) == 2

    def test_cdna3_triple_default_reduced_to_one(self):
        """CDNA3 has three identical 'default' conditions; only one survives."""
        conds = _make_conds_node(['default', 'default', 'default'])
        result = self.parser.parse_encoding_conditions(conds, 'ENC_FLAT')
        # parse_condition renames 'default' -> 'default_encoding'
        assert len(result) == 1
        assert result[0][0] == 'default_encoding'

    def test_mixed_unique_and_duplicates(self):
        """Mixed unique and duplicate names: unique preserved, dups collapsed."""
        conds = _make_conds_node(['a', 'b', 'a', 'c', 'b', 'd'])
        result = self.parser.parse_encoding_conditions(conds, 'ENC_TEST')
        assert [r[0] for r in result] == ['a', 'b', 'c', 'd']

    def test_empty_conditions_node(self):
        """No child elements → empty result."""
        conds = ET.Element(xs.ENCODING_CONDS)
        result = self.parser.parse_encoding_conditions(conds, 'ENC_TEST')
        assert result == []

    def test_expressions_are_preserved(self):
        """The expression from the first occurrence is kept, not from duplicates."""
        # Override stub so each call returns a distinct expression
        call_order = []

        def tracking_parse(cond_node: ET.Element, enc_name: str) -> tuple[str, str]:
            name_node = cond_node.find(xs.COND_NAME)
            name = name_node.text or ''  # type: ignore[union-attr]
            n = len(call_order)
            call_order.append(name)
            return (name, f'expr_{n}')

        self.parser.parse_condition = tracking_parse  # type: ignore[method-assign]
        conds = _make_conds_node(['x', 'x'])
        result = self.parser.parse_encoding_conditions(conds, 'ENC_TEST')
        assert len(result) == 1
        # Expression from the first call (expr_0), not the duplicate (expr_1)
        assert result[0][1] == 'expr_0'

    def test_sanitize_condition_name_handles_xml_operators(self):
        """XML condition operators are converted to C++ identifier text."""
        sanitize = Parser._sanitize_condition_name
        assert sanitize('!has_lit_0') == 'not_has_lit_0'
        assert sanitize('!has_lit_0&!has_lit_1') == ('not_has_lit_0_and_not_has_lit_1')
        assert sanitize('a|b') == 'a_or_b'
        assert sanitize('') == 'condition'
        assert sanitize('1foo') == 'cond_1foo'

    def test_sanitize_condition_name_is_not_injective(self):
        """Sanitization is a best-effort identifier mapping, not a unique key."""
        sanitize = Parser._sanitize_condition_name
        assert sanitize('a&b') == sanitize('a_and_b')
