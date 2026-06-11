# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for AST-based DBT rule generation."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.codegen.dbt.rule_gen import (
    ExpansionStrategy,
    RuleAction,
    TranslationRule,
    generate_rules,
    summarize_rules,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _make_add(name: str, pragma: ExecModel = ExecModel.SCALAR) -> SemaBlock:
    body = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(
                SemaNodeKind.ADD,
                ty=SemaType.U32,
                children=(
                    SemaNode(
                        SemaNodeKind.INSTOPERAND,
                        ty=SemaType.B32,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='S'),
                            SemaNode(SemaNodeKind.LIT, lit_value='0'),
                        ),
                    ),
                    SemaNode(
                        SemaNodeKind.INSTOPERAND,
                        ty=SemaType.B32,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='S'),
                            SemaNode(SemaNodeKind.LIT, lit_value='1'),
                        ),
                    ),
                ),
            ),
        ),
    )
    return SemaBlock(name, pragma, body)


def _make_sub(name: str) -> SemaBlock:
    body = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(
                SemaNodeKind.SUB,
                ty=SemaType.U32,
                children=(
                    SemaNode(
                        SemaNodeKind.INSTOPERAND,
                        ty=SemaType.B32,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='S'),
                            SemaNode(SemaNodeKind.LIT, lit_value='0'),
                        ),
                    ),
                    SemaNode(
                        SemaNodeKind.INSTOPERAND,
                        ty=SemaType.B32,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='S'),
                            SemaNode(SemaNodeKind.LIT, lit_value='1'),
                        ),
                    ),
                ),
            ),
        ),
    )
    return SemaBlock(name, ExecModel.SCALAR, body)


def _make_stub(name: str) -> SemaBlock:
    return SemaBlock(name, ExecModel.UNKNOWN, SemaNode(SemaNodeKind.SEQ, children=()))


def _make_mfma(name: str) -> SemaBlock:
    body = SemaNode(
        SemaNodeKind.CALL,
        ty=SemaType.F32,
        call_name='mfma_compute',
        children=(SemaNode(SemaNodeKind.ID, id_name='mfma_compute'),),
    )
    return SemaBlock(name, ExecModel.VECTOR, body)


class TestGenerateRulesBasic:
    def test_identity(self):
        src = {'ADD': _make_add('ADD')}
        dst = {'ADD': _make_add('ADD')}
        rules = generate_rules('a', 'b', src, dst)
        assert len(rules) == 1
        assert rules[0].action == RuleAction.IDENTITY
        assert rules[0].dst_mnemonic == 'ADD'

    def test_substitute(self):
        src = {'ADD_OLD': _make_add('ADD_OLD')}
        dst = {'ADD_NEW': _make_add('ADD_NEW')}
        rules = generate_rules('a', 'b', src, dst)
        assert len(rules) == 1
        assert rules[0].action == RuleAction.SUBSTITUTE
        assert rules[0].dst_mnemonic == 'ADD_NEW'

    def test_no_match_lower(self):
        src = {'ADD': _make_add('ADD')}
        dst = {'SUB': _make_sub('SUB')}
        rules = generate_rules('a', 'b', src, dst)
        assert len(rules) == 1
        assert rules[0].action == RuleAction.LOWER

    def test_stub_expands(self):
        src = {'NOP': _make_stub('NOP')}
        dst = {'NOP': _make_stub('NOP')}
        rules = generate_rules('a', 'b', src, dst)
        assert len(rules) == 1
        assert rules[0].action == RuleAction.EXPAND
        assert rules[0].expansion == ExpansionStrategy.GENERIC

    def test_mfma_no_match_expands(self):
        src = {'V_MFMA_F32': _make_mfma('V_MFMA_F32')}
        dst = {'V_ADD_F32': _make_add('V_ADD_F32', ExecModel.VECTOR)}
        rules = generate_rules('a', 'b', src, dst)
        assert len(rules) == 1
        assert rules[0].action == RuleAction.EXPAND
        assert rules[0].expansion == ExpansionStrategy.MFMA_TO_WMMA

    def test_mixed_scenario(self):
        src = {
            'ADD': _make_add('ADD'),
            'SUB': _make_sub('SUB'),
            'ADD_OLD': _make_add('ADD_OLD'),
            'NOP': _make_stub('NOP'),
        }
        dst = {
            'ADD': _make_add('ADD'),
            'ADD_RENAMED': _make_add('ADD_RENAMED'),
        }
        rules = generate_rules('a', 'b', src, dst)
        summary = summarize_rules(rules)
        assert summary.total == 4
        assert summary.identity >= 1
        assert summary.substitute >= 1
        assert summary.expand >= 1


class TestGenerateRulesProperties:
    def test_identity_carries_properties(self):
        src = {'ADD': _make_add('ADD', ExecModel.VECTOR)}
        dst = {'ADD': _make_add('ADD', ExecModel.VECTOR)}
        rules = generate_rules('a', 'b', src, dst)
        from amdisa.sema_properties import InstructionProperty

        assert InstructionProperty.EXEC_MASKED in rules[0].src_properties
        assert InstructionProperty.EXEC_MASKED in rules[0].dst_properties


class TestSummarizeRules:
    def test_empty(self):
        s = summarize_rules([])
        assert s.total == 0

    def test_counts(self):
        rules = [
            TranslationRule('A', RuleAction.IDENTITY, 'A'),
            TranslationRule('B', RuleAction.SUBSTITUTE, 'B2'),
            TranslationRule('C', RuleAction.LOWER),
            TranslationRule(
                'D', RuleAction.EXPAND, expansion=ExpansionStrategy.GENERIC
            ),
        ]
        s = summarize_rules(rules)
        assert s.total == 4
        assert s.identity == 1
        assert s.substitute == 1
        assert s.lower == 1
        assert s.expand == 1


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlRuleGen:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml

        return parse_semantics_xml(SEMA_XML_PATH)

    def test_self_rules_all_identity_or_substitute(self, blocks):
        rules = generate_rules('cdna4', 'cdna4', blocks, blocks)
        s = summarize_rules(rules)
        assert s.total == len(blocks)
        assert s.lower == 0
        non_stub = sum(1 for b in blocks.values() if not b.is_empty)
        assert s.identity + s.substitute == non_stub

    def test_rules_have_properties(self, blocks):
        rules = generate_rules('cdna4', 'cdna4', blocks, blocks)
        vector_rules = [
            r
            for r in rules
            if r.action in (RuleAction.IDENTITY, RuleAction.SUBSTITUTE)
            and 'V_ADD_F32' == r.src_mnemonic
        ]
        if vector_rules:
            from amdisa.sema_properties import InstructionProperty

            assert InstructionProperty.EXEC_MASKED in vector_rules[0].src_properties


class TestMatrixExpandRules:
    def test_mfma_to_wmma_16x16(self):
        from amdisa.codegen.dbt.rule_gen import generate_matrix_expand_rules

        rules = generate_matrix_expand_rules(
            ['V_MFMA_F32_16X16X16_F16'],
            ['V_WMMA_F32_16X16X16_F16'],
        )
        assert len(rules) == 1
        r = rules[0]
        assert r.xor_byte_mask == 192
        assert r.range_start == 16
        assert r.range_end == 48
        assert r.dst_vgprs == 4

    def test_no_match_returns_empty(self):
        from amdisa.codegen.dbt.rule_gen import generate_matrix_expand_rules

        rules = generate_matrix_expand_rules(
            ['V_MFMA_F32_32X32X8_F16'],
            ['V_WMMA_I32_16X16X32_I8'],
        )
        assert len(rules) == 0

    def test_emit_header_compiles(self):
        from amdisa.codegen.dbt.rule_gen import (
            generate_matrix_expand_rules,
            emit_matrix_conversions_header,
        )

        rules = generate_matrix_expand_rules(
            ['V_MFMA_F32_16X16X16_F16'],
            ['V_WMMA_F32_16X16X16_F16'],
        )
        header = emit_matrix_conversions_header(rules)
        assert '#pragma once' in header
        assert 'kMatrixConversions' in header
        assert '192u' in header
