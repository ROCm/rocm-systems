# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the semantics parser."""

import os
import textwrap
import tempfile

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_parser import parse_semantics_xml

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _write_xml(content: str) -> str:
    """Write a minimal semantics file to a temp file and return the path."""
    xml = textwrap.dedent(f"""\
    <Spec>
      <Document>
        <SchemaVersion>1.0.0</SchemaVersion>
      </Document>
      <ISASemanticsExtension>
        <Instructions>
          {content}
        </Instructions>
      </ISASemanticsExtension>
    </Spec>
    """)
    fd, path = tempfile.mkstemp(suffix='.xml')
    os.write(fd, xml.encode())
    os.close(fd)
    return path


class TestParserSyntheticXml:
    """Tests using small hand-crafted XML snippets."""

    def test_empty_stub(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>S_NOP</InstructionName>
                <InstructionSemantics>
                    <op type=""/>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        assert 'S_NOP' in blocks
        assert blocks['S_NOP'].is_empty
        assert blocks['S_NOP'].pragma == ExecModel.UNKNOWN

    def test_scalar_pragma(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>S_ADD_U32</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="scalar"><ty><t base="S"/></ty></lit>
                        <op type=":seq">
                            <op type="=">
                                <op type=".instoperand">
                                    <id val="D"><ty><t base="B" size="32"/></ty></id>
                                    <lit val="0"><ty><t base="I" size="32"/></ty></lit>
                                </op>
                                <op type="+">
                                    <op type=".instoperand">
                                        <id val="S"><ty><t base="B" size="32"/></ty></id>
                                        <lit val="0"><ty><t base="I" size="32"/></ty></lit>
                                    </op>
                                    <op type=".instoperand">
                                        <id val="S"><ty><t base="B" size="32"/></ty></id>
                                        <lit val="1"><ty><t base="I" size="32"/></ty></lit>
                                    </op>
                                    <ty><t base="B" size="32"/></ty>
                                </op>
                            </op>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['S_ADD_U32']
        assert b.pragma == ExecModel.SCALAR
        assert not b.is_empty
        assert b.body.kind == SemaNodeKind.SEQ

    def test_vector_pragma(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>V_ADD_F32</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="vector"><ty><t base="S"/></ty></lit>
                        <op type="=">
                            <op type=".instoperand">
                                <id val="D"/>
                                <lit val="0"/>
                            </op>
                            <op type="+">
                                <op type=".instoperand">
                                    <id val="S"/>
                                    <lit val="0"/>
                                </op>
                                <op type=".instoperand">
                                    <id val="S"/>
                                    <lit val="1"/>
                                </op>
                            </op>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['V_ADD_F32']
        assert b.pragma == ExecModel.VECTOR
        assert b.body.kind == SemaNodeKind.ASSIGN

    def test_cast_target_extraction(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>TEST_CAST</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="scalar"><ty><t base="S"/></ty></lit>
                        <op type="=">
                            <id val="tmp"/>
                            <op type=".cast">
                                <id val="src0"><ty><t base="B" size="32"/></ty></id>
                                <type><t base="U" size="64"/></type>
                                <ty><t base="U" size="64"/></ty>
                            </op>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['TEST_CAST']
        assign = b.body
        assert assign.kind == SemaNodeKind.ASSIGN
        cast_node = assign.children[1]
        assert cast_node.kind == SemaNodeKind.CAST
        assert cast_node.cast_target == SemaType('U', 64)

    def test_call_name_extraction(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>TEST_CALL</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="scalar"><ty><t base="S"/></ty></lit>
                        <op type="=">
                            <id val="addr"/>
                            <op type=".call">
                                <id val="CalcBufferAddr"><ty><lambda><ret><t base="B" size="64"/></ret></lambda></ty></id>
                                <id val="base"><ty><t base="B" size="32"/></ty></id>
                            </op>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['TEST_CALL']
        assign = b.body
        call_node = assign.children[1]
        assert call_node.kind == SemaNodeKind.CALL
        assert call_node.call_name == 'CalcBufferAddr'

    def test_laneid_normalization(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>TEST_LANEID</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="vector"><ty><t base="S"/></ty></lit>
                        <op type="=">
                            <id val="tmp"/>
                            <id val="laneID"><ty><t base="U" size="32"/></ty></id>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['TEST_LANEID']
        assign = b.body
        id_node = assign.children[1]
        assert id_node.kind == SemaNodeKind.ID
        assert id_node.id_name == 'laneId'

    def test_schema_version_too_high_raises(self):
        xml = textwrap.dedent("""\
        <Spec>
          <Document>
            <SchemaVersion>2.0.0</SchemaVersion>
          </Document>
          <ISASemanticsExtension>
            <Instructions/>
          </ISASemanticsExtension>
        </Spec>
        """)
        fd, path = tempfile.mkstemp(suffix='.xml')
        os.write(fd, xml.encode())
        os.close(fd)
        with pytest.raises(ValueError, match="Unsupported schema version"):
            parse_semantics_xml(path)
        os.unlink(path)

    def test_multiple_instructions(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>INST_A</InstructionName>
                <InstructionSemantics><op type=""/></InstructionSemantics>
            </Instruction>
            <Instruction>
                <InstructionName>INST_B</InstructionName>
                <InstructionSemantics><op type=""/></InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        assert len(blocks) == 2
        assert 'INST_A' in blocks
        assert 'INST_B' in blocks

    def test_statement_nodes_have_no_type(self):
        path = _write_xml("""
            <Instruction>
                <InstructionName>TEST_STMT_TY</InstructionName>
                <InstructionSemantics>
                    <op type=":pragma">
                        <lit val="scalar"><ty><t base="S"/></ty></lit>
                        <op type=":seq">
                            <op type="=">
                                <id val="tmp"/>
                                <lit val="0"><ty><t base="U" size="32"/></ty></lit>
                                <ty><t base="U" size="32"/></ty>
                            </op>
                            <ty><t base="U" size="32"/></ty>
                        </op>
                    </op>
                </InstructionSemantics>
            </Instruction>
        """)
        blocks = parse_semantics_xml(path)
        os.unlink(path)
        b = blocks['TEST_STMT_TY']
        assert b.body.ty is None
        assert b.body.children[0].ty is None


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestParserSemaXml:
    """Tests against a real semantics file."""

    @pytest.fixture(scope='class')
    def blocks(self):
        return parse_semantics_xml(SEMA_XML_PATH)

    def test_total_instruction_count(self, blocks):
        assert len(blocks) == 1325

    def test_non_stub_count(self, blocks):
        non_stubs = sum(1 for b in blocks.values() if not b.is_empty)
        assert non_stubs == 1244

    def test_stub_count(self, blocks):
        stubs = sum(1 for b in blocks.values() if b.is_empty)
        assert stubs == 81

    def test_pragma_distribution(self, blocks):
        from collections import Counter
        pragmas = Counter(b.pragma for b in blocks.values())
        assert pragmas[ExecModel.VECTOR] == 931
        assert pragmas[ExecModel.SCALAR] == 301
        assert pragmas[ExecModel.BRANCH] == 12
        assert pragmas[ExecModel.UNKNOWN] == 81

    def test_s_add_co_u32_structure(self, blocks):
        b = blocks['S_ADD_CO_U32']
        assert b.pragma == ExecModel.SCALAR
        assert b.body.kind == SemaNodeKind.SEQ
        assert len(b.body.children) >= 3
        first_assign = b.body.children[0]
        assert first_assign.kind == SemaNodeKind.ASSIGN

    def test_v_fma_f32_structure(self, blocks):
        b = blocks['V_FMA_F32']
        assert b.pragma == ExecModel.VECTOR
        assert b.body.kind == SemaNodeKind.ASSIGN
        rhs = b.body.children[1]
        assert rhs.kind == SemaNodeKind.FMA

    def test_v_cmp_eq_f32_structure(self, blocks):
        b = blocks['V_CMP_EQ_F32']
        assert b.pragma == ExecModel.VECTOR
        assert b.body.kind == SemaNodeKind.SEQ
        first = b.body.children[0]
        assert first.kind == SemaNodeKind.ASSIGN
        lhs = first.children[0]
        assert lhs.kind == SemaNodeKind.ARRAYDEREF

    def test_ds_load_b32_has_call(self, blocks):
        b = blocks['DS_LOAD_B32']
        assert b.pragma == ExecModel.VECTOR
        all_nodes = list(b.body.walk())
        call_nodes = [n for n in all_nodes if n.kind == SemaNodeKind.CALL]
        assert len(call_nodes) >= 1
        assert any(n.call_name == 'CalcDsAddr' for n in call_nodes)

    def test_s_load_b32_has_call(self, blocks):
        b = blocks['S_LOAD_B32']
        assert b.pragma == ExecModel.SCALAR
        all_nodes = list(b.body.walk())
        call_nodes = [n for n in all_nodes if n.kind == SemaNodeKind.CALL]
        assert any(n.call_name == 'CalcScalarGlobalAddr' for n in call_nodes)

    def test_s_endpgm_is_stub(self, blocks):
        b = blocks['S_ENDPGM']
        assert b.is_empty
        assert b.pragma == ExecModel.UNKNOWN

    def test_s_branch_is_branch_pragma(self, blocks):
        b = blocks['S_BRANCH']
        assert b.pragma == ExecModel.BRANCH

    def test_laneid_normalized(self, blocks):
        all_ids = set()
        for b in blocks.values():
            for n in b.body.walk():
                if n.kind == SemaNodeKind.ID and n.id_name:
                    all_ids.add(n.id_name)
        assert 'laneId' in all_ids
        assert 'laneID' not in all_ids

    def test_no_stubs_have_non_empty_body(self, blocks):
        for name, b in blocks.items():
            if b.pragma == ExecModel.UNKNOWN and b.is_empty:
                assert len(b.body.children) == 0, f"{name} is stub but has children"

    def test_all_nodes_have_valid_kind(self, blocks):
        for name, b in blocks.items():
            for n in b.body.walk():
                assert isinstance(n.kind, SemaNodeKind), (
                    f"{name}: node has invalid kind {n.kind}"
                )

    def test_walk_covers_all_nodes(self, blocks):
        b = blocks['S_ADD_CO_U32']
        nodes = list(b.body.walk())
        assert len(nodes) > 10
