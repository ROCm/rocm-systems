# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for AST-based instruction property derivation."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_properties import (
    InstructionProperty,
    derive_properties,
    summarize_properties,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _id(name, ty=None):
    return SemaNode(SemaNodeKind.ID, id_name=name, ty=ty)


def _lit(val, ty=SemaType.U32):
    return SemaNode(SemaNodeKind.LIT, lit_value=val, ty=ty)


def _assign(lhs, rhs):
    return SemaNode(SemaNodeKind.ASSIGN, children=(lhs, rhs))


def _cast(inner, target):
    return SemaNode(SemaNodeKind.CAST, ty=target, cast_target=target, children=(inner,))


class TestDeriveEmpty:
    def test_empty_block(self):
        block = SemaBlock(
            'NOP', ExecModel.UNKNOWN, SemaNode(SemaNodeKind.SEQ, children=())
        )
        props = derive_properties(block)
        assert props == InstructionProperty.NONE


class TestDeriveExecMasked:
    def test_vector_pragma(self):
        body = _assign(_id('tmp'), _lit('0'))
        block = SemaBlock('V_NOP', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.EXEC_MASKED in props

    def test_scalar_not_masked(self):
        body = _assign(_id('tmp'), _lit('0'))
        block = SemaBlock('S_NOP', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.EXEC_MASKED not in props


class TestDeriveWritesScc:
    def test_scc_assign(self):
        body = _assign(_id('SCC', SemaType.U1), _lit('1', SemaType.U1))
        block = SemaBlock('S_CMP', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.WRITES_SCC in props

    def test_no_scc(self):
        body = _assign(_id('tmp'), _lit('0'))
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.WRITES_SCC not in props


class TestDeriveWritesVcc:
    def test_vcc_arrayderef(self):
        body = _assign(
            SemaNode(
                SemaNodeKind.ARRAYDEREF,
                ty=SemaType.U1,
                children=(
                    _id('VCC', SemaType.U64),
                    _id('laneId', SemaType.U32),
                ),
            ),
            _lit('1', SemaType.U1),
        )
        block = SemaBlock('V_CMP', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.WRITES_VCC in props


class TestDeriveWritesExec:
    def test_exec_assign(self):
        body = _assign(_id('EXEC', SemaType.U64), _lit('0', SemaType.U64))
        block = SemaBlock('S_WREXEC', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.WRITES_EXEC in props

    def test_exec_arrayderef(self):
        body = _assign(
            SemaNode(
                SemaNodeKind.ARRAYDEREF,
                ty=SemaType.U1,
                children=(
                    _id('EXEC', SemaType.U64),
                    _id('laneId', SemaType.U32),
                ),
            ),
            _lit('1', SemaType.U1),
        )
        block = SemaBlock('V_CMPX', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.WRITES_EXEC in props


class TestDeriveCrossLane:
    def test_permlane_call(self):
        body = SemaNode(
            SemaNodeKind.CALL,
            ty=SemaType.U32,
            call_name='v_permlane16',
            children=(_id('v_permlane16'), _id('src')),
        )
        block = SemaBlock('V_PERMLANE16', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.CROSS_LANE in props

    def test_ds_bpermute(self):
        body = SemaNode(
            SemaNodeKind.CALL,
            ty=SemaType.U32,
            call_name='ds_bpermute',
            children=(_id('ds_bpermute'), _id('addr'), _id('src')),
        )
        block = SemaBlock('DS_BPERMUTE', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.CROSS_LANE in props
        assert (
            InstructionProperty.DS_PERMUTE not in props
        )  # ds_bpermute != ds_permute flag


class TestDeriveMatrix:
    def test_mfma(self):
        body = SemaNode(
            SemaNodeKind.CALL,
            ty=SemaType.F32,
            call_name='mfma_compute',
            children=(_id('mfma_compute'), _id('a'), _id('b'), _id('c')),
        )
        block = SemaBlock('V_MFMA', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_MATRIX in props


class TestDeriveMemory:
    def test_mem_arrayderef(self):
        body = _assign(
            _id('data'),
            SemaNode(
                SemaNodeKind.ARRAYDEREF,
                ty=SemaType.B32,
                children=(_id('MEM'), _id('addr')),
            ),
        )
        block = SemaBlock('LOAD', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_MEMORY in props

    def test_addr_calc_call(self):
        body = SemaNode(
            SemaNodeKind.CALL,
            ty=SemaType.U32,
            call_name='CalcBufferAddr',
            children=(_id('CalcBufferAddr'), _id('base')),
        )
        block = SemaBlock('BUFFER', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_MEMORY in props


class TestDeriveBranch:
    def test_branch_pragma(self):
        body = SemaNode(
            SemaNodeKind.CALL,
            call_name='branch',
            children=(_id('branch'), _id('target')),
        )
        block = SemaBlock('S_BRANCH', ExecModel.BRANCH, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_BRANCH in props
        assert InstructionProperty.IGNORES_EXEC in props


class TestDeriveBarrier:
    def test_barrier(self):
        body = SemaNode(
            SemaNodeKind.CALL, call_name='barrier', children=(_id('barrier'),)
        )
        block = SemaBlock('S_BARRIER', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_BARRIER in props


class TestDeriveWaitcnt:
    def test_waitcnt(self):
        body = SemaNode(
            SemaNodeKind.CALL, call_name='waitcnt', children=(_id('waitcnt'),)
        )
        block = SemaBlock('S_WAITCNT', ExecModel.SCALAR, body)
        props = derive_properties(block)
        assert InstructionProperty.IS_WAITCNT in props


class TestDeriveReadsVcc:
    def test_vcc_read(self):
        body = _assign(
            _id('tmp'),
            SemaNode(
                SemaNodeKind.ARRAYDEREF,
                ty=SemaType.U1,
                children=(_id('VCC', SemaType.U64), _id('laneId')),
            ),
        )
        block = SemaBlock('V_CNDMASK', ExecModel.VECTOR, body)
        props = derive_properties(block)
        assert InstructionProperty.READS_VCC in props


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlProperties:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml

        return parse_semantics_xml(SEMA_XML_PATH)

    def test_summary_counts(self, blocks):
        s = summarize_properties(blocks)
        assert s.total == 1244
        assert s.exec_masked > 900
        assert s.writes_scc > 100  # XML only models explicit SCC writes
        assert s.is_memory > 100
        assert s.is_branch > 5

    def test_s_add_co_u32_writes_scc(self, blocks):
        props = derive_properties(blocks['S_ADD_CO_U32'])
        assert InstructionProperty.WRITES_SCC in props
        assert InstructionProperty.EXEC_MASKED not in props

    def test_v_add_f32_exec_masked(self, blocks):
        props = derive_properties(blocks['V_ADD_F32'])
        assert InstructionProperty.EXEC_MASKED in props
        assert InstructionProperty.WRITES_SCC not in props

    def test_v_cmp_eq_f32_is_vector(self, blocks):
        b = blocks.get('V_CMP_EQ_F32')
        if b and not b.is_empty:
            props = derive_properties(b)
            assert InstructionProperty.EXEC_MASKED in props
            # XML writes VCC via INSTOPERAND(D,0), not literal ID('VCC'),
            # so WRITES_VCC is not detected from the XML AST alone.
            # It would be detected from the derived AST (Flow B) which
            # uses explicit VCC arrayderef.

    def test_ds_load_is_memory(self, blocks):
        props = derive_properties(blocks['DS_LOAD_B32'])
        assert InstructionProperty.IS_MEMORY in props

    def test_s_branch_is_branch(self, blocks):
        props = derive_properties(blocks['S_BRANCH'])
        assert InstructionProperty.IS_BRANCH in props
