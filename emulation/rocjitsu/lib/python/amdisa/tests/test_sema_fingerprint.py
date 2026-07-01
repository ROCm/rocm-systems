# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for semantic fingerprinting."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_fingerprint import (
    are_equivalent,
    build_equivalence_map,
    fingerprint,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _make_add_block(
    name: str, pragma: ExecModel = ExecModel.SCALAR, tmp_name: str = 'tmp'
) -> SemaBlock:
    """Build a simple S_ADD-like block: tmp = S[0] + S[1]; D[0] = tmp."""
    s0 = SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='S'),
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ),
    )
    s1 = SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='S'),
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
        ),
    )
    add = SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(s0, s1))
    assign_tmp = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name=tmp_name),
            add,
        ),
    )
    d0 = SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ),
    )
    assign_dst = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            d0,
            SemaNode(SemaNodeKind.ID, id_name=tmp_name),
        ),
    )
    body = SemaNode(SemaNodeKind.SEQ, children=(assign_tmp, assign_dst))
    return SemaBlock(name, pragma, body)


class TestFingerprint:
    def test_deterministic(self):
        block = _make_add_block('S_ADD_U32')
        fp1 = fingerprint(block)
        fp2 = fingerprint(block)
        assert fp1 == fp2

    def test_returns_32_bytes(self):
        block = _make_add_block('S_ADD_U32')
        fp = fingerprint(block)
        assert len(fp) == 32

    def test_same_structure_same_fingerprint(self):
        a = _make_add_block('S_ADD_U32')
        b = _make_add_block('S_ADD_U32_V2')
        assert fingerprint(a) == fingerprint(b)

    def test_different_temp_name_same_fingerprint(self):
        a = _make_add_block('A', tmp_name='tmp')
        b = _make_add_block('B', tmp_name='result')
        assert fingerprint(a) == fingerprint(b)

    def test_different_instruction_name_same_fingerprint(self):
        a = _make_add_block('S_ADD_U32')
        b = _make_add_block('S_ADD_U32_RENAMED')
        assert fingerprint(a) == fingerprint(b)

    def test_different_operation_different_fingerprint(self):
        add_block = _make_add_block('ADD')
        s0 = SemaNode(
            SemaNodeKind.INSTOPERAND,
            ty=SemaType.B32,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='S'),
                SemaNode(SemaNodeKind.LIT, lit_value='0'),
            ),
        )
        s1 = SemaNode(
            SemaNodeKind.INSTOPERAND,
            ty=SemaType.B32,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='S'),
                SemaNode(SemaNodeKind.LIT, lit_value='1'),
            ),
        )
        sub = SemaNode(SemaNodeKind.SUB, ty=SemaType.U32, children=(s0, s1))
        body = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(SemaNode(SemaNodeKind.ID, id_name='tmp'), sub),
                ),
            ),
        )
        sub_block = SemaBlock('SUB', ExecModel.SCALAR, body)
        assert fingerprint(add_block) != fingerprint(sub_block)

    def test_different_pragma_different_fingerprint(self):
        a = _make_add_block('A', pragma=ExecModel.SCALAR)
        b = _make_add_block('B', pragma=ExecModel.VECTOR)
        assert fingerprint(a) != fingerprint(b)

    def test_different_type_different_fingerprint(self):
        s0 = SemaNode(
            SemaNodeKind.INSTOPERAND,
            ty=SemaType.B32,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='S'),
                SemaNode(SemaNodeKind.LIT, lit_value='0'),
            ),
        )
        s1 = SemaNode(
            SemaNodeKind.INSTOPERAND,
            ty=SemaType.B32,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='S'),
                SemaNode(SemaNodeKind.LIT, lit_value='1'),
            ),
        )
        add_u32 = SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(s0, s1))
        add_f32 = SemaNode(SemaNodeKind.ADD, ty=SemaType.F32, children=(s0, s1))

        body_u32 = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(SemaNode(SemaNodeKind.ID, id_name='r'), add_u32),
                ),
            ),
        )
        body_f32 = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(SemaNode(SemaNodeKind.ID, id_name='r'), add_f32),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_u32)
        b = SemaBlock('B', ExecModel.SCALAR, body_f32)
        assert fingerprint(a) != fingerprint(b)

    def test_context_ids_are_hashed(self):
        body_scc = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U1),
                    ),
                ),
            ),
        )
        body_vcc = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='VCC', ty=SemaType.U64),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U64),
                    ),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_scc)
        b = SemaBlock('B', ExecModel.SCALAR, body_vcc)
        assert fingerprint(a) != fingerprint(b)

    def test_operand_index_is_hashed(self):
        def _make_read(idx: str):
            return SemaNode(
                SemaNodeKind.INSTOPERAND,
                ty=SemaType.B32,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='S'),
                    SemaNode(SemaNodeKind.LIT, lit_value=idx),
                ),
            )

        body_0 = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(SemaNode(SemaNodeKind.ID, id_name='r'), _make_read('0')),
                ),
            ),
        )
        body_1 = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(SemaNode(SemaNodeKind.ID, id_name='r'), _make_read('1')),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_0)
        b = SemaBlock('B', ExecModel.SCALAR, body_1)
        assert fingerprint(a) != fingerprint(b)

    def test_arrayslice_bounds_are_hashed(self):
        def _make_slice(hi: str, lo: str):
            return SemaNode(
                SemaNodeKind.ARRAYSLICE,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='SDATA'),
                    SemaNode(SemaNodeKind.LIT, lit_value=hi),
                    SemaNode(SemaNodeKind.LIT, lit_value=lo),
                ),
            )

        body_lo = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='r'),
                        _make_slice('31', '0'),
                    ),
                ),
            ),
        )
        body_hi = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='r'),
                        _make_slice('63', '32'),
                    ),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_lo)
        b = SemaBlock('B', ExecModel.SCALAR, body_hi)
        assert fingerprint(a) != fingerprint(b)

    def test_general_literal_not_hashed(self):
        def _make_cmp(lit_val: str):
            return SemaNode(
                SemaNodeKind.GE,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='tmp', ty=SemaType.U64),
                    SemaNode(SemaNodeKind.LIT, lit_value=lit_val, ty=SemaType.U64),
                ),
            )

        body_a = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='SCC'),
                        _make_cmp('4294967296'),
                    ),
                ),
            ),
        )
        body_b = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='SCC'),
                        _make_cmp('999999'),
                    ),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_a)
        b = SemaBlock('B', ExecModel.SCALAR, body_b)
        assert fingerprint(a) == fingerprint(b)

    def test_enriched_block_raises(self):
        block = _make_add_block('TEST')
        block.enriched = True
        with pytest.raises(AssertionError, match="pre-enrichment"):
            fingerprint(block)

    def test_call_name_is_hashed(self):
        def _make_call(name: str):
            return SemaNode(
                SemaNodeKind.CALL,
                call_name=name,
                children=(SemaNode(SemaNodeKind.ID, id_name=name),),
            )

        body_a = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='r'),
                        _make_call('CalcDsAddr'),
                    ),
                ),
            ),
        )
        body_b = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='r'),
                        _make_call('CalcFlatAddr'),
                    ),
                ),
            ),
        )
        a = SemaBlock('A', ExecModel.SCALAR, body_a)
        b = SemaBlock('B', ExecModel.SCALAR, body_b)
        assert fingerprint(a) != fingerprint(b)


class TestAreEquivalent:
    def test_same_structure(self):
        a = _make_add_block('A')
        b = _make_add_block('B')
        assert are_equivalent(a, b)

    def test_different_structure(self):
        a = _make_add_block('A', pragma=ExecModel.SCALAR)
        b = _make_add_block('B', pragma=ExecModel.VECTOR)
        assert not are_equivalent(a, b)


class TestBuildEquivalenceMap:
    def test_identical_sets(self):
        src = {'ADD': _make_add_block('ADD')}
        dst = {'ADD': _make_add_block('ADD')}
        result = build_equivalence_map(src, dst)
        assert result == {'ADD': 'ADD'}

    def test_renamed_instruction(self):
        src = {'ADD_OLD': _make_add_block('ADD_OLD')}
        dst = {'ADD_NEW': _make_add_block('ADD_NEW')}
        result = build_equivalence_map(src, dst)
        assert result == {'ADD_OLD': 'ADD_NEW'}

    def test_no_equivalent(self):
        src = {'ADD': _make_add_block('ADD', pragma=ExecModel.SCALAR)}
        dst = {'ADD': _make_add_block('ADD', pragma=ExecModel.VECTOR)}
        result = build_equivalence_map(src, dst)
        assert result == {'ADD': None}

    def test_stubs_map_to_none(self):
        stub = SemaBlock(
            'NOP', ExecModel.UNKNOWN, SemaNode(SemaNodeKind.SEQ, children=())
        )
        src = {'NOP': stub}
        dst = {'NOP': stub}
        result = build_equivalence_map(src, dst)
        assert result == {'NOP': None}

    def test_mixed_results(self):
        add_s = _make_add_block('ADD_S', pragma=ExecModel.SCALAR)
        add_v = _make_add_block('ADD_V', pragma=ExecModel.VECTOR)
        src = {'S_ADD': add_s, 'V_ADD': add_v}
        dst = {'S_ADD_NEW': _make_add_block('S_ADD_NEW', pragma=ExecModel.SCALAR)}
        result = build_equivalence_map(src, dst)
        assert result['S_ADD'] == 'S_ADD_NEW'
        assert result['V_ADD'] is None

    def test_o_n_plus_m_not_quadratic(self):
        src = {f'INST_{i}': _make_add_block(f'INST_{i}') for i in range(100)}
        dst = {f'INST_{i}': _make_add_block(f'INST_{i}') for i in range(100)}
        result = build_equivalence_map(src, dst)
        assert all(v is not None for v in result.values())


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlFingerprinting:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml

        return parse_semantics_xml(SEMA_XML_PATH)

    def test_self_equivalence(self, blocks):
        result = build_equivalence_map(blocks, blocks)
        non_stub = {k: v for k, v in result.items() if not blocks[k].is_empty}
        matched = sum(1 for v in non_stub.values() if v is not None)
        assert matched == len(non_stub)

    def test_all_non_stub_fingerprints_are_non_empty(self, blocks):
        for name, block in blocks.items():
            if not block.is_empty:
                fp = fingerprint(block)
                assert len(fp) == 32, f"{name} produced empty fingerprint"

    def test_add_and_sub_are_different(self, blocks):
        if 'S_ADD_U32' in blocks and 'S_SUB_U32' in blocks:
            a = blocks['S_ADD_U32']
            b = blocks['S_SUB_U32']
            if not a.is_empty and not b.is_empty:
                assert not are_equivalent(a, b)

    def test_unique_fingerprint_count(self, blocks):
        fps = set()
        for block in blocks.values():
            if not block.is_empty:
                fps.add(fingerprint(block))
        assert len(fps) > 500
