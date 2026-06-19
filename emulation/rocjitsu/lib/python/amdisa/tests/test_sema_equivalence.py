# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for AST-based semantic equivalence for DBT legalization."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.codegen.dbt.sema_equivalence import (
    SemaEquivalence,
    build_sema_equivalences,
    merge_equivalences_into_union_find,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _make_add_block(name: str, pragma: ExecModel = ExecModel.SCALAR) -> SemaBlock:
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
    body = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(s0, s1)),
        ),
    )
    return SemaBlock(name, pragma, body)


def _make_sub_block(name: str) -> SemaBlock:
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
    body = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(SemaNodeKind.SUB, ty=SemaType.U32, children=(s0, s1)),
        ),
    )
    return SemaBlock(name, ExecModel.SCALAR, body)


def _make_stub(name: str) -> SemaBlock:
    return SemaBlock(name, ExecModel.UNKNOWN, SemaNode(SemaNodeKind.SEQ, children=()))


class TestBuildSemaEquivalences:
    def test_identical_isas(self):
        src = {'ADD': _make_add_block('ADD'), 'SUB': _make_sub_block('SUB')}
        dst = {'ADD': _make_add_block('ADD'), 'SUB': _make_sub_block('SUB')}
        result = build_sema_equivalences('isa_a', 'isa_b', src, dst)
        assert result.identity_count == 2
        assert result.rename_count == 0
        assert result.no_match_count == 0

    def test_renamed_instruction(self):
        src = {'ADD_OLD': _make_add_block('ADD_OLD')}
        dst = {'ADD_NEW': _make_add_block('ADD_NEW')}
        result = build_sema_equivalences('isa_a', 'isa_b', src, dst)
        assert result.rename_count == 1
        assert result.equivalences['ADD_OLD'] == 'ADD_NEW'

    def test_no_match(self):
        src = {'ADD': _make_add_block('ADD')}
        dst = {'SUB': _make_sub_block('SUB')}
        result = build_sema_equivalences('isa_a', 'isa_b', src, dst)
        assert result.no_match_count == 1
        assert result.equivalences['ADD'] is None

    def test_stubs_counted(self):
        src = {'NOP': _make_stub('NOP'), 'ADD': _make_add_block('ADD')}
        dst = {'ADD': _make_add_block('ADD')}
        result = build_sema_equivalences('isa_a', 'isa_b', src, dst)
        assert result.stub_count == 1
        assert result.identity_count == 1

    def test_mixed_scenario(self):
        src = {
            'ADD': _make_add_block('ADD'),
            'SUB': _make_sub_block('SUB'),
            'ADD_OLD': _make_add_block('ADD_OLD'),
            'UNIQUE': _make_add_block('UNIQUE', ExecModel.VECTOR),
            'NOP': _make_stub('NOP'),
        }
        dst = {
            'ADD': _make_add_block('ADD'),
            'SUB': _make_sub_block('SUB'),
            'ADD_RENAMED': _make_add_block('ADD_RENAMED'),
        }
        result = build_sema_equivalences('isa_a', 'isa_b', src, dst)
        assert result.identity_count == 2  # ADD, SUB
        assert result.rename_count == 1  # ADD_OLD -> ADD or ADD_RENAMED
        assert result.no_match_count == 1  # UNIQUE (vector, no vector target)
        assert result.stub_count == 1  # NOP

    def test_empty_isas(self):
        result = build_sema_equivalences('a', 'b', {}, {})
        assert result.identity_count == 0
        assert result.rename_count == 0
        assert result.no_match_count == 0

    def test_isa_names_stored(self):
        result = build_sema_equivalences('cdna4', 'rdna4', {}, {})
        assert result.src_isa == 'cdna4'
        assert result.dst_isa == 'rdna4'


class TestMergeIntoUnionFind:
    def test_merges_renamed_equivalences(self):
        equiv = SemaEquivalence(
            src_isa='a',
            dst_isa='b',
            equivalences={'OLD': 'NEW', 'SAME': 'SAME', 'GONE': None},
        )

        class FakeUF:
            def __init__(self):
                self.merged = []

            def union(self, a, b):
                self.merged.append((a, b))

        uf = FakeUF()
        canon_to_ids = {'OLD': [10], 'NEW': [20], 'SAME': [30], 'GONE': [40]}
        count = merge_equivalences_into_union_find(equiv, canon_to_ids, uf)
        assert count == 1
        assert uf.merged == [(10, 20)]

    def test_skips_missing_ids(self):
        equiv = SemaEquivalence(
            src_isa='a',
            dst_isa='b',
            equivalences={'OLD': 'NEW'},
        )

        class FakeUF:
            def __init__(self):
                self.merged = []

            def union(self, a, b):
                self.merged.append((a, b))

        uf = FakeUF()
        count = merge_equivalences_into_union_find(equiv, {}, uf)
        assert count == 0


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlEquivalence:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml

        return parse_semantics_xml(SEMA_XML_PATH)

    def test_self_equivalence_covers_all(self, blocks):
        result = build_sema_equivalences('cdna4', 'cdna4', blocks, blocks)
        non_stub = sum(1 for b in blocks.values() if not b.is_empty)
        matched = result.identity_count + result.rename_count
        assert matched == non_stub
        assert result.no_match_count == 0
        assert result.stub_count == 81
        # Some instructions have identical semantics under different mnemonics
        # (e.g., S_BUFFER_LOAD_U8 and S_LOAD_U8), so rename_count > 0 is expected.
        assert result.identity_count > 1000
        assert result.rename_count > 0

    def test_counts_sum_to_total(self, blocks):
        result = build_sema_equivalences('cdna4', 'cdna4', blocks, blocks)
        total = (
            result.identity_count
            + result.rename_count
            + result.no_match_count
            + result.stub_count
        )
        assert total == len(blocks)
