# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for SemaAST enrichment and the composable pipeline."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_enrich import (
    build_sema_block,
    enrich_block,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _src(idx: int) -> SemaNode:
    return SemaNode(SemaNodeKind.INSTOPERAND, ty=SemaType.B32, children=(
        SemaNode(SemaNodeKind.ID, id_name='S'),
        SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
    ))


def _dst(idx: int) -> SemaNode:
    return SemaNode(SemaNodeKind.INSTOPERAND, ty=SemaType.B32, children=(
        SemaNode(SemaNodeKind.ID, id_name='D'),
        SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
    ))


def _cast(inner: SemaNode, target: SemaType) -> SemaNode:
    return SemaNode(SemaNodeKind.CAST, ty=target, cast_target=target,
                    children=(inner,))


class TestEnrichBlock:
    def test_sets_enriched_flag(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                _cast(_dst(0), SemaType.U32),
                _cast(_src(0), SemaType.U32),
            )),
        ))
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        assert not block.enriched
        enriched = enrich_block(block)
        assert enriched.enriched

    def test_does_not_mutate_original(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                _cast(_dst(0), SemaType.U32),
                _cast(_src(0), SemaType.U32),
            )),
        ))
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        enrich_block(block)
        assert not block.enriched

    def test_empty_block_stays_empty(self):
        block = SemaBlock('NOP', ExecModel.UNKNOWN,
                          SemaNode(SemaNodeKind.SEQ, children=()))
        enriched = enrich_block(block)
        assert enriched.is_empty
        assert enriched.enriched


class TestRtnAtomicFix:
    def _make_ds_atomic(self, name: str) -> SemaBlock:
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='addr'),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.ADD_ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='mem_val'),
                SemaNode(SemaNodeKind.ID, id_name='data'),
            )),
            SemaNode(SemaNodeKind.ASSIGN, children=(
                _cast(SemaNode(SemaNodeKind.ID, id_name='RETURN_DATA'),
                      SemaType.U32),
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
            )),
        ))
        return SemaBlock(name, ExecModel.VECTOR, body)

    def test_non_rtn_strips_return_data(self):
        block = self._make_ds_atomic('DS_ADD_U32')
        enriched = enrich_block(block)
        for c in enriched.body.children:
            if c.kind == SemaNodeKind.ASSIGN:
                lhs = c.children[0]
                while lhs.kind == SemaNodeKind.CAST and lhs.children:
                    lhs = lhs.children[0]
                assert not (lhs.kind == SemaNodeKind.ID
                            and lhs.id_name == 'RETURN_DATA'), \
                    "RETURN_DATA should be stripped from non-RTN"

    def test_rtn_keeps_return_data(self):
        block = self._make_ds_atomic('DS_ADD_RTN_U32')
        enriched = enrich_block(block)
        has_return = any(
            c.kind == SemaNodeKind.ASSIGN
            and c.children[0].kind == SemaNodeKind.CAST
            for c in enriched.body.children
        )
        assert has_return

    def test_non_atomic_untouched(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                _cast(SemaNode(SemaNodeKind.ID, id_name='RETURN_DATA'),
                      SemaType.U32),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
        ))
        block = SemaBlock('DS_LOAD_B32', ExecModel.VECTOR, body)
        enriched = enrich_block(block)
        assert len(enriched.body.children) == 1


class TestVop3Modifiers:
    def _make_vop3_add(self) -> SemaBlock:
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.F32),
            SemaNode(SemaNodeKind.ADD, ty=SemaType.F32, children=(
                _cast(_src(0), SemaType.F32),
                _cast(_src(1), SemaType.F32),
            )),
        ))
        return SemaBlock('V_ADD_F32', ExecModel.VECTOR, body)

    def test_no_fields_no_modifiers(self):
        block = self._make_vop3_add()
        enriched = enrich_block(block, enc_field_names=frozenset())
        all_calls = [n for n in enriched.body.walk()
                     if n.kind == SemaNodeKind.CALL]
        assert len(all_calls) == 0

    def test_neg_abs_wraps_sources(self):
        block = self._make_vop3_add()
        enriched = enrich_block(block,
                                enc_field_names=frozenset({'neg', 'abs'}))
        src_mod_calls = [n for n in enriched.body.walk()
                         if n.kind == SemaNodeKind.CALL
                         and n.call_name == 'apply_src_mod']
        assert len(src_mod_calls) == 2

    def test_clamp_omod_wraps_dest(self):
        block = self._make_vop3_add()
        enriched = enrich_block(block,
                                enc_field_names=frozenset({'clamp', 'omod'}))
        clamp_calls = [n for n in enriched.body.walk()
                       if n.kind == SemaNodeKind.CALL
                       and n.call_name == 'apply_clamp']
        omod_calls = [n for n in enriched.body.walk()
                      if n.kind == SemaNodeKind.CALL
                      and n.call_name == 'apply_omod']
        assert len(clamp_calls) == 1
        assert len(omod_calls) == 1

    def test_all_modifiers(self):
        block = self._make_vop3_add()
        enriched = enrich_block(
            block,
            enc_field_names=frozenset({'neg', 'abs', 'clamp', 'omod'}),
        )
        call_names = [n.call_name for n in enriched.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'apply_src_mod' in call_names
        assert 'apply_clamp' in call_names
        assert 'apply_omod' in call_names


class TestBuildSemaBlock:
    def test_none_input_returns_none(self):
        result = build_sema_block('TEST', None)
        assert result is None

    def test_empty_block_returns_none(self):
        empty = SemaBlock('NOP', ExecModel.UNKNOWN,
                          SemaNode(SemaNodeKind.SEQ, children=()))
        result = build_sema_block('NOP', empty)
        assert result is None

    def test_valid_block_returns_enriched(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.U32),
            _cast(_src(0), SemaType.U32),
        ))
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        result = build_sema_block('S_MOV', block)
        assert result is not None
        assert result.enriched

    def test_with_enc_fields(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.F32),
            SemaNode(SemaNodeKind.ADD, ty=SemaType.F32, children=(
                _cast(_src(0), SemaType.F32),
                _cast(_src(1), SemaType.F32),
            )),
        ))
        block = SemaBlock('V_ADD_F32', ExecModel.VECTOR, body)
        result = build_sema_block(
            'V_ADD_F32', block,
            enc_field_names=frozenset({'neg', 'abs', 'clamp'}),
        )
        assert result is not None
        call_names = [n.call_name for n in result.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'apply_src_mod' in call_names
        assert 'apply_clamp' in call_names


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlEnrichment:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml
        return parse_semantics_xml(SEMA_XML_PATH)

    def test_ds_add_u32_strips_return_data(self, blocks):
        b = blocks['DS_ADD_U32']
        enriched = enrich_block(b)
        for c in enriched.body.children:
            if c.kind == SemaNodeKind.ASSIGN:
                lhs = c.children[0]
                while lhs.kind == SemaNodeKind.CAST and lhs.children:
                    lhs = lhs.children[0]
                assert not (lhs.kind == SemaNodeKind.ID
                            and lhs.id_name == 'RETURN_DATA')

    def test_ds_add_rtn_u32_keeps_return_data(self, blocks):
        b = blocks['DS_ADD_RTN_U32']
        enriched = enrich_block(b)
        has_return = False
        for c in enriched.body.children:
            if c.kind == SemaNodeKind.ASSIGN:
                lhs = c.children[0]
                while lhs.kind == SemaNodeKind.CAST and lhs.children:
                    lhs = lhs.children[0]
                if lhs.kind == SemaNodeKind.ID and lhs.id_name == 'RETURN_DATA':
                    has_return = True
        assert has_return

    def test_non_rtn_count(self, blocks):
        count = 0
        for name, b in blocks.items():
            if not name.startswith('DS_') or '_RTN_' in name or b.is_empty:
                continue
            enriched = enrich_block(b)
            orig_len = len(b.body.children) if b.body.kind == SemaNodeKind.SEQ else 0
            new_len = len(enriched.body.children) if enriched.body.kind == SemaNodeKind.SEQ else 0
            if new_len < orig_len:
                count += 1
        assert count == 39

    def test_all_enrich_without_error(self, blocks):
        errors = []
        for name, block in blocks.items():
            try:
                enriched = enrich_block(block)
                assert enriched.enriched
            except Exception as e:
                errors.append(f'{name}: {e}')
        assert errors == [], f'{len(errors)} enrichment errors:\n' + '\n'.join(errors[:10])
