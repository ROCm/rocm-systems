# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the SemaAST data model."""

import dataclasses
import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
    validate_types,
    validate_well_formed,
    _BINARY_KINDS,
    _UNARY_KINDS,
    _TERNARY_KINDS,
    _STATEMENT_KINDS,
)


class TestSemaType:
    def test_basic_construction(self):
        t = SemaType('U', 32)
        assert t.base == 'U'
        assert t.size == 32

    def test_default_size_is_zero(self):
        t = SemaType('S')
        assert t.size == 0

    def test_classvar_constants(self):
        assert SemaType.U32 == SemaType('U', 32)
        assert SemaType.F32 == SemaType('F', 32)
        assert SemaType.B64 == SemaType('B', 64)
        assert SemaType.I64 == SemaType('I', 64)
        assert SemaType.U1 == SemaType('U', 1)
        assert SemaType.F16 == SemaType('F', 16)
        assert SemaType.F64 == SemaType('F', 64)
        assert SemaType.I32 == SemaType('I', 32)

    def test_cpp_type_float(self):
        assert SemaType.F16.cpp_type == 'uint16_t'
        assert SemaType.F32.cpp_type == 'float'
        assert SemaType.F64.cpp_type == 'double'

    def test_cpp_type_unsigned(self):
        assert SemaType('U', 8).cpp_type == 'uint8_t'
        assert SemaType('U', 16).cpp_type == 'uint16_t'
        assert SemaType.U32.cpp_type == 'uint32_t'
        assert SemaType.U64.cpp_type == 'uint64_t'

    def test_cpp_type_signed(self):
        assert SemaType('I', 8).cpp_type == 'int8_t'
        assert SemaType('I', 16).cpp_type == 'int16_t'
        assert SemaType.I32.cpp_type == 'int32_t'
        assert SemaType.I64.cpp_type == 'int64_t'

    def test_cpp_type_bits(self):
        assert SemaType.B32.cpp_type == 'uint32_t'
        assert SemaType.B64.cpp_type == 'uint64_t'

    def test_cpp_type_fallback(self):
        assert SemaType('S', 0).cpp_type == 'uint32_t'

    def test_cpp_type_bf16(self):
        assert SemaType('BF', 16).cpp_type == 'uint16_t'

    def test_cpp_type_fp8(self):
        assert SemaType('FP', 8).cpp_type == 'uint8_t'
        assert SemaType('FP', 4).cpp_type == 'uint8_t'

    def test_frozen(self):
        t = SemaType.U32
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.base = 'I'

    def test_equality(self):
        assert SemaType('U', 32) == SemaType('U', 32)
        assert SemaType('U', 32) != SemaType('I', 32)
        assert SemaType('U', 32) != SemaType('U', 64)

    def test_hashable(self):
        s = {SemaType.U32, SemaType.F32, SemaType.U32}
        assert len(s) == 2


class TestSemaNodeKind:
    def test_total_count(self):
        assert len(SemaNodeKind) == 72

    def test_all_xml_op_types_mapped(self):
        op_map = {
            k.value: k for k in SemaNodeKind
            if not k.value.startswith('_')
        }
        xml_ops = [
            '+', '-', '*', '/', '%', '**', '+=', '-=',
            '==', '!=', '<', '>', '<=', '>=', '<>',
            '&', '|', '^', '<<', '>>', '.bitneg', '.bitcat',
            '&&', '||', '.boolneg',
            '.fma', '.sqrt', '.sin', '.cos', '.log2', '.ldexp',
            '.floor', '.trunc', '.fract', '.pow',
            '.abs', '.uminus', '.uplus', '.sign', '.signext', '.signext_from_bit',
            '.cast',
            '.instoperand', '.arrayderef', '.arrayslice', '.arrayslicesize',
            '.fieldderef', '.cons_array',
            '.call', '.lambda',
            ':seq', ':if', ':for', ':while', ':break', ':return',
            ':declare', ':comment', ':eval', ':pragma',
            '=', '?:',
            '.sum', '.within', '.exponent', '.mantissa',
        ]
        for op in xml_ops:
            assert op in op_map, f"XML op type {op!r} not in SemaNodeKind"

    def test_xml_op_type_values(self):
        assert SemaNodeKind.ADD.value == '+'
        assert SemaNodeKind.SUB.value == '-'
        assert SemaNodeKind.EQ.value == '=='
        assert SemaNodeKind.SEQ.value == ':seq'
        assert SemaNodeKind.IF.value == ':if'
        assert SemaNodeKind.CAST.value == '.cast'
        assert SemaNodeKind.INSTOPERAND.value == '.instoperand'
        assert SemaNodeKind.CALL.value == '.call'
        assert SemaNodeKind.ASSIGN.value == '='
        assert SemaNodeKind.TERNARY.value == '?:'
        assert SemaNodeKind.PRAGMA.value == ':pragma'

    def test_synthetic_leaf_values(self):
        assert SemaNodeKind.LIT.value == '_lit'
        assert SemaNodeKind.ID.value == '_id'

    def test_forward_compat_values(self):
        assert SemaNodeKind.SWITCH.value == ':switch'
        assert SemaNodeKind.CASE.value == ':case'
        assert SemaNodeKind.DEFAULT.value == ':default'

    def test_op_type_map_excludes_synthetics(self):
        op_map = {
            k.value: k for k in SemaNodeKind
            if not k.value.startswith('_')
        }
        assert '_lit' not in op_map
        assert '_id' not in op_map
        assert '+' in op_map
        assert ':seq' in op_map

    def test_kind_sets_are_disjoint(self):
        assert _BINARY_KINDS.isdisjoint(_UNARY_KINDS)
        assert _BINARY_KINDS.isdisjoint(_TERNARY_KINDS)
        assert _UNARY_KINDS.isdisjoint(_TERNARY_KINDS)


class TestSemaNode:
    def test_leaf_lit(self):
        n = SemaNode(SemaNodeKind.LIT, lit_value='42', ty=SemaType.U32)
        assert n.is_leaf()
        assert n.lit_value == '42'
        assert n.children == ()

    def test_leaf_id(self):
        n = SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1)
        assert n.is_leaf()
        assert n.id_name == 'SCC'

    def test_non_leaf(self):
        left = SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32)
        right = SemaNode(SemaNodeKind.LIT, lit_value='2', ty=SemaType.U32)
        add = SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(left, right))
        assert not add.is_leaf()
        assert len(add.children) == 2

    def test_walk_preorder(self):
        leaf0 = SemaNode(SemaNodeKind.LIT, lit_value='0')
        leaf1 = SemaNode(SemaNodeKind.LIT, lit_value='1')
        add = SemaNode(SemaNodeKind.ADD, children=(leaf0, leaf1))
        seq = SemaNode(SemaNodeKind.SEQ, children=(add,))
        walked = list(seq.walk())
        assert len(walked) == 4
        assert walked[0] is seq
        assert walked[1] is add
        assert walked[2] is leaf0
        assert walked[3] is leaf1

    def test_frozen(self):
        n = SemaNode(SemaNodeKind.LIT, lit_value='0')
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.kind = SemaNodeKind.ID

    def test_replace_creates_new_node(self):
        n = SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32)
        n2 = dataclasses.replace(n, lit_value='1')
        assert n.lit_value == '0'
        assert n2.lit_value == '1'
        assert n is not n2

    def test_cast_target(self):
        inner = SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32)
        cast = SemaNode(SemaNodeKind.CAST, ty=SemaType.U64,
                        cast_target=SemaType.U64, children=(inner,))
        assert cast.cast_target == SemaType.U64

    def test_call_name(self):
        callee = SemaNode(SemaNodeKind.ID, id_name='CalcBufferAddr')
        call = SemaNode(SemaNodeKind.CALL, call_name='CalcBufferAddr',
                        children=(callee,))
        assert call.call_name == 'CalcBufferAddr'

    def test_unknown_op(self):
        n = SemaNode(SemaNodeKind.SEQ, unknown_op='.some_future_op')
        assert n.unknown_op == '.some_future_op'


class TestExecModel:
    def test_values(self):
        assert ExecModel.SCALAR.value == 'scalar'
        assert ExecModel.VECTOR.value == 'vector'
        assert ExecModel.BRANCH.value == 'branch'
        assert ExecModel.UNKNOWN.value == 'unknown'

    def test_from_string(self):
        assert ExecModel('scalar') == ExecModel.SCALAR
        assert ExecModel('vector') == ExecModel.VECTOR


class TestSemaBlock:
    def _make_non_empty_block(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ))
        return SemaBlock('S_NOP', ExecModel.SCALAR, body)

    def _make_empty_block(self):
        body = SemaNode(SemaNodeKind.SEQ, children=())
        return SemaBlock('S_ENDPGM', ExecModel.UNKNOWN, body)

    def test_is_empty_stub(self):
        block = self._make_empty_block()
        assert block.is_empty

    def test_is_empty_non_stub(self):
        block = self._make_non_empty_block()
        assert not block.is_empty

    def test_enriched_default_false(self):
        block = self._make_non_empty_block()
        assert not block.enriched

    def test_enriched_mutable(self):
        block = self._make_non_empty_block()
        block.enriched = True
        assert block.enriched


class TestValidateTypes:
    def test_valid_tree_passes(self):
        tree = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(
                    SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                    SemaNode(SemaNodeKind.LIT, lit_value='2', ty=SemaType.U32),
                )),
            )),
        ))
        validate_types(tree)

    def test_statement_with_type_fails(self):
        tree = SemaNode(SemaNodeKind.SEQ, ty=SemaType.U32, children=())
        with pytest.raises(AssertionError, match="Statement node"):
            validate_types(tree)

    @pytest.mark.parametrize('kind', [
        SemaNodeKind.ASSIGN, SemaNodeKind.IF, SemaNodeKind.FOR,
        SemaNodeKind.WHILE, SemaNodeKind.BREAK, SemaNodeKind.CONTINUE,
        SemaNodeKind.RETURN, SemaNodeKind.ADD_ASSIGN, SemaNodeKind.SUB_ASSIGN,
    ])
    def test_each_statement_kind_rejects_type(self, kind):
        n = SemaNode(kind, ty=SemaType.U32, children=())
        with pytest.raises(AssertionError):
            validate_types(n)


class TestValidateWellFormed:
    def test_valid_add(self):
        tree = SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
            SemaNode(SemaNodeKind.LIT, lit_value='2', ty=SemaType.U32),
        ))
        validate_well_formed(tree)

    def test_add_wrong_children_fails(self):
        tree = SemaNode(SemaNodeKind.ADD, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
        ))
        with pytest.raises(AssertionError, match="must have 2 children"):
            validate_well_formed(tree)

    def test_valid_if_two_children(self):
        tree = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
            SemaNode(SemaNodeKind.SEQ, children=()),
        ))
        validate_well_formed(tree)

    def test_valid_if_three_children(self):
        tree = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
            SemaNode(SemaNodeKind.SEQ, children=()),
            SemaNode(SemaNodeKind.SEQ, children=()),
        ))
        validate_well_formed(tree)

    def test_if_one_child_fails(self):
        tree = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
        ))
        with pytest.raises(AssertionError, match="IF must have >= 2"):
            validate_well_formed(tree)

    def test_if_many_children_ok(self):
        children = tuple(
            SemaNode(SemaNodeKind.LIT, lit_value=str(i))
            for i in range(9)
        )
        tree = SemaNode(SemaNodeKind.IF, children=children)
        validate_well_formed(tree)

    def test_for_wrong_children_fails(self):
        tree = SemaNode(SemaNodeKind.FOR, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
        ))
        with pytest.raises(AssertionError, match="FOR must have 4"):
            validate_well_formed(tree)

    def test_bitcat_multi_children_ok(self):
        children = tuple(
            SemaNode(SemaNodeKind.LIT, lit_value=str(i), ty=SemaType.B32)
            for i in range(4)
        )
        tree = SemaNode(SemaNodeKind.BITCAT, children=children)
        validate_well_formed(tree)

    def test_signext_from_bit_two_children_ok(self):
        tree = SemaNode(SemaNodeKind.SIGNEXT_FROM_BIT, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            SemaNode(SemaNodeKind.LIT, lit_value='7', ty=SemaType.U32),
        ))
        validate_well_formed(tree)

    def test_lit_without_value_fails(self):
        tree = SemaNode(SemaNodeKind.LIT)
        with pytest.raises(AssertionError, match="lit_value"):
            validate_well_formed(tree)

    def test_id_without_name_fails(self):
        tree = SemaNode(SemaNodeKind.ID)
        with pytest.raises(AssertionError, match="id_name"):
            validate_well_formed(tree)

    def test_call_needs_at_least_one_child(self):
        tree = SemaNode(SemaNodeKind.CALL, children=())
        with pytest.raises(AssertionError, match="CALL must have >= 1"):
            validate_well_formed(tree)

    def test_valid_fma(self):
        args = tuple(
            SemaNode(SemaNodeKind.LIT, lit_value=str(i), ty=SemaType.F32)
            for i in range(3)
        )
        tree = SemaNode(SemaNodeKind.FMA, ty=SemaType.F32, children=args)
        validate_well_formed(tree)

    def test_valid_return(self):
        tree = SemaNode(SemaNodeKind.RETURN, children=(
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ))
        validate_well_formed(tree)

    def test_valid_complex_tree(self):
        """Validate a tree resembling S_ADD_CO_U32."""
        s0 = SemaNode(SemaNodeKind.INSTOPERAND, ty=SemaType.B32, children=(
            SemaNode(SemaNodeKind.ID, id_name='S'),
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ))
        s1 = SemaNode(SemaNodeKind.INSTOPERAND, ty=SemaType.B32, children=(
            SemaNode(SemaNodeKind.ID, id_name='S'),
            SemaNode(SemaNodeKind.LIT, lit_value='1'),
        ))
        add = SemaNode(SemaNodeKind.ADD, ty=SemaType.U64, children=(
            SemaNode(SemaNodeKind.CAST, cast_target=SemaType.U64, children=(s0,)),
            SemaNode(SemaNodeKind.CAST, cast_target=SemaType.U64, children=(s1,)),
        ))
        assign_tmp = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            add,
        ))
        assign_scc = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
            SemaNode(SemaNodeKind.TERNARY, ty=SemaType.U1, children=(
                SemaNode(SemaNodeKind.GE, children=(
                    SemaNode(SemaNodeKind.ID, id_name='tmp', ty=SemaType.U64),
                    SemaNode(SemaNodeKind.LIT, lit_value='4294967296',
                             ty=SemaType.U64),
                )),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U1),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U1),
            )),
        ))
        tree = SemaNode(SemaNodeKind.SEQ, children=(assign_tmp, assign_scc))
        validate_well_formed(tree)
        validate_types(tree)
