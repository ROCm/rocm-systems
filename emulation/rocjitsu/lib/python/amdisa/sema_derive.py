# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Derive SemaAST from mnemonic-based InstructionSemantics.

For ISAs without expression-tree semantics, this module
converts :class:`~amdisa.semantics.InstructionSemantics` metadata into
:class:`~amdisa.sema_ast.SemaBlock` trees. These trees are structurally
equivalent to what the XML parser produces, enabling both paths
to converge on the same lowering backend.

Covers scalar semantic classes:
  scalar_unary, scalar_binop, scalar_bfe, scalar_cmp, scalar_cmpk,
  scalar_bitcmp, scalar_saveexec, scalar_mov, scalar_cmov

Covers vector ALU + cmp semantic classes:
  vector_mov, vector_unary, vector_binop, vector_ternary,
  vector_cmp, vector_cmpx, vector_cmp_class, vector_cmpx_class,
  vector_add_co, vector_cndmask, vector_readfirstlane, vector_readlane,
  vector_writelane, vector_swap, vector_fmaak, vector_fmamk
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)

if TYPE_CHECKING:
    from amdisa.semantics import InstructionSemantics


def _src(idx: int, ty: SemaType = SemaType.B32) -> SemaNode:
    return SemaNode(SemaNodeKind.INSTOPERAND, ty=ty, children=(
        SemaNode(SemaNodeKind.ID, id_name='S'),
        SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
    ))


def _dst(idx: int, ty: SemaType = SemaType.B32) -> SemaNode:
    return SemaNode(SemaNodeKind.INSTOPERAND, ty=ty, children=(
        SemaNode(SemaNodeKind.ID, id_name='D'),
        SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
    ))


def _cast(inner: SemaNode, target: SemaType) -> SemaNode:
    return SemaNode(SemaNodeKind.CAST, ty=target, cast_target=target,
                    children=(inner,))


def _lit(val: str, ty: SemaType = SemaType.U32) -> SemaNode:
    return SemaNode(SemaNodeKind.LIT, lit_value=val, ty=ty)


def _id(name: str, ty: SemaType | None = None) -> SemaNode:
    return SemaNode(SemaNodeKind.ID, id_name=name, ty=ty)


def _assign(lhs: SemaNode, rhs: SemaNode) -> SemaNode:
    return SemaNode(SemaNodeKind.ASSIGN, children=(lhs, rhs))


def _scc_write(expr: SemaNode) -> SemaNode:
    return _assign(_id('SCC', SemaType.U1), expr)


def _dtype_to_sema(dtype: str | None) -> SemaType:
    """Map canonical dtype string to SemaType.

    Unmapped narrow dtypes (i16, u16, i8, u8, i24, u24, b16) intentionally
    fall back to U32 — the wired classes handle narrowing via instruction-
    specific inline templates. Add explicit entries here when wiring packed
    or sub-32-bit classes through SemaAST.
    """
    if dtype is None:
        return SemaType.U32
    mapping = {
        'u32': SemaType.U32, 'i32': SemaType.I32, 'b32': SemaType.B32,
        'u64': SemaType.U64, 'i64': SemaType.I64, 'b64': SemaType.B64,
        'u16': SemaType('U', 16), 'i16': SemaType('I', 16),
        'b16': SemaType('B', 16),
        'f32': SemaType.F32, 'f64': SemaType.F64, 'f16': SemaType.F16,
        'bf16': SemaType.BF16, 'fp8': SemaType.FP8, 'bf8': SemaType.BF8,
        'fp6': SemaType.FP6, 'fp4': SemaType.FP4,
    }
    return mapping.get(dtype, SemaType.U32)


_OP_TO_KIND: dict[str, SemaNodeKind] = {
    'add': SemaNodeKind.ADD, 'sub': SemaNodeKind.SUB,
    'mul': SemaNodeKind.MUL,
    'mulhi': SemaNodeKind.CALL, 'mul_hi': SemaNodeKind.CALL,
    'and': SemaNodeKind.AND, 'or': SemaNodeKind.OR,
    'xor': SemaNodeKind.XOR, 'nand': SemaNodeKind.AND,
    'nor': SemaNodeKind.OR, 'xnor': SemaNodeKind.XOR,
    'andn2': SemaNodeKind.AND, 'orn2': SemaNodeKind.OR,
    'shl': SemaNodeKind.SHL, 'shr': SemaNodeKind.SHR,
    'ashr': SemaNodeKind.CALL, 'ashrrev': SemaNodeKind.CALL,
    'min': SemaNodeKind.CALL, 'max': SemaNodeKind.CALL,
    'absdiff': SemaNodeKind.CALL,
    'not': SemaNodeKind.BITNEG, 'brev': SemaNodeKind.CALL,
    'bcnt1': SemaNodeKind.CALL, 'bcnt0': SemaNodeKind.CALL,
    'ff0': SemaNodeKind.CALL, 'ff1': SemaNodeKind.CALL,
    'flbit': SemaNodeKind.CALL,
    'wqm': SemaNodeKind.CALL, 'quadmask': SemaNodeKind.CALL,
    'bitset0': SemaNodeKind.CALL, 'bitset1': SemaNodeKind.CALL,
}

_CMP_OP_TO_KIND: dict[str, SemaNodeKind] = {
    'eq': SemaNodeKind.EQ, 'ne': SemaNodeKind.NE,
    'lg': SemaNodeKind.NE, 'neq': SemaNodeKind.NE,
    'lt': SemaNodeKind.LT, 'gt': SemaNodeKind.GT,
    'le': SemaNodeKind.LE, 'ge': SemaNodeKind.GE,
}

_CMP_NEGATED: dict[str, SemaNodeKind] = {
    'nlt': SemaNodeKind.LT,
    'ngt': SemaNodeKind.GT,
    'nle': SemaNodeKind.LE,
    'nge': SemaNodeKind.GE,
    'nlg': SemaNodeKind.NE,
}


def _make_cmp(op: str, src0: SemaNode, src1: SemaNode) -> SemaNode:
    """Build a comparison node, handling ordered, unordered, and constant ops."""
    if op == 'f':
        return _lit('0', SemaType.U1)
    if op == 't':
        return _lit('1', SemaType.U1)
    if op == 'o':
        return SemaNode(SemaNodeKind.CALL, ty=SemaType.U1,
                        call_name='is_ordered',
                        children=(_id('is_ordered'), src0, src1))
    if op == 'u':
        return SemaNode(SemaNodeKind.CALL, ty=SemaType.U1,
                        call_name='is_unordered',
                        children=(_id('is_unordered'), src0, src1))
    neg_kind = _CMP_NEGATED.get(op)
    if neg_kind is not None:
        inner = SemaNode(neg_kind, ty=SemaType.U1, children=(src0, src1))
        return SemaNode(SemaNodeKind.BOOLNEG, ty=SemaType.U1, children=(inner,))
    kind = _CMP_OP_TO_KIND.get(op, SemaNodeKind.EQ)
    return SemaNode(kind, ty=SemaType.U1, children=(src0, src1))


_DERIVE_REGISTRY: dict[str, type['_ScalarDeriver']] = {}


def derive_sema_block(sem: InstructionSemantics) -> SemaBlock | None:
    """Convert an InstructionSemantics into a SemaBlock.

    Returns None if the semantic class is not supported yet.
    """
    cls = _DERIVE_REGISTRY.get(sem.semantic_class)
    if cls is None:
        return None
    return cls.derive(sem)


class _ScalarDeriver:
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        raise NotImplementedError


def _register(semantic_class: str):
    def decorator(cls):
        _DERIVE_REGISTRY[semantic_class] = cls
        return cls
    return decorator


@_register('scalar_mov')
class _ScalarMov(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        body = _assign(_cast(_dst(0, ty), ty), _cast(_src(0, ty), ty))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_cmov')
class _ScalarCmov(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        body = SemaNode(SemaNodeKind.IF, children=(
            _id('SCC', SemaType.U1),
            _assign(_cast(_dst(0, ty), ty), _cast(_src(0, ty), ty)),
        ))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_unary')
class _ScalarUnary(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation
        # bitset0/1 are read-modify-write on dst — not expressible as
        # a simple unary template. Fall through to inline handler.
        if op in ('bitset0', 'bitset1'):
            return SemaBlock(sem.name, ExecModel.SCALAR,
                             SemaNode(SemaNodeKind.SEQ, children=()))
        # 64-bit unary ops need 64-bit-specific templates. Use distinct
        # call names so the lowering selects the right implementation.
        dtype = sem.data_type
        if dtype in ('b64', 'u64', 'i64') and op in (
            'brev', 'wqm', 'quadmask', 'ff0', 'ff1', 'flbit', 'ctz',
            'bcnt1', 'bcnt0',
        ):
            call_name = f'{op}64'
            ty64 = _dtype_to_sema(dtype)
            src0_64 = _cast(_src(0, ty64), ty64)
            result = SemaNode(SemaNodeKind.CALL, ty=ty64,
                              call_name=call_name,
                              children=(_id(call_name), src0_64))
            stmts_64: list[SemaNode] = [
                _assign(_id('result', ty64), result),
                _assign(_cast(_dst(0, ty64), ty64), _id('result', ty64)),
            ]
            if sem.sets_scc and sem.sets_scc != 'none':
                stmts_64.append(_scc_write(
                    SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                        _id('result', ty64), _lit('0', ty64),
                    )),
                ))
            body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts_64))
            return SemaBlock(sem.name, ExecModel.SCALAR, body)
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        stmts: list[SemaNode] = []

        if op == 'not':
            result = SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(src0,))
        elif op == 'sext8':
            result = SemaNode(SemaNodeKind.SIGNEXT_FROM_BIT, ty=ty,
                              children=(src0, _lit('7')))
        elif op == 'sext16':
            result = SemaNode(SemaNodeKind.SIGNEXT_FROM_BIT, ty=ty,
                              children=(src0, _lit('15')))
        elif op in ('ceil', 'floor', 'trunc', 'rndne'):
            kind_map = {
                'ceil': SemaNodeKind.CALL, 'floor': SemaNodeKind.FLOOR,
                'trunc': SemaNodeKind.TRUNC, 'rndne': SemaNodeKind.CALL,
            }
            k = kind_map[op]
            if k == SemaNodeKind.CALL:
                fn = 'std::ceil' if op == 'ceil' else 'std::nearbyint'
                result = SemaNode(SemaNodeKind.CALL, ty=ty,
                                  call_name=fn, children=(
                    SemaNode(SemaNodeKind.ID, id_name=fn), src0))
            else:
                result = SemaNode(k, ty=ty, children=(src0,))
        elif op and op.startswith('cvt_'):
            result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name=op,
                              children=(_id(op), src0))
        else:
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name=op or 'unknown',
                              children=(_id(op or 'unknown'), src0))

        result_ty = SemaType.F32 if ty.base in ('F', 'BF') and ty.size == 16 else ty
        stmts.append(_assign(_id('result', result_ty), result))
        stmts.append(_assign(_cast(_dst(0), ty), _id('result', result_ty)))

        if sem.sets_scc and sem.sets_scc != 'none':
            stmts.append(_scc_write(
                SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                    _id('result', ty), _lit('0', ty),
                )),
            ))

        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_binop')
class _ScalarBinop(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        stmts: list[SemaNode] = []

        kind = _OP_TO_KIND.get(op or '', SemaNodeKind.CALL)

        if op in ('shl', 'shr'):
            mask_val = '63u' if ty.size == 64 else '31u'
            masked = SemaNode(SemaNodeKind.AND, ty=SemaType.U32,
                              children=(src1, _lit(mask_val)))
            shift_kind = SemaNodeKind.SHL if op == 'shl' else SemaNodeKind.SHR
            result = SemaNode(shift_kind, ty=ty, children=(src0, masked))
        elif op == 'ashr':
            call_name = 'ashr_i64' if ty.size == 64 else 'util::arithmetic_shr'
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name=call_name,
                              children=(_id(call_name), src0, src1))
        elif op in ('nand', 'nor', 'xnor'):
            inner = SemaNode(kind, ty=ty, children=(src0, src1))
            result = SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(inner,))
        elif op == 'andn2':
            result = SemaNode(SemaNodeKind.AND, ty=ty, children=(
                src0, SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(src1,)),
            ))
        elif op == 'orn2':
            result = SemaNode(SemaNodeKind.OR, ty=ty, children=(
                src0, SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(src1,)),
            ))
        elif op in ('min', 'max'):
            fn = f'std::{op}'
            result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name=fn,
                              children=(_id(fn), src0, src1))
        elif op == 'absdiff':
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='ABSDIFF',
                              children=(_id('ABSDIFF'), src0, src1))
        elif kind == SemaNodeKind.CALL:
            fn = op or 'unknown'
            if fn == 'bfe' and ty.size == 64:
                fn = 'bfe_i64' if ty.base == 'I' else 'bfe64'
            elif fn == 'bfe' and ty.base == 'I':
                fn = 'bfe_i32'
            result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name=fn,
                              children=(_id(fn), src0, src1))
        else:
            result = SemaNode(kind, ty=ty, children=(src0, src1))

        scc_handled_by_template = op in (
            'addc', 'subb',
            'lshl1_add', 'lshl2_add', 'lshl3_add', 'lshl4_add',
        )

        needs_cached_ops = (sem.sets_scc and sem.sets_scc != 'none'
                            and not scc_handled_by_template
                            and sem.sets_scc in ('carry', 'borrow', 'overflow',
                                                 'compare'))
        if needs_cached_ops:
            stmts.append(_assign(_id('s0', ty), src0))
            stmts.append(_assign(_id('s1', ty), src1))
            src0_scc = _id('s0', ty)
            src1_scc = _id('s1', ty)
        else:
            src0_scc = src0
            src1_scc = src1

        result_ty = SemaType.F32 if ty.base in ('F', 'BF') and ty.size == 16 else ty
        stmts.append(_assign(_id('result', result_ty), result))
        stmts.append(_assign(_cast(_dst(0), ty), _id('result', result_ty)))

        if sem.sets_scc and sem.sets_scc != 'none' and not scc_handled_by_template:
            if sem.sets_scc == 'carry':
                wide = SemaNode(SemaNodeKind.ADD, ty=SemaType.U64, children=(
                    _cast(src0_scc, SemaType.U64), _cast(src1_scc, SemaType.U64),
                ))
                scc_expr = SemaNode(SemaNodeKind.GT, ty=SemaType.U1, children=(
                    wide, _lit('4294967295', SemaType.U64),
                ))
            elif sem.sets_scc == 'borrow':
                scc_expr = SemaNode(SemaNodeKind.LT, ty=SemaType.U1, children=(
                    src0_scc, src1_scc,
                ))
            elif sem.sets_scc == 'overflow':
                overflow_kind = SemaNodeKind.SUB if op == 'sub' else SemaNodeKind.ADD
                wide = SemaNode(overflow_kind, ty=SemaType.I64, children=(
                    _cast(src0_scc, SemaType.I64), _cast(src1_scc, SemaType.I64),
                ))
                scc_expr = SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                    wide, _cast(_id('result', ty), SemaType.I64),
                ))
            elif sem.sets_scc == 'compare':
                cmp_kind = SemaNodeKind.GE if op == 'max' else SemaNodeKind.LT
                scc_expr = SemaNode(cmp_kind, ty=SemaType.U1, children=(
                    src0_scc, src1_scc,
                ))
            else:
                scc_expr = SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                    _id('result', ty), _lit('0', ty),
                ))
            stmts.append(_scc_write(scc_expr))

        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_cmp')
class _ScalarCmp(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)

        cmp = _make_cmp(op or "", src0, src1)
        
        body = _scc_write(cmp)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_cmpk')
class _ScalarCmpk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        imm = _cast(_src(1), ty)

        
        cmp = _make_cmp(op or "", src0, imm)
        body = _scc_write(cmp)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_bitcmp')
class _ScalarBitcmp(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)

        bit_extract = SemaNode(SemaNodeKind.AND, ty=SemaType.U1, children=(
            SemaNode(SemaNodeKind.SHR, ty=ty, children=(src0, src1)),
            _lit('1', SemaType.U1),
        ))
        if op == 'bitset0':
            cmp = SemaNode(SemaNodeKind.EQ, ty=SemaType.U1, children=(
                bit_extract, _lit('0', SemaType.U1),
            ))
        else:
            cmp = SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                bit_extract, _lit('0', SemaType.U1),
            ))
        body = _scc_write(cmp)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_bfe')
class _ScalarBfe(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)

        offset = SemaNode(SemaNodeKind.AND, ty=SemaType.U32, children=(
            src1, _lit('31'),
        ))
        width = SemaNode(SemaNodeKind.AND, ty=SemaType.U32, children=(
            SemaNode(SemaNodeKind.SHR, ty=ty, children=(src1, _lit('16'))),
            _lit('127'),
        ))
        extract = SemaNode(SemaNodeKind.CALL, ty=ty,
                           call_name='util::bfe',
                           children=(_id('util::bfe'), src0, offset, width))

        stmts = [
            _assign(_id('result', ty), extract),
            _assign(_cast(_dst(0), ty), _id('result', ty)),
            _scc_write(SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                _id('result', ty), _lit('0', ty),
            ))),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_saveexec')
class _ScalarSaveexec(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation or 'and'
        stmts: list[SemaNode] = []

        # Cache source before writing dst (prevents aliasing when sdst == ssrc0).
        src0 = _cast(_src(0), SemaType.U64)
        stmts.append(_assign(_id('src', SemaType.U64), src0))
        cached_src = _id('src', SemaType.U64)

        stmts.append(_assign(
            _cast(_dst(0), SemaType.U64),
            _id('EXEC', SemaType.U64),
        ))

        exec_read = _id('EXEC', SemaType.U64)

        kind = _OP_TO_KIND.get(op, SemaNodeKind.AND)
        if op in ('nand', 'nor', 'xnor'):
            inner = SemaNode(kind, ty=SemaType.U64, children=(exec_read, cached_src))
            new_exec = SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64,
                                children=(inner,))
        elif op == 'andn2':
            new_exec = SemaNode(SemaNodeKind.AND, ty=SemaType.U64, children=(
                cached_src,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(exec_read,)),
            ))
        elif op == 'orn2':
            new_exec = SemaNode(SemaNodeKind.OR, ty=SemaType.U64, children=(
                cached_src,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(exec_read,)),
            ))
        elif op == 'and_not0':
            new_exec = SemaNode(SemaNodeKind.AND, ty=SemaType.U64, children=(
                exec_read,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(cached_src,)),
            ))
        elif op == 'and_not1':
            new_exec = SemaNode(SemaNodeKind.AND, ty=SemaType.U64, children=(
                cached_src,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(exec_read,)),
            ))
        elif op == 'or_not0':
            new_exec = SemaNode(SemaNodeKind.OR, ty=SemaType.U64, children=(
                exec_read,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(cached_src,)),
            ))
        elif op == 'or_not1':
            new_exec = SemaNode(SemaNodeKind.OR, ty=SemaType.U64, children=(
                cached_src,
                SemaNode(SemaNodeKind.BITNEG, ty=SemaType.U64, children=(exec_read,)),
            ))
        elif kind == SemaNodeKind.CALL:
            new_exec = SemaNode(SemaNodeKind.CALL, ty=SemaType.U64,
                                call_name=op, children=(_id(op), exec_read, cached_src))
        else:
            new_exec = SemaNode(kind, ty=SemaType.U64, children=(exec_read, cached_src))

        stmts.append(_assign(_id('EXEC', SemaType.U64), new_exec))
        stmts.append(_scc_write(
            SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                _id('EXEC', SemaType.U64), _lit('0', SemaType.U64),
            )),
        ))

        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


# =========================================================================
# Vector ALU + cmp semantic classes
# =========================================================================

_UNARY_KIND_MAP: dict[str, SemaNodeKind] = {
    'not': SemaNodeKind.BITNEG, 'bfrev': SemaNodeKind.CALL,
    'ceil': SemaNodeKind.CALL, 'floor': SemaNodeKind.FLOOR,
    'trunc': SemaNodeKind.TRUNC, 'fract': SemaNodeKind.FRACT,
    'rndne': SemaNodeKind.CALL, 'rcp': SemaNodeKind.CALL,
    'rcp_iflag': SemaNodeKind.CALL, 'rsq': SemaNodeKind.CALL,
    'sqrt': SemaNodeKind.SQRT, 'sin': SemaNodeKind.SIN,
    'cos': SemaNodeKind.COS, 'log': SemaNodeKind.LOG2,
    'exp': SemaNodeKind.CALL, 'bcnt': SemaNodeKind.CALL,
    'ffbl': SemaNodeKind.CALL, 'ffbh_u32': SemaNodeKind.CALL,
    'ffbh_i32': SemaNodeKind.CALL,
    'cls_i32': SemaNodeKind.CALL,
}

_BINOP_KIND_MAP: dict[str, SemaNodeKind] = {
    'add': SemaNodeKind.ADD, 'sub': SemaNodeKind.SUB,
    'subrev': SemaNodeKind.SUB,
    'mul': SemaNodeKind.MUL, 'mul_lo': SemaNodeKind.MUL,
    'mul_hi': SemaNodeKind.CALL, 'mulhi': SemaNodeKind.CALL,
    'and': SemaNodeKind.AND, 'or': SemaNodeKind.OR,
    'xor': SemaNodeKind.XOR,
    'shl': SemaNodeKind.SHL, 'shr': SemaNodeKind.SHR,
    'ashr': SemaNodeKind.CALL, 'ashrrev': SemaNodeKind.CALL,
    'min': SemaNodeKind.CALL, 'max': SemaNodeKind.CALL,
    'lshlrev': SemaNodeKind.SHL, 'lshrrev': SemaNodeKind.SHR,
    'ldexp': SemaNodeKind.LDEXP,
}


def _vec_unary_expr(op: str | None, src0: SemaNode, ty: SemaType) -> SemaNode:
    if op is None:
        return src0
    kind = _UNARY_KIND_MAP.get(op, SemaNodeKind.CALL)
    if op == 'not':
        return SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(src0,))
    if kind != SemaNodeKind.CALL:
        return SemaNode(kind, ty=ty, children=(src0,))
    return SemaNode(SemaNodeKind.CALL, ty=ty, call_name=op,
                    children=(_id(op), src0))


def _vec_binop_expr(
    op: str | None, src0: SemaNode, src1: SemaNode, ty: SemaType,
) -> SemaNode:
    if op is None:
        return SemaNode(SemaNodeKind.ADD, ty=ty, children=(src0, src1))
    if op in ('subrev', 'rsub'):
        return SemaNode(SemaNodeKind.SUB, ty=ty, children=(src1, src0))
    if op in ('lshlrev', 'lshrrev', 'shl', 'shr'):
        base = {'lshlrev': SemaNodeKind.SHL, 'lshrrev': SemaNodeKind.SHR,
                'shl': SemaNodeKind.SHL, 'shr': SemaNodeKind.SHR}
        mask_val = '63u' if ty.size == 64 else ('15u' if ty.size == 16 else '31u')
        masked_amt = SemaNode(SemaNodeKind.AND, ty=SemaType.U32,
                              children=(src0, _lit(mask_val)))
        return SemaNode(base[op], ty=ty, children=(src1, masked_amt))
    if op == 'fmac':
        return SemaNode(SemaNodeKind.FMA, ty=ty,
                        children=(src0, src1, _cast(_dst(0, ty), ty)))
    if op == 'xnor':
        return SemaNode(SemaNodeKind.BITNEG, ty=ty, children=(
            SemaNode(SemaNodeKind.XOR, ty=ty, children=(src0, src1)),))
    if op == 'mul_legacy':
        return SemaNode(SemaNodeKind.CALL, ty=ty, call_name='mul_legacy',
                        children=(_id('mul_legacy'), src0, src1))
    if op in ('ashr', 'ashrrev'):
        call_name = 'ashr_i64' if ty.size == 64 else 'util::arithmetic_shr'
        return SemaNode(SemaNodeKind.CALL, ty=ty,
                        call_name=call_name,
                        children=(_id(call_name), src1, src0))
    kind = _BINOP_KIND_MAP.get(op, SemaNodeKind.CALL)
    if kind == SemaNodeKind.CALL:
        if op in ('min', 'max'):
            fn = f'std::f{op}' if ty.base in ('F', 'BF') else f'std::{op}'
        else:
            fn = op
        return SemaNode(SemaNodeKind.CALL, ty=ty, call_name=fn,
                        children=(_id(fn), src0, src1))
    return SemaNode(kind, ty=ty, children=(src0, src1))


@_register('vector_mov')
class _VectorMov(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        body = _assign(_cast(_dst(0), ty), _cast(_src(0), ty))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_unary')
class _VectorUnary(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        op = sem.operation
        dtype = sem.data_type

        if op == 'frexp_exp_f32' and dtype == 'f64':
            src0 = _cast(_src(0, SemaType.F64), SemaType.F64)
            result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                              call_name='frexp_exp_f64',
                              children=(_id('frexp_exp_f64'), src0))
            body = _assign(_cast(_dst(0), SemaType.U32), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op == 'frexp_mant_f32' and dtype == 'f64':
            src0 = _cast(_src(0, SemaType.F64), SemaType.F64)
            result = SemaNode(SemaNodeKind.CALL, ty=SemaType.F64,
                              call_name='frexp_mant_f64',
                              children=(_id('frexp_mant_f64'), src0))
            body = _assign(_cast(_dst(0, SemaType.F64), SemaType.F64), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op == 'cvt' and dtype:
            call_name = f'cvt_{dtype}'
            src0 = _src(0)
            body = _assign(_cast(_dst(0), SemaType.B32),
                           SemaNode(SemaNodeKind.CALL, ty=SemaType.B32,
                                    call_name=call_name,
                                    children=(_id(call_name), src0)))
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op and op.startswith('cvt_'):
            src0 = _src(0)
        else:
            src0 = _cast(_src(0, ty), ty)
        result = _vec_unary_expr(op, src0, ty)
        body = _assign(_cast(_dst(0, ty), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_binop')
class _VectorBinop(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        op = sem.operation
        name_lower = sem.name.lower()

        if op == 'ldexp':
            src0 = _cast(_src(0), ty)
            if ty.size == 16:
                src1 = _cast(_cast(_src(1, SemaType('B', 16)),
                                   SemaType('I', 16)), SemaType.I32)
            else:
                src1 = _cast(_src(1), SemaType.I32)
            result = SemaNode(SemaNodeKind.LDEXP, ty=ty,
                              children=(src0, src1))
            body = _assign(_cast(_dst(0), ty), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op in ('mul_hi', 'mulhi'):
            if 'u24' in name_lower or 'u32_u24' in name_lower:
                call_name = 'mul_hi_u24'
            elif 'i24' in name_lower or 'i32_i24' in name_lower:
                call_name = 'mul_hi_i24'
            else:
                call_name = 'mul_hi'
            src0 = _cast(_src(0), ty)
            src1 = _cast(_src(1), ty)
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name=call_name,
                              children=(_id(call_name), src0, src1))
            body = _assign(_cast(_dst(0), ty), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op in ('mul', 'mul_lo'):
            if 'u24' in name_lower or 'u32_u24' in name_lower:
                src0 = _cast(_src(0), ty)
                src1 = _cast(_src(1), ty)
                result = SemaNode(SemaNodeKind.CALL, ty=ty,
                                  call_name='mul_u24',
                                  children=(_id('mul_u24'), src0, src1))
                body = _assign(_cast(_dst(0), ty), result)
                return SemaBlock(sem.name, ExecModel.VECTOR, body)
            if 'i24' in name_lower or 'i32_i24' in name_lower:
                src0 = _cast(_src(0), ty)
                src1 = _cast(_src(1), ty)
                result = SemaNode(SemaNodeKind.CALL, ty=ty,
                                  call_name='mul_i24',
                                  children=(_id('mul_i24'), src0, src1))
                body = _assign(_cast(_dst(0), ty), result)
                return SemaBlock(sem.name, ExecModel.VECTOR, body)

        if op in ('mul', 'mul_lo') and 'u16' in name_lower:
            src0 = _cast(_src(0), ty)
            src1 = _cast(_src(1), ty)
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='mul_lo_u16',
                              children=(_id('mul_lo_u16'), src0, src1))
            body = _assign(_cast(_dst(0), ty), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        result = _vec_binop_expr(op, src0, src1, ty)
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_ternary')
class _VectorTernary(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        op = sem.operation or 'fma'
        dtype = sem.data_type

        if op in ('fma', 'mad', 'fmac') and dtype in ('i24', 'u24'):
            call_name = 'mad_i24' if dtype == 'i24' else 'mad_u24'
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name=call_name,
                              children=(_id(call_name),
                                        _cast(_src(0), ty),
                                        _cast(_src(1), ty),
                                        _cast(_src(2), ty)))
            body = _assign(_cast(_dst(0), ty), result)
            return SemaBlock(sem.name, ExecModel.VECTOR, body)

        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        src2 = _cast(_src(2), ty)
        if op == 'mad':
            mul = SemaNode(SemaNodeKind.MUL, ty=ty, children=(src0, src1))
            result = SemaNode(SemaNodeKind.ADD, ty=ty, children=(mul, src2))
        elif op in ('fma', 'fmac'):
            result = SemaNode(SemaNodeKind.FMA, ty=ty,
                              children=(src0, src1, src2))
        else:
            result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name=op,
                              children=(_id(op), src0, src1, src2))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_cmp')
class _VectorCmp(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        cmp = _make_cmp(sem.operation or "", src0, src1)
        body = _assign(
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1, children=(
                _id('VCC', SemaType.U64), _id('laneId', SemaType.U32),
            )),
            cmp,
        )
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_cmpx')
class _VectorCmpx(_ScalarDeriver):
    # TODO: This derive only writes EXEC, but GFX9 (CDNA) cmpx also writes
    # VCC. Must add VCC[laneId] = cmp before the EXEC write, gated on ISA
    # generation, before this class can be wired through _SEMA_CLASSES.
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        cmp = _make_cmp(sem.operation or "", src0, src1)
        body = _assign(
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1, children=(
                _id('EXEC', SemaType.U64), _id('laneId', SemaType.U32),
            )),
            cmp,
        )
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_cmp_class')
class _VectorCmpClass(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), SemaType.U32)
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U1,
                          call_name='fp_class_test',
                          children=(_id('fp_class_test'), src0, src1))
        body = _assign(
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1, children=(
                _id('VCC', SemaType.U64), _id('laneId', SemaType.U32),
            )),
            result,
        )
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_cmpx_class')
class _VectorCmpxClass(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), SemaType.U32)
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U1,
                          call_name='fp_class_test',
                          children=(_id('fp_class_test'), src0, src1))
        body = _assign(
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1, children=(
                _id('EXEC', SemaType.U64), _id('laneId', SemaType.U32),
            )),
            result,
        )
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_add_co')
class _VectorAddCo(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        op = sem.operation or 'add'
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)

        if op == 'sub':
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='sub_co',
                              children=(_id('sub_co'), src0, src1))
        elif op == 'rsub':
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='sub_co',
                              children=(_id('sub_co'), src1, src0))
        elif op == 'addc':
            cin = SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1,
                           children=(_id('VCC', SemaType.U64),
                                     _id('laneId', SemaType.U32)))
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='addc_co',
                              children=(_id('addc_co'), src0, src1, cin))
        elif op == 'subbc':
            cin = SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1,
                           children=(_id('VCC', SemaType.U64),
                                     _id('laneId', SemaType.U32)))
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='subbc_co',
                              children=(_id('subbc_co'), src0, src1, cin))
        elif op == 'subbrevco':
            cin = SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1,
                           children=(_id('VCC', SemaType.U64),
                                     _id('laneId', SemaType.U32)))
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='subbc_co',
                              children=(_id('subbc_co'), src1, src0, cin))
        else:
            result = SemaNode(SemaNodeKind.CALL, ty=ty,
                              call_name='add_co',
                              children=(_id('add_co'), src0, src1))

        stmts = [
            _assign(_cast(_dst(0), ty), result),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_cndmask')
class _VectorCndmask(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        src0 = _cast(_src(0), ty)
        src1 = _cast(_src(1), ty)
        cond = SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.U1, children=(
            _id('VCC', SemaType.U64), _id('laneId', SemaType.U32),
        ))
        result = SemaNode(SemaNodeKind.TERNARY, ty=ty,
                          children=(cond, src1, src0))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_readfirstlane')
class _VectorReadfirstlane(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(
            _cast(_dst(0), SemaType.U32),
            SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                     call_name='v_readfirstlane',
                     children=(_id('v_readfirstlane'),
                               _cast(_src(0), SemaType.U32))),
        )
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('vector_readlane')
class _VectorReadlane(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(
            _cast(_dst(0), SemaType.U32),
            SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                     call_name='v_readlane',
                     children=(_id('v_readlane'),
                               _cast(_src(0), SemaType.U32),
                               _cast(_src(1), SemaType.U32))),
        )
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('vector_writelane')
class _VectorWritelane(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name='v_writelane',
                          children=(_id('v_writelane'),
                                    _cast(_dst(0), SemaType.U32),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_swap')
class _VectorSwap(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        stmts = [
            _assign(_id('tmp', ty), _cast(_dst(0), ty)),
            _assign(_cast(_dst(0), ty), _cast(_src(0), ty)),
            _assign(_cast(_src(0), ty), _id('tmp', ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_fmaak')
class _VectorFmaak(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.FMA, ty=ty, children=(
            _cast(_src(0), ty), _cast(_src(1), ty), _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_fmamk')
class _VectorFmamk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.FMA, ty=ty, children=(
            _cast(_src(0), ty), _cast(_src(1), ty), _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


# =========================================================================
# Memory semantic classes
# =========================================================================

def _elem_type(elem_size: int | None, sign_extend: bool = False) -> SemaType:
    """Map element byte size to SemaType."""
    sz = (elem_size or 4) * 8
    if sign_extend:
        return SemaType('I', sz)
    return SemaType('B', sz)


def _mem_read(addr: SemaNode, elem_ty: SemaType) -> SemaNode:
    return SemaNode(SemaNodeKind.ARRAYDEREF, ty=elem_ty, children=(
        _id('MEM', elem_ty), addr,
    ))


def _mem_write(addr: SemaNode, value: SemaNode) -> SemaNode:
    return _assign(
        SemaNode(SemaNodeKind.ARRAYDEREF, children=(
            _id('MEM'), addr,
        )),
        value,
    )


def _addr_call(calc_name: str, *args: SemaNode,
               addr_ty: SemaType = SemaType.U32) -> SemaNode:
    return SemaNode(SemaNodeKind.CALL, ty=addr_ty,
                    call_name=calc_name,
                    children=(_id(calc_name), *args))


@_register('smem_load')
class _SmemLoad(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size, sem.sign_extend)
        addr = _addr_call('CalcScalarGlobalAddr',
                          _cast(_src(0), SemaType.U64),
                          _cast(_src(1), SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('smem_store')
class _SmemStore(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcScalarGlobalAddr',
                          _cast(_src(0), SemaType.U64),
                          _cast(_src(1), SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _mem_write(_id('addr', SemaType.U32), _cast(_src(2), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('buffer_load')
class _BufferLoad(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size, sem.sign_extend)
        addr = _addr_call('CalcBufferAddr',
                          _cast(_src(0), SemaType.B32),
                          _cast(_src(1), SemaType.B32),
                          _cast(_src(2), SemaType.B32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('buffer_store')
class _BufferStore(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcBufferAddr',
                          _cast(_src(0), SemaType.B32),
                          _cast(_src(1), SemaType.B32),
                          _cast(_src(2), SemaType.B32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _mem_write(_id('addr', SemaType.U32), _cast(_src(3), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('buffer_atomic')
class _BufferAtomic(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcBufferAddr',
                          _cast(_src(0), SemaType.B32),
                          _cast(_src(1), SemaType.B32),
                          _cast(_src(2), SemaType.B32))
        op = sem.operation or 'add'
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_id('old', elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
            SemaNode(SemaNodeKind.CALL, ty=elem_ty, call_name=f'atomic_{op}',
                     children=(_id(f'atomic_{op}'),
                               _id('addr', SemaType.U32),
                               _cast(_src(3), elem_ty))),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('flat_load')
class _FlatLoad(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size, sem.sign_extend)
        addr = _addr_call('CalcFlatAddr',
                          _cast(_src(0), SemaType.U64),
                          addr_ty=SemaType.U64)
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('flat_store')
class _FlatStore(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcFlatAddr',
                          _cast(_src(0), SemaType.U64),
                          addr_ty=SemaType.U64)
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _mem_write(_id('addr', SemaType.U32), _cast(_src(1), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('flat_atomic')
class _FlatAtomic(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        op = sem.operation or 'add'
        addr = _addr_call('CalcFlatAddr',
                          _cast(_src(0), SemaType.U64),
                          addr_ty=SemaType.U64)
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            SemaNode(SemaNodeKind.CALL, ty=elem_ty, call_name=f'atomic_{op}',
                     children=(_id(f'atomic_{op}'),
                               _id('addr', SemaType.U32),
                               _cast(_src(1), elem_ty))),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_read')
class _DsRead(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size, sem.sign_extend)
        addr = _addr_call('CalcDsAddr',
                          _cast(_src(0), SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_write')
class _DsWrite(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcDsAddr',
                          _cast(_src(0), SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _mem_write(_id('addr', SemaType.U32), _cast(_src(1), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_read2')
class _DsRead2(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr0 = _addr_call('CalcDsAddr',
                           _cast(_src(0), SemaType.U32),
                           _id('OFFSET0', SemaType.U32))
        addr1 = _addr_call('CalcDsAddr',
                           _cast(_src(0), SemaType.U32),
                           _id('OFFSET1', SemaType.U32))
        stmts = [
            _assign(_id('addr0', SemaType.U32), addr0),
            _assign(_id('addr1', SemaType.U32), addr1),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr0', SemaType.U32), elem_ty)),
            _assign(_cast(_dst(1), elem_ty), _mem_read(_id('addr1', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_write2')
class _DsWrite2(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr0 = _addr_call('CalcDsAddr',
                           _cast(_src(0), SemaType.U32),
                           _id('OFFSET0', SemaType.U32))
        addr1 = _addr_call('CalcDsAddr',
                           _cast(_src(0), SemaType.U32),
                           _id('OFFSET1', SemaType.U32))
        stmts = [
            _assign(_id('addr0', SemaType.U32), addr0),
            _assign(_id('addr1', SemaType.U32), addr1),
            _mem_write(_id('addr0', SemaType.U32), _cast(_src(1), elem_ty)),
            _mem_write(_id('addr1', SemaType.U32), _cast(_src(2), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_atomic')
class _DsAtomic(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        op = sem.operation or 'add'
        addr = _addr_call('CalcDsAddr',
                          _cast(_src(0), SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            SemaNode(SemaNodeKind.CALL, ty=elem_ty, call_name=f'atomic_{op}',
                     children=(_id(f'atomic_{op}'),
                               _id('addr', SemaType.U32),
                               _cast(_src(1), elem_ty))),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_read_addtid')
class _DsReadAddtid(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcDsAddr',
                          _id('laneId', SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), elem_ty), _mem_read(_id('addr', SemaType.U32), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_write_addtid')
class _DsWriteAddtid(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        elem_ty = _elem_type(sem.elem_size)
        addr = _addr_call('CalcDsAddr',
                          _id('laneId', SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _mem_write(_id('addr', SemaType.U32), _cast(_src(0), elem_ty)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_permute')
class _DsPermute(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        op = sem.operation or 'bpermute'
        call_name = f'ds_{op}'
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name=call_name,
                          children=(_id(call_name),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_swizzle')
class _DsSwizzle(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name='ds_swizzle',
                          children=(_id('ds_swizzle'),
                                    _cast(_src(0), SemaType.U32),
                                    _id('OFFSET', SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_read_tr_b4')
class _DsReadTrB4(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        addr = _addr_call('CalcDsAddr',
                          _cast(_src(0), SemaType.U32),
                          _id('OFFSET', SemaType.U32))
        stmts = [
            _assign(_id('addr', SemaType.U32), addr),
            _assign(_cast(_dst(0), SemaType.B32),
                    _mem_read(_id('addr', SemaType.U32), SemaType.B32)),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('ds_read_tr_b6')
class _DsReadTrB6(_DsReadTrB4):
    pass


@_register('ds_read_tr_b8')
class _DsReadTrB8(_DsReadTrB4):
    pass


@_register('ds_read_tr_b16')
class _DsReadTrB16(_DsReadTrB4):
    pass


# =========================================================================
# Packed, matrix, special, and remaining semantic classes
# =========================================================================

# --- Packed 16-bit / 32-bit ops ---

@_register('pk_binop_f32')
class _PkBinopF32(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = _vec_binop_expr(sem.operation, _cast(_src(0), ty),
                                 _cast(_src(1), ty), ty)
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('pk_ternary')
class _PkTernary(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.FMA, ty=ty, children=(
            _cast(_src(0), ty), _cast(_src(1), ty), _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('pk_ternary_f32')
class _PkTernaryF32(_PkTernary):
    pass


@_register('pk_mov_b32')
class _PkMovB32(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(_cast(_dst(0), SemaType.B32), _cast(_src(0), SemaType.B32))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


# --- Dot product ops ---

def _derive_dot(sem: InstructionSemantics) -> SemaBlock:
    ty = _dtype_to_sema(sem.data_type)
    result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name=sem.operation or 'dot',
                      children=(_id(sem.operation or 'dot'),
                                _cast(_src(0), ty), _cast(_src(1), ty),
                                _cast(_src(2), ty)))
    body = _assign(_cast(_dst(0), ty), result)
    return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_dot')
class _VectorDot(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return _derive_dot(sem)


@_register('vector_dot2c_bf16')
class _VectorDot2cBf16(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return _derive_dot(sem)


for _dot_cls in ('dot2_f32_f16', 'dot2_i32_i16', 'dot2_u32_u16',
                 'dot4_i32_i8', 'dot4_u32_u8', 'dot4_f32_fp8',
                 'dot8_i32_i4', 'dot8_u32_u4'):
    @_register(_dot_cls)
    class _DotN(_ScalarDeriver):
        @staticmethod
        def derive(sem: InstructionSemantics) -> SemaBlock:
            return _derive_dot(sem)


# --- Mad mix ops ---

@_register('mad_mix_f32')
class _MadMixF32(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = SemaType.F32
        result = SemaNode(SemaNodeKind.FMA, ty=ty, children=(
            _cast(_src(0), ty), _cast(_src(1), ty), _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('mad_mixlo_f16')
class _MadMixloF16(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = SemaType.F32
        result = SemaNode(SemaNodeKind.FMA, ty=ty, children=(
            _cast(_src(0), ty), _cast(_src(1), ty), _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), SemaType.F16), _cast(result, SemaType.F16))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('mad_mixhi_f16')
class _MadMixhiF16(_MadMixloF16):
    pass


# --- Matrix ops ---

@_register('mfma')
class _Mfma(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty,
                          call_name='mfma_compute',
                          children=(_id('mfma_compute'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('accvgpr_read')
class _AccvgprRead(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(_cast(_dst(0), SemaType.B32), _cast(_src(0), SemaType.B32))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('accvgpr_write')
class _AccvgprWrite(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(_cast(_dst(0), SemaType.B32), _cast(_src(0), SemaType.B32))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


# --- Cross-lane / special vector ops ---

@_register('vector_permlane16')
class _VectorPermlane16(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name='v_permlane16',
                          children=(_id('v_permlane16'),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32),
                                    _cast(_src(2), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_permlanex16')
class _VectorPermlanex16(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name='v_permlanex16',
                          children=(_id('v_permlanex16'),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32),
                                    _cast(_src(2), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_mbcnt')
class _VectorMbcnt(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U32,
                          call_name='mbcnt',
                          children=(_id('mbcnt'),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_bitop3')
class _VectorBitop3(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='bitop3',
                          children=(_id('bitop3'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_pack_b32_f16')
class _VectorPackB32F16(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.B32,
                          call_name='pack_b32_f16',
                          children=(_id('pack_b32_f16'),
                                    _cast(_src(0), SemaType.F16),
                                    _cast(_src(1), SemaType.F16)))
        body = _assign(_cast(_dst(0), SemaType.B32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_mad_32_16')
class _VectorMad3216(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='mad_32_16',
                          children=(_id('mad_32_16'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_mad_64_32')
class _VectorMad6432(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = SemaType.U64
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='mad_64_32',
                          children=(_id('mad_64_32'),
                                    _cast(_src(0), SemaType.U32),
                                    _cast(_src(1), SemaType.U32),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_div_fixup')
class _VectorDivFixup(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='div_fixup',
                          children=(_id('div_fixup'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_div_fmas')
class _VectorDivFmas(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='div_fmas',
                          children=(_id('div_fmas'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('vector_div_scale')
class _VectorDivScale(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.CALL, ty=ty, call_name='div_scale',
                          children=(_id('div_scale'),
                                    _cast(_src(0), ty), _cast(_src(1), ty),
                                    _cast(_src(2), ty)))
        body = _assign(_cast(_dst(0), ty), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


# --- Conversion pack ops ---

def _derive_cvt_pk(sem: InstructionSemantics) -> SemaBlock:
    ty = _dtype_to_sema(sem.data_type)
    op = sem.operation or sem.semantic_class
    result = SemaNode(SemaNodeKind.CALL, ty=SemaType.B32, call_name=op,
                      children=(_id(op), _cast(_src(0), ty),
                                _cast(_src(1), ty)))
    body = _assign(_cast(_dst(0), SemaType.B32), result)
    return SemaBlock(sem.name, ExecModel.VECTOR, body)


for _cvt_cls in ('vector_cvt_pk', 'vector_cvt_pk_bf16_f32',
                 'vector_cvt_pk_f16_f32', 'vector_cvt_pk_u8_f32',
                 'vector_cvt_pknorm', 'vector_cvt_pkrtz_f16_f32',
                 'vector_cvt_sr_bf16_f32', 'vector_cvt_sr_f16_f32'):
    @_register(_cvt_cls)
    class _CvtPk(_ScalarDeriver):
        @staticmethod
        def derive(sem: InstructionSemantics) -> SemaBlock:
            return _derive_cvt_pk(sem)


# --- Remaining scalar ops ---

@_register('scalar_movk')
class _ScalarMovk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(_cast(_dst(0), SemaType.U32), _cast(_src(0), SemaType.U32))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_cmovk')
class _ScalarCmovk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.IF, children=(
            _id('SCC', SemaType.U1),
            _assign(_cast(_dst(0), SemaType.U32), _cast(_src(0), SemaType.U32)),
        ))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_addk')
class _ScalarAddk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(
            _cast(_dst(0), SemaType.U32), _cast(_src(0), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_mulk')
class _ScalarMulk(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.MUL, ty=SemaType.U32, children=(
            _cast(_dst(0), SemaType.U32), _cast(_src(0), SemaType.U32)))
        body = _assign(_cast(_dst(0), SemaType.U32), result)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_cselect')
class _ScalarCselect(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        ty = _dtype_to_sema(sem.data_type)
        result = SemaNode(SemaNodeKind.TERNARY, ty=ty, children=(
            _id('SCC', SemaType.U1), _cast(_src(0, ty), ty), _cast(_src(1, ty), ty)))
        body = _assign(_cast(_dst(0, ty), ty), result)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_wrexec')
class _ScalarWrexec(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        stmts = [
            _assign(_id('EXEC', SemaType.U64), _cast(_src(0), SemaType.U64)),
            _scc_write(SemaNode(SemaNodeKind.NE, ty=SemaType.U1, children=(
                _id('EXEC', SemaType.U64), _lit('0', SemaType.U64)))),
        ]
        body = SemaNode(SemaNodeKind.SEQ, children=tuple(stmts))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('scalar_call')
class _ScalarCall(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.U64,
                          call_name='scalar_call',
                          children=(_id('scalar_call'), _cast(_src(0), SemaType.U64)))
        body = _assign(_cast(_dst(0), SemaType.U64), result)
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


# --- Control flow / system ---

@_register('branch')
class _Branch(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.CALL, ty=SemaType.U64,
                        call_name='branch',
                        children=(_id('branch'), _cast(_src(0), SemaType.U64)))
        return SemaBlock(sem.name, ExecModel.BRANCH, body)


@_register('cbranch')
class _Cbranch(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        cond = sem.branch_condition or 'scc1'
        body = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.CALL, ty=SemaType.U1,
                     call_name=f'eval_{cond}',
                     children=(_id(f'eval_{cond}'),)),
            SemaNode(SemaNodeKind.CALL, ty=SemaType.U64,
                     call_name='branch',
                     children=(_id('branch'), _cast(_src(0), SemaType.U64))),
        ))
        return SemaBlock(sem.name, ExecModel.BRANCH, body)


@_register('endpgm')
class _Endpgm(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.CALL, call_name='endpgm',
                        children=(_id('endpgm'),))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('nop')
class _Nop(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return SemaBlock(sem.name, ExecModel.SCALAR,
                         SemaNode(SemaNodeKind.SEQ, children=()))


@_register('true_nop')
class _TrueNop(_Nop):
    pass


@_register('barrier')
class _Barrier(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.CALL, call_name='barrier',
                        children=(_id('barrier'),))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('waitcnt')
class _Waitcnt(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.CALL, call_name='waitcnt',
                        children=(_id('waitcnt'),))
        return SemaBlock(sem.name, ExecModel.SCALAR, body)


@_register('wait_counter')
class _WaitCounter(_Waitcnt):
    pass


@_register('dcache_inv')
class _DcacheInv(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return SemaBlock(sem.name, ExecModel.SCALAR,
                         SemaNode(SemaNodeKind.SEQ, children=()))


@_register('dcache_wb')
class _DcacheWb(_DcacheInv):
    pass


@_register('gl1_inv')
class _Gl1Inv(_DcacheInv):
    pass


@_register('export')
class _Export(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = SemaNode(SemaNodeKind.CALL, call_name='export_pixel',
                        children=(_id('export_pixel'),
                                  _cast(_src(0), SemaType.B32)))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('interp')
class _Interp(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        result = SemaNode(SemaNodeKind.CALL, ty=SemaType.F32,
                          call_name='interp',
                          children=(_id('interp'),
                                    _cast(_src(0), SemaType.F32)))
        body = _assign(_cast(_dst(0), SemaType.F32), result)
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


@_register('lds_direct')
class _LdsDirect(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        body = _assign(_cast(_dst(0), SemaType.B32),
                       SemaNode(SemaNodeKind.CALL, ty=SemaType.B32,
                                call_name='lds_direct_load',
                                children=(_id('lds_direct_load'),)))
        return SemaBlock(sem.name, ExecModel.VECTOR, body)


# --- Image ops ---

def _derive_image(sem: InstructionSemantics, is_store: bool = False) -> SemaBlock:
    op = sem.semantic_class
    ty = SemaType.B32
    if is_store:
        body = SemaNode(SemaNodeKind.CALL, call_name=op,
                        children=(_id(op), _cast(_src(0), ty),
                                  _cast(_src(1), ty)))
    else:
        body = _assign(_cast(_dst(0), ty),
                       SemaNode(SemaNodeKind.CALL, ty=ty, call_name=op,
                                children=(_id(op), _cast(_src(0), ty),
                                          _cast(_src(1), ty))))
    return SemaBlock(sem.name, ExecModel.VECTOR, body)


for _img_cls in ('image_load', 'image_sample', 'image_query', 'image_bvh'):
    @_register(_img_cls)
    class _ImageLoad(_ScalarDeriver):
        @staticmethod
        def derive(sem: InstructionSemantics) -> SemaBlock:
            return _derive_image(sem, is_store=False)


@_register('image_store')
class _ImageStore(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return _derive_image(sem, is_store=True)


@_register('image_atomic')
class _ImageAtomic(_ScalarDeriver):
    @staticmethod
    def derive(sem: InstructionSemantics) -> SemaBlock:
        return _derive_image(sem, is_store=False)
