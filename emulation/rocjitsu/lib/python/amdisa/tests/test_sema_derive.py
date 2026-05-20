# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for mnemonic-to-SemaAST derivation."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_derive import derive_sema_block
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.sema_fingerprint import fingerprint


class _FakeSem:
    """Minimal InstructionSemantics stand-in for testing."""
    def __init__(self, name, semantic_class, operation=None,
                 data_type=None, sets_scc=None):
        self.name = name
        self.semantic_class = semantic_class
        self.operation = operation
        self.data_type = data_type
        self.sets_scc = sets_scc


class TestDeriveScalarMov:
    def test_produces_assign(self):
        sem = _FakeSem('S_MOV_B32', 'scalar_mov', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.SCALAR
        assert block.body.kind == SemaNodeKind.ASSIGN

    def test_lowers_to_cpp(self):
        sem = _FakeSem('S_MOV_B32', 'scalar_mov', data_type='b32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scalar' in cpp
        assert 'read_scalar' in cpp


class TestDeriveScalarCmov:
    def test_produces_if(self):
        sem = _FakeSem('S_CMOV_B32', 'scalar_cmov', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.body.kind == SemaNodeKind.IF

    def test_condition_is_scc(self):
        sem = _FakeSem('S_CMOV_B32', 'scalar_cmov', data_type='b32')
        block = derive_sema_block(sem)
        cond = block.body.children[0]
        assert cond.kind == SemaNodeKind.ID
        assert cond.id_name == 'SCC'


class TestDeriveScalarUnary:
    def test_not(self):
        sem = _FakeSem('S_NOT_B32', 'scalar_unary', 'not', 'b32', 'nonzero')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.BITNEG in all_kinds

    def test_sext8(self):
        sem = _FakeSem('S_SEXT_I32_I8', 'scalar_unary', 'sext8', 'i32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.SIGNEXT_FROM_BIT in all_kinds

    def test_floor(self):
        sem = _FakeSem('S_FLOOR_F32', 'scalar_unary', 'floor', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FLOOR in all_kinds

    def test_with_scc(self):
        sem = _FakeSem('S_NOT_B32', 'scalar_unary', 'not', 'b32', 'nonzero')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp

    def test_lowers_without_error(self):
        for op in ['not', 'brev', 'sext8', 'sext16', 'floor', 'trunc']:
            sem = _FakeSem(f'S_{op.upper()}', 'scalar_unary', op, 'b32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert len(cpp) > 0


class TestDeriveScalarBinop:
    def test_add_u32(self):
        sem = _FakeSem('S_ADD_U32', 'scalar_binop', 'add', 'u32', 'carry')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.ADD in all_kinds

    def test_and_b32(self):
        sem = _FakeSem('S_AND_B32', 'scalar_binop', 'and', 'b32', 'nonzero')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.AND in all_kinds

    def test_nand_b32(self):
        sem = _FakeSem('S_NAND_B32', 'scalar_binop', 'nand', 'b32', 'nonzero')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.AND in all_kinds
        assert SemaNodeKind.BITNEG in all_kinds

    def test_andn2(self):
        sem = _FakeSem('S_ANDN2_B32', 'scalar_binop', 'andn2', 'b32', 'nonzero')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.AND in all_kinds
        assert SemaNodeKind.BITNEG in all_kinds

    def test_scc_carry(self):
        sem = _FakeSem('S_ADD_U32', 'scalar_binop', 'add', 'u32', 'carry')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp

    def test_absdiff(self):
        sem = _FakeSem('S_ABSDIFF_I32', 'scalar_binop', 'absdiff', 'i32', 'nonzero')
        block = derive_sema_block(sem)
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'ABSDIFF' in call_names

    def test_lowers_without_error(self):
        for op in ['add', 'sub', 'and', 'or', 'xor', 'shl', 'shr',
                    'nand', 'nor', 'xnor', 'andn2', 'orn2', 'min', 'max']:
            sem = _FakeSem(f'S_{op.upper()}_B32', 'scalar_binop', op, 'b32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert len(cpp) > 0


class TestDeriveScalarCmp:
    def test_eq(self):
        sem = _FakeSem('S_CMP_EQ_U32', 'scalar_cmp', 'eq', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.body.kind == SemaNodeKind.ASSIGN
        assert block.body.children[0].id_name == 'SCC'

    def test_all_ops(self):
        for op in ['eq', 'ne', 'lt', 'gt', 'le', 'ge']:
            sem = _FakeSem(f'S_CMP_{op.upper()}_U32', 'scalar_cmp', op, 'u32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert 'write_scc' in cpp


class TestDeriveScalarCmpk:
    def test_eq(self):
        sem = _FakeSem('S_CMPK_EQ_U32', 'scalar_cmpk', 'eq', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp


class TestDeriveScalarBitcmp:
    def test_bitset0(self):
        sem = _FakeSem('S_BITCMP0_B32', 'scalar_bitcmp', 'bitset0', 'b32')
        block = derive_sema_block(sem)
        assert block is not None
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp

    def test_bitset1(self):
        sem = _FakeSem('S_BITCMP1_B32', 'scalar_bitcmp', 'bitset1', 'b32')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveScalarBfe:
    def test_bfe(self):
        sem = _FakeSem('S_BFE_U32', 'scalar_bfe', data_type='u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'util::bfe' in call_names
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp


class TestDeriveScalarSaveexec:
    def test_and(self):
        sem = _FakeSem('S_AND_SAVEEXEC_B64', 'scalar_saveexec', 'and')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.AND in all_kinds

    def test_writes_exec_and_scc(self):
        sem = _FakeSem('S_AND_SAVEEXEC_B64', 'scalar_saveexec', 'and')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'set_exec' in cpp
        assert 'write_scc' in cpp

    def test_saves_old_exec(self):
        sem = _FakeSem('S_AND_SAVEEXEC_B64', 'scalar_saveexec', 'and')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scalar' in cpp
        assert 'wf.exec()' in cpp


# =========================================================================
# Vector ALU + cmp
# =========================================================================

class TestDeriveVectorMov:
    def test_produces_vector_pragma(self):
        sem = _FakeSem('V_MOV_B32', 'vector_mov', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR

    def test_lowers_with_lane_loop(self):
        sem = _FakeSem('V_MOV_B32', 'vector_mov', data_type='b32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'for (uint32_t lane' in cpp
        assert 'write_lane' in cpp


class TestDeriveVectorUnary:
    def test_not(self):
        sem = _FakeSem('V_NOT_B32', 'vector_unary', 'not', 'b32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.BITNEG in all_kinds

    def test_sqrt(self):
        sem = _FakeSem('V_SQRT_F32', 'vector_unary', 'sqrt', 'f32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'std::sqrt' in cpp

    def test_floor(self):
        sem = _FakeSem('V_FLOOR_F32', 'vector_unary', 'floor', 'f32')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FLOOR in all_kinds

    def test_lowers_all(self):
        for op in ['not', 'sqrt', 'sin', 'cos', 'floor', 'trunc', 'fract',
                    'rcp', 'rsq', 'log', 'bcnt', 'ffbl']:
            sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_unary', op, 'f32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert len(cpp) > 0


class TestDeriveVectorBinop:
    def test_add_f32(self):
        sem = _FakeSem('V_ADD_F32', 'vector_binop', 'add', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.ADD in all_kinds

    def test_subrev(self):
        sem = _FakeSem('V_SUBREV_U32', 'vector_binop', 'subrev', 'u32')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.SUB in all_kinds

    def test_lshlrev(self):
        sem = _FakeSem('V_LSHLREV_B32', 'vector_binop', 'lshlrev', 'b32')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.SHL in all_kinds

    def test_min_max_use_call(self):
        for op in ['min', 'max']:
            sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_binop', op, 'f32')
            block = derive_sema_block(sem)
            call_names = [n.call_name for n in block.body.walk()
                          if n.kind == SemaNodeKind.CALL]
            assert f'std::f{op}' in call_names
        for op in ['min', 'max']:
            sem = _FakeSem(f'V_{op.upper()}_I32', 'vector_binop', op, 'i32')
            block = derive_sema_block(sem)
            call_names = [n.call_name for n in block.body.walk()
                          if n.kind == SemaNodeKind.CALL]
            assert f'std::{op}' in call_names

    def test_lowers_all(self):
        for op in ['add', 'sub', 'subrev', 'mul', 'and', 'or', 'xor',
                    'shl', 'shr', 'lshlrev', 'lshrrev', 'ashrrev',
                    'min', 'max', 'ldexp']:
            sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_binop', op, 'f32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert 'for (uint32_t lane' in cpp


class TestDeriveVectorTernary:
    def test_fma(self):
        sem = _FakeSem('V_FMA_F32', 'vector_ternary', 'fma', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds

    def test_mad(self):
        sem = _FakeSem('V_MAD_F32', 'vector_ternary', 'mad', 'f32')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.MUL in all_kinds
        assert SemaNodeKind.ADD in all_kinds

    def test_lowers_to_std_fma(self):
        sem = _FakeSem('V_FMA_F32', 'vector_ternary', 'fma', 'f32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'std::fma(' in cpp


class TestDeriveVectorCmp:
    def test_cmp_eq_writes_vcc(self):
        sem = _FakeSem('V_CMP_EQ_F32', 'vector_cmp', 'eq', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR
        ids = {n.id_name for n in block.body.walk()
               if n.kind == SemaNodeKind.ID and n.id_name}
        assert 'VCC' in ids

    def test_all_ops(self):
        for op in ['eq', 'ne', 'lt', 'gt', 'le', 'ge']:
            sem = _FakeSem(f'V_CMP_{op.upper()}_F32', 'vector_cmp', op, 'f32')
            block = derive_sema_block(sem)
            assert block is not None


class TestDeriveVectorCmpx:
    def test_cmpx_writes_exec(self):
        sem = _FakeSem('V_CMPX_EQ_F32', 'vector_cmpx', 'eq', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        ids = {n.id_name for n in block.body.walk()
               if n.kind == SemaNodeKind.ID and n.id_name}
        assert 'EXEC' in ids


class TestDeriveVectorCmpClass:
    def test_cmp_class(self):
        sem = _FakeSem('V_CMP_CLASS_F32', 'vector_cmp_class', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'fp_class_test' in call_names


class TestDeriveVectorAddCo:
    def test_add_co(self):
        sem = _FakeSem('V_ADD_CO_U32', 'vector_add_co', 'add', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = {n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL}
        assert 'add_co' in call_names

    def test_sub_co(self):
        sem = _FakeSem('V_SUB_CO_U32', 'vector_add_co', 'sub', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = {n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL}
        assert 'sub_co' in call_names


class TestDeriveVectorCndmask:
    def test_cndmask(self):
        sem = _FakeSem('V_CNDMASK_B32', 'vector_cndmask', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.TERNARY in all_kinds
        ids = {n.id_name for n in block.body.walk()
               if n.kind == SemaNodeKind.ID and n.id_name}
        assert 'VCC' in ids


class TestDeriveVectorLaneOps:
    def test_readfirstlane_is_scalar(self):
        sem = _FakeSem('V_READFIRSTLANE_B32', 'vector_readfirstlane')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.SCALAR

    def test_readlane_is_scalar(self):
        sem = _FakeSem('V_READLANE_B32', 'vector_readlane')
        block = derive_sema_block(sem)
        assert block.pragma == ExecModel.SCALAR

    def test_writelane(self):
        sem = _FakeSem('V_WRITELANE_B32', 'vector_writelane')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR

    def test_swap(self):
        sem = _FakeSem('V_SWAP_B32', 'vector_swap', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR
        assert block.body.kind == SemaNodeKind.SEQ
        assert len(block.body.children) == 3


class TestDeriveVectorFmaVariants:
    def test_fmaak(self):
        sem = _FakeSem('V_FMAAK_F32', 'vector_fmaak', 'fma', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds

    def test_fmamk(self):
        sem = _FakeSem('V_FMAMK_F32', 'vector_fmamk', 'fma', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds


# =========================================================================
# Memory
# =========================================================================

class TestDeriveSmemLoad:
    def test_produces_scalar(self):
        sem = _FakeSem('S_LOAD_B32', 'smem_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.SCALAR

    def test_has_addr_calc(self):
        sem = _FakeSem('S_LOAD_B32', 'smem_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'CalcScalarGlobalAddr' in call_names

    def test_lowers_to_scalar_mem(self):
        sem = _FakeSem('S_LOAD_B32', 'smem_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'scalar_mem().read' in cpp


class TestDeriveSmemStore:
    def test_store(self):
        sem = _FakeSem('S_STORE_B32', 'smem_store')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.SCALAR


class TestDeriveBufferLoad:
    def test_buffer_load(self):
        sem = _FakeSem('BUFFER_LOAD_B32', 'buffer_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'CalcBufferAddr' in call_names


class TestDeriveFlatLoad:
    def test_flat_load(self):
        sem = _FakeSem('FLAT_LOAD_B32', 'flat_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'CalcFlatAddr' in call_names


class TestDeriveDsRead:
    def test_ds_read(self):
        sem = _FakeSem('DS_LOAD_B32', 'ds_read')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'CalcDsAddr' in call_names

    def test_lowers_to_lds(self):
        sem = _FakeSem('DS_LOAD_B32', 'ds_read')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'for (uint32_t lane' in cpp


class TestDeriveDsRead2:
    def test_ds_read2(self):
        sem = _FakeSem('DS_READ2_B32', 'ds_read2')
        sem.elem_size = 4
        sem.num_elems = 2
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.body.kind == SemaNodeKind.SEQ
        assert len(block.body.children) == 4


class TestDeriveDsAtomic:
    def test_ds_atomic(self):
        sem = _FakeSem('DS_ADD_U32', 'ds_atomic', 'add')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'atomic_add' in call_names
        assert 'CalcDsAddr' in call_names


class TestDeriveDsPermute:
    def test_ds_permute(self):
        sem = _FakeSem('DS_BPERMUTE_B32', 'ds_permute')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'ds_bpermute' in call_names


class TestDeriveDsSwizzle:
    def test_ds_swizzle(self):
        sem = _FakeSem('DS_SWIZZLE_B32', 'ds_swizzle')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveMemoryLowerAll:
    def test_all_memory_classes_lower(self):
        classes = [
            ('S_LOAD_B32', 'smem_load', ExecModel.SCALAR),
            ('S_STORE_B32', 'smem_store', ExecModel.SCALAR),
            ('BUFFER_LOAD_B32', 'buffer_load', ExecModel.VECTOR),
            ('BUFFER_STORE_B32', 'buffer_store', ExecModel.VECTOR),
            ('BUFFER_ATOMIC_ADD_F32', 'buffer_atomic', ExecModel.VECTOR),
            ('FLAT_LOAD_B32', 'flat_load', ExecModel.VECTOR),
            ('FLAT_STORE_B32', 'flat_store', ExecModel.VECTOR),
            ('FLAT_ATOMIC_ADD_F32', 'flat_atomic', ExecModel.VECTOR),
            ('DS_LOAD_B32', 'ds_read', ExecModel.VECTOR),
            ('DS_STORE_B32', 'ds_write', ExecModel.VECTOR),
            ('DS_READ2_B32', 'ds_read2', ExecModel.VECTOR),
            ('DS_WRITE2_B32', 'ds_write2', ExecModel.VECTOR),
            ('DS_ADD_U32', 'ds_atomic', ExecModel.VECTOR),
            ('DS_BPERMUTE_B32', 'ds_permute', ExecModel.VECTOR),
            ('DS_SWIZZLE_B32', 'ds_swizzle', ExecModel.VECTOR),
            ('DS_LOAD_ADDTID_B32', 'ds_read_addtid', ExecModel.VECTOR),
            ('DS_STORE_ADDTID_B32', 'ds_write_addtid', ExecModel.VECTOR),
            ('DS_READ_TR_B32', 'ds_read_tr_b4', ExecModel.VECTOR),
        ]
        for name, cls, expected_pragma in classes:
            sem = _FakeSem(name, cls, 'add')
            sem.elem_size = 4
            sem.num_elems = 1
            sem.sign_extend = False
            block = derive_sema_block(sem)
            assert block is not None, f'{cls} returned None'
            assert block.pragma == expected_pragma, f'{cls}: expected {expected_pragma}, got {block.pragma}'
            cpp = lower_sema_block(block)
            assert len(cpp) > 0, f'{cls} produced empty C++'


# =========================================================================
# Packed, matrix, special, remaining
# =========================================================================

class TestDerivePacked:
    def test_pk_ternary(self):
        sem = _FakeSem('V_PK_FMA_F16', 'pk_ternary', 'fma', 'f16')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds

    def test_pk_binop_f32(self):
        sem = _FakeSem('V_PK_ADD_F32', 'pk_binop_f32', 'add', 'f32')
        block = derive_sema_block(sem)
        assert block is not None

    def test_pk_mov_b32(self):
        sem = _FakeSem('V_PK_MOV_B32', 'pk_mov_b32')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveDot:
    def test_vector_dot(self):
        sem = _FakeSem('V_DOT2_F32_F16', 'vector_dot', 'dot2_f32_f16', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'dot2_f32_f16' in call_names

    def test_dot_variants(self):
        for cls in ['dot2_f32_f16', 'dot4_i32_i8', 'dot8_u32_u4']:
            sem = _FakeSem(f'V_{cls.upper()}', cls, cls)
            block = derive_sema_block(sem)
            assert block is not None


class TestDeriveMfma:
    def test_mfma(self):
        sem = _FakeSem('V_MFMA_F32_16X16X16_F16', 'mfma', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'mfma_compute' in call_names


class TestDerivePermlane:
    def test_permlane16(self):
        sem = _FakeSem('V_PERMLANE16_B32', 'vector_permlane16')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [n.call_name for n in block.body.walk()
                      if n.kind == SemaNodeKind.CALL]
        assert 'v_permlane16' in call_names

    def test_permlanex16(self):
        sem = _FakeSem('V_PERMLANEX16_B32', 'vector_permlanex16')
        block = derive_sema_block(sem)
        assert block is not None

    def test_mbcnt(self):
        sem = _FakeSem('V_MBCNT_LO_U32_B32', 'vector_mbcnt')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveSpecialScalar:
    def test_cselect(self):
        sem = _FakeSem('S_CSELECT_B32', 'scalar_cselect', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.TERNARY in all_kinds

    def test_wrexec(self):
        sem = _FakeSem('S_OR_SAVEEXEC_B64', 'scalar_wrexec')
        block = derive_sema_block(sem)
        assert block is not None
        cpp = lower_sema_block(block)
        assert 'set_exec' in cpp

    def test_movk(self):
        sem = _FakeSem('S_MOVK_I32', 'scalar_movk')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveControlFlow:
    def test_branch(self):
        sem = _FakeSem('S_BRANCH', 'branch')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.BRANCH

    def test_cbranch(self):
        sem = _FakeSem('S_CBRANCH_SCC1', 'cbranch')
        sem.branch_condition = 'scc1'
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.BRANCH

    def test_endpgm(self):
        sem = _FakeSem('S_ENDPGM', 'endpgm')
        block = derive_sema_block(sem)
        assert block is not None

    def test_nop(self):
        sem = _FakeSem('S_NOP', 'nop')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.is_empty

    def test_waitcnt(self):
        sem = _FakeSem('S_WAITCNT', 'waitcnt')
        block = derive_sema_block(sem)
        assert block is not None

    def test_barrier(self):
        sem = _FakeSem('S_BARRIER', 'barrier')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveImage:
    def test_image_load(self):
        sem = _FakeSem('IMAGE_LOAD', 'image_load')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR

    def test_image_store(self):
        sem = _FakeSem('IMAGE_STORE', 'image_store')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveCvtPk:
    def test_cvt_pk(self):
        sem = _FakeSem('V_CVT_PK_F16_F32', 'vector_cvt_pk_f16_f32', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None

    def test_cvt_sr(self):
        sem = _FakeSem('V_CVT_SR_F16_F32', 'vector_cvt_sr_f16_f32', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None


class TestDeriveAllClassesLower:
    def test_all_registered_classes_lower(self):
        from amdisa.sema_derive import _DERIVE_REGISTRY
        errors = []
        for cls_name in sorted(_DERIVE_REGISTRY.keys()):
            sem = _FakeSem(f'TEST_{cls_name.upper()}', cls_name, 'add', 'f32')
            sem.elem_size = 4
            sem.num_elems = 1
            sem.sign_extend = False
            sem.branch_condition = 'scc1'
            try:
                block = derive_sema_block(sem)
                assert block is not None, f'{cls_name} returned None'
                cpp = lower_sema_block(block)
                assert len(cpp) > 0, f'{cls_name} empty C++'
            except Exception as e:
                errors.append(f'{cls_name}: {e}')
        assert errors == [], f'{len(errors)} errors:\n' + '\n'.join(errors[:10])


class TestDeriveUnsupported:
    def test_unknown_class_returns_none(self):
        sem = _FakeSem('UNKNOWN', 'unknown_class', 'add', 'f32')
        block = derive_sema_block(sem)
        assert block is None


class TestDeriveFingerprinting:
    def test_same_op_same_fingerprint(self):
        sem_a = _FakeSem('S_ADD_U32', 'scalar_binop', 'add', 'u32', 'carry')
        sem_b = _FakeSem('S_ADD_U32_V2', 'scalar_binop', 'add', 'u32', 'carry')
        block_a = derive_sema_block(sem_a)
        block_b = derive_sema_block(sem_b)
        assert fingerprint(block_a) == fingerprint(block_b)

    def test_different_op_different_fingerprint(self):
        sem_add = _FakeSem('S_ADD_U32', 'scalar_binop', 'add', 'u32')
        sem_sub = _FakeSem('S_SUB_U32', 'scalar_binop', 'sub', 'u32')
        block_add = derive_sema_block(sem_add)
        block_sub = derive_sema_block(sem_sub)
        assert fingerprint(block_add) != fingerprint(block_sub)
