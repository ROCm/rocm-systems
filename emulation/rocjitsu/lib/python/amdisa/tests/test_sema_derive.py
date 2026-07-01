# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for mnemonic-to-SemaAST derivation."""

import os
import re
from types import SimpleNamespace

import pytest

from amdisa.semantics import derive_semantics
from amdisa.sema_ast import (
    ExecModel,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_derive import derive_sema_block
from amdisa.codegen.execute.sema_lower import (
    LoweringContext,
    OperandBinding,
    OperandMap,
    RegClass,
    lower_sema_block,
)
from amdisa.codegen.execute.packed import gen_pk_binop, gen_pk_ternary
from amdisa.codegen.execute.vector_special import (
    gen_cvt_fp8,
    gen_vector_cvt_pk,
    gen_vector_cvt_scale,
)
from amdisa.sema_fingerprint import fingerprint


class _FakeSem:
    """Minimal InstructionSemantics stand-in for testing."""

    def __init__(
        self, name, semantic_class, operation=None, data_type=None, sets_scc=None
    ):
        self.name = name
        self.semantic_class = semantic_class
        self.operation = operation
        self.data_type = data_type
        self.sets_scc = sets_scc


def test_gfx1250_bf16_fma_mix_semantics_are_explicit():
    cases = {
        'V_FMA_MIX_F32_BF16': 'mad_mix_f32_bf16',
        'V_FMA_MIXLO_BF16': 'mad_mixlo_bf16',
        'V_FMA_MIXHI_BF16': 'mad_mixhi_bf16',
    }

    for name, semantic_class in cases.items():
        sem = derive_semantics(name, 'ENC_VOP3P')
        assert sem is not None
        assert sem.semantic_class == semantic_class


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

    @pytest.mark.parametrize('name', ['S_CLZ_I32_U32', 'S_CLZ_I32_U64'])
    def test_clz_zero_returns_all_ones(self, name):
        sem = derive_semantics(name, 'ENC_SOP1')
        assert sem.semantic_class == 'scalar_unary'
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'static_cast<uint32_t>(-1)' in cpp
        assert '? 32u' not in cpp
        assert '? 64u' not in cpp

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

    def test_fmac_f32_reads_destination(self):
        sem = _FakeSem('S_FMAC_F32', 'scalar_binop', 'fmac', 'f32', 'none')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds
        cpp = lower_sema_block(block)
        assert 'std::fma' in cpp

    def test_scalar_fma_reads_third_source(self):
        sem = _FakeSem('S_FMAAK_F32', 'scalar_binop', 'fma', 'f32', 'none')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FMA in all_kinds
        cpp = lower_sema_block(block)
        assert 'std::fma' in cpp
        assert 'src2' in cpp

    def test_scc_carry(self):
        sem = _FakeSem('S_ADD_U32', 'scalar_binop', 'add', 'u32', 'carry')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp

    def test_signed_mul_uses_unsigned_result_slot(self):
        sem = _FakeSem('S_MUL_I32', 'scalar_binop', 'mul', 'i32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert 'uint32_t result' in cpp
        assert re.search(r'\bint32_t\s+result\b', cpp) is None

    def test_signed_co_uses_unsigned_overflow(self):
        # Signed s_add_co_i32 / s_sub_co_i32 must emulate the hardware's
        # wrap-around add/sub entirely in unsigned (signed overflow is undefined
        # behavior and unnecessary).
        sem = derive_semantics('S_ADD_CO_I32', 'ENC_SOP2')
        assert sem.sets_scc == 'overflow'
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert 'uint32_t result = (s0 + s1)' in cpp
        assert re.search(r'\bint32_t\b', cpp) is None
        assert re.search(r'\bint64_t\b', cpp) is None
        # SCC overflow is detected by the unsigned helper (simd_glue.h).
        assert 'wf.write_scc(signed_add_overflows(s0, s1))' in cpp

        sem = derive_semantics('S_SUB_CO_I32', 'ENC_SOP2')
        assert sem.sets_scc == 'overflow'
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'uint32_t result = (s0 - s1)' in cpp
        assert re.search(r'\bint32_t\b', cpp) is None
        assert re.search(r'\bint64_t\b', cpp) is None
        assert 'wf.write_scc(signed_sub_overflows(s0, s1))' in cpp

    @pytest.mark.parametrize(
        'name,operation,dtype,scc',
        [
            ('S_ADD_CO_U32', 'add', 'u32', 'carry'),
            ('S_SUB_CO_U32', 'sub', 'u32', 'borrow'),
            ('S_ADD_CO_I32', 'add', 'i32', 'overflow'),
            ('S_SUB_CO_I32', 'sub', 'i32', 'overflow'),
            ('S_ADD_CO_CI_U32', 'addc', 'u32', 'carry'),
            ('S_SUB_CO_CI_U32', 'subb', 'u32', 'borrow'),
            ('S_ADD_NC_U64', 'add', 'u64', 'none'),
            ('S_SUB_NC_U64', 'sub', 'u64', 'none'),
            ('S_AND_NOT1_B32', 'andn2', 'b32', 'nonzero'),
            ('S_OR_NOT1_B32', 'orn2', 'b32', 'nonzero'),
            ('S_FMAAK_F32', 'fma', 'f32', 'none'),
            ('S_FMAMK_F32', 'fma', 'f32', 'none'),
        ],
    )
    def test_sop2_carry_forms_derive_scalar_binop(self, name, operation, dtype, scc):
        sem = derive_semantics(name, 'ENC_SOP2')
        assert sem is not None
        assert sem.semantic_class == 'scalar_binop'
        assert sem.operation == operation
        assert sem.data_type == dtype
        assert sem.sets_scc == scc

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        if scc == 'none':
            assert 'write_scc' not in cpp
        else:
            assert 'write_scc' in cpp

    def test_absdiff(self):
        sem = _FakeSem('S_ABSDIFF_I32', 'scalar_binop', 'absdiff', 'i32', 'nonzero')
        block = derive_sema_block(sem)
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'ABSDIFF' in call_names

    def test_lowers_without_error(self):
        for op in [
            'add',
            'sub',
            'and',
            'or',
            'xor',
            'shl',
            'shr',
            'nand',
            'nor',
            'xnor',
            'andn2',
            'orn2',
            'min',
            'max',
        ]:
            sem = _FakeSem(f'S_{op.upper()}_B32', 'scalar_binop', op, 'b32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert len(cpp) > 0

    @pytest.mark.parametrize(
        'name,operation,dtype,expected_cpp',
        [
            ('S_PACK_HL_B32_B16', 'pack_hl', 'b32', '0xFFFFu) << 16'),
            ('S_MIN_NUM_F32', 'min_num', 'f32', 'std::fmin'),
            ('S_MAX_NUM_F32', 'max_num', 'f32', 'std::fmax'),
            ('S_MIN_NUM_F16', 'min_num', 'f16', 'std::fmin'),
            ('S_MAX_NUM_F16', 'max_num', 'f16', 'std::fmax'),
            ('S_MINIMUM_F32', 'minimum', 'f32', 'quiet_NaN'),
            ('S_MAXIMUM_F32', 'maximum', 'f32', 'quiet_NaN'),
            ('S_MINIMUM_F16', 'minimum', 'f16', 'quiet_NaN'),
            ('S_MAXIMUM_F16', 'maximum', 'f16', 'quiet_NaN'),
        ],
    )
    def test_gfx1250_scalar_float_and_pack_ops_derive(
        self, name, operation, dtype, expected_cpp
    ):
        sem = derive_semantics(name, 'ENC_SOP2')
        assert sem is not None
        assert sem.semantic_class == 'scalar_binop'
        assert sem.operation == operation
        assert sem.data_type == dtype
        assert sem.sets_scc == 'none'

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert expected_cpp in cpp
        assert 'write_scc' not in cpp

    def test_gfx1250_scalar_cvt_pk_rtz_uses_rtz_helper(self):
        sem = derive_semantics('S_CVT_PK_RTZ_F16_F32', 'ENC_SOP2')
        assert sem is not None
        assert sem.semantic_class == 'scalar_cvt_pkrtz_f16_f32'
        assert sem.operation == 'cvt_pkrtz_f16_f32'
        assert sem.data_type == 'f32'

        block = derive_sema_block(sem)
        assert block.pragma == ExecModel.SCALAR
        cpp = lower_sema_block(block)
        assert 'util::f32_to_f16_rtz' in cpp
        assert 'write_scc' not in cpp


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

    def test_float_compare_derives_and_lowers(self):
        sem = derive_semantics('S_CMP_LT_F32', 'ENC_SOPC')
        assert sem is not None
        assert sem.semantic_class == 'scalar_cmp'
        assert sem.operation == 'lt'
        assert sem.data_type == 'f32'
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp
        assert 'std::bit_cast<float>' in cpp


class TestDeriveScalarCmpk:
    def test_eq(self):
        sem = _FakeSem('S_CMPK_EQ_U32', 'scalar_cmpk', 'eq', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp


class TestDeriveScalarSopk:
    def test_s_version_is_metadata_noop(self):
        sem = derive_semantics('S_VERSION', 'ENC_SOPK')
        assert sem is not None
        assert sem.semantic_class == 'true_nop'


class TestDeriveScalarBitcmp:
    def test_bitcmp0(self):
        sem = _FakeSem('S_BITCMP0_B32', 'scalar_bitcmp', 'bitcmp0', 'b32')
        block = derive_sema_block(sem)
        assert block is not None
        cpp = lower_sema_block(block)
        assert 'write_scc' in cpp
        assert '& 31' in cpp

    def test_bitcmp1(self):
        sem = _FakeSem('S_BITCMP1_B32', 'scalar_bitcmp', 'bitcmp1', 'b32')
        block = derive_sema_block(sem)
        assert block is not None

    def test_bitcmp_b64_uses_32_bit_index_operand(self):
        sem = _FakeSem('S_BITCMP0_B64', 'scalar_bitcmp', 'bitcmp0', 'b64')
        block = derive_sema_block(sem)
        assert block is not None
        omap = OperandMap.from_operand_names(
            ['ssrc0', 'ssrc1'], [], block.pragma, 'b64', src_widths={1: 32}
        )
        ctx = LoweringContext(exec_model=block.pragma, operand_map=omap)
        cpp = lower_sema_block(block, ctx)
        assert 'ssrc0.read_scalar64(wf)' in cpp
        assert 'ssrc1.read_scalar(wf)' in cpp
        assert 'ssrc1.read_scalar64(wf)' not in cpp
        assert '& 63' in cpp


class TestDeriveScalarBfe:
    def test_bfe(self):
        sem = _FakeSem('S_BFE_U32', 'scalar_bfe', data_type='u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
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

    def test_not1_saveexec_uses_source_and_negated_exec(self):
        sem = _FakeSem(
            'S_AND_NOT1_SAVEEXEC_B32',
            'scalar_saveexec',
            'and_not1',
            'b32',
        )
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'src & (~old_exec)' in cpp
        assert '(~src) & (~old_exec)' not in cpp

    def test_or_not1_saveexec_uses_source_or_negated_exec(self):
        sem = _FakeSem(
            'S_OR_NOT1_SAVEEXEC_B32',
            'scalar_saveexec',
            'or_not1',
            'b32',
        )
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'src | (~old_exec)' in cpp
        assert '(~src) | old_exec' not in cpp


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


class TestDeriveVectorMovrel:
    @pytest.mark.parametrize(
        'name,operation',
        [
            ('V_MOVRELS_B32', 'src'),
            ('V_MOVRELD_B32', 'dst'),
        ],
    )
    def test_movrel_derives_vector_special(self, name, operation):
        sem = derive_semantics(name, 'ENC_VOP1')
        assert sem is not None
        assert sem.semantic_class == 'vector_movrel'
        assert sem.operation == operation
        assert sem.data_type == 'b32'


class TestDeriveScalarPc:
    @pytest.mark.parametrize(
        ('name', 'semantic_class'),
        [
            ('S_GET_PC_I64', 'scalar_getpc'),
            ('S_SET_PC_I64', 'scalar_setpc'),
            ('S_SWAP_PC_I64', 'scalar_swappc'),
            ('S_ADD_PC_I64', 'scalar_addpc'),
        ],
    )
    def test_gfx1250_pc_spelling_derives_pc_semantics(self, name, semantic_class):
        sem = derive_semantics(name, 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.data_type == 'i64'


class TestDeriveScalarBitreplicate:
    def test_bitreplicate_derives_mixed_width_semantics(self):
        sem = derive_semantics('S_BITREPLICATE_B64_B32', 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'scalar_bitreplicate'
        assert sem.data_type == 'b32'
        assert sem.sets_scc is None


class TestDeriveScalarShaderCycles:
    def test_shader_cycles_derives_clock_read_semantics(self):
        sem = derive_semantics('S_GET_SHADER_CYCLES_U64', 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'scalar_shader_cycles'
        assert sem.data_type == 'u64'
        assert sem.sets_scc is None


class TestDeriveScalarSendmsgRtn:
    @pytest.mark.parametrize(
        ('name', 'dtype'),
        [
            ('S_SENDMSG_RTN_B32', 'b32'),
            ('S_SENDMSG_RTN_B64', 'b64'),
        ],
    )
    def test_sendmsg_rtn_derives_return_message_semantics(self, name, dtype):
        sem = derive_semantics(name, 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'scalar_sendmsg_rtn'
        assert sem.data_type == dtype
        assert sem.sets_scc is None


class TestDeriveScalarMovrel:
    @pytest.mark.parametrize(
        ('name', 'operation', 'dtype'),
        [
            ('S_MOVRELS_B32', 'src', 'b32'),
            ('S_MOVRELS_B64', 'src', 'b64'),
            ('S_MOVRELD_B32', 'dst', 'b32'),
            ('S_MOVRELD_B64', 'dst', 'b64'),
            ('S_MOVRELSD_2_B32', 'srcdst2', 'b32'),
        ],
    )
    def test_scalar_movrel_derives_indexed_register_semantics(
        self, name, operation, dtype
    ):
        sem = derive_semantics(name, 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'scalar_movrel'
        assert sem.operation == operation
        assert sem.data_type == dtype
        assert sem.sets_scc is None


class TestDeriveScalarSplitBarrier:
    def test_get_barrier_state_derives_idle_state_read(self):
        sem = derive_semantics('S_GET_BARRIER_STATE', 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'scalar_barrier_state'
        assert sem.data_type == 'b32'
        assert sem.sets_scc is None

    @pytest.mark.parametrize(
        'name',
        [
            'S_BARRIER_INIT',
            'S_BARRIER_JOIN',
            'S_WAKEUP_BARRIER',
        ],
    )
    def test_named_barrier_ops_derive_current_noop_model(self, name):
        sem = derive_semantics(name, 'ENC_SOP1')
        assert sem is not None
        assert sem.semantic_class == 'true_nop'


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
        assert 'sqrt' in cpp

    def test_floor(self):
        sem = _FakeSem('V_FLOOR_F32', 'vector_unary', 'floor', 'f32')
        block = derive_sema_block(sem)
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.FLOOR in all_kinds

    def test_lowers_all(self):
        for op in [
            'not',
            'sqrt',
            'sin',
            'cos',
            'floor',
            'trunc',
            'fract',
            'rcp',
            'rsq',
            'log',
            'bcnt',
            'ffbl',
        ]:
            sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_unary', op, 'f32')
            block = derive_sema_block(sem)
            assert block is not None
            cpp = lower_sema_block(block)
            assert len(cpp) > 0

    @pytest.mark.parametrize(
        ('name', 'enc', 'op', 'scale'),
        [
            ('V_CVT_NORM_I16_F16', 'ENC_VOP1', 'cvt_norm_i16_f16', '32767.0f'),
            ('V_CVT_NORM_I16_F16', 'ENC_VOP3', 'cvt_norm_i16_f16', '32767.0f'),
            ('V_CVT_NORM_U16_F16', 'ENC_VOP1', 'cvt_norm_u16_f16', '65535.0f'),
            ('V_CVT_NORM_U16_F16', 'ENC_VOP3', 'cvt_norm_u16_f16', '65535.0f'),
        ],
    )
    def test_cvt_norm_i16_u16_f16_lowers_to_scaled_saturating_convert(
        self, name, enc, op, scale
    ):
        sem = derive_semantics(name, enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_unary'
        assert sem.operation == op
        assert sem.data_type == 'f16'

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert op not in cpp
        assert 'util::f16_to_f32' in cpp
        assert 'std::isnan' in cpp
        assert 'std::clamp' in cpp
        assert scale in cpp

    @pytest.mark.parametrize('enc', ['ENC_VOP1', 'ENC_VOP3'])
    def test_cos_bf16_lowers_through_shared_transcendental(self, enc):
        sem = derive_semantics('V_COS_BF16', enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_unary'
        assert sem.operation == 'cos'
        assert sem.data_type == 'bf16'

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'util::bf16_to_f32' in cpp
        assert 'amdgpu::transcendental::cos_f32' in cpp
        assert 'util::f32_to_bf16' in cpp
        assert 'UnimplementedInst' not in cpp

    @pytest.mark.parametrize(
        ('name', 'op', 'dtype', 'helper'),
        [
            ('V_RCP_BF16', 'rcp', 'bf16', 'amdgpu::transcendental::rcp_f32'),
            ('V_SQRT_BF16', 'sqrt', 'bf16', 'amdgpu::transcendental::sqrt_f32'),
            ('V_RSQ_BF16', 'rsq', 'bf16', 'amdgpu::transcendental::rsq_f32'),
            ('V_LOG_BF16', 'log2', 'bf16', 'amdgpu::transcendental::log_f32'),
            ('V_EXP_BF16', 'exp2', 'bf16', 'amdgpu::transcendental::exp_f32'),
            ('V_SIN_BF16', 'sin', 'bf16', 'amdgpu::transcendental::sin_f32'),
            ('V_TANH_BF16', 'tanh', 'bf16', 'amdgpu::transcendental::tanh_f32'),
            ('V_TANH_F16', 'tanh', 'f16', 'amdgpu::transcendental::tanh_f32'),
            ('V_TANH_F32', 'tanh', 'f32', 'amdgpu::transcendental::tanh_f32'),
        ],
    )
    @pytest.mark.parametrize('enc', ['ENC_VOP1', 'ENC_VOP3'])
    def test_bf16_and_tanh_transcendentals_lower_through_shared_helpers(
        self, enc, name, op, dtype, helper
    ):
        sem = derive_semantics(name, enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_unary'
        assert sem.operation == op
        assert sem.data_type == dtype

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert helper in cpp
        if dtype == 'bf16':
            assert 'util::bf16_to_f32' in cpp
            assert 'util::f32_to_bf16' in cpp
        if dtype == 'f16':
            assert 'util::f16_to_f32' in cpp
            assert 'util::f32_to_f16' in cpp
        assert 'UnimplementedInst' not in cpp

    @pytest.mark.parametrize('enc', ['ENC_VOP1', 'ENC_VOP3'])
    def test_cvt_off_f32_i4_lowers_to_signed_nibble_table(self, enc):
        sem = derive_semantics('V_CVT_OFF_F32_I4', enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_unary'
        assert sem.operation == 'cvt_off_f32_i4'
        assert sem.data_type == 'f32'

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert '& 0xfu' in cpp
        assert 'nibble -= 16' in cpp
        assert '* 0.0625f' in cpp
        assert 'cvt_off_f32_i4(' not in cpp

    @pytest.mark.parametrize(
        ('name', 'helper'),
        [
            ('V_CVT_F16_FP8', 'util::fp8_e4m3_to_f32'),
            ('V_CVT_F16_BF8', 'util::bf8_e5m2_to_f32'),
        ],
    )
    @pytest.mark.parametrize('enc', ['ENC_VOP1', 'ENC_VOP3'])
    def test_cvt_f16_fp8_bf8_lowers_through_f32_to_half(self, enc, name, helper):
        sem = derive_semantics(name, enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_unary'
        assert sem.operation == name.lower()[2:]
        assert sem.data_type == 'f16'

        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert helper in cpp
        assert 'util::f32_to_f16' in cpp
        assert 'static_cast<uint8_t>' in cpp
        assert f'{sem.operation}(' not in cpp

    @pytest.mark.parametrize(
        ('name', 'helper'),
        [
            ('V_CVT_F16_FP8', 'util::fp8_e4m3_to_f32'),
            ('V_CVT_F16_BF8', 'util::bf8_e5m2_to_f32'),
        ],
    )
    def test_cvt_f16_fp8_bf8_vop3_uses_opsel_byte_select(self, name, helper):
        sem = derive_semantics(name, 'ENC_VOP3')
        assert sem is not None
        block = derive_sema_block(sem)
        ctx = LoweringContext(
            exec_model=block.pragma,
            fp8_byte_select='((inst_.opsel & 0x1u) << 1) | ((inst_.opsel & 0x2u) >> 1)',
        )

        cpp = lower_sema_block(block, ctx)

        assert helper in cpp
        assert (
            '>> ((((inst_.opsel & 0x1u) << 1) | ((inst_.opsel & 0x2u) >> 1)) * 8u)'
            in cpp
        )
        assert '& 0xFFu' in cpp

    def test_cvt_f32_fp8_gfx1250_clamp_selects_e5m3_decode(self):
        sem = derive_semantics('V_CVT_F32_FP8', 'ENC_VOP3')
        assert sem is not None
        block = derive_sema_block(sem)
        ctx = LoweringContext(
            exec_model=block.pragma,
            fp8_byte_select='((inst_.opsel & 0x1u) << 1) | ((inst_.opsel & 0x2u) >> 1)',
            fp8_decode_e5m3_select='inst_.clamp',
        )

        cpp = lower_sema_block(block, ctx)

        assert 'inst_.clamp' in cpp
        assert 'util::fp8_e5m3_to_f32' in cpp
        assert 'util::fp8_e4m3_to_f32' in cpp

    @pytest.mark.parametrize(
        ('name', 'op', 'helper', 'write_fn', 'needs_f16'),
        [
            (
                'V_CVT_PK_F32_FP8',
                'f32_fp8',
                'util::fp8_e4m3_to_f32',
                'write_lane64',
                False,
            ),
            (
                'V_CVT_PK_F32_BF8',
                'f32_bf8',
                'util::bf8_e5m2_to_f32',
                'write_lane64',
                False,
            ),
            (
                'V_CVT_PK_F16_FP8',
                'f16_fp8',
                'util::fp8_e4m3_to_f32',
                'write_lane',
                True,
            ),
            (
                'V_CVT_PK_F16_BF8',
                'f16_bf8',
                'util::bf8_e5m2_to_f32',
                'write_lane',
                True,
            ),
        ],
    )
    @pytest.mark.parametrize('enc', ['ENC_VOP1', 'ENC_VOP3'])
    def test_cvt_pk_fp8_bf8_unpack_conversions_use_packed_generator(
        self, enc, name, op, helper, write_fn, needs_f16
    ):
        sem = derive_semantics(name, enc)
        assert sem is not None
        assert sem.semantic_class == 'vector_cvt_pk'
        assert sem.operation == op

        cpp = gen_vector_cvt_pk(['vdst'], ['src0'], sem.semantic_class, sem.operation)
        assert helper in cpp
        assert 'src_hi' in cpp
        assert 'packed & 0xFFFFu' in cpp
        assert 'half & 0xFFu' in cpp
        assert '(half >> 8) & 0xFFu' in cpp
        assert write_fn in cpp
        assert ('util::f32_to_f16' in cpp) == needs_f16
        assert 'src1' not in cpp

    @pytest.mark.parametrize(
        ('name', 'op', 'helper', 'needs_src1', 'needs_f16'),
        [
            ('V_CVT_PK_FP8_F32', 'fp8_f32', 'util::f32_to_fp8_e4m3_rne', True, False),
            ('V_CVT_PK_BF8_F32', 'bf8_f32', 'util::f32_to_bf8_e5m2_rne', True, False),
            ('V_CVT_PK_FP8_F16', 'fp8_f16', 'util::f32_to_fp8_e4m3_rne', False, True),
            ('V_CVT_PK_BF8_F16', 'bf8_f16', 'util::f32_to_bf8_e5m2_rne', False, True),
        ],
    )
    def test_cvt_pk_fp8_bf8_output_conversions_use_rne_packing(
        self, name, op, helper, needs_src1, needs_f16
    ):
        sem = derive_semantics(name, 'ENC_VOP3')
        assert sem is not None
        assert sem.semantic_class == 'vector_cvt_pk'
        assert sem.operation == op

        src = ['src0', 'src1'] if needs_src1 else ['src0']
        cpp = gen_vector_cvt_pk(['vdst'], src, sem.semantic_class, sem.operation)
        assert helper in cpp
        assert 'static_cast<uint32_t>(lo)' in cpp
        assert 'static_cast<uint32_t>(hi) << 8' in cpp
        assert 'write_vop3_true16_dst' in cpp
        assert ('src1' in cpp) == needs_src1
        assert ('util::f16_to_f32' in cpp) == needs_f16

    def test_gfx1250_cvt_pk_fp8_clamp_selects_e5m3_encoder(self):
        cpp = gen_vector_cvt_pk(
            ['vdst'],
            ['src0', 'src1'],
            'vector_cvt_pk',
            'fp8_f32',
            opsel='inst_.opsel',
            fp8_format_select='inst_.clamp',
        )

        assert 'inst_.clamp' in cpp
        assert 'util::f32_to_fp8_e5m3_rne(s0)' in cpp
        assert 'util::f32_to_fp8_e4m3_rne(s0)' in cpp
        assert 'inst_.opsel' in cpp

    def test_gfx1250_cvt_sr_fp8_clamp_selects_e5m3_encoder(self):
        ctx = SimpleNamespace(
            op='sr_fp8_f32',
            dst_ops=['vdst'],
            src_ops=['src0', 'src1'],
            is_vop3=True,
            enc_field_names={'opsel'},
            encoding_map=None,
            enc_name='',
            arch_name='gfx1250',
        )

        cpp = gen_cvt_fp8(ctx)

        assert 'inst_.clamp' in cpp
        assert 'util::f32_to_fp8_e5m3_sr(s0, seed)' in cpp
        assert 'util::f32_to_fp8_e4m3_sr(s0, seed)' in cpp
        assert 'inst_.opsel' in cpp

    def test_gfx1250_cvt_sr_fp8_f16_clamp_selects_e5m3_encoder(self):
        cpp = gen_vector_cvt_pk(
            ['vdst'],
            ['src0', 'src1'],
            'vector_cvt_sr_fp8_f16',
            None,
            opsel='inst_.opsel',
            fp8_format_select='inst_.clamp',
        )

        assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in cpp
        assert 'inst_.clamp' in cpp
        assert 'util::f32_to_fp8_e5m3_sr(s0, seed)' in cpp
        assert 'util::f32_to_fp8_e4m3_sr(s0, seed)' in cpp

    def test_cvt_pk_bf16_f32_uses_rne_packing(self):
        sem = derive_semantics('V_CVT_PK_BF16_F32', 'ENC_VOP3')
        assert sem is not None
        assert sem.semantic_class == 'vector_cvt_pk_bf16_f32'

        cpp = gen_vector_cvt_pk(
            ['vdst'], ['src0', 'src1'], sem.semantic_class, sem.operation
        )
        assert 'util::f32_to_bf16_rne' in cpp
        assert 'util::f32_to_bf16(s' not in cpp
        assert 'lo | (hi << 16)' in cpp

    @pytest.mark.parametrize(
        ('name', 'op', 'decode_helper', 'encode_helper'),
        [
            (
                'V_CVT_SCALE_PK16_BF16_BF6',
                'unpack_pk16_bf16_bf6',
                'util::bf6_e3m2_to_f32',
                'util::f32_to_bf16',
            ),
            (
                'V_CVT_SCALE_PK8_F32_FP4',
                'unpack_pk8_f32_fp4',
                'util::fp4_e2m1_to_f32',
                'std::bit_cast<uint32_t>',
            ),
        ],
    )
    def test_cvt_scale_unpack_conversions_use_scaled_generator(
        self, name, op, decode_helper, encode_helper
    ):
        sem = derive_semantics(name, 'ENC_VOP3')
        assert sem is not None
        assert sem.semantic_class == 'vector_cvt_scale'
        assert sem.operation == op

        cpp = gen_vector_cvt_scale(
            ['vdst'], ['src0', 'src1'], sem.semantic_class, sem.operation
        )
        assert decode_helper in cpp
        assert encode_helper in cpp
        assert 'util::e8m0_to_f32' in cpp
        assert '((inst_.opsel & 0x3u) * 8u)' in cpp
        assert 'std::bit_cast<float>(src1.read_lane(wf, lane))' not in cpp
        assert 'read_scaled_src(index) * scale' in cpp
        assert 'Isa::resolved_vgpr_offset' in cpp
        assert 'wf.cu().write_vgpr' in cpp

    @pytest.mark.parametrize(
        ('name', 'op', 'read_helper', 'encode_helper'),
        [
            (
                'V_CVT_SCALEF32_PK16_BF6_BF16',
                'pack_pk16_bf6_bf16',
                'util::bf16_to_f32',
                'util::f32_to_bf6_e3m2_rne',
            ),
            (
                'V_CVT_SCALEF32_PK8_FP4_F32',
                'pack_pk8_fp4_f32',
                'std::bit_cast<float>',
                'util::f32_to_fp4_e2m1_rne',
            ),
        ],
    )
    def test_cvt_scalef32_pack_conversions_use_scaled_generator(
        self, name, op, read_helper, encode_helper
    ):
        sem = derive_semantics(name, 'ENC_VOP3')
        assert sem is not None
        assert sem.semantic_class == 'vector_cvt_scale'
        assert sem.operation == op

        cpp = gen_vector_cvt_scale(
            ['vdst'], ['src0', 'src1'], sem.semantic_class, sem.operation
        )
        assert read_helper in cpp
        assert encode_helper in cpp
        assert 'std::bit_cast<float>(src1.read_lane(wf, lane))' in cpp
        assert 'util::e8m0_to_f32' not in cpp
        assert 'pack_scaled_dst(index' in cpp
        assert 'read_scaled_input(index) / scale' in cpp
        assert 'Isa::resolved_vgpr_offset' in cpp


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

    def test_i24_mul_lowers_through_unsigned_helper(self):
        sem = _FakeSem('V_MUL_I32_I24', 'vector_binop', 'mul_i24', 'i24')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert '::rocjitsu::amdgpu::mul_i24_u32' in cpp
        assert 'a * b' not in cpp

    def test_min_max_use_call(self):
        for op in ['min', 'max']:
            sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_binop', op, 'f32')
            block = derive_sema_block(sem)
            call_names = [
                n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
            ]
            assert f'std::f{op}' in call_names
        for op in ['min', 'max']:
            sem = _FakeSem(f'V_{op.upper()}_I32', 'vector_binop', op, 'i32')
            block = derive_sema_block(sem)
            call_names = [
                n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
            ]
            assert f'std::{op}' in call_names

    def test_lowers_all(self):
        for op in [
            'add',
            'sub',
            'subrev',
            'mul',
            'and',
            'or',
            'xor',
            'shl',
            'shr',
            'lshlrev',
            'lshrrev',
            'ashrrev',
            'min',
            'max',
            'ldexp',
        ]:
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

    def test_add_minmax_i32_u32_wraps_add_before_clamp(self):
        cases = [
            ('V_ADD_MAX_I32', 'add_max', 'i32', 'std::max'),
            ('V_ADD_MAX_U32', 'add_max', 'u32', 'std::max'),
            ('V_ADD_MIN_I32', 'add_min', 'i32', 'std::min'),
            ('V_ADD_MIN_U32', 'add_min', 'u32', 'std::min'),
        ]
        for name, op, dtype, clamp_fn in cases:
            sem = derive_semantics(name, 'ENC_VOP3')
            assert sem is not None
            assert sem.semantic_class == 'vector_ternary'
            assert sem.operation == op
            assert sem.data_type == dtype

            block = derive_sema_block(sem)
            cpp = lower_sema_block(block)
            assert clamp_fn in cpp
            assert 'static_cast<uint32_t>' in cpp
            assert f'{op}_{dtype}(' not in cpp

    def test_ashr_pk_i8_u8_i32_packs_shifted_saturated_bytes(self):
        cases = [
            ('V_ASHR_PK_I8_I32', 'ashr_pk_i8_i32', '-128', '127'),
            ('V_ASHR_PK_U8_I32', 'ashr_pk_u8_i32', '0', '255'),
        ]
        for name, op, clamp_lo, clamp_hi in cases:
            sem = derive_semantics(name, 'ENC_VOP3')
            assert sem is not None
            assert sem.semantic_class == 'vector_ternary'
            assert sem.operation == op
            assert sem.data_type == 'b32'

            block = derive_sema_block(sem)
            cpp = lower_sema_block(block)
            assert 'static_cast<int32_t>(src)' in cpp
            assert 'shift = static_cast<uint32_t>' in cpp
            assert f'static_cast<int32_t>({clamp_lo})' in cpp
            assert f'static_cast<int32_t>({clamp_hi})' in cpp
            assert 'return pack(' in cpp
            assert '| (pack(' in cpp
            assert ' << 8);' in cpp
            assert f'{op}(' not in cpp

    def test_lshl_add_lowers_through_masked_helper(self):
        sem = _FakeSem('V_LSHL_ADD_U32', 'vector_ternary', 'lshl_add', 'u32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert '::rocjitsu::amdgpu::lshl_masked' in cpp
        assert (
            'inst.src0.read_lane(wf, lane) << inst.src1.read_lane(wf, lane)' not in cpp
        )

    def test_i24_mad_lowers_through_unsigned_helper(self):
        sem = _FakeSem('V_MAD_I32_I24', 'vector_ternary', 'mad_i24', 'i24')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert '::rocjitsu::amdgpu::mad_i24_u32' in cpp
        assert 'a * b' not in cpp

    def test_signed_bfe_keeps_braced_one_literal(self):
        sem = _FakeSem('V_BFE_I32', 'vector_ternary', 'bfe_i', 'i32')
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)

        assert 'uint32_t{1}' in cpp
        assert 'uint32_tstatic_cast' not in cpp


class TestDeriveVectorCmp:
    def test_cmp_eq_writes_vcc(self):
        sem = _FakeSem('V_CMP_EQ_F32', 'vector_cmp', 'eq', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR
        ids = {
            n.id_name
            for n in block.body.walk()
            if n.kind == SemaNodeKind.ID and n.id_name
        }
        assert 'VCC' in ids

    def test_all_ops(self):
        for op in ['eq', 'ne', 'lt', 'gt', 'le', 'ge']:
            sem = _FakeSem(f'V_CMP_{op.upper()}_F32', 'vector_cmp', op, 'f32')
            block = derive_sema_block(sem)
            assert block is not None

    def test_true16_cmp_lowering_selects_sources_and_writes_wave32_mask(self):
        sem = _FakeSem('V_CMP_LT_I16', 'vector_cmp', 'lt', 'i16')
        block = derive_sema_block(sem)
        assert block is not None
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 32),
                1: OperandBinding('src1', RegClass.VGPR, 32),
            },
            dst_bindings={0: OperandBinding('vdst', RegClass.SGPR, 64)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_src_selects={
                0: 'inst_.opsel & 0x1u',
                1: 'inst_.opsel & 0x2u',
            },
            clear_false_lane_mask_writes=False,
        )

        cpp = lower_sema_block(block, ctx)

        assert '((inst_.opsel & 0x1u) != 0 ? (src0.read_lane(wf, lane) >> 16)' in cpp
        assert '((inst_.opsel & 0x2u) != 0 ? (src1.read_lane(wf, lane) >> 16)' in cpp
        assert 'vcc &= ~(1ULL << lane)' not in cpp
        assert 'vdst.write_scalar(wf, static_cast<uint32_t>(vcc));' in cpp
        assert 'vdst.write_scalar64(wf, vcc);' in cpp


class TestDeriveVectorCmpx:
    def test_cmpx_writes_exec(self):
        sem = _FakeSem('V_CMPX_EQ_F32', 'vector_cmpx', 'eq', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        ids = {
            n.id_name
            for n in block.body.walk()
            if n.kind == SemaNodeKind.ID and n.id_name
        }
        assert 'EXEC' in ids


class TestDeriveVectorCmpClass:
    def test_cmp_class(self):
        sem = _FakeSem('V_CMP_CLASS_F32', 'vector_cmp_class', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'fp_class_test' in call_names


class TestDeriveVectorAddCo:
    def test_add_co(self):
        sem = _FakeSem('V_ADD_CO_U32', 'vector_add_co', 'add', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = {
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        }
        assert 'add_co' in call_names

    def test_sub_co(self):
        sem = _FakeSem('V_SUB_CO_U32', 'vector_add_co', 'sub', 'u32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = {
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        }
        assert 'sub_co' in call_names


class TestDeriveVectorCndmask:
    def test_cndmask(self):
        sem = _FakeSem('V_CNDMASK_B32', 'vector_cndmask', data_type='b32')
        block = derive_sema_block(sem)
        assert block is not None
        all_kinds = {n.kind for n in block.body.walk()}
        assert SemaNodeKind.TERNARY in all_kinds
        ids = {
            n.id_name
            for n in block.body.walk()
            if n.kind == SemaNodeKind.ID and n.id_name
        }
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
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'CalcScalarGlobalAddr' in call_names

    def test_lowers_to_scalar_mem(self):
        sem = _FakeSem('S_LOAD_B32', 'smem_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'scalar_mem().read' in cpp

    @pytest.mark.parametrize(
        ('name', 'elem_size', 'sign_extend'),
        [
            ('S_LOAD_I8', 1, True),
            ('S_LOAD_U8', 1, False),
            ('S_LOAD_I16', 2, True),
            ('S_LOAD_U16', 2, False),
            ('S_BUFFER_LOAD_I8', 1, True),
            ('S_BUFFER_LOAD_U8', 1, False),
            ('S_BUFFER_LOAD_I16', 2, True),
            ('S_BUFFER_LOAD_U16', 2, False),
        ],
    )
    def test_gfx1250_narrow_loads(self, name, elem_size, sign_extend):
        sem = derive_semantics(name, 'ENC_SMEM')
        assert sem is not None
        assert sem.semantic_class == 'smem_load'
        assert sem.elem_size == elem_size
        assert sem.num_elems == 1
        assert sem.sign_extend is sign_extend


class TestDeriveSmemStore:
    def test_store(self):
        sem = _FakeSem('S_STORE_B32', 'smem_store')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.SCALAR


class TestDeriveSmemHints:
    @pytest.mark.parametrize(
        'name',
        [
            'S_PREFETCH_INST',
            'S_PREFETCH_INST_PC_REL',
            'S_PREFETCH_DATA',
            'S_PREFETCH_DATA_PC_REL',
            'S_BUFFER_PREFETCH_DATA',
            'S_ATC_PROBE',
            'S_ATC_PROBE_BUFFER',
        ],
    )
    def test_gfx1250_prefetch_hints_are_true_nop(self, name):
        sem = derive_semantics(name, 'ENC_SMEM')
        assert sem is not None
        assert sem.semantic_class == 'true_nop'
        block = derive_sema_block(sem)
        assert block is not None
        assert block.is_empty


class TestDeriveBufferLoad:
    def test_buffer_load(self):
        sem = _FakeSem('BUFFER_LOAD_B32', 'buffer_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        assert block.pragma == ExecModel.VECTOR
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'CalcBufferAddr' in call_names


class TestDeriveFlatLoad:
    def test_flat_load(self):
        sem = _FakeSem('FLAT_LOAD_B32', 'flat_load')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'CalcFlatAddr' in call_names

    @pytest.mark.parametrize(
        ('name', 'num_elems', 'transpose_kind'),
        [
            ('GLOBAL_LOAD_TR4_B64', 2, 1),
            ('GLOBAL_LOAD_TR6_B96', 3, 2),
            ('GLOBAL_LOAD_TR8_B64', 2, 3),
            ('GLOBAL_LOAD_TR_B64', 2, 3),
            ('GLOBAL_LOAD_TR16_B128', 4, 4),
            ('GLOBAL_LOAD_TR_B128', 4, 4),
        ],
    )
    def test_gfx1250_global_transpose_loads(self, name, num_elems, transpose_kind):
        sem = derive_semantics(name, 'ENC_VGLOBAL')
        assert sem is not None
        assert sem.semantic_class == 'flat_load'
        assert sem.elem_size == 4
        assert sem.num_elems == num_elems
        assert sem.transpose_kind == transpose_kind

    @pytest.mark.parametrize(
        ('name', 'semantic_class'),
        [
            ('GLOBAL_INV', 'gl1_inv'),
            ('GLOBAL_WB', 'true_nop'),
            ('GLOBAL_WBINV', 'gl1_wbinv'),
        ],
    )
    def test_gfx1250_global_cache_control(self, name, semantic_class):
        sem = derive_semantics(name, 'ENC_VGLOBAL')
        assert sem is not None
        assert sem.semantic_class == semantic_class

    @pytest.mark.parametrize(
        ('name', 'enc_name'),
        [
            ('FLAT_PREFETCH_B8', 'ENC_VFLAT'),
            ('GLOBAL_PREFETCH_B8', 'ENC_VGLOBAL'),
        ],
    )
    def test_gfx1250_flat_global_prefetch_hints_are_true_nop(self, name, enc_name):
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == 'true_nop'
        block = derive_sema_block(sem)
        assert block is not None
        assert block.is_empty

    @pytest.mark.parametrize(
        ('name', 'num_elems'),
        [
            ('CLUSTER_LOAD_B32', 1),
            ('CLUSTER_LOAD_B64', 2),
            ('CLUSTER_LOAD_B128', 4),
        ],
    )
    def test_gfx1250_cluster_loads_use_global_load_path(self, name, num_elems):
        sem = derive_semantics(name, 'ENC_VGLOBAL')
        assert sem is not None
        assert sem.semantic_class == 'flat_load'
        assert sem.elem_size == 4
        assert sem.num_elems == num_elems

    @pytest.mark.parametrize(
        ('name', 'semantic_class'),
        [
            ('GLOBAL_LOAD_ADDTID_B32', 'global_load_addtid'),
            ('GLOBAL_STORE_ADDTID_B32', 'global_store_addtid'),
        ],
    )
    def test_gfx1250_global_addtid_uses_lane_id_addressing(self, name, semantic_class):
        sem = derive_semantics(name, 'ENC_VGLOBAL')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.elem_size == 4
        assert sem.num_elems == 1

    @pytest.mark.parametrize(
        ('name', 'enc_name', 'semantic_class'),
        [
            ('GLOBAL_LOAD_BLOCK', 'ENC_VGLOBAL', 'flat_load'),
            ('GLOBAL_STORE_BLOCK', 'ENC_VGLOBAL', 'flat_store'),
            ('SCRATCH_LOAD_BLOCK', 'ENC_VSCRATCH', 'flat_load'),
            ('SCRATCH_STORE_BLOCK', 'ENC_VSCRATCH', 'flat_store'),
        ],
    )
    def test_gfx1250_block_memory_uses_32_dword_flat_path(
        self, name, enc_name, semantic_class
    ):
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.elem_size == 4
        assert sem.num_elems == 32

    @pytest.mark.parametrize(
        ('name', 'enc_name', 'num_elems'),
        [
            ('FLAT_LOAD_MONITOR_B32', 'ENC_VFLAT', 1),
            ('FLAT_LOAD_MONITOR_B64', 'ENC_VFLAT', 2),
            ('FLAT_LOAD_MONITOR_B128', 'ENC_VFLAT', 4),
            ('GLOBAL_LOAD_MONITOR_B32', 'ENC_VGLOBAL', 1),
            ('GLOBAL_LOAD_MONITOR_B64', 'ENC_VGLOBAL', 2),
            ('GLOBAL_LOAD_MONITOR_B128', 'ENC_VGLOBAL', 4),
        ],
    )
    def test_gfx1250_monitor_loads_use_flat_load_path(self, name, enc_name, num_elems):
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == 'flat_load'
        assert sem.elem_size == 4
        assert sem.num_elems == num_elems
        assert sem.sign_extend is False

    @pytest.mark.parametrize(
        ('name', 'semantic_class', 'elem_size', 'num_elems'),
        [
            ('GLOBAL_LOAD_ASYNC_TO_LDS_B8', 'global_load_async_to_lds', 1, 1),
            ('GLOBAL_LOAD_ASYNC_TO_LDS_B32', 'global_load_async_to_lds', 4, 1),
            ('GLOBAL_LOAD_ASYNC_TO_LDS_B64', 'global_load_async_to_lds', 4, 2),
            ('GLOBAL_LOAD_ASYNC_TO_LDS_B128', 'global_load_async_to_lds', 4, 4),
            ('CLUSTER_LOAD_ASYNC_TO_LDS_B8', 'global_load_async_to_lds', 1, 1),
            ('CLUSTER_LOAD_ASYNC_TO_LDS_B32', 'global_load_async_to_lds', 4, 1),
            ('CLUSTER_LOAD_ASYNC_TO_LDS_B64', 'global_load_async_to_lds', 4, 2),
            ('CLUSTER_LOAD_ASYNC_TO_LDS_B128', 'global_load_async_to_lds', 4, 4),
            ('GLOBAL_STORE_ASYNC_FROM_LDS_B8', 'global_store_async_from_lds', 1, 1),
            ('GLOBAL_STORE_ASYNC_FROM_LDS_B32', 'global_store_async_from_lds', 4, 1),
            ('GLOBAL_STORE_ASYNC_FROM_LDS_B64', 'global_store_async_from_lds', 4, 2),
            ('GLOBAL_STORE_ASYNC_FROM_LDS_B128', 'global_store_async_from_lds', 4, 4),
        ],
    )
    def test_gfx1250_global_cluster_async_lds_ops(
        self, name, semantic_class, elem_size, num_elems
    ):
        sem = derive_semantics(name, 'ENC_VGLOBAL')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.elem_size == elem_size
        assert sem.num_elems == num_elems

    @pytest.mark.parametrize(
        ('name', 'enc_name', 'semantic_class', 'elem_size', 'num_elems'),
        [
            ('GLOBAL_ATOMIC_CMPSWAP_B32', 'ENC_VGLOBAL', 'flat_atomic', 4, 2),
            ('FLAT_ATOMIC_CMPSWAP', 'ENC_VFLAT', 'flat_atomic', 4, 2),
            ('FLAT_ATOMIC_CMPSWAP_B32', 'ENC_VFLAT', 'flat_atomic', 4, 2),
            ('BUFFER_ATOMIC_CMPSWAP_B32', 'ENC_VBUFFER', 'buffer_atomic', 4, 2),
            ('GLOBAL_ATOMIC_CMPSWAP_B64', 'ENC_VGLOBAL', 'flat_atomic', 8, 4),
            ('FLAT_ATOMIC_CMPSWAP_X2', 'ENC_VFLAT', 'flat_atomic', 8, 4),
            ('FLAT_ATOMIC_CMPSWAP_B64', 'ENC_VFLAT', 'flat_atomic', 8, 4),
            ('BUFFER_ATOMIC_CMPSWAP_B64', 'ENC_VBUFFER', 'buffer_atomic', 8, 4),
        ],
    )
    def test_gfx1250_compare_swap_memory_element_and_source_payload_width(
        self, name, enc_name, semantic_class, elem_size, num_elems
    ):
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.operation == 'cmpswap'
        assert sem.elem_size == elem_size
        assert sem.num_elems == num_elems

    @pytest.mark.parametrize(
        ('name', 'enc_name', 'semantic_class'),
        [
            ('GLOBAL_ATOMIC_ADD_U64', 'ENC_VGLOBAL', 'flat_atomic'),
            ('FLAT_ATOMIC_ADD_U64', 'ENC_VFLAT', 'flat_atomic'),
            ('BUFFER_ATOMIC_ADD_U64', 'ENC_VBUFFER', 'buffer_atomic'),
        ],
    )
    def test_gfx1250_u64_atomic_source_payload_uses_two_dwords(
        self, name, enc_name, semantic_class
    ):
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.operation == 'add'
        assert sem.elem_size == 8
        assert sem.num_elems == 2


class TestDeriveDsRead:
    def test_ds_read(self):
        sem = _FakeSem('DS_LOAD_B32', 'ds_read')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'CalcDsAddr' in call_names

    def test_lowers_to_lds(self):
        sem = _FakeSem('DS_LOAD_B32', 'ds_read')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        cpp = lower_sema_block(block)
        assert 'for (uint32_t lane' in cpp

    @pytest.mark.parametrize(
        ('name', 'semantic_class', 'num_elems', 'transpose_kind'),
        [
            ('DS_LOAD_TR4_B64', 'ds_read_tr_b4', 2, 1),
            ('DS_LOAD_TR6_B96', 'ds_read_tr_b6', 3, 2),
            ('DS_LOAD_TR8_B64', 'ds_read_tr_b8', 2, 3),
            ('DS_LOAD_TR_B64', 'ds_read_tr_b8', 2, 3),
            ('DS_LOAD_TR16_B128', 'ds_read_tr_b16', 4, 4),
            ('DS_LOAD_TR_B128', 'ds_read_tr_b16', 4, 4),
            ('DS_READ_B64_TR_B16', 'ds_read_tr_b16', 2, 4),
        ],
    )
    def test_gfx1250_ds_transpose_loads(
        self, name, semantic_class, num_elems, transpose_kind
    ):
        sem = derive_semantics(name, 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.elem_size == 4
        assert sem.num_elems == num_elems
        assert sem.transpose_kind == transpose_kind

    def test_gfx1250_ds_nop_is_true_nop(self):
        sem = derive_semantics('DS_NOP', 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == 'true_nop'
        block = derive_sema_block(sem)
        assert block is not None
        assert block.is_empty


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
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'atomic_add' in call_names
        assert 'CalcDsAddr' in call_names

    @pytest.mark.parametrize(
        ('name', 'semantic_class', 'operation', 'elem_size', 'num_elems'),
        [
            ('DS_MSKOR_B32', 'ds_mskor', 'mskor', 4, 2),
            ('DS_MSKOR_RTN_B32', 'ds_mskor', 'mskor', 4, 2),
            ('DS_MSKOR_B64', 'ds_mskor', 'mskor', 8, 4),
            ('DS_MSKOR_RTN_B64', 'ds_mskor', 'mskor', 8, 4),
            ('DS_APPEND', 'ds_append_consume', 'append', 4, 1),
            ('DS_CONSUME', 'ds_append_consume', 'consume', 4, 1),
            (
                'DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64',
                'ds_barrier_arrive',
                'async_barrier_arrive',
                8,
                0,
            ),
            (
                'DS_ATOMIC_BARRIER_ARRIVE_RTN_B64',
                'ds_barrier_arrive',
                'barrier_arrive',
                8,
                2,
            ),
        ],
    )
    def test_gfx1250_ds_special_atomics(
        self, name, semantic_class, operation, elem_size, num_elems
    ):
        sem = derive_semantics(name, 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.operation == operation
        assert sem.elem_size == elem_size
        assert sem.num_elems == num_elems


class TestDeriveDsPermute:
    def test_ds_permute(self):
        sem = _FakeSem('DS_BPERMUTE_B32', 'ds_permute')
        sem.elem_size = 4
        sem.num_elems = 1
        sem.sign_extend = False
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'ds_bpermute' in call_names

    def test_gfx1250_ds_bpermute_fi(self):
        sem = derive_semantics('DS_BPERMUTE_FI_B32', 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == 'ds_permute'


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
            assert (
                block.pragma == expected_pragma
            ), f'{cls}: expected {expected_pragma}, got {block.pragma}'
            cpp = lower_sema_block(block)
            assert len(cpp) > 0, f'{cls} produced empty C++'


# =========================================================================
# Packed, matrix, special, remaining
# =========================================================================


class TestDerivePacked:
    @pytest.mark.parametrize(
        ('name', 'semantic_class', 'operation'),
        [
            ('V_PK_ADD_BF16', 'pk_binop', 'add'),
            ('V_PK_MUL_BF16', 'pk_binop', 'mul'),
            ('V_PK_MIN_NUM_BF16', 'pk_binop', 'min'),
            ('V_PK_MAX_NUM_BF16', 'pk_binop', 'max'),
            ('V_PK_FMA_BF16', 'pk_ternary', 'fma'),
        ],
    )
    def test_bf16_pk_ops_map_to_packed_generators(
        self, name, semantic_class, operation
    ):
        sem = derive_semantics(name, 'ENC_VOP3P')
        assert sem is not None
        assert sem.semantic_class == semantic_class
        assert sem.operation == operation
        assert sem.data_type == 'bf16'

    def test_pk_binop_bf16_generator_uses_bf16_helpers(self):
        cpp = gen_pk_binop(
            ['vdst'], ['src0', 'src1'], 'add', 'bf16', ('inst_.opsel', 'inst_.opsel_hi')
        )
        assert 'util::bf16_to_f32' in cpp
        assert 'util::f32_to_bf16' in cpp
        assert 'util::f32_to_f16' not in cpp

    def test_pk_ternary_bf16_generator_uses_fma_and_bf16_helpers(self):
        cpp = gen_pk_ternary(
            ['vdst'],
            ['src0', 'src1', 'src2'],
            'fma',
            'bf16',
            '((inst_.opsel_hi >> 2) & 1)',
            ('inst_.opsel', 'inst_.opsel_hi'),
        )
        assert 'std::fma' in cpp
        assert 'util::bf16_to_f32' in cpp
        assert 'util::f32_to_bf16' in cpp

    @pytest.mark.parametrize(
        ('name', 'operation', 'data_type'),
        [
            ('V_PK_MAX3_I16', 'max3', 'i16'),
            ('V_PK_MAX3_U16', 'max3', 'u16'),
            ('V_PK_MIN3_I16', 'min3', 'i16'),
            ('V_PK_MIN3_U16', 'min3', 'u16'),
            ('V_PK_MIN3_NUM_F16', 'min3', 'f16'),
            ('V_PK_MAX3_NUM_F16', 'max3', 'f16'),
        ],
    )
    def test_pk_min3_max3_ops_map_to_packed_ternary(self, name, operation, data_type):
        sem = derive_semantics(name, 'ENC_VOP3P')
        assert sem is not None
        assert sem.semantic_class == 'pk_ternary'
        assert sem.operation == operation
        assert sem.data_type == data_type

    def test_pk_ternary_i16_min3_uses_component_min(self):
        cpp = gen_pk_ternary(
            ['vdst'],
            ['src0', 'src1', 'src2'],
            'min3',
            'i16',
            '((inst_.opsel_hi >> 2) & 1)',
            ('inst_.opsel', 'inst_.opsel_hi'),
        )
        assert 'std::min(std::min(a_lo, b_lo), c_lo)' in cpp
        assert 'a_lo * b_lo + c_lo' not in cpp

    def test_pk_ternary_u16_max3_uses_component_max(self):
        cpp = gen_pk_ternary(
            ['vdst'],
            ['src0', 'src1', 'src2'],
            'max3',
            'u16',
            '((inst_.opsel_hi >> 2) & 1)',
            ('inst_.opsel', 'inst_.opsel_hi'),
        )
        assert 'std::max(std::max(a_lo, b_lo), c_lo)' in cpp
        assert 'a_lo * b_lo + c_lo' not in cpp

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
    def test_rdna3_dot2acc_vop2_is_functional_dot2c(self):
        sem = derive_semantics('V_DOT2ACC_F32_F16', 'ENC_VOP2')
        assert sem is not None
        assert sem.semantic_class == 'vector_dot'
        assert sem.operation == 'dot2c'
        assert sem.data_type == 'f32'

    def test_vop3_dot2_true16_semantics(self):
        cases = {
            'V_DOT2_F16_F16': 'dot2_f16_f16',
            'V_DOT2_BF16_BF16': 'dot2_bf16_bf16',
        }
        for name, semantic_class in cases.items():
            sem = derive_semantics(name, 'ENC_VOP3')
            assert sem is not None
            assert sem.semantic_class == semantic_class

    def test_vector_dot(self):
        sem = _FakeSem('V_DOT2_F32_F16', 'vector_dot', 'dot2_f32_f16', 'f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'dot2_f32_f16' in call_names

    def test_dot_variants(self):
        for cls in [
            'dot2_f32_f16',
            'dot4_i32_i8',
            'dot4_i32_iu8',
            'dot8_i32_iu4',
            'dot8_u32_u4',
        ]:
            sem = _FakeSem(f'V_{cls.upper()}', cls, cls)
            block = derive_sema_block(sem)
            assert block is not None

    @pytest.mark.parametrize(
        ('name', 'expected_class'),
        [
            ('V_DOT4_I32_IU8', 'dot4_i32_iu8'),
            ('V_DOT8_I32_IU4', 'dot8_i32_iu4'),
        ],
    )
    def test_dot_iu_variants_preserve_mixed_signedness(self, name, expected_class):
        sem = derive_semantics(name, 'ENC_VOP3P')
        assert sem is not None
        assert sem.semantic_class == expected_class


class TestDeriveMfma:
    def test_mfma(self):
        sem = _FakeSem('V_MFMA_F32_16X16X16_F16', 'mfma', data_type='f32')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
        assert 'mfma_compute' in call_names

    @pytest.mark.parametrize(
        'name',
        [
            'V_WMMA_BF16F32_16X16X32_BF16',
            'V_WMMA_F32_16X16X128_F8F6F4',
            'V_WMMA_F32_32X16X128_F4',
            'V_SWMMAC_BF16F32_16X16X64_BF16',
        ],
    )
    def test_gfx1250_low_precision_wmma_derives_mfma(self, name):
        sem = derive_semantics(name, 'ENC_VOP3P')
        assert sem is not None
        assert sem.semantic_class == 'mfma'


class TestDeriveWaitCounters:
    @pytest.mark.parametrize(
        ('name', 'operation'),
        [
            ('S_WAITCNT_VSCNT', 'waitcnt_vscnt'),
            ('S_WAITCNT_VMCNT', 'waitcnt_vmcnt'),
            ('S_WAITCNT_LGKMCNT', 'waitcnt_lgkmcnt'),
            ('S_WAITCNT_EXPCNT', 'waitcnt_expcnt'),
            ('S_WAIT_TENSORCNT', 'wait_tensorcnt'),
            ('S_WAIT_ASYNCCNT', 'wait_asynccnt'),
        ],
    )
    def test_named_counter_is_split_wait(self, name, operation):
        enc_name = 'ENC_SOPK' if name.startswith('S_WAITCNT_') else 'ENC_SOPP'
        sem = derive_semantics(name, enc_name)
        assert sem is not None
        assert sem.semantic_class == 'wait_counter'
        assert sem.operation == operation


class TestDerivePermlane:
    def test_permlane16(self):
        sem = _FakeSem('V_PERMLANE16_B32', 'vector_permlane16')
        block = derive_sema_block(sem)
        assert block is not None
        call_names = [
            n.call_name for n in block.body.walk() if n.kind == SemaNodeKind.CALL
        ]
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

    @pytest.mark.parametrize('name', ['S_CALL_B64', 'S_CALL_I64'])
    def test_scalar_call_derives_for_b64_and_i64_spellings(self, name):
        sem = derive_semantics(name, 'ENC_SOPK')
        assert sem is not None
        assert sem.semantic_class == 'scalar_call'

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

    def test_trap_is_classified_as_control_flow_terminator(self):
        sem = derive_semantics('S_TRAP', 'ENC_SOPP')
        assert sem is not None
        assert sem.semantic_class == 'trap'

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
    @pytest.mark.parametrize(
        ('name', 'semantic_class'),
        [
            ('TENSOR_LOAD_TO_LDS', 'tensor_load_to_lds'),
            ('TENSOR_STORE_FROM_LDS', 'tensor_store_from_lds'),
        ],
    )
    def test_tensor_dma(self, name, semantic_class):
        sem = derive_semantics(name, 'ENC_VIMAGE')
        assert sem is not None
        assert sem.semantic_class == semantic_class

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
