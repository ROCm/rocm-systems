# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for SemaAST to C++ lowering."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.codegen.execute.sema_lower import (
    LoweringContext,
    OperandBinding,
    OperandMap,
    RegClass,
    lower_sema_block,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


def _src(idx: int) -> SemaNode:
    return SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='S'),
            SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
        ),
    )


def _dst(idx: int) -> SemaNode:
    return SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='D'),
            SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
        ),
    )


def _cast(inner: SemaNode, target: SemaType) -> SemaNode:
    return SemaNode(SemaNodeKind.CAST, ty=target, cast_target=target, children=(inner,))


def _cvt_f32_fp8_assignment() -> SemaNode:
    return SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            SemaNode(
                SemaNodeKind.CALL,
                call_name='cvt_f32_fp8',
                ty=SemaType.U32,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='cvt_f32_fp8'),
                    SemaNode(SemaNodeKind.LIT, lit_value='0x40', ty=SemaType.U32),
                ),
            ),
        ),
    )


class TestLowerEmptyBlock:
    def test_empty_block(self):
        block = SemaBlock(
            'NOP', ExecModel.UNKNOWN, SemaNode(SemaNodeKind.SEQ, children=())
        )
        result = lower_sema_block(block)
        assert '(void)wf;' in result


class TestLowerScalarAdd:
    def test_scalar_add_u32(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.U32),
                SemaNode(
                    SemaNodeKind.ADD,
                    ty=SemaType.U32,
                    children=(
                        _cast(_src(0), SemaType.U32),
                        _cast(_src(1), SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('S_ADD_U32', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'write_scalar' in result
        assert 'read_scalar' in result
        assert 'for (' not in result

    def test_scalar_no_exec_loop(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.U32),
                _cast(_src(0), SemaType.U32),
            ),
        )
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'lane' not in result
        assert 'exec()' not in result


class TestLowerVectorAdd:
    def test_vector_add_f32(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.F32),
                SemaNode(
                    SemaNodeKind.ADD,
                    ty=SemaType.F32,
                    children=(
                        _cast(_src(0), SemaType.F32),
                        _cast(_src(1), SemaType.F32),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_ADD_F32', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'for (uint32_t lane = 0' in result
        assert 'wf.exec()' in result
        assert 'read_lane(wf, lane)' in result
        assert 'write_lane(wf, lane' in result

    def test_vector_fma(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.F32),
                SemaNode(
                    SemaNodeKind.FMA,
                    ty=SemaType.F32,
                    children=(
                        _cast(_src(0), SemaType.F32),
                        _cast(_src(1), SemaType.F32),
                        _cast(_src(2), SemaType.F32),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_FMA_F32', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'std::fma(' in result

    def test_vector_explicit_vcc_dst_preserves_high_sgpr_on_wave32(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.U32),
                SemaNode(
                    SemaNodeKind.CALL,
                    call_name='add_co',
                    ty=SemaType.U32,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='add_co'),
                        _cast(_src(0), SemaType.U32),
                        _cast(_src(1), SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_ADD_CO_U32', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 32),
                1: OperandBinding('src1', RegClass.VGPR, 32),
            },
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 32)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            vcc_dst='sdst',
        )

        result = lower_sema_block(block, ctx)

        assert 'if (wf.wf_size() <= 32)' in result
        assert 'sdst.write_scalar(wf, static_cast<uint32_t>(vcc))' in result
        assert 'sdst.write_scalar64(wf, vcc)' in result

    def test_vector_block_can_write_scalar_destination(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.F32),
                _cast(_src(0), SemaType.F32),
            ),
        )
        block = SemaBlock('V_S_RCP_F32', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.SGPR, 32)},
            dst_bindings={0: OperandBinding('vdst', RegClass.SGPR, 32)},
        )
        ctx = LoweringContext(exec_model=ExecModel.VECTOR, operand_map=omap)

        result = lower_sema_block(block, ctx)

        assert 'if (exec != 0)' not in result
        assert 'for (uint32_t lane = 0' in result
        assert 'src0.read_scalar(wf)' in result
        assert 'vdst.write_scalar(wf' in result
        assert 'vdst.write_lane' not in result

    def test_vector_sgpr_once_avoids_repeated_aliased_writes(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.F32),
                _cast(_src(0), SemaType.F32),
            ),
        )
        block = SemaBlock('V_S_RCP_F32', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.SGPR, 32)},
            dst_bindings={0: OperandBinding('vdst', RegClass.SGPR, 32)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            vector_sgpr_once=True,
        )

        result = lower_sema_block(block, ctx)

        assert 'uint64_t exec = wf.exec();' in result
        assert 'if (exec != 0)' in result
        assert 'for (uint32_t lane = 0' not in result
        assert 'src0.read_scalar(wf)' in result
        assert 'vdst.write_scalar(wf' in result

    def test_less_greater_compare_reads_operands_once(self):
        s0 = _cast(_src(0), SemaType.F32)
        s1 = _cast(_src(1), SemaType.F32)
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(
                    SemaNodeKind.ARRAYDEREF,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='VCC'),
                        SemaNode(SemaNodeKind.ID, id_name='laneId'),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.LOR,
                    ty=SemaType.U1,
                    children=(
                        SemaNode(SemaNodeKind.LT, ty=SemaType.U1, children=(s0, s1)),
                        SemaNode(SemaNodeKind.GT, ty=SemaType.U1, children=(s0, s1)),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_CMP_LG_F32', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 32),
                1: OperandBinding('src1', RegClass.VGPR, 32),
            },
        )

        result = lower_sema_block(
            block, LoweringContext(exec_model=ExecModel.VECTOR, operand_map=omap)
        )

        assert 'return (a < b) || (a > b);' in result
        assert result.count('src0.read_lane(wf, lane)') == 1
        assert result.count('src1.read_lane(wf, lane)') == 1

    def test_zero_initialized_vcc_mask_omits_false_lane_clear(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(
                    SemaNodeKind.ARRAYDEREF,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='VCC'),
                        SemaNode(SemaNodeKind.ID, id_name='laneId'),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.EQ,
                    ty=SemaType.U1,
                    children=(
                        _cast(_src(0), SemaType.U32),
                        _cast(_src(1), SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_CMP_EQ_U32', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 32),
                1: OperandBinding('src1', RegClass.VGPR, 32),
            },
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            clear_false_lane_mask_writes=False,
        )

        result = lower_sema_block(block, ctx)

        assert 'uint64_t vcc = 0;' in result
        assert 'vcc |= (1ULL << lane);' in result
        assert 'vcc &= ~(1ULL << lane);' not in result

    def test_true16_destination_select_merges_half(self):
        b16 = SemaType('B', 16)
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), b16),
                _cast(_src(0), b16),
            ),
        )
        block = SemaBlock('V_MOV_B16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.VGPR, 16)},
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.pad_16',
            true16_dst_reg='inst_.vdst & 0x7fu',
        )

        result = lower_sema_block(block, ctx)

        assert (
            'wf.cu().read_vgpr(wf.vgpr_alloc().base + (inst_.vdst & 0x7fu), lane)'
            in result
        )
        assert (
            'wf.cu().write_vgpr(wf.vgpr_alloc().base + (inst_.vdst & 0x7fu), lane, merged)'
            in result
        )
        assert '0x0000ffffu' in result
        assert '0xffff0000u' in result
        assert 'inst_.pad_16' in result

    def test_true16_source_select_uses_raw_source(self):
        u16 = SemaType('U', 16)
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), u16),
                _cast(_src(0), u16),
            ),
        )
        block = SemaBlock('V_MOV_B16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.VGPR, 16)},
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.opsel & 0x2u',
            true16_src_select='inst_.opsel & 0x1u',
            true16_src_raw='src0_raw',
        )

        result = lower_sema_block(block, ctx)

        assert '(src0_raw >> 16)' in result
        assert '(static_cast<uint16_t>(src0.read_lane(wf, lane)) >> 16)' not in result

    def test_true16_source_selects_are_per_operand(self):
        u16 = SemaType('U', 16)
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), u16),
                SemaNode(
                    SemaNodeKind.OR,
                    ty=u16,
                    children=(
                        _cast(_src(0), u16),
                        _cast(_src(1), u16),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_OR_B16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 16),
                1: OperandBinding('src1', RegClass.VGPR, 16),
            },
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.opsel & 0x8u',
            true16_src_selects={
                0: 'inst_.opsel & 0x1u',
                1: 'inst_.opsel & 0x2u',
            },
            true16_vop3_opsel='inst_.opsel',
        )

        result = lower_sema_block(block, ctx)

        assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in result
        assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in result
        assert (
            '::rocjitsu::amdgpu::write_vop3_true16_dst('
            'vdst, wf, lane, inst_.opsel, src_half, true);' in result
        )

    def test_true16_cndmask_keeps_selector_scalar(self):
        b16 = SemaType('B', 16)
        cond = SemaNode(
            SemaNodeKind.ARRAYDEREF,
            ty=SemaType.U1,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='VCC', ty=SemaType.U64),
                SemaNode(SemaNodeKind.ID, id_name='laneId', ty=SemaType.U32),
            ),
        )
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), b16),
                SemaNode(
                    SemaNodeKind.TERNARY,
                    ty=b16,
                    children=(cond, _cast(_src(1), b16), _cast(_src(0), b16)),
                ),
            ),
        )
        block = SemaBlock('V_CNDMASK_B16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={
                0: OperandBinding('src0', RegClass.VGPR, 16),
                1: OperandBinding('src1', RegClass.VGPR, 16),
                2: OperandBinding('src2', RegClass.SGPR, 64),
            },
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.opsel & 0x8u',
            true16_src_selects={
                0: 'inst_.opsel & 0x1u',
                1: 'inst_.opsel & 0x2u',
            },
            true16_vop3_opsel='inst_.opsel',
            vcc_read='src2.read_scalar64(wf)',
        )

        result = lower_sema_block(block, ctx)

        assert 'src2.read_scalar64(wf)' in result
        assert 'src2.read_lane' not in result
        assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in result
        assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in result
        assert (
            'write_vop3_true16_dst(vdst, wf, lane, inst_.opsel, src_half, true);'
            in result
        )

    def test_true16_bf16_destination_converts_float_result(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), SemaType.BF16),
                SemaNode(
                    SemaNodeKind.COS,
                    ty=SemaType.F32,
                    children=(_cast(_src(0), SemaType.BF16),),
                ),
            ),
        )
        block = SemaBlock('V_COS_BF16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.VGPR, 16)},
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.opsel & 0x8u',
            true16_src_selects={0: 'inst_.opsel & 0x1u'},
            true16_vop3_opsel='inst_.opsel',
        )

        result = lower_sema_block(block, ctx)

        assert 'util::bf16_to_f32' in result
        assert (
            'uint32_t src_half = static_cast<uint32_t>(static_cast<uint16_t>(util::f32_to_bf16('
            in result
        )
        assert (
            'write_vop3_true16_dst(vdst, wf, lane, inst_.opsel, src_half, true);'
            in result
        )
        assert 'std::cos' in result

    def test_true16_destination_read_selects_matching_half(self):
        f16 = SemaType('F', 16)
        dst_read = SemaNode(
            SemaNodeKind.INSTOPERAND,
            ty=f16,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='D'),
                SemaNode(SemaNodeKind.LIT, lit_value='0'),
            ),
        )
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                _cast(_dst(0), f16),
                SemaNode(
                    SemaNodeKind.ADD,
                    ty=SemaType.F32,
                    children=(
                        _cast(_src(0), f16),
                        _cast(dst_read, f16),
                    ),
                ),
            ),
        )
        block = SemaBlock('V_FMAC_F16', ExecModel.VECTOR, body)
        omap = OperandMap(
            src_bindings={0: OperandBinding('src0', RegClass.VGPR, 16)},
            dst_bindings={0: OperandBinding('vdst', RegClass.VGPR, 16)},
        )
        ctx = LoweringContext(
            exec_model=ExecModel.VECTOR,
            operand_map=omap,
            true16_dst_select='inst_.opsel & 0x8u',
            true16_dst_reg='inst_.vdst & 0x7fu',
        )

        result = lower_sema_block(block, ctx)

        assert (
            '((inst_.opsel & 0x8u) != 0 ? '
            '(wf.cu().read_vgpr(wf.vgpr_alloc().base + (inst_.vdst & 0x7fu), lane) >> 16)'
            in result
        )


class TestLowerCast:
    def test_instoperand_uses_bit_cast(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                _cast(_src(0), SemaType.F32),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'std::bit_cast<float>' in result

    def test_widening_uses_static_cast(self):
        inner_id = SemaNode(SemaNodeKind.ID, id_name='tmp', ty=SemaType.U32)
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='wide'),
                _cast(inner_id, SemaType.U64),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'static_cast<uint64_t>' in result


class TestLowerControlFlow:
    def test_if_two_branches(self):
        body = SemaNode(
            SemaNodeKind.IF,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='tmp'),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'if (wf.read_scc())' in result

    def test_if_else(self):
        body = SemaNode(
            SemaNodeKind.IF,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='tmp'),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='tmp'),
                        SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '} else {' in result

    def test_for_loop(self):
        body = SemaNode(
            SemaNodeKind.FOR,
            children=(
                SemaNode(
                    SemaNodeKind.ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='i'),
                        SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.LT,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='i'),
                        SemaNode(SemaNodeKind.LIT, lit_value='4', ty=SemaType.U32),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.ADD_ASSIGN,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='i'),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                    ),
                ),
                SemaNode(
                    SemaNodeKind.COMMENT,
                    children=(SemaNode(SemaNodeKind.LIT, lit_value='loop body'),),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'for (' in result

    @pytest.mark.parametrize('control', ['if', 'for', 'while'])
    def test_nested_control_flow_preserves_arch_name(self, control):
        nested_body = _cvt_f32_fp8_assignment()
        if control == 'if':
            body = SemaNode(
                SemaNodeKind.IF,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                    nested_body,
                ),
            )
        elif control == 'for':
            body = SemaNode(
                SemaNodeKind.FOR,
                children=(
                    SemaNode(
                        SemaNodeKind.ASSIGN,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='i'),
                            SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
                        ),
                    ),
                    SemaNode(
                        SemaNodeKind.LT,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='i'),
                            SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                        ),
                    ),
                    SemaNode(
                        SemaNodeKind.ADD_ASSIGN,
                        children=(
                            SemaNode(SemaNodeKind.ID, id_name='i'),
                            SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                        ),
                    ),
                    nested_body,
                ),
            )
        else:
            body = SemaNode(
                SemaNodeKind.WHILE,
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                    nested_body,
                ),
            )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        ctx = LoweringContext(exec_model=ExecModel.SCALAR, arch_name='cdna3')

        result = lower_sema_block(block, ctx)

        assert 'util::fp8_e4m3_fnuz_to_f32' in result
        assert 'util::fp8_e4m3_to_f32' not in result


class TestLowerContextIds:
    def test_scc_write(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U1),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'wf.write_scc(1)' in result

    def test_scc_read(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'wf.read_scc()' in result


class TestLowerCall:
    def test_inline_cpp_call(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='addr'),
                SemaNode(
                    SemaNodeKind.CALL,
                    call_name='CalcDsAddr',
                    ty=SemaType.U32,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='CalcDsAddr'),
                        SemaNode(SemaNodeKind.ID, id_name='base'),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'calc_ds_addr(base)' in result

    def test_opaque_nop_call(self):
        body = SemaNode(
            SemaNodeKind.SEQ,
            children=(
                SemaNode(
                    SemaNodeKind.CALL,
                    call_name='nop',
                    children=(SemaNode(SemaNodeKind.ID, id_name='nop'),),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'no-op' in result


class TestLowerTernary:
    def test_ternary(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='result'),
                SemaNode(
                    SemaNodeKind.TERNARY,
                    ty=SemaType.U32,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                        SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                        SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '?' in result
        assert 'wf.read_scc()' in result


class TestLowerMemory:
    def test_mem_read_scalar(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='data'),
                SemaNode(
                    SemaNodeKind.ARRAYDEREF,
                    ty=SemaType.B32,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='MEM', ty=SemaType.B32),
                        SemaNode(SemaNodeKind.ID, id_name='addr'),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'scalar_mem().read<uint32_t>(addr)' in result

    def test_mem_read_vector(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='data'),
                SemaNode(
                    SemaNodeKind.ARRAYDEREF,
                    ty=SemaType.B32,
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='MEM', ty=SemaType.B32),
                        SemaNode(SemaNodeKind.ID, id_name='addr'),
                    ),
                ),
            ),
        )
        block = SemaBlock('TEST', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'vmem().read<uint32_t>(addr)' in result


class TestLowerLiterals:
    def test_float_literal_suffix(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.F32),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '0.0f' in result

    def test_u64_literal_suffix(self):
        body = SemaNode(
            SemaNodeKind.ASSIGN,
            children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='4294967296', ty=SemaType.U64),
            ),
        )
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '4294967296ULL' in result


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlLowering:
    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml

        return parse_semantics_xml(SEMA_XML_PATH)

    def test_v_add_f32(self, blocks):
        result = lower_sema_block(blocks['V_ADD_F32'])
        assert 'for (uint32_t lane' in result
        assert 'read_lane' in result
        assert 'write_lane' in result
        assert 'std::bit_cast<float>' in result

    def test_v_fma_f32(self, blocks):
        result = lower_sema_block(blocks['V_FMA_F32'])
        assert 'std::fma(' in result

    def test_s_add_co_u32(self, blocks):
        result = lower_sema_block(blocks['S_ADD_CO_U32'])
        assert 'write_scc(' in result
        assert 'write_scalar' in result
        assert 'for (' not in result

    def test_s_mov_b32(self, blocks):
        result = lower_sema_block(blocks['S_MOV_B32'])
        assert 'write_scalar' in result
        assert 'read_scalar' in result

    def test_ds_load_b32_has_addr_calc(self, blocks):
        result = lower_sema_block(blocks['DS_LOAD_B32'])
        assert 'calc_ds_addr(' in result

    def test_s_load_b32_has_addr_calc(self, blocks):
        result = lower_sema_block(blocks['S_LOAD_B32'])
        assert 'calc_scalar_global_addr(' in result

    def test_all_non_stub_lower_without_error(self, blocks):
        errors = []
        for name, block in blocks.items():
            if block.is_empty:
                continue
            try:
                result = lower_sema_block(block)
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                errors.append(f'{name}: {e}')
        assert errors == [], f'{len(errors)} lowering errors:\n' + '\n'.join(
            errors[:10]
        )

    def test_s_endpgm_stub(self, blocks):
        result = lower_sema_block(blocks['S_ENDPGM'])
        assert '(void)wf;' in result
