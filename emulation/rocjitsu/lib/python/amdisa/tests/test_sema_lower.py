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
    RegClass,
    lower_sema_block,
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


class TestLowerEmptyBlock:
    def test_empty_block(self):
        block = SemaBlock('NOP', ExecModel.UNKNOWN,
                          SemaNode(SemaNodeKind.SEQ, children=()))
        result = lower_sema_block(block)
        assert '(void)wf;' in result


class TestLowerScalarAdd:
    def test_scalar_add_u32(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.U32),
            SemaNode(SemaNodeKind.ADD, ty=SemaType.U32, children=(
                _cast(_src(0), SemaType.U32),
                _cast(_src(1), SemaType.U32),
            )),
        ))
        block = SemaBlock('S_ADD_U32', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'write_scalar' in result
        assert 'read_scalar' in result
        assert 'for (' not in result

    def test_scalar_no_exec_loop(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.U32),
            _cast(_src(0), SemaType.U32),
        ))
        block = SemaBlock('S_MOV', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'lane' not in result
        assert 'exec()' not in result


class TestLowerVectorAdd:
    def test_vector_add_f32(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.F32),
            SemaNode(SemaNodeKind.ADD, ty=SemaType.F32, children=(
                _cast(_src(0), SemaType.F32),
                _cast(_src(1), SemaType.F32),
            )),
        ))
        block = SemaBlock('V_ADD_F32', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'for (uint32_t lane = 0' in result
        assert 'wf.exec()' in result
        assert 'read_lane(wf, lane)' in result
        assert 'write_lane(wf, lane' in result

    def test_vector_fma(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            _cast(_dst(0), SemaType.F32),
            SemaNode(SemaNodeKind.FMA, ty=SemaType.F32, children=(
                _cast(_src(0), SemaType.F32),
                _cast(_src(1), SemaType.F32),
                _cast(_src(2), SemaType.F32),
            )),
        ))
        block = SemaBlock('V_FMA_F32', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'std::fma(' in result


class TestLowerCast:
    def test_instoperand_uses_bit_cast(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            _cast(_src(0), SemaType.F32),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'std::bit_cast<float>' in result

    def test_widening_uses_static_cast(self):
        inner_id = SemaNode(SemaNodeKind.ID, id_name='tmp', ty=SemaType.U32)
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='wide'),
            _cast(inner_id, SemaType.U64),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'static_cast<uint64_t>' in result


class TestLowerControlFlow:
    def test_if_two_branches(self):
        body = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'if (wf.read_scc())' in result

    def test_if_else(self):
        body = SemaNode(SemaNodeKind.IF, children=(
            SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='tmp'),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '} else {' in result

    def test_for_loop(self):
        body = SemaNode(SemaNodeKind.FOR, children=(
            SemaNode(SemaNodeKind.ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='i'),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.LT, children=(
                SemaNode(SemaNodeKind.ID, id_name='i'),
                SemaNode(SemaNodeKind.LIT, lit_value='4', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.ADD_ASSIGN, children=(
                SemaNode(SemaNodeKind.ID, id_name='i'),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
            )),
            SemaNode(SemaNodeKind.COMMENT, children=(
                SemaNode(SemaNodeKind.LIT, lit_value='loop body'),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'for (' in result


class TestLowerContextIds:
    def test_scc_write(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
            SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U1),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'wf.write_scc(1)' in result

    def test_scc_read(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'wf.read_scc()' in result


class TestLowerCall:
    def test_inline_cpp_call(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='addr'),
            SemaNode(SemaNodeKind.CALL, call_name='CalcDsAddr',
                     ty=SemaType.U32, children=(
                SemaNode(SemaNodeKind.ID, id_name='CalcDsAddr'),
                SemaNode(SemaNodeKind.ID, id_name='base'),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'calc_ds_addr(base)' in result

    def test_opaque_nop_call(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.CALL, call_name='nop', children=(
                SemaNode(SemaNodeKind.ID, id_name='nop'),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'no-op' in result


class TestLowerTernary:
    def test_ternary(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='result'),
            SemaNode(SemaNodeKind.TERNARY, ty=SemaType.U32, children=(
                SemaNode(SemaNodeKind.ID, id_name='SCC', ty=SemaType.U1),
                SemaNode(SemaNodeKind.LIT, lit_value='1', ty=SemaType.U32),
                SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.U32),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '?' in result
        assert 'wf.read_scc()' in result


class TestLowerMemory:
    def test_mem_read_scalar(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='data'),
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.B32, children=(
                SemaNode(SemaNodeKind.ID, id_name='MEM', ty=SemaType.B32),
                SemaNode(SemaNodeKind.ID, id_name='addr'),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert 'scalar_mem().read<uint32_t>(addr)' in result

    def test_mem_read_vector(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='data'),
            SemaNode(SemaNodeKind.ARRAYDEREF, ty=SemaType.B32, children=(
                SemaNode(SemaNodeKind.ID, id_name='MEM', ty=SemaType.B32),
                SemaNode(SemaNodeKind.ID, id_name='addr'),
            )),
        ))
        block = SemaBlock('TEST', ExecModel.VECTOR, body)
        result = lower_sema_block(block)
        assert 'vmem().read<uint32_t>(addr)' in result


class TestLowerLiterals:
    def test_float_literal_suffix(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            SemaNode(SemaNodeKind.LIT, lit_value='0', ty=SemaType.F32),
        ))
        block = SemaBlock('TEST', ExecModel.SCALAR, body)
        result = lower_sema_block(block)
        assert '0.0f' in result

    def test_u64_literal_suffix(self):
        body = SemaNode(SemaNodeKind.ASSIGN, children=(
            SemaNode(SemaNodeKind.ID, id_name='tmp'),
            SemaNode(SemaNodeKind.LIT, lit_value='4294967296', ty=SemaType.U64),
        ))
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
        assert errors == [], f'{len(errors)} lowering errors:\n' + '\n'.join(errors[:10])

    def test_s_endpgm_stub(self, blocks):
        result = lower_sema_block(blocks['S_ENDPGM'])
        assert '(void)wf;' in result
