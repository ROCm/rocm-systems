# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Side-by-side validation: SemaAST lowering vs old generators.

Validates that the SemaAST pipeline (derive → enrich → lower) produces
C++ that has the same structural properties as the old string-tag
generators. Does NOT require byte-identical output — instead checks
that both outputs:
  - Use the same operand names (ssrc0, vdst, etc.)
  - Use the same read/write methods (read_scalar, write_lane, etc.)
  - Write the same side-effect registers (SCC, VCC, EXEC)
  - Have the same execution model (EXEC loop vs no loop)
"""

import re

import pytest

from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.codegen.execute.sema_lower import (
    ExecModel,
    LoweringContext,
    OperandMap,
    lower_sema_block,
)
from amdisa.codegen.execute.scalar import (
    gen_scalar_binop,
    gen_scalar_cmp,
    gen_scalar_saveexec,
    gen_scalar_unary,
)
from amdisa.codegen.execute.vector_alu import (
    gen_vector_binop,
    gen_vector_ternary,
    gen_vector_unary,
)


class _FakeSem:
    def __init__(self, name, cls, op=None, dtype=None, scc=None):
        self.name = name
        self.semantic_class = cls
        self.operation = op
        self.data_type = dtype
        self.sets_scc = scc
        self.branch_condition = None
        self.elem_size = None
        self.num_elems = None
        self.sign_extend = False


def _extract_properties(cpp: str) -> dict:
    """Extract structural properties from generated C++."""
    return {
        'has_exec_loop': 'for (uint32_t lane' in cpp,
        'writes_scc': 'write_scc' in cpp or 'wf.write_scc' in cpp,
        'writes_vcc': 'write_vcc' in cpp or 'vcc' in cpp.lower(),
        'writes_exec': 'write_exec' in cpp or 'set_exec' in cpp,
        'reads_scalar': 'read_scalar' in cpp,
        'reads_lane': 'read_lane' in cpp,
        'writes_scalar': 'write_scalar' in cpp,
        'writes_lane': 'write_lane' in cpp,
        'uses_fabs': 'fabs' in cpp,
        'uses_fma': 'fma' in cpp or 'std::fma' in cpp,
        'operand_names': set(re.findall(r'\b(ssrc\d|sdst|src\d|vsrc\d|vdst)\b', cpp)),
    }


def _new_output(sem, src_ops, dst_ops, dtype=None, enc_fields=None):
    """Generate C++ via the SemaAST pipeline."""
    block = derive_sema_block(sem)
    if block is None:
        return ''
    exec_model = block.pragma
    if enc_fields:
        block = enrich_block(block, enc_field_names=frozenset(enc_fields))
    omap = OperandMap.from_operand_names(
        src_ops, dst_ops, exec_model, dtype or sem.data_type
    )
    ctx = LoweringContext(exec_model=exec_model, operand_map=omap)
    return lower_sema_block(block, ctx)


class TestScalarBinopValidation:
    @pytest.mark.parametrize(
        'op,dtype,scc',
        [
            ('add', 'u32', 'carry'),
            ('sub', 'u32', 'borrow'),
            ('and', 'b32', 'nonzero'),
            ('or', 'b64', 'nonzero'),
            ('xor', 'b32', 'nonzero'),
            ('shl', 'b32', 'nonzero'),
        ],
    )
    def test_scalar_binop_properties_match(self, op, dtype, scc):
        sem = _FakeSem(
            f'S_{op.upper()}_{dtype.upper()}', 'scalar_binop', op, dtype, scc
        )
        old = gen_scalar_binop(['sdst'], ['ssrc0', 'ssrc1'], op, dtype, scc)
        new = _new_output(sem, ['ssrc0', 'ssrc1'], ['sdst'], dtype)
        old_props = _extract_properties(old)
        new_props = _extract_properties(new)
        assert (
            old_props['writes_scc'] == new_props['writes_scc']
        ), f"SCC mismatch for {op}/{dtype}: old={old_props['writes_scc']}, new={new_props['writes_scc']}"
        assert not old_props['has_exec_loop'], "scalar should not have EXEC loop"
        assert not new_props['has_exec_loop'], "scalar should not have EXEC loop"
        assert 'ssrc0' in new_props['operand_names']
        assert 'sdst' in new_props['operand_names']


class TestScalarCmpValidation:
    @pytest.mark.parametrize('op', ['eq', 'ne', 'lt', 'gt', 'le', 'ge'])
    def test_scalar_cmp_writes_scc(self, op):
        sem = _FakeSem(f'S_CMP_{op.upper()}_U32', 'scalar_cmp', op, 'u32')
        old = gen_scalar_cmp(['ssrc0', 'ssrc1'], op, 'u32')
        new = _new_output(sem, ['ssrc0', 'ssrc1'], [], 'u32')
        assert 'write_scc' in old
        assert 'write_scc' in new


class TestScalarUnaryValidation:
    @pytest.mark.parametrize('op', ['not', 'floor', 'trunc'])
    def test_scalar_unary_no_exec_loop(self, op):
        sem = _FakeSem(f'S_{op.upper()}_B32', 'scalar_unary', op, 'b32', 'nonzero')
        old = gen_scalar_unary(['sdst'], ['ssrc0'], op, 'b32', 'nonzero')
        new = _new_output(sem, ['ssrc0'], ['sdst'], 'b32')
        assert 'for (uint32_t lane' not in old
        assert 'for (uint32_t lane' not in new
        assert 'write_scc' in new


class TestVectorBinopValidation:
    @pytest.mark.parametrize(
        'op,dtype',
        [
            ('add', 'f32'),
            ('sub', 'f32'),
            ('mul', 'f32'),
            ('and', 'b32'),
        ],
    )
    def test_vector_binop_has_exec_loop(self, op, dtype):
        sem = _FakeSem(f'V_{op.upper()}_{dtype.upper()}', 'vector_binop', op, dtype)
        old = gen_vector_binop(['vdst'], ['src0', 'vsrc1'], op, dtype)
        new = _new_output(sem, ['src0', 'vsrc1'], ['vdst'], dtype)
        old_props = _extract_properties(old)
        new_props = _extract_properties(new)
        assert old_props['has_exec_loop']
        assert new_props['has_exec_loop']
        assert 'vdst' in new_props['operand_names']
        assert new_props['reads_lane']
        assert new_props['writes_lane']


class TestVectorTernaryValidation:
    def test_fma_f32(self):
        sem = _FakeSem('V_FMA_F32', 'vector_ternary', 'fma', 'f32')
        old = gen_vector_ternary(['vdst'], ['src0', 'src1', 'src2'], 'fma', 'f32')
        new = _new_output(sem, ['src0', 'src1', 'src2'], ['vdst'], 'f32')
        old_props = _extract_properties(old)
        new_props = _extract_properties(new)
        assert old_props['uses_fma']
        assert new_props['uses_fma']
        assert new_props['has_exec_loop']


class TestVectorUnaryValidation:
    @pytest.mark.parametrize('op', ['floor', 'trunc', 'sqrt'])
    def test_vector_unary_exec_loop(self, op):
        sem = _FakeSem(f'V_{op.upper()}_F32', 'vector_unary', op, 'f32')
        old = gen_vector_unary(['vdst'], ['src0'], op, 'f32')
        new = _new_output(sem, ['src0'], ['vdst'], 'f32')
        assert 'for (uint32_t lane' in old
        assert 'for (uint32_t lane' in new


class TestVop3ModifierValidation:
    def test_enriched_has_modifier_code(self):
        sem = _FakeSem('V_ADD_F32', 'vector_binop', 'add', 'f32')
        new = _new_output(
            sem,
            ['src0', 'vsrc1'],
            ['vdst'],
            'f32',
            enc_fields={'neg', 'abs', 'clamp', 'omod'},
        )
        assert 'inst_.neg' in new
        assert 'inst_.abs' in new
        assert 'inst_.omod' in new
        assert 'inst_.clamp' in new

    def test_unenriched_has_no_modifier_code(self):
        sem = _FakeSem('V_ADD_F32', 'vector_binop', 'add', 'f32')
        new = _new_output(sem, ['src0', 'vsrc1'], ['vdst'], 'f32')
        assert 'inst_.neg' not in new
        assert 'inst_.abs' not in new


class TestSaveexecValidation:
    def test_saveexec_writes_exec(self):
        sem = _FakeSem('S_AND_SAVEEXEC_B64', 'scalar_saveexec', 'and')
        old = gen_scalar_saveexec(['sdst'], ['ssrc0'], 'and')
        new = _new_output(sem, ['ssrc0'], ['sdst'])
        assert 'set_exec' in old or 'set_exec' in old
        assert 'set_exec' in new
        assert 'write_scc' in new


class TestAllClassesLowerWithOperandMap:
    """Verify every registered semantic class lowers with OperandMap."""

    def test_all_classes_produce_output(self):
        from amdisa.sema_derive import _DERIVE_REGISTRY

        errors = []
        for cls_name in sorted(_DERIVE_REGISTRY.keys()):
            sem = _FakeSem(f'TEST_{cls_name.upper()}', cls_name, 'add', 'f32')
            sem.elem_size = 4
            sem.num_elems = 1
            sem.sign_extend = False
            sem.branch_condition = 'scc1'
            block = derive_sema_block(sem)
            if block is None:
                continue
            omap = OperandMap.from_operand_names(
                ['src0', 'src1', 'src2'], ['dst0'], block.pragma, 'f32'
            )
            ctx = LoweringContext(exec_model=block.pragma, operand_map=omap)
            try:
                cpp = lower_sema_block(block, ctx)
                assert len(cpp) > 0
                if block.pragma == ExecModel.VECTOR:
                    assert 'for (uint32_t lane' in cpp, f'{cls_name}: missing EXEC loop'
            except Exception as e:
                errors.append(f'{cls_name}: {e}')
        assert errors == [], '\n'.join(errors[:10])
