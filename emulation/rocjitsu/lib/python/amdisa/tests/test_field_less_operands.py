# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for capturing field-less operands in the amdisa model.

Fixtures below mirror the MR ISA operand layouts (confirmed against the specs,
not parsed from them):
  - V_ADD_CO_U32 (VOP2): vdst, field-less VCC carry-out, src0, vsrc1.
  - V_ADD_CO_CI_U32 (VOP2): vdst, field-less VCC carry-out, src0, vsrc1,
    field-less VCC carry-in.
  - V_CMPX_EQ_F32 (VOPC, RDNA): field-less OPR_EXEC dest + field-less implicit
    OPR_SDST_EXEC side effect.
  - S_AND_SAVEEXEC_B64 (SOP1): sdst, implicit field-less SDST_EXEC (out), SCC
    (out), SDST_EXEC (in).
"""

import re
from types import SimpleNamespace

from amdisa.codegen import CodeGenerator
from amdisa.cross_isa import _operand_signature
from amdisa.gpuisa import (
    Instruction,
    Operand,
    OperandSelector,
    synthesize_fieldless_name,
)
from amdisa.parser import _uniquify_fieldless_names

_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _v_add_co_ci_u32():
    """VOP2 form with field-less VCC carry-out and carry-in (both explicit)."""
    ops = [
        Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
        Operand('vcc', 64, 'OPR_VCC', False, True, False, False, 2, True),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 3),
        Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 4),
        Operand('vcc', 64, 'OPR_VCC', True, False, False, False, 5, True),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('V_ADD_CO_CI_U32', 'ENC_VOP2', 0, ops)


def _v_cmpx_eq_f32():
    """VOPC form (RDNA): field-less OPR_EXEC dest + implicit OPR_SDST_EXEC."""
    ops = [
        Operand('exec', 64, 'OPR_EXEC', False, True, False, False, 1, True),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
        Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 3),
        Operand('sdst_exec', 64, 'OPR_SDST_EXEC', False, True, True, False, 4, True),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('V_CMPX_EQ_F32', 'ENC_VOPC', 0, ops)


def _s_and_saveexec_b64():
    ops = [
        Operand('sdst', 32, 'OPR_SREG', False, True, False, False, 1),
        Operand('ssrc0', 32, 'OPR_SSRC', True, False, False, False, 2),
        Operand('sdst_exec', 64, 'OPR_SDST_EXEC', False, True, True, False, 3, True),
        Operand('scc', 32, 'OPR_SSRC_SPECIAL_SCC', False, True, True, False, 4, True),
        Operand('sdst_exec', 64, 'OPR_SDST_EXEC', True, False, True, False, 5, True),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('S_AND_SAVEEXEC_B64', 'ENC_SOP1', 0, ops)


# ---------------------------------------------------------------------------
# (a) operands now includes field-less operands, nothing dropped.
# ---------------------------------------------------------------------------
def test_operands_include_field_less():
    inst = _v_add_co_ci_u32()
    assert len(inst.operands) == 5
    field_less = [op for op in inst.operands if op.field_less]
    assert [op.operand_type for op in field_less] == ['OPR_VCC', 'OPR_VCC']
    # Both a field-bearing and a field-less operand are present.
    assert any(not op.field_less for op in inst.operands)


# ---------------------------------------------------------------------------
# (b) implicit_operands is exactly the is_implicit subset (NOT field-less-ness).
# ---------------------------------------------------------------------------
def test_implicit_operands_is_is_implicit_subset():
    # VCC carry is field-less but explicit -> not implicit.
    add = _v_add_co_ci_u32()
    assert add.implicit_operands == []
    assert len(add.explicit_operands) == 5

    # SAVEEXEC: SDST_EXEC and SCC side effects are implicit; sdst/ssrc0 are not.
    save = _s_and_saveexec_b64()
    assert [op.operand_type for op in save.implicit_operands] == [
        'OPR_SDST_EXEC',
        'OPR_SSRC_SPECIAL_SCC',
        'OPR_SDST_EXEC',
    ]
    assert all(op.is_implicit for op in save.implicit_operands)
    assert all(not op.is_implicit for op in save.explicit_operands)
    # implicit_operands is a subset view of operands.
    assert set(id(o) for o in save.implicit_operands) <= set(
        id(o) for o in save.operands
    )


def test_src_dst_operand_subsets():
    save = _s_and_saveexec_b64()
    assert [op.name for op in save.dst_operands] == ['sdst', 'sdst_exec', 'scc']
    assert [op.name for op in save.src_operands] == ['ssrc0', 'sdst_exec_in']


# ---------------------------------------------------------------------------
# (c) Name synthesis is deterministic, valid, and unique -- including the two
#     two-field-less-operands-of-the-same-type cases.
# ---------------------------------------------------------------------------
def test_synthesize_fieldless_name_is_deterministic_and_valid():
    cases = {
        'OPR_VCC': 'vcc',
        'OPR_EXEC': 'exec',
        'OPR_SDST_EXEC': 'sdst_exec',  # deliberately distinct from OPR_EXEC
        'OPR_SSRC_SPECIAL_SCC': 'scc',
        'OPR_PC': 'pc',
        'OPR_SDST_M0': 'm0',
        'OPR_DSMEM': 'dsmem',
        'OPR_UNKNOWN_TYPE': 'unknown_type',  # fallback: strip OPR_, lowercase
    }
    for opr_type, expected in cases.items():
        got = synthesize_fieldless_name(opr_type)
        assert got == expected, (opr_type, got, expected)
        assert _IDENTIFIER.match(got), got
        # deterministic
        assert synthesize_fieldless_name(opr_type) == got
    # exec and sdst_exec must not collapse to the same base.
    assert synthesize_fieldless_name('OPR_EXEC') != synthesize_fieldless_name(
        'OPR_SDST_EXEC'
    )


def test_uniquify_vcc_in_out_case():
    inst = _v_add_co_ci_u32()
    names = [op.name for op in inst.operands]
    assert len(names) == len(set(names)), names  # all unique
    vcc = [op for op in inst.operands if op.operand_type == 'OPR_VCC']
    out_name = next(op.name for op in vcc if op.is_output)
    in_name = next(op.name for op in vcc if op.is_input)
    assert out_name == 'vcc'
    assert in_name == 'vcc_in'


def test_uniquify_cmpx_exec_x2_case():
    inst = _v_cmpx_eq_f32()
    names = [op.name for op in inst.operands]
    assert len(names) == len(set(names)), names
    # OPR_EXEC and OPR_SDST_EXEC are distinct bases -> no suffixing needed.
    assert 'exec' in names
    assert 'sdst_exec' in names
    for name in names:
        assert _IDENTIFIER.match(name), name


def test_uniquify_is_stable_across_repeat_runs():
    a = [op.name for op in _s_and_saveexec_b64().operands]
    b = [op.name for op in _s_and_saveexec_b64().operands]
    assert a == b == ['sdst', 'ssrc0', 'sdst_exec', 'scc', 'sdst_exec_in']


# ---------------------------------------------------------------------------
# (d) has_implicit_operand query (drives PC_OPERAND-style flags).
# ---------------------------------------------------------------------------
def test_has_implicit_operand():
    pc_inst = Instruction(
        'S_GETPC_B64',
        'ENC_SOP1',
        0,
        [
            Operand('sdst', 64, 'OPR_SDST', False, True, False, False, 1),
            Operand('pc', 64, 'OPR_PC', True, False, True, False, 2, True),
        ],
    )
    assert pc_inst.has_implicit_operand('OPR_PC')
    assert not pc_inst.has_implicit_operand('OPR_VCC')
    # A field-less-but-explicit VCC operand is NOT reported as implicit.
    assert not _v_add_co_ci_u32().has_implicit_operand('OPR_VCC')


# ---------------------------------------------------------------------------
# Cross-ISA signature must ignore field-less operands (keeps execute_shared.h
# partitioning stable when field-less operands are added to the model).
# ---------------------------------------------------------------------------
def test_operand_signature_excludes_field_less():
    with_vcc = _v_add_co_ci_u32()
    without_field_less = Instruction(
        'V_ADD_CO_CI_U32',
        'ENC_VOP2',
        0,
        [op for op in with_vcc.operands if not op.field_less],
    )
    assert _operand_signature(with_vcc) == _operand_signature(without_field_less)


# ---------------------------------------------------------------------------
# Canonical fixed encoding value is computed from the ISA's selectors (min of
# the selector values), matching what name()/to_register_ref() gate on.
# ---------------------------------------------------------------------------
def test_fieldless_canonical_value_from_selectors():
    selectors = [
        OperandSelector('OPR_PC', [('OPR_PC_PC_ALL', '0')]),
        OperandSelector('OPR_VCC', [('OPR_VCC_VCC', '0')]),
        OperandSelector(
            'OPR_SSRC_SPECIAL_SCC', [('OPR_SSRC_SPECIAL_SCC_SRC_SCC', '253')]
        ),
        OperandSelector(
            'OPR_SDST_EXEC',
            [('OPR_SDST_EXEC_EXEC_LO', '126'), ('OPR_SDST_EXEC_EXEC_HI', '127')],
        ),
    ]
    fake = SimpleNamespace(isa_spec=SimpleNamespace(opnd_selectors=selectors))
    canon = CodeGenerator._fieldless_canonical_value
    assert canon(fake, 'OPR_PC') == 0
    assert canon(fake, 'OPR_VCC') == 0
    assert canon(fake, 'OPR_SSRC_SPECIAL_SCC') == 253
    # EXEC LO/HI -> the LO (minimum) value, which name() renders as exec_lo.
    assert canon(fake, 'OPR_SDST_EXEC') == 126
    # No selector -> 0 fallback.
    assert canon(fake, 'OPR_GPUMEM') == 0


# ---------------------------------------------------------------------------
# Execute codegen keeps side-effect field-less operands out of positional
# src/dst lists, but keeps field-less OPR_SIMM32 because it is value-bearing.
# ---------------------------------------------------------------------------
def test_execute_operand_participation_keeps_only_fieldless_simm32():
    vcc = Operand('vcc', 64, 'OPR_VCC', True, False, False, False, 0, True)
    simm32 = Operand('simm32', 32, 'OPR_SIMM32', True, False, False, False, 1, True)
    src0 = Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2)

    assert not CodeGenerator._execute_operand_participates(vcc)
    assert CodeGenerator._execute_operand_participates(simm32)
    assert CodeGenerator._execute_operand_participates(src0)


def test_value_bearing_fieldless_types_is_locked():
    # Golden lock on the single source of truth for "which field-less operands
    # stay live". Changing this set deliberately flips runtime inertness for a
    # type across is_vgpr / simd_capable / read_* (via
    # _fieldless_inert_cpp_cond) AND its execute-body visibility (via
    # _execute_operand_participates), so update this assertion consciously
    # rather than by accident.
    assert CodeGenerator._VALUE_BEARING_FIELDLESS_TYPES == frozenset({'OPR_SIMM32'})


def test_fieldless_inert_cpp_cond_is_derived_from_value_bearing_set():
    # The C++ guard emitted into every generated read_*/simd_capable body must
    # be derived from _VALUE_BEARING_FIELDLESS_TYPES so it can never drift from
    # the participation rule (the whole point of single-sourcing the set).
    cond = CodeGenerator._fieldless_inert_cpp_cond()
    assert cond.startswith('field_less_ && ')
    for t in CodeGenerator._VALUE_BEARING_FIELDLESS_TYPES:
        assert f'opr_type_ != OperandType::{t}' in cond
    # Golden for the current single-type set.
    assert cond == 'field_less_ && opr_type_ != OperandType::OPR_SIMM32'
