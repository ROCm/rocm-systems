# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for capturing fieldless operands in the amdisa model.

Fixtures below mirror the MR ISA operand layouts (confirmed against the specs,
not parsed from them):
  - V_ADD_CO_U32 (VOP2): vdst, fieldless VCC carry-out, src0, vsrc1.
  - V_ADD_CO_CI_U32 (VOP2): vdst, fieldless VCC carry-out, src0, vsrc1,
    fieldless VCC carry-in.
  - V_CMPX_EQ_F32 (VOPC, RDNA): fieldless OPR_EXEC dest + fieldless implicit
    OPR_SDST_EXEC side effect.
  - S_AND_SAVEEXEC_B64 (SOP1): sdst, implicit fieldless SDST_EXEC (out), SCC
    (out), SDST_EXEC (in).
"""

import re
from types import SimpleNamespace

import pytest

import amdisa.fieldless_policy as fieldless_policy_mod
from amdisa.codegen import CodeGenerator
from amdisa.cross_isa import _operand_signature
from amdisa.fieldless_policy import (
    FieldlessCaps,
    FieldlessCategory,
    FieldlessDisplay,
    FieldlessEffect,
    FieldlessPolicy,
    SpecialRegClass,
    fieldless_policy,
    validate_fieldless_taxonomy,
)
from amdisa.gpuisa import (
    Instruction,
    Operand,
    OperandSelector,
    synthesize_fieldless_name,
)
from amdisa.parser import _uniquify_fieldless_names

_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _v_add_co_ci_u32():
    """VOP2 form with fieldless VCC carry-out and carry-in (both explicit)."""
    ops = [
        Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
        Operand('vcc', 64, 'OPR_VCC', False, True, False, False, 2, fieldless=True),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 3),
        Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 4),
        Operand('vcc', 64, 'OPR_VCC', True, False, False, False, 5, fieldless=True),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('V_ADD_CO_CI_U32', 'ENC_VOP2', 0, ops)


def _v_cmpx_eq_f32():
    """VOPC form (RDNA): fieldless OPR_EXEC dest + implicit OPR_SDST_EXEC."""
    ops = [
        Operand('exec', 64, 'OPR_EXEC', False, True, False, False, 1, fieldless=True),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
        Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 3),
        Operand(
            'sdst_exec',
            64,
            'OPR_SDST_EXEC',
            False,
            True,
            True,
            False,
            4,
            fieldless=True,
        ),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('V_CMPX_EQ_F32', 'ENC_VOPC', 0, ops)


def _s_and_saveexec_b64():
    ops = [
        Operand('sdst', 32, 'OPR_SREG', False, True, False, False, 1),
        Operand('ssrc0', 32, 'OPR_SSRC', True, False, False, False, 2),
        Operand(
            'sdst_exec',
            64,
            'OPR_SDST_EXEC',
            False,
            True,
            True,
            False,
            3,
            fieldless=True,
        ),
        Operand(
            'scc',
            32,
            'OPR_SSRC_SPECIAL_SCC',
            False,
            True,
            True,
            False,
            4,
            fieldless=True,
        ),
        Operand(
            'sdst_exec',
            64,
            'OPR_SDST_EXEC',
            True,
            False,
            True,
            False,
            5,
            fieldless=True,
        ),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('S_AND_SAVEEXEC_B64', 'ENC_SOP1', 0, ops)


# ---------------------------------------------------------------------------
# (a) operands now includes fieldless operands, nothing dropped.
# ---------------------------------------------------------------------------
def test_operands_include_fieldless():
    inst = _v_add_co_ci_u32()
    assert len(inst.operands) == 5
    fieldless = [op for op in inst.operands if op.fieldless]
    assert [op.operand_type for op in fieldless] == ['OPR_VCC', 'OPR_VCC']
    # Both a field-bearing and a fieldless operand are present.
    assert any(not op.fieldless for op in inst.operands)


# ---------------------------------------------------------------------------
# (b) implicit_operands is exactly the is_implicit subset (NOT fieldless-ness).
# ---------------------------------------------------------------------------
def test_implicit_operands_is_is_implicit_subset():
    # VCC carry is fieldless but explicit -> not implicit.
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
# (c) Name synthesis is deterministic, valid, and unique, including the two
#     two-fieldless-operands-of-the-same-type cases.
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
        # Fieldless OPR_VGPR is the gfx12 image address placeholder; its curated
        # name must be 'vaddr', not the 'vgpr' fallback.
        'OPR_VGPR': 'vaddr',
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
            Operand('pc', 64, 'OPR_PC', True, False, True, False, 2, fieldless=True),
        ],
    )
    assert pc_inst.has_implicit_operand('OPR_PC')
    assert not pc_inst.has_implicit_operand('OPR_VCC')
    # A fieldless-but-explicit VCC operand is NOT reported as implicit.
    assert not _v_add_co_ci_u32().has_implicit_operand('OPR_VCC')


# ---------------------------------------------------------------------------
# Cross-ISA signature uses the same participation rule as execute generation:
# inert fieldless operands are excluded (they don't change the body, so they
# must not fragment a shared plan), but value-bearing fieldless OPR_SIMM32 is
# included (it changes the body, so it must change the signature too).
# ---------------------------------------------------------------------------
def test_operand_signature_excludes_inert_fieldless():
    # VCC carry-in/out are inert fieldless -> dropping them leaves the signature
    # unchanged, so an ISA that models them still shares with one that doesn't.
    with_vcc = _v_add_co_ci_u32()
    without_fieldless = Instruction(
        'V_ADD_CO_CI_U32',
        'ENC_VOP2',
        0,
        [op for op in with_vcc.operands if not op.fieldless],
    )
    assert _operand_signature(with_vcc) == _operand_signature(without_fieldless)


def test_operand_signature_includes_value_bearing_simm32():
    # A fieldless OPR_SIMM32 reads a value and appears in the execute body, so it
    # MUST be in the signature -- otherwise two instructions differing only in
    # the literal would look shareable while their bodies differ.
    with_simm32 = Instruction(
        'V_FMAMK_F32',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
            Operand(
                'simm32',
                32,
                'OPR_SIMM32',
                True,
                False,
                False,
                False,
                3,
                fieldless=True,
            ),
            Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 4),
        ],
    )
    without_simm32 = Instruction(
        'V_FMAMK_F32',
        'ENC_VOP2',
        0,
        [op for op in with_simm32.operands if op.operand_type != 'OPR_SIMM32'],
    )
    # The value-bearing fieldless literal is part of the signature.
    assert _operand_signature(with_simm32) != _operand_signature(without_simm32)
    assert any(entry[1] == 'OPR_SIMM32' for entry in _operand_signature(with_simm32))


def _fmaak_f16(simm32_size):
    # V_FMAAK_F16 shape (D = fma(s0, s1, K)); simm32 size varies across ISAs
    # (16 on rdna1/rdna2, 32 on rdna3/rdna4) for the same instruction.
    return Instruction(
        'V_FMAAK_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 1),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 3),
            Operand(
                'simm32',
                simm32_size,
                'OPR_SIMM32',
                True,
                False,
                False,
                False,
                4,
                fieldless=True,
            ),
        ],
    )


def test_operand_signature_canonicalizes_simm32_size():
    # The literal is always a 32-bit encoding word; its declared size (the
    # datatype width) is recorded inconsistently for the same instruction across
    # ISAs. Canonicalizing it keeps identical instructions from fragmenting --
    # this is what restores Identity DBT legalization for V_FMAAK_F16 across
    # rdna generations. The size-16 and size-32 forms must share a signature.
    assert _operand_signature(_fmaak_f16(16)) == _operand_signature(_fmaak_f16(32))


def test_operand_signature_still_distinguishes_f16_from_f32_fma():
    # Canonicalizing the literal must NOT merge f16 and f32 FMA-K: they stay
    # distinct via their register operand sizes (16-bit vs 32-bit vdst/src0/
    # vsrc1), so no false Identity/Substitute across the datatype boundary.
    fmaak_f32 = Instruction(
        'V_FMAAK_F32',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
            Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 3),
            Operand(
                'simm32', 32, 'OPR_SIMM32', True, False, False, False, 4, fieldless=True
            ),
        ],
    )
    assert _operand_signature(_fmaak_f16(32)) != _operand_signature(fmaak_f32)


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
    # Mirror the real CodeGenerator instance state: the canonical-value cache is
    # declared in __init__ and populated lazily on first call.
    fake = SimpleNamespace(
        isa_spec=SimpleNamespace(opnd_selectors=selectors),
        _fieldless_canon_cache=None,
    )
    canon = CodeGenerator._fieldless_canonical_value
    assert canon(fake, 'OPR_PC') == 0
    assert canon(fake, 'OPR_VCC') == 0
    assert canon(fake, 'OPR_SSRC_SPECIAL_SCC') == 253
    # EXEC LO/HI -> the LO (minimum) value, which name() renders as exec_lo.
    assert canon(fake, 'OPR_SDST_EXEC') == 126
    # No selector -> 0 fallback.
    assert canon(fake, 'OPR_GPUMEM') == 0


# ---------------------------------------------------------------------------
# Execute codegen keeps side-effect fieldless operands out of positional
# src/dst lists, but keeps fieldless OPR_SIMM32 because it is value-bearing.
# ---------------------------------------------------------------------------
def test_execute_operand_participation_keeps_only_fieldless_simm32():
    vcc = Operand('vcc', 64, 'OPR_VCC', True, False, False, False, 0, fieldless=True)
    simm32 = Operand(
        'simm32', 32, 'OPR_SIMM32', True, False, False, False, 1, fieldless=True
    )
    src0 = Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2)

    assert not CodeGenerator._execute_operand_participates(vcc)
    assert CodeGenerator._execute_operand_participates(simm32)
    assert CodeGenerator._execute_operand_participates(src0)


# ---------------------------------------------------------------------------
# Fieldless operand policy table. Capability, role, display, and the (stubbed)
# architectural-effect column are fields of ONE FieldlessPolicy entry per
# operand type. Runtime inertness is driven by construction-time capability
# flags on Operand.
# ---------------------------------------------------------------------------

# The 11 fieldless operand types observed currently.
_OBSERVED_FIELDLESS_TYPES = [
    'OPR_DSMEM',
    'OPR_EXEC',
    'OPR_FLAT_SCRATCH',
    'OPR_GPUMEM',
    'OPR_PC',
    'OPR_SDST_EXEC',
    'OPR_SDST_M0',
    'OPR_SIMM32',
    'OPR_SSRC_SPECIAL_SCC',
    'OPR_VCC',
    'OPR_VGPR',
]

_INERT_CAPS = FieldlessCaps(reads_value=False, writable=False, is_vgpr=False)


def test_fieldless_policy_table_is_golden():
    # Golden lock on the ENTIRE policy table -- role + caps + display + effect
    # for every fieldless operand type. This is the single place each type's
    # policy is pinned, so changing any one type is a conscious, reviewed edit
    # here. Captured invariants: OPR_SIMM32 is the only value-reading (literal)
    # type, every other type is fully inert, and the image OPR_VGPR placeholder
    # is NOT vgpr-capable (so it never reads as a real v0).
    sr = FieldlessCategory.SPECIAL_REGISTER
    mp = FieldlessCategory.MEMORY_PSEUDO
    hidden = FieldlessDisplay.HIDDEN
    value = FieldlessCaps(reads_value=True, writable=False, is_vgpr=False)

    def sreg(cls):
        return FieldlessEffect(special_reg=cls)

    assert fieldless_policy_mod._FIELDLESS_POLICY == {
        'OPR_SIMM32': FieldlessPolicy(FieldlessCategory.LITERAL, value, hidden, None),
        'OPR_VCC': FieldlessPolicy(sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.VCC)),
        'OPR_EXEC': FieldlessPolicy(
            sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.EXEC)
        ),
        'OPR_SDST_EXEC': FieldlessPolicy(
            sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.EXEC)
        ),
        'OPR_SSRC_SPECIAL_SCC': FieldlessPolicy(
            sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.SCC)
        ),
        'OPR_PC': FieldlessPolicy(sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.PC)),
        'OPR_SDST_M0': FieldlessPolicy(
            sr, _INERT_CAPS, hidden, sreg(SpecialRegClass.M0)
        ),
        'OPR_DSMEM': FieldlessPolicy(mp, _INERT_CAPS, hidden, None),
        'OPR_GPUMEM': FieldlessPolicy(mp, _INERT_CAPS, hidden, None),
        # FLAT_SCRATCH is the per-wave scratch-base register state (SGPR pair),
        # so it is classified as a special register.
        'OPR_FLAT_SCRATCH': FieldlessPolicy(sr, _INERT_CAPS, hidden, None),
        'OPR_VGPR': FieldlessPolicy(
            FieldlessCategory.PLACEHOLDER, _INERT_CAPS, hidden, None
        ),
    }


def test_only_simm32_fieldless_type_reads_a_value():
    # Current status of fieldless operands: exactly one fieldless
    # type is value-bearing. Computed from the table, so promoting any other
    # type to readable fails here and forces a conscious decision.
    value_reading = [
        t for t in _OBSERVED_FIELDLESS_TYPES if fieldless_policy(t).caps.reads_value
    ]
    assert value_reading == ['OPR_SIMM32']


def test_fieldless_policy_placeholder_vaddr_is_not_vgpr():
    # The image vaddr placeholder is type OPR_VGPR but must never be
    # vgpr-capable, or it would read as a real v0.
    policy = fieldless_policy('OPR_VGPR')
    assert policy.role == FieldlessCategory.PLACEHOLDER
    assert policy.caps.is_vgpr is False
    assert policy.caps.reads_value is False


def test_fieldless_policy_every_observed_type_is_classified():
    # Ties the observed type list to the table: every observed fieldless type
    # has a real (non-UNKNOWN) policy entry, so nothing silently falls through
    # to the inert-unknown default.
    for opr_type in _OBSERVED_FIELDLESS_TYPES:
        assert fieldless_policy(opr_type).role != FieldlessCategory.UNKNOWN, opr_type


def test_fieldless_policy_unknown_type_defaults_inert():
    policy = fieldless_policy('OPR_MADE_UP_TYPE')
    assert policy.role == FieldlessCategory.UNKNOWN
    assert policy.caps == _INERT_CAPS


def _fake_spec(fieldless_types):
    return SimpleNamespace(
        arch_name='fake_isa',
        fieldless_operand_types=set(fieldless_types),
    )


def test_validate_fieldless_taxonomy_accepts_classified_types():
    # A spec whose fieldless types are all classified passes cleanly. This is
    # the fatal parse-time gate (runs inside Parser.parse); it enforces the
    # taxonomy against exactly what a real regeneration parses, including
    # non-committed targets such as gfx1250.
    validate_fieldless_taxonomy(_fake_spec(_OBSERVED_FIELDLESS_TYPES))
    validate_fieldless_taxonomy(_fake_spec(set()))


def test_validate_fieldless_taxonomy_is_fatal_on_unknown_type():
    # The tripwire: an unclassified fieldless type raises, and the message
    # names the offender and the arch without flagging a classified sibling.
    with pytest.raises(ValueError) as exc:
        validate_fieldless_taxonomy(_fake_spec({'OPR_VCC', 'OPR_BRAND_NEW'}))
    msg = str(exc.value)
    assert 'OPR_BRAND_NEW' in msg
    assert 'fake_isa' in msg
    assert 'OPR_VCC' not in msg


def test_fieldless_policy_entry_carries_all_columns_on_one_object():
    # Single-table design: capability, role, display, and effect are fields of
    # one FieldlessPolicy object, so they cannot drift as independent type-keyed
    # maps.
    policy = fieldless_policy('OPR_VCC')
    assert policy.role == FieldlessCategory.SPECIAL_REGISTER
    assert isinstance(policy.caps, FieldlessCaps)
    assert policy.display == FieldlessDisplay.HIDDEN
    assert policy.effect == FieldlessEffect(special_reg=SpecialRegClass.VCC)


# The special register each SPECIAL_REGISTER fieldless type denotes, consumed by
# the generated to_special_reg_class() helper. SDST_EXEC is the SDST-namespace
# alias for EXEC, so both map to EXEC.
_EXPECTED_SPECIAL_REG = {
    'OPR_VCC': SpecialRegClass.VCC,
    'OPR_EXEC': SpecialRegClass.EXEC,
    'OPR_SDST_EXEC': SpecialRegClass.EXEC,
    'OPR_SSRC_SPECIAL_SCC': SpecialRegClass.SCC,
    'OPR_PC': SpecialRegClass.PC,
    'OPR_SDST_M0': SpecialRegClass.M0,
}


def test_special_register_types_carry_expected_effect():
    # Every SPECIAL_REGISTER fieldless type maps to its architectural special
    # register through the effect column; the mapping is the sole source the
    # generator lowers into to_special_reg_class().
    for opr_type, reg in _EXPECTED_SPECIAL_REG.items():
        policy = fieldless_policy(opr_type)
        assert policy.role == FieldlessCategory.SPECIAL_REGISTER, opr_type
        assert policy.effect == FieldlessEffect(special_reg=reg), opr_type


def test_special_register_role_and_effect_agree():
    # Every type carrying a special-register effect is a SPECIAL_REGISTER, and
    # the effect set is exactly the mapped types. FLAT_SCRATCH is the one
    # SPECIAL_REGISTER whose architectural effect is intentionally deferred (no
    # SpecialRegClass produced yet), so role and effect are not identical sets.
    # Guards against a mapped special reg gaining/losing its effect, or a
    # non-special type sprouting one, without a conscious edit.
    with_effect = {
        t
        for t in _OBSERVED_FIELDLESS_TYPES
        if fieldless_policy(t).effect is not None
        and fieldless_policy(t).effect.special_reg is not None
    }
    is_special = {
        t
        for t in _OBSERVED_FIELDLESS_TYPES
        if fieldless_policy(t).role == FieldlessCategory.SPECIAL_REGISTER
    }
    assert with_effect == set(_EXPECTED_SPECIAL_REG)
    assert with_effect < is_special
    assert is_special - with_effect == {'OPR_FLAT_SCRATCH'}


def test_types_without_special_reg_mapping_have_no_effect():
    # Everything outside the mapped special-register set models no architectural
    # effect yet: literal, memory-pseudo, placeholder, and the deferred
    # FLAT_SCRATCH special register.
    for opr_type in _OBSERVED_FIELDLESS_TYPES:
        if opr_type in _EXPECTED_SPECIAL_REG:
            continue
        assert fieldless_policy(opr_type).effect is None, opr_type


def test_fieldless_caps_stmt_emits_policy_caps():
    # Locks the caps-statement emission FORMAT (a caps triple -> C++), for a
    # value-bearing and an inert operand. Per-type caps values are pinned by the
    # golden table above, so this need not enumerate every type.
    assert (
        CodeGenerator._fieldless_caps_stmt('simm32', 'OPR_SIMM32')
        == 'simm32.apply_fieldless_caps(true, false, false);'
    )
    assert (
        CodeGenerator._fieldless_caps_stmt('vcc', 'OPR_VCC')
        == 'vcc.apply_fieldless_caps(false, false, false);'
    )


# ---------------------------------------------------------------------------
# FieldlessCaps.__post_init__ enforces the cross-language contract: writable or
# is_vgpr each imply reads_value. This mirrors the reads_value() gates on
# simd_capable() / resolved_vgpr_offset_for_operand in isa_operand_simd_inl.h
# (a writable-but-!reads_value operand would be dropped from the SIMD path), so
# pin it directly rather than only through full-regen exercise.
# ---------------------------------------------------------------------------
def test_fieldless_caps_rejects_writable_without_reads_value():
    with pytest.raises(ValueError, match='writable/is_vgpr implies reads_value'):
        FieldlessCaps(reads_value=False, writable=True, is_vgpr=False)


def test_fieldless_caps_rejects_vgpr_without_reads_value():
    with pytest.raises(ValueError, match='writable/is_vgpr implies reads_value'):
        FieldlessCaps(reads_value=False, writable=False, is_vgpr=True)


def test_fieldless_caps_accepts_invariant_respecting_combos():
    # Fully inert, value-reading-only, and fully-capable are all valid.
    FieldlessCaps(reads_value=False, writable=False, is_vgpr=False)
    FieldlessCaps(reads_value=True, writable=False, is_vgpr=False)
    FieldlessCaps(reads_value=True, writable=True, is_vgpr=True)


# ---------------------------------------------------------------------------
# An interior fieldless operand (the FMAMK/MADMK inline literal) means the
# field-bearing sources are NOT at their positional src indices. This is the
# invariant the generator's DPP/SDWA permute path relies on when it addresses
# permuted sources by name (and asserts they are field-bearing) instead of by
# src_operands_[] index.
# ---------------------------------------------------------------------------
def _v_fmamk_f32():
    """VOP2 FMAMK layout: vdst, src0, fieldless simm32, vsrc1."""
    ops = [
        Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
        Operand(
            'simm32', 32, 'OPR_SIMM32', True, False, False, False, 3, fieldless=True
        ),
        Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 4),
    ]
    _uniquify_fieldless_names(ops)
    return Instruction('V_FMAMK_F32', 'ENC_VOP2', 0, ops)


def test_interior_fieldless_shifts_positional_source_index():
    inst = _v_fmamk_f32()
    # The literal sits between the two field-bearing sources, so a by-index read
    # of src_operands_[1] would grab the fieldless literal, not vsrc1.
    assert [op.name for op in inst.src_operands] == ['src0', 'simm32', 'vsrc1']
    assert inst.src_operands[1].name == 'simm32'
    assert inst.src_operands[1].fieldless

    # The generator selects DPP/SDWA permute sources by filtering out fieldless
    # operands, then addresses them by name. That selection yields the real
    # field-bearing sources in order, and both are field-bearing (matching the
    # tripwire assert in the permute path).
    field_bearing_inputs = [
        op for op in inst.operands if op.is_input and not op.fieldless
    ]
    assert [op.name for op in field_bearing_inputs] == ['src0', 'vsrc1']
    assert all(not op.fieldless for op in field_bearing_inputs[:2])
