# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for fieldless operand policy.

A fieldless operand (no ``<FieldName>`` in the MR ISA) has no encoding field to
decode; the generator constructs it from a canonical value and must decide how
it behaves at runtime, how it is classified, how it displays, and (later) which
architectural effect it represents. This module is one policy table: every
fieldless operand type maps to a single :class:`FieldlessPolicy` object whose
columns are

  * ``role``    -- coarse semantic category (:class:`FieldlessCategory`),
  * ``caps``    -- runtime capability for the normal read/write/SIMD accessors
                   (:class:`FieldlessCaps`),
  * ``display`` -- disassembly display policy (:class:`FieldlessDisplay`),
  * ``effect``  -- architectural-effect metadata (:class:`FieldlessEffect`): the
                   special register a fieldless operand corresponds to, or
                   ``None`` for types with no effect modeled yet.

``role``, ``caps`` and ``effect`` are live today: the generator consumes them to
classify and lower each operand and to emit ``to_special_reg_class()``.
``display`` is a staged extension point.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FieldlessCategory(enum.Enum):
    """Coarse semantic category of a fieldless operand.

    Describes what the operand fundamentally represents, independent of its
    synthesized display name and its MR ISA ``operand_type`` spelling. This is a
    generation-time (Python) classification consumed by the code generator to
    decide how to lower the operand; it is not emitted into C++.

      SPECIAL_REGISTER: a real architectural register that has no encoding
        field -- VCC/EXEC/SCC/M0/PC and FLAT_SCRATCH (per-wave scratch-base
        register state held in SGPRs). The concrete operand_type -> C++
        RegClass mapping is carried by ``FieldlessEffect.special_reg`` (a
        :class:`SpecialRegClass`) for VCC/EXEC/SCC/M0/PC; FLAT_SCRATCH is
        classified here but supplies no special_reg yet.
      MEMORY_PSEUDO: a memory-resource pseudo-operand (global/LDS) that carries
        memory-effect meaning but never a register value.
      LITERAL: a value-bearing fieldless literal (``OPR_SIMM32``).
      PLACEHOLDER: an inert placeholder modeling no effect yet (gfx12 image
        ``vaddr`` fieldless ``OPR_VGPR``).
      UNKNOWN: no classification -- a fieldless type nobody has triaged. Used
        as the default for an unknown type so it stays safely inert
    """

    SPECIAL_REGISTER = 'special_register'
    MEMORY_PSEUDO = 'memory_pseudo'
    LITERAL = 'literal'
    PLACEHOLDER = 'placeholder'
    UNKNOWN = 'unknown'


class FieldlessDisplay(enum.Enum):
    """Disassembly display policy for a fieldless operand.

    Everything is HIDDEN today (fieldless operands are suppressed from
    disassembly). VISIBLE is reserved for a future display-policy slice, e.g.
    if the assembler form requires a value-bearing literal to be printed.
    """

    HIDDEN = 'hidden'


class SpecialRegClass(enum.Enum):
    """An architectural special register a fieldless operand corresponds to.

    Mirrors the special (non-indexed) members of the C++ ``RegClass`` enum
    (``register_set.h``): each value's name is the C++ enumerator (so generator
    lowers ``SpecialRegClass.VCC`` to ``RegClass::VCC``). Only the classes a
    fieldless operand currently denotes are listed. Two C++ ``RegClass`` values
    are not produced here: FLAT_SCRATCH (its operand ``OPR_FLAT_SCRATCH``
    classifies as ``SPECIAL_REGISTER`` in the policy table but supplies no
    ``FieldlessEffect.special_reg``) and TTMP (a per-index trap-temporary file
    that is not a fieldless operand at all), so no fieldless operand lowers to
    either.
    """

    EXEC = 'EXEC'
    VCC = 'VCC'
    SCC = 'SCC'
    M0 = 'M0'
    PC = 'PC'


@dataclass(frozen=True)
class FieldlessEffect:
    """Architectural-effect metadata for a fieldless operand (one effect object).

    Lives on the :class:`FieldlessPolicy` row so effect data cannot drift into a
    separate type-keyed map. For now it carries only the special register the
    operand denotes; memory-effect fields (for the ``MEMORY_PSEUDO`` types) are
    deferred.

    Attributes:
        special_reg: The special register this operand reads/writes as an
            architectural effect, or ``None`` if it denotes no special register.
    """

    special_reg: SpecialRegClass | None = None


@dataclass(frozen=True)
class FieldlessCaps:
    """Runtime capability of a fieldless operand for the normal accessors.

    These map directly to construction-time capability flags on the C++
    ``Operand``; the normal read/write/SIMD accessors query the stored flags
    instead of re-deriving them from ``fieldless_`` plus operand type.

    Attributes:
        reads_value: The operand yields a real value through the normal read /
            SIMD accessors (``read_scalar``/``read_lane``/``simd_capable``/...).
            False means those accessors are inert (benign zero / no-op). Only
            the literal ``OPR_SIMM32`` reads a value today.
        writable: The operand is a valid write target for the normal write
            accessors. False for every fieldless operand today (no fieldless
            operand is a write/def target through the ordinary write path).
        is_vgpr: The operand resolves to VGPR/AccVGPR storage. False for every
            fieldless operand today -- notably including the image ``vaddr``
            placeholder, whose type is ``OPR_VGPR`` but which must NOT behave
            like a real ``v0``.

    Invariant: ``writable`` and ``is_vgpr`` each imply ``reads_value``. In
    ``isa_operand_simd_inl.h`` the SIMD write path (``write_lane_chunk``) gates
    on ``is_writable()``, but ``simd_capable()`` -- required on every operand,
    including the destination, for the SIMD fast path to run at all -- gates on
    ``reads_value()``, and ``resolved_vgpr_offset_for_operand`` (the write fast
    path's address resolution) is inert when ``!reads_value()``. So a writable
    (or VGPR) operand that does not read a value would be dropped from the SIMD
    path and its write silently no-op the fast path. ``__post_init__`` enforces
    the invariant to keep that combination unrepresentable.
    """

    reads_value: bool
    writable: bool
    is_vgpr: bool

    def __post_init__(self) -> None:
        if (self.writable or self.is_vgpr) and not self.reads_value:
            raise ValueError(
                f'FieldlessCaps invariant violated: writable/is_vgpr implies '
                f'reads_value ({self!r}). Before adding such an entry, revisit '
                f'the reads_value() gates on simd_capable() and '
                f'resolved_vgpr_offset_for_operand in isa_operand_simd_inl.h.'
            )


#: Inert capability: benign-zero reads, non-writable, non-VGPR. The default for
#: every fieldless operand except the value-bearing literal.
_INERT = FieldlessCaps(reads_value=False, writable=False, is_vgpr=False)

#: Value-only capability: reads a real value (the literal), but is not a write
#: target and is not VGPR storage.
_VALUE_ONLY = FieldlessCaps(reads_value=True, writable=False, is_vgpr=False)


@dataclass(frozen=True)
class FieldlessPolicy:
    """One fieldless operand type's complete policy (one row of the table).

    Attributes:
        role: Semantic category.
        caps: Runtime capability for the normal accessors.
        display: Disassembly display policy.
        effect: Architectural-effect metadata (:class:`FieldlessEffect`), or
            ``None`` for a type with no effect modeled yet (literal, memory
            pseudo, placeholder). Present here so effect data lives on the same
            policy row rather than in a separate type-keyed map.
    """

    role: FieldlessCategory
    caps: FieldlessCaps
    display: FieldlessDisplay = FieldlessDisplay.HIDDEN
    effect: FieldlessEffect | None = None


#: The fieldless operand policy table: every fieldless ``operand_type`` ->
#: :class:`FieldlessPolicy`. This is the single source of truth. Adding a new
#: fieldless type to an ISA XML requires a conscious entry here.
_FIELDLESS_POLICY: dict[str, FieldlessPolicy] = {
    # Value-bearing literal: the one fieldless operand read as a real value.
    'OPR_SIMM32': FieldlessPolicy(FieldlessCategory.LITERAL, _VALUE_ONLY),
    # Architectural special registers: inert through normal accessors, but their
    # architectural effect (the special register they denote) is carried in the
    # effect column and consumed by the generated to_special_reg_class() helper.
    # SDST_EXEC is the SDST-namespace alias for EXEC, so both map to EXEC.
    'OPR_VCC': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.VCC),
    ),
    'OPR_EXEC': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.EXEC),
    ),
    'OPR_SDST_EXEC': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.EXEC),
    ),
    'OPR_SSRC_SPECIAL_SCC': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.SCC),
    ),
    'OPR_PC': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.PC),
    ),
    'OPR_SDST_M0': FieldlessPolicy(
        FieldlessCategory.SPECIAL_REGISTER,
        _INERT,
        effect=FieldlessEffect(SpecialRegClass.M0),
    ),
    # Memory pseudo-operands: no register value; drive memory-effect metadata
    # in a later slice.
    'OPR_DSMEM': FieldlessPolicy(FieldlessCategory.MEMORY_PSEUDO, _INERT),
    'OPR_GPUMEM': FieldlessPolicy(FieldlessCategory.MEMORY_PSEUDO, _INERT),
    # FLAT_SCRATCH is architectural register state so consider it special.
    'OPR_FLAT_SCRATCH': FieldlessPolicy(FieldlessCategory.SPECIAL_REGISTER, _INERT),
    # gfx12 image address/coordinate placeholder (fieldless OPR_VGPR). Inert,
    # and explicitly non-VGPR so it never reads as a real v0.
    'OPR_VGPR': FieldlessPolicy(FieldlessCategory.PLACEHOLDER, _INERT),
}

#: Default policy for a fieldless type not in the table: inert + UNKNOWN role.
#: Keeps an untriaged type safely inert; the taxonomy gate
#: (validate_fieldless_taxonomy) then rejects any observed UNKNOWN at parse time.
_UNKNOWN_POLICY = FieldlessPolicy(FieldlessCategory.UNKNOWN, _INERT)


def fieldless_policy(operand_type: str) -> FieldlessPolicy:
    """Return the :class:`FieldlessPolicy` for a fieldless operand type.

    The caller is responsible for only asking about operands that are actually
    fieldless (e.g. fieldless ``OPR_VGPR`` is the image placeholder, but an
    ordinary field-bearing ``OPR_VGPR`` is a plain vector register and must not
    be routed through here). An unclassified type returns
    :data:`_UNKNOWN_POLICY` (inert).
    """
    return _FIELDLESS_POLICY.get(operand_type, _UNKNOWN_POLICY)


def operand_participates(fieldless: bool, operand_type: str) -> bool:
    """Whether an operand is visible to execute generation and the cross-ISA
    sharing signature.

    Field-bearing operands always participate. A fieldless operand participates
    only if its policy says it does. This is the single predicate behind both
    ``_execute_operand_participates`` (which operands the execute body sees) and
    ``_operand_signature`` (which operands define shareability), so the two
    cannot drift: an operand that changes the generated body is exactly one that
    changes the sharing signature.
    """
    return not fieldless or fieldless_policy(operand_type).caps.reads_value


def validate_fieldless_taxonomy(spec) -> None:
    """Fatally reject a parsed spec containing an unclassified fieldless type.

    This is the production tripwire for the fieldless taxonomy: it runs on
    every parse (see ``Parser.parse``), so regen cannot silently let a new
    fieldless operand type fall through to inert handling. Every observed
    fieldless type must have a real (non-UNKNOWN) entry in the shared policy
    table.

    ``spec`` is duck-typed: it only needs ``fieldless_operand_types`` (a set of
    observed fieldless ``operand_type`` strings) and ``arch_name``.

    Raises:
        ValueError: if any observed fieldless operand type classifies as
            :attr:`FieldlessCategory.UNKNOWN`.
    """
    unknown = sorted(
        t
        for t in spec.fieldless_operand_types
        if fieldless_policy(t).role == FieldlessCategory.UNKNOWN
    )
    if unknown:
        raise ValueError(
            f'Unclassified fieldless operand type(s) while parsing '
            f'{spec.arch_name}: {unknown}. Add each to _FIELDLESS_POLICY '
            f'(fieldless_policy.py) with its role/caps/display and pin it in '
            f'tests/test_fieldless_operands.py.'
        )
