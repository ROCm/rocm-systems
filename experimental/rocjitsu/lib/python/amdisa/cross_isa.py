# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-ISA instruction analysis for shared execute() deduplication.

The ``CrossIsaAnalyzer`` compares instructions across all 9 AMDGPU ISAs
and classifies each into one of three categories:

- **universal** — identical encoding fields and semantics on all ISAs where
  the instruction appears.  The execute() body can be emitted once as a
  shared template.
- **family_shared** — identical within a family (e.g., all CDNA or all RDNA)
  but differs across families.  A family-scoped shared template is emitted.
- **isa_exclusive** — present on only one ISA, or semantically/structurally
  unique.  The execute() body is emitted per-ISA as today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec, MicrocodeField
    from amdisa.semantics import InstructionSemantics, SemanticsSpec


@dataclass
class SharedInstInfo:
    """Metadata for a shared instruction."""

    mnemonic: str
    encoding_name: str
    field_layout: tuple[tuple[str, int], ...]  # ((field_name, bit_cnt), ...)
    semantic_class: str
    operation: str | None
    data_type: str | None
    isa_names: list[str]


@dataclass
class SharedInstructionPlan:
    """Result of cross-ISA analysis.  Consumed by the codegen."""

    universal: dict[str, SharedInstInfo] = field(default_factory=dict)
    family_shared: dict[str, dict[str, SharedInstInfo]] = field(default_factory=dict)
    isa_exclusive: dict[str, set[str]] = field(default_factory=dict)

    @property
    def total_universal(self) -> int:
        return len(self.universal)

    @property
    def total_family_shared(self) -> int:
        return sum(len(v) for v in self.family_shared.values())

    @property
    def total_exclusive(self) -> int:
        return sum(len(v) for v in self.isa_exclusive.values())


def _field_signature(enc: InstEncoding) -> tuple[tuple[str, int], ...]:
    """Return a canonical tuple of (field_name, bit_count) for an encoding."""
    return tuple(
        (f.name, f.bit_cnt)
        for f in sorted(enc.ucode_fields, key=lambda f: f.bit_offset)
    )


def _sem_key(sem: InstructionSemantics | None) -> tuple[str, str | None, str | None]:
    """Return a hashable semantic identity for comparison."""
    if sem is None:
        return ('nop', None, None)
    return (sem.semantic_class, sem.operation, sem.data_type)


def _operand_signature(inst: Instruction) -> tuple[tuple[str, str, int, bool, bool], ...]:
    """Return a canonical tuple of operand (name, type, size, is_input, is_output)."""
    return tuple(
        (op.name, op.operand_type, op.size, op.is_input, op.is_output)
        for op in inst.operands
    )


# ISA family groupings for family_shared classification.
CDNA_ISAS = frozenset({'cdna1', 'cdna2', 'cdna3', 'cdna4'})
RDNA_ISAS = frozenset({'rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4'})
ALL_ISAS = CDNA_ISAS | RDNA_ISAS

# Finer sub-families for encoding-level sharing.
_FAMILIES: list[tuple[str, frozenset[str]]] = [
    ('cdna', CDNA_ISAS),
    ('rdna', RDNA_ISAS),
    ('gfx9', frozenset({'cdna1', 'cdna2', 'cdna3', 'cdna4'})),
    ('gfx10', frozenset({'rdna1', 'rdna2'})),
    ('gfx11', frozenset({'rdna3', 'rdna3_5'})),
]


class CrossIsaAnalyzer:
    """Analyze instruction overlap across multiple ISA specs."""

    def analyze(
        self,
        specs: list[tuple[str, IsaSpec, SemanticsSpec | None]],
    ) -> SharedInstructionPlan:
        """Run the cross-ISA analysis.

        Args:
            specs: List of ``(isa_name, isa_spec, semantics_spec)`` tuples,
                one per ISA.

        Returns:
            A ``SharedInstructionPlan`` classifying each instruction.
        """
        plan = SharedInstructionPlan()

        # Step 1: Build mnemonic → list of (isa_name, encoding, instruction, sem)
        inst_map: dict[str, list[tuple[str, InstEncoding, Instruction, InstructionSemantics | None]]] = {}
        for isa_name, spec, sem_spec in specs:
            for enc in spec.inst_encodings:
                if not enc.insts:
                    continue
                # Skip alt encodings — their instructions are classified
                # under the parent encoding for cross-ISA comparison.
                if spec.profile.is_alt_encoding(enc.enc_name):
                    continue
                # Collect instructions from this encoding plus any child
                # alt encodings whose instructions reside in the parent's
                # file.
                all_insts = list(enc.insts)
                for child_enc in spec.inst_encodings:
                    if (
                        child_enc.insts
                        and spec.profile.is_alt_encoding(child_enc.enc_name)
                        and spec.profile.derive_parent_enc_name(
                            child_enc.enc_name
                        ) == enc.enc_name
                    ):
                        all_insts.extend(child_enc.insts)
                for inst in all_insts:
                    sem = sem_spec.instructions.get(inst.name) if sem_spec else None
                    inst_map.setdefault(inst.mnemonic, []).append(
                        (isa_name, enc, inst, sem)
                    )

        isa_names_set = {name for name, _, _ in specs}

        # Step 2: Classify each instruction.
        for mnemonic, entries in inst_map.items():
            present_isas = {e[0] for e in entries}

            if len(present_isas) == 1:
                # Only on one ISA → exclusive.
                isa = next(iter(present_isas))
                plan.isa_exclusive.setdefault(isa, set()).add(mnemonic)
                continue

            # Check if all entries have the same encoding layout, semantics,
            # AND operand types.  Operand type names differ across ISAs
            # (e.g., OPR_SLEEP exists on RDNA4 but not CDNA1), so instructions
            # with ISA-specific operand types cannot be shared.
            field_sigs = set()
            sem_keys = set()
            opnd_sigs = set()
            for isa_name, enc, inst, sem in entries:
                field_sigs.add(_field_signature(enc))
                sem_keys.add(_sem_key(sem))
                opnd_sigs.add(_operand_signature(inst))

            same_structure = (
                len(field_sigs) == 1
                and len(sem_keys) == 1
                and len(opnd_sigs) == 1
            )

            if same_structure and present_isas == isa_names_set:
                # Universal — identical on ALL ISAs.
                _, enc0, inst0, sem0 = entries[0]
                plan.universal[mnemonic] = SharedInstInfo(
                    mnemonic=mnemonic,
                    encoding_name=enc0.enc_name,
                    field_layout=_field_signature(enc0),
                    semantic_class=_sem_key(sem0)[0],
                    operation=_sem_key(sem0)[1],
                    data_type=_sem_key(sem0)[2],
                    isa_names=sorted(present_isas),
                )
                continue

            if same_structure and len(present_isas) >= 2:
                # Same structure on a subset of ISAs → family_shared.
                # Determine which family this belongs to.
                family_name = self._classify_family(present_isas)
                _, enc0, inst0, sem0 = entries[0]
                plan.family_shared.setdefault(family_name, {})[mnemonic] = SharedInstInfo(
                    mnemonic=mnemonic,
                    encoding_name=enc0.enc_name,
                    field_layout=_field_signature(enc0),
                    semantic_class=_sem_key(sem0)[0],
                    operation=_sem_key(sem0)[1],
                    data_type=_sem_key(sem0)[2],
                    isa_names=sorted(present_isas),
                )
                continue

            # Different structure across ISAs → each gets its own classification.
            # Group by (field_sig, sem_key, operand_sig) and classify each group.
            groups: dict[tuple, list[str]] = {}
            for isa_name, enc, inst, sem in entries:
                key = (_field_signature(enc), _sem_key(sem), _operand_signature(inst))
                groups.setdefault(key, []).append(isa_name)

            for (fsig, skey, _osig), group_isas in groups.items():
                if len(group_isas) >= 2:
                    family_name = self._classify_family(set(group_isas))
                    _, enc0, inst0, sem0 = next(
                        e for e in entries if e[0] == group_isas[0]
                    )
                    plan.family_shared.setdefault(family_name, {})[mnemonic] = SharedInstInfo(
                        mnemonic=mnemonic,
                        encoding_name=enc0.enc_name,
                        field_layout=fsig,
                        semantic_class=skey[0],
                        operation=skey[1],
                        data_type=skey[2],
                        isa_names=sorted(group_isas),
                    )
                else:
                    plan.isa_exclusive.setdefault(group_isas[0], set()).add(mnemonic)

        return plan

    @staticmethod
    def _classify_family(isas: set[str]) -> str:
        """Return a family name for a set of ISAs."""
        if isas <= CDNA_ISAS:
            return 'cdna'
        if isas <= RDNA_ISAS:
            return 'rdna'
        # Mixed — use a descriptive name.
        return '_'.join(sorted(isas))
