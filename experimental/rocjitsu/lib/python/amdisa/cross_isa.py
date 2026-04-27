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
    """Result of cross-ISA analysis.  Consumed by the codegen.

    Key schemes differ across the three dicts:

    - ``universal``: keyed by mnemonic alone.  Safe because an instruction
      can only be universal if ALL entries across ALL ISAs have identical
      field layout, semantics, and operands — which implies a single
      encoding.  Multi-encoding mnemonics (e.g., v_mov_b32 in VOP1 + VOP3)
      always have differing field layouts, so they never reach universal;
      they go through the grouping path into ``family_shared`` instead.

    - ``family_shared``: keyed by family name → ``(mnemonic, encoding_name)``.
      The composite key prevents collisions between different encodings of
      the same mnemonic (e.g., VOP1 vs VOP3 variants).

    - ``isa_exclusive``: keyed by ISA name → set of mnemonics.  Encoding
      information is not tracked because the per-ISA codegen processes each
      encoding file independently and doesn't need it here.
    """

    universal: dict[str, SharedInstInfo] = field(default_factory=dict)
    family_shared: dict[str, dict[tuple[str, str], SharedInstInfo]] = field(default_factory=dict)
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
    """Return a canonical tuple of (field_name, bit_count) for an encoding.

    Padding fields (``pad_*``) and the format identifier (``encoding``) are
    excluded — they are never referenced by execute() bodies, and their
    layout varies across XML spec versions without affecting semantics.
    """
    return tuple(
        (f.name, f.bit_cnt)
        for f in sorted(enc.ucode_fields, key=lambda f: f.bit_offset)
        if not f.name.startswith('pad_') and f.name != 'encoding'
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
# Sub-family groupings reflect encoding layout generations: ISAs within
# the same sub-family share encoding field layouts (e.g., ENC_MUBUF
# changed between RDNA2 and RDNA3), while ISAs across sub-families may
# diverge.  _classify_family() checks sub-families first (finest to
# coarsest), so instructions that are identical within a generation get
# their own bucket rather than colliding at the coarse family level.
CDNA_GFX9 = frozenset({'cdna1', 'cdna2', 'cdna3', 'cdna4'})
RDNA_GFX10 = frozenset({'rdna1', 'rdna2'})
RDNA_GFX11 = frozenset({'rdna3', 'rdna3_5'})
RDNA_GFX12 = frozenset({'rdna4'})

CDNA_ISAS = CDNA_GFX9  # All CDNA ISAs currently share one generation.
RDNA_ISAS = RDNA_GFX10 | RDNA_GFX11 | RDNA_GFX12
ALL_ISAS = CDNA_ISAS | RDNA_ISAS

# Family table used by _classify_family() to pick the tightest-fitting
# bucket for a set of ISAs.  Order is informational only — the lookup
# picks the entry with the smallest ``members`` set that is a superset
# of the input.  On exact-size ties (e.g., CDNA_GFX9 vs CDNA_ISAS today,
# both size 4), the first matching entry wins, so finest-first ordering
# is preferred.
#
# Note for Tony (re commit 2316f85): you introduced an equivalent
# `_FAMILIES` table and a smallest-fits ``_classify_family`` in your
# v_pack_b32_f16 commit.  This branch adopts that same shape so your
# rebase/merge should be near-trivial here.  The deliberate differences
# vs. your version are the bucket *names* — we use ``rdna_gfx10``,
# ``cdna_gfx9``, etc. instead of bare ``gfx10`` / ``cdna``, because the
# generated code reads more clearly with the family prefix and our new
# tests in ``tests/test_cross_isa_analyzer.py`` hardcode these names.
# If you'd rather keep your shorter names, just rename the strings here
# and update the matching assertions in
# tests/test_cross_isa_analyzer.py (test_classify_family_sub_families,
# test_rdna_cross_generation_no_collision, and
# test_family_shared_uses_sub_family_keys) — the algorithm is identical.
_FAMILIES: list[tuple[str, frozenset[str]]] = [
    # Sub-families (finest) — encoding layouts are identical within these.
    ('rdna_gfx12', RDNA_GFX12),
    ('rdna_gfx11', RDNA_GFX11),
    ('rdna_gfx10', RDNA_GFX10),
    ('cdna_gfx9',  CDNA_GFX9),
    # Coarse families — spans sub-families but stays within one family.
    ('rdna',       RDNA_ISAS),
    ('cdna',       CDNA_ISAS),
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

        # Step 1: Build mnemonic → list of (isa_name, encoding, instruction, sem).
        # Keyed by mnemonic only — entries from different encodings of the
        # same mnemonic (e.g., v_mov_b32 in VOP1 and VOP3) land in the same
        # bucket.  Step 2's classification logic handles multi-encoding
        # mnemonics: their differing field layouts cause same_structure to
        # be False, routing them through the grouping path which separates
        # them by (field_sig, sem_key, operand_sig) and assigns each
        # per-encoding group independently.
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
                # Any entry is representative: same_structure guarantees
                # identical field_sig, sem_key, and opnd_sig across all.
                _, enc0, inst0, sem0 = entries[0]
                # Defensive: same_structure guarantees identical field
                # layouts, but verify that the encoding name is also
                # consistent — the mnemonic-only key in plan.universal
                # assumes a single encoding name across all ISAs.
                enc_names = {e[1].enc_name for e in entries}
                assert len(enc_names) == 1, (
                    f"Universal instruction '{mnemonic}' has identical field "
                    f"layouts but inconsistent encoding names: {enc_names}. "
                    f"The universal dict is keyed by mnemonic alone and "
                    f"cannot represent multiple encoding names. This needs "
                    f"a composite key or per-encoding classification."
                )
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
                plan.family_shared.setdefault(family_name, {})[(mnemonic, enc0.enc_name)] = SharedInstInfo(
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
            # This path also handles multi-encoding mnemonics (e.g., v_mov_b32
            # in VOP1 + VOP3): the differing field layouts across encodings
            # cause same_structure to be False, so each encoding is classified
            # independently via the sub-groups below.
            # Group by (field_sig, sem_key, operand_sig) and classify each group.
            groups: dict[tuple, list[str]] = {}
            for isa_name, enc, inst, sem in entries:
                key = (_field_signature(enc), _sem_key(sem), _operand_signature(inst))
                groups.setdefault(key, []).append(isa_name)

            for (fsig, skey, osig), group_isas in groups.items():
                if len(group_isas) >= 2:
                    # Shareable across 2+ ISAs
                    family_name = self._classify_family(set(group_isas))
                    # Find an entry that matches this group's signatures.
                    # _sem_key is not checked: semantics are derived from the
                    # mnemonic, which is identical for all entries in this
                    # inst_map bucket, so matching fsig + osig is sufficient.
                    next_match = next(
                        (e for e in entries
                         if e[0] == group_isas[0]
                         and _field_signature(e[1]) == fsig
                         and _operand_signature(e[2]) == osig),
                        None
                    )
                    if next_match is None:
                        raise ValueError(
                            f"Failed to find matching entry for mnemonic '{mnemonic}' "
                            f"in ISA '{group_isas[0]}' with field_sig={fsig} and "
                            f"operand_sig={osig}. This indicates an internal analyzer bug."
                        )
                    _, enc0, inst0, sem0 = next_match
                    inst_key = (mnemonic, enc0.enc_name)
                    fam_dict = plan.family_shared.setdefault(family_name, {})
                    # Guard against silent overwrites: two sub-groups with
                    # different field layouts can collide on the same
                    # (mnemonic, enc_name) key if they share a family name
                    # and encoding name but diverge in layout across ISA
                    # generations.  This doesn't happen with current ISA
                    # specs, but if future specs introduce intra-family
                    # encoding layout changes, this assertion will fire.
                    # Fix by introducing finer sub-family groupings or by
                    # incorporating the field layout into the key.
                    assert inst_key not in fam_dict or fam_dict[inst_key].field_layout == fsig, (
                        f"Key collision in family_shared['{family_name}'] for "
                        f"{inst_key}: existing entry has field_layout="
                        f"{fam_dict[inst_key].field_layout}, new group has "
                        f"field_layout={fsig}. Two sub-groups with different "
                        f"field layouts map to the same (mnemonic, enc_name) "
                        f"key. Consider finer sub-family groupings or a "
                        f"layout-aware key."
                    )
                    fam_dict[inst_key] = SharedInstInfo(
                        mnemonic=mnemonic,
                        encoding_name=enc0.enc_name,
                        field_layout=fsig,
                        semantic_class=skey[0],
                        operation=skey[1],
                        data_type=skey[2],
                        isa_names=sorted(group_isas),
                    )
                else:
                    # ISA-exclusive — only 1 ISA has this particular
                    # encoding variant (field layout / semantics / operands),
                    # even if the mnemonic itself exists on other ISAs
                    # with a different structure.  Gets inline code.
                    plan.isa_exclusive.setdefault(group_isas[0], set()).add(mnemonic)

        return plan

    @staticmethod
    def _classify_family(isas: set[str]) -> str:
        """Return the family name for a set of ISAs.

        Picks the smallest entry from ``_FAMILIES`` that contains all
        input ISAs.  This avoids collisions where, e.g., {rdna1, rdna2}
        and {rdna3, rdna3_5} would otherwise both map to the coarse
        ``'rdna'`` bucket and overwrite each other in
        ``family_shared``.  Falls back to a joined ISA-name string for
        sets that cross families (e.g., {cdna1, rdna1}).
        """
        # See the note above _FAMILIES re commit 2316f85.  This loop
        # body is the same shape as Tony's; the difference is purely
        # the bucket-name strings in the table.
        best_name = '_'.join(sorted(isas))
        best_size = float('inf')
        for name, members in _FAMILIES:
            if isas <= members and len(members) < best_size:
                best_name = name
                best_size = len(members)
        return best_name
