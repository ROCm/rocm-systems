# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-ISA legalization table generator for DBT.

Classifies every instruction on a source ISA into one of five
legalization actions for a given target ISA:

- **IDENTITY** — encoding-identical; copy instruction word(s) verbatim.
- **SUBSTITUTE** — same encoding layout, different opcode; swap opcode field.
- **LOWER** — semantically equivalent but encoding differs; emit a
  target-native sequence via the matching ``lower_*()`` method in
  ``BinaryTranslator``.
- **EXPAND** — no target equivalent; emit a software emulation sequence.
- **ILLEGAL** — no legalization rule exists for this instruction on this
  target pair.

The classifier reuses ``CrossIsaAnalyzer`` for structural comparison and
adds a mnemonic rename map for cross-generation name changes plus
domain-specific override rules for waitcnt, barrier, flat memory,
MFMA, and AccVGPR instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from amdisa.cross_isa import CrossIsaAnalyzer, _field_signature, _operand_signature

if TYPE_CHECKING:
    from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec
    from amdisa.semantics import SemanticsSpec


# ---------------------------------------------------------------------------
# Legalization action types
# ---------------------------------------------------------------------------

class LoweringKind(Enum):
    WAITCNT = auto()
    BARRIER = auto()
    FLAT_MEMORY = auto()
    WAVE_EXEC = auto()
    GENERIC = auto()


class ExpansionKind(Enum):
    MFMA = auto()
    ACCVGPR = auto()
    WMMA = auto()
    CMP_REMOVED = auto()


@dataclass
class LegalizationAction:
    kind: str  # 'identity', 'substitute', 'lower', 'expand', 'illegal'
    target_opcode: int = 0
    lowering_kind: LoweringKind | None = None
    expansion_kind: ExpansionKind | None = None

    @staticmethod
    def identity() -> LegalizationAction:
        return LegalizationAction('identity')

    @staticmethod
    def substitute(target_opcode: int) -> LegalizationAction:
        return LegalizationAction('substitute', target_opcode=target_opcode)

    @staticmethod
    def lower(kind: LoweringKind = LoweringKind.GENERIC) -> LegalizationAction:
        return LegalizationAction('lower', lowering_kind=kind)

    @staticmethod
    def expand(kind: ExpansionKind) -> LegalizationAction:
        return LegalizationAction('expand', expansion_kind=kind)

    @staticmethod
    def illegal() -> LegalizationAction:
        # Retained for API compatibility but should never appear in
        # generated tables.  All unmatched instructions route through
        # _no_match_action() which returns LOWER or EXPAND.
        return LegalizationAction('illegal')


@dataclass
class LegalizationEntry:
    src_mnemonic: str
    src_encoding: str
    src_encoding_order: int
    src_encoding_bits: int
    src_opcode: int
    action: LegalizationAction


# ---------------------------------------------------------------------------
# Mnemonic rename map — cross-generation name changes that preserve semantics
# ---------------------------------------------------------------------------

def _build_rename_map() -> dict[str, str]:
    """Build canonical mnemonic rename map.

    Maps old-style mnemonics to their canonical (newest) form so that
    instructions with different names but identical semantics are
    recognized as equivalent.  The canonical form is the RDNA4/CDNA4
    (newest generation) mnemonic.
    """
    renames: dict[str, str] = {}

    def _add(old: str, new: str) -> None:
        if old != new:
            renames[old] = new

    # --- Memory size nomenclature: DWORD → B32 ---
    _dword_to_b = {
        'DWORD': 'B32', 'DWORDX2': 'B64', 'DWORDX3': 'B96',
        'DWORDX4': 'B128', 'DWORDX8': 'B256', 'DWORDX16': 'B512',
    }
    for prefix in ('S_LOAD_', 'S_STORE_', 'S_BUFFER_LOAD_', 'S_BUFFER_STORE_',
                    'S_ATOMIC_', 'S_BUFFER_ATOMIC_',
                    'BUFFER_LOAD_', 'BUFFER_STORE_',
                    'SCRATCH_LOAD_', 'SCRATCH_STORE_',
                    'DS_LOAD_', 'DS_STORE_',
                    'DS_READ_', 'DS_WRITE_',
                    'GLOBAL_LOAD_', 'GLOBAL_STORE_'):
        for old_suffix, new_suffix in _dword_to_b.items():
            _add(f'{prefix}{old_suffix}', f'{prefix}{new_suffix}')

    # FLAT → GLOBAL for load/store/atomic (FLAT is the legacy unified form)
    for op in ('LOAD_', 'STORE_', 'ATOMIC_'):
        for old_suffix, new_suffix in _dword_to_b.items():
            _add(f'FLAT_{op}{old_suffix}', f'GLOBAL_{op}{new_suffix}')

    # --- Sub-dword type nomenclature: UBYTE→U8, SBYTE→I8, etc. ---
    _sub_dword = {
        'UBYTE': 'U8', 'SBYTE': 'I8', 'USHORT': 'U16', 'SSHORT': 'I16',
        'BYTE': 'B8', 'SHORT': 'B16',
        'UBYTE_D16': 'U8_D16', 'SBYTE_D16': 'I8_D16',
        'USHORT_D16': 'U16_D16', 'SSHORT_D16': 'I16_D16',
        'UBYTE_D16_HI': 'U8_D16_HI', 'SBYTE_D16_HI': 'I8_D16_HI',
        'USHORT_D16_HI': 'U16_D16_HI', 'SSHORT_D16_HI': 'I16_D16_HI',
        'SHORT_D16': 'B16_D16', 'BYTE_D16_HI': 'B8_D16_HI',
        'SHORT_D16_HI': 'B16_D16_HI',
    }
    for prefix in ('FLAT_LOAD_', 'FLAT_STORE_',
                    'GLOBAL_LOAD_', 'GLOBAL_STORE_',
                    'SCRATCH_LOAD_', 'SCRATCH_STORE_',
                    'BUFFER_LOAD_', 'BUFFER_STORE_'):
        for old_suffix, new_suffix in _sub_dword.items():
            _add(f'{prefix}{old_suffix}', f'{prefix}{new_suffix}')
    # FLAT sub-dword → GLOBAL sub-dword
    for op in ('LOAD_', 'STORE_'):
        for old_suffix, new_suffix in _sub_dword.items():
            _add(f'FLAT_{op}{old_suffix}', f'GLOBAL_{op}{new_suffix}')

    # --- BUFFER_LOAD/STORE format renames ---
    for suffix in ('D16_X', 'D16_XY', 'D16_XYZ', 'D16_XYZW',
                    'D16_HI_X',
                    'X', 'XY', 'XYZ', 'XYZW'):
        _add(f'BUFFER_LOAD_FORMAT_{suffix}', f'BUFFER_LOAD_FORMAT_{suffix}')
        _add(f'BUFFER_STORE_FORMAT_{suffix}', f'BUFFER_STORE_FORMAT_{suffix}')
        _add(f'BUFFER_LOAD_FORMAT_D16_{suffix}', f'BUFFER_LOAD_D16_FORMAT_{suffix}')
        _add(f'BUFFER_STORE_FORMAT_D16_{suffix}', f'BUFFER_STORE_D16_FORMAT_{suffix}')
        _add(f'BUFFER_LOAD_FORMAT_D16_HI_{suffix}', f'BUFFER_LOAD_D16_HI_FORMAT_{suffix}')
        _add(f'BUFFER_STORE_FORMAT_D16_HI_{suffix}', f'BUFFER_STORE_D16_HI_FORMAT_{suffix}')

    # --- Atomic operation type suffixes ---
    # GFX9/GFX10 atomics have no type suffix; GFX12 adds _U32/_U64/_B32/_B64/_I32/_I64
    _atomic_type_suffixes = {
        'ADD': 'ADD_U32', 'ADD_X2': 'ADD_U64',
        'SUB': 'SUB_U32', 'SUB_X2': 'SUB_U64',
        'INC': 'INC_U32', 'INC_X2': 'INC_U64',
        'DEC': 'DEC_U32', 'DEC_X2': 'DEC_U64',
        'SMIN': 'MIN_I32', 'SMIN_X2': 'MIN_I64',
        'SMAX': 'MAX_I32', 'SMAX_X2': 'MAX_I64',
        'UMIN': 'MIN_U32', 'UMIN_X2': 'MIN_U64',
        'UMAX': 'MAX_U32', 'UMAX_X2': 'MAX_U64',
        'AND': 'AND_B32', 'AND_X2': 'AND_B64',
        'OR': 'OR_B32', 'OR_X2': 'OR_B64',
        'XOR': 'XOR_B32', 'XOR_X2': 'XOR_B64',
        'SWAP': 'SWAP_B32', 'SWAP_X2': 'SWAP_B64',
        'CMPSWAP': 'CMPSWAP_B32', 'CMPSWAP_X2': 'CMPSWAP_B64',
    }
    for prefix in ('BUFFER_ATOMIC_', 'FLAT_ATOMIC_', 'GLOBAL_ATOMIC_',
                    'SCRATCH_ATOMIC_', 'IMAGE_ATOMIC_', 'DS_'):
        for old_suffix, new_suffix in _atomic_type_suffixes.items():
            _add(f'{prefix}{old_suffix}', f'{prefix}{new_suffix}')
    # FLAT_ATOMIC → GLOBAL_ATOMIC
    for old_suffix, new_suffix in _atomic_type_suffixes.items():
        _add(f'FLAT_ATOMIC_{old_suffix}', f'GLOBAL_ATOMIC_{new_suffix}')

    # DS atomics also have return variants
    for old_suffix, new_suffix in _atomic_type_suffixes.items():
        _add(f'DS_{old_suffix}_RTN', f'DS_{new_suffix}_RTN')
        _add(f'DS_RTN_{old_suffix}', f'DS_{new_suffix}_RTN')

    # --- DS compare-store rename: CMPST → CMPSTORE ---
    for suffix in ('B32', 'B64', 'F32', 'F64',
                    'RTN_B32', 'RTN_B64', 'RTN_F32', 'RTN_F64'):
        _add(f'DS_CMPST_{suffix}', f'DS_CMPSTORE_{suffix}')

    # --- DS read/write → load/store ---
    for suffix in ('B32', 'B64', 'B128', 'B96',
                    'U8', 'I8', 'U16', 'I16', 'U8_D16', 'U8_D16_HI',
                    'I8_D16', 'I8_D16_HI', 'U16_D16', 'U16_D16_HI',
                    'I16_D16', 'I16_D16_HI', 'ADDTID_B32'):
        _add(f'DS_READ_{suffix}', f'DS_LOAD_{suffix}')
        _add(f'DS_WRITE_{suffix}', f'DS_STORE_{suffix}')
    # 2-address variants
    for suffix in ('B32', 'B64'):
        _add(f'DS_READ2_B32', f'DS_LOAD_2ADDR_B32')
        _add(f'DS_READ2_B64', f'DS_LOAD_2ADDR_B64')
        _add(f'DS_READ2ST64_B32', f'DS_LOAD_2ADDR_STRIDE64_B32')
        _add(f'DS_READ2ST64_B64', f'DS_LOAD_2ADDR_STRIDE64_B64')
        _add(f'DS_WRITE2_B32', f'DS_STORE_2ADDR_B32')
        _add(f'DS_WRITE2_B64', f'DS_STORE_2ADDR_B64')
        _add(f'DS_WRITE2ST64_B32', f'DS_STORE_2ADDR_STRIDE64_B32')
        _add(f'DS_WRITE2ST64_B64', f'DS_STORE_2ADDR_STRIDE64_B64')

    # --- Float min/max → min_num/max_num (IEEE 754-2019 quiet NaN) ---
    for size in ('F16', 'F32', 'F64'):
        _add(f'V_MIN_{size}', f'V_MIN_NUM_{size}')
        _add(f'V_MAX_{size}', f'V_MAX_NUM_{size}')
        _add(f'DS_MIN_{size}', f'DS_MIN_NUM_{size}')
        _add(f'DS_MAX_{size}', f'DS_MAX_NUM_{size}')
        _add(f'DS_MIN_RTN_{size}', f'DS_MIN_NUM_RTN_{size}')
        _add(f'DS_MAX_RTN_{size}', f'DS_MAX_NUM_RTN_{size}')
    # Packed
    _add('V_PK_MIN_F16', 'V_PK_MIN_NUM_F16')
    _add('V_PK_MAX_F16', 'V_PK_MAX_NUM_F16')
    _add('V_PK_MIN_F32', 'V_PK_MIN_NUM_F32')
    _add('V_PK_MAX_F32', 'V_PK_MAX_NUM_F32')
    # Buffer/global/flat atomic min/max float
    for prefix in ('BUFFER_ATOMIC_', 'FLAT_ATOMIC_', 'GLOBAL_ATOMIC_'):
        _add(f'{prefix}FMIN', f'{prefix}MIN_NUM_F32')
        _add(f'{prefix}FMAX', f'{prefix}MAX_NUM_F32')
        _add(f'{prefix}MIN_F32', f'{prefix}MIN_NUM_F32')
        _add(f'{prefix}MAX_F32', f'{prefix}MAX_NUM_F32')

    # --- Scalar bitwise renames (GFX9→GFX11): S_ANDN2→S_AND_NOT1, etc. ---
    for w in ('32', '64'):
        _add(f'S_ANDN2_B{w}', f'S_AND_NOT1_B{w}')
        _add(f'S_ORN2_B{w}', f'S_OR_NOT1_B{w}')
        _add(f'S_NAND_B{w}', f'S_NAND_B{w}')
        _add(f'S_NOR_B{w}', f'S_NOR_B{w}')
        _add(f'S_XNOR_B{w}', f'S_XNOR_B{w}')
    _add('S_ANDN1_SAVEEXEC_B64', 'S_AND_NOT0_SAVEEXEC_B64')
    _add('S_ANDN2_SAVEEXEC_B64', 'S_AND_NOT1_SAVEEXEC_B64')
    _add('S_ORN1_SAVEEXEC_B64', 'S_OR_NOT0_SAVEEXEC_B64')
    _add('S_ORN2_SAVEEXEC_B64', 'S_OR_NOT1_SAVEEXEC_B64')

    # --- EXP → EXPORT ---
    _add('EXP', 'EXPORT')

    # --- Conditional subtract: CSUB → COND_SUB ---
    for prefix in ('BUFFER_ATOMIC_', 'FLAT_ATOMIC_', 'GLOBAL_ATOMIC_',
                    'SCRATCH_ATOMIC_', 'DS_'):
        _add(f'{prefix}CSUB_U32', f'{prefix}COND_SUB_U32')
        _add(f'{prefix}CSUB_RTN_U32', f'{prefix}COND_SUB_RTN_U32')

    # --- BUFFER_WBL2 / BUFFER_INV / BUFFER_GL0_INV etc. ---
    _add('BUFFER_WBL2', 'BUFFER_GL1_INV')
    _add('BUFFER_INV', 'BUFFER_INV')

    # --- V_INTERP renames (GFX9→GFX11) ---
    _add('V_INTERP_P1_F32', 'V_INTERP_P10_F32_INREG')
    _add('V_INTERP_P2_F32', 'V_INTERP_P2_F32_INREG')
    _add('V_INTERP_MOV_F32', 'V_INTERP_P10_F32_INREG')

    # --- V_CMP / V_CMPX: F (false) and T/TRU (true) are removed in GFX11+ ---
    # These always-false/always-true comparisons are trivially emulated with
    # s_mov_b{32,64} of 0 or exec. Map them to the same canonical so the
    # classifier sees them as needing LOWER.  We use the mnemonic of the
    # equivalent s_mov instruction as the canonical form.
    # (Handled in domain_rules rather than rename map — see _apply_domain_rules)

    # --- S_CMPK → same (these exist on both, just encoding differences) ---
    # These are handled by encoding comparison, not rename.

    # --- V_MAD → V_FMA (GFX9→GFX11: V_MAD removed, V_FMA is the replacement)
    # These are NOT renames — MAD is fused multiply-add without IEEE
    # intermediate rounding, FMA is with.  They must be LOWER, not rename.
    # Handled in domain_rules.

    return renames


_RENAME_MAP: dict[str, str] | None = None


def _rename_map() -> dict[str, str]:
    global _RENAME_MAP
    if _RENAME_MAP is None:
        _RENAME_MAP = _build_rename_map()
    return _RENAME_MAP


def canonical_mnemonic(name: str) -> str:
    """Normalize an instruction mnemonic via the rename map."""
    return _rename_map().get(name, name)


# ---------------------------------------------------------------------------
# Minimal e-graph (union-find over mnemonic equivalence classes)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressing union-find for transitive mnemonic equivalence."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
        return ra


# ---------------------------------------------------------------------------
# ISA capability queries (used by domain-specific rules)
# ---------------------------------------------------------------------------

class WaitcntModel(Enum):
    GFX9_CLASSIC = auto()
    GFX10_PLUS_VSCNT = auto()
    GFX11_SPLIT = auto()
    GFX12_SPLIT = auto()


_WAITCNT_MODELS: dict[str, WaitcntModel] = {
    'cdna1': WaitcntModel.GFX9_CLASSIC,
    'cdna2': WaitcntModel.GFX9_CLASSIC,
    'cdna3': WaitcntModel.GFX9_CLASSIC,
    'cdna4': WaitcntModel.GFX9_CLASSIC,
    'rdna1': WaitcntModel.GFX10_PLUS_VSCNT,
    'rdna2': WaitcntModel.GFX10_PLUS_VSCNT,
    'rdna3': WaitcntModel.GFX11_SPLIT,
    'rdna3_5': WaitcntModel.GFX11_SPLIT,
    'rdna4': WaitcntModel.GFX12_SPLIT,
}

_SPLIT_BARRIER_ISAS = frozenset({'cdna4', 'rdna3', 'rdna3_5', 'rdna4'})
_SEPARATE_GLOBAL_ISAS = frozenset({
    'cdna3', 'cdna4', 'rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4',
})
_MFMA_ISAS = frozenset({'cdna1', 'cdna2', 'cdna3', 'cdna4'})
_ACCVGPR_ISAS = frozenset({'cdna2', 'cdna3', 'cdna4'})


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

@dataclass
class _InstRecord:
    """Internal record for one instruction on one ISA."""
    mnemonic: str
    canonical: str
    isa_name: str
    enc_name: str
    enc_order: int
    enc_bits: int
    opcode: int
    field_sig: tuple[tuple[str, int], ...]
    opnd_sig: tuple
    eclass_id: int = -1


class LegalizationGenerator:
    """Generate cross-ISA legalization tables for all supported pairs.

    Usage::

        specs = [('cdna3', cdna3_spec, cdna3_sem), ('cdna4', ...)]
        gen = LegalizationGenerator(specs)
        for src, dst, entries in gen.generate_all():
            ...
    """

    def __init__(
        self,
        specs: list[tuple[str, IsaSpec, SemanticsSpec | None]],
    ) -> None:
        self._specs = specs
        self._records: dict[str, list[_InstRecord]] = {}
        self._uf = _UnionFind()
        self._next_id = 0
        self._canon_to_ids: dict[str, list[int]] = {}
        self._id_to_record: dict[int, _InstRecord] = {}

        self._build_records()
        self._discover_equivalences()

    @staticmethod
    def _dt_index(enc, spec) -> int:
        if enc.primary_dt_ptrs is None:
            parent_name = spec.profile.derive_parent_enc_name(enc.enc_name)
            if parent_name in spec.encoding_map:
                return LegalizationGenerator._dt_index(
                    spec.encoding_map[parent_name], spec)
            return 0
        for ptr in enc.primary_dt_ptrs:
            if ptr != -1:
                return ptr
        return 0

    @staticmethod
    def _enc_field_value(enc, spec) -> int:
        dt_idx = LegalizationGenerator._dt_index(enc, spec)
        max_enc_bits = spec.profile.max_enc_bits
        dont_care = max_enc_bits - enc.enc_field_bit_cnt
        return dt_idx >> dont_care

    def _alloc_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        self._uf.make_set(eid)
        return eid

    def _build_records(self) -> None:
        for isa_name, spec, _ in self._specs:
            records: list[_InstRecord] = []
            for enc in spec.inst_encodings:
                if not enc.insts:
                    continue
                fsig = _field_signature(enc)
                for inst in enc.insts:
                    opsig = _operand_signature(inst)
                    eid = self._alloc_id()
                    rec = _InstRecord(
                        mnemonic=inst.name,
                        canonical=canonical_mnemonic(inst.name),
                        isa_name=isa_name,
                        enc_name=enc.enc_name,
                        enc_order=self._dt_index(enc, spec),
                        enc_bits=enc.enc_field_bit_cnt,
                        opcode=inst.opcode,
                        field_sig=fsig,
                        opnd_sig=opsig,
                        eclass_id=eid,
                    )
                    self._id_to_record[eid] = rec
                    self._canon_to_ids.setdefault(rec.canonical, []).append(eid)
                    records.append(rec)
            self._records[isa_name] = records

    def _discover_equivalences(self) -> None:
        """Merge e-classes for instructions with the same canonical mnemonic."""
        for canon, ids in self._canon_to_ids.items():
            if len(ids) < 2:
                continue
            first = ids[0]
            for other in ids[1:]:
                self._uf.union(first, other)

    def _equivalent(self, a: _InstRecord, b: _InstRecord) -> bool:
        return self._uf.find(a.eclass_id) == self._uf.find(b.eclass_id)

    def classify(
        self, src_isa: str, dst_isa: str,
    ) -> list[LegalizationEntry]:
        """Classify every source instruction for a (src, dst) pair."""
        src_records = self._records.get(src_isa, [])
        dst_records = self._records.get(dst_isa, [])

        dst_by_class: dict[int, list[_InstRecord]] = {}
        for rec in dst_records:
            root = self._uf.find(rec.eclass_id)
            dst_by_class.setdefault(root, []).append(rec)

        entries: list[LegalizationEntry] = []
        for src_rec in src_records:
            src_root = self._uf.find(src_rec.eclass_id)
            candidates = dst_by_class.get(src_root, [])

            if not candidates:
                action = self._no_match_action(src_rec)
            else:
                action = self._best_match_action(src_rec, candidates)

            entries.append(LegalizationEntry(
                src_mnemonic=src_rec.mnemonic,
                src_encoding=src_rec.enc_name,
                src_encoding_order=src_rec.enc_order,
                src_encoding_bits=src_rec.enc_bits,
                src_opcode=src_rec.opcode,
                action=action,
            ))

        _apply_domain_rules(entries, src_isa, dst_isa)
        return entries

    @staticmethod
    def _best_match_action(
        src: _InstRecord, candidates: list[_InstRecord],
    ) -> LegalizationAction:
        for dst in candidates:
            if src.field_sig == dst.field_sig and src.opnd_sig == dst.opnd_sig:
                if src.opcode == dst.opcode:
                    return LegalizationAction('identity', target_opcode=dst.opcode)
                return LegalizationAction('substitute', target_opcode=dst.opcode)
        # Encoding differs — LOWER.  Prefer a candidate in the same encoding
        # format; fall back to any candidate; if none has a valid opcode, Expand.
        same_enc = [c for c in candidates if c.enc_name == src.enc_name]
        best = same_enc[0] if same_enc else candidates[0] if candidates else None
        if best:
            if best.opcode == 0 and src.opcode != 0:
                return LegalizationAction.expand(ExpansionKind.CMP_REMOVED)
            return LegalizationAction('lower', target_opcode=best.opcode)
        return LegalizationAction.expand(ExpansionKind.CMP_REMOVED)

    @staticmethod
    def _no_match_action(src: _InstRecord) -> LegalizationAction:
        """Classify an instruction with no equivalent on the target.

        Never returns ILLEGAL — every instruction can be lowered or
        expanded.  The BinaryTranslator dispatches to the appropriate
        handler based on the action kind and lowering/expansion tag.
        """
        name = src.mnemonic
        if name.startswith('V_MFMA_') or name.startswith('V_SMFMAC_'):
            return LegalizationAction.expand(ExpansionKind.MFMA)
        if name.startswith('V_ACCVGPR_'):
            return LegalizationAction.expand(ExpansionKind.ACCVGPR)
        if name.startswith('V_WMMA_'):
            return LegalizationAction.expand(ExpansionKind.WMMA)
        if name.startswith('DS_GWS_'):
            return LegalizationAction.lower(LoweringKind.GENERIC)
        if name.startswith('V_MAD_'):
            return LegalizationAction.lower(LoweringKind.GENERIC)
        if name.startswith('V_INTERP_'):
            return LegalizationAction.lower(LoweringKind.GENERIC)
        if name.startswith('IMAGE_'):
            return LegalizationAction.lower(LoweringKind.GENERIC)
        # All remaining unmatched instructions are EXPAND — no target
        # equivalent exists. BinaryTranslator emits s_nop placeholder.
        return LegalizationAction.expand(ExpansionKind.CMP_REMOVED)

    def generate_all(
        self,
        pairs: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str, list[LegalizationEntry]]]:
        """Generate legalization tables for all (or specified) ISA pairs."""
        if pairs is None:
            pairs = _default_pairs(self._specs)
        results = []
        for src, dst in pairs:
            entries = self.classify(src, dst)
            results.append((src, dst, entries))
        return results

    def summary(
        self, entries: list[LegalizationEntry],
    ) -> dict[str, int]:
        """Count entries by action kind."""
        counts: dict[str, int] = {
            'identity': 0, 'substitute': 0, 'lower': 0,
            'expand': 0, 'illegal': 0,
        }
        for e in entries:
            counts[e.action.kind] += 1
        return counts


# ---------------------------------------------------------------------------
# Domain-specific override rules
# ---------------------------------------------------------------------------

def _apply_domain_rules(
    entries: list[LegalizationEntry],
    src_isa: str, dst_isa: str,
) -> None:
    src_wc = _WAITCNT_MODELS.get(src_isa)
    dst_wc = _WAITCNT_MODELS.get(dst_isa)
    dst_has_split_barrier = dst_isa in _SPLIT_BARRIER_ISAS
    src_has_split_barrier = src_isa in _SPLIT_BARRIER_ISAS
    dst_has_mfma = dst_isa in _MFMA_ISAS
    dst_has_accvgpr = dst_isa in _ACCVGPR_ISAS

    for entry in entries:
        name = entry.src_mnemonic

        if name in ('S_WAITCNT', 'S_WAITCNT_VSCNT',
                     'S_WAITCNT_VMCNT', 'S_WAITCNT_LGKMCNT',
                     'S_WAITCNT_EXPCNT'):
            if src_wc != dst_wc:
                entry.action = LegalizationAction.lower(LoweringKind.WAITCNT)

        if name.startswith('S_WAIT_') and name not in (
            'S_WAITCNT', 'S_WAITCNT_VSCNT', 'S_WAITCNT_VMCNT',
            'S_WAITCNT_LGKMCNT', 'S_WAITCNT_EXPCNT',
        ):
            if src_wc != dst_wc:
                entry.action = LegalizationAction.lower(LoweringKind.WAITCNT)

        if name == 'S_BARRIER':
            if not src_has_split_barrier and dst_has_split_barrier:
                entry.action = LegalizationAction.lower(LoweringKind.BARRIER)
        if name in ('S_BARRIER_SIGNAL', 'S_BARRIER_WAIT'):
            if src_has_split_barrier and not dst_has_split_barrier:
                entry.action = LegalizationAction.lower(LoweringKind.BARRIER)

        if name.startswith('FLAT_LOAD_') or name.startswith('FLAT_STORE_'):
            if dst_isa in _SEPARATE_GLOBAL_ISAS and src_isa not in _SEPARATE_GLOBAL_ISAS:
                entry.action = LegalizationAction.lower(LoweringKind.FLAT_MEMORY)

        if name.startswith('V_MFMA_') and not dst_has_mfma:
            entry.action = LegalizationAction.expand(ExpansionKind.MFMA)

        if name.startswith('V_ACCVGPR_') and not dst_has_accvgpr:
            entry.action = LegalizationAction.expand(ExpansionKind.ACCVGPR)

        # V_CMP_F_* (always-false) and V_CMP_TRU_* / V_CMPX_TRU_* (always-true)
        # are removed on GFX11+ (RDNA3/3.5/4). They have no target equivalent
        # and must be expanded to s_mov_b32 vcc, 0 / s_mov_b64 vcc, exec.
        if name.startswith(('V_CMP_F_', 'V_CMPX_F_',
                            'V_CMP_TRU_', 'V_CMPX_TRU_',
                            'V_CMP_T_', 'V_CMPX_T_')):
            if entry.action.target_opcode == 0:
                entry.action = LegalizationAction.expand(ExpansionKind.CMP_REMOVED)


# ---------------------------------------------------------------------------
# Default translation pairs
# ---------------------------------------------------------------------------

def _default_pairs(
    specs: list[tuple[str, IsaSpec, SemanticsSpec | None]],
) -> list[tuple[str, str]]:
    """Return the set of supported translation pairs from the loaded ISAs."""
    names = {name for name, _, _ in specs}
    all_pairs = [
        # Intra-CDNA upgrades
        ('cdna1', 'cdna2'), ('cdna1', 'cdna3'), ('cdna1', 'cdna4'),
        ('cdna2', 'cdna3'), ('cdna2', 'cdna4'), ('cdna3', 'cdna4'),
        # CDNA → RDNA
        ('cdna1', 'rdna1'), ('cdna1', 'rdna2'), ('cdna1', 'rdna3'),
        ('cdna1', 'rdna4'), ('cdna2', 'rdna3'), ('cdna2', 'rdna4'),
        ('cdna3', 'rdna3'), ('cdna3', 'rdna4'), ('cdna4', 'rdna3'),
        ('cdna4', 'rdna4'),
        # Intra-RDNA upgrades
        ('rdna1', 'rdna2'), ('rdna1', 'rdna3'), ('rdna1', 'rdna4'),
        ('rdna2', 'rdna3'), ('rdna2', 'rdna4'), ('rdna3', 'rdna4'),
        ('rdna3_5', 'rdna4'),
        # RDNA → CDNA
        ('rdna1', 'cdna3'), ('rdna1', 'cdna4'), ('rdna3', 'cdna4'),
        ('rdna4', 'cdna4'),
    ]
    return [(s, d) for s, d in all_pairs if s in names and d in names]
