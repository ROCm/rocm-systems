# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ISA-specific encoding rules and constants.

Separates ISA-specific knowledge (naming conventions, encoding behavior,
hardware constants, disassembly formatting) from the generic data model,
parser, and code generator so that supporting a new ISA version requires
only a new profile, not changes to the core infrastructure.

Each concrete profile captures the encoding structure differences, mnemonic
formatting rules, and instruction modifier conventions for one or more
ISA generations in that family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

_FLOAT_NAME_MAP: dict[float, str] = {
    -0.5: 'NEG_HALF',
    -1.0: 'NEG_ONE',
    -2.0: 'NEG_TWO',
    -4.0: 'NEG_FOUR',
    0.5: 'HALF',
    1.0: 'ONE',
    2.0: 'TWO',
    4.0: 'FOUR',
    0.15915494: 'ONE_OVER_TWO_PI',
}


class MemoryCoherencyModel(Enum):
    """Identifies the memory coherency encoding model for an ISA family.

    The five models are NOT backward-compatible: field positions, field
    names, and semantic meanings all change across generations. execute()
    for memory instructions must use ISA-specific logic keyed on this enum.

    GFX9_GLC covers CDNA1/2 (GFX908/GFX90A). GFX940_SC0_SC1_NT covers
    CDNA3/4. GFX10/11/12 cover RDNA generations.
    """

    GFX9_GLC        = auto()  # CDNA1, CDNA2 — GLC bit only, all memory
    GFX940_SC0_SC1_NT = auto()  # CDNA3, CDNA4 — SC0/SC1+NT vector; GLC scalar
    GFX10_GLC_DLC_SLC = auto()  # RDNA1, RDNA2 — GLC + DLC + SLC
    GFX11_SC0_SC1_TH  = auto()  # RDNA3, RDNA3.5 — SC0+SC1 scope + TH hint
    GFX12_SCOPE_TH    = auto()  # RDNA4 — 2-bit SCOPE + TH hint


@dataclass
class EncodingModifier:
    """A disassembly modifier to append to an encoding's mnemonic output.

    Describes a single modifier field from an encoding's microcode format
    that should be included in the disassembled instruction output.

    Attributes:
        field: Microcode field name (e.g., ``glc``, ``sc0``, ``offen``).
            When ``preamble`` is set, this is a local variable name
            defined by the preamble rather than an ``inst->`` member.
        display: Display string appended to the mnemonic (e.g., ``" glc"``).
        is_offset: True if this field is a numeric offset that should be
            printed as ``" offset:<value>"`` instead of a boolean flag.
        condition: Optional C++ expression (with ``inst->`` prefixes)
            that must evaluate to true for this modifier to appear.
            Example: ``"inst->soffset_en && inst->imm"``.
        preamble: Optional C++ statement emitted before the modifier
            check. When set, ``field`` refers to the local variable
            defined by the preamble (not ``inst->field``). Used for
            computed values like the FLAT offset that combines multiple
            bitfields.
    """

    field: str
    display: str = ''
    is_offset: bool = False
    condition: str = ''
    preamble: str = ''

    def __post_init__(self) -> None:
        if not self.display:
            self.display = f' {self.field}'


@dataclass
class MnemonicRule:
    """Rules for transforming an encoding class's mnemonic at construction.

    Attributes:
        suffix: String appended to the mnemonic (e.g., ``"_e32"``).
        use_flat_mnemonic: True if the ``flat_mnemonic()`` helper should
            be used to rewrite the mnemonic prefix based on the ``seg``
            field (e.g., ``flat_load`` to ``global_load``).
    """

    suffix: str = ''
    use_flat_mnemonic: bool = False


class IsaProfile(ABC):
    """Defines ISA-specific encoding rules and constants.

    Subclasses implement the rules for a specific ISA family. The parser,
    semantic derivation engine, and C++ code generator all consult the
    profile to handle ISA-specific naming conventions, encoding behavior,
    mnemonic formatting, and instruction modifier display.

    The profile interface is split into several groups:

    **Parsing rules** control how the XML spec is interpreted:
    ``supported_versions``, ``max_enc_bits``, ``max_enc_order``,
    ``skip_encodings``, ``is_alt_encoding``, ``derive_parent_enc_name``,
    ``is_implied_literal_encoding``, ``skip_inst_encoding``.

    **Semantic overrides** let a profile supply execution semantics for
    instructions that the generic mnemonic-based derivation cannot handle:
    ``semantic_overrides``.

    **Code generation rules** control how generated C++ encoding classes
    format mnemonics and modifiers: ``mnemonic_rule``,
    ``encoding_modifiers``.

    Attributes:
        supported_versions: Schema versions this profile supports.
        max_enc_bits: Maximum bits used for the primary decode table index.
        max_enc_order: Maximum order value for primary encoding types.
        flt_name_map: Maps floating-point literal values to C++ enum names.
    """

    @property
    @abstractmethod
    def supported_versions(self) -> list[str]:
        """Schema versions this profile supports."""
        ...

    @property
    @abstractmethod
    def max_enc_bits(self) -> int:
        """Maximum bits used for the primary decode table index."""
        ...

    @property
    @abstractmethod
    def max_enc_order(self) -> int:
        """Maximum order value for primary encoding types."""
        ...

    @property
    @abstractmethod
    def flt_name_map(self) -> dict[float, str]:
        """Maps floating-point literal values to C++ enum-friendly names."""
        ...

    @property
    def skip_encodings(self) -> frozenset[str]:
        """Encoding names to skip entirely during parsing.

        Used for encodings with non-standard structures that the parser
        cannot handle (e.g., dual-opcode formats like VOPDXY, or
        encodings with incomplete XML data like VOP3PX2).
        """
        return frozenset()

    @property
    def inst_size_overrides(self) -> dict[str, int]:
        """Per-instruction size overrides in bytes.

        Used for instructions whose encoding size differs from their
        parent encoding (e.g., VOP3PX2 instructions are 128-bit but
        decoded under the 64-bit VOP3P_MFMA encoding).
        """
        return {}

    @property
    def semantic_overrides(self) -> dict[str, tuple[str, ...]]:
        """Per-instruction semantic overrides for this ISA.

        Maps instruction mnemonic to a tuple of
        ``(semantic_class, operation, data_type)`` where ``operation``
        and ``data_type`` may be empty strings to indicate None.

        These take priority over the generic mnemonic-based derivation
        in :func:`~amdisa.semantics.derive_semantics`. Use this for
        instructions whose semantics cannot be inferred from the mnemonic
        alone (e.g., ISA-specific intrinsics, instructions whose naming
        conventions diverge from the common AMD pattern).
        """
        return {}

    @property
    def cmpx_writes_vcc(self) -> bool:
        """True if V_CMPX instructions write both EXEC and VCC.

        On CDNA (GFX9-based), V_CMPX writes both EXEC and VCC. On RDNA,
        V_CMPX writes only EXEC.
        """
        return False

    @property
    def waitcnt_decode(self) -> str:
        """Return C++ code block that decodes a WAITCNT immediate into
        vmcnt, expcnt, and lgkmcnt local variables.

        The field layout varies by ISA family. CDNA uses a split vmcnt
        field: bits [3:0] | (bits [15:14] << 4), expcnt at [6:4],
        lgkmcnt at [12:8]. RDNA uses the same layout but with a 6-bit
        lgkmcnt at [13:8]. Subclasses may override for different layouts.
        """
        return (
            'uint32_t vmcnt = (encoding_value_ & 0xF) | '
            '(((encoding_value_ >> 14) & 0x3) << 4);\n'
            f'uint32_t expcnt = (encoding_value_ >> 4) & 0x7;\n'
            f'uint32_t lgkmcnt = (encoding_value_ >> 8) & {self.waitcnt_lgkmcnt_mask};\n'
        )

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        """Hex mask for the lgkmcnt field in the S_WAITCNT immediate.

        CDNA (GFX9 family): 4-bit field at bits [11:8] → mask 0x0F.
        RDNA1/2 (GFX10): 6-bit field at bits [13:8] → mask 0x3F.
        RDNA3/3.5 (GFX11): layout changed; use Isa::WAITCNT_LGKMCNT_MASK.
        RDNA4 (GFX12): S_WAITCNT removed; this property is unused.
        """
        return '0x0F'

    def has_src_modifiers(self, enc_name: str) -> bool:
        """True if the encoding format has source input modifiers.

        VOP3 encodings have per-source ``neg`` and ``abs`` fields plus
        output ``omod`` and ``clamp`` fields. VOP3P has ``neg`` but not
        ``abs``. VOP1/VOP2/VOPC do not have modifiers in the base (e32)
        encoding.

        Args:
            enc_name: Encoding format name (e.g., ``ENC_VOP3``).

        Returns:
            True if the encoding has neg/abs/omod/clamp modifier fields.
        """
        return False

    def has_abs_modifier(self, enc_name: str) -> bool:
        """True if the encoding format has ``abs`` source modifier fields.

        Most VOP3 variants have ``abs``, but ``VOP3_SDST_ENC`` has
        ``neg``/``omod``/``clamp`` without ``abs``.

        Args:
            enc_name: Encoding format name (e.g., ``ENC_VOP3``).

        Returns:
            True if the encoding has per-source ``abs`` modifier fields.
        """
        return False

    def mnemonic_rule(self, enc_name: str) -> MnemonicRule:
        """Return the mnemonic formatting rule for an encoding format.

        Controls how the generated C++ encoding class constructor
        transforms the instruction mnemonic string.

        Args:
            enc_name: Uppercase encoding name (e.g., ``ENC_VOP1``).

        Returns:
            A ``MnemonicRule`` with suffix and/or flat-mnemonic flag.
            The default returns an empty rule (no transformation).
        """
        return MnemonicRule()

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        """Return the disassembly modifier fields for an encoding format.

        The code generator uses these to emit ``if (inst->field)``
        modifier-append lines in the generated encoding class constructor.

        Args:
            enc_name: Uppercase encoding name (e.g., ``ENC_SMEM``).

        Returns:
            List of ``EncodingModifier`` descriptors. The default is an
            empty list (no modifiers).
        """
        return []

    def field_renames(self, enc_name: str) -> dict[str, str]:
        """Return field name remaps for an encoding's microcode fields.

        Used to correct XML field names that differ from the ISA PDF.
        The returned dict maps XML field names (lowercased) to the
        canonical spec name. Applied during bitmap parsing so that the
        generated C++ struct uses the correct field names.

        Args:
            enc_name: Uppercase encoding name (e.g., ``ENC_FLAT``).

        Returns:
            Dict of ``{xml_name: canonical_name}``. Default is empty
            (no renames).
        """
        return {}

    @abstractmethod
    def is_alt_encoding(self, enc_name: str) -> bool:
        """True if the encoding name indicates an alternate encoding.

        Alternate encodings share their parent's primary decode table
        entries and represent a variant of the parent format (e.g., with
        an implied literal operand, a different memory segment, or a
        specialized opcode subset).
        """
        ...

    @abstractmethod
    def derive_parent_enc_name(self, enc_name: str) -> str:
        """Derive the parent encoding name from an alternate encoding name.

        Only called when ``is_alt_encoding()`` returns True.
        """
        ...

    @abstractmethod
    def is_implied_literal_encoding(
        self,
        enc_name: str,
        enc_conds: list[tuple[str, str]],
        bit_cnt: int,
        parent_bit_cnt: int,
    ) -> bool:
        """True if this alternate encoding has an implied literal DWORD.

        Implied-literal encodings consume an extra DWORD (the literal
        constant) beyond the base encoding size. Detection uses a
        combination of:

        * Encoding condition names (e.g., ``has_lit``).
        * Encoding name and bit count relative to the parent (e.g., a
          64-bit alternate of a 32-bit parent named ``*_INST_LITERAL``).
        """
        ...

    @abstractmethod
    def skip_inst_encoding(self, enc_name: str, enc_cond: str) -> bool:
        """True if instructions under this encoding/condition should be skipped.

        The base decoder only handles instructions under the ``default``
        encoding condition. Modifier variants (DPP, SDWA) and
        segment-specific FLAT encodings are skipped because they share
        the parent's decode table and are distinguished at runtime.
        """
        ...

_VOP_E32_RULE = MnemonicRule(suffix='_e32')

# GFX940 (CDNA3/4): SC0+SC1+NT coherency model.
_SMEM_MODIFIERS = [
    EncodingModifier(
        'offset', is_offset=True,
        condition='inst->soffset_en && inst->imm',
    ),
    EncodingModifier('glc'),
    EncodingModifier('nv'),
]

_MUBUF_MODIFIERS = [
    EncodingModifier('offen'),
    EncodingModifier('idxen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('sc0'),
    EncodingModifier('sc1'),
    EncodingModifier('nt'),
    EncodingModifier('lds'),
]

_MTBUF_MODIFIERS = [
    EncodingModifier('offen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('sc0'),
    EncodingModifier('sc1'),
    EncodingModifier('nt'),
]

_FLAT_MODIFIERS = [
    EncodingModifier(
        'flat_offset', is_offset=True,
        preamble=(
            'int flat_offset = (inst->seg != 0) ?'
            ' (inst->offset | (inst->pad_12 << 12)) : inst->offset;'
        ),
    ),
    EncodingModifier('sc0'),
    EncodingModifier('sc1'),
    EncodingModifier('nt'),
]

# GFX9 (CDNA1/2): GLC+SLC coherency model; SMEM unchanged (soffset_en/imm present).
_MUBUF_MODIFIERS_GLC = [
    EncodingModifier('offen'),
    EncodingModifier('idxen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('glc'),
    EncodingModifier('slc'),
    EncodingModifier('lds'),
]

_MTBUF_MODIFIERS_GLC = [
    EncodingModifier('offen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('glc'),
    EncodingModifier('slc'),
]

_FLAT_MODIFIERS_GLC = [
    EncodingModifier(
        'flat_offset', is_offset=True,
        preamble=(
            'int flat_offset = (inst->seg != 0) ?'
            ' (inst->offset | (inst->pad_12 << 12)) : inst->offset;'
        ),
    ),
    EncodingModifier('glc'),
    EncodingModifier('slc'),
]

# GFX10/GFX11 (RDNA1/2/3/3.5): GLC+DLC+SLC; SMEM has no soffset_en/imm.
_SMEM_MODIFIERS_GLC_DLC = [
    EncodingModifier('glc'),
    EncodingModifier('dlc'),
]

_MUBUF_MODIFIERS_GLC_DLC = [
    EncodingModifier('offen'),
    EncodingModifier('idxen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('glc'),
    EncodingModifier('dlc'),
    EncodingModifier('slc'),
    EncodingModifier('lds'),
]

_MTBUF_MODIFIERS_GLC_DLC = [
    EncodingModifier('offen'),
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('glc'),
    EncodingModifier('dlc'),
    EncodingModifier('slc'),
]

_FLAT_MODIFIERS_GLC_DLC = [
    EncodingModifier('offset', is_offset=True),
    EncodingModifier('glc'),
    EncodingModifier('dlc'),
    EncodingModifier('slc'),
]

# GFX12 (RDNA4): SCOPE+TH model; flag modifier is NV only.
_SMEM_MODIFIERS_RDNA4 = [
    EncodingModifier('nv'),
]

_VBUFFER_MODIFIERS_RDNA4 = [
    EncodingModifier('offen'),
    EncodingModifier('idxen'),
    EncodingModifier('ioffset', is_offset=True),
    EncodingModifier('nv'),
]

_VFLAT_MODIFIERS_RDNA4 = [
    EncodingModifier('nv'),
]


class _AmdgpuProfileBase(IsaProfile):
    """Shared behaviour for all AMDGPU ISA profiles (CDNA and RDNA).

    Provides default implementations for ``is_implied_literal_encoding``
    and ``flt_name_map`` that are identical across all AMDGPU generations.
    Subclasses override only the methods that differ.

    Subclass knobs:

    * ``_FLAT_SEGMENTS``: frozenset of FLAT segment suffixes that form
      alternate encodings (e.g. ``{'GLBL', 'SCRATCH'}``). Empty by
      default (RDNA4 has independent VFLAT/VGLOBAL/VSCRATCH instead).
    * ``_SKIP_DPP_SDWA``: when True, ``skip_inst_encoding`` also rejects
      DPP and SDWA encoding variants.
    """

    _FLAT_SEGMENTS: frozenset[str] = frozenset()
    _SKIP_DPP_SDWA: bool = False

    @property
    def flt_name_map(self) -> dict[float, str]:
        return _FLOAT_NAME_MAP

    def is_implied_literal_encoding(
        self,
        enc_name: str,
        enc_conds: list[tuple[str, str]],
        bit_cnt: int,
        parent_bit_cnt: int,
    ) -> bool:
        """Detect implied-literal alternate encodings.

        An encoding is an implied-literal variant when either:

        * One of its condition names contains ``has_lit`` (the XML's way
          of saying "this encoding condition selects the literal path").
        * The encoding name contains ``LITERAL`` and the encoding is
          wider than its parent (indicating the extra DWORD).
        """
        if any('has_lit' in name.lower() for name, _ in enc_conds):
            return True
        if 'LITERAL' in enc_name.upper() and bit_cnt > parent_bit_cnt:
            return True
        return False

    def has_src_modifiers(self, enc_name: str) -> bool:
        """VOP3 (excluding VOP3P) has source modifiers."""
        upper = enc_name.upper()
        return 'VOP3' in upper and 'VOP3P' not in upper

    def has_abs_modifier(self, enc_name: str) -> bool:
        """VOP3 has abs, but VOP3_SDST_ENC does not."""
        upper = enc_name.upper()
        return ('VOP3' in upper and 'VOP3P' not in upper
                and 'SDST_ENC' not in upper)

    def mnemonic_rule(self, enc_name: str) -> MnemonicRule:
        """Default AMDGPU mnemonic rules.

        * VOP1, VOP2, VOPC: append ``_e32`` suffix.
        * FLAT (when using segment variants): rewrite prefix via
          ``flat_mnemonic()`` helper.
        * All others: no transformation.
        """
        upper = enc_name.upper()
        if upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
            return _VOP_E32_RULE
        if upper == 'ENC_FLAT':
            return MnemonicRule(use_flat_mnemonic=True)
        return MnemonicRule()

    def is_alt_encoding(self, enc_name: str) -> bool:
        parts = enc_name.split('_')
        if parts[0] != 'ENC':
            return True
        return (
            len(parts) == 3
            and parts[1] == 'FLAT'
            and parts[2] in self._FLAT_SEGMENTS
        )

    def derive_parent_enc_name(self, enc_name: str) -> str:
        parts = enc_name.split('_')
        if (parts[0] == 'ENC' and len(parts) >= 3
                and parts[1] == 'FLAT' and parts[2] in self._FLAT_SEGMENTS):
            return 'ENC_FLAT'
        return f'ENC_{parts[0]}'

    def skip_inst_encoding(self, enc_name: str, enc_cond: str) -> bool:
        if enc_cond != 'default':
            return True
        if self._SKIP_DPP_SDWA:
            if '_VOP_DPP' in enc_name or '_VOP_SDWA' in enc_name:
                return True
        parts = enc_name.split('_')
        return (
            parts[0] == 'ENC'
            and len(parts) == 3
            and parts[1] == 'FLAT'
            and parts[2] in self._FLAT_SEGMENTS
        )

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        """Default AMDGPU encoding modifiers.

        Returns modifier field lists for SMEM, MUBUF, MTBUF, and FLAT.
        """
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS
        if upper == 'ENC_MUBUF':
            return _MUBUF_MODIFIERS
        if upper == 'ENC_MTBUF':
            return _MTBUF_MODIFIERS
        if upper == 'ENC_FLAT':
            return _FLAT_MODIFIERS
        return []

    # --- ISA dimension properties ---

    @property
    def wave_size(self) -> int:
        """Default wavefront size in lanes (32 or 64)."""
        return 64  # CDNA is Wave64-only; RDNA subclasses override to 32

    @property
    def wave_size_max(self) -> int:
        """Maximum wavefront size. RDNA supports Wave32 and Wave64;
        CDNA is Wave64-only."""
        return self.wave_size

    @property
    def has_acc_vgpr(self) -> bool:
        """True if this ISA has AccVGPRs (CDNA2/3/4 only)."""
        return False

    @property
    def acc_vgpr_encoding_base(self) -> int:
        """Encoding index where AccVGPR range begins (512 for CDNA2, 768 for CDNA3/4)."""
        return 0

    @property
    def max_acc_vgprs(self) -> int:
        """Number of AccVGPRs per wavefront (0 if not present)."""
        return 0

    @property
    def waitcnt_family(self) -> str:
        """Waitcnt encoding family name.

        'gfx9'  — single S_WAITCNT; vmcnt split, lgkmcnt 4-bit at [11:8].
                   ISAs: CDNA1, CDNA2, CDNA3, CDNA4.
        'gfx10' — S_WAITCNT (lgkmcnt 6-bit at [13:8]) + S_WAITCNT_VSCNT.
                   ISAs: RDNA1, RDNA2.
        'gfx11' — S_WAITCNT with changed layout (expcnt at [2:0], lgkmcnt
                   6-bit at [9:4], vmcnt 6-bit at [15:10]).
                   ISAs: RDNA3, RDNA3.5.
        'gfx12' — S_WAITCNT removed; replaced by split S_WAIT_* instructions.
                   ISAs: RDNA4.
        """
        return 'gfx9'

    @property
    def has_mfma(self) -> bool:
        """True if this ISA has MFMA matrix instructions (all CDNA)."""
        return False

    @property
    def has_wmma(self) -> bool:
        """True if this ISA has WMMA matrix instructions (RDNA3+)."""
        return False

    @property
    def flat_scratch_mechanism(self) -> str:
        """How scratch base is located: 'hwreg' (CDNA3/4) or 'sgpr_pair'."""
        return 'sgpr_pair'

    @property
    def has_vopd(self) -> bool:
        """True if this ISA supports VOPD dual-issue instructions (RDNA3+)."""
        return False

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        """Memory coherency encoding model for this ISA family."""
        return MemoryCoherencyModel.GFX9_GLC

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        """Return ``(sc0_field, sc1_field, nt_field_or_None)`` for execute() bodies.

        These names index into the machine-instruction struct fields that
        carry the two cache-scope bits and the non-temporal hint.  On ISAs
        that lack a dedicated NT field, ``nt_field`` is ``None`` and the
        code generator substitutes the literal ``0``.

        Default (CDNA3/4): ``('sc0', 'sc1', 'nt')``.
        """
        return ('sc0', 'sc1', 'nt')

    @property
    def vop3p_opsel_fields(self) -> tuple[str, str]:
        """Return ``(op_sel_field, op_sel_hi_field)`` for VOP3P execute() bodies.

        Default: ``('op_sel', 'op_sel_hi')``.
        RDNA4 renames these to ``('opsel', 'opsel_hi')`` (no underscores).
        """
        return ('op_sel', 'op_sel_hi')

    @property
    def smem_direct_offset_field(self) -> str | None:
        """Field name of the direct SMEM immediate offset, or ``None``.

        When ``None``, the ISA uses the three-field CDNA model:
        ``soffset_en``, ``imm``, and ``offset``/``soffset``.  When a
        string (e.g. ``'offset'`` or ``'ioffset'``), the generated
        ``make_smem_offset`` helper always returns
        ``enc-><field>`` directly with no conditional logic.

        CDNA1/2/3/4 → ``None`` (three-field model).
        RDNA1/2/3/3.5 → ``'offset'``.
        RDNA4 → ``'ioffset'``.
        """
        return None

    @property
    def flat_store_src_field(self) -> str:
        """Field name in the flat/global/scratch machine inst for store source data.

        CDNA3/4 and older flat: ``'data'``.
        RDNA4 vflat/vglobal/vscratch: ``'vsrc'``.
        """
        return 'data'


class CdnaProfile(_AmdgpuProfileBase):
    """ISA profile for the CDNA family (CDNA1 through CDNA4).

    Encoding name conventions:

    - Primary encodings are named ``ENC_<FORMAT>`` (e.g., ``ENC_VOP2``).
    - Alternate encodings either omit the ``ENC_`` prefix
      (e.g., ``VOP2_INST_LITERAL``, ``VOP3_SDST_ENC``) or use a
      three-part ``ENC_FLAT_<SEGMENT>`` name (e.g., ``ENC_FLAT_GLBL``).
    - Implied-literal alternates are identified by encoding conditions
      whose names contain ``has_lit``.
    - FLAT segment variants (GLBL, SCRATCH) share the parent FLAT
      encoding's primary decode table entries and are distinguished at
      runtime by the segment field (bits 14-15 of the first DWORD:
      ``00`` = FLAT, ``01`` = SCRATCH, ``10`` = GLOBAL).

    XML bugs worked around:

    - ENC_VOP3PX2 (CDNA4 only) has zero encoding identifier entries and
      an all-zeros mask; it is skipped entirely.
    - V_SWAP_B32 operands are marked output-only in CDNA4 XML even though
      the instruction reads both registers; the codegen compensates (see
      ``codegen.py:_gen_execute_body``).
    - V_FMAMK/V_FMAAK are missing the ``simm32`` operand in CDNA4 XML;
      the codegen falls back to ``inst_.simm32`` (see
      ``codegen.py:_gen_execute_body``).
    - Read-modify-write destinations (V_FMAC, V_DOT2C, V_DOT4C, etc.)
      are marked output-only in CDNA4 XML even though the instruction
      reads the accumulator; the codegen compensates via
      ``_dst_is_also_source()`` which adds ``vdst``/``sdst`` to
      ``src_operands_`` for instructions with accumulator semantics.
    - All 64-bit encodings (SMEM, VOP3, VOP3P, DS, MUBUF, MTBUF, FLAT)
      have an empty ``<Value />`` in their ``default`` condition in CDNA4
      XML, which parses as ``false``. The codegen handles this by
      skipping the ``default_encoding()`` size check entirely for
      encodings with ``bit_cnt >= 64``, since their ``OpEncoding``
      struct already spans the full instruction width.
    """

    _FLAT_SEGMENTS = frozenset({'GLBL', 'SCRATCH'})

    # XML bug (P1): CDNA3/4 ENC_FLAT lists field 'SVE' at bit 13 but the
    # ISA PDF (CDNA3 Table 100, CDNA4 Table 101) names the field 'LDS'.
    # The LDS field controls whether FLAT accesses local data store vs. VGPR.
    _FLAT_FIELD_RENAMES: dict[str, str] = {'sve': 'lds'}

    def field_renames(self, enc_name: str) -> dict[str, str]:
        if enc_name.upper() == 'ENC_FLAT':
            return self._FLAT_FIELD_RENAMES
        return {}

    @property
    def cmpx_writes_vcc(self) -> bool:
        return True

    @property
    def supported_versions(self) -> list[str]:
        return ['1.0.0', '1.1.0', '1.1.1']

    @property
    def max_enc_bits(self) -> int:
        return 9

    @property
    def max_enc_order(self) -> int:
        return 34

    @property
    def skip_encodings(self) -> frozenset[str]:
        return frozenset({'ENC_VOP3PX2'})

    @property
    def inst_size_overrides(self) -> dict[str, int]:
        # VOP3PX2 instructions are 128-bit (16 bytes) but decoded under
        # the 64-bit VOP3P_MFMA encoding. Override their size so the PC
        # advances correctly past the 128-bit instruction.
        return {
            'V_MFMA_F32_16X16X128_F8F6F4': 16,
            'V_MFMA_F32_32X32X64_F8F6F4': 16,
        }

    # ISA dimension properties for CDNA3/4 (the two ISAs this profile covers).
    # Cdna1Profile and Cdna2Profile override the ones that differ.

    @property
    def has_mfma(self) -> bool:
        return True

    @property
    def has_acc_vgpr(self) -> bool:
        return True

    @property
    def acc_vgpr_encoding_base(self) -> int:
        return 768  # CDNA3/4: AccVGPR range starts at encoding 768

    @property
    def max_acc_vgprs(self) -> int:
        return 256

    @property
    def flat_scratch_mechanism(self) -> str:
        return 'hwreg'  # CDNA3/4 use HW register for scratch base

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX940_SC0_SC1_NT


class Cdna1Profile(CdnaProfile):
    """ISA profile for CDNA1 (GFX908 / MI100).

    Differs from CDNA3/4 (the CdnaProfile defaults):
    - No AccVGPRs.
    - GFX9-style GLC-only coherency model.
    - Scratch base via SGPR pair (not HW register).
    - ENC_VOP3PX2 does not exist in CDNA1 XML.
    """

    @property
    def has_acc_vgpr(self) -> bool:
        return False

    @property
    def acc_vgpr_encoding_base(self) -> int:
        return 0

    @property
    def max_acc_vgprs(self) -> int:
        return 0

    @property
    def flat_scratch_mechanism(self) -> str:
        return 'sgpr_pair'

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX9_GLC

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        return ('glc', 'slc', None)

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS  # soffset_en/imm/glc/nv present in CDNA1
        if upper == 'ENC_MUBUF':
            return _MUBUF_MODIFIERS_GLC
        if upper == 'ENC_MTBUF':
            return _MTBUF_MODIFIERS_GLC
        if upper == 'ENC_FLAT':
            return _FLAT_MODIFIERS_GLC
        return []

    @property
    def skip_encodings(self) -> frozenset[str]:
        return frozenset()


class Cdna2Profile(CdnaProfile):
    """ISA profile for CDNA2 (GFX90A / MI200).

    Differs from CDNA3/4 (the CdnaProfile defaults):
    - AccVGPRs start at encoding 512 (not 768).
    - GFX9-style GLC-only coherency model.
    - Scratch base via SGPR pair.
    - ENC_VOP3PX2 does not exist in CDNA2 XML.
    """

    @property
    def acc_vgpr_encoding_base(self) -> int:
        return 512  # CDNA2: AccVGPR range starts at encoding 512

    @property
    def flat_scratch_mechanism(self) -> str:
        return 'sgpr_pair'

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX9_GLC

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        return ('glc', 'slc', None)

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS  # soffset_en/imm/glc/nv present in CDNA2
        if upper == 'ENC_MUBUF':
            return _MUBUF_MODIFIERS_GLC
        if upper == 'ENC_MTBUF':
            return _MTBUF_MODIFIERS_GLC
        if upper == 'ENC_FLAT':
            return _FLAT_MODIFIERS_GLC
        return []

    @property
    def skip_encodings(self) -> frozenset[str]:
        return frozenset()


class Rdna1Profile(_AmdgpuProfileBase):
    """ISA profile for RDNA1 (GFX10.1, Navi1x).

    Encoding name conventions follow the same pattern as CDNA:

    - Primary encodings: ``ENC_<FORMAT>``.
    - FLAT segment variants: ``ENC_FLAT_GLBL``, ``ENC_FLAT_SCRATCH``.
    - MIMG NSA variants (``MIMG_NSA1``, ``MIMG_NSA2``, ``MIMG_NSA3``)
      have non-default encoding conditions (``has_nsa_*``) and are
      skipped automatically by the ``enc_cond != 'default'`` check.
    - DPP variants (``*_VOP_DPP16``, ``*_VOP_DPP8``) and SDWA variants
      (``*_VOP_SDWA``, ``*_VOP_SDWA_SDST_ENC``) use the ``default``
      encoding condition (unlike CDNA where they use ``has_dpp`` /
      ``has_sdwa``), so they must be skipped by encoding name pattern.

    XML bugs worked around:

    - Reserved field omissions (versions 1.0.0): the parser synthesizes
      padding fields for gaps in the MicrocodeFormat bitfield layout.
    """

    _FLAT_SEGMENTS = frozenset({'GLBL', 'SCRATCH'})
    _SKIP_DPP_SDWA = True

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        # RDNA1/2 uses a 6-bit lgkmcnt field at bits [13:8].
        return '0x3F'

    @property
    def supported_versions(self) -> list[str]:
        return ['1.0.0']

    @property
    def max_enc_bits(self) -> int:
        return 9

    @property
    def max_enc_order(self) -> int:
        return 46

    @property
    def wave_size(self) -> int:
        return 32  # RDNA default is Wave32

    @property
    def wave_size_max(self) -> int:
        return 64  # RDNA supports Wave32 and Wave64

    @property
    def waitcnt_family(self) -> str:
        return 'gfx10'

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX10_GLC_DLC_SLC

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        return ('glc', 'slc', None)

    @property
    def smem_direct_offset_field(self) -> str | None:
        return 'offset'

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS_GLC_DLC
        if upper == 'ENC_MUBUF':
            return _MUBUF_MODIFIERS_GLC_DLC
        if upper == 'ENC_MTBUF':
            return _MTBUF_MODIFIERS_GLC_DLC
        if upper == 'ENC_FLAT':
            return _FLAT_MODIFIERS_GLC_DLC
        return []


class Rdna2Profile(Rdna1Profile):
    """ISA profile for RDNA2 (GFX10.3, Navi2x).

    Inherits all properties from ``Rdna1Profile`` except:

    - Wave64 is not supported: ``wave_size_max == 32``.
    - DPP/SDWA variants still skipped (``_SKIP_DPP_SDWA = True``).
    """

    @property
    def wave_size_max(self) -> int:
        return 32


class Rdna3Profile(_AmdgpuProfileBase):
    """ISA profile for RDNA3 (GFX11, Navi3x).

    Key differences from RDNA1/2:

    - FLAT segment variants use ``GLOBAL`` instead of ``GLBL``
      (``ENC_FLAT_GLOBAL``, ``ENC_FLAT_SCRATCH``).
    - VOPDXY dual-issue encoding uses ``opx``/``opy`` fields instead of
      a single ``op`` field, which the parser cannot handle. Both
      ``VOPDXY`` and ``VOPDXY_INST_LITERAL`` are skipped.
    - DPP support expanded to VOP3/VOP3P/VOPC/VOP3_SDST_ENC.
    - SDWA removed (no ``_VOP_SDWA`` variants).
    - ``ENC_LDSDIR`` and ``ENC_VINTERP`` replace CDNA's ``ENC_VINTRP``.

    XML bugs worked around:

    - VOPDXY dual-opcode format: uses ``opx``/``opy`` instead of ``op``,
      which breaks the parser's single-opcode assumption. Skipped.
    - Reserved field omissions (version 1.0.0): synthesized by the parser.
    """

    _FLAT_SEGMENTS = frozenset({'GLOBAL', 'SCRATCH'})
    _SKIP_DPP_SDWA = True
    _SKIP = frozenset({'VOPDXY', 'VOPDXY_INST_LITERAL'})

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        # GFX11 layout differs from GFX10; this mask applies to the
        # lgkmcnt field at bits [9:4] in the new S_WAITCNT encoding.
        return '0x3F'

    @property
    def waitcnt_decode(self) -> str:
        """GFX11 (RDNA3/3.5) S_WAITCNT SIMM16 layout:

        expcnt[2:0]  = bits [2:0]
        lgkmcnt[5:0] = bits [9:4]
        vmcnt[5:0]   = bits [15:10]
        """
        return (
            'uint32_t expcnt = encoding_value_ & 0x7;\n'
            'uint32_t lgkmcnt = (encoding_value_ >> 4) & 0x3F;\n'
            'uint32_t vmcnt = (encoding_value_ >> 10) & 0x3F;\n'
        )

    @property
    def supported_versions(self) -> list[str]:
        return ['1.0.0', '1.1.0']

    @property
    def max_enc_bits(self) -> int:
        return 9

    @property
    def max_enc_order(self) -> int:
        return 54

    @property
    def skip_encodings(self) -> frozenset[str]:
        return self._SKIP

    @property
    def wave_size(self) -> int:
        return 32

    @property
    def wave_size_max(self) -> int:
        return 64

    @property
    def waitcnt_family(self) -> str:
        return 'gfx11'

    @property
    def has_wmma(self) -> bool:
        return True

    @property
    def has_vopd(self) -> bool:
        return True

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX11_SC0_SC1_TH

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        # RDNA3/3.5 MubufMachineInst uses glc+slc (not sc0+sc1).
        return ('glc', 'slc', None)

    @property
    def smem_direct_offset_field(self) -> str | None:
        return 'offset'

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS_GLC_DLC
        if upper == 'ENC_MUBUF':
            return _MUBUF_MODIFIERS_GLC_DLC
        if upper == 'ENC_MTBUF':
            return _MTBUF_MODIFIERS_GLC_DLC
        if upper == 'ENC_FLAT':
            return _FLAT_MODIFIERS_GLC_DLC
        return []


class Rdna3_5Profile(Rdna3Profile):
    """ISA profile for RDNA3.5 (GFX11.5, Navi3.5x).

    Inherits all properties from ``Rdna3Profile``.  Provided as a distinct
    class so the codegen pipeline can auto-detect RDNA3.5 XML files
    separately from RDNA3.
    """


class Rdna4Profile(_AmdgpuProfileBase):
    """ISA profile for RDNA4.

    Key differences from RDNA3:

    - FLAT memory encodings completely restructured: ``ENC_VFLAT``,
      ``ENC_VGLOBAL``, ``ENC_VSCRATCH`` are independent primary
      encodings (not FLAT segment variants).
    - Buffer/image/LDS encodings renamed: ``ENC_VBUFFER``,
      ``ENC_VIMAGE``, ``ENC_VDS``, ``ENC_VEXPORT``, ``ENC_VSAMPLE``,
      ``ENC_VDSDIR``, ``ENC_VINTERP``.
    - ``ENC_VEXPORT`` has no ``op`` field (single instruction), handled
      by the parser as ``op_field_bit_cnt = 0``.
    - VOPDXY dual-issue encoding still present, still skipped.
    - Memory instruction mnemonics use ``B32``/``B64``/``B96``/``B128``
      suffixes instead of ``DWORD``/``DWORDX2``/``DWORDX3``/``DWORDX4``.
    - DS instructions use ``DS_LOAD_*``/``DS_STORE_*`` instead of
      ``DS_READ_*``/``DS_WRITE_*``.

    XML bugs worked around:

    - VOPDXY dual-opcode format: same issue as RDNA3, skipped.
    """

    _SKIP_DPP_SDWA = True
    _SKIP = frozenset({'VOPDXY', 'VOPDXY_INST_LITERAL'})

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        # RDNA4 removed S_WAITCNT; this property is unused but kept for
        # completeness. Returns 0x3F as a safe no-op default.
        return '0x3F'

    @property
    def supported_versions(self) -> list[str]:
        return ['1.1.0']

    @property
    def max_enc_bits(self) -> int:
        return 9

    @property
    def max_enc_order(self) -> int:
        return 53

    @property
    def skip_encodings(self) -> frozenset[str]:
        return self._SKIP

    @property
    def wave_size(self) -> int:
        return 32

    @property
    def wave_size_max(self) -> int:
        return 64

    @property
    def waitcnt_family(self) -> str:
        return 'gfx12'

    @property
    def has_wmma(self) -> bool:
        return True

    @property
    def has_vopd(self) -> bool:
        return True

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX12_SCOPE_TH

    def mnemonic_rule(self, enc_name: str) -> MnemonicRule:
        """RDNA4 mnemonic rules.

        Same as the base AMDGPU rules except FLAT encodings do not use
        the ``flat_mnemonic()`` rewrite (RDNA4 has independent
        ``ENC_VFLAT``, ``ENC_VGLOBAL``, ``ENC_VSCRATCH``).
        """
        upper = enc_name.upper()
        if upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
            return _VOP_E32_RULE
        return MnemonicRule()

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        # RDNA4 VbufferMachineInst/VflatMachineInst use nv (no sc0/sc1).
        # mtype_from_bits is called with (nv, 0) as a placeholder.
        return ('nv', 'nv', None)

    @property
    def vop3p_opsel_fields(self) -> tuple[str, str]:
        return ('opsel', 'opsel_hi')

    @property
    def smem_direct_offset_field(self) -> str | None:
        return 'ioffset'

    @property
    def flat_store_src_field(self) -> str:
        return 'vsrc'

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        """RDNA4 encoding modifiers.

        Uses GFX12 SCOPE+TH model: SMEM/VBUFFER/VFLAT show only NV.
        """
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS_RDNA4
        if upper in ('ENC_VBUFFER', 'ENC_MUBUF'):
            return _VBUFFER_MODIFIERS_RDNA4
        if upper in ('ENC_VFLAT', 'ENC_VGLOBAL', 'ENC_VSCRATCH'):
            return _VFLAT_MODIFIERS_RDNA4
        return []
