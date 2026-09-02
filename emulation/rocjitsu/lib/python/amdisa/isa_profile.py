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

# \NPI new ISA family: (1) sync shared/machine-readable-isa via download.py and \
# add amdgpu_isa_<isa>.xml, (2) add its profile in this module, (3) regenerate \
# per docs/codegen.md, (4) author the hand-written isa.h / insts.h / mma_exec.h \
# / addr_calc.* under lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<isa>/.
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

    GFX9_GLC = auto()  # CDNA1, CDNA2 — GLC bit only, all memory
    GFX940_SC0_SC1_NT = auto()  # CDNA3, CDNA4 — SC0/SC1+NT vector; GLC scalar
    GFX10_GLC_DLC_SLC = auto()  # RDNA1, RDNA2 — GLC + DLC + SLC
    GFX11_SC0_SC1_TH = auto()  # RDNA3, RDNA3.5 — SC0+SC1 scope + TH hint
    GFX12_SCOPE_TH = auto()  # RDNA4 — 2-bit SCOPE + TH hint


class SwmmacLayout(Enum):
    """Executor calling convention used by sparse WMMA instructions."""

    NONE = auto()
    FIXED_WAVE = auto()
    RUNTIME_WAVE = auto()


class MatrixLayout(Enum):
    """Register/lane layout used by matrix instruction executors."""

    MFMA_ACCUMULATOR = auto()
    WMMA_REPLICATED_HALFWAVE = auto()
    WMMA_SPLIT_K = auto()


class DppOpcodeRule(Enum):
    """Opcode-level availability of a DPP source extension."""

    ALLOW = auto()
    FORBID = auto()
    ROW_SELECT_ONLY = auto()


def _opcode_uses_64bit_data(inst_name: str) -> bool:
    """Return whether an opcode name identifies a 64-bit data operation."""

    return any(part in ('B64', 'F64', 'I64', 'U64') for part in inst_name.split('_'))


_DPP_VOP3P_FORBIDDEN = frozenset(
    {
        'V_DOT4_I32_IU8',
        'V_DOT4_U32_U8',
        'V_DOT8_I32_IU4',
        'V_DOT8_U32_U4',
    }
)


_LEGACY_DPP_COMMON_FORBIDDEN = frozenset(
    {
        'V_READFIRSTLANE_B32',
        'V_CVT_I32_F64',
        'V_CVT_F64_I32',
        'V_CVT_F32_F64',
        'V_CVT_F64_F32',
        'V_CVT_U32_F64',
        'V_CVT_F64_U32',
        'V_TRUNC_F64',
        'V_CEIL_F64',
        'V_RNDNE_F64',
        'V_FLOOR_F64',
        'V_RCP_F64',
        'V_RSQ_F64',
        'V_SQRT_F64',
        'V_FREXP_EXP_I32_F64',
        'V_FREXP_MANT_F64',
        'V_FRACT_F64',
        'V_CLREXCP',
        'V_SWAP_B32',
        'V_CMP_CLASS_F64',
        'V_CMPX_CLASS_F64',
    }
)


def _legacy_dpp_opcode_rule(inst_name: str) -> DppOpcodeRule:
    """Implement the CDNA1-4 and RDNA1-2 DPP limitation tables."""

    name = inst_name.upper()
    if name in _LEGACY_DPP_COMMON_FORBIDDEN:
        return DppOpcodeRule.FORBID
    if name.startswith(('V_CMP_', 'V_CMPX_')) and _opcode_uses_64bit_data(name):
        return DppOpcodeRule.FORBID
    return DppOpcodeRule.ALLOW


def _modern_rdna_dpp_opcode_rule(
    enc_name: str,
    inst_name: str,
    *,
    allow_dot2_vop3p: bool,
    forbid_cvt_pk_f32_vop3: bool,
    rdna4_vop1_exclusions: bool,
) -> DppOpcodeRule:
    """Implement the GFX11/GFX12 opcode tables for DPP16/DPP8."""

    enc = enc_name.upper()
    name = inst_name.upper()
    if enc == 'VOP3_SDST_ENC':
        enc = 'ENC_VOP3'

    if enc == 'ENC_VOP1':
        if _opcode_uses_64bit_data(name):
            return DppOpcodeRule.FORBID
        rdna3_exclusions = {
            'V_READFIRSTLANE_B32',
            'V_SWAP_B32',
            'V_PIPEFLUSH',
            'V_WRITELANE_REGWR_B32',
            'V_PERMUTE64',
            'V_PERMLANE64_B32',
        }
        if rdna4_vop1_exclusions:
            prohibited = name in {'V_READFIRSTLANE_B32', 'V_NOP'} or name.startswith(
                ('V_SWAP', 'V_PERMLANE')
            )
        else:
            prohibited = name in rdna3_exclusions
        if prohibited:
            return DppOpcodeRule.FORBID
        return DppOpcodeRule.ALLOW

    if enc == 'ENC_VOP2':
        if _opcode_uses_64bit_data(name) or name.startswith(('V_FMAMK_', 'V_FMAAK_')):
            return DppOpcodeRule.FORBID
        return DppOpcodeRule.ALLOW

    if enc == 'ENC_VOP3P':
        if name.startswith('V_FMA_MIX'):
            return DppOpcodeRule.ALLOW
        if allow_dot2_vop3p and name in (
            'V_DOT2_F32_F16',
            'V_DOT2_F32_BF16',
        ):
            return DppOpcodeRule.ALLOW
        if (
            name in _DPP_VOP3P_FORBIDDEN
            or name.startswith(('V_PK_', 'V_DOT4_', 'V_DOT8_'))
            or 'WMMA' in name
        ):
            return DppOpcodeRule.FORBID
        # The manuals enumerate the legal VOP3P operations rather than giving
        # an open-ended "others" category. Fail closed for future opcodes.
        return DppOpcodeRule.FORBID

    if enc == 'ENC_VOP3':
        if _opcode_uses_64bit_data(name):
            return DppOpcodeRule.FORBID
        if forbid_cvt_pk_f32_vop3 and name.startswith('V_CVT_PK_F32_'):
            return DppOpcodeRule.FORBID
        if name in {'V_MUL_LO_U32', 'V_MUL_HI_U32', 'V_MUL_HI_I32'} or name.startswith(
            (
                'V_QSAD_PK_',
                'V_MQSAD_',
                'V_READLANE',
                'V_WRITELANE',
                'V_PERMLANE',
                'V_SWAP',
            )
        ):
            return DppOpcodeRule.FORBID
        return DppOpcodeRule.ALLOW

    if enc == 'ENC_VOPC':
        return (
            DppOpcodeRule.FORBID
            if _opcode_uses_64bit_data(name)
            else DppOpcodeRule.ALLOW
        )

    return DppOpcodeRule.ALLOW


class WaveStateLayout(Enum):
    """Architectural layout of wave status, exception, and trap-control state."""

    LEGACY = 'Legacy'
    GFX12 = 'Gfx12'
    GFX12_5 = 'Gfx12_5'


class DppCtrlDialect(Enum):
    """Names and validity rules for DPP_CTRL values."""

    GFX9 = auto()
    GFX10_PLUS = auto()


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


@dataclass(frozen=True)
class VopdSlotOp:
    """One MRISA V_DUAL_* slot opcode used by the generated VOPD decoder."""

    enum_name: str
    opcode: int
    mnemonic: str


@dataclass(frozen=True)
class VopdEncodingPrefix:
    """A primary-table prefix routed to the generated VOPD decoder."""

    prefix: int
    prefix_bits: int
    is_vopd3: bool = False


@dataclass(frozen=True)
class MfmaScaleVop3px2Spec:
    """One compound CDNA block-scale MFMA decoded from a VOP3PX2 pair."""

    dense_name: str
    dense_class_name: str
    scaled_name: str
    class_name: str
    mnemonic: str
    opcode: int
    m: int
    n: int
    k: int
    prefix_opcode: int = 0x2C


_VOPD_COMMON_F32_SLOT_OPS = (
    VopdSlotOp('VopdFmacF32', 0, 'v_dual_fmac_f32'),
    VopdSlotOp('VopdFmaakF32', 1, 'v_dual_fmaak_f32'),
    VopdSlotOp('VopdFmamkF32', 2, 'v_dual_fmamk_f32'),
    VopdSlotOp('VopdMulF32', 3, 'v_dual_mul_f32'),
    VopdSlotOp('VopdAddF32', 4, 'v_dual_add_f32'),
    VopdSlotOp('VopdSubF32', 5, 'v_dual_sub_f32'),
    VopdSlotOp('VopdSubrevF32', 6, 'v_dual_subrev_f32'),
    VopdSlotOp('VopdMulDx9ZeroF32', 7, 'v_dual_mul_dx9_zero_f32'),
    VopdSlotOp('VopdMovB32', 8, 'v_dual_mov_b32'),
    VopdSlotOp('VopdCndmaskB32', 9, 'v_dual_cndmask_b32'),
)

_RDNA3_VOPD_SLOT_OPS = _VOPD_COMMON_F32_SLOT_OPS + (
    VopdSlotOp('VopdMaxF32', 10, 'v_dual_max_f32'),
    VopdSlotOp('VopdMinF32', 11, 'v_dual_min_f32'),
    VopdSlotOp('VopdDot2AccF32F16', 12, 'v_dual_dot2acc_f32_f16'),
    VopdSlotOp('VopdDot2AccF32Bf16', 13, 'v_dual_dot2acc_f32_bf16'),
    VopdSlotOp('VopdAddNcU32', 16, 'v_dual_add_nc_u32'),
    VopdSlotOp('VopdLshlrevB32', 17, 'v_dual_lshlrev_b32'),
    VopdSlotOp('VopdAndB32', 18, 'v_dual_and_b32'),
)

_RDNA4_VOPD_SLOT_OPS = _VOPD_COMMON_F32_SLOT_OPS + (
    VopdSlotOp('VopdMaxNumF32', 10, 'v_dual_max_num_f32'),
    VopdSlotOp('VopdMinNumF32', 11, 'v_dual_min_num_f32'),
    VopdSlotOp('VopdDot2AccF32F16', 12, 'v_dual_dot2acc_f32_f16'),
    VopdSlotOp('VopdDot2AccF32Bf16', 13, 'v_dual_dot2acc_f32_bf16'),
    VopdSlotOp('VopdAddNcU32', 16, 'v_dual_add_nc_u32'),
    VopdSlotOp('VopdLshlrevB32', 17, 'v_dual_lshlrev_b32'),
    VopdSlotOp('VopdAndB32', 18, 'v_dual_and_b32'),
)

_CDNA5_VOPD_SLOT_OPS = _VOPD_COMMON_F32_SLOT_OPS + (
    VopdSlotOp('VopdMaxNumF32', 10, 'v_dual_max_num_f32'),
    VopdSlotOp('VopdMinNumF32', 11, 'v_dual_min_num_f32'),
    VopdSlotOp('VopdAddNcU32', 16, 'v_dual_add_nc_u32'),
    VopdSlotOp('VopdLshlrevB32', 17, 'v_dual_lshlrev_b32'),
    VopdSlotOp('VopdBitop2B32', 18, 'v_dual_bitop2_b32'),
    VopdSlotOp('VopdFmaF32', 19, 'v_dual_fma_f32'),
    VopdSlotOp('VopdSubNcU32', 20, 'v_dual_sub_nc_u32'),
    VopdSlotOp('VopdLshrrevB32', 21, 'v_dual_lshrrev_b32'),
    VopdSlotOp('VopdAshrrevI32', 22, 'v_dual_ashrrev_i32'),
    VopdSlotOp('VopdMaxI32', 23, 'v_dual_max_i32'),
    VopdSlotOp('VopdMinI32', 24, 'v_dual_min_i32'),
    VopdSlotOp('VopdFmaF64', 32, 'v_dual_fma_f64'),
    VopdSlotOp('VopdAddF64', 33, 'v_dual_add_f64'),
    VopdSlotOp('VopdMulF64', 34, 'v_dual_mul_f64'),
    VopdSlotOp('VopdMaxNumF64', 35, 'v_dual_max_num_f64'),
    VopdSlotOp('VopdMinNumF64', 36, 'v_dual_min_num_f64'),
)


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
    def generated_arch_name(self) -> str | None:
        """Override for the logical architecture name used by code generation."""
        return None

    @property
    def generated_dir_name(self) -> str | None:
        """Override for the generated and handwritten filesystem directory."""
        return None

    @property
    def cpp_namespace(self) -> str | None:
        """Override for the generated C++ architecture namespace."""
        return None

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
        """Per-instruction size overrides in bytes."""
        return {}

    @property
    def compatibility_instruction_slots(self) -> dict[tuple[str, int], str]:
        """Opcode slots owned by instructions synthesized after XML parsing."""
        return {}

    @property
    def vop3px2_prefix_opcode(self) -> int | None:
        """VOP3P opcode slot for a VOP3PX2 prefix decoder, if any."""
        return None

    @property
    def mfma_scale_vop3px2_specs(self) -> tuple[MfmaScaleVop3px2Spec, ...]:
        """Compound block-scale MFMA encodings supplied by this profile."""
        return ()

    @property
    def source_split_max_bytes(self) -> dict[str, int]:
        """Maximum generated source chunk size by encoding.

        Large generated instruction implementation files can be split into
        multiple translation units. Keys are XML encoding names such as
        ``ENC_VOP3``; values are soft byte limits used by the generator.
        """
        return {}

    def source_split_file_stem(
        self, enc_name: str, inst_name: str, semantics: object | None
    ) -> str | None:
        """Optional logical source-file stem for a generated instruction.

        When ``source_split_max_bytes`` asks the generator to split an
        encoding's implementation file, profiles can return a descriptive
        stem such as ``cvt_pack`` or ``cmpx_f32``. The generator writes
        matching ``<encoding>_<stem>.cpp`` chunks while still enforcing the
        configured byte limit.
        """
        return None

    @property
    def uses_vgpr_msb_indexing(self) -> bool:
        """True when VGPR operands use MODE-controlled high-bank bits."""
        return False

    @property
    def d16_loads_zero_unselected_half(self) -> bool:
        """True when D16 loads zero the unselected destination half."""
        return False

    @property
    def supports_gpr_idx(self) -> bool:
        """True when MODE.GPR_IDX_EN applies M0 indexing to VALU operands."""
        return False

    @property
    def uses_packed_16bit_e32_source_selectors(self) -> bool:
        """True when E32 16-bit source selectors can address packed high halves."""
        return False

    @property
    def renders_gfx11_image_syntax(self) -> bool:
        """Whether GFX11 image operands and modifiers use canonical syntax."""
        return False

    @property
    def split_ds_2addr_offsets(self) -> bool:
        """Whether DS 2ADDR instructions render two independent offsets."""
        return False

    @property
    def vop3p_absolute_source_instructions(self) -> frozenset[str]:
        """VOP3P instructions whose NEG fields encode source abs and negate."""
        return frozenset()

    @property
    def gfx11_mimg_gather_style_instructions(self) -> frozenset[str]:
        """GFX11 MIMG instructions with gather-style VDATA sizing."""
        return frozenset()

    @property
    def gfx11_mimg_fixed_vdata_words(self) -> dict[str, int]:
        """GFX11 MIMG instructions with a fixed VDATA width in DWORDs."""
        return {}

    @property
    def gfx11_mimg_fixed_vaddr_words(self) -> dict[str, tuple[int, int]]:
        """GFX11 MIMG VADDR widths as ``(default, a16)`` DWORD counts."""
        return {}

    @property
    def gfx11_mimg_nsa_group_words(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        """GFX11 partial-NSA group widths as ``(default, a16)`` tuples."""
        return {}

    @property
    def sendmsg_return_symbolic(self) -> bool:
        """Whether return-message selectors use symbolic assembly syntax."""
        return False

    @property
    def split_execution_sources(self) -> bool:
        """True when execution bodies are emitted to separate source files."""
        return False

    @property
    def uses_true16_vop3_opsel(self) -> bool:
        """True when VOP3 16-bit operands use op_sel half selectors."""
        return False

    @property
    def renders_true16_vop3_operands(self) -> bool:
        """True when VOP3 operands use explicit ``.l``/``.h`` suffixes."""
        return False

    @property
    def vop3_opsel_omissions(self) -> frozenset[str]:
        """VOP3 instructions whose half selection is shown only on operands."""
        return frozenset()

    @property
    def vop3p_source_modifier_omissions(self) -> frozenset[str]:
        """VOP3P instructions that do not use packed source modifier fields."""
        return frozenset()

    @property
    def integer_clamp_dtypes(self) -> dict[str, str]:
        """Integer VOP3 instructions whose CLAMP modifier saturates the result.

        CLAMP is present in the shared VOP3 encoding, but it is not legal for
        every opcode in that encoding. Keep the instruction-level policy here
        instead of inferring support from the encoding field alone.
        """
        return {
            'V_ADDC_CO_U32': 'u32',
            'V_ADD_CO_CI_U32': 'u32',
            'V_ADD_CO_U32': 'u32',
            'V_ADD_I16': 'i16',
            'V_ADD_I32': 'i32',
            'V_ADD_NC_I16': 'i16',
            'V_ADD_NC_I32': 'i32',
            'V_ADD_NC_U16': 'u16',
            'V_ADD_NC_U32': 'u32',
            'V_ADD_NC_U64': 'u64',
            'V_ADD_U16': 'u16',
            'V_ADD_U32': 'u32',
            'V_MAD_I16': 'i16',
            'V_MAD_I32_I16': 'i32',
            'V_MAD_I32_I24': 'i32',
            'V_MAD_U16': 'u16',
            'V_MAD_U32_U16': 'u32',
            'V_MAD_U32_U24': 'u32',
            'V_MSAD_U8': 'u32',
            'V_MUL_I32_I24': 'i32',
            'V_MUL_U32_U24': 'u32',
            'V_SAD_U8': 'u32',
            'V_SAD_U16': 'u32',
            'V_SAD_U32': 'u32',
            'V_SUBBREV_CO_U32': 'u32',
            'V_SUBB_CO_U32': 'u32',
            'V_SUBREV_CO_CI_U32': 'u32',
            'V_SUBREV_CO_U32': 'u32',
            'V_SUBREV_NC_U32': 'u32',
            'V_SUBREV_U16': 'u16',
            'V_SUBREV_U32': 'u32',
            'V_SUB_CO_CI_U32': 'u32',
            'V_SUB_CO_U32': 'u32',
            'V_SUB_I16': 'i16',
            'V_SUB_I32': 'i32',
            'V_SUB_NC_I16': 'i16',
            'V_SUB_NC_I32': 'i32',
            'V_SUB_NC_U16': 'u16',
            'V_SUB_NC_U32': 'u32',
            'V_SUB_NC_U64': 'u64',
            'V_SUB_U16': 'u16',
            'V_SUB_U32': 'u32',
        }

    @property
    def dpp_ctrl_dialect(self) -> DppCtrlDialect:
        """DPP_CTRL naming and validity rules for this ISA."""
        return DppCtrlDialect.GFX9

    @property
    def scalar_null_precedes_m0(self) -> bool:
        """True when scalar operand code 124 is null and 125 is m0."""
        return False

    @property
    def vbuffer_store_data_uses_dst_vgpr_msb_role(self) -> bool:
        """True when buffer-store data operands use the destination VGPR-MSB bank."""
        return False

    @property
    def buffer_payload_reads_use_effective_exec_mask(self) -> bool:
        """True when buffer store/atomic payload reads use post-address-calc EXEC."""
        return False

    @property
    def generate_scaled_wmma_vop3px2(self) -> bool:
        """True when generator should synthesize scaled-WMMA VOP3PX2 support."""
        return False

    @property
    def smem_address_uses_access_size(self) -> bool:
        """True when generated SMEM address helpers need the access size."""
        return False

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
    def semantic_class_overrides(self) -> dict[str, str]:
        """Per-instruction semantic-class refinements for this ISA.

        Unlike :attr:`semantic_overrides`, these preserve the generic
        derivation's operation, element size, and other metadata. Use this
        when an ISA-specific instruction shape needs a different codegen
        template but the remaining mnemonic-derived metadata is still valid.
        """
        return {}

    @property
    def ds_addtid_uses_m0_byte_base(self) -> bool:
        """True when DS ADDTID addresses use M0 as a byte-address base."""
        return False

    @property
    def ds_transpose_ignores_exec(self) -> bool:
        """True when DS transpose loads replace EXEC with an all-lanes mask.

        The effective mask applies to issue, address generation, and load
        writeback, not only to address formation.
        """
        return False

    @property
    def ds_b8_transpose_kind(self) -> int:
        """Cross-lane routing used by 8-bit B64 DS transpose loads.

        CDNA4 and older targets use the MFMA-oriented B64_TR_B8 routing.
        Architectures with target-specific DS B8 routing override this property
        so opcode aliases cannot acquire different behavior from their spelling.
        """
        return 3

    @property
    def global_b8_transpose_kind(self) -> int:
        """Cross-lane routing used by 8-bit B64 global transpose loads.

        Current targets use the 16x16 WMMA routing. Keep this selection in
        the ISA profile so a future architecture can change the global-load
        routing without making mnemonic aliases disagree.
        """
        return 6

    @property
    def cmpx_writes_vcc(self) -> bool:
        """True if V_CMPX instructions write both EXEC and VCC.

        On CDNA (GFX9-based), V_CMPX writes both EXEC and VCC. On RDNA,
        V_CMPX writes only EXEC.
        """
        return False

    @property
    def dpp_bound_ctrl_applies_to_inactive_sources(self) -> bool:
        """True when FI=0 makes an inactive DPP source subject to BOUND_CTRL.

        GFX11 and newer distinguish fetching an inactive source (FI) from the
        invalid-source action (BOUND_CTRL). Earlier RDNA generations instead
        define FI=0 as supplying zero directly. Physical gfx1201 Wave32 and
        Wave64 results confirm that FI does not affect out-of-range sources;
        BOUND_CTRL alone governs them.
        """
        return False

    @property
    def dpp_suppressed_compare_lanes_zero(self) -> bool:
        """True when DPP-suppressed compare result bits are cleared.

        This covers inactive destinations, row/bank-masked destinations, and
        invalid sources whose write is suppressed by BOUND_CTRL=0.
        """
        return False

    @property
    def dpp_supports_row_xmask(self) -> bool:
        """True when DPP_ROW_XMASK[0:15] is a documented control range."""
        return False

    @property
    def dpp_supports_wave_controls(self) -> bool:
        """True when the four single-step DPP_WF controls are documented."""
        return True

    @property
    def dpp_supports_row_broadcast_controls(self) -> bool:
        """True when DPP_ROW_BCAST15/31 are documented controls."""
        return True

    @property
    def dpp_requires_opsel_lane_alignment(self) -> bool:
        """True when DPP VOP3/VOP3P OPSEL halves must remain aligned."""
        return False

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        """Return the opcode-level DPP rule for an instruction.

        Legacy profiles retain their existing encoding-level behavior. Modern
        profiles override this with the opcode tables from their ISA manuals.
        """
        del enc_name, inst_name, src0_size_bits
        return DppOpcodeRule.ALLOW

    def supports_sdwa_opcode(
        self,
        enc_name: str,
        inst_name: str,
        *,
        has_modifier_encoding: bool,
    ) -> bool:
        """Return whether an opcode supports the architecture's SDWA form.

        Most profiles can use the alternate encodings listed in the MR ISA
        directly. Profiles may override this when the architecture manual has
        a more complete opcode rule than the available XML.
        """
        del enc_name, inst_name
        return has_modifier_encoding

    @property
    def vop3_cmp_sdst_size_bits(self) -> int | None:
        """Explicit VOP3 compare destination width, if target-specific."""
        return None

    @property
    def vop3_cndmask_selector_size_bits(self) -> int | None:
        """Explicit VOP3 cndmask scalar-selector width, if target-specific."""
        return None

    @property
    def vop3_carry_mask_size_bits(self) -> int | None:
        """Explicit VOP3 carry input/output mask width, if target-specific."""
        return None

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

    def saddr_null_selector_expr(self, enc_name: str) -> str | None:
        """Return the generated NULL-SADDR selector for an encoding.

        The selector is an encoding property rather than a generic scalar
        operand-table property. Profiles return ``None`` for encodings that do
        not carry an optional scalar address.
        """
        return None

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

    def normalize_operand_field_name(self, enc_name: str, field_name: str) -> str:
        """Return the normalized field name used by an instruction operand."""
        return self.field_renames(enc_name).get(field_name, field_name)

    def normalize_operand_type(
        self, enc_name: str, field_name: str, operand_type: str
    ) -> str:
        """Return the normalized type used by an instruction operand."""
        return operand_type

    @property
    def lowercase_operand_selector_names(self) -> bool:
        """Whether predefined symbolic operand names normalize to lowercase."""
        return False

    def normalize_encoding_condition(self, enc_name: str, cond_name: str) -> str:
        """Return the logical condition name to use in generated code.

        Some XML revisions spell an encoding's base condition as an expression
        rather than the literal ``default`` name. Profiles can normalize those
        names here while leaving instruction filtering decisions in
        ``skip_inst_encoding``.
        """
        return cond_name

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
        'offset',
        is_offset=True,
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
        'flat_offset',
        is_offset=True,
        preamble=(
            'int flat_offset = inst->offset | (inst->pad_12 << 12);'
            'if (inst->seg == 0) flat_offset = inst->offset;'
            'else if (flat_offset & 0x1000) flat_offset -= 0x2000;'
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
        'flat_offset',
        is_offset=True,
        preamble=(
            'int flat_offset = inst->offset | (inst->pad_12 << 12);'
            'if (inst->seg == 0) flat_offset = inst->offset;'
            'else if (flat_offset & 0x1000) flat_offset -= 0x2000;'
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

# GFX12 (RDNA4): encoding-specific modifiers beyond the data-driven SCOPE+TH
# cache policy emitted for every encoding that carries op/scope/th fields.
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

_IMAGE_MODIFIERS_RDNA4 = [
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
    def split_execution_sources(self) -> bool:
        """Split every built-in AMDGPU target into model and execution sources.

        Custom profiles may override this for compatibility with the generator's
        non-split fallback, which remains covered independently.
        """
        return True

    def field_renames(self, enc_name: str) -> dict[str, str]:
        # The public 1.1.1 snapshot gives literal extension DWORDs an
        # explicit LITERAL field.  Preserve the established generated member
        # name used by the earlier AMDGPU specifications.
        return {'literal': 'simm32'}

    @property
    def lowercase_operand_selector_names(self) -> bool:
        # The 1.1.1 XML snapshot uppercases symbolic selector names, unlike
        # the preceding AMDGPU specs and their established disassembly.
        return True

    def normalize_operand_type(
        self, enc_name: str, field_name: str, operand_type: str
    ) -> str:
        # Before schema 1.1.1, the 16-bit K operand in VOP2 MADMK/FMAMK
        # instructions was represented by the 32-bit literal extension
        # operand.  The newer XML calls the same encoded value OPR_SIMM16.
        # Preserve the established literal identity (and hexadecimal
        # disassembly) when the field rename identifies that extension.
        if (
            field_name == 'literal'
            and operand_type == 'OPR_SIMM16'
            and self.normalize_operand_field_name(enc_name, field_name) == 'simm32'
        ):
            return 'OPR_SIMM32'
        return operand_type

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
        return 'VOP3' in upper and 'VOP3P' not in upper and 'SDST_ENC' not in upper

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

    def saddr_null_selector_expr(self, enc_name: str) -> str | None:
        """Legacy FLAT reserves the all-ones 7-bit SADDR selector."""
        if enc_name.upper() == 'ENC_FLAT':
            return '0x7F'
        return None

    def is_alt_encoding(self, enc_name: str) -> bool:
        parts = enc_name.split('_')
        if parts[0] != 'ENC':
            return True
        return (
            len(parts) == 3 and parts[1] == 'FLAT' and parts[2] in self._FLAT_SEGMENTS
        )

    def derive_parent_enc_name(self, enc_name: str) -> str:
        parts = enc_name.split('_')
        if (
            parts[0] == 'ENC'
            and len(parts) >= 3
            and parts[1] == 'FLAT'
            and parts[2] in self._FLAT_SEGMENTS
        ):
            return 'ENC_FLAT'
        for suffix in (
            '_INST_LITERAL64',
            '_INST_LITERAL',
            '_VOP_DPP16',
            '_VOP_DPP8',
            '_VOP_DPP',
            '_VOP_SDWA',
        ):
            if enc_name.endswith(suffix):
                parent_name = enc_name[: -len(suffix)]
                if parent_name.endswith('_ENC'):
                    return parent_name
                return f'ENC_{parent_name}'
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
    def supports_wgp_mode(self) -> bool:
        """Whether COMPUTE_PGM_RSRC1.WGP_MODE exists."""
        return False

    @property
    def uses_ttmp_workgroup_ids(self) -> bool:
        """Whether dispatch workgroup IDs are carried in TTMP registers."""
        return False

    @property
    def uses_cluster_ttmp_workgroup_ids(self) -> bool:
        """Whether the TTMP workgroup-ID payload uses cluster coordinates."""
        return False

    @property
    def wave_state_layout(self) -> WaveStateLayout:
        """Layout of shader-visible wave state and the first-level trap ABI."""
        return WaveStateLayout.LEGACY

    @property
    def compute_tmpring_wavesize_granule(self) -> int:
        """Bytes represented by one COMPUTE_TMPRING_SIZE.WAVESIZE unit."""
        return 1024

    @property
    def compute_tmpring_wavesize_bits(self) -> int:
        """Width of the COMPUTE_TMPRING_SIZE.WAVESIZE field."""
        return 13

    @property
    def max_addressable_vgprs_per_wf(self) -> int:
        """Maximum VGPR index space addressable by one wavefront."""
        return 256

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        """Wave32 VGPR count granule encoded in compute descriptors."""
        return 0

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        """Wave64 VGPR count granule encoded in compute descriptors."""
        return 4

    @property
    def descriptor_sgpr_count_encoded(self) -> bool:
        """Whether zero SGPR granule fields still use descriptor encoding."""
        return True

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
    def swmmac_layout(self) -> SwmmacLayout:
        """Sparse WMMA executor layout supported by this ISA."""
        return SwmmacLayout.NONE

    @property
    def matrix_layout(self) -> MatrixLayout:
        """Register/lane mapping used by matrix instructions."""
        return MatrixLayout.MFMA_ACCUMULATOR

    @property
    def flat_scratch_mechanism(self) -> str:
        """How scratch base is located: 'hwreg' (CDNA3/4) or 'sgpr_pair'."""
        return 'sgpr_pair'

    @property
    def has_vopd(self) -> bool:
        """True if this ISA supports VOPD dual-issue instructions (RDNA3+)."""
        return False

    @property
    def has_vopd3(self) -> bool:
        """True if this ISA supports the VOPD3 encoding form."""
        return any(prefix.is_vopd3 for prefix in self.vopd_encoding_prefixes)

    @property
    def vopd_encoding_prefixes(self) -> tuple[VopdEncodingPrefix, ...]:
        """Primary encoding prefixes accepted by the generated VOPD decoder."""
        if not self.has_vopd:
            return ()
        return (VopdEncodingPrefix(0x32, 6),)

    @property
    def vopd_slot_ops(self) -> tuple[VopdSlotOp, ...]:
        """MRISA V_DUAL_* slot opcode table for generated VOPD support."""
        return ()

    @property
    def vopd_x_slot_opcodes(self) -> frozenset[int]:
        """Opcode values accepted in the VOPD X slot."""
        return frozenset(op.opcode for op in self.vopd_slot_ops)

    @property
    def vopd_y_slot_opcodes(self) -> frozenset[int]:
        """Opcode values accepted in the VOPD Y slot."""
        return frozenset(op.opcode for op in self.vopd_slot_ops)

    @property
    def vopd3_x_slot_opcodes(self) -> frozenset[int]:
        """Opcode values accepted in the VOPD3 X slot."""
        return frozenset()

    @property
    def vopd3_y_slot_opcodes(self) -> frozenset[int]:
        """Opcode values accepted in the VOPD3 Y slot."""
        return frozenset()

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
    def vop3p_opsel_hi_high_field(self) -> str:
        """Return the machine field carrying VOP3P op_sel_hi bit 2."""
        return 'op_sel_hi_2'

    @property
    def vop3_opsel_field(self) -> str:
        """Return the source and destination half-selector field for VOP3."""
        return 'op_sel'

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
      an all-zeros mask; it is skipped entirely by the CDNA4 profile.
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

    @property
    def supports_gpr_idx(self) -> bool:
        return True

    _FLAT_SEGMENTS = frozenset({'GLBL', 'SCRATCH'})
    # XML bug (P1): CDNA3/4 ENC_FLAT lists field 'SVE' at bit 13 but the
    # ISA PDF (CDNA3 Table 100, CDNA4 Table 101) names the field 'LDS'.
    # The LDS field controls whether FLAT accesses local data store vs. VGPR.
    _FLAT_FIELD_RENAMES: dict[str, str] = {'sve': 'lds'}

    # XML bug (P2): CDNA4 ENC_VOP3P lists bit 14 as 'PAD' but the ISA PDF
    # names it 'OP_SEL_HI_2' (the third bit of op_sel_hi for src2 hi/lo
    # half selection in packed FP16/BF16 instructions). CDNA3's XML already
    # uses the correct name; the rename is a no-op there.
    _VOP3P_FIELD_RENAMES: dict[str, str] = {'pad_14': 'op_sel_hi_2'}

    def field_renames(self, enc_name: str) -> dict[str, str]:
        upper = enc_name.upper()
        renames = dict(super().field_renames(enc_name))
        if upper == 'ENC_FLAT':
            renames.update(self._FLAT_FIELD_RENAMES)
        if upper == 'ENC_VOP3P':
            renames.update(self._VOP3P_FIELD_RENAMES)
        return renames

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
        return {spec.dense_name: 16 for spec in self.mfma_scale_vop3px2_specs}

    @property
    def vop3px2_prefix_opcode(self) -> int | None:
        specs = self.mfma_scale_vop3px2_specs
        return specs[0].prefix_opcode if specs else None

    @property
    def mfma_scale_vop3px2_specs(self) -> tuple[MfmaScaleVop3px2Spec, ...]:
        return ()

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
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 0

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 8

    @property
    def flat_scratch_mechanism(self) -> str:
        return 'hwreg'  # CDNA3/4 use HW register for scratch base

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX940_SC0_SC1_NT

    @property
    def uses_true16_vop3_opsel(self) -> bool:
        # CDNA VOP3 OP_SEL uses bits [0:2] for source half selection and
        # bit [3] for destination half selection. Low-destination writes
        # zero the upper half; see the CDNA ISA OP_SEL field description.
        return True

    @property
    def vop3p_source_modifier_omissions(self) -> frozenset[str]:
        # These B32 accumulator moves reuse the VOP3P encoding but do not have
        # packed half-selection or per-half negate semantics.
        return frozenset({'V_ACCVGPR_READ', 'V_ACCVGPR_WRITE'})

    @property
    def d16_loads_zero_unselected_half(self) -> bool:
        # CDNA2/3/4 enable SRAM ECC in the runtime ISA traits.
        return True

    @property
    def dpp_64bit_input_row_select_only(self) -> bool:
        """Whether 64-bit DPP inputs are restricted to DPP_ROW[0:15]."""
        return True

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del enc_name
        base_rule = _legacy_dpp_opcode_rule(inst_name)
        # The complete opcode prohibition table takes precedence over the
        # row-select-only allowance for otherwise legal 64-bit inputs.
        if base_rule is DppOpcodeRule.FORBID:
            return base_rule
        if self.dpp_64bit_input_row_select_only and src0_size_bits == 64:
            return DppOpcodeRule.ROW_SELECT_ONLY
        return base_rule

    def supports_sdwa_opcode(
        self,
        enc_name: str,
        inst_name: str,
        *,
        has_modifier_encoding: bool,
    ) -> bool:
        # CDNA's SDWA limitation table permits V_PK_FMAC_F16, and the hardware
        # encoding is exercised by LLVM, but the MR ISA omits its SDWA alternate.
        if enc_name.upper() == 'ENC_VOP2' and inst_name == 'V_PK_FMAC_F16':
            return True
        return has_modifier_encoding


class Cdna4Profile(CdnaProfile):
    """ISA profile for CDNA4-only encoding capabilities."""

    @property
    def mfma_scale_vop3px2_specs(self) -> tuple[MfmaScaleVop3px2Spec, ...]:
        return (
            MfmaScaleVop3px2Spec(
                dense_name='V_MFMA_F32_16X16X128_F8F6F4',
                dense_class_name='VMfmaF3216x16x128F8f6f4Vop3pMfma',
                scaled_name='V_MFMA_SCALE_F32_16X16X128_F8F6F4',
                class_name='VMfmaScaleF3216x16x128F8f6f4Vop3px2',
                mnemonic='v_mfma_scale_f32_16x16x128_f8f6f4',
                opcode=45,
                m=16,
                n=16,
                k=128,
            ),
            MfmaScaleVop3px2Spec(
                dense_name='V_MFMA_F32_32X32X64_F8F6F4',
                dense_class_name='VMfmaF3232x32x64F8f6f4Vop3pMfma',
                scaled_name='V_MFMA_SCALE_F32_32X32X64_F8F6F4',
                class_name='VMfmaScaleF3232x32x64F8f6f4Vop3px2',
                mnemonic='v_mfma_scale_f32_32x32x64_f8f6f4',
                opcode=46,
                m=32,
                n=32,
                k=64,
            ),
        )


class Cdna1Profile(CdnaProfile):
    """ISA profile for CDNA1 (GFX908 / MI100).

    Differs from CDNA3/4 (the CdnaProfile defaults):
    - No AccVGPRs.
    - GFX9-style GLC-only coherency model.
    - Scratch base via SGPR pair (not HW register).
    - ENC_VOP3PX2 does not exist in CDNA1 XML.
    - The current runtime ISA trait leaves SRAM ECC disabled.
    """

    @property
    def d16_loads_zero_unselected_half(self) -> bool:
        return False

    @property
    def has_acc_vgpr(self) -> bool:
        return False

    @property
    def dpp_64bit_input_row_select_only(self) -> bool:
        return False

    @property
    def acc_vgpr_encoding_base(self) -> int:
        return 0

    @property
    def max_acc_vgprs(self) -> int:
        return 0

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 0

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 4

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

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del enc_name, src0_size_bits
        return _legacy_dpp_opcode_rule(inst_name)


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
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 0

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 8

    @property
    def dpp_64bit_input_row_select_only(self) -> bool:
        return False

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

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del enc_name, src0_size_bits
        return _legacy_dpp_opcode_rule(inst_name)


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
    def dpp_ctrl_dialect(self) -> DppCtrlDialect:
        return DppCtrlDialect.GFX10_PLUS

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        # RDNA1/2 uses a 6-bit lgkmcnt field at bits [13:8].
        return '0x3F'

    @property
    def supported_versions(self) -> list[str]:
        return ['1.0.0', '1.1.1']

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
    def supports_wgp_mode(self) -> bool:
        return True

    @property
    def descriptor_sgpr_count_encoded(self) -> bool:
        return False

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 8

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 4

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

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del enc_name, src0_size_bits
        return _legacy_dpp_opcode_rule(inst_name)


class Rdna2Profile(Rdna1Profile):
    """ISA profile for RDNA2 (GFX10.3, Navi2x).

    Inherits the RDNA1 Wave32 default and Wave64 maximum. DPP/SDWA variants
    remain skipped (``_SKIP_DPP_SDWA = True``).
    """


class Rdna3Profile(_AmdgpuProfileBase):
    """ISA profile for RDNA3 (GFX11, Navi3x).

    Key differences from RDNA1/2:

    - FLAT segment variants use ``GLOBAL`` instead of ``GLBL``
      (``ENC_FLAT_GLOBAL``, ``ENC_FLAT_SCRATCH``).
    - VOPDXY dual-issue encoding uses ``opx``/``opy`` fields instead of
      a single ``op`` field, which the parser cannot handle. The normal
      XML instruction generator skips these formats; ``gen_vopd`` emits the
      manual dual-slot implementation.
    - DPP support expanded to VOP3/VOP3P/VOPC/VOP3_SDST_ENC.
    - SDWA removed (no ``_VOP_SDWA`` variants).
    - ``ENC_LDSDIR`` and ``ENC_VINTERP`` replace CDNA's ``ENC_VINTRP``.

    XML bugs worked around:

    - VOPDXY dual-opcode format: uses ``opx``/``opy`` instead of ``op``,
      which breaks the parser's single-opcode assumption. Handled by
      ``gen_vopd`` after XML parsing skips normal instruction generation.
    - Reserved field omissions (version 1.0.0): synthesized by the parser.
    """

    _FLAT_SEGMENTS = frozenset({'GLOBAL', 'SCRATCH'})
    _SKIP_DPP_SDWA = True
    _SKIP = frozenset({'VOPDXY', 'VOPDXY_INST_LITERAL'})
    _SOP1_BASE_COND = 'Nothas_lit_0_Nothas_lit_1'

    def normalize_encoding_condition(self, enc_name: str, cond_name: str) -> str:
        if enc_name.upper() == 'ENC_SOP1' and cond_name == self._SOP1_BASE_COND:
            return 'default'
        return super().normalize_encoding_condition(enc_name, cond_name)

    def skip_inst_encoding(self, enc_name: str, enc_cond: str) -> bool:
        if enc_name.upper() == 'ENC_SOP1' and enc_cond == self._SOP1_BASE_COND:
            return False
        return super().skip_inst_encoding(enc_name, enc_cond)

    @property
    def dpp_bound_ctrl_applies_to_inactive_sources(self) -> bool:
        return True

    @property
    def dpp_suppressed_compare_lanes_zero(self) -> bool:
        return True

    @property
    def dpp_supports_row_xmask(self) -> bool:
        return True

    @property
    def dpp_supports_wave_controls(self) -> bool:
        return False

    @property
    def dpp_supports_row_broadcast_controls(self) -> bool:
        return False

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del src0_size_bits
        return _modern_rdna_dpp_opcode_rule(
            enc_name,
            inst_name,
            allow_dot2_vop3p=True,
            forbid_cvt_pk_f32_vop3=False,
            rdna4_vop1_exclusions=False,
        )

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
        return ['1.0.0', '1.1.0', '1.1.1']

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
    def supports_wgp_mode(self) -> bool:
        return True

    @property
    def compute_tmpring_wavesize_granule(self) -> int:
        return 256

    @property
    def compute_tmpring_wavesize_bits(self) -> int:
        return 15

    @property
    def descriptor_sgpr_count_encoded(self) -> bool:
        return False

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 8

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 4

    @property
    def waitcnt_family(self) -> str:
        return 'gfx11'

    @property
    def has_wmma(self) -> bool:
        return True

    @property
    def matrix_layout(self) -> MatrixLayout:
        return MatrixLayout.WMMA_REPLICATED_HALFWAVE

    @property
    def has_vopd(self) -> bool:
        return True

    @property
    def vopd_slot_ops(self) -> tuple[VopdSlotOp, ...]:
        return _RDNA3_VOPD_SLOT_OPS

    @property
    def vopd_x_slot_opcodes(self) -> frozenset[int]:
        return frozenset(range(14))

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX11_SC0_SC1_TH

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        # RDNA3/3.5 MubufMachineInst uses glc+slc (not sc0+sc1).
        return ('glc', 'slc', None)

    @property
    def uses_packed_16bit_e32_source_selectors(self) -> bool:
        # LLVM accepts gfx1100 E32 true16 operands such as
        # ``v_mov_b16_e32 v2.h, v0.l`` with vdst[7] selecting the high half.
        return True

    @property
    def scalar_null_precedes_m0(self) -> bool:
        return True

    @property
    def uses_true16_vop3_opsel(self) -> bool:
        return True

    @property
    def renders_true16_vop3_operands(self) -> bool:
        return True

    @property
    def vop3_opsel_omissions(self) -> frozenset[str]:
        return frozenset({'V_CNDMASK_B16'})

    @property
    def dpp_ctrl_dialect(self) -> DppCtrlDialect:
        return DppCtrlDialect.GFX10_PLUS

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

    @property
    def renders_gfx11_image_syntax(self) -> bool:
        return True

    @property
    def split_ds_2addr_offsets(self) -> bool:
        return True

    @property
    def vop3p_absolute_source_instructions(self) -> frozenset[str]:
        return frozenset({'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16'})

    @property
    def gfx11_mimg_gather_style_instructions(self) -> frozenset[str]:
        return frozenset({'IMAGE_MSAA_LOAD'})

    @property
    def gfx11_mimg_fixed_vdata_words(self) -> dict[str, int]:
        return {
            'IMAGE_BVH_INTERSECT_RAY': 4,
            'IMAGE_BVH64_INTERSECT_RAY': 4,
        }

    @property
    def gfx11_mimg_fixed_vaddr_words(self) -> dict[str, tuple[int, int]]:
        return {
            'IMAGE_BVH_INTERSECT_RAY': (11, 8),
            'IMAGE_BVH64_INTERSECT_RAY': (12, 9),
        }

    @property
    def gfx11_mimg_nsa_group_words(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        return {
            'IMAGE_BVH_INTERSECT_RAY': ((1, 1, 3, 3, 3), (1, 1, 3, 3)),
            'IMAGE_BVH64_INTERSECT_RAY': ((2, 1, 3, 3, 3), (2, 1, 3, 3)),
        }

    @property
    def sendmsg_return_symbolic(self) -> bool:
        return True


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
    - VOPDXY dual-issue encoding still uses the manual ``gen_vopd`` path.
    - Memory instruction mnemonics use ``B32``/``B64``/``B96``/``B128``
      suffixes instead of ``DWORD``/``DWORDX2``/``DWORDX3``/``DWORDX4``.
    - DS instructions use ``DS_LOAD_*``/``DS_STORE_*`` instead of
      ``DS_READ_*``/``DS_WRITE_*``.

    XML bugs worked around:

    - VOPDXY dual-opcode format: same parser issue as RDNA3, handled by
      ``gen_vopd`` after normal instruction generation skips it.
    """

    _SKIP_DPP_SDWA = True
    _SKIP = frozenset({'VOPDXY', 'VOPDXY_INST_LITERAL'})
    _SOP1_BASE_COND = 'Nothas_lit_0_Nothas_lit_1'

    def normalize_encoding_condition(self, enc_name: str, cond_name: str) -> str:
        if enc_name.upper() == 'ENC_SOP1' and cond_name == self._SOP1_BASE_COND:
            return 'default'
        return super().normalize_encoding_condition(enc_name, cond_name)

    def skip_inst_encoding(self, enc_name: str, enc_cond: str) -> bool:
        if enc_name.upper() == 'ENC_SOP1' and enc_cond == self._SOP1_BASE_COND:
            return False
        return super().skip_inst_encoding(enc_name, enc_cond)

    @property
    def dpp_bound_ctrl_applies_to_inactive_sources(self) -> bool:
        return True

    @property
    def dpp_suppressed_compare_lanes_zero(self) -> bool:
        return True

    @property
    def dpp_supports_row_xmask(self) -> bool:
        return True

    @property
    def dpp_supports_wave_controls(self) -> bool:
        return False

    @property
    def dpp_supports_row_broadcast_controls(self) -> bool:
        return False

    @property
    def dpp_requires_opsel_lane_alignment(self) -> bool:
        return False

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del src0_size_bits
        return _modern_rdna_dpp_opcode_rule(
            enc_name,
            inst_name,
            allow_dot2_vop3p=False,
            forbid_cvt_pk_f32_vop3=True,
            rdna4_vop1_exclusions=True,
        )

    @property
    def ds_b8_transpose_kind(self) -> int:
        return 6

    @property
    def waitcnt_lgkmcnt_mask(self) -> str:
        # RDNA4 removed S_WAITCNT; this property is unused but kept for
        # completeness. Returns 0x3F as a safe no-op default.
        return '0x3F'

    @property
    def waitcnt_decode(self) -> str:
        """GFX12 compatibility S_WAITCNT SIMM16 layout.

        RDNA4 XML exposes split S_WAIT_* opcodes, but LLVM still accepts the
        monolithic opcode-9 S_WAITCNT form. The injected compatibility opcode
        uses the GFX11 bit layout.
        """
        return Rdna3Profile.waitcnt_decode.fget(self)

    @property
    def compatibility_instruction_slots(self) -> dict[tuple[str, int], str]:
        return {('ENC_SOPP', 9): 'S_WAITCNT'}

    @property
    def supported_versions(self) -> list[str]:
        return ['1.1.0', '1.1.1']

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
    def supports_wgp_mode(self) -> bool:
        return True

    @property
    def uses_ttmp_workgroup_ids(self) -> bool:
        return True

    @property
    def wave_state_layout(self) -> WaveStateLayout:
        return WaveStateLayout.GFX12

    @property
    def compute_tmpring_wavesize_granule(self) -> int:
        return 256

    @property
    def compute_tmpring_wavesize_bits(self) -> int:
        return 18

    @property
    def descriptor_sgpr_count_encoded(self) -> bool:
        return False

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        return 8

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 4

    @property
    def waitcnt_family(self) -> str:
        return 'gfx12'

    @property
    def has_wmma(self) -> bool:
        return True

    @property
    def matrix_layout(self) -> MatrixLayout:
        return MatrixLayout.WMMA_SPLIT_K

    @property
    def swmmac_layout(self) -> SwmmacLayout:
        return SwmmacLayout.RUNTIME_WAVE

    @property
    def has_vopd(self) -> bool:
        return True

    @property
    def vopd_slot_ops(self) -> tuple[VopdSlotOp, ...]:
        return _RDNA4_VOPD_SLOT_OPS

    @property
    def vopd_x_slot_opcodes(self) -> frozenset[int]:
        return frozenset(range(14))

    @property
    def coherency_model(self) -> MemoryCoherencyModel:
        return MemoryCoherencyModel.GFX12_SCOPE_TH

    @property
    def uses_packed_16bit_e32_source_selectors(self) -> bool:
        return True

    @property
    def scalar_null_precedes_m0(self) -> bool:
        return True

    @property
    def uses_true16_vop3_opsel(self) -> bool:
        return True

    @property
    def semantic_overrides(self) -> dict[str, tuple[str, ...]]:
        return {
            'S_BARRIER_SIGNAL': ('true_nop', '', ''),
            'S_BARRIER_SIGNAL_ISFIRST': ('true_nop', '', ''),
            'S_BARRIER_WAIT': ('barrier', '', ''),
        }

    @property
    def renders_true16_vop3_operands(self) -> bool:
        return True

    @property
    def vop3_opsel_omissions(self) -> frozenset[str]:
        return frozenset({'V_CNDMASK_B16'})

    @property
    def dpp_ctrl_dialect(self) -> DppCtrlDialect:
        return DppCtrlDialect.GFX10_PLUS

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

    def saddr_null_selector_expr(self, enc_name: str) -> str | None:
        """GFX12 VFLAT/VGLOBAL use the architectural scalar NULL value."""
        if enc_name.upper() in ('ENC_VFLAT', 'ENC_VGLOBAL'):
            return 'OPR_SREG_NULL'
        return super().saddr_null_selector_expr(enc_name)

    @property
    def coherency_field_names(self) -> tuple[str, str, str | None]:
        # RDNA4 VbufferMachineInst/VflatMachineInst use nv (no sc0/sc1).
        # mtype_from_bits is called with (nv, 0) as a placeholder.
        return ('nv', 'nv', None)

    @property
    def vop3p_opsel_fields(self) -> tuple[str, str]:
        return ('opsel', 'opsel_hi')

    @property
    def vop3p_opsel_hi_high_field(self) -> str:
        return 'opsel_hi_2'

    @property
    def vop3_opsel_field(self) -> str:
        return 'opsel'

    @property
    def smem_direct_offset_field(self) -> str | None:
        return 'ioffset'

    @property
    def flat_store_src_field(self) -> str:
        return 'vsrc'

    def encoding_modifiers(self, enc_name: str) -> list[EncodingModifier]:
        """RDNA4 encoding modifiers.

        Render the GFX12 SCOPE+TH policy and NV flag for every memory encoding
        that carries those fields.
        """
        upper = enc_name.upper()
        if upper == 'ENC_SMEM':
            return _SMEM_MODIFIERS_RDNA4
        if upper in ('ENC_VBUFFER', 'ENC_MUBUF'):
            return _VBUFFER_MODIFIERS_RDNA4
        if upper in ('ENC_VFLAT', 'ENC_VGLOBAL', 'ENC_VSCRATCH'):
            return _VFLAT_MODIFIERS_RDNA4
        if upper in ('ENC_VIMAGE', 'ENC_VSAMPLE'):
            return _IMAGE_MODIFIERS_RDNA4
        return []


class Cdna5Profile(Rdna4Profile):
    """ISA profile for gfx1250.

    The gfx1250 encoding model is RDNA4/GFX12-like. Use ``cdna5`` as the
    logical target used by parser/codegen rules while generated and handwritten
    C++ lives under ``amdgpu/cdna5`` in the ``cdna5`` namespace.
    """

    @property
    def generated_arch_name(self) -> str | None:
        return 'cdna5'

    @property
    def generated_dir_name(self) -> str | None:
        return 'cdna5'

    @property
    def cpp_namespace(self) -> str | None:
        return 'cdna5'

    @property
    def ds_b8_transpose_kind(self) -> int:
        # gfx1250 groups source lanes by four rows and two eight-column halves.
        # Keep every mnemonic alias on the corresponding CDNA5 DS routing.
        return 7

    @property
    def semantic_overrides(self) -> dict[str, tuple[str, ...]]:
        overrides = dict(super().semantic_overrides)
        for mnemonic in (
            'S_BARRIER_SIGNAL',
            'S_BARRIER_SIGNAL_ISFIRST',
            'S_BARRIER_WAIT',
        ):
            overrides.pop(mnemonic, None)
        return overrides

    @property
    def semantic_class_overrides(self) -> dict[str, str]:
        return {
            'DS_STORE_ADDTID_B32': 'ds_write_addtid',
            'DS_STOREXCHG_2ADDR_RTN_B32': 'ds_atomic2',
            'DS_STOREXCHG_2ADDR_RTN_B64': 'ds_atomic2',
            'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32': 'ds_atomic2',
            'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64': 'ds_atomic2',
        }

    @property
    def ds_addtid_uses_m0_byte_base(self) -> bool:
        return True

    @property
    def ds_transpose_ignores_exec(self) -> bool:
        return True

    _SKIP = frozenset(
        {
            'ENC_VOP3PX2',
            'ENC_VOP3PX3',
            'VOPD3XY',
            'VOPDXY_X',
            'VOPDXY_INST_LITERAL_X',
            'VOPDXY_Y',
            'VOPDXY_INST_LITERAL_Y',
        }
    )

    _SOP1_BASE_COND = '!has_lit64_0&!has_lit64_1&!has_lit_0&!has_lit_1'

    # MI400 Table 55: these DPMACC operations may use the 0x150-0x15f
    # row-select controls (named DPP_ROW_SHARE there) but no other DPP16
    # control. DPP8 has no corresponding form and is therefore rejected.
    _DPP_ROW_SELECT_ONLY = frozenset(
        {
            'V_CVT_I32_F64',
            'V_CVT_F64_I32',
            'V_CVT_F32_F64',
            'V_CVT_F64_F32',
            'V_CVT_U32_F64',
            'V_CVT_F64_U32',
            'V_TRUNC_F64',
            'V_CEIL_F64',
            'V_RNDNE_F64',
            'V_FLOOR_F64',
            'V_FREXP_EXP_I32_F64',
            'V_FREXP_MANT_F64',
            'V_FRACT_F64',
            'V_FMA_F64',
            'V_FMAC_F64',
            'V_DIV_FIXUP_F64',
            'V_DIV_FMAS_F64',
            'V_DIV_SCALE_F64',
            'V_MAD_NC_U64_U32',
            'V_MAD_NC_I64_I32',
            'V_MAD_CO_U64_U32',
            'V_MAD_CO_I64_I32',
            'V_ADD_F64',
            'V_MUL_F64',
            'V_MINIMUM_F64',
            'V_MAXIMUM_F64',
            'V_MIN_NUM_F64',
            'V_MAX_NUM_F64',
            'V_LDEXP_F64',
            'V_MUL_LO_U32',
            'V_MUL_HI_U32',
            'V_MUL_HI_I32',
            'V_MAD_U32',
            'V_LSHLREV_B64',
            'V_LSHRREV_B64',
            'V_ASHRREV_I64',
            'V_MOV_B64',
            'V_LSHL_ADD_U64',
            'V_ADD_NC_U64',
            'V_SUB_NC_U64',
            'V_MAX_I64',
            'V_MAX_U64',
            'V_MIN_I64',
            'V_MIN_U64',
        }
    )

    _DPP_FORBIDDEN_64 = frozenset(
        {
            'V_FMAAK_F64',
            'V_FMAMK_F64',
            'V_TRIG_PREOP_F64',
            'V_MUL_U64',
            'V_SQRT_F64',
            'V_RCP_F64',
            'V_RSQ_F64',
        }
    )

    def normalize_encoding_condition(self, enc_name: str, cond_name: str) -> str:
        if enc_name.upper() == 'ENC_SOP1' and cond_name == self._SOP1_BASE_COND:
            return 'default'
        return super().normalize_encoding_condition(enc_name, cond_name)

    def skip_inst_encoding(self, enc_name: str, enc_cond: str) -> bool:
        if enc_name.upper() == 'ENC_SOP1' and enc_cond == self._SOP1_BASE_COND:
            return False
        return super().skip_inst_encoding(enc_name, enc_cond)

    def dpp_opcode_rule(
        self,
        enc_name: str,
        inst_name: str,
        *,
        src0_size_bits: int | None = None,
    ) -> DppOpcodeRule:
        del src0_size_bits
        name = inst_name.upper()
        enc = enc_name.upper()
        if name in self._DPP_ROW_SELECT_ONLY:
            return DppOpcodeRule.ROW_SELECT_ONLY
        if name in self._DPP_FORBIDDEN_64 or (
            name.startswith('V_CMP') and _opcode_uses_64bit_data(name)
        ):
            return DppOpcodeRule.FORBID
        if _opcode_uses_64bit_data(name):
            return DppOpcodeRule.FORBID
        if enc == 'ENC_VOP3' and name.startswith(('V_CVT_PK_F16_', 'V_CVT_SCALE_')):
            return DppOpcodeRule.FORBID
        return super().dpp_opcode_rule(enc_name, inst_name)

    def field_renames(self, enc_name: str) -> dict[str, str]:
        renames = dict(super().field_renames(enc_name))
        renames['literal'] = 'simm32'
        return renames

    @property
    def compatibility_instruction_slots(self) -> dict[tuple[str, int], str]:
        return {('ENC_VOP1', 103): 'V_PERMLANE64_B32'}

    def normalize_operand_field_name(self, enc_name: str, field_name: str) -> str:
        # Keep the concrete gfx1250 operand identity distinct from earlier
        # CDNA/RDNA specs; its local generator path maps it to the renamed
        # microcode member and must not join simm32-specific shared bodies.
        return field_name

    @property
    def lowercase_operand_selector_names(self) -> bool:
        # CDNA5 schema 1.2.0 intentionally uses uppercase symbolic names and
        # the existing gfx1250 disassembly contract follows those spellings.
        return False

    @property
    def supported_versions(self) -> list[str]:
        return ['1.2.0']

    @property
    def wave_size_max(self) -> int:
        return 32

    @property
    def swmmac_layout(self) -> SwmmacLayout:
        return SwmmacLayout.FIXED_WAVE

    @property
    def vop3_cmp_sdst_size_bits(self) -> int | None:
        return 32

    @property
    def vop3_cndmask_selector_size_bits(self) -> int | None:
        return 32

    @property
    def vop3_carry_mask_size_bits(self) -> int | None:
        return 32

    @property
    def supports_wgp_mode(self) -> bool:
        return False

    @property
    def uses_cluster_ttmp_workgroup_ids(self) -> bool:
        return True

    @property
    def wave_state_layout(self) -> WaveStateLayout:
        return WaveStateLayout.GFX12_5

    @property
    def max_addressable_vgprs_per_wf(self) -> int:
        return 1024

    @property
    def descriptor_vgpr_count_granule_wave32(self) -> int:
        # Matches LLVM AMDGPUBaseInfo::getVGPREncodingGranule():
        # gfx1250 has Feature1024AddressableVGPRs, so Wave32 descriptors
        # encode VGPR counts in 16-register blocks. The separate
        # s_set_vgpr_msb high-bank indexing needed above v255 is modeled by
        # uses_vgpr_msb_indexing.
        return 16

    @property
    def descriptor_vgpr_count_granule_wave64(self) -> int:
        return 0

    @property
    def vopd_encoding_prefixes(self) -> tuple[VopdEncodingPrefix, ...]:
        return super().vopd_encoding_prefixes + (
            VopdEncodingPrefix(0xCF, 8, is_vopd3=True),
        )

    @property
    def vopd_slot_ops(self) -> tuple[VopdSlotOp, ...]:
        return _CDNA5_VOPD_SLOT_OPS

    @property
    def vopd_x_slot_opcodes(self) -> frozenset[int]:
        return frozenset(range(12))

    @property
    def vopd_y_slot_opcodes(self) -> frozenset[int]:
        return frozenset((*range(12), 16, 17, *range(20, 25)))

    @property
    def vopd3_x_slot_opcodes(self) -> frozenset[int]:
        return frozenset((0, *range(3, 12), 16, 17, *range(19, 23), *range(32, 37)))

    @property
    def vopd3_y_slot_opcodes(self) -> frozenset[int]:
        return frozenset((0, *range(3, 12), *range(16, 25)))

    @property
    def uses_vgpr_msb_indexing(self) -> bool:
        return True

    @property
    def d16_loads_zero_unselected_half(self) -> bool:
        return True

    @property
    def uses_packed_16bit_e32_source_selectors(self) -> bool:
        return True

    @property
    def vbuffer_store_data_uses_dst_vgpr_msb_role(self) -> bool:
        return True

    @property
    def buffer_payload_reads_use_effective_exec_mask(self) -> bool:
        return True

    @property
    def generate_scaled_wmma_vop3px2(self) -> bool:
        return True

    @property
    def smem_address_uses_access_size(self) -> bool:
        return True

    @property
    def source_split_max_bytes(self) -> dict[str, int]:
        # Keep generated gfx1250 instruction sources under the repository's
        # added-file size hook without changing the hook policy for all users.
        # clang-format expands constructor-heavy generated sources, so leave
        # enough room below the hook instead of targeting the hook limit itself.
        return {
            'ENC_VOP3': 450 * 1024,
            'ENC_VOPC': 450 * 1024,
        }

    def source_split_file_stem(
        self, enc_name: str, inst_name: str, semantics: object | None
    ) -> str | None:
        enc = enc_name.upper()
        name = inst_name.upper()
        sem_class = getattr(semantics, 'semantic_class', None)

        if enc == 'ENC_VOPC':
            return self._vopc_source_split_file_stem(name)
        if enc == 'ENC_VOP3':
            return self._vop3_source_split_file_stem(name, sem_class)
        return None

    @staticmethod
    def _vopc_source_split_file_stem(inst_name: str) -> str:
        if inst_name.startswith('V_CMPX_'):
            return 'cmpx'
        if inst_name.startswith('V_CMP_'):
            return 'cmp'
        return 'misc'

    @staticmethod
    def _vop3_source_split_file_stem(
        inst_name: str,
        sem_class: str | None,
    ) -> str:
        if inst_name.startswith(('V_CMPX_', 'V_CMP_')):
            return 'cmp'

        if inst_name.startswith(('V_CVT_', 'V_PACK_', 'V_FREXP_')):
            return 'cvt'

        if inst_name.startswith('V_DIV_'):
            return 'alu'
        if inst_name.startswith(('V_PERM', 'V_CUBE')):
            return 'data'
        if inst_name in ('V_READFIRSTLANE_B32', 'V_READLANE_B32', 'V_WRITELANE_B32'):
            return 'data'
        # Carry-out instructions are grouped with ALU even when their generic
        # semantic class is a multiply-add family.
        if '_CO_' in inst_name:
            return 'alu'

        if sem_class == 'vector_cvt_scale':
            return 'cvt'
        if sem_class in {
            'vector_cvt_pk',
            'vector_cvt_pknorm',
            'vector_cvt_pk_u8_f32',
            'vector_cvt_pkrtz_f16_f32',
            'vector_cvt_pk_f16_f32',
            'vector_cvt_pk_bf16_f32',
            'vector_cvt_sr_f16_f32',
            'vector_cvt_sr_bf16_f32',
            'vector_pack_b32_f16',
        }:
            return 'cvt'
        if sem_class in {'vector_readfirstlane', 'vector_readlane', 'vector_writelane'}:
            return 'data'
        if sem_class in {
            'vector_permlane16',
            'vector_permlanex16',
            'vector_permlane16_swap',
            'vector_permlane32_swap',
            'vector_permlane64',
        }:
            return 'data'
        if sem_class in {'vector_div_fixup', 'vector_div_scale', 'vector_div_fmas'}:
            return 'alu'
        if sem_class in {'vector_mad_32_16', 'vector_mad_64_32'}:
            return 'ternary'
        if sem_class in {'vector_mbcnt', 'vector_bitop3'}:
            return 'alu'
        if sem_class == 'vector_add_co':
            return 'alu'
        if sem_class == 'vector_binop':
            return 'alu'
        if sem_class == 'vector_ternary':
            return 'ternary'
        if sem_class in {
            'vector_cndmask',
            'vector_mov',
            'vector_movrel',
        }:
            return 'data'
        if sem_class in {'vector_dot', 'vector_dot2c_bf16'}:
            return 'alu'
        if sem_class in {'pseudo_scalar_unary', 'vector_unary'}:
            return 'alu'
        if sem_class in {'nop', 'true_nop'}:
            return 'misc'

        return 'misc'
