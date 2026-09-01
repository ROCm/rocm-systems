# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amdisa.isa_profile import IsaProfile


@dataclass
class MicrocodeField:
    """A bitfield in an encoding's microcode.

    Attributes:
        name: Name of the field.
        bit_cnt: Size of the field in bits.
        bit_offset: Offset, in bits, of the field within the encoding.
    """

    name: str
    bit_cnt: int
    bit_offset: int


#: Base C++ identifier synthesized for a fieldless operand of each operand
#: type (fieldless operands have no ``<FieldName>`` in the MR ISA to derive a
#: name from). Bases are deliberately distinct per type so the name reflects
#: which operand it is -- in particular ``OPR_EXEC`` (``exec``) and
#: ``OPR_SDST_EXEC`` (``sdst_exec``) are kept apart. This is a readability
#: choice, not a correctness one. Any type not listed falls back to the type
#: name with the ``OPR_`` prefix stripped and lowercased (see
#: ``synthesize_fieldless_name``).
_FIELDLESS_NAME_MAP: dict[str, str] = {
    'OPR_VCC': 'vcc',
    'OPR_EXEC': 'exec',
    'OPR_SDST_EXEC': 'sdst_exec',
    'OPR_SSRC_SPECIAL_SCC': 'scc',
    'OPR_PC': 'pc',
    'OPR_SDST_M0': 'm0',
    'OPR_DSMEM': 'dsmem',
    'OPR_GPUMEM': 'gpumem',
    'OPR_FLAT_SCRATCH': 'flat_scratch',
    # Fieldless OPR_VGPR is the image ADDRESS/coordinate operand (vaddr).
    'OPR_VGPR': 'vaddr',
    'OPR_SIMM32': 'simm32',
}


def format_encoding_name(enc_name: str) -> str:
    """Format an MR-ISA encoding name as a C++ PascalCase fragment."""
    parts = enc_name.split('_')
    if parts[0] == 'ENC':
        parts = parts[1:]
    return ''.join(part.capitalize() for part in parts)


def format_true_encoding_name(
    enc_name: str, *, is_implied_literal_enc: bool = False
) -> str:
    """Format the concrete C++ encoding class used by an instruction."""
    if is_implied_literal_enc:
        return enc_name.split('_')[0].capitalize()
    return format_encoding_name(enc_name)


def format_instruction_name(
    name: str, enc_name: str, *, is_implied_literal_enc: bool = False
) -> str:
    """Format an instruction's generated C++ class/decode-function name."""
    mnemonic = ''.join(part.capitalize() for part in name.split('_'))
    encoding = format_true_encoding_name(
        enc_name, is_implied_literal_enc=is_implied_literal_enc
    )
    return f'{mnemonic}{encoding}'


def opcode_name_fragment(token: str) -> str:
    """Return one C++ opcode-constant fragment for a mnemonic token."""
    token = token.lower()
    if token.endswith('exec') and len(token) > len('exec'):
        return f'{token[:-4].capitalize()}Exec'
    if token.endswith('pc') and len(token) > len('pc'):
        return f'{token[:-2].capitalize()}Pc'
    return token.capitalize()


def opcode_constant_base_name(mnemonic: str) -> str:
    """Return the generated C++ opcode-constant base for a mnemonic."""
    return 'k' + ''.join(
        opcode_name_fragment(token) for token in mnemonic.lower().split('_') if token
    )


def opcode_constant_name(
    mnemonic: str, enc_name: str, *, is_implied_literal_enc: bool = False
) -> str:
    """Return the concrete encoding-qualified C++ opcode-constant name."""
    return opcode_constant_base_name(mnemonic) + format_true_encoding_name(
        enc_name, is_implied_literal_enc=is_implied_literal_enc
    )


def synthesize_fieldless_name(operand_type: str) -> str:
    """Return a stable, valid C++ base identifier for a fieldless operand.

    Fieldless operands (no ``<FieldName>`` in the MR ISA) cannot derive their
    name from an encoding field, so one is synthesized from the operand type.
    The result is a *base* name; callers must still guarantee uniqueness within
    an instruction (two fieldless operands can share a type). See
    ``_FIELDLESS_NAME_MAP`` for the curated bases; anything else strips the
    ``OPR_`` prefix and lowercases.
    """
    if operand_type in _FIELDLESS_NAME_MAP:
        return _FIELDLESS_NAME_MAP[operand_type]
    base = operand_type
    if base.startswith('OPR_'):
        base = base[len('OPR_') :]
    return base.lower()


@dataclass
class Operand:
    """An instruction operand.

    Attributes:
        name: Name of the operand (``<FieldName>`` ? name : synthesized id).
        size: Size of the operand in bits.
        operand_type: ISA-specific operand type.
        is_input: True if the operand is an input.
        is_output: True if the operand is an output.
        is_implicit: True if the operand is implicit (the MR ISA
            ``IsImplicit`` attribute)
        is_binary_ucode_required: True if the operand is missing from the
            encoding but implied by the type of the operand.
        order: Order of the operand.
        data_format_name: ISA data format name from the XML operand signature.
        fieldless: True if the operand has no ``<FieldName>`` in the MR ISA.
    """

    name: str
    size: int
    operand_type: str
    is_input: bool
    is_output: bool
    is_implicit: bool
    is_binary_ucode_required: bool
    order: int
    data_format_name: str = ''
    fieldless: bool = False


@dataclass
class OperandNamePattern:
    """Pattern for resolving operand encoding values to operand name strings.

    Attributes:
        kind: Pattern type (one of the class constants below).
        prefix: Register prefix for reg_range patterns (e.g., "s", "v").
        min_enum: OpSel enum name for the minimum of a range.
        max_enum: OpSel enum name for the maximum of a range.
        operand_name: Operand name string for named/float_const patterns.
        enum_name: OpSel enum name for single-value patterns.
    """

    REG_RANGE = 'reg_range'
    POS_INT = 'pos_int'
    NEG_INT = 'neg_int'
    FLOAT_CONST = 'float_const'
    NAMED = 'named'
    LITERAL = 'literal'

    kind: str
    prefix: str = ''
    min_enum: str = ''
    max_enum: str = ''
    operand_name: str = ''
    enum_name: str = ''


@dataclass
class OperandSelector:
    """Operand encoding values that determine how to select the operand value.

    Attributes:
        operand_type: Operand's type.
        op_sel_vals: Operand value names and their associated values.
        name_patterns: Patterns for resolving encoding values to operand name
            strings.
    """

    operand_type: str
    op_sel_vals: list[tuple[str, str]]
    name_patterns: list[OperandNamePattern] = field(default_factory=list)


class InstBase:
    """Base class for instruction and encoding objects.

    Provides encoding name formatting shared by InstEncoding and Instruction.

    Attributes:
        enc_name: Name of the encoding.
        is_implied_literal_enc: True if this encoding is an implied-literal
            alternate. Used by ``fmt_true_enc_name`` to resolve the display
            name back to the parent encoding format.
    """

    def __init__(self, enc_name: str, is_implied_literal_enc: bool = False) -> None:
        self.enc_name = enc_name
        self.is_implied_literal_enc = is_implied_literal_enc

    @cached_property
    def fmt_enc_name(self) -> str:
        """Encoding name formatted to C++ PascalCase style."""
        return format_encoding_name(self.enc_name)

    @cached_property
    def fmt_true_enc_name(self) -> str:
        """True encoding name, resolving implied-literal alternate encodings.

        Implied-literal alternates (e.g., ``VOP2_INST_LITERAL``) format as
        their parent encoding name (e.g., ``Vop2``) because the C++ class
        hierarchy inherits from the parent encoding class.
        """
        return format_true_encoding_name(
            self.enc_name, is_implied_literal_enc=self.is_implied_literal_enc
        )


class InstEncoding(InstBase):
    """An encoding in the ISA specification.

    Attributes:
        is_primary_decode: True if decoded from the primary decode table.
        primary_dt_ptrs: Pointers to entries in the primary decode table.
        order: Decoding order for this encoding.
        bit_cnt: Size in bits of the encoding.
        enc_field_bit_cnt: Size in bits of the encoding field.
        op_field_bit_cnt: Size in bits of the opcode field.
        ucode_fields: All microcode fields in this encoding.
        enc_conds: Encoding condition (name, logic) pairs.
        insts: Instructions encoded under this encoding.
        implied_literal_ops: Opcode-to-extension-DWORD-count mapping for
            instructions whose encoding carries an implied literal.
    """

    def __init__(
        self,
        name: str,
        order: int,
        bit_cnt: int,
        enc_field_bit_cnt: int,
        op_field_bit_cnt: int,
        ucode_fields: list[MicrocodeField],
        enc_conds: list[tuple[str, str]],
    ) -> None:
        super().__init__(name)
        self.is_primary_decode = True
        self.primary_dt_ptrs: list[int] | None = None
        self.order = order
        self.bit_cnt = bit_cnt
        self.enc_field_bit_cnt = enc_field_bit_cnt
        self.op_field_bit_cnt = op_field_bit_cnt
        self.opm_field_bit_cnt = 0
        self.ucode_fields = ucode_fields
        self.enc_conds = enc_conds
        self.insts: list[Instruction] = []
        self.implied_literal_ops: dict[str, int] = {}

    @cached_property
    def has_implied_literal_ops(self) -> bool:
        """True if this encoding has any implied literal opcodes."""
        return len(self.implied_literal_ops) > 0

    @cached_property
    def has_variable_implied_literal_size(self) -> bool:
        """True if an implied literal is not the usual one DWORD."""
        return any(words != 1 for words in self.implied_literal_ops.values())


class Instruction(InstBase):
    """An AMD GPU ISA instruction.

    Attributes:
        name: Name of the instruction.
        opcode: Opcode of the instruction.
        operands: The instruction's operands.
        available_encodings: All encoding names listed for this instruction in
            the machine-readable ISA, including alternate DPP, SDWA, and
            literal forms that are not emitted as independent instruction
            classes. ``None`` means encoding provenance is unknown; modifier
            generation rejects that state, so synthetic VOP instructions must
            provide an explicit set.
        source_addition: Additions provenance for a repository-supplied instruction,
            or ``None`` when the instruction came from the base MR ISA XML.
    """

    def __init__(
        self,
        name: str,
        enc_name: str,
        opcode: int,
        operands: list[Operand],
        is_implied_literal_enc: bool = False,
        available_encodings: frozenset[str] | None = None,
        source_addition: IsaAdditionProvenance | None = None,
    ) -> None:
        super().__init__(enc_name, is_implied_literal_enc)
        self.name = name
        self.opcode = opcode
        self.operands = operands
        self.available_encodings = available_encodings
        self.source_addition = source_addition
        # Populated by an optional ISA-variant manifest after the base XML and
        # additive deltas have been parsed.  The generator copies these masks
        # into decoded instructions; the runtime decoder then fails closed when
        # its concrete target lacks a required feature.
        self.required_feature_mask = 0
        self.encoding_feature_masks: dict[str, int] = {}
        # A model-only instruction is decoded and represented, but deliberately
        # omitted from generated execution callback tables.
        self.model_only = False

    @cached_property
    def fmt_name(self) -> str:
        """Instruction name formatted to C++ PascalCase style."""
        return format_instruction_name(
            self.name,
            self.enc_name,
            is_implied_literal_enc=self.is_implied_literal_enc,
        )

    @cached_property
    def mnemonic(self) -> str:
        """Instruction mnemonic (lowercase name)."""
        return self.name.lower()

    @property
    def implicit_operands(self) -> list[Operand]:
        """Subset of ``operands`` that are implicit (``is_implicit=True``)."""
        return [op for op in self.operands if op.is_implicit]

    @property
    def explicit_operands(self) -> list[Operand]:
        """Subset of ``operands`` that are explicit (``is_implicit=False``)."""
        return [op for op in self.operands if not op.is_implicit]

    @property
    def src_operands(self) -> list[Operand]:
        """Subset of ``operands`` that are inputs (``is_input=True``).

        Includes fieldless inputs, so this is NOT the execute-visible source
        set: the code generator filters those through
        ``CodeGenerator._execute_operand_participates``.
        """
        return [op for op in self.operands if op.is_input]

    @property
    def dst_operands(self) -> list[Operand]:
        """Subset of ``operands`` that are outputs (``is_output=True``).

        Includes fieldless outputs (e.g. the SCC/EXEC side-effect defs), so like
        :attr:`src_operands` this is NOT the execute-visible destination set. No
        in-tree consumer yet; filter via
        ``CodeGenerator._execute_operand_participates`` for the execute set.
        """
        return [op for op in self.operands if op.is_output]

    def has_implicit_operand(self, operand_type: str) -> bool:
        """True if any implicit operand has the given ``operand_type``."""
        return any(op.operand_type == operand_type for op in self.implicit_operands)


@dataclass
class DecodeTableEntry:
    """An entry in the primary decode table.

    Attributes:
        enc: Instruction encoding that maps to this entry.
        num_dupe_entries: Number of duplicate entries due to don't-care bits.
        inst_name: Name of the instruction if directly decodable.
        is_primary: True if this entry can directly decode an instruction.
        sub_decode_table: Sub-decode table name (if not primary).
        sub_decode_funcs: Sub-decoder function names (if not primary).
        decode_func: Name of this entry's decode function.
    """

    enc: InstEncoding
    num_dupe_entries: int
    inst_name: str | None = None
    is_primary: bool = True
    sub_decode_table: str | None = None
    sub_decode_funcs: list[str] | None = None
    decode_func: str | None = None


@dataclass(frozen=True)
class IsaAdditionProvenance:
    """Identity and source path for one applied ISA additions document."""

    identifier: str
    path: str


class IsaSpec:
    """Internal representation of a machine-readable ISA spec.

    Holds the data extracted during parsing. ISA-specific constants and
    encoding rules are delegated to the ``IsaProfile`` (see
    ``isa_profile.py``). XML schema constants (element/attribute names)
    live in ``xml_schema.py``.

    Attributes:
        profile: ISA-specific encoding rules and constants.
        arch_name: Logical architecture name used by parser/codegen rules.
        cpp_namespace: Namespace used by generated C++ declarations.
        generated_dir_name: Filesystem directory for generated and handwritten
            architecture files.
        version: Schema version string.
        encoding_map: Maps encoding names to InstEncoding objects.
        inst_encodings: All encodings parsed from the spec.
        operand_types: All operand type names.
        opnd_selectors: All operand selectors.
        primary_decode_table: The primary decode table (indexed by the top
            ``profile.max_enc_bits`` bits of the instruction word).
        alt_encs_with_implied_literal: Alternate encoding names whose
            encoding conditions indicate an implied literal DWORD.
            Populated during parsing using the profile's
            ``is_implied_literal_encoding()`` method.
        applied_additions: Ordered provenance for ISA additions documents
            merged into this specification.
    """

    def __init__(
        self,
        arch_name: str,
        version: str,
        profile: IsaProfile,
        generated_dir_name: str | None = None,
        cpp_namespace: str | None = None,
    ) -> None:
        self.profile = profile
        self.arch_name = arch_name
        self.generated_dir_name = generated_dir_name or arch_name
        self.cpp_namespace = cpp_namespace or arch_name
        self.version = version
        self.encoding_map: dict[str, InstEncoding] = {}
        self.inst_encodings: list[InstEncoding] = []
        self.operand_types: list[str] = []
        self.opnd_selectors: list[OperandSelector] = []
        self.primary_decode_table: list[DecodeTableEntry | None] = [None] * pow(
            2, profile.max_enc_bits
        )
        self.alt_encs_with_implied_literal: set[str] = set()
        self.applied_additions: tuple[IsaAdditionProvenance, ...] = ()
        self.isa_features: tuple[str, ...] = ()
        self.isa_variants: dict[str, int] = {}
        # Every fieldless operand type observed while parsing this spec.
        # Consumed by fieldless_policy.validate_fieldless_taxonomy.
        self.fieldless_operand_types: set[str] = set()
        if self.version not in profile.supported_versions:
            raise ValueError(
                f'Unsupported machine-readable ISA spec version: {self.version}'
            )
