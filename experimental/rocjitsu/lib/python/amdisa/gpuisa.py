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


@dataclass
class Operand:
    """An instruction operand.

    Attributes:
        name: Name of the operand.
        size: Size of the operand in bits.
        operand_type: ISA-specific operand type.
        is_input: True if the operand is an input.
        is_output: True if the operand is an output.
        is_implicit: True if the operand is implicit.
        is_binary_ucode_required: True if the operand is missing from the
            encoding but implied by the type of the operand.
        order: Order of the operand.
    """

    name: str
    size: int
    operand_type: str
    is_input: bool
    is_output: bool
    is_implicit: bool
    is_binary_ucode_required: bool
    order: int


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

    def __init__(
        self, enc_name: str, is_implied_literal_enc: bool = False
    ) -> None:
        self.enc_name = enc_name
        self.is_implied_literal_enc = is_implied_literal_enc

    @cached_property
    def fmt_enc_name(self) -> str:
        """Encoding name formatted to C++ PascalCase style."""
        if self.enc_name.split('_')[0] == 'ENC':
            return ''.join(
                x.capitalize() for x in self.enc_name.split('_')[1:]
            )
        return ''.join(x.capitalize() for x in self.enc_name.split('_'))

    @cached_property
    def fmt_true_enc_name(self) -> str:
        """True encoding name, resolving implied-literal alternate encodings.

        Implied-literal alternates (e.g., ``VOP2_INST_LITERAL``) format as
        their parent encoding name (e.g., ``Vop2``) because the C++ class
        hierarchy inherits from the parent encoding class.
        """
        if self.is_implied_literal_enc:
            return self.enc_name.split('_')[0].capitalize()
        return self.fmt_enc_name


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
        implied_literal_ops: Opcodes with an implied literal (second DWORD).
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
        self.implied_literal_ops: list[str] = []

    @cached_property
    def has_implied_literal_ops(self) -> bool:
        """True if this encoding has any implied literal opcodes."""
        return len(self.implied_literal_ops) > 0


class Instruction(InstBase):
    """An AMD GPU ISA instruction.

    Attributes:
        name: Name of the instruction.
        opcode: Opcode of the instruction.
        operands: The instruction's operands.
    """

    def __init__(
        self,
        name: str,
        enc_name: str,
        opcode: int,
        operands: list[Operand],
        is_implied_literal_enc: bool = False,
    ) -> None:
        super().__init__(enc_name, is_implied_literal_enc)
        self.name = name
        self.opcode = opcode
        self.operands = operands

    @cached_property
    def fmt_name(self) -> str:
        """Instruction name formatted to C++ PascalCase style."""
        return (
            f'{"".join(x.capitalize() for x in self.name.split("_"))}'
            f'{self.fmt_true_enc_name}'
        )

    @cached_property
    def mnemonic(self) -> str:
        """Instruction mnemonic (lowercase name)."""
        return self.name.lower()


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


class IsaSpec:
    """Internal representation of a machine-readable ISA spec.

    Holds the data extracted during parsing. ISA-specific constants and
    encoding rules are delegated to the ``IsaProfile`` (see
    ``isa_profile.py``). XML schema constants (element/attribute names)
    live in ``xml_schema.py``.

    Attributes:
        profile: ISA-specific encoding rules and constants.
        arch_name: Architecture name.
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
    """

    def __init__(
        self, arch_name: str, version: str, profile: IsaProfile
    ) -> None:
        self.profile = profile
        self.arch_name = arch_name
        self.version = version
        self.encoding_map: dict[str, InstEncoding] = {}
        self.inst_encodings: list[InstEncoding] = []
        self.operand_types: list[str] = []
        self.opnd_selectors: list[OperandSelector] = []
        self.primary_decode_table: list[DecodeTableEntry | None] = [
            None
        ] * pow(2, profile.max_enc_bits)
        self.alt_encs_with_implied_literal: set[str] = set()
        if self.version not in profile.supported_versions:
            raise ValueError(
                f'Unsupported machine-readable ISA spec version: {self.version}'
            )
