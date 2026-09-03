# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""C++ code generator for AMD GPU ISA decoders and instruction classes.

Generates the following C++ files from a parsed ``IsaSpec``:

* ``machine_insts.h`` - bitfield structs for each encoding's microcode.
* ``opcodes.h`` - per-ISA symbolic opcode constants keyed by mnemonic.
* ``builders.h`` - per-ISA instruction builders keyed by encoding format.
* ``encodings.h/.cpp`` - encoding classes (mnemonic, size, modifiers).
* ``operand_types.h`` - operand type enum and operand selector metadata.
* ``operand.h/.cpp`` - ISA-specific operand class with read/write methods.
* ``<inst>.cpp`` - per-encoding instruction files with ``execute()`` bodies
  (only when ``SemanticsSpec`` is provided).
* ``decoder.h/.cpp`` - primary decode table and per-format sub-decoders.

ISA-specific mnemonic formatting and modifier rules are delegated to the
``IsaProfile`` (via ``mnemonic_rule()`` and ``encoding_modifiers()``).
Execution semantics are provided by ``SemanticsSpec`` from
:mod:`amdisa.semantics`.
"""

import cgen
import textwrap
import re
import os

from dataclasses import dataclass, field as _field
from collections import defaultdict
from enum import Enum, auto

from amdisa.gpuisa import (
    InstEncoding,
    Instruction,
    IsaSpec,
    Operand,
    OperandNamePattern,
    opcode_constant_base_name,
    opcode_name_fragment,
)
from amdisa.fieldless_policy import (
    FieldlessCategory,
    fieldless_policy,
    operand_participates,
)
from amdisa.semantics import InstructionSemantics, SemanticsSpec
from amdisa.isa_profile import DppOpcodeRule

from amdisa.codegen.config import CodegenConfig
from amdisa.codegen.cpp_file import CppFile
from amdisa.codegen.shared_baselines import (
    SCALAR_SHARED_INCLUDE as _SCALAR_SHARED_INCLUDE,
    CDNA_SHARED_INCLUDE as _CDNA_SHARED_INCLUDE,
    SCALAR_BASELINE as _SCALAR_BASELINE,
    CDNA_BASELINE as _CDNA_BASELINE,
    CDNA_ARCHES as _CDNA_ARCHES,
)
from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
    vop3_dst_mod,
    vop3_dst_mod_f64,
)
from amdisa.codegen.execute.vector_special import (
    gen_vector_mbcnt,
    gen_vector_mad_64_32,
    gen_vector_mad_32_16,
    gen_vector_div_fixup,
    gen_vector_div_scale,
    gen_vector_div_fmas,
    gen_vector_dot,
    gen_vector_dot2c_bf16,
    gen_vector_bitop3,
    gen_vector_permlane_swap,
    gen_vector_permlane,
    gen_vector_permlane64,
    gen_vector_cvt_pk,
    gen_vector_cvt_scale,
)
from amdisa.codegen.execute.vector_cmp import (
    gen_vector_cmp_class,
    gen_vector_cmp,
    gen_vector_cmpx,
    gen_vector_add_co,
)
from amdisa.codegen.execute.packed import (
    gen_pk_binop,
    gen_pk_ternary,
    gen_pk_binop_f32,
    gen_pk_ternary_f32,
    gen_pk_mov_b32,
    gen_mad_mix_f32,
    gen_mad_mix_lo_hi,
    gen_dot2,
    gen_dot4,
    gen_dot8,
)
from amdisa.codegen.execute.matrix import (
    gen_accvgpr_read,
    gen_accvgpr_write,
    gen_mfma,
)


class _Literal32Widening(Enum):
    ZERO_EXTEND = 'ZeroExtend'
    SIGN_EXTEND = 'SignExtend'
    REPLICATE_32 = 'Replicate32'
    F64_HIGH_BITS = 'F64HighBits'


# Keep every shipped 64-bit source format explicit. PK_F32 replicates its literal
# into both DWORDs, while I64 and F64 reinterpret the extension word by type.
_SIMM32_64BIT_WIDENING = {
    'FMT_NUM_B32': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_B64': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_BF16': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_BF8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_F16': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_F32': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_FP8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_I8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_M64': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK16_U4': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK2_B32': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK2_F32': _Literal32Widening.REPLICATE_32,
    'FMT_NUM_PK2_U32': _Literal32Widening.REPLICATE_32,
    'FMT_NUM_PK4_BF16': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK4_F16': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK8_BF8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK8_FP8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_PK8_I8': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_U64': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_16X16_BF8': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_16X16_FP8': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_16X16_IU8': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_16X32_IU4': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_16X4_F32': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_AB_IU4': _Literal32Widening.ZERO_EXTEND,
    'FMT_WMMA_INDEX_SET2': _Literal32Widening.ZERO_EXTEND,
    'FMT_NUM_I64': _Literal32Widening.SIGN_EXTEND,
    'FMT_NUM_F64': _Literal32Widening.F64_HIGH_BITS,
}

_LITERAL_ENCODING_OPERANDS = {
    'ENC_SOP1': ('Sop1InstLiteralMachineInst', ('ssrc0',)),
    'ENC_SOP2': ('Sop2InstLiteralMachineInst', ('ssrc0', 'ssrc1')),
    'ENC_SOPC': ('SopcInstLiteralMachineInst', ('ssrc0', 'ssrc1')),
    'ENC_VOP1': ('Vop1InstLiteralMachineInst', ('src0',)),
    'ENC_VOP2': ('Vop2InstLiteralMachineInst', ('src0',)),
    'VOP2_INST_LITERAL64': ('Vop2InstLiteral64MachineInst', ('src0',)),
    'ENC_VOPC': ('VopcInstLiteralMachineInst', ('src0',)),
    'ENC_VOP3': ('Vop3InstLiteralMachineInst', ('src0', 'src1', 'src2')),
    'ENC_VOP3P': ('Vop3pInstLiteralMachineInst', ('src0', 'src1', 'src2')),
    'VOP3_SDST_ENC': (
        'Vop3SdstEncInstLiteralMachineInst',
        ('src0', 'src1', 'src2'),
    ),
}

# Immediate operands carry their value directly in the instruction encoding.
_IMMEDIATE_OPERAND_TYPES = (
    'OPR_SIMM4',
    'OPR_SIMM8',
    'OPR_SIMM16',
    'OPR_SIMM32',
    'OPR_SIMM64',
    'OPR_LABEL',
    'OPR_WAITCNT',
)

# Only operand selector families that define SRC_LITERAL may consume a literal
# extension word. Keeping this as an allowlist makes new selector types fail
# closed until their literal capability is explicitly established.
_LITERAL_CAPABLE_OPERAND_TYPES = frozenset(
    {
        'OPR_SRC',
        'OPR_SRC_NOINLINE',
        'OPR_SRC_NOLDS',
        'OPR_SREG_LITERAL',
        'OPR_SSRC',
        'OPR_SSRC_NOLDS',
    }
)


def _exec_mask_flag_stmts(sem) -> list[str]:
    """Return ``flags_ |= ...;`` statements for EXEC instruction metadata.

    The flags are derived from the instruction's semantic AST so generic
    liveness/dataflow analyses do not have to know AMDGPU instruction names:

    * ``IGNORES_EXEC`` - branch-style instructions that run regardless of EXEC.
    * ``WRITES_EXEC``  - the instruction writes EXEC.

    Returns an empty list when ``sem`` is None or its semantic class has no
    deriver (``derive_sema_block`` returns None) or derives to an empty stub;
    those instructions stay flag-free and analyses treat them conservatively.
    """
    if sem is None:
        return []
    # Lazy import mirrors the execute-body generation path and avoids a
    # module-load import cycle with the sema_* modules.
    from amdisa.sema_derive import derive_sema_block
    from amdisa.sema_properties import InstructionProperty, derive_properties

    block = derive_sema_block(sem)
    if block is None or block.is_empty:
        return []
    props = derive_properties(block)
    return [
        f'flags_ |= {name};'
        for prop, name in (
            (InstructionProperty.IGNORES_EXEC, 'IGNORES_EXEC'),
            (InstructionProperty.WRITES_EXEC, 'WRITES_EXEC'),
        )
        if prop in props
    ]


def _result_combinator_flag_stmts(sem) -> list[str]:
    """Return ``flags_ |= ...;`` for how the instruction forms its result value.
    Only returns flags for scalar operations.


    * ``RESULT_COPY`` - result is a plain copy of a single source (s_mov).
    * ``RESULT_OR``   - result is the bitwise OR of its sources (s_or,
      s_or_saveexec)

    Derived from the per-opcode semantic class/operation.
    EXEC-state analysis uses this to prove an all-ones EXEC write:
    """
    if sem is None:
        return []
    cls = sem.semantic_class
    op = (sem.operation or '').lower()
    if cls == 'scalar_mov':
        return ['flags_ |= RESULT_COPY;']
    if cls in ('scalar_binop', 'scalar_saveexec') and op == 'or':
        return ['flags_ |= RESULT_OR;']
    return []


@dataclass
class _SourceImplUnit:
    file_stem: str | None
    impls: list[object]


@dataclass
class _ImplOutputs:
    """Implementation fragments destined for model and execution sources."""

    model: list[object] = _field(default_factory=list)
    execution: list[object] = _field(default_factory=list)

    def execution_target(self, split: bool) -> list[object]:
        """Return the execution output, or the model output for unsplit profiles."""
        return self.execution if split else self.model


@dataclass(frozen=True)
class _True16Vop3Info:
    force_src: bool
    has_dst: bool
    body_uses_true16: bool
    enabled: bool


class _MaskResultKind(Enum):
    COMPARE = auto()
    EXEC = auto()
    SECONDARY = auto()
    IMPLICIT_VCC = auto()


@dataclass(frozen=True)
class _OperandCtx:
    opnd_name: str
    enc_name: str
    packed_16bit: bool
    operand_type: str | None = None
    has_acc_field: bool = False
    has_acc_cd_field: bool = False


class _SemanticEmitter:
    """Entry point for execute() body generation.

    This class provides a named abstraction;
    the full method extraction (one ``emit_<cls>`` method per semantic class,
    replacing the ~600-line ``if cls == ...`` chain in ``_gen_execute_body``).

    Attributes:
        _spec: Parsed ISA specification.
        _semantics: Optional semantic metadata for execute() bodies.
    """

    def __init__(self, spec: IsaSpec, semantics: SemanticsSpec | None) -> None:
        self._spec = spec
        self._semantics = semantics


class CodeGenerator:
    """Generates C++ code from a parsed machine-readable ISA specification.

    Produces encoding classes, instruction classes with execute() bodies,
    operand types, and the primary/sub-decode tables. ISA-specific
    mnemonic and modifier rules come from the ``IsaProfile`` attached to
    the ``IsaSpec``. Execution-semantics templates (scalar ALU, vector
    compare, memory load/store, etc.) come from ``SemanticsSpec``.

    Attributes:
        isa_spec: Parsed ISA specification with encodings and instructions.
        out_path: Output directory for generated C++ files.
        semantics: Optional semantic metadata for generating execute() bodies.
        config: Code generation configuration (namespace, include paths).
    """

    # Capacities of the fixed-size operand arrays declared in the hand-written
    # instruction.h (``std::array<Operand *, N> src_operands_ / dst_operands_``).
    # Mirrored here to bound the generation-time overflow tripwire. KEEP IN SYNC
    # with instruction.h: if you resize either std::array, update these too (and
    # vice versa).
    _SRC_OPERANDS_CAPACITY = 6
    _DST_OPERANDS_CAPACITY = 3

    # Shared scalar execution uses these encoding values without including one
    # ISA's generated operand enums. Validate the corresponding OPR_SSRC
    # contract for every generated ISA so ISA description changes cannot
    # silently make the shared resolver stale.
    _SHARED_SCALAR_PAIR_VALUES = frozenset((*range(0, 107), *range(108, 123), 126))

    # These selectors name a canonical unified register namespace, while their
    # instruction fields use a format-dependent bank namespace completed by
    # separate ACC/ACC_CD bits or format selectors. Some malformed fields are
    # intentionally preserved for instruction-specific diagnostics, so the
    # generic Operand constructor must not compare those raw values with the
    # canonical selector intervals.
    _NON_CANONICAL_SELECTOR_OPERAND_TYPES = frozenset(
        {
            'OPR_SRC_ACCVGPR',
            'OPR_SRC_ACCVGPR_OR_CONST',
            'OPR_SRC_VGPR_OR_ACCVGPR',
            'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST',
            'OPR_VGPR_OR_ACCVGPR',
        }
    )

    # These gfx1250 K=128 forms accept an ordinary scalar source for src2 in
    # addition to the VGPR/inline spellings declared by the shared XML operand
    # type. Keep that instruction-specific namespace explicit while validating
    # the other OPR_SRC_VGPR_OR_INLINE users against their declared intervals.
    _GENERIC_WMMA_ACCUMULATOR_INSTRUCTIONS = frozenset(
        f'V_WMMA_F{result_bits}_16X16X128_{lhs}_{rhs}'
        for result_bits in (16, 32)
        for lhs in ('FP8', 'BF8')
        for rhs in ('FP8', 'BF8')
    )

    def __init__(
        self,
        isa_spec: IsaSpec,
        out_path: str,
        semantics: SemanticsSpec | None = None,
        config: CodegenConfig | None = None,
        shared_plan: 'SharedInstructionPlan | None' = None,
    ) -> None:
        self.isa_spec = isa_spec
        self.out_path = out_path
        self.semantics = semantics
        self.config = config if config is not None else CodegenConfig()
        self.shared_plan = shared_plan
        # Keyed by (mnemonic, enc_name) to avoid cross-encoding conflicts.
        self._shared_execute_bodies: dict[tuple[str, str], tuple] = {}
        # Canonical fixed encoding value per fieldless operand type, computed
        # lazily from the spec's selectors on first use (see
        # _fieldless_canonical_value).
        self._fieldless_canon_cache: dict[str, int] | None = None
        self._selector_interval_cache: dict[str, list[tuple[int, int]]] = {}
        self._emitter = _SemanticEmitter(isa_spec, semantics)
        # Split profiles assign a dense, named ID to every concrete instruction
        # class. The matching immutable callback table is emitted after all
        # instruction classes have been generated.
        self._split_execution_classes: list[str] = []

    @property
    def generated_dir_name(self) -> str:
        """Filesystem directory for this ISA's generated files."""
        return self.isa_spec.generated_dir_name

    @property
    def handwritten_dir_name(self) -> str:
        """Profile directory containing this ISA's handwritten files."""
        return self.isa_spec.arch_name

    @property
    def cpp_namespace(self) -> str:
        """C++ namespace for this ISA's generated declarations."""
        return self.isa_spec.cpp_namespace

    def _split_execute_expr(self, class_name: str) -> str:
        """Return the model constructor expression for one split instruction."""
        if not self.isa_spec.profile.split_execution_sources:
            return f'make_exec_fn<{class_name}>()'
        classes = getattr(self, '_split_execution_classes', None)
        if classes is None:
            classes = []
            self._split_execution_classes = classes
        if class_name in classes:
            raise ValueError(f'duplicate split execution class: {class_name}')
        classes.append(class_name)
        return f'selected_exec_fn(InstructionExecutionId::{class_name})'

    def _supports_simm64_literal_operands(self) -> bool:
        return 'OPR_SIMM64' in self.isa_spec.operand_types

    @staticmethod
    def _literal64_condition_names(inst_enc: InstEncoding) -> list[str]:
        """Return sizing conditions for explicit and implied Literal64 forms."""
        conditions = [
            name for name, _ in inst_enc.enc_conds if name.startswith('has_lit64')
        ]
        if any(
            inst.is_implied_literal_enc
            and inst.enc_name.upper().endswith('_INST_LITERAL64')
            for inst in inst_enc.insts
        ):
            conditions.append('hasImpliedLiteral64')
        return conditions

    @staticmethod
    def _rejects_unencoded_vop3p_literal64(inst_enc: InstEncoding) -> bool:
        """Whether VOP3P must reject the global Literal64 selector.

        gfx1250 exposes selector 254 in its global source enum, but VOP3P has
        only the one-DWORD Literal32 extension. Keep this format-level rule
        conditional on the complete encoding capability so a future
        authoritative explicit or implied VOP3P Literal64 encoding becomes
        usable without a conflicting rejection.
        """
        return (
            inst_enc.enc_name.upper() == 'ENC_VOP3P'
            and not CodeGenerator._literal64_condition_names(inst_enc)
        )

    def _unsupported_literal64_selector_fields(
        self, inst_enc: InstEncoding
    ) -> tuple[str, ...]:
        """Return source fields whose Literal64 selector this encoding rejects."""
        literal32_conds = {
            name
            for name, _ in inst_enc.enc_conds
            if name.startswith('has_lit') and not name.startswith('has_lit64')
        }
        if not (
            self._supports_simm64_literal_operands()
            and literal32_conds
            and self._rejects_unencoded_vop3p_literal64(inst_enc)
        ):
            return ()

        fields: set[str] = set()
        for name, condition in inst_enc.enc_conds:
            if name not in literal32_conds:
                continue
            fields.update(
                re.findall(r'inst_\.([A-Za-z_][A-Za-z0-9_]*)\s*==\s*255', condition)
            )
        return tuple(sorted(fields))

    @staticmethod
    def _opcode_name_fragment(token: str) -> str:
        """Return one C++ constant-name fragment for a mnemonic token.

        The generated instruction class names use Python ``capitalize()``,
        which turns ``s_getpc_b64`` into ``SGetpcB64``.  Opcode constants are
        meant to be read and used directly by handwritten DBT code, so split a
        few packed ISA abbreviations that commonly appear inside one XML token.
        """
        return opcode_name_fragment(token)

    @classmethod
    def _opcode_const_base_name(cls, mnemonic: str) -> str:
        return opcode_constant_base_name(mnemonic)

    @staticmethod
    def _emit_opcode_constant(name: str, value: int) -> str:
        return f'inline constexpr uint16_t {name} = {value};'

    @staticmethod
    def _emit_encoding_constant(name: str, value: int) -> str:
        return f'inline constexpr uint16_t {name} = {value};'

    @staticmethod
    def _raw_opcode_value(enc: InstEncoding, inst: Instruction) -> int:
        if enc.op_field_bit_cnt == 0:
            return inst.opcode
        return inst.opcode & ((1 << enc.op_field_bit_cnt) - 1)

    def _primary_decode_values(self, enc: InstEncoding) -> list[int]:
        ptrs = enc.primary_dt_ptrs
        if ptrs is None:
            parent_name = self.isa_spec.profile.derive_parent_enc_name(enc.enc_name)
            parent = self.isa_spec.encoding_map.get(parent_name)
            return self._primary_decode_values(parent) if parent else []

        return sorted({value for value in ptrs if value != -1})

    def _primary_decode_duplicate_count(self, value: int) -> int:
        dt = self.isa_spec.primary_decode_table
        if value < 0 or value >= len(dt) or dt[value] is None:
            return 1
        return dt[value].num_dupe_entries

    @staticmethod
    def _encoding_group_name(
        base_name: str, offset: int, has_multiple_values: bool
    ) -> str:
        # The base constant already names the first decode value.  Reuse it for
        # offset zero instead of emitting a redundant ``OpHi0`` alias that is
        # absent from the checked-in generated headers.
        if offset == 0:
            return base_name
        if has_multiple_values:
            return f'{base_name}OpHi{offset}'
        return f'{base_name}Hi{offset}'

    def _encoding_constants_block(self) -> cgen.Line | None:
        """Generate primary-decode selector constants for ``encodings.h``."""
        constants: list[tuple[str, int]] = []
        seen: dict[str, int] = {}

        for enc in self.isa_spec.inst_encodings:
            if not enc.insts:
                continue

            values = self._primary_decode_values(enc)
            if not values:
                continue

            base_name = f'k{enc.fmt_enc_name}'
            has_multiple_values = len(values) > 1
            existing = seen.get(base_name)
            if existing is None:
                seen[base_name] = values[0]
                constants.append((base_name, values[0]))
            elif existing != values[0]:
                raise ValueError(
                    f'encoding constant collision for '
                    f'{self.isa_spec.arch_name}::encoding::{base_name}: '
                    f'{existing} vs {values[0]}'
                )
            for value in values:
                value_offset = value - values[0]
                name = self._encoding_group_name(
                    base_name, value_offset, has_multiple_values
                )
                existing = seen.get(name)
                if existing is None:
                    seen[name] = value
                    constants.append((name, value))
                elif existing != value:
                    raise ValueError(
                        f'encoding constant collision for '
                        f'{self.isa_spec.arch_name}::encoding::{name}: '
                        f'{existing} vs {value}'
                    )

                duplicate_count = self._primary_decode_duplicate_count(value)
                for duplicate_offset in range(1, duplicate_count):
                    duplicate_name = self._encoding_group_name(
                        base_name,
                        value_offset + duplicate_offset,
                        has_multiple_values,
                    )
                    duplicate_value = value + duplicate_offset
                    existing = seen.get(duplicate_name)
                    if existing is None:
                        seen[duplicate_name] = duplicate_value
                        constants.append((duplicate_name, duplicate_value))
                    elif existing != duplicate_value:
                        raise ValueError(
                            f'encoding constant collision for '
                            f'{self.isa_spec.arch_name}::encoding::{duplicate_name}: '
                            f'{existing} vs {duplicate_value}'
                        )

        if not constants:
            return None

        lines = [
            'namespace encoding {',
            '',
            '/// @brief Primary decode selector constants generated from the ISA XML.',
            '///',
            '/// These values match Instruction::encoding_id(), which is word0 >> 23.',
            '/// They are not necessarily the narrower MachineInst::encoding bitfield value.',
        ]
        lines.extend(
            self._emit_encoding_constant(name, value) for name, value in constants
        )
        lines.extend(['', '} // namespace encoding'])
        return cgen.Line('\n'.join(lines))

    def gen_opcode_constants(self) -> None:
        """Generate namespace-level opcode constants for every instruction.

        AMDGPU raw opcode fields are scoped by encoding format, and some
        mnemonics exist in multiple formats with different opcode values.  To
        make handwritten DBT code both readable and unambiguous, every concrete
        instruction gets an encoding-suffixed name (for example
        ``kVMovB32Vop1``).  A shorter bare mnemonic alias (for example
        ``kSGetPcB64``) is emitted only when all instances of that mnemonic in
        this ISA use the same raw opcode.
        """
        arch = self.cpp_namespace
        mnemonic_values: dict[str, list[int]] = defaultdict(list)
        concrete: list[tuple[str, int]] = []
        seen: dict[str, int] = {}

        # Parser collections preserve XML declaration order.  Retaining that
        # order here makes regenerated headers deterministic while keeping the
        # constants grouped like the source ISA specification.
        for enc in self.isa_spec.inst_encodings:
            for inst in enc.insts:
                base_name = self._opcode_const_base_name(inst.name)
                concrete_name = f'{base_name}{inst.fmt_true_enc_name}'
                opcode = self._raw_opcode_value(enc, inst)
                existing = seen.get(concrete_name)
                if existing is None:
                    seen[concrete_name] = opcode
                    concrete.append((concrete_name, opcode))
                elif existing != opcode:
                    raise ValueError(
                        f'opcode constant collision for {arch}::{concrete_name}: '
                        f'{existing} vs {opcode}'
                    )
                mnemonic_values[base_name].append(opcode)

        aliases = [
            (name, values[0])
            for name, values in mnemonic_values.items()
            if len(set(values)) == 1 and name not in seen
        ]

        lines = [
            CppFile._prologue_comment(),
            f'#ifndef ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_OPCODES_H_',
            f'#define ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_OPCODES_H_',
            '',
            '#include <cstdint>',
            '',
            'namespace rocjitsu {',
            f'namespace {arch} {{',
            '',
            '/// @brief Encoding-qualified raw opcode constants generated from the ISA XML.',
        ]
        lines.extend(
            self._emit_opcode_constant(name, value) for name, value in concrete
        )
        if aliases:
            lines.extend(
                [
                    '',
                    '/// @brief Bare mnemonic aliases emitted only when the raw opcode is unambiguous.',
                ]
            )
            lines.extend(
                self._emit_opcode_constant(name, value) for name, value in aliases
            )
        lines.extend(
            [
                '',
                f'}} // namespace {arch}',
                '} // namespace rocjitsu',
                '',
                f'#endif // ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_OPCODES_H_',
                '',
            ]
        )

        arch_out_path = os.path.join(self.out_path, self.generated_dir_name)
        os.makedirs(arch_out_path, exist_ok=True)
        with open(os.path.join(arch_out_path, 'opcodes.h'), 'w') as f:
            f.write('\n'.join(lines))

    @staticmethod
    def _builder_field_type(bit_count: int) -> str:
        """Return the smallest conventional type able to hold an XML field."""
        if bit_count <= 8:
            return 'uint8_t'
        if bit_count <= 16:
            return 'uint16_t'
        if bit_count <= 32:
            return 'uint32_t'
        return 'uint64_t'

    @staticmethod
    def _builder_name(enc: InstEncoding) -> str:
        """Return the snake-case builder name for an encoding format."""
        name = enc.enc_name
        if name.startswith('ENC_'):
            name = name[len('ENC_') :]
        return f'build_{name.lower()}'

    def _builder_encoding_value(self, enc: InstEncoding) -> int:
        """Return the MachineInst ``encoding`` bitfield value for *enc*.

        Primary decode selectors are indexed by ``word0 >> 23``.  The XML
        encoding field can begin above bit 23 (VOP3 begins at bit 26, for
        example), so its MachineInst value is the selector shifted back down
        to the field's origin.  Opcode bits below the encoding field disappear
        in that shift, leaving the fixed format selector.
        """
        encoding_field = next(
            (field for field in enc.ucode_fields if field.name == 'encoding'),
            None,
        )
        if encoding_field is None:
            raise ValueError(f'{enc.enc_name} has no encoding field')
        if encoding_field.bit_offset < 23:
            raise ValueError(
                f'{enc.enc_name} encoding field starts below primary decode bit 23'
            )

        values = self._primary_decode_values(enc)
        if not values:
            raise ValueError(f'{enc.enc_name} has no primary decode selector')
        mask = (1 << encoding_field.bit_cnt) - 1
        return (min(values) >> (encoding_field.bit_offset - 23)) & mask

    def gen_instruction_builders(self) -> None:
        """Generate compact, format-level instruction encoding helpers.

        A builder fixes the XML-derived encoding selector and accepts the raw
        generated opcode plus a value-initialized field structure.  Padding,
        opcode, and encoding fields are intentionally absent from that public
        structure: callers should describe only operands and modifiers, while
        the generator remains responsible for the binary layout.
        """
        arch = self.cpp_namespace
        body: list[str] = [
            CppFile._prologue_comment(),
            f'#ifndef ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_BUILDERS_H_',
            f'#define ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_BUILDERS_H_',
            '',
            '#include <array>',
            '#include <cstddef>',
            '#include <cstdint>',
            '',
            'namespace rocjitsu {',
            f'namespace {arch} {{',
            '',
            'namespace builder_detail {',
            '',
            '/// @brief Insert one XML field into an encoded instruction.',
            'template <std::size_t NumWords>',
            'constexpr void set_field(std::array<uint32_t, NumWords> &words, uint64_t value,',
            '                         std::size_t bit_offset, std::size_t bit_count) {',
            '  while (bit_count != 0) {',
            '    const std::size_t word_index = bit_offset / 32;',
            '    const std::size_t word_offset = bit_offset % 32;',
            '    const std::size_t bits_here = bit_count < (32 - word_offset) ? bit_count :',
            '                                                                  (32 - word_offset);',
            '    const uint32_t mask = bits_here == 32 ? ~uint32_t{0} :',
            '                                                  ((uint32_t{1} << bits_here) - 1);',
            '    words[word_index] |= (static_cast<uint32_t>(value) & mask) << word_offset;',
            '    value >>= bits_here;',
            '    bit_offset += bits_here;',
            '    bit_count -= bits_here;',
            '  }',
            '}',
            '',
            '} // namespace builder_detail',
        ]

        for enc in self.isa_spec.inst_encodings:
            # Alternate decoder-only encodings have no instructions of their
            # own.  Their selector conditions (for example src0 == 0xfa for
            # DPP) are not fixed format fields and therefore are not safe to
            # expose as standalone builders yet.
            if not enc.insts:
                continue

            fields = [
                field
                for field in enc.ucode_fields
                if field.name not in ('op', 'encoding')
                and not field.name.startswith('pad_')
            ]
            field_struct = f'{enc.fmt_enc_name}BuilderFields'
            word_count = (enc.bit_cnt + 31) // 32
            opcode_field = next(
                (field for field in enc.ucode_fields if field.name == 'op'), None
            )
            encoding_field = next(
                field for field in enc.ucode_fields if field.name == 'encoding'
            )
            encoding_value = self._builder_encoding_value(enc)

            body.extend(
                [
                    '',
                    f'/// @brief Caller-controlled fields for {enc.enc_name}.',
                    f'struct {field_struct} {{',
                ]
            )
            body.extend(
                f'  {self._builder_field_type(field.bit_cnt)} {field.name} = 0;'
                for field in fields
            )
            body.extend(
                [
                    '};',
                    '',
                    f'/// @brief Build one {enc.enc_name} instruction.',
                    f'[[nodiscard]] constexpr std::array<uint32_t, {word_count}>',
                    (
                        f'{self._builder_name(enc)}(uint16_t op, '
                        f'{field_struct} fields = {{}}) {{'
                        if opcode_field is not None
                        else f'{self._builder_name(enc)}({field_struct} fields = {{}}) {{'
                    ),
                    f'  std::array<uint32_t, {word_count}> words{{}};',
                ]
            )
            if opcode_field is not None:
                body.extend(
                    [
                        f'  builder_detail::set_field(words, op, {opcode_field.bit_offset},',
                        f'                            {opcode_field.bit_cnt});',
                    ]
                )
            body.extend(
                [
                    f'  builder_detail::set_field(words, {encoding_value},',
                    f'                            {encoding_field.bit_offset},',
                    f'                            {encoding_field.bit_cnt});',
                ]
            )
            for field in fields:
                body.extend(
                    [
                        f'  builder_detail::set_field(words, fields.{field.name},',
                        f'                            {field.bit_offset}, {field.bit_cnt});',
                    ]
                )
            body.extend(['  return words;', '}'])

        body.extend(
            [
                '',
                f'}} // namespace {arch}',
                '} // namespace rocjitsu',
                '',
                f'#endif // ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_BUILDERS_H_',
                '',
            ]
        )

        arch_out_path = os.path.join(self.out_path, self.generated_dir_name)
        os.makedirs(arch_out_path, exist_ok=True)
        with open(os.path.join(arch_out_path, 'builders.h'), 'w') as f:
            f.write('\n'.join(body))

    def _constructor_operand_type(
        self, inst_sem: InstructionSemantics | None, opnd: Operand
    ) -> str:
        if self._uses_generic_wmma_accumulator_selector(inst_sem, opnd):
            return 'OPR_SRC'
        if (
            inst_sem
            and inst_sem.semantic_class in ('vector_readfirstlane', 'vector_readlane')
            and opnd.name == 'src0'
            and opnd.is_input
            and 'OPR_SRC_VGPR' in self.isa_spec.operand_types
        ):
            return 'OPR_SRC_VGPR'
        if inst_sem and inst_sem.accvgpr_srcs and opnd.is_input:
            return 'OPR_SRC_VGPR_OR_ACCVGPR'
        return opnd.operand_type

    def _uses_generic_wmma_accumulator_selector(
        self, inst_sem: InstructionSemantics | None, opnd: Operand
    ) -> bool:
        return (
            getattr(self.isa_spec, 'arch_name', None) == 'cdna5'
            and inst_sem is not None
            and inst_sem.name in self._GENERIC_WMMA_ACCUMULATOR_INSTRUCTIONS
            and opnd.name == 'src2'
            and opnd.operand_type == 'OPR_SRC_VGPR_OR_INLINE'
        )

    @staticmethod
    def _sdwa_source_modifier_format(
        sem: InstructionSemantics, source_index: int, opnd: Operand
    ) -> str:
        """Return the C++ floating format for one SDWA source modifier.

        MRISA operand formats distinguish conversion inputs from outputs, which
        instruction-level ``data_type`` cannot do by itself. Two instructions
        have mixed semantic source types despite homogeneous XML operand
        formats: class compares use an integer class mask as src1, and LDEXP
        uses an integer exponent as src1.
        """
        if source_index == 1 and (
            sem.semantic_class in ('vector_cmp_class', 'vector_cmpx_class')
            or (sem.semantic_class == 'vector_binop' and sem.operation == 'ldexp')
        ):
            return 'amdgpu::sdwa::SourceModifierFormat::NONE'

        # GFX9 accepts the SDWA modifier bits for V_PK_FMAC_F16, but its packed
        # operation ignores source negate and absolute-value modifiers. Source
        # selection and sign extension remain active through the NONE format.
        if sem.name == 'V_PK_FMAC_F16':
            return 'amdgpu::sdwa::SourceModifierFormat::NONE'

        suffix = {
            'FMT_NUM_F16': 'F16',
            'FMT_NUM_BF16': 'BF16',
            'FMT_NUM_F32': 'F32',
        }.get(opnd.data_format_name, 'NONE')
        return f'amdgpu::sdwa::SourceModifierFormat::{suffix}'

    @staticmethod
    def _literal_encoding_info(
        enc: InstEncoding, inst_enc_obj: InstEncoding | None, inst: Instruction
    ) -> tuple[str, tuple[str, ...]] | None:
        # Implied-literal instructions are parsed from an alternate XML
        # encoding but generated in the parent encoding class.
        # The 64-bit literal form has a distinct three-DWORD MachineInst whose
        # literal operand must be read in full, rather than through the normal
        # one-DWORD literal extension used by the parent encoding.
        if inst.is_implied_literal_enc and inst.enc_name.upper().endswith(
            '_INST_LITERAL64'
        ):
            lit_enc = inst_enc_obj or enc
        else:
            lit_enc = enc if inst.is_implied_literal_enc else (inst_enc_obj or enc)
        return _LITERAL_ENCODING_OPERANDS.get(lit_enc.enc_name.upper())

    def _encoded_dpp_opcodes(
        self, inst_enc: InstEncoding, modifier: str
    ) -> tuple[int, ...]:
        if modifier == 'dpp':
            supports = self._instruction_supports_dpp
        elif modifier == 'dpp8':
            supports = self._instruction_supports_dpp8
        else:
            raise ValueError(f'unknown DPP modifier encoding: {modifier}')
        return tuple(
            sorted(
                inst.opcode
                for inst in inst_enc.insts
                if supports(inst, inst_enc.enc_name)
            )
        )

    @staticmethod
    def _opcode_set_condition(opcodes: list[int], opcode_max: int | None = None) -> str:
        """Render a compact membership test for a sorted opcode list."""
        ranges: list[tuple[int, int]] = []
        for opcode in sorted(opcodes):
            if ranges and opcode == ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], opcode)
            else:
                ranges.append((opcode, opcode))
        return ' || '.join(
            (
                f'inst_.op == {first}'
                if first == last
                else (
                    'true'
                    if first == 0 and last == opcode_max
                    else (
                        f'inst_.op <= {last}'
                        if first == 0
                        else (
                            f'inst_.op >= {first}'
                            if last == opcode_max
                            else f'(inst_.op >= {first} && inst_.op <= {last})'
                        )
                    )
                )
            )
            for first, last in ranges
        )

    def _gfx12_cache_policy_modifier_impl(
        self, inst_enc: InstEncoding, enc_field_names: set[str]
    ) -> str:
        """Render opcode-aware GFX12 TH and SCOPE disassembly."""
        from amdisa.isa_profile import MemoryCoherencyModel

        if (
            self.isa_spec.profile.coherency_model != MemoryCoherencyModel.GFX12_SCOPE_TH
            or not {'op', 'scope', 'th'} <= enc_field_names
        ):
            return ''

        # Mirrors LLVM getTemporalHintType: atomics use the atomic spelling;
        # store-only plus async/tensor stores use the store spelling; all other
        # operations use the load spelling. Keep the classification explicit so
        # a new semantic class cannot silently change category after a rename.
        atomic_classes = frozenset({'buffer_atomic', 'flat_atomic', 'image_atomic'})
        store_classes = frozenset(
            {
                'buffer_store',
                'buffer_store_format_d16',
                'flat_store',
                'global_store_addtid',
                'global_store_async_from_lds',
                'image_store',
                'tensor_store_from_lds',
            }
        )
        load_classes = frozenset(
            {
                'buffer_load',
                'buffer_load_format_d16',
                'dcache_inv',
                'flat_load',
                'gl1_inv',
                'gl1_wbinv',
                'global_load_addtid',
                'global_load_async_to_lds',
                'image_bvh',
                'image_load',
                'image_query',
                'image_sample',
                'nop',
                'smem_load',
                'tensor_load_to_lds',
                'true_nop',
            }
        )
        op_kinds: dict[int, str] = {}
        for inst in inst_enc.insts:
            sem = self.semantics.instructions.get(inst.name) if self.semantics else None
            if sem is None:
                raise ValueError(
                    f'{inst_enc.enc_name} {inst.name} has TH/SCOPE fields but '
                    'no instruction semantics'
                )
            semantic_class = sem.semantic_class
            if semantic_class in atomic_classes:
                kind = 'Atomic'
            elif semantic_class in store_classes:
                kind = 'Store'
            elif semantic_class in load_classes:
                kind = 'Load'
            else:
                raise ValueError(
                    f'{inst_enc.enc_name} {inst.name} has TH/SCOPE fields but '
                    f'unknown temporal-hint semantic class {semantic_class}'
                )
            previous = op_kinds.setdefault(inst.opcode, kind)
            if previous != kind:
                raise ValueError(
                    f'{inst_enc.enc_name} opcode {inst.opcode} has conflicting '
                    f'temporal hint kinds: {previous} and {kind}'
                )

        grouped: dict[str, list[int]] = {'Atomic': [], 'Store': []}
        for opcode, kind in op_kinds.items():
            if kind in grouped:
                grouped[kind].append(opcode)

        if not any(grouped.values()):
            return (
                'amdgpu::append_gfx12_cache_policy('
                'modifiers_, inst->th, inst->scope, '
                'amdgpu::Gfx12TemporalHintKind::Load);'
            )

        lines = [
            'amdgpu::Gfx12TemporalHintKind hint_kind = '
            'amdgpu::Gfx12TemporalHintKind::Load;',
            'switch (inst->op) {',
        ]
        for kind in ('Atomic', 'Store'):
            for opcode in sorted(grouped[kind]):
                lines.append(f'case {opcode}:')
            if grouped[kind]:
                lines.extend(
                    (
                        f'  hint_kind = amdgpu::Gfx12TemporalHintKind::{kind};',
                        '  break;',
                    )
                )
        lines.extend(
            (
                'default:',
                '  break;',
                '}',
                'amdgpu::append_gfx12_cache_policy('
                'modifiers_, inst->th, inst->scope, hint_kind);',
            )
        )
        return ''.join(lines)

    @classmethod
    def _opcode_predicate_helper_impl(
        cls,
        inst_enc: InstEncoding,
        name: str,
        opcodes: list[int],
        selector_condition: str | None = None,
    ) -> str:
        """Render an opcode membership predicate for an encoding feature."""
        lines = [f'bool {inst_enc.fmt_enc_name}::{name}() const {{']
        if selector_condition:
            lines.extend((f'  if (!({selector_condition}))', '    return false;'))
        opcode_max = (
            (1 << inst_enc.op_field_bit_cnt) - 1 if inst_enc.op_field_bit_cnt else None
        )
        condition = cls._opcode_set_condition(opcodes, opcode_max)
        lines.extend((f'  return {condition or "false"};', '}'))
        return '\n'.join(lines)

    @staticmethod
    def _encoded_literal_field_masks(
        inst_enc: InstEncoding, literal_fields: tuple[str, ...]
    ) -> dict[int, tuple[str, ...]]:
        """Return the active literal-selector fields for each opcode.

        Encoding formats describe every possible source field, but individual
        opcodes may use only a subset of them or reinterpret one as inline
        immediate data. Extension sizing must follow the selected opcode's
        operands rather than every field present in the format.
        """
        fields_by_opcode: dict[int, set[str]] = {}
        for inst in inst_enc.insts:
            active = fields_by_opcode.setdefault(inst.opcode, set())
            active.update(
                opnd.name
                for opnd in inst.operands
                if opnd.name in literal_fields
                and opnd.operand_type in _LITERAL_CAPABLE_OPERAND_TYPES
            )
        return {
            opcode: tuple(field for field in literal_fields if field in active)
            for opcode, active in fields_by_opcode.items()
        }

    @classmethod
    def _encoded_literal_helper_impl(
        cls,
        inst_enc: InstEncoding,
        literal_fields: tuple[str, ...],
        selector: int,
    ) -> str:
        """Render an opcode-aware encoded-literal predicate."""
        grouped_opcodes: dict[tuple[str, ...], list[int]] = {}
        for opcode, fields in cls._encoded_literal_field_masks(
            inst_enc, literal_fields
        ).items():
            grouped_opcodes.setdefault(fields, []).append(opcode)

        name = f'has_encoded_literal{64 if selector == 254 else 32}'
        lines = [
            f'bool {inst_enc.fmt_enc_name}::{name}() const {{',
            '  switch (inst_.op) {',
        ]
        for fields, opcodes in sorted(
            grouped_opcodes.items(), key=lambda item: (item[0], item[1])
        ):
            if not fields:
                continue
            lines.extend(f'  case {opcode}:' for opcode in sorted(opcodes))
            condition = ' || '.join(f'inst_.{field} == {selector}' for field in fields)
            lines.append(f'    return {condition};')
        lines.extend(('  default:', '    return false;', '  }', '}'))
        return '\n'.join(lines)

    @staticmethod
    def _literal_operand_from_expr_stmt(
        opnd: Operand,
        literal_expr: str,
        size_expr: str | None = None,
        literal_operand_type: str | None = None,
        dynamic_true16_opsel_bit: int | None = None,
        arch_name: str = '<unknown>',
        inst_name: str = '<unknown>',
        enc_name: str = '<unknown>',
    ) -> str | None:
        operand_type = literal_operand_type or opnd.operand_type
        if operand_type == 'OPR_SIMM64':
            return (
                f'{opnd.name} = Operand({size_expr or opnd.size}, '
                f'OperandType::OPR_SIMM64, '
                '(static_cast<uint64_t>(reinterpret_cast<const uint32_t *>(inst)[2]) '
                '<< 32) | reinterpret_cast<const uint32_t *>(inst)[1], true);'
            )
        if operand_type not in ('OPR_SIMM16', 'OPR_SIMM32'):
            return None
        operand_size = size_expr or opnd.size
        if dynamic_true16_opsel_bit is not None:
            display_expr = (
                f'static_cast<uint16_t>(({literal_expr} >> '
                f'(((amdgpu::vop3_opsel(inst_) >> {dynamic_true16_opsel_bit}) & 1u) * 16u)) '
                f'& 0xFFFFu)'
            )
            return (
                f'{opnd.name} = Operand({operand_size}, '
                f'OperandType::{operand_type}, static_cast<int>({literal_expr}), '
                f'{display_expr}, true);'
            )
        if operand_type == 'OPR_SIMM16' or opnd.size == 16:
            literal_expr = f'({literal_expr} & 0xFFFFu)'
        widening = CodeGenerator._literal_operand_64bit_widening(
            opnd, operand_type, arch_name, inst_name, enc_name
        )
        if widening is not None:
            return (
                f'{opnd.name} = Operand::make_literal32('
                f'{operand_size}, static_cast<uint32_t>({literal_expr}), '
                f'Operand::Literal32Widening::{widening.value});'
            )
        return (
            f'{opnd.name} = Operand({operand_size}, '
            f'OperandType::{operand_type}, static_cast<int>({literal_expr}));'
        )

    @staticmethod
    def _literal_operand_64bit_widening(
        opnd: Operand,
        operand_type: str,
        arch_name: str,
        inst_name: str,
        enc_name: str,
    ) -> _Literal32Widening | None:
        if operand_type != 'OPR_SIMM32' or not opnd.is_input or opnd.size != 64:
            return None
        try:
            return _SIMM32_64BIT_WIDENING[opnd.data_format_name]
        except KeyError as exc:
            data_format = opnd.data_format_name or '<missing>'
            raise ValueError(
                f'architecture {arch_name!r}, instruction {inst_name!r}, encoding '
                f'{enc_name!r}: 64-bit SIMM32 input operand {opnd.name!r} has '
                f'unsupported data format {data_format!r}'
            ) from exc

    @staticmethod
    def _literal_operand_fixup_stmt(
        opnd: Operand,
        lit_struct: str,
        size_expr: str | None = None,
        literal_operand_type: str | None = None,
        dynamic_true16_opsel_bit: int | None = None,
        arch_name: str = '<unknown>',
        inst_name: str = '<unknown>',
        enc_name: str = '<unknown>',
    ) -> str | None:
        literal_expr = f'reinterpret_cast<const {lit_struct} *>(inst)->simm32'
        return CodeGenerator._literal_operand_from_expr_stmt(
            opnd,
            literal_expr,
            size_expr,
            literal_operand_type,
            dynamic_true16_opsel_bit,
            arch_name,
            inst_name,
            enc_name,
        )

    @staticmethod
    def _has_inline_literal_operand(inst: Instruction) -> bool:
        return any(
            opnd.name == 'literal' and opnd.operand_type in ('OPR_SIMM16', 'OPR_SIMM32')
            for opnd in inst.operands
        )

    @staticmethod
    def _semantic_source_operands(
        inst: Instruction, src_operands: list[Operand]
    ) -> list[Operand]:
        if inst.name != 'S_FMAMK_F32':
            return src_operands

        by_name = {op.name: op for op in src_operands}
        # The multiplicand literal is the fieldless simm32 (or the field-bearing
        # `literal` on ISAs that name it so).
        mul_literal = by_name.get('simm32') or by_name.get('literal')
        ssrc0 = by_name.get('ssrc0')
        ssrc1 = by_name.get('ssrc1')
        if ssrc0 is None or ssrc1 is None or mul_literal is None:
            return src_operands

        ordered = [ssrc0, mul_literal, ssrc1]
        ordered_names = {op.name for op in ordered}
        ordered.extend(op for op in src_operands if op.name not in ordered_names)
        return ordered

    @classmethod
    def _execute_operand_participates(cls, opnd: Operand) -> bool:
        """Whether codegen should expose this operand to execute templates.

        Field-bearing operands always participate. A fieldless operand
        participates only if it reads a real value (its policy ``reads_value``
        capability), which today is just the literal ``OPR_SIMM32``. Most
        fieldless operands are hardwired side effects still handled by the
        semantic emitters (VCC/EXEC/SCC/M0/PC) and stay out of the positional
        execute lists. Shares one predicate with ``_operand_signature`` (via
        ``operand_participates``) so the execute-visible set and the sharing
        signature cannot drift apart.
        """
        return operand_participates(opnd.fieldless, opnd.operand_type)

    @staticmethod
    def _fieldless_caps_stmt(opnd_name: str, operand_type: str) -> str:
        """C++ statement marking a fieldless operand and applying its caps.

        Emitted for every fieldless operand in place of a bare fieldless
        marker: it marks the operand fieldless (structural) AND applies its
        runtime capability policy (readable / writable / vgpr) from the shared
        fieldless operand policy table, so the normal accessors query stored
        flags instead of re-checking ``fieldless_`` plus operand type.
        """
        caps = fieldless_policy(operand_type).caps

        def _b(v: bool) -> str:
            return 'true' if v else 'false'

        return (
            f'{opnd_name}.apply_fieldless_caps('
            f'{_b(caps.reads_value)}, {_b(caps.writable)}, {_b(caps.is_vgpr)});'
        )

    def _has_machine_inst_struct(self, struct_name: str) -> bool:
        return struct_name in {
            f'{enc.fmt_enc_name}MachineInst' for enc in self.isa_spec.inst_encodings
        }

    def _machine_inst_struct_fields(self, struct_name: str) -> frozenset[str]:
        for enc in self.isa_spec.inst_encodings:
            if f'{enc.fmt_enc_name}MachineInst' == struct_name:
                return frozenset(
                    field.name for field in getattr(enc, 'ucode_fields', ())
                )
        return frozenset()

    def _machine_inst_struct_has_field(
        self, struct_name: str | None, field: str
    ) -> bool:
        if struct_name is None:
            return False
        return field in self._machine_inst_struct_fields(struct_name)

    def _supports_vop_dpp8(self) -> bool:
        return any(
            self._has_machine_inst_struct(f'{base}VopDpp8MachineInst')
            for base in ('Vop1', 'Vop2', 'Vopc')
        )

    def _vop_dpp_struct_names(self, enc_name: str) -> tuple[str | None, str | None]:
        enc_upper = enc_name.upper()
        dpp_bases = {
            'ENC_VOP1': ('Vop1',),
            'ENC_VOP2': ('Vop2',),
            'ENC_VOPC': ('Vopc', 'Vop1'),
            'ENC_VOP3': ('Vop3',),
            'ENC_VOP3P': ('Vop3p',),
            'VOP3_SDST_ENC': ('Vop3SdstEnc',),
        }
        dpp8_bases = {
            'ENC_VOP1': 'Vop1',
            'ENC_VOP2': 'Vop2',
            'ENC_VOPC': 'Vopc',
            'ENC_VOP3': 'Vop3',
            'ENC_VOP3P': 'Vop3p',
            'VOP3_SDST_ENC': 'Vop3SdstEnc',
        }
        enc_bases = dpp_bases.get(enc_upper)
        if enc_bases is None:
            return None, None

        is_rdna = any(
            ie.enc_name.startswith('VOP1_VOP_DPP16')
            for ie in self.isa_spec.inst_encodings
        )
        dpp_suffix = 'VopDpp16' if is_rdna else 'VopDpp'
        dpp_struct = None
        for enc_base in enc_bases:
            candidate = f'{enc_base}{dpp_suffix}MachineInst'
            if self._has_machine_inst_struct(candidate):
                dpp_struct = candidate
                break

        dpp8_struct = None
        dpp8_base = dpp8_bases.get(enc_upper)
        if dpp8_base is not None:
            candidate = f'{dpp8_base}VopDpp8MachineInst'
            if self._has_machine_inst_struct(candidate):
                dpp8_struct = candidate

        return dpp_struct, dpp8_struct

    def _supports_vop_dpp_encoding(self, enc_name: str) -> bool:
        dpp_struct, dpp8_struct = self._vop_dpp_struct_names(enc_name)
        return (
            dpp_struct is not None and self._supports_dpp_for_encoding(enc_name)
        ) or dpp8_struct is not None

    def _supports_sdwa_for_encoding(self, enc_name: str) -> bool:
        """Whether any instruction in an encoding has an SDWA alternate."""
        enc_upper = enc_name.upper()
        if enc_upper not in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
            return False
        return any(
            self._instruction_supports_sdwa(inst, enc_name)
            for inst_enc in self.isa_spec.inst_encodings
            if inst_enc.enc_name.upper() == enc_upper
            for inst in inst_enc.insts
        )

    def _supports_dpp_for_encoding(self, enc_name: str) -> bool:
        enc_upper = enc_name.upper()
        if enc_upper != 'ENC_VOPC':
            return True

        # RDNA DPP16 uses shared VOP1/VOPC machine-inst layouts. RDNA1/2 have
        # VOP1/VOP2 DPP16 encodings but their VOPC XML explicitly forbids DPP,
        # so VOPC needs an encoding-level availability check.
        has_rdna_dpp16 = any(
            ie.enc_name.startswith('VOP1_VOP_DPP16')
            for ie in self.isa_spec.inst_encodings
        )
        if not has_rdna_dpp16:
            return True

        return any(
            ie.enc_name.startswith('VOPC_VOP_DPP16')
            for ie in self.isa_spec.inst_encodings
        )

    def _instruction_supports_modifier_encoding(
        self, inst: Instruction, parent_enc_name: str, modifier: str
    ) -> bool:
        """Whether the MR ISA lists an alternate modifier encoding.

        Alternate instruction entries are not emitted as independent classes,
        but their presence is still the instruction-level legality fact. Keep
        that fact separate from architecture-wide machine-inst availability.
        """
        if inst.available_encodings is None:
            raise ValueError(
                f'{inst.name}: cannot determine {modifier.upper()} support '
                'without instruction encoding provenance'
            )

        parent_upper = parent_enc_name.upper()
        for enc_name in inst.available_encodings:
            enc_upper = enc_name.upper()
            if modifier == 'dpp':
                matches = '_VOP_DPP' in enc_upper and '_VOP_DPP8' not in enc_upper
            elif modifier == 'dpp8':
                matches = '_VOP_DPP8' in enc_upper
            elif modifier == 'sdwa':
                matches = '_VOP_SDWA' in enc_upper
            else:
                raise ValueError(f'unknown VOP modifier encoding: {modifier}')
            if (
                matches
                and self.isa_spec.profile.is_alt_encoding(enc_name)
                and self.isa_spec.profile.derive_parent_enc_name(enc_name).upper()
                == parent_upper
            ):
                return True
        return False

    def _instruction_supports_dpp(self, inst: Instruction, enc_name: str) -> bool:
        return (
            self._supports_dpp_for_encoding(enc_name)
            and self._dpp_opcode_rule(inst, enc_name) is not DppOpcodeRule.FORBID
            and self._instruction_supports_modifier_encoding(inst, enc_name, 'dpp')
        )

    def _instruction_supports_dpp8(self, inst: Instruction, enc_name: str) -> bool:
        return self._dpp_opcode_rule(
            inst, enc_name
        ) is DppOpcodeRule.ALLOW and self._instruction_supports_modifier_encoding(
            inst, enc_name, 'dpp8'
        )

    def _instruction_supports_sdwa(self, inst: Instruction, enc_name: str) -> bool:
        has_modifier_encoding = self._instruction_supports_modifier_encoding(
            inst, enc_name, 'sdwa'
        )
        return self.isa_spec.profile.supports_sdwa_opcode(
            enc_name,
            inst.name,
            has_modifier_encoding=has_modifier_encoding,
        )

    def _instruction_supports_literal_encoding(
        self, inst: Instruction, parent_enc_name: str, width: int
    ) -> bool:
        """Whether the instruction lists a literal alternate for this format."""
        if inst.available_encodings is None:
            raise ValueError(
                f'{inst.name}: cannot determine literal support without '
                'instruction encoding provenance'
            )

        suffix = '_INST_LITERAL64' if width == 64 else '_INST_LITERAL'
        parent_upper = parent_enc_name.upper()
        for enc_name in inst.available_encodings:
            enc_upper = enc_name.upper()
            if not enc_upper.endswith(suffix):
                continue
            if (
                self.isa_spec.profile.is_alt_encoding(enc_name)
                and self.isa_spec.profile.derive_parent_enc_name(enc_name).upper()
                == parent_upper
            ):
                return True
        return False

    def _instruction_literal_support(
        self, inst: Instruction, inst_enc: InstEncoding
    ) -> tuple[bool, bool, bool]:
        """Return whether support is tracked and the accepted literal widths."""
        tracks_support = self._supports_simm64_literal_operands()
        if not tracks_support:
            return False, True, False

        base_enc_name = self._instruction_base_encoding_name(inst)
        supports_literal32 = self._instruction_supports_literal_encoding(
            inst, base_enc_name, 32
        )
        supports_literal64 = bool(self._literal64_condition_names(inst_enc)) and (
            self._instruction_supports_literal_encoding(inst, base_enc_name, 64)
        )
        return tracks_support, supports_literal32, supports_literal64

    @staticmethod
    def _literal_support_enum_name(
        supports_literal32: bool, supports_literal64: bool
    ) -> str:
        if supports_literal32 and supports_literal64:
            return 'Both'
        if supports_literal32:
            return 'Literal32'
        if supports_literal64:
            return 'Literal64'
        return 'None'

    @staticmethod
    def _active_literal_selector_fields(
        inst: Instruction, literal_fields: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the encoded source fields that are literal selectors for inst."""
        non_literal_operand_types = ('OPR_SENDMSG', 'OPR_SENDMSG_RTN')
        return tuple(
            field
            for field in literal_fields
            if any(
                opnd.is_input
                and opnd.name == field
                and opnd.operand_type not in non_literal_operand_types
                for opnd in inst.operands
            )
        )

    def _instruction_base_encoding_name(self, inst: Instruction) -> str:
        if inst.is_implied_literal_enc:
            return self.isa_spec.profile.derive_parent_enc_name(inst.enc_name)
        return inst.enc_name

    def _modifier_feature_mask(
        self, inst: Instruction, parent_enc_name: str, modifier: str
    ) -> int:
        """Return requirements attached to a runtime modifier encoding."""
        mask = 0
        parent_upper = parent_enc_name.upper()
        for enc_name, feature_mask in inst.encoding_feature_masks.items():
            enc_upper = enc_name.upper()
            if modifier == 'dpp':
                matches = '_VOP_DPP' in enc_upper and '_VOP_DPP8' not in enc_upper
            elif modifier == 'dpp8':
                matches = '_VOP_DPP8' in enc_upper
            elif modifier == 'sdwa':
                matches = '_VOP_SDWA' in enc_upper
            else:
                raise ValueError(f'unknown VOP modifier encoding: {modifier}')
            if (
                matches
                and self.isa_spec.profile.derive_parent_enc_name(enc_name).upper()
                == parent_upper
            ):
                mask |= feature_mask
        return mask

    def _uses_full_dpp_write_mask(self, enc_name: str) -> bool:
        return self._supports_dpp_for_encoding(enc_name)

    def _dpp_opcode_rule(self, inst: Instruction, enc_name: str) -> DppOpcodeRule:
        # Implied-literal VALU forms consume the DWORD immediately following
        # the base instruction. DPP16, DPP8, and SDWA use that same extension
        # DWORD, so the formats cannot be combined even when an opcode
        # limitation table happens to omit the instruction. The SDWA half of
        # this invariant is emitted separately in the constructor below.
        if inst.is_implied_literal_enc:
            return DppOpcodeRule.FORBID

        src0 = next(
            (op for op in inst.operands if op.is_input and not op.fieldless),
            None,
        )
        return self.isa_spec.profile.dpp_opcode_rule(
            enc_name,
            inst.name,
            src0_size_bits=src0.size if src0 is not None else None,
        )

    def _supports_dpp_for_instruction(self, inst: Instruction, enc_name: str) -> bool:
        return self._instruction_supports_dpp(inst, enc_name)

    def gen_all(self) -> None:
        """Generate all C++ objects.

        Note: ``gen_isa_types()`` is intentionally excluded because its
        output file (``isa.h``) contains hand-maintained content (ISA
        traits, status register bitfields, etc.) that the generator does
        not produce. Use ``--gen-isa`` only for bootstrapping a new arch.
        """
        self.gen_machine_inst_encodings()
        self.gen_opcode_constants()
        self.gen_instruction_builders()
        self.gen_encodings()
        self.gen_operand_types()
        self.gen_operand()
        # VOPD is generated from a hand-written C++ template because the XML
        # describes the packed dual-slot encoding, while the normal emitters
        # model one instruction and one operand list at a time. Keeping the
        # template here preserves one-step regeneration for VOPD-capable profiles.
        self.gen_vopd()
        self.gen_insts()
        self.gen_execution_backend()
        self.gen_isa_features()
        self.gen_decoder()
        self.gen_test_encodings()

    def gen_isa_features(self) -> None:
        """Emit stable feature and concrete-variant masks for this ISA input."""
        if not self.isa_spec.isa_features and not self.isa_spec.isa_variants:
            return
        guard = (
            f'ROCJITSU_ISA_ARCH_AMDGPU_GENERATED_'
            f'{self.generated_dir_name.upper()}_ISA_FEATURES_H_'
        )
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f'#ifndef {guard}',
            f'#define {guard}',
            '',
            '#include <cstdint>',
            '',
            f'namespace rocjitsu::{self.cpp_namespace} {{',
            '',
        ]

        def cpp_name(name: str) -> str:
            return ''.join(part[:1].upper() + part[1:] for part in name.split('_'))

        for index, feature in enumerate(self.isa_spec.isa_features):
            lines.append(
                f'inline constexpr uint64_t kIsaFeature{cpp_name(feature)} = '
                f'uint64_t{{1}} << {index};'
            )
        if self.isa_spec.isa_features:
            lines.append('')
        for variant, mask in self.isa_spec.isa_variants.items():
            lines.append(
                f'inline constexpr uint64_t k{cpp_name(variant)}IsaFeatures = '
                f'uint64_t{{{mask}}};'
            )
        lines += [
            '',
            f'}} // namespace rocjitsu::{self.cpp_namespace}',
            '',
            f'#endif // {guard}',
            '',
        ]
        out_path = os.path.join(
            self.out_path, self.generated_dir_name, 'isa_features.h'
        )
        with open(out_path, 'w') as output:
            output.write('\n'.join(lines))

    def gen_execution_backend(self) -> None:
        """Emit one immutable execution descriptor for a split ISA profile."""
        if not self.isa_spec.profile.split_execution_sources:
            return

        arch = self.cpp_namespace
        generated_arch = self.config.generated_include(self.generated_dir_name)
        handwritten_arch = self.config.handwritten_include(self.handwritten_dir_name)
        execution_ids = ''.join(
            f'  {class_name},\n' for class_name in self._split_execution_classes
        )
        header = textwrap.dedent(f'''\
            {CppFile._prologue_comment()}
            #pragma once

            #include "{handwritten_arch}/isa.h"
            #include "rocjitsu/isa/execution_backend.h"

            namespace rocjitsu {{
            namespace {arch} {{

            /// @brief Named entries for the dense execution callback table.
            ///
            /// Model constructors use these names instead of opaque ordinals.
            /// Keep this enum and kInstructionCallbacks generated from the same
            /// ordered class list.
            enum class InstructionExecutionId : size_t {{
            {execution_ids}  Count,
            }};

            const IsaExecutionBackend &execution_backend();

            }} // namespace {arch}
            }} // namespace rocjitsu
            ''')
        entries = ''.join(
            f'    &execute_with_backend<{class_name}>,\n'
            for class_name in self._split_execution_classes
        )
        source = textwrap.dedent(f'''\
            {CppFile._prologue_comment()}
            #include "{generated_arch}/execution_backend.h"
            #include "{generated_arch}/insts.h"
            #include "{generated_arch}/operand.h"

            #include <array>

            namespace rocjitsu {{
            namespace {arch} {{
            namespace {{

            template <typename Derived>
            void execute_with_backend(Instruction &instruction, void *context) {{
              // execute_impl may construct temporary operands (for example,
              // indexed SOP1/VOP1 sources). Re-enter the ISA scope so those
              // operands capture the same backend as decode-time operands.
              ScopedIsaExecutionBackend scope(&execution_backend());
              static_cast<Derived &>(instruction).execute_impl(
                  *static_cast<Isa::Context *>(context));
            }}

            constexpr size_t kInstructionCallbackCount =
                static_cast<size_t>(InstructionExecutionId::Count);
            using InstructionCallbackTable =
                std::array<Instruction::ExecuteFn, kInstructionCallbackCount>;
            constexpr InstructionCallbackTable kInstructionCallbacks{{{{
            {entries}    }}}};

            }} // namespace

            const IsaExecutionBackend &execution_backend() {{
              static const IsaExecutionBackend backend{{
                  .instruction_callbacks = kInstructionCallbacks.data(),
                  .instruction_callback_count = kInstructionCallbacks.size(),
                  .operand_backend = Operand::full_execution_backend(),
              }};
              return backend;
            }}

            }} // namespace {arch}
            }} // namespace rocjitsu
            ''')
        arch_dir = os.path.join(self.out_path, self.generated_dir_name)
        os.makedirs(arch_dir, exist_ok=True)
        with open(os.path.join(arch_dir, 'execution_backend.h'), 'w') as f:
            f.write(header)
        with open(os.path.join(arch_dir, 'execution_backend_exec.cpp'), 'w') as f:
            f.write(source)

    def _supports_generated_vopd(self) -> bool:
        return self.isa_spec.profile.has_vopd

    def gen_vopd(self) -> None:
        """Generate VOPD dual-issue decoder/executor files.

        VOPD is skipped by the normal XML instruction generation because it
        uses a dual-slot encoding with bespoke operand packing. Keeping the
        target-specific C++ body here lets the same regeneration path recreate
        the VOPD files and generated decoder-table entries together.
        """
        if not self._supports_generated_vopd():
            return

        logical_arch = self.isa_spec.arch_name
        arch = self.cpp_namespace
        vopd_encoding_prefixes = self.isa_spec.profile.vopd_encoding_prefixes
        vopd_prefixes = [
            prefix for prefix in vopd_encoding_prefixes if not prefix.is_vopd3
        ]
        vopd3_prefixes = [
            prefix for prefix in vopd_encoding_prefixes if prefix.is_vopd3
        ]
        if len(vopd_prefixes) != 1 or len(vopd3_prefixes) > 1:
            raise ValueError(
                f'{logical_arch} has invalid VOPD encoding prefix metadata'
            )
        vopd_prefix = vopd_prefixes[0]
        vopd3_prefix = vopd3_prefixes[0] if vopd3_prefixes else None
        has_vopd3 = vopd3_prefix is not None
        if self.isa_spec.profile.split_execution_sources:
            vopd_exec_fn = self._split_execute_expr('Vopd')
            vopd_registration_decl = ''
            vopd_registration_def = ''
        else:
            vopd_exec_fn = 'make_exec_fn<Vopd>()'
            vopd_registration_decl = ''
            vopd_registration_def = ''

        def cpp_block(text: str) -> str:
            return textwrap.dedent(text).strip('\n')

        vopd_slot_ops = self.isa_spec.profile.vopd_slot_ops
        if not vopd_slot_ops:
            raise ValueError(
                f'{logical_arch} has VOPD enabled without a slot opcode table'
            )
        vopd_slot_op_names = {op.enum_name for op in vopd_slot_ops}

        def has_op(enum_name: str) -> bool:
            return enum_name in vopd_slot_op_names

        def case_labels(enum_names: tuple[str, ...]) -> str:
            return ''.join(
                f'  case k{enum_name}:\n'
                for enum_name in enum_names
                if has_op(enum_name)
            )

        def case_block(enum_names: tuple[str, ...], body: str) -> str:
            labels = case_labels(enum_names)
            if not labels:
                return ''
            return labels + cpp_block(body)

        def join_cases(*cases: str) -> str:
            return '\n'.join(case for case in cases if case)

        vopd_slot_constants = '\n'.join(
            f'constexpr uint16_t k{op.enum_name} = {op.opcode};' for op in vopd_slot_ops
        )
        vopd_execution_slot_constants = '\n'.join(
            f'[[maybe_unused]] constexpr uint16_t k{op.enum_name} = {op.opcode};'
            for op in vopd_slot_ops
        )
        vopd_op_name_cases = '\n'.join(
            f'  case k{op.enum_name}:\n    return "{op.mnemonic}";'
            for op in vopd_slot_ops
        )

        def opcode_mask(name: str, opcodes: frozenset[int]) -> str:
            unknown_opcodes = opcodes.difference(op.opcode for op in vopd_slot_ops)
            if unknown_opcodes:
                raise ValueError(
                    f'{logical_arch} {name} references unknown VOPD opcodes: '
                    f'{sorted(unknown_opcodes)}'
                )
            mask = sum(1 << opcode for opcode in opcodes)
            return f'constexpr uint64_t {name} = 0x{mask:016X}ULL;'

        opcode_masks = [
            opcode_mask('kVopdXOpcodeMask', self.isa_spec.profile.vopd_x_slot_opcodes),
            opcode_mask('kVopdYOpcodeMask', self.isa_spec.profile.vopd_y_slot_opcodes),
        ]
        if has_vopd3:
            opcode_masks.extend(
                (
                    opcode_mask(
                        'kVopd3XOpcodeMask',
                        self.isa_spec.profile.vopd3_x_slot_opcodes,
                    ),
                    opcode_mask(
                        'kVopd3YOpcodeMask',
                        self.isa_spec.profile.vopd3_y_slot_opcodes,
                    ),
                )
            )
        vopd_opcode_validation_helpers = '\n'.join(opcode_masks) + textwrap.dedent('''

            bool is_valid_opcode(uint16_t opcode, uint64_t mask) {
              return opcode < 64 && (mask & (1ULL << opcode));
            }
        ''')
        vopd_src_neg_case_labels = case_labels(
            (
                'VopdFmacF32',
                'VopdFmaakF32',
                'VopdFmamkF32',
                'VopdMulF32',
                'VopdAddF32',
                'VopdSubF32',
                'VopdSubrevF32',
                'VopdMulDx9ZeroF32',
                'VopdMaxF32',
                'VopdMinF32',
                'VopdMaxNumF32',
                'VopdMinNumF32',
                'VopdFmaF32',
                'VopdCndmaskB32',
            )
        ).rstrip()
        vopd_execute_slot_cases = join_cases(
            *(
                case_block(
                    ('VopdFmacF32',),
                    '''
                    {
                      float result = std::fma(std::bit_cast<float>(src0),
                                              std::bit_cast<float>(src1),
                                              std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane(*slot.dst, lane)));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdFmaakF32',),
                    '''
                    {
                      float result = std::fma(std::bit_cast<float>(src0),
                                              std::bit_cast<float>(src1),
                                              std::bit_cast<float>(src2));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdFmamkF32',),
                    '''
                    {
                      float result = std::fma(std::bit_cast<float>(src0),
                                              std::bit_cast<float>(src2),
                                              std::bit_cast<float>(src1));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdMulF32',),
                    '''
                    {
                      float result = std::bit_cast<float>(src0) * std::bit_cast<float>(src1);
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdMulDx9ZeroF32',),
                    '''
                    {
                      float lhs = std::bit_cast<float>(src0);
                      float rhs = std::bit_cast<float>(src1);
                      if (lhs == 0.0f || rhs == 0.0f)
                        return std::bit_cast<uint32_t>(0.0f);
                      return std::bit_cast<uint32_t>(lhs * rhs);
                    }
                    ''',
                ),
                case_block(
                    ('VopdAddF32',),
                    '''
                    {
                      float result = std::bit_cast<float>(src0) + std::bit_cast<float>(src1);
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdSubF32',),
                    '''
                    {
                      float result = std::bit_cast<float>(src0) - std::bit_cast<float>(src1);
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdSubrevF32',),
                    '''
                    {
                      float result = std::bit_cast<float>(src1) - std::bit_cast<float>(src0);
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdMovB32',),
                    '''
                      return src0;
                    ''',
                ),
                case_block(
                    ('VopdCndmaskB32',),
                    '''
                    {
                      uint64_t condition = slot.uses_vcc ? wf.vcc() : amdgpu::read_wave_mask_scalar(*slot.src2, wf);
                      return ((condition >> lane) & 1u) ? src1 : src0;
                    }
                    ''',
                ),
                case_block(
                    ('VopdMaxF32', 'VopdMaxNumF32'),
                    '''
                    {
                      float result = std::fmax(std::bit_cast<float>(src0),
                                               std::bit_cast<float>(src1));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdMinF32', 'VopdMinNumF32'),
                    '''
                    {
                      float result = std::fmin(std::bit_cast<float>(src0),
                                               std::bit_cast<float>(src1));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdAddNcU32',),
                    '''
                      return src0 + src1;
                    ''',
                ),
                case_block(
                    ('VopdLshlrevB32',),
                    '''
                      return src1 << (src0 & 31u);
                    ''',
                ),
                case_block(
                    ('VopdAndB32',),
                    '''
                      return src0 & src1;
                    ''',
                ),
                case_block(
                    ('VopdBitop2B32',),
                    '''
                      return bitop2(src0, src1, slot.src2_imm);
                    ''',
                ),
                case_block(
                    ('VopdFmaF32',),
                    '''
                    {
                      float result = std::fma(std::bit_cast<float>(src0),
                                              std::bit_cast<float>(src1),
                                              std::bit_cast<float>(src2));
                      return std::bit_cast<uint32_t>(result);
                    }
                    ''',
                ),
                case_block(
                    ('VopdSubNcU32',),
                    '''
                      return src0 - src1;
                    ''',
                ),
                case_block(
                    ('VopdLshrrevB32',),
                    '''
                      return src1 >> (src0 & 31u);
                    ''',
                ),
                case_block(
                    ('VopdAshrrevI32',),
                    '''
                      return static_cast<uint32_t>(static_cast<int32_t>(src1) >> (src0 & 31u));
                    ''',
                ),
                case_block(
                    ('VopdMaxI32',),
                    '''
                      return static_cast<uint32_t>(std::max(static_cast<int32_t>(src0),
                                                            static_cast<int32_t>(src1)));
                    ''',
                ),
                case_block(
                    ('VopdMinI32',),
                    '''
                      return static_cast<uint32_t>(std::min(static_cast<int32_t>(src0),
                                                            static_cast<int32_t>(src1)));
                    ''',
                ),
            )
        )
        vopd_src2_operand_exprs = ['opx_ == kVopdCndmaskB32']
        if has_op('VopdFmaF32'):
            vopd_src2_operand_exprs.append('opx_ == kVopdFmaF32')
        if has_op('VopdFmaF64'):
            vopd_src2_operand_exprs.append('opx_ == kVopdFmaF64')
        vopd_x_has_src2_operand = ' || '.join(vopd_src2_operand_exprs)
        vopd_y_has_src2_operand = vopd_x_has_src2_operand.replace('opx_', 'opy_')
        vopd_x_src2_is_imm = (
            'opx_ == kVopdBitop2B32' if has_op('VopdBitop2B32') else 'false'
        )
        vopd_y_src2_is_imm = vopd_x_src2_is_imm.replace('opx_', 'opy_')
        vopd_add_slot_source_cases = join_cases(
            case_block(
                ('VopdFmacF32',),
                '''
                  add_src(slot.dst);
                  add_src(slot.src0);
                  add_src(slot.src1);
                  break;
                ''',
            ),
            case_block(
                ('VopdFmaF32', 'VopdFmaF64'),
                '''
                  add_src(slot.src0);
                  add_src(slot.src1);
                  add_src(slot.src2);
                  break;
                ''',
            ),
            case_block(
                ('VopdMovB32',),
                '''
                  add_src(slot.src0);
                  break;
                ''',
            ),
            case_block(
                ('VopdCndmaskB32',),
                '''
                  add_src(slot.src0);
                  add_src(slot.src1);
                  if (!slot.uses_vcc)
                    add_src(slot.src2);
                  break;
                ''',
            ),
        )
        vopd_format_slot_cases = join_cases(
            case_block(
                ('VopdMovB32',),
                '''
                  out += slot.dst->name() + ", " + slot.src0->name();
                  break;
                ''',
            ),
            case_block(
                ('VopdCndmaskB32',),
                '''
                  out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                  if (!slot.uses_vcc)
                    out += ", " + slot.src2->name();
                  break;
                ''',
            ),
            case_block(
                ('VopdBitop2B32',),
                '''
                  out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                  out += std::format(" bitop3:0x{:02x}", slot.src2_imm & 0xFF);
                  break;
                ''',
            ),
            case_block(
                ('VopdFmaF32', 'VopdFmaF64'),
                '''
                  out += operand_list(*slot.dst, *slot.src0, *slot.src1) + ", " +
                         slot.src2->name();
                  break;
                ''',
            ),
            case_block(
                ('VopdFmaakF32',),
                '''
                  out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                  out += std::format(", 0x{:08x}", slot.src2_imm);
                  break;
                ''',
            ),
            case_block(
                ('VopdFmamkF32',),
                '''
                  out += slot.dst->name() + ", " + slot.src0->name();
                  out += std::format(", 0x{:08x}, ", slot.src2_imm);
                  out += slot.src1->name();
                  break;
                ''',
            ),
        )
        vopd_float64_case_labels = case_labels(
            (
                'VopdFmaF64',
                'VopdAddF64',
                'VopdMulF64',
                'VopdMaxNumF64',
                'VopdMinNumF64',
            )
        ).rstrip()
        vopd3_src0_type = (
            'OperandType::OPR_SRC_SIMPLE'
            if 'OPR_SRC_SIMPLE' in self.isa_spec.operand_types
            else 'OperandType::OPR_SRC'
        )
        vopd_src0_type_expr = (
            vopd3_src0_type
            if vopd3_src0_type == 'OperandType::OPR_SRC'
            else f'vopd3 ? {vopd3_src0_type} : OperandType::OPR_SRC'
        )
        vopd3_unused_attr = (
            '' if 'vopd3 ?' in vopd_src0_type_expr else '[[maybe_unused]] '
        )
        # Preserve the visual separator before bitop2 for VOPD profiles that
        # do not have the VOPD3-only declarations.
        vopd3_header_decls = '\n'
        vopd3_model_helpers = ''
        vopd3_f64_helpers = ''
        vopd3_execute_slot_cases = ''
        vopd3_constructor_branch = ''
        vopd3_constructor_close = ''
        vopd3_validation_branch = ''
        vopd3_init_operands_prefix = ''
        vopd3_init_operands_suffix = ''
        vopdxy_bits_decl = cpp_block('''
                constexpr uint32_t x_bits = 32;
                constexpr uint32_t y_bits = 32;
            ''')
        execute_impl_body = cpp_block('''
              uint64_t exec = wf.exec();
              for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
                if (!(exec & (1ULL << lane)))
                  continue;
                uint32_t x_result = execute_slot(x_, wf, lane);
                uint32_t y_result = execute_slot(y_, wf, lane);
                amdgpu::RegisterAccess(wf).write_lane(*x_.dst, lane, x_result);
                amdgpu::RegisterAccess(wf).write_lane(*y_.dst, lane, y_result);
              }
            ''')
        if has_vopd3:
            vopd3_header_decls = cpp_block('''
              static bool is_float64_op(uint16_t op);
              static uint64_t apply_neg64(uint64_t value, uint8_t neg_bits, uint8_t src_idx);
              static uint64_t execute_slot64(const Slot &slot, amdgpu::Wavefront &wf,
                                             uint32_t lane);
            ''') + '\n'
            vopd3_model_helpers = cpp_block('''
            bool Vopd::is_float64_op(uint16_t op) {
              switch (op) {
            @VOPD_FLOAT64_CASES@
                return true;
              default:
                return false;
              }
            }
            ''')
            vopd3_model_helpers = vopd3_model_helpers.replace(
                '@VOPD_FLOAT64_CASES@', vopd_float64_case_labels
            )
            vopd3_f64_helpers = cpp_block('''

            uint64_t Vopd::apply_neg64(uint64_t value, uint8_t neg_bits, uint8_t src_idx) {
              return (neg_bits & (1u << src_idx)) ? (value ^ 0x8000000000000000ULL) : value;
            }

            uint64_t Vopd::execute_slot64(const Slot &slot, amdgpu::Wavefront &wf,
                                          uint32_t lane) {
              uint64_t src0 = apply_neg64(amdgpu::RegisterAccess(wf).read_lane64(*slot.src0, lane), slot.neg, 0);
              uint64_t src1 = apply_neg64(amdgpu::RegisterAccess(wf).read_lane64(*slot.src1, lane), slot.neg, 1);

              switch (slot.op) {
              case kVopdFmaF64: {
                uint64_t src2 = apply_neg64(amdgpu::RegisterAccess(wf).read_lane64(*slot.src2, lane), slot.neg, 2);
                double result = std::fma(std::bit_cast<double>(src0),
                                         std::bit_cast<double>(src1),
                                         std::bit_cast<double>(src2));
                return std::bit_cast<uint64_t>(result);
              }
              case kVopdAddF64: {
                double result = std::bit_cast<double>(src0) + std::bit_cast<double>(src1);
                return std::bit_cast<uint64_t>(result);
              }
              case kVopdMulF64: {
                double result = std::bit_cast<double>(src0) * std::bit_cast<double>(src1);
                return std::bit_cast<uint64_t>(result);
              }
              case kVopdMinNumF64: {
                double result = std::fmin(std::bit_cast<double>(src0),
                                          std::bit_cast<double>(src1));
                return std::bit_cast<uint64_t>(result);
              }
              case kVopdMaxNumF64: {
                double result = std::fmax(std::bit_cast<double>(src0),
                                          std::bit_cast<double>(src1));
                return std::bit_cast<uint64_t>(result);
              }
              default:
                throw util::UnimplementedInst(op_name(slot.op));
              }
            }
            ''')
            vopd3_execute_slot_cases = cpp_block('''
            @VOPD_FLOAT64_CASES@
                throw util::UnimplementedInst(
                    std::string(op_name(slot.op)) +
                    " (VOPD F64 execution requires 64-bit VGPR pair support)");
            ''')
            vopd3_execute_slot_cases = vopd3_execute_slot_cases.replace(
                '@VOPD_FLOAT64_CASES@', vopd_float64_case_labels
            )
            vopd3_constructor_branch = cpp_block('''
              if ((word0_ >> @VOPD3_PREFIX_SHIFT@) == @VOPD3_PREFIX@) {
                format_ = Format::Vopd3;
                size_ = 12;
                encoding_id_ = @VOPD3_PREFIX@;
                word2_ = words[2];
                opx_ = static_cast<uint16_t>((word0_ >> 18) & 0x3F);
                opy_ = static_cast<uint16_t>((word0_ >> 12) & 0x3F);
                uint16_t srcx0 = static_cast<uint16_t>(word0_ & 0x1FF);
                uint16_t srcy0 = static_cast<uint16_t>(word1_ & 0x1FF);
                negx_ = static_cast<uint8_t>((word1_ >> 9) & 0x7);
                negy_ = static_cast<uint8_t>((word1_ >> 12) & 0x7);
                uint16_t vsrcx1 = static_cast<uint16_t>((word1_ >> 16) & 0xFF);
                uint16_t vsrcx2 = static_cast<uint16_t>((word1_ >> 24) & 0xFF);
                uint16_t vdstx = static_cast<uint16_t>(word2_ & 0xFF);
                uint16_t vsrcy1 = static_cast<uint16_t>((word2_ >> 8) & 0xFF);
                uint16_t vsrcy2 = static_cast<uint16_t>((word2_ >> 16) & 0xFF);
                uint16_t vdsty = static_cast<uint16_t>((word2_ >> 24) & 0xFF);

                uint32_t x_bits = is_float64_op(opx_) ? 64 : 32;
                uint32_t y_bits = is_float64_op(opy_) ? 64 : 32;
                dstx_ = Operand(x_bits, OperandType::OPR_VGPR, vdstx);
                dsty_ = Operand(y_bits, OperandType::OPR_VGPR, vdsty);
                srcx0_ = make_src0(x_bits, true, false, 0, srcx0);
                srcy0_ = make_src0(y_bits, true, false, 0, srcy0);
                srcx1_ = Operand(x_bits, OperandType::OPR_VGPR, vsrcx1);
                srcy1_ = Operand(y_bits, OperandType::OPR_VGPR, vsrcy1);
                srcx2_ = (opx_ == kVopdCndmaskB32) ? Operand(64, OperandType::OPR_SREG, vsrcx2)
                                                   : Operand(x_bits, OperandType::OPR_VGPR, vsrcx2);
                srcy2_ = (opy_ == kVopdCndmaskB32) ? Operand(64, OperandType::OPR_SREG, vsrcy2)
                                                   : Operand(y_bits, OperandType::OPR_VGPR, vsrcy2);
              } else {
            ''')
            vopd3_constructor_branch = vopd3_constructor_branch.replace(
                '@VOPD3_PREFIX_SHIFT@', str(32 - vopd3_prefix.prefix_bits)
            ).replace('@VOPD3_PREFIX@', f'0x{vopd3_prefix.prefix:X}')
            vopd3_validation_branch = cpp_block('''
              if ((word0 >> @VOPD3_PREFIX_SHIFT@) == @VOPD3_PREFIX@) {
                const uint16_t opx = static_cast<uint16_t>((word0 >> 18) & 0x3F);
                const uint16_t opy = static_cast<uint16_t>((word0 >> 12) & 0x3F);
                if (!is_valid_opcode(opx, kVopd3XOpcodeMask)) [[unlikely]]
                  return emit_error.emit() << "invalid VOPD3 X opcode";
                if (!is_valid_opcode(opy, kVopd3YOpcodeMask)) [[unlikely]]
                  return emit_error.emit() << "invalid VOPD3 Y opcode";
                const uint16_t srcx0 = static_cast<uint16_t>(word0 & 0x1FF);
                const uint16_t srcy0 = static_cast<uint16_t>(word1 & 0x1FF);
                if (srcx0 == 254 || srcx0 == 255 || srcy0 == 254 || srcy0 == 255) [[unlikely]]
                  return emit_error.emit() << "VOPD3 does not support literal selectors";
                const uint32_t word2 = words[2];
                const uint16_t vdstx = static_cast<uint16_t>(word2 & 0xFF);
                const uint16_t vdsty = static_cast<uint16_t>((word2 >> 24) & 0xFF);
                const uint32_t x_end = vdstx + (is_float64_op(opx) ? 2 : 1);
                const uint32_t y_end = vdsty + (is_float64_op(opy) ? 2 : 1);
                if (vdstx < y_end && vdsty < x_end) [[unlikely]]
                  return emit_error.emit() << "VOPD3 destination ranges overlap";
                return Result::success();
              }
            ''')
            vopd3_validation_branch = vopd3_validation_branch.replace(
                '@VOPD3_PREFIX_SHIFT@', str(32 - vopd3_prefix.prefix_bits)
            ).replace('@VOPD3_PREFIX@', f'0x{vopd3_prefix.prefix:X}')
            vopd3_constructor_close = '              }'
            vopd3_init_operands_prefix = cpp_block('''
              const bool vopd3 = format_ == Format::Vopd3;

              if (vopd3) {
                x_.has_src2_operand = @VOPD_X_HAS_SRC2_OPERAND@;
                y_.has_src2_operand = @VOPD_Y_HAS_SRC2_OPERAND@;
                x_.src2_is_imm = @VOPD_X_SRC2_IS_IMM@;
                y_.src2_is_imm = @VOPD_Y_SRC2_IS_IMM@;
                x_.src2_imm = static_cast<uint32_t>(srcx2_.encoding_value());
                y_.src2_imm = static_cast<uint32_t>(srcy2_.encoding_value());
              } else {
            ''')
            vopd3_init_operands_prefix = (
                vopd3_init_operands_prefix.replace(
                    '@VOPD_X_HAS_SRC2_OPERAND@', vopd_x_has_src2_operand
                )
                .replace('@VOPD_Y_HAS_SRC2_OPERAND@', vopd_y_has_src2_operand)
                .replace('@VOPD_X_SRC2_IS_IMM@', vopd_x_src2_is_imm)
                .replace('@VOPD_Y_SRC2_IS_IMM@', vopd_y_src2_is_imm)
            )
            vopd3_init_operands_suffix = '              }'
            vopdxy_bits_decl = cpp_block('''
                uint32_t x_bits = is_float64_op(opx_) ? 64 : 32;
                uint32_t y_bits = is_float64_op(opy_) ? 64 : 32;
            ''')
            execute_impl_body = cpp_block('''
              uint64_t exec = wf.exec();
              for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
                if (!(exec & (1ULL << lane)))
                  continue;
                bool x64 = is_float64_op(x_.op);
                bool y64 = is_float64_op(y_.op);
                uint64_t x_result64 = x64 ? execute_slot64(x_, wf, lane) : 0;
                uint64_t y_result64 = y64 ? execute_slot64(y_, wf, lane) : 0;
                uint32_t x_result32 = x64 ? 0 : execute_slot(x_, wf, lane);
                uint32_t y_result32 = y64 ? 0 : execute_slot(y_, wf, lane);
                if (x64)
                  amdgpu::RegisterAccess(wf).write_lane64(*x_.dst, lane, x_result64);
                else
                  amdgpu::RegisterAccess(wf).write_lane(*x_.dst, lane, x_result32);
                if (y64)
                  amdgpu::RegisterAccess(wf).write_lane64(*y_.dst, lane, y_result64);
                else
                  amdgpu::RegisterAccess(wf).write_lane(*y_.dst, lane, y_result32);
              }
            ''')
        vopd_execute_slot_cases = join_cases(
            vopd_execute_slot_cases, vopd3_execute_slot_cases
        )
        vopd_impl_includes = textwrap.dedent('''\
            #include "@GENERATED_ARCH@/vopd.h"
            #include "util/except.h"
            #include "rocjitsu/vm/amdgpu/wavefront.h"
            #include <algorithm>
            #include <bit>
            #include <cmath>
            #include <format>
            #include <string>
            ''')
        if self.isa_spec.profile.split_execution_sources:
            vopd_impl_includes = textwrap.dedent('''\
                #include "@GENERATED_ARCH@/vopd.h"
                #include "@GENERATED_ARCH@/execution_backend.h"
                #include "util/except.h"
                #include <format>
                #include <string>
                ''')
        vopd_impl_includes = vopd_impl_includes.replace(
            '@GENERATED_ARCH@',
            self.config.generated_include(self.generated_dir_name),
        )
        out_dir = os.path.join(self.out_path, self.generated_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        guard = f'ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_VOPD_H_'

        header = (
            textwrap.dedent('''
            // Copyright (c) 2026 Advanced Micro Devices, Inc.
            // SPDX-License-Identifier: MIT
            //
            // AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.
            // See lib/python/amdisa/README.md for regeneration instructions.

            #ifndef @GUARD@
            #define @GUARD@

            #include "@GENERATED_ARCH@/encodings.h"
            #include "@GENERATED_ARCH@/operand.h"
            #include "rocjitsu/isa/decode_result.h"
            #include <cstdint>
            #include <string>

            namespace rocjitsu {
            namespace @ARCH@ {

            class Vopd : public IsaInstruction<Isa>
            {
              public:
              explicit Vopd(const MachineInst *inst);
              static Result validate_encoding(const MachineInst *inst,
                                              const util::DiagnosticEmitter &emit_error);
              void execute_impl(amdgpu::Wavefront &wf);

              private:
              enum class Format : uint8_t { VopdXy@VOPD3_FORMAT_ENUM@ };

              struct Slot {
                uint16_t op = 0;
                Operand *dst = nullptr;
                Operand *src0 = nullptr;
                Operand *src1 = nullptr;
                Operand *src2 = nullptr;
                uint32_t src2_imm = 0;
                uint8_t neg = 0;
                bool has_src2_operand = false;
                bool src2_is_imm = false;
                bool uses_vcc = false;
              };

              static const char *op_name(uint16_t op);
              static bool uses_src_neg_modifier(uint16_t op);
              static uint32_t apply_neg(uint32_t value, uint8_t neg_bits, uint8_t src_idx);
              static uint32_t execute_slot(const Slot &slot, amdgpu::Wavefront &wf,
                                           uint32_t lane);
            @VOPD3_HEADER_DECLS@  static uint32_t bitop2(uint32_t src0, uint32_t src1, uint32_t truth_table);
              @VOPD_REGISTRATION_DECL@std::string format_slot(const Slot &slot) const;
              void init_operands();

              Format format_ = Format::VopdXy;
              uint32_t word0_ = 0;
              uint32_t word1_ = 0;
              uint32_t word2_ = 0;
              uint32_t literal_ = 0;
              bool has_literal_ = false;
              uint16_t opx_ = 0;
              uint16_t opy_ = 0;
              uint8_t negx_ = 0;
              uint8_t negy_ = 0;
              std::string mnemonic_storage_;
              Operand dstx_;
              Operand dsty_;
              Operand srcx0_;
              Operand srcx1_;
              Operand srcx2_;
              Operand srcy0_;
              Operand srcy1_;
              Operand srcy2_;
              Slot x_;
              Slot y_;
            } ;

            } // namespace @ARCH@
            } // namespace rocjitsu

            #endif // @GUARD@
            ''')
            .lstrip()
            .replace('@ARCH@', arch)
            .replace(
                '@GENERATED_ARCH@',
                self.config.generated_include(self.generated_dir_name),
            )
            .replace('@GUARD@', guard)
            .replace('@VOPD3_FORMAT_ENUM@', ', Vopd3' if has_vopd3 else '')
            .replace('@VOPD3_HEADER_DECLS@', vopd3_header_decls)
            .replace('@VOPD_REGISTRATION_DECL@', vopd_registration_decl)
        )

        execution_helpers = (
            textwrap.dedent('''\
            @VOPD_REGISTRATION_DEF@
            bool Vopd::uses_src_neg_modifier(uint16_t op) {
              switch (op) {
            @VOPD_SRC_NEG_CASES@
                return true;
              default:
                return false;
              }
            }

            uint32_t Vopd::apply_neg(uint32_t value, uint8_t neg_bits, uint8_t src_idx) {
              return (neg_bits & (1u << src_idx)) ? (value ^ 0x80000000u) : value;
            }

            uint32_t Vopd::bitop2(uint32_t src0, uint32_t src1, uint32_t truth_table) {
              uint32_t result = 0;
              for (uint32_t bit = 0; bit < 32; ++bit) {
                uint32_t idx = (((src0 >> bit) & 1u) << 2) |
                               (((src1 >> bit) & 1u) << 1);
                result |= ((truth_table >> idx) & 1u) << bit;
              }
              return result;
            }
            @VOPD3_F64_HELPERS@

            uint32_t Vopd::execute_slot(const Slot &slot, amdgpu::Wavefront &wf,
                                        uint32_t lane) {
              uint32_t src0 = amdgpu::RegisterAccess(wf).read_lane(*slot.src0, lane);
              uint32_t src1 = amdgpu::RegisterAccess(wf).read_lane(*slot.src1, lane);
              uint32_t src2 = slot.has_src2_operand ? amdgpu::RegisterAccess(wf).read_lane(*slot.src2, lane)
                                                     : slot.src2_imm;
              if (uses_src_neg_modifier(slot.op)) {
                src0 = apply_neg(src0, slot.neg, 0);
                src1 = apply_neg(src1, slot.neg, 1);
                src2 = apply_neg(src2, slot.neg, 2);
              }

              switch (slot.op) {
            @VOPD_EXECUTE_SLOT_CASES@
              default:
                throw util::UnimplementedInst(op_name(slot.op));
              }
            }
            ''')
            .replace('@VOPD_REGISTRATION_DEF@', vopd_registration_def)
            .replace('@VOPD_SRC_NEG_CASES@', vopd_src_neg_case_labels)
            .replace('@VOPD3_F64_HELPERS@', vopd3_f64_helpers)
            .replace('@VOPD_EXECUTE_SLOT_CASES@', vopd_execute_slot_cases)
        )
        execution_method = textwrap.dedent('''\
            void Vopd::execute_impl(amdgpu::Wavefront &wf) {
              if (wf.wf_size() != 32)
                throw util::UnimplementedInst("VOPD requires Wave32");
            @EXECUTE_IMPL_BODY@
            }
            ''').replace('@EXECUTE_IMPL_BODY@', execute_impl_body)

        impl = (
            textwrap.dedent('''
            // Copyright (c) 2026 Advanced Micro Devices, Inc.
            // SPDX-License-Identifier: MIT
            //
            // AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.
            // See lib/python/amdisa/README.md for regeneration instructions.

            @VOPD_IMPL_INCLUDES@

            namespace rocjitsu {
            namespace @ARCH@ {

            namespace {

            // VOPD slot opcode values are the MRISA <Opcode> values for V_DUAL_* encodings.
            @VOPD_SLOT_CONSTANTS@

            @VOPD_OPCODE_VALIDATION_HELPERS@

            Operand make_src0(uint32_t bits, @VOPD3_UNUSED_ATTR@bool vopd3, bool use_literal,
                              uint32_t literal, uint16_t encoded) {
              if (use_literal && encoded == 255)
                return Operand(bits, OperandType::OPR_SIMM32, static_cast<int>(literal));
              return Operand(bits, @VOPD_SRC0_TYPE_EXPR@, encoded);
            }

            std::string operand_list(const Operand &dst, const Operand &src0,
                                     const Operand &src1) {
              return dst.name() + ", " + src0.name() + ", " + src1.name();
            }

            } // namespace

            const char *Vopd::op_name(uint16_t op) {
              switch (op) {
            @VOPD_OP_NAME_CASES@
              default:
                return "v_dual_unknown";
              }
            }

            @VOPD3_MODEL_HELPERS@

            @INLINE_EXECUTION_HELPERS@

            Result
            Vopd::validate_encoding(const MachineInst *inst,
                                    const util::DiagnosticEmitter &emit_error) {
              const auto *words = reinterpret_cast<const uint32_t *>(inst);
              const uint32_t word0 = words[0];
              const uint32_t word1 = words[1];
            @VOPD3_VALIDATION_BRANCH@
              const uint16_t opx = static_cast<uint16_t>((word0 >> 22) & 0xF);
              const uint16_t opy = static_cast<uint16_t>((word0 >> 17) & 0x1F);
              if (!is_valid_opcode(opx, kVopdXOpcodeMask)) [[unlikely]]
                return emit_error.emit() << "invalid VOPD X opcode";
              if (!is_valid_opcode(opy, kVopdYOpcodeMask)) [[unlikely]]
                return emit_error.emit() << "invalid VOPD Y opcode";
              const uint16_t srcx0 = static_cast<uint16_t>(word0 & 0x1FF);
              const uint16_t srcy0 = static_cast<uint16_t>(word1 & 0x1FF);
              if (srcx0 == 254 || srcy0 == 254) [[unlikely]]
                return emit_error.emit() << "VOPD does not support 64-bit literals";
              return Result::success();
            }

            Vopd::Vopd(const MachineInst *inst)
                : IsaInstruction<Isa>("vopd", @VOPD_EXEC_FN@),
                  dstx_(32, OperandType::OPR_VGPR, 0),
                  dsty_(32, OperandType::OPR_VGPR, 0),
                  srcx0_(32, OperandType::OPR_SRC, 0),
                  srcx1_(32, OperandType::OPR_VGPR, 0),
                  srcx2_(32, OperandType::OPR_VGPR, 0),
                  srcy0_(32, OperandType::OPR_SRC, 0),
                  srcy1_(32, OperandType::OPR_VGPR, 0),
                  srcy2_(32, OperandType::OPR_VGPR, 0) {
              const auto *words = reinterpret_cast<const uint32_t *>(inst);
              raw_encoding_ = words;
              word0_ = words[0];
              word1_ = words[1];

            @VOPD3_CONSTRUCTOR_BRANCH@
                format_ = Format::VopdXy;
                encoding_id_ = @VOPD_PREFIX@;
                opx_ = static_cast<uint16_t>((word0_ >> 22) & 0xF);
                opy_ = static_cast<uint16_t>((word0_ >> 17) & 0x1F);
                uint16_t srcx0 = static_cast<uint16_t>(word0_ & 0x1FF);
                uint16_t vsrcx1 = static_cast<uint16_t>((word0_ >> 9) & 0xFF);
                uint16_t srcy0 = static_cast<uint16_t>(word1_ & 0x1FF);
                uint16_t vsrcy1 = static_cast<uint16_t>((word1_ >> 9) & 0xFF);
                uint16_t vdstx = static_cast<uint16_t>((word1_ >> 24) & 0xFF);
                uint16_t vdsty_hi = static_cast<uint16_t>((word1_ >> 17) & 0x7F);
                uint16_t vdsty = static_cast<uint16_t>((vdsty_hi << 1) | ((~vdstx) & 1u));
                has_literal_ = srcx0 == 255 || srcy0 == 255 || opx_ == kVopdFmaakF32 ||
                               opx_ == kVopdFmamkF32 || opy_ == kVopdFmaakF32 ||
                               opy_ == kVopdFmamkF32;
                size_ = has_literal_ ? 12 : 8;
                if (has_literal_) {
                  word2_ = words[2];
                  literal_ = word2_;
                }

            @VOPDXY_BITS_DECL@
                dstx_ = Operand(x_bits, OperandType::OPR_VGPR, vdstx);
                dsty_ = Operand(y_bits, OperandType::OPR_VGPR, vdsty);
                srcx0_ = make_src0(x_bits, false, has_literal_, literal_, srcx0);
                srcy0_ = make_src0(y_bits, false, has_literal_, literal_, srcy0);
                srcx1_ = Operand(x_bits, OperandType::OPR_VGPR, vsrcx1);
                srcy1_ = Operand(y_bits, OperandType::OPR_VGPR, vsrcy1);
            @VOPD3_CONSTRUCTOR_CLOSE@

              dstx_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);
              dsty_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);
              srcx0_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
              srcy0_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
              srcx1_.set_vgpr_msb_role(opx_ == kVopdFmamkF32
                                           ? amdgpu::VgprMsbRole::Src2
                                           : amdgpu::VgprMsbRole::Src1);
              srcy1_.set_vgpr_msb_role(opy_ == kVopdFmamkF32
                                           ? amdgpu::VgprMsbRole::Src2
                                           : amdgpu::VgprMsbRole::Src1);
              srcx2_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);
              srcy2_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);

              opcode_ = static_cast<uint16_t>((opx_ << 8) | opy_);
              init_operands();
              mnemonic_storage_ = std::string(op_name(opx_)) + " :: " + op_name(opy_);
              mnemonic_ = mnemonic_storage_;
              disassembly_ = format_slot(x_) + " :: " + format_slot(y_);
            }

            void Vopd::init_operands() {
              x_ = Slot{opx_, &dstx_, &srcx0_, &srcx1_, &srcx2_, 0, negx_, false, false,
                        false};
              y_ = Slot{opy_, &dsty_, &srcy0_, &srcy1_, &srcy2_, 0, negy_, false, false,
                        false};

            @VOPD3_INIT_OPERANDS_PREFIX@
                x_.uses_vcc = opx_ == kVopdCndmaskB32;
                y_.uses_vcc = opy_ == kVopdCndmaskB32;
                if (opx_ == kVopdFmaakF32 || opx_ == kVopdFmamkF32) {
                  x_.src2_is_imm = true;
                  x_.src2_imm = literal_;
                }
                if (opy_ == kVopdFmaakF32 || opy_ == kVopdFmamkF32) {
                  y_.src2_is_imm = true;
                  y_.src2_imm = literal_;
                }
            @VOPD3_INIT_OPERANDS_SUFFIX@

              dst_operands_[0] = &dstx_;
              dst_operands_[1] = &dsty_;
              num_dst_ = 2;
              num_src_ = 0;

              const auto add_src = [this](Operand *op) {
                if (op)
                  src_operands_[num_src_++] = op;
              };
              const auto add_slot_sources = [&](const Slot &slot) {
                switch (slot.op) {
            @VOPD_ADD_SLOT_SOURCE_CASES@
                default:
                  add_src(slot.src0);
                  add_src(slot.src1);
                  break;
                }
              };

              add_slot_sources(x_);
              add_slot_sources(y_);
            }

            std::string Vopd::format_slot(const Slot &slot) const {
              std::string out = op_name(slot.op);
              out += " ";
              switch (slot.op) {
            @VOPD_FORMAT_SLOT_CASES@
              default:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                break;
              }
              return out;
            }

            @INLINE_EXECUTION_METHOD@

            } // namespace @ARCH@
            } // namespace rocjitsu
            ''')
            .lstrip()
            .replace('@ARCH@', arch)
            .replace('@VOPD_IMPL_INCLUDES@', vopd_impl_includes.rstrip())
            .replace('@VOPD_SRC0_TYPE_EXPR@', vopd_src0_type_expr)
            .replace('@VOPD_EXEC_FN@', vopd_exec_fn)
            .replace(
                '@INLINE_EXECUTION_HELPERS@',
                (
                    ''
                    if self.isa_spec.profile.split_execution_sources
                    else execution_helpers
                ),
            )
            .replace(
                '@INLINE_EXECUTION_METHOD@',
                (
                    ''
                    if self.isa_spec.profile.split_execution_sources
                    else execution_method
                ),
            )
            .replace('@VOPD3_FORMAT_ENUM@', ', Vopd3' if has_vopd3 else '')
            .replace('@VOPD3_HEADER_DECLS@', vopd3_header_decls)
            .replace('@VOPD_SLOT_CONSTANTS@', vopd_slot_constants)
            .replace('@VOPD_OPCODE_VALIDATION_HELPERS@', vopd_opcode_validation_helpers)
            .replace('@VOPD3_UNUSED_ATTR@', vopd3_unused_attr)
            .replace('@VOPD_OP_NAME_CASES@', vopd_op_name_cases)
            .replace('@VOPD3_MODEL_HELPERS@', vopd3_model_helpers)
            .replace('@VOPD3_VALIDATION_BRANCH@', vopd3_validation_branch)
            .replace('@VOPD3_CONSTRUCTOR_BRANCH@', vopd3_constructor_branch)
            .replace('@VOPD3_CONSTRUCTOR_CLOSE@', vopd3_constructor_close)
            .replace('@VOPD_PREFIX@', f'0x{vopd_prefix.prefix:X}')
            .replace('@VOPD3_INIT_OPERANDS_PREFIX@', vopd3_init_operands_prefix)
            .replace('@VOPD3_INIT_OPERANDS_SUFFIX@', vopd3_init_operands_suffix)
            .replace('@VOPD_ADD_SLOT_SOURCE_CASES@', vopd_add_slot_source_cases)
            .replace('@VOPD_FORMAT_SLOT_CASES@', vopd_format_slot_cases)
            .replace('@VOPDXY_BITS_DECL@', vopdxy_bits_decl)
        )
        impl_outputs = _ImplOutputs(
            model=[impl], execution=[execution_helpers, execution_method]
        )

        with open(os.path.join(out_dir, 'vopd.h'), 'w') as f:
            f.write(header)
        if self.isa_spec.profile.split_execution_sources:
            execution_impl = '\n\n'.join(
                str(fragment) for fragment in impl_outputs.execution
            )
            execution_file = (
                '// Copyright (c) 2026 Advanced Micro Devices, Inc.\n'
                '// SPDX-License-Identifier: MIT\n'
                '//\n'
                '// AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.\n'
                '// See lib/python/amdisa/README.md for regeneration instructions.\n\n'
                f'#include "{self.config.generated_include(self.generated_dir_name, "vopd.h")}"\n'
                '#include "util/except.h"\n'
                '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"\n'
                '#include "rocjitsu/vm/amdgpu/register_access.h"\n'
                '#include "rocjitsu/vm/amdgpu/wavefront.h"\n'
                '#include <algorithm>\n'
                '#include <bit>\n'
                '#include <cmath>\n'
                '#include <format>\n'
                '#include <string>\n\n'
                'namespace rocjitsu {\n'
                f'namespace {arch} {{\n\n'
                'namespace {\n'
                f'{vopd_execution_slot_constants}\n'
                '} // namespace\n\n'
                f'{execution_impl}\n\n'
                f'}} // namespace {arch}\n'
                '} // namespace rocjitsu\n'
            )
            with open(os.path.join(out_dir, 'vopd.cpp'), 'w') as f:
                f.write(str(impl_outputs.model[0]))
            with open(os.path.join(out_dir, 'vopd_exec.cpp'), 'w') as f:
                f.write(execution_file)
        else:
            with open(os.path.join(out_dir, 'vopd.cpp'), 'w') as f:
                f.write(str(impl_outputs.model[0]))

    def _shared_baseline(self) -> dict[str, tuple[str, list[tuple[str, int]]]]:
        """Build the shared-struct baseline for the current ISA.

        Returns a dict mapping struct name -> (include_path, expected_fields)
        that ``gen_machine_inst_encodings`` uses when ``config.use_shared``
        is True.
        """
        baseline: dict[str, tuple[str, list[tuple[str, int]]]] = {}
        for name, fields in _SCALAR_BASELINE.items():
            baseline[name] = (_SCALAR_SHARED_INCLUDE, fields)
        if self.isa_spec.arch_name in _CDNA_ARCHES:
            for name, fields in _CDNA_BASELINE.items():
                baseline[name] = (_CDNA_SHARED_INCLUDE, fields)
        return baseline

    def gen_machine_inst_encodings(self) -> None:
        """Generate machine instruction encoding structs as bitfields.

        When ``config.use_shared`` is True, structs whose name and field
        layout match a shared baseline are emitted as ``using`` aliases
        referencing the ``amdgpu::`` namespace version from the
        corresponding shared header.  Non-matching structs are emitted
        inline as before.
        """
        baseline = self._shared_baseline()
        shared_includes: set[str] = set()

        enc_structs = [cgen.Statement('using MachineInst = uint32_t')]
        for inst_enc in self.isa_spec.inst_encodings:
            struct_name = f'{inst_enc.fmt_enc_name}MachineInst'
            fields = [(x.name, x.bit_cnt) for x in inst_enc.ucode_fields]

            if struct_name in baseline:
                inc_path, expected_fields = baseline[struct_name]
                if fields == expected_fields:
                    shared_includes.add(inc_path)
                    enc_structs.append(
                        cgen.Statement(f'using {struct_name} = amdgpu::{struct_name}')
                    )
                    continue

            s = cgen.Struct(
                struct_name,
                [
                    cgen.Value(
                        'uint64_t' if x.bit_cnt > 32 else 'uint32_t',
                        f'{x.name} : {x.bit_cnt}',
                    )
                    for x in inst_enc.ucode_fields
                ],
            )
            enc_structs.append(s)

        includes: list[tuple[str, bool]] = [('cstdint', True)]
        for inc in sorted(shared_includes):
            includes.append((inc, False))

        cpp_file = CppFile(
            'machine_insts',
            self.out_path,
            True,
            includes,
            [],
            enc_structs,
            self.cpp_namespace,
            generated_dir_name=self.generated_dir_name,
        )
        cpp_file.gen_code()

    def _encoding_extension_word_capacity(self, inst_enc: InstEncoding) -> int:
        """Return the largest extension storage owned by an encoding."""
        dpp_struct, dpp8_struct = self._vop_dpp_struct_names(inst_enc.enc_name)
        owns_dpp_extension = (
            dpp_struct is not None
            and self._supports_dpp_for_encoding(inst_enc.enc_name)
        ) or dpp8_struct is not None
        default_cond = dict(inst_enc.enc_conds).get('default_encoding', 'true')
        has_real_default_check = inst_enc.bit_cnt < 64 and default_cond != 'false'
        literal32_condition = any(
            name.startswith('has_lit') and not name.startswith('has_lit64')
            for name, _ in inst_enc.enc_conds
        )

        capacity = int(owns_dpp_extension)
        if has_real_default_check and default_cond != 'true':
            capacity = max(capacity, 1)
        if literal32_condition:
            capacity = max(capacity, 1)
        if self._literal64_condition_names(inst_enc):
            capacity = max(capacity, 2)
        if inst_enc.has_implied_literal_ops:
            capacity = max(capacity, 1, *inst_enc.implied_literal_ops.values())
        return capacity

    def _max_instruction_word_count(self) -> int:
        """Return the maximum encoded width and lookahead for this generated decoder."""
        maximum = 1
        for inst_enc in self.isa_spec.inst_encodings:
            if not inst_enc.insts:
                continue
            base_words = (inst_enc.bit_cnt + 31) // 32
            maximum = max(
                maximum,
                base_words + self._encoding_extension_word_capacity(inst_enc),
            )

        for size_bytes in self.isa_spec.profile.inst_size_overrides.values():
            maximum = max(maximum, (size_bytes + 3) // 4)

        # Bespoke generated decoders do not necessarily have a matching XML
        # encoding object. VOPD can own one literal DWORD, while VOP3PX2 forms
        # are a two-DWORD prefix followed by a two-DWORD instruction.
        if self._supports_generated_vopd():
            maximum = max(maximum, 3)
        if (
            self._supports_cdna5_scaled_wmma_vop3px2()
            or self._supports_cdna_mfma_f8f6f4_vop3px2()
        ):
            maximum = max(maximum, 4)
        return maximum

    def gen_encodings(self) -> None:
        """Generate encoding classes wrapping raw encoding types."""
        enc_classes = []
        class_func_impls = []
        cond_emitted: set[str] = set()
        for inst_enc in self.isa_spec.inst_encodings:
            if not inst_enc.insts:
                continue
            enc_upper = inst_enc.enc_name.upper()
            enc_field_names = {f.name for f in inst_enc.ucode_fields}
            literal_fields = _LITERAL_ENCODING_OPERANDS.get(
                inst_enc.enc_name.upper(), ('', ())
            )[1]
            encoded_literal_fields = tuple(
                field for field in literal_fields if field in enc_field_names
            )
            uses_instruction_literal_policy = (
                self._supports_simm64_literal_operands()
                and bool(encoded_literal_fields)
            )
            unsupported_literal64_fields = self._unsupported_literal64_selector_fields(
                inst_enc
            )
            supports_sdwa_extension = self._supports_sdwa_for_encoding(
                inst_enc.enc_name
            )
            supports_fixed_size_embedding = (
                self._supports_cdna5_scaled_wmma_vop3px2()
                and inst_enc.fmt_enc_name == 'Vop3p'
            )
            dpp_struct, dpp8_struct = self._vop_dpp_struct_names(inst_enc.enc_name)
            dpp_extension_conditions = []
            dpp_marker_conditions = []
            dpp_opcodes: tuple[int, ...] = ()
            dpp8_opcodes: tuple[int, ...] = ()
            if dpp_struct is not None and self._supports_dpp_for_encoding(
                inst_enc.enc_name
            ):
                dpp_opcodes = self._encoded_dpp_opcodes(inst_enc, 'dpp')
                if dpp_opcodes:
                    dpp_extension_conditions.append('has_encoded_dpp()')
                    dpp_marker_conditions.append('inst_.src0 == amdgpu::SRC_DPP')
            if dpp8_struct is not None:
                dpp8_opcodes = self._encoded_dpp_opcodes(inst_enc, 'dpp8')
                if dpp8_opcodes:
                    dpp_extension_conditions.append('has_encoded_dpp8()')
                    dpp_marker_conditions.append('amdgpu::dpp::is_src_dpp8(inst_.src0)')
            owns_dpp_extension = bool(dpp_extension_conditions)
            # Compact VOP1/VOP2/VOPC encodings already account for DPP through
            # !default_encoding(). VOP3-family base encodings are 64 bits, so
            # their DPP control DWORD must be counted explicitly.
            needs_explicit_dpp_size = owns_dpp_extension and inst_enc.bit_cnt >= 64
            class_members = []
            constructor_args = [
                cgen.Value('std::string_view', 'mnemonic'),
                cgen.Value(
                    f'const {inst_enc.fmt_enc_name}MachineInst',
                    '*inst',
                ),
                cgen.Value('ExecuteFn', 'exec_fn'),
            ]
            if supports_fixed_size_embedding:
                constructor_args.append(
                    cgen.Value(
                        'ExtensionDecodePolicy',
                        'extension_policy = ExtensionDecodePolicy::Decode',
                    )
                )
            public_members = [cgen.Line('public:')]
            if supports_fixed_size_embedding:
                public_members.append(
                    cgen.Line('enum class ExtensionDecodePolicy { Decode, Skip };')
                )
            public_members.append(
                cgen.FunctionDeclaration(
                    cgen.Value('', f'{inst_enc.fmt_enc_name}'),
                    constructor_args,
                )
            )
            validation_args = [
                '[[maybe_unused]] std::string_view mnemonic',
                f'const {inst_enc.fmt_enc_name}MachineInst *inst',
                'const util::DiagnosticEmitter &emit_error',
            ]
            if uses_instruction_literal_policy:
                validation_args.extend(
                    [
                        'LiteralSupport literal_support = LiteralSupport::Both',
                        f'int num_encoded_sources = {len(encoded_literal_fields)}',
                    ]
                )
            elif unsupported_literal64_fields:
                validation_args.append('int num_encoded_sources = 3')
            if supports_fixed_size_embedding:
                validation_args.append(
                    'ExtensionDecodePolicy extension_policy = '
                    'ExtensionDecodePolicy::Decode'
                )
            shared_dpp_opcode_set = bool(dpp_opcodes) and dpp_opcodes == dpp8_opcodes
            if shared_dpp_opcode_set:
                public_members.append(cgen.Line('bool supports_dpp_opcode() const;'))
                class_func_impls.append(
                    cgen.Line(
                        self._opcode_predicate_helper_impl(
                            inst_enc, 'supports_dpp_opcode', list(dpp_opcodes)
                        )
                    )
                )
            if dpp_opcodes:
                public_members.append(cgen.Line('bool has_encoded_dpp() const;'))
                if shared_dpp_opcode_set:
                    class_func_impls.append(
                        cgen.Line(
                            f'bool {inst_enc.fmt_enc_name}::has_encoded_dpp() const {{ '
                            'return supports_dpp_opcode() && '
                            'inst_.src0 == amdgpu::SRC_DPP; }'
                        )
                    )
                else:
                    class_func_impls.append(
                        cgen.Line(
                            self._opcode_predicate_helper_impl(
                                inst_enc,
                                'has_encoded_dpp',
                                list(dpp_opcodes),
                                'inst_.src0 == amdgpu::SRC_DPP',
                            )
                        )
                    )
            if dpp8_opcodes:
                public_members.append(cgen.Line('bool has_encoded_dpp8() const;'))
                if shared_dpp_opcode_set:
                    class_func_impls.append(
                        cgen.Line(
                            f'bool {inst_enc.fmt_enc_name}::has_encoded_dpp8() const {{ '
                            'return supports_dpp_opcode() && '
                            'amdgpu::dpp::is_src_dpp8(inst_.src0); }'
                        )
                    )
                else:
                    class_func_impls.append(
                        cgen.Line(
                            self._opcode_predicate_helper_impl(
                                inst_enc,
                                'has_encoded_dpp8',
                                list(dpp8_opcodes),
                                'amdgpu::dpp::is_src_dpp8(inst_.src0)',
                            )
                        )
                    )
            if owns_dpp_extension:
                public_members.append(
                    cgen.Line('void append_mnemonic(std::string &out) const override;')
                )
                dpp_condition = ' || '.join(dpp_extension_conditions)
                if enc_upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
                    rule = profile.mnemonic_rule(inst_enc.enc_name)
                    assert (
                        rule.suffix == '_e32'
                    ), f'{inst_enc.enc_name}: compact DPP mnemonic rule must use _e32'
                    dpp_suffix = '' if enc_upper == 'ENC_VOPC' else '_dpp'
                    dpp_mnemonic_body = (
                        f'if (!({dpp_condition}) || !mnemonic_.ends_with("_e32")) '
                        '{ out += mnemonic_; return; } '
                        'out.append(mnemonic_.data(), mnemonic_.size() - 4); '
                        f'out += "{dpp_suffix}";'
                    )
                else:
                    dpp_mnemonic_body = (
                        f'out += mnemonic_; if ({dpp_condition}) out += "_e64_dpp";'
                    )
                class_func_impls.append(
                    cgen.Line(
                        f'void {inst_enc.fmt_enc_name}::append_mnemonic'
                        f'(std::string &out) const {{ {dpp_mnemonic_body} }}'
                    )
                )
            # Determine whether the constructor needs a runtime size
            # check for an extension DWORD beyond the base encoding.
            #
            # The ``default_encoding()`` method distinguishes the compact
            # base form (32-bit) from extended forms (DPP, SDWA, literal).
            # Its expression comes from the XML ``default`` condition.
            #
            # Two cases where no size check is needed:
            #  1. 64-bit+ encodings: OpEncoding already spans the full
            #     instruction; there is no extension DWORD.
            #  2. The ``default`` condition is a constant ``false`` (empty
            #     ``<Value />`` in CDNA4 XML): the encoding is fixed-size
            #     and the XML simply lacks a meaningful condition.
            #
            # For case 2 with implied-literal encodings (SOPK), the
            # ``hasImpliedLiteral()`` check alone suffices.
            default_cond = dict(inst_enc.enc_conds).get('default_encoding', 'true')
            has_real_default_check = inst_enc.bit_cnt < 64 and default_cond != 'false'

            if (
                has_real_default_check
                and inst_enc.has_implied_literal_ops
                and not inst_enc.has_variable_implied_literal_size
            ):
                size_condition = '!default_encoding() || hasImpliedLiteral()'
            elif has_real_default_check:
                size_condition = '!default_encoding()'
            elif (
                inst_enc.has_implied_literal_ops
                and not inst_enc.has_variable_implied_literal_size
            ):
                size_condition = 'hasImpliedLiteral()'
            else:
                size_condition = None
            if has_real_default_check and dpp_marker_conditions:
                raw_dpp_condition = ' || '.join(dpp_marker_conditions)
                encoded_dpp_condition = ' || '.join(dpp_extension_conditions)
                size_condition = (
                    f'(!default_encoding() && !({raw_dpp_condition}))'
                    f' || {encoded_dpp_condition}'
                )
                if (
                    inst_enc.has_implied_literal_ops
                    and not inst_enc.has_variable_implied_literal_size
                ):
                    size_condition += ' || hasImpliedLiteral()'

            profile = self.isa_spec.profile
            rule = profile.mnemonic_rule(inst_enc.enc_name)
            if rule.use_flat_mnemonic:
                mnemonic_expr = 'flat_mnemonic(mnemonic, inst->seg)'
            else:
                # Suffix is pre-baked into the literal by the instruction
                # constructor, so the encoding base just passes through.
                mnemonic_expr = 'mnemonic'

            vop3_opsel_field = profile.vop3_opsel_field
            vop3_family = enc_upper in ('ENC_VOP3', 'VOP3_SDST_ENC')
            standard_vop3_disassembly = (
                vop3_family
                and {
                    'clamp',
                    'omod',
                    'neg',
                }
                <= enc_field_names
            )
            vop3_has_abs = 'abs' in enc_field_names
            vop3_has_opsel = vop3_opsel_field in enc_field_names
            vop3_opsel_opcodes = (
                tuple(
                    sorted(
                        inst.opcode
                        for inst in inst_enc.insts
                        if vop3_has_opsel
                        and profile.uses_true16_vop3_opsel
                        and inst.name not in profile.vop3_opsel_omissions
                        and any(opnd.size == 16 for opnd in inst.operands)
                    )
                )
                if standard_vop3_disassembly
                else ()
            )
            if vop3_opsel_opcodes:
                public_members.append(cgen.Line('bool displays_vop3_op_sel() const;'))
                class_func_impls.append(
                    cgen.Line(
                        self._opcode_predicate_helper_impl(
                            inst_enc,
                            'displays_vop3_op_sel',
                            list(vop3_opsel_opcodes),
                        )
                    )
                )

            absolute_source_modifier_opcodes = [
                inst.opcode
                for inst in inst_enc.insts
                if inst.name in profile.vop3p_absolute_source_instructions
            ]
            if absolute_source_modifier_opcodes:
                public_members.append(
                    cgen.Line('bool uses_vop3p_absolute_source_syntax() const;')
                )
                class_func_impls.append(
                    cgen.Line(
                        self._opcode_predicate_helper_impl(
                            inst_enc,
                            'uses_vop3p_absolute_source_syntax',
                            sorted(set(absolute_source_modifier_opcodes)),
                        )
                    )
                )

            gfx11_mimg_no_dim_dmask_opcodes = []
            if profile.renders_gfx11_image_syntax and enc_upper == 'ENC_MIMG':
                gfx11_mimg_no_dim_dmask_opcodes = [
                    inst.opcode
                    for inst in inst_enc.insts
                    if inst.name in profile.gfx11_mimg_fixed_vaddr_words
                ]
                if gfx11_mimg_no_dim_dmask_opcodes:
                    public_members.append(
                        cgen.Line('bool omits_gfx11_mimg_dim_dmask() const;')
                    )
                    class_func_impls.append(
                        cgen.Line(
                            self._opcode_predicate_helper_impl(
                                inst_enc,
                                'omits_gfx11_mimg_dim_dmask',
                                sorted(set(gfx11_mimg_no_dim_dmask_opcodes)),
                            )
                        )
                    )

                nsa_group_rules = []
                for inst in inst_enc.insts:
                    groups = profile.gfx11_mimg_nsa_group_words.get(inst.name)
                    if groups is not None:
                        nsa_group_rules.append((inst.opcode, *groups))
                if nsa_group_rules:
                    public_members.append(
                        cgen.Line(
                            'uint32_t gfx11_mimg_nsa_group_width('
                            'uint32_t index, uint32_t vaddr_words) const;'
                        )
                    )
                    nsa_group_lines = [
                        f'uint32_t {inst_enc.fmt_enc_name}::gfx11_mimg_nsa_group_width('
                        'uint32_t index, uint32_t vaddr_words) const {',
                        '  switch (inst_.op) {',
                    ]
                    for opcode, default_groups, a16_groups in nsa_group_rules:
                        nsa_group_lines.extend(
                            (
                                f'  case {opcode}:',
                                '    if (inst_.a16) {',
                                '      static constexpr uint8_t widths[] = {'
                                + ', '.join(str(width) for width in a16_groups)
                                + '};',
                                f'      return index < {len(a16_groups)} ? widths[index] : 0;',
                                '    }',
                                '    {',
                                '      static constexpr uint8_t widths[] = {'
                                + ', '.join(str(width) for width in default_groups)
                                + '};',
                                f'      return index < {len(default_groups)} ? widths[index] : 0;',
                                '    }',
                            )
                        )
                    nsa_group_lines.extend(
                        (
                            '  default:',
                            '    if (index < 4) return 1;',
                            '    return index == 4 && vaddr_words > 4 ? vaddr_words - 4 : 0;',
                            '  }',
                            '}',
                        )
                    )
                    class_func_impls.append(cgen.Line('\n'.join(nsa_group_lines)))

            # TH/SCOPE rendering follows the encoding data itself. Keeping it
            # outside the profile modifier lists prevents a new memory format
            # with the same fields from silently losing its cache attributes.
            gfx12_cache_modifier = self._gfx12_cache_policy_modifier_impl(
                inst_enc, enc_field_names
            )

            modifier_lines = ''
            if profile.renders_gfx11_image_syntax and enc_upper == 'ENC_MIMG':
                modifier_lines += (
                    'if (!omits_gfx11_mimg_dim_dmask()) {'
                    'modifiers_ += " dmask:0x"; '
                    'modifiers_ += "0123456789abcdef"[inst->dmask & 0xfu]; }'
                    'if (!omits_gfx11_mimg_dim_dmask()) {'
                    'static constexpr std::string_view dims[] = {'
                    '"1D", "2D", "3D", "CUBE", "1D_ARRAY", '
                    '"2D_ARRAY", "2D_MSAA", "2D_MSAA_ARRAY"};'
                    'modifiers_ += " dim:SQ_RSRC_IMG_";'
                    'modifiers_ += dims[inst->dim & 7u];}'
                    'if (!omits_gfx11_mimg_dim_dmask()) {'
                    'if (inst->unorm) modifiers_ += " unorm";'
                    'if (inst->glc) modifiers_ += " glc";'
                    'if (inst->slc) modifiers_ += " slc";'
                    'if (inst->dlc) modifiers_ += " dlc";'
                    'if (inst->r128) modifiers_ += " r128";}'
                    'if (inst->a16) modifiers_ += " a16";'
                    'if (!omits_gfx11_mimg_dim_dmask()) {'
                    'if (inst->tfe) modifiers_ += " tfe";'
                    'if (inst->lwe) modifiers_ += " lwe";'
                    'if (inst->d16) modifiers_ += " d16";}'
                )
            for mod in profile.encoding_modifiers(inst_enc.enc_name):
                if not mod.preamble and mod.field not in enc_field_names:
                    continue
                # GFX12 assemblers require VBUFFER address modifiers before
                # TH/SCOPE, while flags such as NV follow the cache policy.
                if gfx12_cache_modifier and mod.field == 'nv':
                    modifier_lines += gfx12_cache_modifier
                    gfx12_cache_modifier = ''
                field_ref = mod.field if mod.preamble else f'inst->{mod.field}'
                if mod.preamble:
                    modifier_lines += mod.preamble
                if mod.is_offset:
                    cond = mod.condition if mod.condition else field_ref
                    modifier_lines += (
                        f'if ({cond}) modifiers_ += " offset:"'
                        f' + std::to_string({field_ref});'
                    )
                else:
                    modifier_lines += f'if ({field_ref}) modifiers_ += "{mod.display}";'
            modifier_lines += gfx12_cache_modifier
            if (
                enc_upper == 'ENC_DS'
                and {'offset0', 'offset1', 'gds'} <= enc_field_names
            ):
                split_offset_opcodes = []
                for inst in inst_enc.insts:
                    sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    if (
                        sem is not None
                        and sem.semantic_class
                        in (
                            'ds_read2',
                            'ds_write2',
                            'ds_atomic2',
                        )
                        or (profile.split_ds_2addr_offsets and '_2ADDR_' in inst.name)
                    ):
                        split_offset_opcodes.append(inst.opcode)
                if split_offset_opcodes:
                    public_members.append(
                        cgen.Line('bool uses_split_ds_offsets() const;')
                    )
                    class_func_impls.append(
                        cgen.Line(
                            self._opcode_predicate_helper_impl(
                                inst_enc,
                                'uses_split_ds_offsets',
                                sorted(set(split_offset_opcodes)),
                            )
                        )
                    )
                split_offset_condition = (
                    'uses_split_ds_offsets()' if split_offset_opcodes else 'false'
                )
                modifier_lines += (
                    f'if ({split_offset_condition}) {{'
                    'if (inst->offset0) modifiers_ += " offset0:"'
                    ' + std::to_string(inst->offset0);'
                    'if (inst->offset1) modifiers_ += " offset1:"'
                    ' + std::to_string(inst->offset1);'
                    '} else {'
                    'const uint32_t offset = inst->offset0 | (inst->offset1 << 8);'
                    'if (offset) modifiers_ += " offset:" + std::to_string(offset);'
                    '}'
                    'if (inst->gds) modifiers_ += " gds";'
                )
            if standard_vop3_disassembly:
                public_members.append(
                    cgen.Line('uint32_t vop3_encoded_source_count() const;')
                )
                public_members.append(
                    cgen.Line(
                        'int32_t vop3_encoded_source_index(uint8_t operand_index) const;'
                    )
                )
                class_func_impls.append(
                    cgen.Line(
                        f'uint32_t {inst_enc.fmt_enc_name}::vop3_encoded_source_count() const {{ '
                        'uint32_t count = 0; '
                        'for (uint8_t src = 0; src < num_src_; ++src) { '
                        'if (src_operands_[src]->is_fieldless()) continue; '
                        'bool repeats_dst = false; '
                        'for (uint8_t dst = 0; dst < num_dst_; ++dst) '
                        'repeats_dst |= src_operands_[src] == dst_operands_[dst]; '
                        'if (!repeats_dst) ++count; } return count; }'
                    )
                )
                class_func_impls.append(
                    cgen.Line(
                        f'int32_t {inst_enc.fmt_enc_name}::vop3_encoded_source_index'
                        '(uint8_t operand_index) const { int32_t encoded_index = 0; '
                        'for (uint8_t src = 0; src <= operand_index; ++src) { '
                        'if (src_operands_[src]->is_fieldless()) continue; '
                        'bool repeats_dst = false; '
                        'for (uint8_t dst = 0; dst < num_dst_; ++dst) '
                        'repeats_dst |= src_operands_[src] == dst_operands_[dst]; '
                        'if (repeats_dst) { if (src == operand_index) return -1; continue; } '
                        'if (src == operand_index) return encoded_index; ++encoded_index; } '
                        'return -1; }'
                    )
                )
                displays_op_sel = (
                    'displays_vop3_op_sel()' if vop3_opsel_opcodes else 'false'
                )
                modifier_lines += (
                    'amdgpu::vop::append_vop3_disassembly('
                    f'modifiers_, {f"inst->{vop3_opsel_field}" if vop3_has_opsel else "0"}, '
                    'inst->clamp, inst->omod, '
                    f'vop3_encoded_source_count(), {displays_op_sel});'
                )
            if enc_upper == 'ENC_VOP3P':
                op_sel, op_sel_hi = profile.vop3p_opsel_fields
                op_sel_hi_high = profile.vop3p_opsel_hi_high_field
                required_fields = {
                    op_sel,
                    op_sel_hi,
                    op_sel_hi_high,
                    'neg',
                    'neg_hi',
                    'clamp',
                }
                modifier_fields = {op_sel, op_sel_hi, 'neg', 'neg_hi', 'clamp'}
                if (
                    modifier_fields <= enc_field_names
                    and op_sel_hi_high not in enc_field_names
                ):
                    raise ValueError(
                        f'{inst_enc.enc_name} has VOP3P modifier fields but is missing '
                        f'the profile high op_sel field {op_sel_hi_high}'
                    )
                if required_fields <= enc_field_names:
                    public_members.append(
                        cgen.Line('uint32_t vop3p_encoded_source_count() const;')
                    )
                    class_func_impls.append(
                        cgen.Line(
                            f'uint32_t {inst_enc.fmt_enc_name}::vop3p_encoded_source_count() const {{ '
                            'uint32_t count = 0; '
                            'for (uint8_t src = 0; src < num_src_; ++src) { '
                            'if (src_operands_[src]->is_fieldless()) continue; '
                            'bool repeats_dst = false; '
                            'for (uint8_t dst = 0; dst < num_dst_; ++dst) '
                            'repeats_dst |= src_operands_[src] == dst_operands_[dst]; '
                            'if (!repeats_dst) ++count; } return count; }'
                        )
                    )
                    packed_default_opcodes = []
                    omitted_source_modifier_opcodes = []
                    for inst in inst_enc.insts:
                        sem = (
                            self.semantics.instructions.get(inst.name)
                            if self.semantics
                            else None
                        )
                        # Packed disassembly defaults are part of the ISA
                        # encoding, not its execution implementation.  A
                        # model-only V_PK_* instruction intentionally has no
                        # semantic class, but still defaults OP_SEL_HI to one
                        # for every encoded source.
                        if inst.name.startswith('V_PK_') or (
                            sem is not None
                            and sem.semantic_class.startswith(('pk_', 'dot2_'))
                        ):
                            packed_default_opcodes.append(inst.opcode)
                        if inst.name in profile.vop3p_source_modifier_omissions:
                            omitted_source_modifier_opcodes.append(inst.opcode)
                    if omitted_source_modifier_opcodes:
                        public_members.append(
                            cgen.Line('bool omits_vop3p_source_modifiers() const;')
                        )
                        class_func_impls.append(
                            cgen.Line(
                                self._opcode_predicate_helper_impl(
                                    inst_enc,
                                    'omits_vop3p_source_modifiers',
                                    sorted(set(omitted_source_modifier_opcodes)),
                                )
                            )
                        )
                    packed_defaults = (
                        self._opcode_set_condition(
                            packed_default_opcodes,
                            (1 << inst_enc.op_field_bit_cnt) - 1,
                        )
                        if packed_default_opcodes
                        else 'false'
                    )
                    source_count = 'vop3p_encoded_source_count()'
                    if omitted_source_modifier_opcodes:
                        source_count = (
                            f'omits_vop3p_source_modifiers() ? 0 : {source_count}'
                        )
                    absolute_syntax = (
                        'uses_vop3p_absolute_source_syntax()'
                        if absolute_source_modifier_opcodes
                        else 'false'
                    )
                    if absolute_source_modifier_opcodes:
                        source_count = f'{absolute_syntax} ? 3 : ({source_count})'
                        packed_defaults = (
                            f'{absolute_syntax} ? false : ({packed_defaults})'
                        )
                    modifier_lines += (
                        'amdgpu::vop::append_vop3p_disassembly('
                        f'modifiers_, inst->{op_sel}, '
                        f'inst->{op_sel_hi} | (inst->{op_sel_hi_high} << 2), '
                        f'{absolute_syntax} ? 0 : inst->neg, '
                        f'{absolute_syntax} ? 0 : inst->neg_hi, inst->clamp, '
                        f'{source_count}, '
                        f'{packed_defaults});'
                    )
            dpp_modifier_line = ''
            if dpp_opcodes:
                from amdisa.isa_profile import DppCtrlDialect

                dpp_has_fi = str(
                    self._machine_inst_struct_has_field(dpp_struct, 'fi')
                ).lower()
                dpp_dialect = (
                    'Gfx9'
                    if profile.dpp_ctrl_dialect == DppCtrlDialect.GFX9
                    else 'Gfx10Plus'
                )
                dpp_modifier_line = (
                    'if (has_encoded_dpp()) '
                    'amdgpu::dpp::append_dpp16_disassembly('
                    'out, dpp_ctrl_, dpp_row_mask_, dpp_bank_mask_, '
                    f'dpp_bound_ctrl_, dpp_fi_, {dpp_has_fi}, '
                    f'amdgpu::dpp::DppCtrlDialect::{dpp_dialect});'
                )
            dpp8_modifier_line = ''
            if dpp8_opcodes:
                dpp8_modifier_line = (
                    'if (has_encoded_dpp8()) '
                    'amdgpu::dpp::append_dpp8_disassembly('
                    'out, dpp8_lane_sel_, dpp_fi_);'
                )

            has_op = any(f.name == 'op' for f in inst_enc.ucode_fields)
            size_line = (
                ' size_ = sizeof(OpEncoding);\n'
                '  raw_encoding_ = reinterpret_cast<const uint32_t *>(&inst_);\n'
                '  encoding_id_ = raw_encoding_[0] >> 23;'
            )
            validation_body = ' const auto &inst_ = *inst;'
            has_encoding_validation = False
            if has_op:
                size_line += '\n  opcode_ = inst_.op;'
            literal64_conds = self._literal64_condition_names(inst_enc)
            explicit_literal64_conds = [
                name for name, _ in inst_enc.enc_conds if name.startswith('has_lit64')
            ]
            if explicit_literal64_conds and encoded_literal_fields:
                public_members.append(cgen.Line('bool has_encoded_literal64() const;'))
                class_func_impls.append(
                    cgen.Line(
                        self._encoded_literal_helper_impl(
                            inst_enc, encoded_literal_fields, 254
                        )
                    )
                )
                literal64_conds = [
                    name
                    for name in literal64_conds
                    if name not in explicit_literal64_conds
                ]
                literal64_conds.append('has_encoded_literal64')
            implied_literal64_ops = [
                str(inst.opcode)
                for inst in inst_enc.insts
                if inst.is_implied_literal_enc
                and inst.enc_name.upper().endswith('_INST_LITERAL64')
            ]
            literal64_condition = ' || '.join(f'{name}()' for name in literal64_conds)
            literal32_conds = [
                name
                for name, _ in inst_enc.enc_conds
                if name.startswith('has_lit') and not name.startswith('has_lit64')
            ]
            if literal32_conds and encoded_literal_fields:
                public_members.append(cgen.Line('bool has_encoded_literal32() const;'))
                class_func_impls.append(
                    cgen.Line(
                        self._encoded_literal_helper_impl(
                            inst_enc, encoded_literal_fields, 255
                        )
                    )
                )
                literal32_conds = ['has_encoded_literal32']
            literal32_condition = ' || '.join(f'{name}()' for name in literal32_conds)
            if supports_fixed_size_embedding:
                size_line += ' if (extension_policy == ExtensionDecodePolicy::Decode) {'
                validation_body += (
                    ' if (extension_policy == ExtensionDecodePolicy::Decode) {'
                )
            # Some profiles expose SRC_LITERAL64 in their global operand
            # selector table even when a particular encoding has no 64-bit
            # literal extension form.  Derive the affected source fields from
            # that encoding's Literal32 conditions and fail before sizing or
            # copying the instruction.  Otherwise a selector-254 instruction
            # can be reported as the base size while a generated instruction
            # constructor reads two extension DWORDs that the encoding does
            # not own.
            if unsupported_literal64_fields:
                has_encoding_validation = True
                reject_condition = ' || '.join(
                    f'(num_encoded_sources > {source_idx} && inst_.{field} == 254)'
                    for source_idx, field in enumerate(unsupported_literal64_fields)
                )
                validation_body += (
                    f'\n  if ({reject_condition})'
                    f' [[unlikely]] return emit_error.emit() << "{inst_enc.fmt_enc_name} does not support '
                    'Literal64";'
                )
            if uses_instruction_literal_policy:
                has_encoding_validation = True
                for width, selector in ((32, 255), (64, 254)):
                    reject_condition = ' || '.join(
                        f'(num_encoded_sources > {source_idx} && '
                        f'inst_.{field} == {selector})'
                        for source_idx, field in enumerate(encoded_literal_fields)
                    )
                    validation_body += (
                        f'\n  if (!supports_literal(literal_support, '
                        f'LiteralSupport::Literal{width}) && ({reject_condition}))'
                        f' [[unlikely]] return emit_error.emit() << mnemonic << " does not support {width}-bit literals";'
                    )
            if needs_explicit_dpp_size and encoded_literal_fields:
                has_encoding_validation = True
                raw_dpp_condition = ' || '.join(dpp_marker_conditions)
                literal_selector_condition = ' || '.join(
                    f'inst_.{field} == 255' for field in encoded_literal_fields
                )
                validation_body += (
                    f' if (({raw_dpp_condition}) && '
                    f'({literal_selector_condition}))'
                    ' [[unlikely]] return emit_error.emit() << '
                    '"DPP and literal operands cannot be combined";'
                )
            # Size owned storage from this encoding's declared extension forms,
            # not the architecture-wide operand selector table. For example,
            # gfx1250 defines selector 254 for 64-bit literals, but the 64-bit
            # VOP3 and VOP3P encodings only support a 32-bit literal extension.
            encoded_sdwa_opcodes = (
                [
                    inst.opcode
                    for inst in inst_enc.insts
                    if self._instruction_supports_sdwa(inst, inst_enc.enc_name)
                ]
                if supports_sdwa_extension
                else []
            )
            encoded_sdwa_condition = ''
            if encoded_sdwa_opcodes:
                encoded_sdwa_condition = 'has_encoded_sdwa()'
                public_members.append(cgen.Line('bool has_encoded_sdwa() const;'))
                class_func_impls.append(
                    cgen.Line(
                        self._opcode_predicate_helper_impl(
                            inst_enc,
                            'has_encoded_sdwa',
                            encoded_sdwa_opcodes,
                            'inst_.src0 == amdgpu::SRC_SDWA',
                        )
                    )
                )
            extension_word_capacity = self._encoding_extension_word_capacity(inst_enc)
            owns_extension_words = extension_word_capacity > 0
            literal32_size_condition = size_condition
            compact_modifier_encoding = has_real_default_check and (
                owns_dpp_extension or supports_sdwa_extension
            )
            if compact_modifier_encoding:
                compact_extension_conditions = [
                    *dpp_extension_conditions,
                    *([encoded_sdwa_condition] if encoded_sdwa_condition else []),
                    *([literal32_condition] if literal32_condition else []),
                ]
                if (
                    inst_enc.has_implied_literal_ops
                    and not inst_enc.has_variable_implied_literal_size
                ):
                    compact_extension_conditions.append('hasImpliedLiteral()')
                literal32_size_condition = (
                    ' || '.join(compact_extension_conditions) or None
                )
            elif not owns_dpp_extension and literal32_condition:
                literal32_size_condition = literal32_condition
                if (
                    inst_enc.has_implied_literal_ops
                    and not inst_enc.has_variable_implied_literal_size
                ):
                    literal32_size_condition += ' || hasImpliedLiteral()'
            elif literal32_size_condition is None:
                literal32_size_condition = literal32_condition or None
            extension_size_line = ''
            if literal64_condition and literal32_size_condition is not None:
                extension_size_line = (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                    f' else if ({literal32_size_condition}) size_ += sizeof(MachineInst);'
                )
            elif literal64_condition and literal32_condition:
                extension_size_line = (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                    f' else if ({literal32_condition}) size_ += sizeof(MachineInst);'
                )
            elif literal64_condition:
                extension_size_line = (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                )
            elif literal32_size_condition is not None:
                extension_size_line = (
                    f' if ({literal32_size_condition}) size_ += sizeof(MachineInst);'
                )
            elif literal32_condition:
                extension_size_line = (
                    f' if ({literal32_condition}) size_ += sizeof(MachineInst);'
                )
            if inst_enc.has_variable_implied_literal_size:
                size_line += (
                    ' if (hasImpliedLiteral()) size_ += impliedLiteralWordCount()'
                    ' * sizeof(MachineInst); else' + extension_size_line
                )
            else:
                size_line += extension_size_line
            if needs_explicit_dpp_size:
                dpp_extension_condition = ' || '.join(dpp_extension_conditions)
                size_line += (
                    f' if ({dpp_extension_condition})' ' size_ += sizeof(MachineInst);'
                )
            if inst_enc.has_implied_literal_ops:
                size_line += (
                    ' if (hasImpliedLiteral())'
                    ' literal_ = reinterpret_cast<const uint32_t *>(inst)[1];'
                )
            if owns_extension_words:
                size_line += (
                    ' std::memcpy(raw_words_.data(), inst, size_);'
                    ' raw_encoding_ = raw_words_.data();'
                )
            if supports_fixed_size_embedding:
                size_line += ' }'
                validation_body += ' }'
            validation_body += ' return Result::success();'
            if has_encoding_validation:
                public_members.append(
                    cgen.Line(
                        'static Result validate_encoding('
                        + ', '.join(validation_args)
                        + f') {{{validation_body}}}'
                    )
                )
            constructor_extra_params = ''
            if supports_fixed_size_embedding:
                constructor_extra_params += ', ExtensionDecodePolicy extension_policy'
            if rule.use_flat_mnemonic:
                # FLAT mnemonics are dynamically constructed ("scratch_*",
                # "global_*"). Store the owned string in a member so the
                # string_view in Instruction doesn't dangle.
                class_ctor_impl = (
                    f'{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}'
                    f'(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst, '
                    f'ExecuteFn exec_fn{constructor_extra_params}) '
                    f': IsaInstruction<Isa>("", exec_fn), inst_(*inst), '
                    f'owned_mnemonic_({mnemonic_expr}) '
                    f'{{ mnemonic_ = owned_mnemonic_;{size_line}}}'
                )
            else:
                class_ctor_impl = (
                    f'{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}'
                    f'(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst'
                    f', ExecuteFn exec_fn{constructor_extra_params}) '
                    f': IsaInstruction<Isa>({mnemonic_expr}, exec_fn), inst_(*inst) '
                    f'{{{size_line}}}'
                )
            class_func_impls.append(cgen.Line(class_ctor_impl))
            if profile.renders_gfx11_image_syntax and enc_upper == 'ENC_MIMG':
                public_members.append(
                    cgen.Line(
                        'void capture_nsa_words(const MachineInst *inst, '
                        'const Operand *vaddr);'
                    )
                )
                public_members.append(
                    cgen.Line(
                        'void append_src_operand(std::string &out, '
                        'uint8_t operand_index) const override {\n'
                        '  const Operand *operand = src_operands_[operand_index];\n'
                        '  if (!inst_.nsa || operand != nsa_vaddr_operand_) {\n'
                        '    Instruction::append_src_operand(out, operand_index);\n'
                        '    return;\n'
                        '  }\n'
                        '  const uint32_t vaddr_words = (operand->size_bits() + 31) / 32;\n'
                        '  out += "[";\n'
                        '  uint32_t consumed_words = 0;\n'
                        '  for (uint32_t index = 0; index < 5 && consumed_words < vaddr_words; ++index) {\n'
                        '    const uint32_t group_words = gfx11_mimg_nsa_group_width(index, vaddr_words);\n'
                        '    if (group_words == 0) break;\n'
                        '    if (index != 0) out += ", ";\n'
                        '    const uint32_t selector = index == 0\n'
                        '        ? inst_.vaddr\n'
                        '        : (raw_words_[2] >> ((index - 1) * 8)) & 0xffu;\n'
                        '    if (group_words > 1) {\n'
                        '      out += "v[" + std::to_string(selector) + ":";\n'
                        '      out += std::to_string(selector + group_words - 1) + "]";\n'
                        '    } else {\n'
                        '      out += "v" + std::to_string(selector);\n'
                        '    }\n'
                        '    consumed_words += group_words;\n'
                        '  }\n'
                        '  out += "]";\n'
                        '}'
                    )
                )
                class_func_impls.append(
                    cgen.Line(
                        f'void {inst_enc.fmt_enc_name}::capture_nsa_words('
                        'const MachineInst *inst, const Operand *vaddr) {\n'
                        '  if (!inst_.nsa) return;\n'
                        '  nsa_vaddr_operand_ = vaddr;\n'
                        '  size_ = sizeof(OpEncoding) + sizeof(MachineInst);\n'
                        '  std::memcpy(raw_words_.data(), inst, size_);\n'
                        '  raw_encoding_ = raw_words_.data();\n'
                        '}'
                    )
                )
            if absolute_source_modifier_opcodes:
                public_members.append(
                    cgen.Line(
                        'void append_src_operand(std::string &out, '
                        'uint8_t operand_index) const override {\n'
                        '  const Operand *operand = src_operands_[operand_index];\n'
                        '  if (!uses_vop3p_absolute_source_syntax() || operand_index >= 3) {\n'
                        '    Instruction::append_src_operand(out, operand_index);\n'
                        '    return;\n'
                        '  }\n'
                        '  if ((inst_.neg >> operand_index) & 1u) out += \'-\';\n'
                        '  const bool absolute = ((inst_.neg_hi >> operand_index) & 1u) != 0;\n'
                        '  if (absolute) out += \'|\';\n'
                        '  const uint32_t selector = operand_index == 0 ? inst_.src0\n'
                        '      : (operand_index == 1 ? inst_.src1 : inst_.src2);\n'
                        '  if (selector == 255) {\n'
                        '    out += "lit(";\n'
                        '    out += operand->name();\n'
                        '    out += \')\';\n'
                        '  } else {\n'
                        '    out += operand->name();\n'
                        '  }\n'
                        '  if (absolute) out += \'|\';\n'
                        '}'
                    )
                )
            # Generate build_modifiers() overrides for encoding bases that
            # display memory flags, DPP controls, or other attributes. This is
            # called lazily by disassemble() instead of eagerly in the
            # constructor, avoiding string allocation on the hot path.
            # The modifier_lines were written for the constructor where
            # they appended to modifiers_ and accessed inst->field.
            # Rewrite to append to 'out' and access via local pointer.
            modifier_impl = (
                modifier_lines.replace('modifiers_', 'out')
                + dpp_modifier_line
                + dpp8_modifier_line
            )
            modifier_prologue = (
                '  auto *inst = &inst_;\n  (void)inst;\n'
                if 'inst->' in modifier_impl
                else ''
            )
            modifier_tail = (
                f'{modifier_prologue}  {modifier_impl}\n' if modifier_impl else ''
            )
            # Keep encoding-base rendering overrides inline. Fixed-profile DBT
            # libraries use these polymorphic bases without linking every ISA
            # model object; an out-of-line virtual can otherwise become the key
            # function and move the class RTTI into an unlinked model object.
            if modifier_impl and not supports_sdwa_extension:
                public_members.append(
                    cgen.Line(
                        'void build_modifiers(std::string &out) const override '
                        f'{{{modifier_prologue}'
                        f'{modifier_impl}}}'
                    ),
                )
            fmt_enc_name = inst_enc.fmt_enc_name
            implicit_uses_impl = self._encoding_implicit_uses_impl(
                inst_enc, enc_field_names
            )
            if implicit_uses_impl:
                public_members.append(
                    cgen.Line('void implicit_uses(RegisterSet &uses) const override;')
                )
                class_func_impls.append(
                    cgen.Line(
                        f'void {fmt_enc_name}::implicit_uses'
                        f'(RegisterSet &uses) const '
                        f'{{ {implicit_uses_impl} }}'
                    )
                )
            # Only profiles with MODE-controlled VGPR high-bank bits (gfx1250)
            # have a consumer: InstDefUse calls this hook exclusively on the
            # vgpr_msb != nullptr path. Emitting it elsewhere would add dead
            # overrides -- generated lines, compile time and vtable slots -- to
            # every other architecture, so leave the base-class no-op in place.
            implicit_use_operands_impl = (
                self._encoding_implicit_use_operands_impl(inst_enc, enc_field_names)
                if self.isa_spec.profile.uses_vgpr_msb_indexing
                else ''
            )
            if implicit_use_operands_impl:
                public_members.append(
                    cgen.Line(
                        'void implicit_use_operands('
                        'std::vector<const ::rocjitsu::Operand *> &operands) const override;'
                    )
                )
                class_func_impls.append(
                    cgen.Line(
                        f'void {fmt_enc_name}::implicit_use_operands'
                        f'(std::vector<const ::rocjitsu::Operand *> &operands) const '
                        f'{{ {implicit_use_operands_impl} }}'
                    )
                )

            if supports_sdwa_extension:
                public_members.append(
                    cgen.Line(
                        'void append_src_operand(std::string &out, '
                        'uint8_t operand_index) const override {\n'
                        '  const ::rocjitsu::Operand *operand = src_operands_[operand_index];\n'
                        '  if (inst_.src0 == amdgpu::SRC_SDWA && operand == sdwa_src0_operand_) {\n'
                        '    amdgpu::sdwa::append_source(out, *operand, sdwa_src0_format_,\n'
                        '                                sdwa_src0_sext_, sdwa_src0_neg_, sdwa_src0_abs_);\n'
                        '    return;\n'
                        '  }\n'
                        '  if (inst_.src0 == amdgpu::SRC_SDWA && operand == sdwa_src1_operand_) {\n'
                        '    amdgpu::sdwa::append_source(out, *operand, sdwa_src1_format_,\n'
                        '                                sdwa_src1_sext_, sdwa_src1_neg_, sdwa_src1_abs_);\n'
                        '    return;\n'
                        '  }\n'
                        '  Instruction::append_src_operand(out, operand_index);\n'
                        '}'
                    )
                )
                if enc_upper == 'ENC_VOPC':
                    public_members.append(
                        cgen.Line(
                            'void build_modifiers(std::string &out) const override {\n'
                            '  if (inst_.src0 == amdgpu::SRC_SDWA)\n'
                            '    amdgpu::sdwa::append_source_attributes(\n'
                            '        out, sdwa_src0_sel_, sdwa_src1_operand_, sdwa_src1_sel_);\n'
                            f'{modifier_tail}'
                            '}'
                        )
                    )
                else:
                    public_members.append(
                        cgen.Line(
                            'void build_modifiers(std::string &out) const override {\n'
                            '  if (inst_.src0 == amdgpu::SRC_SDWA)\n'
                            '    amdgpu::sdwa::append_destination_attributes(\n'
                            '        out, sdwa_clamp_, sdwa_omod_, sdwa_dst_sel_, sdwa_dst_unused_,\n'
                            '        sdwa_src0_sel_, sdwa_src1_operand_, sdwa_src1_sel_);\n'
                            f'{modifier_tail}'
                            '}'
                        )
                    )

            if standard_vop3_disassembly:
                renders_true16_operands = str(
                    profile.renders_true16_vop3_operands and vop3_has_opsel
                ).lower()
                abs_expr = 'inst_.abs' if vop3_has_abs else '0'
                opsel_expr = f'inst_.{vop3_opsel_field}' if vop3_has_opsel else '0'
                public_members.append(
                    cgen.Line(
                        'void append_src_operand(std::string &out, '
                        'uint8_t operand_index) const override {\n'
                        '  const ::rocjitsu::Operand *operand = src_operands_[operand_index];\n'
                        '  const int32_t modifier_index = '
                        'vop3_encoded_source_index(operand_index);\n'
                        '  const auto reg = operand->to_register_ref();\n'
                        f'  const bool half_width = modifier_index >= 0 && {renders_true16_operands} &&\n'
                        '                          operand->size_bits() == 16 &&\n'
                        '                          reg && reg->cls == RegClass::VGPR;\n'
                        '  if (modifier_index < 0) { out += operand->name(); return; }\n'
                        '  amdgpu::vop::append_vop3_operand(\n'
                        f'      out, operand->name(), ({abs_expr} >> modifier_index) & 1,\n'
                        '      (inst_.neg >> modifier_index) & 1, half_width,\n'
                        f'      ({opsel_expr} >> modifier_index) & 1);\n'
                        '}'
                    )
                )
                public_members.append(
                    cgen.Line(
                        'void append_dst_operand(std::string &out, '
                        'uint8_t operand_index) const override {\n'
                        '  const ::rocjitsu::Operand *operand = dst_operands_[operand_index];\n'
                        '  const auto reg = operand->to_register_ref();\n'
                        f'  const bool half_width = {renders_true16_operands} && operand->size_bits() == 16 &&\n'
                        '                          reg && reg->cls == RegClass::VGPR;\n'
                        '  amdgpu::vop::append_vop3_operand(\n'
                        '      out, operand->name(), false, false, half_width, '
                        f'({opsel_expr} >> 3) & 1);\n'
                        '}'
                    )
                )

            if fmt_enc_name not in cond_emitted:
                cond_emitted.add(fmt_enc_name)
                seen_conds: set[str] = set()
                for enc_cond in inst_enc.enc_conds:
                    if enc_cond[0] in seen_conds:
                        continue
                    # Skip default_encoding when it's not used in the
                    # constructor: 64-bit+ encodings (OpEncoding is the
                    # full instruction) or constant-false conditions
                    # (empty XML value, no meaningful runtime check).
                    if enc_cond[0] == 'default_encoding' and not has_real_default_check:
                        continue
                    seen_conds.add(enc_cond[0])
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value('bool', f'{enc_cond[0]}'), []
                    )
                    func_body = cgen.FunctionBody(
                        cgen.FunctionDeclaration(
                            cgen.Value('bool', f'{fmt_enc_name}::{enc_cond[0]}'),
                            [],
                        ),
                        cgen.Block([cgen.Statement(f'return {enc_cond[1]}')]),
                    )
                    public_members.append(func_decl)
                    class_func_impls.append(func_body)

            if inst_enc.has_implied_literal_ops:
                func_decl = cgen.FunctionDeclaration(
                    cgen.Value('bool', 'hasImpliedLiteral'), []
                )
                implied_literal_cond = ' || '.join(
                    f'inst_.op == {op}' for op in inst_enc.implied_literal_ops
                )
                func_body = cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value(
                            'bool',
                            f'{fmt_enc_name}::hasImpliedLiteral',
                        ),
                        [],
                    ),
                    cgen.Block([cgen.Statement(f'return {implied_literal_cond}')]),
                )
                public_members.append(func_decl)
                class_func_impls.append(func_body)

            if implied_literal64_ops:
                func_decl = cgen.FunctionDeclaration(
                    cgen.Value('bool', 'hasImpliedLiteral64'), []
                )
                implied_literal64_cond = ' || '.join(
                    f'inst_.op == {op}' for op in implied_literal64_ops
                )
                func_body = cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value(
                            'bool',
                            f'{fmt_enc_name}::hasImpliedLiteral64',
                        ),
                        [],
                    ),
                    cgen.Block([cgen.Statement(f'return {implied_literal64_cond}')]),
                )
                public_members.append(func_decl)
                class_func_impls.append(func_body)

            if inst_enc.has_variable_implied_literal_size:
                word_count_decl = cgen.FunctionDeclaration(
                    cgen.Value('uint32_t', 'impliedLiteralWordCount'), []
                )
                word_count_expr = ' : '.join(
                    f'inst_.op == {op} ? {words}'
                    for op, words in inst_enc.implied_literal_ops.items()
                )
                word_count_body = cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value(
                            'uint32_t',
                            f'{fmt_enc_name}::impliedLiteralWordCount',
                        ),
                        [],
                    ),
                    cgen.Block([cgen.Statement(f'return {word_count_expr} : 0')]),
                )
                public_members.append(word_count_decl)
                class_func_impls.append(word_count_body)

            class_members.extend(public_members)
            class_members.append(
                cgen.Statement(f'using OpEncoding = {inst_enc.fmt_enc_name}MachineInst')
            )
            class_members.append(cgen.Statement('const OpEncoding inst_'))
            if owns_extension_words:
                raw_word_count = (inst_enc.bit_cnt + 31) // 32 + extension_word_capacity
                class_members.append(
                    cgen.Statement(
                        f'std::array<uint32_t, {raw_word_count}> raw_words_{{}}'
                    )
                )
            elif profile.renders_gfx11_image_syntax and enc_upper == 'ENC_MIMG':
                class_members.append(
                    cgen.Statement('std::array<uint32_t, 5> raw_words_{}')
                )
                class_members.append(
                    cgen.Statement('const Operand *nsa_vaddr_operand_ = nullptr')
                )
            if inst_enc.has_implied_literal_ops:
                class_members.append(cgen.Statement('uint32_t literal_ = 0'))
            # FLAT encoding bases need an owned string for the dynamic mnemonic.
            if rule.use_flat_mnemonic:
                class_members.append(cgen.Statement('std::string owned_mnemonic_'))
            # VOP encoding bases store DPP control fields. The execution-local
            # DPP operand proxies are emitted in execute_impl() rather than
            # retained by the decoded instruction.
            _enc_upper = inst_enc.enc_name.upper()
            _dpp_struct, _dpp8_struct = self._vop_dpp_struct_names(_enc_upper)
            if (
                _dpp_struct
                or _dpp8_struct
                or _enc_upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC')
            ):
                class_members.append(cgen.Statement('uint32_t dpp_ctrl_ = 0'))
                class_members.append(cgen.Statement('uint32_t dpp_row_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bank_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bound_ctrl_ = 0'))
                class_members.append(cgen.Statement('uint32_t dpp_fi_ = 1'))
                if _dpp8_struct:
                    class_members.append(cgen.Statement('uint32_t dpp8_lane_sel_ = 0'))
            if _enc_upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
                # SDWA fields (CDNA and RDNA1/2 have hardware SDWA encoding; fields
                # are present on all ISAs for uniform codegen even if unused).
                class_members.append(
                    cgen.Statement('uint32_t sdwa_src0_sel_ = amdgpu::sdwa::DWORD')
                )
                class_members.append(cgen.Statement('bool sdwa_src0_sext_ = false'))
                class_members.append(cgen.Statement('bool sdwa_src0_neg_ = false'))
                class_members.append(cgen.Statement('bool sdwa_src0_abs_ = false'))
                class_members.append(
                    cgen.Statement('uint32_t sdwa_src1_sel_ = amdgpu::sdwa::DWORD')
                )
                class_members.append(cgen.Statement('bool sdwa_src1_sext_ = false'))
                class_members.append(cgen.Statement('bool sdwa_src1_neg_ = false'))
                class_members.append(cgen.Statement('bool sdwa_src1_abs_ = false'))
                if supports_sdwa_extension:
                    class_members.append(
                        cgen.Statement('const Operand *sdwa_src0_operand_ = nullptr')
                    )
                    class_members.append(
                        cgen.Statement('const Operand *sdwa_src1_operand_ = nullptr')
                    )
                    class_members.append(
                        cgen.Statement(
                            'amdgpu::sdwa::SourceModifierFormat sdwa_src0_format_ = '
                            'amdgpu::sdwa::SourceModifierFormat::NONE'
                        )
                    )
                    class_members.append(
                        cgen.Statement(
                            'amdgpu::sdwa::SourceModifierFormat sdwa_src1_format_ = '
                            'amdgpu::sdwa::SourceModifierFormat::NONE'
                        )
                    )
                if inst_enc.enc_name.upper() != 'ENC_VOPC':
                    class_members.append(
                        cgen.Statement('uint32_t sdwa_dst_sel_ = amdgpu::sdwa::DWORD')
                    )
                    class_members.append(
                        cgen.Statement('uint32_t sdwa_dst_unused_ = 0')
                    )
                    class_members.append(cgen.Statement('bool sdwa_clamp_ = false'))
                    if supports_sdwa_extension:
                        class_members.append(cgen.Statement('uint32_t sdwa_omod_ = 0'))
                else:
                    class_members.append(cgen.Statement('uint32_t sdwa_sdst_ = 106'))
                    class_members.append(cgen.Statement('bool sdwa_sd_ = false'))
            s = cgen.Struct(
                f'{inst_enc.fmt_enc_name} : public IsaInstruction<Isa>',
                [x for x in class_members],
            )
            enc_classes.append(s)

        if self._supports_simm64_literal_operands():
            enc_classes.insert(
                0,
                cgen.Line(
                    'enum class LiteralSupport : uint8_t {\n'
                    '  None = 0,\n'
                    '  Literal32 = 1,\n'
                    '  Literal64 = 2,\n'
                    '  Both = 3,\n'
                    '};\n\n'
                    '[[nodiscard]] constexpr bool supports_literal(\n'
                    '    LiteralSupport support, LiteralSupport required) {\n'
                    '  return (static_cast<uint8_t>(support) &\n'
                    '          static_cast<uint8_t>(required)) != 0;\n'
                    '}'
                ),
            )

        encoding_constants = self._encoding_constants_block()
        if encoding_constants is not None:
            enc_classes.insert(0, encoding_constants)

        encoding_helper_header = (
            'rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h'
            if self.isa_spec.profile.split_execution_sources
            else 'rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h'
        )
        from amdisa.isa_profile import MemoryCoherencyModel

        needs_gfx12_cache_helpers = (
            self.isa_spec.profile.coherency_model == MemoryCoherencyModel.GFX12_SCOPE_TH
        )
        needs_sdwa_helpers = any(
            self._supports_sdwa_for_encoding(enc.enc_name)
            for enc in self.isa_spec.inst_encodings
        )
        _enc_h_includes = [
            (
                self.config.handwritten_include(self.handwritten_dir_name, 'isa.h'),
                False,
            ),
            (
                self.config.generated_include(
                    self.generated_dir_name, 'machine_insts.h'
                ),
                False,
            ),
            ('rocjitsu/isa/instruction.h', False),
            ('rocjitsu/isa/decode_result.h', False),
            (encoding_helper_header, False),
            ('array', True),
            ('cstdint', True),
            ('string', True),
            ('string_view', True),
        ]
        if needs_gfx12_cache_helpers:
            _enc_h_includes.insert(
                5,
                ('rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h', False),
            )
        if (
            needs_sdwa_helpers
            and encoding_helper_header
            != 'rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h'
        ):
            _enc_h_includes.insert(
                5,
                ('rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h', False),
            )
        class_def_file = CppFile(
            'encodings',
            self.out_path,
            True,
            _enc_h_includes,
            [],
            enc_classes,
            self.cpp_namespace,
            True,
            generated_dir_name=self.generated_dir_name,
        )
        needs_flat_mnemonic = any(
            self.isa_spec.profile.mnemonic_rule(enc.enc_name).use_flat_mnemonic
            for enc in self.isa_spec.inst_encodings
        )
        if needs_flat_mnemonic:
            flat_mnemonic_helper = cgen.Line(
                'namespace {\n'
                'std::string flat_mnemonic(std::string_view mnemonic, int seg) {\n'
                '  // seg: 0=FLAT, 1=SCRATCH, 2=GLOBAL\n'
                '  if (seg == 1 && mnemonic.substr(0, 5) == "flat_")\n'
                '    return std::string("scratch_").append(mnemonic.substr(5));\n'
                '  if (seg == 2 && mnemonic.substr(0, 5) == "flat_")\n'
                '    return std::string("global_").append(mnemonic.substr(5));\n'
                '  return std::string(mnemonic);\n'
                '}\n'
                '} // namespace'
            )
            class_func_impls.insert(0, flat_mnemonic_helper)

        _enc_cpp_includes = [
            (
                self.config.generated_include(self.generated_dir_name, 'encodings.h'),
                False,
            ),
            ('util/except.h', False),
            ('cstring', True),
            ('string', True),
        ]
        if needs_gfx12_cache_helpers:
            _enc_cpp_includes.insert(
                1,
                (
                    'rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h',
                    False,
                ),
            )
        if needs_sdwa_helpers:
            _enc_cpp_includes.insert(
                1, ('rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h', False)
            )
        class_impl_file = CppFile(
            'encodings',
            self.out_path,
            False,
            _enc_cpp_includes,
            [],
            class_func_impls,
            self.cpp_namespace,
            generated_dir_name=self.generated_dir_name,
        )
        class_def_file.gen_code()
        class_impl_file.gen_code()

    def _encoding_implicit_uses_impl(
        self, inst_enc: InstEncoding, enc_field_names: set[str]
    ) -> str:
        """Return C++ body for hidden register uses on an encoding base."""
        if (
            inst_enc.enc_name.upper() == 'ENC_FLAT'
            and {'seg', 'saddr'} <= enc_field_names
        ):
            saddr_null = self._saddr_null_expr(inst_enc.enc_name)
            return (
                f'if (inst_.saddr == {saddr_null}) return;'
                'if (inst_.seg == 1) {'
                'uses.expand(RegisterRef{RegClass::SGPR, '
                'static_cast<uint16_t>(inst_.saddr), 1});'
                '} else if (inst_.seg == 2) {'
                'uses.expand(RegisterRef{RegClass::SGPR, '
                'static_cast<uint16_t>(inst_.saddr), 2});'
                '}'
            )
        # SDWA dst_unused:PRESERVE and DPP that keeps lanes (partial row/bank
        # mask, or bound_ctrl == 0 with an invalid source) leave the old vector
        # destination in place, so surface the preserved destination as a use.
        # This matches what the executor actually restores: the non-VOPC SDWA/DPP
        # merge only rewrites old vdst bytes/lanes via write_vgpr, so only a
        # VGPR-class destination is preserved. SGPR destinations are fully
        # written and never restored -- for example a VOP3_SDST_ENC carry
        # (v_add_co_ci) -- so filter this encoding-level hook to RegClass::VGPR.
        # SDWA exists only on VOP1/VOP2; DPP additionally exists on VOP3/VOP3P/
        # VOP3_SDST_ENC on gfx11+ (gated -- GCN/CDNA lack the dpp_* fields).
        # VOPC/VOPCX stay omitted: their result lands in VCC/EXEC, which liveness
        # does not track (see liveness.h).
        return self._encoding_preserved_dst_body(
            inst_enc, enc_field_names, 'uses.expand(*ref);'
        )

    def _encoding_implicit_use_operands_impl(
        self, inst_enc: InstEncoding, enc_field_names: set[str]
    ) -> str:
        """Return C++ body for operand-backed hidden uses on an encoding base.

        Mirrors the operand-backed reads of _encoding_implicit_uses_impl (the
        SDWA/DPP preserved destination), but appends the source Operand pointers
        so a caller can resolve each with its own VGPR-MSB role and width. The
        FLAT/GLOBAL saddr case is intentionally omitted: it reads an encoded
        field with no backing Operand, so it stays exclusive to implicit_uses().
        """
        return self._encoding_preserved_dst_body(
            inst_enc, enc_field_names, 'operands.push_back(dst);'
        )

    def _encoding_preserved_dst_body(
        self, inst_enc: InstEncoding, enc_field_names: set[str], emit: str
    ) -> str:
        """Return the SDWA/DPP preserved-destination body, or '' if not applicable.

        SINGLE SOURCE OF TRUTH for both implicit_uses() and
        implicit_use_operands(): the two hooks must agree on exactly which
        encodings preserve their destination, and differ only in how they report
        it (`emit`). This is load-bearing rather than cosmetic -- on gfx1250,
        InstDefUse clears the VGPR class from the flat implicit_uses() result and
        takes banked VGPR reads solely from implicit_use_operands(), so an
        encoding covered by one predicate but not the other would silently vanish
        from liveness instead of failing a build. Keeping one predicate makes that
        drift impossible to express.
        """
        enc = inst_enc.enc_name.upper()
        has_sdwa = enc in ('ENC_VOP1', 'ENC_VOP2')
        has_dpp = enc in ('ENC_VOP1', 'ENC_VOP2') or (
            enc in ('ENC_VOP3', 'ENC_VOP3P', 'VOP3_SDST_ENC')
            and self._supports_vop_dpp_encoding(enc)
        )
        if not (has_sdwa or has_dpp) or 'vdst' not in enc_field_names:
            return ''
        decls = ''
        guards = []
        if has_sdwa:
            decls += (
                'bool sdwa_preserve = sdwa_dst_sel_ != amdgpu::sdwa::DWORD && '
                'sdwa_dst_unused_ == amdgpu::sdwa::UNUSED_PRESERVE; '
            )
            guards.append('sdwa_preserve')
        if has_dpp:
            invalid_source_partial = (
                '(amdgpu::dpp::dpp_ctrl_produces_oob(dpp_ctrl_) || dpp_fi_ == 0)'
                if getattr(
                    self.isa_spec.profile,
                    'dpp_bound_ctrl_applies_to_inactive_sources',
                    False,
                )
                else 'amdgpu::dpp::dpp_ctrl_produces_oob(dpp_ctrl_)'
            )
            decls += (
                'bool dpp_partial = inst_.src0 == amdgpu::SRC_DPP && '
                '(dpp_row_mask_ != 0xF || dpp_bank_mask_ != 0xF || '
                '(dpp_bound_ctrl_ == 0 && '
                f'{invalid_source_partial})); '
            )
            guards.append('dpp_partial')
        return (
            decls + f'if ({" || ".join(guards)}) '
            'for (int i = 0; i < num_dst_operands(); ++i) '
            'if (const auto *dst = dst_operand(i)) '
            'if (auto ref = dst->to_register_ref()) '
            'if (ref->cls == RegClass::VGPR) '
            f'{emit}'
        )

    def _saddr_null_expr(self, enc_name: str) -> str:
        """Return the profile-defined NULL-SADDR selector for an encoding."""
        expr = self.isa_spec.profile.saddr_null_selector_expr(enc_name)
        if expr is None:
            raise ValueError(f'{enc_name} does not define a NULL-SADDR selector')
        return expr

    def _enc_field_at_bit(self, enc_name: str, bit_offset: int) -> str | None:
        """Return the field name at a given bit offset in an encoding, or None."""
        for enc in self.isa_spec.inst_encodings:
            if enc.enc_name.upper() == enc_name.upper():
                for f in enc.ucode_fields:
                    if f.bit_offset == bit_offset:
                        return f.name
        return None

    def _op_sel_hi_2_field(self, enc_name: str) -> str:
        """Return the encoded field carrying op_sel_hi for the third source.

        Some ISA specs (e.g. CDNA4) removed the explicit op_sel_hi_2
        field from Vop3pMachineInst and declared the bit as reserved
        (pad_14). The hardware behavior is unchanged - access whatever
        field sits at bit 14.
        """
        return self._enc_field_at_bit(enc_name, 14) or 'op_sel_hi_2'

    def _op_sel_hi_2_expr(self, enc_name: str) -> str:
        """Return the instruction-member expression for op_sel_hi source 2."""
        return f'inst_.{self._op_sel_hi_2_field(enc_name)}'

    # Semantic operations/classes where the primary destination register is also
    # read. Several CDNA4 XML operands are marked output-only even though the
    # execute body consumes the old register value. This matters outside the
    # emulator: liveness and DBT scratch allocation are built from the generated
    # ``src_operands_``/``dst_operands_`` tables, so read/write operands must be
    # reflected there even when the XML direction bits are incomplete.
    _READS_DST_OPS = frozenset({'fmac', 'bitset0', 'bitset1'})
    _READS_DST_CLASSES = frozenset(
        {
            'vector_dot',
            'vector_writelane',
            'scalar_addk',
            'scalar_mulk',
            'scalar_cmov',
            'scalar_cmovk',
            'mad_mixlo_f16',
            'mad_mixhi_f16',
            'mad_mixlo_bf16',
            'mad_mixhi_bf16',
        }
    )
    _READS_ALL_OUTPUT_CLASSES = frozenset(
        {
            'vector_swap',
            'vector_swaprel',
            'vector_permlane16_swap',
            'vector_permlane32_swap',
        }
    )
    _READS_DST_MNEMONICS = frozenset(
        {
            # These forms currently use special/stub execute paths, so their
            # semantic class alone is not enough to recover the accumulator.
            'V_CVT_PKACCUM_U8_F32',
            'V_PK_FMAC_F16',
        }
    )
    _READS_DST_PREFIXES = ('V_SMFMAC_',)
    _READS_DATA_OUTPUT_PREFIXES = ('BUFFER_ATOMIC_', 'S_ATOMIC_', 'S_BUFFER_ATOMIC_')
    _READS_DST_DOT_PREFIXES = (
        'V_DOT2ACC_',
        'V_DOT2C_',
        'V_DOT4C_',
        'V_DOT8C_',
    )
    # D16 load semantic classes. Stores share the d16_hi/d16_lo flags, so the
    # class gate is what restricts the partial-def treatment to loads.
    # global/scratch loads use the 'flat_load' class. Byte/short buffer D16 loads
    # (e.g. buffer_load_ubyte_d16) use 'buffer_load'/'tbuffer_load' and execute
    # correctly; packed D16 FORMAT loads use the non-executable
    # 'buffer_load_format_d16' class (metadata-only, see semantics._derive_buffer_data).
    _D16_LOAD_CLASSES = frozenset(
        {
            'flat_load',
            'buffer_load',
            'tbuffer_load',
            'ds_read',
            'buffer_load_format_d16',
        }
    )

    def _dst_is_also_source(self, inst: Instruction) -> bool:
        """Return True if the instruction reads from its destination operand.

        Some ISA XML specs mark accumulator destinations as output-only even
        though the instruction reads the old value (e.g. fused multiply-
        accumulate, dot product accumulate, swap, bitset).  This method
        identifies such instructions via their semantics so the constructor
        can register the destination in both src_operands_ and dst_operands_.
        """
        sem = self.semantics.instructions.get(inst.name) if self.semantics else None
        return self._instruction_reads_primary_dst(inst, sem)

    def _instruction_reads_primary_dst(
        self, inst: Instruction, sem: InstructionSemantics | None
    ) -> bool:
        """Return True when an instruction reads its first destination operand."""
        name = inst.name.upper()
        return (
            (sem is not None and sem.operation in self._READS_DST_OPS)
            or (sem is not None and sem.semantic_class in self._READS_DST_CLASSES)
            or (
                sem is not None and sem.semantic_class in self._READS_ALL_OUTPUT_CLASSES
            )
            or name in self._READS_DST_MNEMONICS
            or name.startswith(self._READS_DST_PREFIXES)
            or name.startswith(self._READS_DST_DOT_PREFIXES)
        )

    def _d16_load_reads_dst(self, inst: Instruction) -> bool:
        """Return True for a D16(_HI) load whose last dst register is partial.

        The last register is partial when the written bytes (num_elems *
        elem_size) do not fill whole 32-bit registers (e.g. ushort_d16, or
        format_d16_xyz). Architectures with SRAM ECC can instead zero-fill the
        unselected half, making the destination a full write rather than a
        read-modify-write.
        """
        if not self.semantics:
            return False
        if self.isa_spec.profile.d16_loads_zero_unselected_half:
            return False
        sem = self.semantics.instructions.get(inst.name)
        if not sem or sem.num_elems is None or sem.elem_size is None:
            return False
        return (
            (sem.d16_hi or sem.d16_lo)
            and (sem.num_elems * sem.elem_size) % 4 != 0
            and sem.semantic_class in self._D16_LOAD_CLASSES
        )

    def _operand_size_override(
        self,
        enc_name: str,
        opnd: Operand,
        sem: InstructionSemantics | None,
    ) -> str | None:
        """Return a target-specific generated operand size override."""
        if (
            sem is not None
            and sem.semantic_class in ('vector_cmp', 'vector_cmp_class')
            and enc_name.upper() == 'ENC_VOP3'
            and opnd.is_output
            and opnd.name in ('vdst', 'sdst')
            and opnd.operand_type.upper() == 'OPR_SREG'
            and self.isa_spec.profile.vop3_cmp_sdst_size_bits is not None
        ):
            return str(self.isa_spec.profile.vop3_cmp_sdst_size_bits)
        if (
            sem is not None
            and sem.semantic_class == 'vector_cndmask'
            and enc_name.upper() == 'ENC_VOP3'
            and not opnd.is_output
            and opnd.name == 'src2'
            and opnd.operand_type.upper() == 'OPR_SREG'
            and self.isa_spec.profile.vop3_cndmask_selector_size_bits is not None
        ):
            return str(self.isa_spec.profile.vop3_cndmask_selector_size_bits)
        if sem is not None and enc_name.upper() == 'ENC_VOP3':
            is_carry_mask = sem.semantic_class == 'vector_add_co' and opnd.name in (
                'sdst',
                'src2',
            )
            is_aux_mask_output = (
                sem.semantic_class in ('vector_div_scale', 'vector_mad_64_32')
                and opnd.is_output
                and opnd.name == 'sdst'
            )
            if (
                (is_carry_mask or is_aux_mask_output)
                and opnd.operand_type.upper() == 'OPR_SREG'
                and self.isa_spec.profile.vop3_carry_mask_size_bits is not None
            ):
                return str(self.isa_spec.profile.vop3_carry_mask_size_bits)
        return None

    def _output_operand_is_also_source(self, inst: Instruction, opnd: Operand) -> bool:
        """Return True if a generated output operand must also be a source.

        The predicate is intentionally operand-specific. Some instructions
        preserve/consume only the primary destination (FMAC, writelane), while
        swap-style instructions read every output operand and atomics read their
        data operand even when it is also the return register.
        """
        if not opnd.is_output:
            return False

        sem = self.semantics.instructions.get(inst.name) if self.semantics else None
        name = inst.name.upper()

        if opnd.name in ('vdata', 'sdata') and (
            (sem is not None and sem.semantic_class == 'buffer_atomic')
            or name.startswith(self._READS_DATA_OUTPUT_PREFIXES)
        ):
            return True

        if sem is not None and sem.semantic_class in self._READS_ALL_OUTPUT_CLASSES:
            return True

        if opnd.name in ('vdst', 'sdst') and self._instruction_reads_primary_dst(
            inst, sem
        ):
            return True

        # A sub-dword (true16/byte) vector destination writes only part of its
        # 32-bit register lane, so the old value is a real input. That partial-def
        # read is surfaced separately as an implicit_uses() override (see
        # _partial_def_outputs), so it must NOT also be appended to src_operands_:
        # doing so would print the destination a second time in disassembly and
        # (for the architectural operand list) misrepresent the instruction's
        # sources. Liveness/DBT already pick up the read via InstDefUse merging
        # implicit_uses, so returning False here keeps the operand tables faithful
        # to the ISA without losing the dependency.
        return False

    _VGPR_MSB_SRC_ROLES = ('Src0', 'Src1', 'Src2')

    @staticmethod
    def _operand_type_can_name_vgpr(opr_type: str) -> bool:
        """Return True if an operand type can resolve to VGPR storage."""
        opr_type = opr_type.upper()
        return (
            opr_type.startswith('OPR_VGPR')
            or opr_type.startswith('OPR_SRC')
            or 'ACCVGPR' in opr_type
        )

    @classmethod
    def _operand_can_use_vgpr_msb(cls, opnd: Operand) -> bool:
        """Return True if an operand can name a VGPR at execution time."""
        return cls._operand_type_can_name_vgpr(opnd.operand_type)

    def _operand_uses_packed_16bit_source(
        self, enc_name: str, opnd: Operand, *, reads_dst: bool = False
    ) -> bool:
        """Return True for E32 16-bit sources with packed-half selectors."""
        profile = self.isa_spec.profile
        if not profile.uses_packed_16bit_e32_source_selectors:
            return False
        if enc_name.upper() not in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
            return False
        if opnd.size != 16:
            return False
        if opnd.is_input:
            return opnd.operand_type in ('OPR_SRC', 'OPR_SRC_VGPR', 'OPR_VGPR')
        if (
            reads_dst
            and opnd.is_output
            and getattr(opnd, 'name', '') in ('vdst', 'sdst')
        ):
            return opnd.operand_type in ('OPR_SRC', 'OPR_SRC_VGPR', 'OPR_VGPR')
        return False

    def _operand_uses_packed_16bit_dst(self, enc_name: str, opnd: Operand) -> bool:
        """Return True for E32 16-bit destinations with packed-half selectors."""
        profile = self.isa_spec.profile
        return (
            profile.uses_packed_16bit_e32_source_selectors
            and enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2')
            and opnd.size == 16
            and opnd.is_output
            and opnd.operand_type == 'OPR_VGPR'
        )

    def _vbuffer_store_data_uses_dst_vgpr_msb_role(
        self, enc_name: str, sem: InstructionSemantics | None, opnd: Operand
    ) -> bool:
        profile = self.isa_spec.profile
        return (
            profile.vbuffer_store_data_uses_dst_vgpr_msb_role
            and enc_name.upper() == 'ENC_VBUFFER'
            and sem is not None
            and sem.semantic_class == 'buffer_store'
            and opnd.name == 'vdata'
        )

    def _fixed_vgpr_msb_role(
        self, enc_name: str, inst_name: str, opnd: Operand
    ) -> str | None:
        """Return the MODE role fixed by an encoding's physical operand slot."""
        enc_upper = enc_name.upper()
        inst_upper = inst_name.upper()
        opnd_name = opnd.name

        if enc_upper in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOP3', 'ENC_VOP3P', 'ENC_VOPC'):
            if (
                enc_upper == 'ENC_VOP2'
                and inst_upper
                in (
                    'V_FMAMK_F16',
                    'V_FMAMK_F32',
                    'V_FMAMK_F64',
                    'V_MADMK_F16',
                    'V_MADMK_F32',
                )
                and opnd_name == 'vsrc1'
            ):
                return 'Src2'
            return {
                'src0': 'Src0',
                'src1': 'Src1',
                'vsrc1': 'Src1',
                'src2': 'Src2',
                'vsrc2': 'Src2',
                'vdst': 'Dst',
            }.get(opnd_name)

        if enc_upper == 'ENC_VDS':
            return {
                'addr': 'Src0',
                'data0': 'Src1',
                'data1': 'Src2',
                'vdst': 'Dst',
            }.get(opnd_name)

        if enc_upper in ('ENC_VFLAT', 'ENC_VGLOBAL'):
            return {
                'vaddr': 'Src0',
                'vdata': 'Src1',
                'vsrc': 'Src1',
                'vdst': 'Dst',
            }.get(opnd_name)

        return None

    def _supports_cdna5_scaled_wmma_vop3px2(self) -> bool:
        return (
            self.isa_spec.arch_name.lower() == 'cdna5'
            and self.isa_spec.profile.generate_scaled_wmma_vop3px2
        )

    def _cdna_mfma_f8f6f4_vop3px2_specs(self):
        instruction_names = {
            inst.name for enc in self.isa_spec.inst_encodings for inst in enc.insts
        }
        return tuple(
            spec
            for spec in getattr(self.isa_spec.profile, "mfma_scale_vop3px2_specs", ())
            if spec.dense_name in instruction_names
        )

    def _supports_cdna_mfma_f8f6f4_vop3px2(self) -> bool:
        return bool(self._cdna_mfma_f8f6f4_vop3px2_specs())

    def _is_cdna_mfma_f8f6f4_vop3px2_suffix(self, inst: Instruction) -> bool:
        return any(
            inst.name == spec.dense_name
            for spec in self._cdna_mfma_f8f6f4_vop3px2_specs()
        )

    def _emit_cdna_mfma_f8f6f4_vop3px2_decoder_helpers(self) -> str:
        opcodes = ' || '.join(
            f'op2 == {spec.opcode}' for spec in self._cdna_mfma_f8f6f4_vop3px2_specs()
        )
        prefix_opcode = self._cdna_mfma_f8f6f4_vop3px2_specs()[0].prefix_opcode
        return textwrap.dedent(f'''\
            namespace {{

            bool isLegalMfmaScaleF8f6f4Selector(uint32_t selector) {{
              return (selector >= 240 && selector <= 248) || (selector >= 256 && selector <= 511);
            }}

            bool isLegalMfmaF8f6f4Format(uint32_t format) {{ return format <= 4; }}

            bool isMfmaScaleF8f6f4Vop3px2(const MachineInst *opcode) {{
              constexpr uint32_t VOP3P_MFMA_ENC = 423;
              constexpr uint32_t PREFIX_OP = {prefix_opcode};
              auto enc0 = (opcode[0] >> 23) & 0x1FFu;
              auto op0 = (opcode[0] >> 16) & 0x7Fu;
              if (enc0 != VOP3P_MFMA_ENC || op0 != PREFIX_OP)
                return false;
              auto enc2 = (opcode[2] >> 23) & 0x1FFu;
              auto op2 = (opcode[2] >> 16) & 0x7Fu;
              auto abid2 = (opcode[2] >> 11) & 0xFu;
              return enc2 == VOP3P_MFMA_ENC && ({opcodes}) && abid2 == 1u;
            }}

            bool isValidMfmaScaleF8f6f4(const MachineInst *opcode) {{
              auto scale_src0 = opcode[1] & 0x1FFu;
              auto scale_src1 = (opcode[1] >> 9) & 0x1FFu;
              if (!isLegalMfmaScaleF8f6f4Selector(scale_src0) ||
                  !isLegalMfmaScaleF8f6f4Selector(scale_src1))
                return false;
              auto suffix = *reinterpret_cast<const Vop3pMfma::OpEncoding *>(opcode + 2);
              if (!isLegalMfmaF8f6f4Format(suffix.cbsz) ||
                  !isLegalMfmaF8f6f4Format(suffix.blgp))
                return false;
              auto neg = (opcode[1] >> 29) & 0x7u;
              auto neg_hi = (opcode[0] >> 8) & 0x7u;
              return (neg & 0x3u) == 0 && (neg_hi & 0x3u) == 0;
            }}

            }} // namespace
            ''')

    def _emit_cdna_mfma_f8f6f4_vop3px2_classes(self) -> str:
        classes = []
        for spec in self._cdna_mfma_f8f6f4_vop3px2_specs():
            classes.append(textwrap.dedent(f'''\
                    class {spec.class_name} : public Vop3pMfma {{
                    public:
                      {spec.class_name}(const MachineInst *inst);
                      void execute_impl(amdgpu::Wavefront &wf);

                      Operand vdst;
                      Operand src0;
                      Operand src1;
                      Operand src2;
                      Operand scale_src0;
                      Operand scale_src1;

                    private:
                      std::array<uint32_t, 4> raw_words_{{}};
                    }};
                    '''))
        return '\n'.join(classes)

    def _emit_cdna_mfma_f8f6f4_vop3px2_impls(self) -> _ImplOutputs:
        outputs = _ImplOutputs()
        for spec in self._cdna_mfma_f8f6f4_vop3px2_specs():
            exec_fn = self._split_execute_expr(spec.class_name)
            output_bits = spec.m * spec.n * 32 // 64
            outputs.model.append(textwrap.dedent(f'''\
                    {spec.class_name}::{spec.class_name}(const MachineInst *inst)
                        : Vop3pMfma("{spec.mnemonic}",
                                    reinterpret_cast<const OpEncoding *>(inst + 2), {exec_fn}),
                          vdst({output_bits}, OperandType::OPR_VGPR_OR_ACCVGPR,
                               (reinterpret_cast<const OpEncoding *>(inst + 2)->vdst +
                                (reinterpret_cast<const OpEncoding *>(inst + 2)->acc_cd
                                     ? OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN
                                     : 0))),
                          src0(cdna4_matrix_fmt_operand_size_bits(
                                   reinterpret_cast<const OpEncoding *>(inst + 2)->cbsz,
                                   {spec.m}, {spec.k}),
                               OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
                               (reinterpret_cast<const OpEncoding *>(inst + 2)->src0 +
                                ((reinterpret_cast<const OpEncoding *>(inst + 2)->acc & 0x1u)
                                     ? (OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN -
                                        OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN)
                                     : 0))),
                          src1(cdna4_matrix_fmt_operand_size_bits(
                                   reinterpret_cast<const OpEncoding *>(inst + 2)->blgp,
                                   {spec.n}, {spec.k}),
                               OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
                               (reinterpret_cast<const OpEncoding *>(inst + 2)->src1 +
                                ((reinterpret_cast<const OpEncoding *>(inst + 2)->acc & 0x2u)
                                     ? (OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN -
                                        OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN)
                                     : 0))),
                          src2({output_bits}, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
                               mfma_src2_encoding(
                                   reinterpret_cast<const OpEncoding *>(inst + 2)->src2,
                                   reinterpret_cast<const OpEncoding *>(inst + 2)->acc_cd)),
                          scale_src0(32, OperandType::OPR_SRC_SIMPLE,
                                     reinterpret_cast<const OpEncoding *>(inst)->src0),
                          scale_src1(32, OperandType::OPR_SRC_SIMPLE,
                                     reinterpret_cast<const OpEncoding *>(inst)->src1),
                          raw_words_{{inst[0], inst[1], inst[2], inst[3]}} {{
                      size_ = 16;
                      raw_encoding_ = raw_words_.data();
                      dst_operands_[0] = &vdst;
                      src_operands_[0] = &src0;
                      src_operands_[1] = &src1;
                      src_operands_[2] = &src2;
                      src_operands_[3] = &scale_src0;
                      src_operands_[4] = &scale_src1;
                      num_src_ = 5;
                      num_dst_ = 1;
                      flags_ |= MFMA;
                    }}
                    '''))
            outputs.execution.append(textwrap.dedent(f'''\
                    void {spec.class_name}::execute_impl(amdgpu::Wavefront &wf) {{
                      auto &cu = wf.cu();
                      uint32_t vb = wf.vgpr_alloc().base;
                      uint32_t dst = amdgpu::dst_base(vb, vdst.encoding_value_, inst_.acc_cd);
                      uint32_t const_acc;
                      uint32_t s2 = amdgpu::resolve_acc(
                          vb, dst, src2.encoding_value_, const_acc,
                          [&] {{ return amdgpu::RegisterAccess(wf).read_scalar(src2); }});
                      uint32_t s0b = amdgpu::src_base(vb, src0.encoding_value_);
                      uint32_t s1b = amdgpu::src_base(vb, src1.encoding_value_);
                      uint32_t scale_a = raw_words_[1] & 0x1FFu;
                      uint32_t scale_b = (raw_words_[1] >> 9) & 0x1FFu;
                      uint32_t scale_a_byte =
                          ((raw_words_[0] >> 11) & 1u) | (((raw_words_[1] >> 27) & 1u) << 1);
                      uint32_t scale_b_byte =
                          ((raw_words_[0] >> 12) & 1u) | (((raw_words_[1] >> 28) & 1u) << 1);
                      uint32_t c_modifier = amdgpu::wmma_c_modifier(
                          (raw_words_[1] >> 29) & 7u, (raw_words_[0] >> 8) & 7u);
                      bool dispatched = amdgpu::dispatch_matrix_fmt_pair(
                          inst_.cbsz, inst_.blgp,
                          [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {{
                            amdgpu::exec_f32_scaled_mixed(
                                cu, {spec.m}, {spec.n}, {spec.k}, 1, a_bits, b_bits, dst, s0b,
                                s1b, s2, ea, eb, const_acc, vb, scale_a, scale_b, scale_a_byte,
                                scale_b_byte, c_modifier);
                          }});
                      if (!dispatched)
                        throw util::UnimplementedInst(mnemonic());
                    }}
                    '''))
        return outputs

    def _cdna5_f8f6f4_wmma_shape(
        self, inst: Instruction
    ) -> tuple[int, int, int] | None:
        if self.isa_spec.arch_name != 'cdna5' or not inst.name.startswith('V_WMMA_'):
            return None
        m = re.match(
            r'V_WMMA_(?:F32|F16|BF16|I32)_(\d+)X(\d+)X(\d+)_?F8F6F4$',
            inst.name,
        )
        if not m:
            return None
        return tuple(int(x) for x in m.groups())

    def _cdna4_f8f6f4_mfma_shape(
        self, inst: Instruction
    ) -> tuple[int, int, int] | None:
        return next(
            (
                (spec.m, spec.n, spec.k)
                for spec in self._cdna_mfma_f8f6f4_vop3px2_specs()
                if inst.name == spec.dense_name
            ),
            None,
        )

    def _cdna5_swmmac_has_modifiers(self, inst: Instruction) -> bool:
        return self.isa_spec.arch_name == 'cdna5' and inst.name.startswith('V_SWMMAC_')

    def _cdna5_matrix_fmt_operand_size_expr(
        self, shape: tuple[int, int, int] | None, opnd_name: str
    ) -> str | None:
        if shape is None or opnd_name not in ('src0', 'src1'):
            return None
        m, n, k = shape
        dim = m if opnd_name == 'src0' else n
        if opnd_name == 'src0':
            fmt_expr = 'reinterpret_cast<const OpEncoding *>(inst)->opsel'
        else:
            opsel_hi_2 = self._op_sel_hi_2_field('ENC_VOP3P')
            fmt_expr = (
                f'((reinterpret_cast<const OpEncoding *>(inst)->{opsel_hi_2} << 2) | '
                'reinterpret_cast<const OpEncoding *>(inst)->opsel_hi)'
            )
        return f'cdna5_matrix_fmt_operand_size_bits({fmt_expr}, {dim}, {k})'

    @staticmethod
    def _cdna4_matrix_fmt_operand_size_expr(
        shape: tuple[int, int, int] | None, opnd_name: str
    ) -> str | None:
        if shape is None or opnd_name not in ('src0', 'src1'):
            return None
        m, n, k = shape
        dim = m if opnd_name == 'src0' else n
        fmt_field = 'cbsz' if opnd_name == 'src0' else 'blgp'
        fmt_expr = f'reinterpret_cast<const OpEncoding *>(inst)->{fmt_field}'
        return f'cdna4_matrix_fmt_operand_size_bits({fmt_expr}, {dim}, {k})'

    @staticmethod
    def _buffer_vaddr_operand_size_expr(enc_name: str, opnd_name: str) -> str | None:
        if (
            enc_name.upper() not in ('ENC_MUBUF', 'ENC_MTBUF', 'ENC_VBUFFER')
            or opnd_name != 'vaddr'
        ):
            return None
        if enc_name.upper() == 'ENC_VBUFFER':
            return 'vbuffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
        return 'buffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'

    @staticmethod
    def _vflat_vaddr_operand_size_expr(enc_name: str, opnd_name: str) -> str | None:
        """Return the semantic VADDR width for GFX12 flat/global encodings.

        LLVM's FLAT pseudos use a 64-bit VGPR address for the vector-only form,
        but a 32-bit VGPR offset when SADDR supplies the scalar base. The ISA
        XML describes both variants with the same encoding and operand, so the
        generated decoder must derive the width from the encoded SADDR value.
        VSCRATCH has its own fixed-width operand and does not use this rule.
        """
        if enc_name.upper() not in ('ENC_VFLAT', 'ENC_VGLOBAL') or opnd_name != 'vaddr':
            return None
        return 'vflat_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'

    def _mimg_operand_size_expr(
        self, enc_name: str, inst_name: str, opnd_name: str
    ) -> str | None:
        if (
            not self.isa_spec.profile.renders_gfx11_image_syntax
            or enc_name.upper() != 'ENC_MIMG'
        ):
            return None
        if opnd_name == 'vdata':
            fixed_words = self.isa_spec.profile.gfx11_mimg_fixed_vdata_words.get(
                inst_name
            )
            if fixed_words is not None:
                return str(fixed_words * 32)
            gather = str(
                inst_name.upper().startswith('IMAGE_GATHER')
                or inst_name
                in self.isa_spec.profile.gfx11_mimg_gather_style_instructions
            ).lower()
            return (
                'mimg_vdata_bits(reinterpret_cast<const OpEncoding *>(inst), '
                f'{gather})'
            )
        if opnd_name == 'vaddr':
            fixed_words = self.isa_spec.profile.gfx11_mimg_fixed_vaddr_words.get(
                inst_name
            )
            if fixed_words is not None:
                default_words, a16_words = fixed_words
                return (
                    '(reinterpret_cast<const OpEncoding *>(inst)->a16 ? '
                    f'{a16_words * 32} : {default_words * 32})'
                )
            return (
                'mimg_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst), '
                f'"{inst_name.lower()}")'
            )
        return None

    @staticmethod
    def _emit_buffer_vaddr_helpers(
        helper_name: str, machine_inst_type: str, *, templated: bool
    ) -> str:
        template_decl = (
            'template <typename BufferMachineInst>\n            ' if templated else ''
        )
        return textwrap.dedent(f'''\
            namespace {{
            {template_decl}uint32_t {helper_name}(const {machine_inst_type} *inst) {{
              if (inst->idxen && inst->offen)
                return 64;
              if (inst->idxen || inst->offen)
                return 32;
              return 0;
            }}
            }} // namespace''')

    @staticmethod
    def _emit_vflat_helpers() -> str:
        return textwrap.dedent('''\
            namespace {
            template <typename VmemMachineInst>
            uint32_t vflat_vaddr_bits(const VmemMachineInst *inst) {
              // SADDR == NULL selects a 64-bit vector address; otherwise VADDR is a 32-bit offset.
              return inst->saddr == OPR_SREG_NULL ? 64 : 32;
            }
            } // namespace''')

    @staticmethod
    def _emit_gfx11_mimg_helpers() -> str:
        return textwrap.dedent('''\
            namespace {
            template <typename MimgMachineInst>
            uint32_t mimg_vdata_bits(const MimgMachineInst *inst, bool gather4) {
              uint32_t words = gather4 ? 4u : 0u;
              if (!gather4) {
                uint32_t mask = inst->dmask & 0xfu;
                while (mask) {
                  words += mask & 1u;
                  mask >>= 1;
                }
                if (words == 0)
                  words = 1;
              }
              if (inst->d16)
                words = (words + 1) / 2;
              if (inst->tfe)
                ++words;
              return words * 32;
            }

            bool mimg_name_has_token(std::string_view name, std::string_view token) {
              size_t pos = 0;
              while (pos < name.size()) {
                const size_t end = name.find('_', pos);
                const size_t count = end == std::string_view::npos ? name.size() - pos : end - pos;
                if (name.substr(pos, count) == token)
                  return true;
                if (end == std::string_view::npos)
                  break;
                pos = end + 1;
              }
              return false;
            }

            template <typename MimgMachineInst>
            uint32_t mimg_vaddr_bits(const MimgMachineInst *inst, std::string_view name) {
              static constexpr uint8_t coords[] = {1, 2, 3, 3, 2, 3, 3, 4};
              static constexpr uint8_t gradients[] = {2, 4, 6, 4, 2, 4, 4, 4};
              const uint32_t dim = inst->dim & 7u;
              const bool resinfo = name == "image_get_resinfo";
              const bool gradient = mimg_name_has_token(name, "d") ||
                                    mimg_name_has_token(name, "cd");
              const bool g16 = mimg_name_has_token(name, "g16");
              const bool lod = resinfo || mimg_name_has_token(name, "mip") ||
                               mimg_name_has_token(name, "l") ||
                               mimg_name_has_token(name, "cl");
              uint32_t words = mimg_name_has_token(name, "c") ? 1u : 0u;
              words += mimg_name_has_token(name, "o") ? 1u : 0u;
              words += mimg_name_has_token(name, "b") ? 1u : 0u;
              const uint32_t coord_words = (resinfo ? 0u : coords[dim]) + (lod ? 1u : 0u);
              words += inst->a16 ? (coord_words + 1) / 2 : coord_words;
              if (gradient) {
                const uint32_t gradient_words = gradients[dim];
                words += g16 ? ((gradient_words / 2 + 1) & ~1u) : gradient_words;
              }
              return (words == 0 ? 1u : words) * 32;
            }
            } // namespace''')

    @staticmethod
    def _emit_mfma_operand_helpers() -> str:
        return textwrap.dedent('''\
            namespace {
            uint32_t mfma_src2_encoding(uint32_t value, bool acc_cd) {
              constexpr uint32_t vgpr_min =
                  OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_VGPR_MIN;
              constexpr uint32_t vgpr_max =
                  OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_VGPR_MAX;
              constexpr uint32_t acc_min =
                  OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN;
              return value + (acc_cd && value >= vgpr_min && value <= vgpr_max
                                  ? acc_min - vgpr_min
                                  : 0);
            }
            } // namespace''')

    @staticmethod
    def _emit_cdna4_matrix_fmt_helpers() -> str:
        return textwrap.dedent('''\
            namespace {
            uint32_t cdna4_matrix_fmt_element_bits(uint32_t fmt) {
              switch (fmt) {
              case 2:
              case 3:
                return 6;
              case 4:
                return 4;
              default:
                return 8;
              }
            }

            int cdna4_matrix_fmt_operand_size_bits(uint32_t fmt, uint32_t dim, uint32_t k) {
              return static_cast<int>(dim * k * cdna4_matrix_fmt_element_bits(fmt) / 64);
            }
            } // namespace''')

    def _emit_cdna5_matrix_fmt_helpers(self) -> _ImplOutputs:
        """Emit C++ helpers for gfx1250 VOP3P packed and matrix quirks."""
        execution = textwrap.dedent('''\
            namespace {
            struct PkF32Words {
              uint32_t lo;
              uint32_t hi;
            };

            PkF32Words read_pk_f32_words(const Operand &operand, const amdgpu::Wavefront &wf, uint32_t lane) {
              const auto pair = amdgpu::RegisterAccess(wf).read_lane_pair32(operand, lane);
              return {pair.lo, pair.hi};
            }

            struct PkU64Pair {
              uint64_t lo;
              uint64_t hi;
            };

            struct PkU32Pair {
              uint32_t lo;
              uint32_t hi;
            };

            Operand packed_register_dword_offset(const Operand &operand, uint32_t dword_offset) {
              Operand shifted = operand;
              shifted.encoding_value_ += static_cast<int>(dword_offset);
              return shifted;
            }

            PkU64Pair read_pk_u64_pair(const Operand &operand, const amdgpu::Wavefront &wf,
                                       uint32_t lane) {
              const uint64_t lo = amdgpu::RegisterAccess(wf).read_lane64(operand, lane);
              const auto reg = operand.to_register_ref();
              if (!reg || reg->cls != RegClass::VGPR)
                return {lo, lo};

              const Operand hi_operand = packed_register_dword_offset(operand, 2);
              return {lo, amdgpu::RegisterAccess(wf).read_lane64(hi_operand, lane)};
            }

            PkU32Pair read_pk_u32_pair(const Operand &operand, const amdgpu::Wavefront &wf,
                                       uint32_t lane) {
              // GFX12+ single-SGPR-read operands read the first SGPR and replicate it.
              // VGPRs and 64-bit special registers such as VCC and EXEC remain pairs.
              const auto reg = operand.to_register_ref();
              if (reg && reg->cls == RegClass::SGPR) {
                const uint32_t value = amdgpu::RegisterAccess(wf).read_lane(operand, lane);
                return {value, value};
              }
              const auto pair = amdgpu::RegisterAccess(wf).read_lane_pair32(operand, lane);
              return {pair.lo, pair.hi};
            }

            void write_pk_u64_pair(const Operand &operand, amdgpu::Wavefront &wf, uint32_t lane,
                                   PkU64Pair value) {
              const Operand hi_operand = packed_register_dword_offset(operand, 2);
              amdgpu::RegisterAccess access(wf);
              access.write_lane64(operand, lane, value.lo);
              access.write_lane64(hi_operand, lane, value.hi);
            }
            ''')
        model = (
            'namespace {\n\n'
            + (
                'const char *cdna5_matrix_fmt_name(uint32_t fmt) {\n'
                '  switch (fmt) {\n'
                '  case 0:\n'
                '    return "MATRIX_FMT_FP8";\n'
                '  case 1:\n'
                '    return "MATRIX_FMT_BF8";\n'
                '  case 2:\n'
                '    return "MATRIX_FMT_FP6";\n'
                '  case 3:\n'
                '    return "MATRIX_FMT_BF6";\n'
                '  case 4:\n'
                '    return "MATRIX_FMT_FP4";\n'
                '  default:\n'
                '    return "MATRIX_FMT_INVALID";\n'
                '  }\n'
                '}\n'
                '\n'
                'const char *cdna5_matrix_scale_fmt_name(uint32_t fmt) {\n'
                '  switch (fmt) {\n'
                '  case 0:\n'
                '    return "MATRIX_SCALE_FMT_E8";\n'
                '  case 1:\n'
                '    return "MATRIX_SCALE_FMT_E5M3";\n'
                '  case 2:\n'
                '    return "MATRIX_SCALE_FMT_E4M3";\n'
                '  default:\n'
                '    return "MATRIX_SCALE_FMT_INVALID";\n'
                '  }\n'
                '}\n'
                '\n'
                'uint32_t cdna5_matrix_fmt_element_bits(uint32_t fmt) {\n'
                '  switch (fmt) {\n'
                '  case 2:\n'
                '  case 3:\n'
                '    return 6;\n'
                '  case 4:\n'
                '    return 4;\n'
                '  default:\n'
                '    return 8;\n'
                '  }\n'
                '}\n'
                '\n'
                'int cdna5_matrix_fmt_operand_size_bits(uint32_t fmt, uint32_t dim, uint32_t k) {\n'
                '  return static_cast<int>((dim * k * cdna5_matrix_fmt_element_bits(fmt)) / 32);\n'
                '}\n'
                '\n'
                'bool cdna5_scaled_wmma_is_scale16(const MachineInst *inst) {\n'
                '  return reinterpret_cast<const Vop3pMachineInst *>(inst)->op == 0x3a;\n'
                '}\n'
                '\n'
                'bool cdna5_scaled_wmma_is_f4_32x16x128(const MachineInst *inst) {\n'
                '  return reinterpret_cast<const Vop3pMachineInst *>(inst + 2)->op == 0x88;\n'
                '}\n'
                '\n'
                'const char *cdna5_scaled_wmma_mnemonic(const MachineInst *inst) {\n'
                '  if (cdna5_scaled_wmma_is_f4_32x16x128(inst))\n'
                '    return cdna5_scaled_wmma_is_scale16(inst) ? "v_wmma_scale16_f32_32x16x128_f4"\n'
                '                                               : "v_wmma_scale_f32_32x16x128_f4";\n'
                '  return cdna5_scaled_wmma_is_scale16(inst) ? "v_wmma_scale16_f32_16x16x128_f8f6f4"\n'
                '                                             : "v_wmma_scale_f32_16x16x128_f8f6f4";\n'
                '}\n'
                '\n'
                'int cdna5_scale_operand_size_bits(const MachineInst *inst, uint32_t selector) {\n'
                '  const bool is_vgpr = selector >= OpSelSrcSimple::OPR_SRC_SIMPLE_VGPR_MIN &&\n'
                '                       selector <= OpSelSrcSimple::OPR_SRC_SIMPLE_VGPR_MAX;\n'
                '  return cdna5_scaled_wmma_is_scale16(inst) && is_vgpr ? 64 : 32;\n'
                '}\n'
                '\n'
                'int cdna5_scaled_wmma_dst_size_bits(const MachineInst *inst) {\n'
                '  return cdna5_scaled_wmma_is_f4_32x16x128(inst) ? 512 : 256;\n'
                '}\n'
                '\n'
                'int cdna5_scaled_wmma_src0_size_bits(const MachineInst *inst) {\n'
                '  const auto *high = reinterpret_cast<const Vop3pMachineInst *>(inst + 2);\n'
                '  if (cdna5_scaled_wmma_is_f4_32x16x128(inst))\n'
                '    return 512;\n'
                '  return cdna5_matrix_fmt_operand_size_bits(high->opsel, 16, 128);\n'
                '}\n'
                '\n'
                'int cdna5_scaled_wmma_src1_size_bits(const MachineInst *inst) {\n'
                '  const auto *high = reinterpret_cast<const Vop3pMachineInst *>(inst + 2);\n'
                '  if (cdna5_scaled_wmma_is_f4_32x16x128(inst))\n'
                '    return 256;\n'
                '  return cdna5_matrix_fmt_operand_size_bits((high->@OPSEL_HI_2@ << 2) | high->opsel_hi, 16, 128);\n'
                '}\n'
            )
            + '\n} // namespace'
        ).replace('@OPSEL_HI_2@', self._op_sel_hi_2_field('ENC_VOP3P'))
        execution += '\n' + (
            'uint16_t read_fma_mix_f16_bits(uint32_t raw, uint32_t src_selector, bool high_half) {\n'
            '  switch (src_selector) {\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_HALF:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_HALF:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_ONE:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_ONE:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_TWO:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_TWO:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_FOUR:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_FOUR:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_ONE_OVER_TWO_PI: {\n'
            '    float value = std::bit_cast<float>(raw);\n'
            '    return util::f32_to_f16(value);\n'
            '  }\n'
            '  default: {\n'
            '    return static_cast<uint16_t>(high_half ? (raw >> 16) : raw);\n'
            '  }\n'
            '  }\n'
            '}\n'
            '\n'
            'uint16_t read_fma_mix_bf16_bits(uint32_t raw, uint32_t src_selector, bool high_half) {\n'
            '  switch (src_selector) {\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_HALF:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_HALF:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_ONE:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_ONE:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_TWO:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_TWO:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_FOUR:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_NEG_FOUR:\n'
            '  case OpSelSrc::OPR_SRC_FLOAT_ONE_OVER_TWO_PI: {\n'
            '    float value = std::bit_cast<float>(raw);\n'
            '    return util::f32_to_bf16(value);\n'
            '  }\n'
            '  default: {\n'
            '    return static_cast<uint16_t>(high_half ? (raw >> 16) : raw);\n'
            '  }\n'
            '  }\n'
            '}\n'
            '\n'
            'float read_fma_mix_source_f32(const Operand &src, const amdgpu::Wavefront &wf, uint32_t lane,\n'
            '                              uint32_t src_selector, bool src_is_f16, bool high_half) {\n'
            '  uint32_t raw = amdgpu::RegisterAccess(wf).read_lane(src, lane);\n'
            '  if (!src_is_f16)\n'
            '    return std::bit_cast<float>(raw);\n'
            '  return util::f16_to_f32(read_fma_mix_f16_bits(raw, src_selector, high_half));\n'
            '}\n'
            '\n'
            'float read_fma_mix_bf16_source_f32(const Operand &src, const amdgpu::Wavefront &wf, uint32_t lane,\n'
            '                                   uint32_t src_selector, bool src_is_bf16, bool high_half) {\n'
            '  uint32_t raw = amdgpu::RegisterAccess(wf).read_lane(src, lane);\n'
            '  if (!src_is_bf16)\n'
            '    return std::bit_cast<float>(raw);\n'
            '  return util::bf16_to_f32(read_fma_mix_bf16_bits(raw, src_selector, high_half));\n'
            '}\n'
            '} // namespace'
        )
        return _ImplOutputs(model=[model], execution=[execution])

    @staticmethod
    def _emit_cdna5_scaled_wmma_vop3px2_class() -> str:
        return textwrap.dedent('''\
            class VWmmaScaleF32Vop3px2 : public Vop3p {
            public:
              VWmmaScaleF32Vop3px2(const MachineInst *inst);
              void execute_impl(amdgpu::Wavefront &wf);
              void build_modifiers(std::string &out) const override;

              Operand vdst;
              Operand src0;
              Operand src1;
              Operand src2;
              Operand scale_src0;
              Operand scale_src1;
              OpEncoding scale_inst_;
              std::array<uint32_t, 4> raw_words_{};
            };
            ''')

    def _emit_cdna5_scaled_wmma_vop3px2_impls(self) -> _ImplOutputs:
        exec_fn = self._split_execute_expr('VWmmaScaleF32Vop3px2')
        model = (
            textwrap.dedent('''\
            VWmmaScaleF32Vop3px2::VWmmaScaleF32Vop3px2(const MachineInst *inst)
                : Vop3p(cdna5_scaled_wmma_mnemonic(inst), reinterpret_cast<const OpEncoding *>(inst + 2),
                        @EXEC_FN@, Vop3p::ExtensionDecodePolicy::Skip),
                  vdst(cdna5_scaled_wmma_dst_size_bits(inst), OperandType::OPR_VGPR,
                       reinterpret_cast<const OpEncoding *>(inst + 2)->vdst),
                  src0(cdna5_scaled_wmma_src0_size_bits(inst), OperandType::OPR_SRC_VGPR,
                       reinterpret_cast<const OpEncoding *>(inst + 2)->src0),
                  src1(cdna5_scaled_wmma_src1_size_bits(inst), OperandType::OPR_SRC_VGPR,
                       reinterpret_cast<const OpEncoding *>(inst + 2)->src1),
                  src2(cdna5_scaled_wmma_dst_size_bits(inst), OperandType::OPR_SRC_VGPR_OR_INLINE,
                       reinterpret_cast<const OpEncoding *>(inst + 2)->src2),
                  scale_src0(cdna5_scale_operand_size_bits(
                                 inst, reinterpret_cast<const OpEncoding *>(inst)->src0),
                             OperandType::OPR_SRC_SIMPLE,
                             reinterpret_cast<const OpEncoding *>(inst)->src0),
                  scale_src1(cdna5_scale_operand_size_bits(
                                 inst, reinterpret_cast<const OpEncoding *>(inst)->src1),
                             OperandType::OPR_SRC_SIMPLE,
                             reinterpret_cast<const OpEncoding *>(inst)->src1),
                  scale_inst_(*reinterpret_cast<const OpEncoding *>(inst)) {
              raw_words_ = {inst[0], inst[1], inst[2], inst[3]};
              raw_encoding_ = raw_words_.data();
              encoding_id_ = raw_encoding_[0] >> 23;
              opcode_ = scale_inst_.op;
              size_ = 4 * sizeof(MachineInst);

              dst_operands_[0] = &vdst;
              src_operands_[0] = &src0;
              src_operands_[1] = &src1;
              src_operands_[2] = &src2;
              src_operands_[3] = &scale_src0;
              src_operands_[4] = &scale_src1;
              num_src_ = 5;
              num_dst_ = 1;
              vdst.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);
              src0.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
              src1.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);
              src2.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);
            }

            void VWmmaScaleF32Vop3px2::build_modifiers(std::string &out) const {
              if (inst_.op != 0x88) {
                const uint32_t matrix_a_fmt = inst_.opsel;
                const uint32_t matrix_b_fmt = (inst_.@OPSEL_HI_2@ << 2) | inst_.opsel_hi;
                if (matrix_a_fmt != 0) {
                  out += " matrix_a_fmt:";
                  out += cdna5_matrix_fmt_name(matrix_a_fmt);
                }
                if (matrix_b_fmt != 0) {
                  out += " matrix_b_fmt:";
                  out += cdna5_matrix_fmt_name(matrix_b_fmt);
                }
              }
              if (scale_inst_.opsel & 0x1u)
                out += " matrix_a_scale:MATRIX_SCALE_ROW1";
              if (scale_inst_.opsel_hi & 0x1u)
                out += " matrix_b_scale:MATRIX_SCALE_ROW1";
              const uint32_t matrix_a_scale_fmt = scale_inst_.neg & 0x3u;
              const uint32_t matrix_b_scale_fmt = scale_inst_.neg_hi & 0x3u;
              if (matrix_a_scale_fmt != 0) {
                out += " matrix_a_scale_fmt:";
                out += cdna5_matrix_scale_fmt_name(matrix_a_scale_fmt);
              }
              if (matrix_b_scale_fmt != 0) {
                out += " matrix_b_scale_fmt:";
                out += cdna5_matrix_scale_fmt_name(matrix_b_scale_fmt);
              }
              if ((scale_inst_.opsel >> 2) & 0x1u)
                out += " matrix_a_reuse";
              if (scale_inst_.@OPSEL_HI_2@)
                out += " matrix_b_reuse";
            }
            ''')
            .replace('@EXEC_FN@', exec_fn)
            .replace('@OPSEL_HI_2@', self._op_sel_hi_2_field('ENC_VOP3P'))
        )

        execution = textwrap.dedent('''\
            void VWmmaScaleF32Vop3px2::execute_impl(amdgpu::Wavefront &wf) {
              auto &cu = wf.cu();
              uint32_t vb = wf.vgpr_alloc().base;
              uint32_t dst = vb + *Isa::resolved_vgpr_offset(wf, vdst.opr_type_, vdst.encoding_value_,
                                                             vdst.vgpr_msb_role());
              uint32_t src0_base = vb + *Isa::resolved_vgpr_offset(wf, src0.opr_type_, src0.encoding_value_,
                                                                   src0.vgpr_msb_role());
              uint32_t src1_base = vb + *Isa::resolved_vgpr_offset(wf, src1.opr_type_, src1.encoding_value_,
                                                                   src1.vgpr_msb_role());
              uint32_t const_acc;
              auto src2_off =
                  Isa::resolved_vgpr_offset(wf, src2.opr_type_, src2.encoding_value_, src2.vgpr_msb_role());
              uint32_t s2 = dst;
              if (src2_off) {
                const_acc = amdgpu::ACC_FROM_VGPR;
                s2 = vb + *src2_off;
              } else {
                const_acc = amdgpu::RegisterAccess(wf).read_scalar(src2);
              }

              const uint32_t matrix_a_fmt = inst_.opsel;
              const uint32_t matrix_b_fmt = (inst_.@OPSEL_HI_2@ << 2) | inst_.opsel_hi;
              const uint32_t matrix_a_scale =
                  (scale_inst_.opsel & 0x1u) | (((scale_inst_.opsel >> 2u) & 0x1u) << 1u);
              const uint32_t matrix_b_scale =
                  (scale_inst_.opsel_hi & 0x1u) | ((scale_inst_.@OPSEL_HI_2@ & 0x1u) << 1u);
              const uint32_t matrix_a_scale_fmt = scale_inst_.neg & 0x3u;
              const uint32_t matrix_b_scale_fmt = scale_inst_.neg_hi & 0x3u;
              const bool scale16 = scale_inst_.op == 0x3a;
              const bool scale0_inline_zero =
                  scale_src0.encoding_value() == OpSelSrcSimple::OPR_SRC_SIMPLE_POS_INT_MIN;
              const bool scale1_inline_zero =
                  scale_src1.encoding_value() == OpSelSrcSimple::OPR_SRC_SIMPLE_POS_INT_MIN;

              auto scale_word = [&](const Operand &operand, uint32_t lane) -> uint64_t {
                // Inline zero supplies a neutral E8M0 scale for every K block.
                if (operand.encoding_value() == OpSelSrcSimple::OPR_SRC_SIMPLE_POS_INT_MIN)
                  return 0x7f7f7f7f7f7f7f7full;
                // Scalar scale sources use only bits 7:0, repeated for every K
                // block. VGPR scale sources retain their packed per-block bytes.
                if (operand.encoding_value() >= 0 && operand.encoding_value() <= 105) {
                  const uint64_t byte = amdgpu::RegisterAccess(wf).read_lane(operand, lane) & 0xffu;
                  return byte * 0x0101010101010101ull;
                }
                return scale16 ? amdgpu::RegisterAccess(wf).read_lane64(operand, lane)
                               : amdgpu::RegisterAccess(wf).read_lane(operand, lane);
              };
              auto scale0 = [&](uint32_t lane) -> uint64_t {
                return scale_word(scale_src0, lane);
              };
              auto scale1 = [&](uint32_t lane) -> uint64_t {
                return scale_word(scale_src1, lane);
              };

              bool dispatched = false;
              if (inst_.op == 0x88) {
                amdgpu::exec_wmma_f32_scaled_mixed(cu, 32, 16, 128, 4, 4, dst, src0_base,
                                                   src1_base, s2, amdgpu::extract_fp4,
                                                   amdgpu::extract_fp4, const_acc, scale0, scale1,
                                                   matrix_a_scale, matrix_b_scale,
                                                   scale0_inline_zero ? 0u : matrix_a_scale_fmt,
                                                   scale1_inline_zero ? 0u : matrix_b_scale_fmt, scale16,
                                                   amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));
                dispatched = true;
              } else {
                dispatched = amdgpu::dispatch_matrix_fmt_pair(
                    matrix_a_fmt, matrix_b_fmt,
                    [&](uint32_t a_bits, uint32_t b_bits, auto extract_a, auto extract_b) {
                      amdgpu::exec_wmma_f32_scaled_mixed(
                          cu, 16, 16, 128, a_bits, b_bits, dst, src0_base, src1_base, s2,
                          extract_a, extract_b, const_acc, scale0, scale1, matrix_a_scale,
                          matrix_b_scale, scale0_inline_zero ? 0u : matrix_a_scale_fmt,
                          scale1_inline_zero ? 0u : matrix_b_scale_fmt, scale16,
                          amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));
                    });
              }
              if (!dispatched)
                throw util::UnimplementedInst(mnemonic());
            }
            ''').replace('@OPSEL_HI_2@', self._op_sel_hi_2_field('ENC_VOP3P'))
        return _ImplOutputs(model=[model], execution=[execution])

    def _emit_cdna5_scaled_wmma_vop3px2_decoder_helpers(self) -> str:
        return textwrap.dedent('''\
            namespace {

            bool isVop3pOp(const MachineInst opcode, uint32_t op) {
              return (opcode >> 24) == 0xcc && ((opcode >> 16) & 0xff) == op;
            }

            bool isGfx1250WmmaScaleSource(uint32_t selector, bool scale16) {
              if (selector <= 105u || selector == 128u)
                return true;
              if (selector < 256u || selector > 511u)
                return false;
              return !scale16 || ((selector - 256u) % 2u == 0u && selector < 511u);
            }

            bool isGfx1250WmmaScaleFormatPairLegal(uint32_t matrix_a_fmt, uint32_t matrix_b_fmt,
                                                   uint32_t scale_a_fmt, uint32_t scale_b_fmt) {
              if (matrix_a_fmt > 4u || matrix_b_fmt > 4u || scale_a_fmt > 2u || scale_b_fmt > 2u)
                return false;
              if (scale_a_fmt == 0u && scale_b_fmt == 0u)
                return true;
              if ((scale_a_fmt != 0u && matrix_a_fmt != 4u) ||
                  (scale_b_fmt != 0u && matrix_b_fmt != 4u))
                return false;
              return matrix_a_fmt != 4u || matrix_b_fmt != 4u || scale_a_fmt == scale_b_fmt;
            }

            bool isGfx1250WmmaScalePairValid(const MachineInst *opcode) {
              const auto *scale = reinterpret_cast<const Vop3pMachineInst *>(opcode);
              const auto *matrix = reinterpret_cast<const Vop3pMachineInst *>(opcode + 2);
              const bool scale16 = scale->op == 0x3au;
              // Accept LLVM's encoding as an alias for the ISA-canonical constant.
              const bool fixed_src2_valid = scale->src2 == 0x080u || scale->src2 == 0x100u;
              if (scale->vdst != 0u || (scale->neg_hi & 0x4u) != 0u ||
                  (scale->opsel & 0x2u) != 0u || scale->clamp != 0u || !fixed_src2_valid ||
                  (scale->opsel_hi & 0x2u) != 0u || (scale->neg & 0x4u) != 0u ||
                  (matrix->neg_hi & 0x3u) != 0u || matrix->clamp != 0u ||
                  (matrix->neg & 0x3u) != 0u)
                return false;
              if (!isGfx1250WmmaScaleSource(scale->src0, scale16) ||
                  !isGfx1250WmmaScaleSource(scale->src1, scale16))
                return false;

              const uint32_t scale_a_fmt = scale->neg & 0x3u;
              const uint32_t scale_b_fmt = scale->neg_hi & 0x3u;
              if (matrix->op == 0x88u)
                return isGfx1250WmmaScaleFormatPairLegal(4u, 4u, scale_a_fmt, scale_b_fmt);
              const uint32_t matrix_a_fmt = matrix->opsel;
              const uint32_t matrix_b_fmt = (matrix->@OPSEL_HI_2@ << 2u) | matrix->opsel_hi;
              return isGfx1250WmmaScaleFormatPairLegal(matrix_a_fmt, matrix_b_fmt, scale_a_fmt,
                                                       scale_b_fmt);
            }

            } // namespace
            ''').replace('@OPSEL_HI_2@', self._op_sel_hi_2_field('ENC_VOP3P'))

    def _execute_operand_roles(
        self, inst: Instruction, sem: InstructionSemantics
    ) -> tuple[list[Operand], list[Operand]]:
        # TODO: Incorporate fieldless operand side effects into execute bodies.
        # Execute bodies reference operands positionally (src_ops[i]/dst_ops[i]).
        # Fieldless side-effect operands (VCC/EXEC/SCC/...) are excluded here
        # so these indices match the field-bearing operand set exactly as
        # before. Fieldless OPR_SIMM32 is value-bearing, so it stays visible.
        visible_operands = [
            op for op in inst.operands if self._execute_operand_participates(op)
        ]
        dst_operands = [op for op in visible_operands if not op.is_input]
        src_operands = [op for op in visible_operands if op.is_input]
        # Some instructions mark their destination as input (read-modify-write,
        # e.g. S_BITSET0, S_CMOV, V_FMAC, V_SWAP). Recover the destination
        # from src_ops when it looks like one.
        if (
            not dst_operands
            and src_operands
            and src_operands[0].name in ('sdst', 'vdst')
        ):
            dst_operands = [src_operands[0]]
            src_operands = src_operands[1:]
        # Some ISA specs mark swap operands as output-only even though the
        # instruction reads both. Treat the second output as a source.
        if (
            not src_operands
            and len(dst_operands) >= 2
            and sem.semantic_class in ('vector_swap', 'vector_swaprel')
        ):
            src_operands = dst_operands[1:]
            dst_operands = dst_operands[:1]
        src_operands = self._semantic_source_operands(inst, src_operands)
        return src_operands, dst_operands

    def _true16_vop3_info(
        self,
        inst: Instruction,
        sem: InstructionSemantics,
        enc_name: str = '',
        *,
        src_operands: list[Operand] | None = None,
        dst_operands: list[Operand] | None = None,
        is_vop3: bool | None = None,
    ) -> _True16Vop3Info:
        profile = self.isa_spec.profile
        if src_operands is None or dst_operands is None:
            src_operands, dst_operands = self._execute_operand_roles(inst, sem)
        if is_vop3 is None:
            is_vop3 = profile.has_src_modifiers(enc_name)

        cls = sem.semantic_class
        dtype = sem.data_type
        force_value = cls in (
            'vector_binop',
            'vector_ternary',
            'vector_unary',
        ) and dtype in (
            'b16',
            'f16',
            'i16',
            'u16',
        )
        force_div_fixup = cls == 'vector_div_fixup' and dtype == 'f16'
        force_cmp = cls == 'vector_cmp' and dtype in ('i16', 'u16')
        force_cmp_class = cls in ('vector_cmp_class', 'vector_cmpx_class') and (
            dtype == 'f16'
        )
        force_cvt_pk_src = (
            cls == 'vector_pack_b32_f16'
            or (cls in ('vector_cvt_pknorm', 'vector_cvt_pk') and dtype == 'f16')
            or (cls == 'vector_cvt_pk' and inst.name.endswith('_F16'))
        )
        force_src = force_value or force_div_fixup or force_cmp or force_cmp_class
        has_src = any(
            opnd.is_input and (opnd.size == 16 or force_src) for opnd in src_operands
        )
        has_dst = any(
            opnd.is_output and (opnd.size == 16 or force_value or force_div_fixup)
            for opnd in dst_operands
        )
        uses_true16_vop3 = bool(getattr(profile, 'uses_true16_vop3_opsel', False))
        ignores_true16_opsel = cls == 'pseudo_scalar_unary' and dtype == 'f16'
        enabled = (
            uses_true16_vop3
            and bool(is_vop3)
            and bool(dst_operands)
            and (has_src or has_dst)
            and not ignores_true16_opsel
        )
        # A few VOP3 f16 special bodies must still use true16 OP_SEL selection
        # even though their semantic operand model is not a plain 16-bit
        # value op. Keep that body classification separate from the profile
        # gate above so the generated scalar body and shared SIMD probe use
        # the same true16 policy.
        body_uses_true16 = enabled or (
            bool(is_vop3) and (force_div_fixup or force_cmp_class or force_cvt_pk_src)
        )
        return _True16Vop3Info(
            force_src=force_src,
            has_dst=has_dst,
            body_uses_true16=body_uses_true16,
            enabled=enabled,
        )

    # The trap-handler control ops are spelled differently per ISA. Every
    # spelling has to be listed: an omitted one derives as `true_nop` and its
    # generated execute_impl() comes out an empty body, which leaves that ISA's
    # trap handlers unable to return at all. GFX1250 spells the return
    # S_RFE_I64 (SOP1 opcode 74).
    _TRAP_RETURN_NAMES = ('S_RFE', 'S_RFE_B64', 'S_RFE_I64')
    _TRAP_SENDMSG_NAMES = ('S_SENDMSG', 'S_SENDMSGHALT')

    def _sleep_body(self, sem: InstructionSemantics) -> str:
        """execute() body for S_SLEEP / S_SLEEP_VAR.

        The MR ISA gives these no pseudocode, so they derive as `true_nop` and
        the generator used to emit the yield alone. The delay *is* the whole
        instruction -- there is no result register -- so retiring it in one step
        leaves it with no effect and lets a sleep loop spin at the speed of its
        own scalar code. That is not only a performance detail: an asynchronous
        debugger suspend then lands uniformly across the loop body instead of
        overwhelmingly on the sleep, and -O0 loop bodies are full of short
        windows where the compiler has forced EXEC to all lanes to spill an
        AGPR. Stopping inside one reports every lane active, which
        gdb.rocm/lane-info.exp catches by comparing the stopped lane states
        against the ones it recorded at a breakpoint. That expect test is the
        only guard -- the C++ ISA harness lists s_sleep in SKIP_PREFIXES -- so
        the policy belongs here, where a regeneration cannot drop it.
        """
        # S_SLEEP idles the wave for 64 * SIMM16[6:0] clocks; S_SLEEP_VAR takes
        # the same 7-bit count from a scalar operand instead of the literal.
        count = (
            'static_cast<uint32_t>(simm16.encoding_value_)'
            if sem.name == 'S_SLEEP'
            else 'amdgpu::RegisterAccess(wf).read_scalar(ssrc0)'
        )
        return (
            '  constexpr uint32_t kSleepClocksPerUnit = 64;\n'
            f'  wf.set_sleep_cycles(kSleepClocksPerUnit * ({count} & 0x7Fu));\n'
            '  wf.cu().request_functional_yield();'
        )

    def _trap_control_body(self, sem: InstructionSemantics) -> str | None:
        """execute() body for a trap-handler control op, or None if not one.

        The MR ISA carries no pseudocode for these, so they derive as
        `true_nop`. They are not nops: the configured GPU trap handler returns
        through S_RFE and reports through S_SENDMSG, and leaving them empty
        silently disables ROCgdb's whole stop/resume path (see
        docs/rocgdb-debugging.md). The bodies belong here rather than in a
        hand-edit of the generated header, which a regeneration would drop.
        """
        if sem.name in self._TRAP_RETURN_NAMES:
            # Return from exception: restore the PC the trap handler saved in
            # ssrc0. gfx12.5 has a 57-bit PC; older targets use 48 address bits.
            # The instruction size is subtracted because the interpreter
            # advances the PC after execute() returns.
            pc_mask = (
                '0x01FFFFFFFFFFFFFFULL'
                if sem.name == 'S_RFE_I64'
                else '0x0000FFFFFFFFFFFFULL'
            )
            split_state_status_restore = (
                '  // GFX12 returns the interrupted wave state while preserving the handler\'s\n'
                '  // STATE_PRIV.HALT decision. Older layouts keep using the live STATUS word.\n'
                '  if (wf.in_trap_handler() && wf.uses_separate_trap_ctrl()) {\n'
                '    uint32_t restored_status = wf.trap_saved_status();\n'
                '    restored_status = keep_halted ? (restored_status | kStatusHalt)\n'
                '                                   : (restored_status & ~kStatusHalt);\n'
                '    wf.set_status_raw(restored_status);\n'
                '  }\n'
            )
            return (
                # Bare operand and size_ spellings: that is the arch-local form
                # every generated execute_impl() uses, and
                # _write_shared_execute_templates() lifts them to inst.ssrc0 /
                # inst.size() for the shared template. Writing the lifted form
                # here instead compiles only on the ISAs that happen to share
                # the body -- S_RFE_I64 is gfx1250-only, so it does not.
                '  uint64_t saved_pc = amdgpu::RegisterAccess(wf).read_scalar64(ssrc0);\n'
                f'  constexpr uint64_t kPcAddressMask = {pc_mask};\n'
                '  wf.pc = (saved_pc & kPcAddressMask) - size_;\n'
                '\n'
                '  constexpr uint32_t kStatusHalt = 1u << 13;\n'
                '  const bool keep_halted = (wf.status_raw() & kStatusHalt) != 0;\n'
                f'{split_state_status_restore}'
                '\n'
                '  // Returning from the handler puts the interrupted EXEC back. The handler runs\n'
                '  // under its own mask -- it parks a doorbell id in EXEC_LO on the way to\n'
                '  // MSG_INTERRUPT -- and restoring that is part of returning, not part of\n'
                '  // stopping for a debugger: a handler that returns without stopping the wave\n'
                '  // used to leave its mask installed, so the application ran on with every lane\n'
                '  // active. That silently un-diverges a branch (gdb.rocm/lane-info.exp sees\n'
                '  // lanes that converged out of a branch reported active again) and is\n'
                '  // permanent, because nothing later puts the application\'s mask back.\n'
                '  if (wf.in_trap_handler())\n'
                '    wf.set_exec(wf.trap_saved_exec());\n'
                '  wf.set_in_trap_handler(false);\n'
                '\n'
                '  // The handler sets STATUS.HALT when it wants the wave to stay\n'
                '  // stopped for the debugger; honour that on the way out.\n'
                '  if (keep_halted) {\n'
                '    wf.set_debug_single_step(false);\n'
                '    wf.set_debug_halted(true);\n'
                '  } else {\n'
                '    // Nothing is halting the wave any more, so an s_sendmsghalt marker\n'
                '    // left over from an earlier stop is stale. Leaving it set would make\n'
                '    // the next resume clear a HALT the handler raises later.\n'
                '    wf.set_self_halted(false);\n'
                '  }'
            )

        if sem.name in self._TRAP_SENDMSG_NAMES:
            # MSG_INTERRUPT (id 1) from inside the trap handler is how the wave
            # tells KFD it has stopped; the CU turns it into a debug event.
            body = (
                '  const uint32_t message = static_cast<uint32_t>(simm16.encoding_value_);\n'
                '  if (wf.in_trap_handler() && (message & 0xFu) == 1u)\n'
                '    wf.set_trap_interrupt_sent(true);\n'
                '  wf.cu().handle_sendmsg(wf, message);'
            )
            if sem.name == 'S_SENDMSGHALT':
                # "...and then HALT the wavefront". The halt is architectural,
                # so publish it in STATUS.HALT and not only in the debugger's
                # private flag: the s_rfe path above reads STATUS.HALT to decide
                # whether the wave stays stopped, and saw 0 for a wave this
                # instruction had already halted. Both halves now agree, and a
                # debugger reading STATUS sees the same thing the wave does.
                body += (
                    '\n'
                    '\n'
                    '  // S_SENDMSGHALT halts the wave. Keep the architectural bit and the\n'
                    '  // scheduler flag in step -- s_rfe consults STATUS.HALT on the way out.\n'
                    '  constexpr uint32_t kStatusHalt = 1u << 13;\n'
                    '  wf.set_status_raw(wf.status_raw() | kStatusHalt);\n'
                    '  wf.set_debug_halted(true);\n'
                    '  // Remember who raised the bit. The trap handler also raises HALT, via\n'
                    '  // s_setreg just before it returns, and there it means "keep the wave\n'
                    '  // stopped" -- the opposite of what it means here, where the wave has\n'
                    '  // already reported and is waiting to be resumed. The CWSR record cannot\n'
                    '  // distinguish the two, so the resume path reads this instead.\n'
                    '  wf.set_self_halted(true);'
                )
            return body

    @staticmethod
    def _mask_result_kind(
        inst: Instruction, sem: InstructionSemantics, enc_name: str
    ) -> _MaskResultKind | None:
        """Return the architectural kind of a structured lane-mask result."""
        if (
            sem.semantic_class in ('vector_cmp', 'vector_cmp_class')
            and enc_name.upper() != 'ENC_VOPC'
        ):
            return _MaskResultKind.COMPARE
        if (
            sem.semantic_class in ('vector_cmpx', 'vector_cmpx_class')
            and enc_name.upper() != 'ENC_VOPC'
        ):
            return _MaskResultKind.EXEC
        if inst.enc_name.upper() == 'VOP3_SDST_ENC' and any(
            op.is_output and op.name == 'sdst' for op in inst.operands
        ):
            return _MaskResultKind.SECONDARY
        if sem.semantic_class == 'vector_add_co' and enc_name.upper() == 'ENC_VOP2':
            return _MaskResultKind.IMPLICIT_VCC
        return None

    def _gen_execute_body(
        self,
        inst: Instruction,
        sem: InstructionSemantics,
        enc_name: str = '',
        result_writer: str | None = None,
    ) -> str:
        """Generate execute() body from instruction semantics."""
        src_operands, dst_operands = self._execute_operand_roles(inst, sem)
        dst_ops = [op.name for op in dst_operands]
        src_ops = [op.name for op in src_operands]
        cls = sem.semantic_class
        op = sem.operation
        dtype = sem.data_type
        scc = sem.sets_scc
        cond = sem.branch_condition
        profile = self.isa_spec.profile
        is_vop3 = profile.has_src_modifiers(enc_name)
        # Use inst.enc_name (not enc_name) because the instruction's encoding
        # may be a sub-format (e.g. VOP3_SDST_ENC) that differs from the
        # parent encoding (ENC_VOP3). VOP3_SDST_ENC has neg/omod/clamp but
        # no abs modifier field.
        has_abs = profile.has_abs_modifier(inst.enc_name)
        self._enc_name = enc_name

        # Try SemaAST pipeline for validated classes.
        from amdisa.sema_derive import derive_sema_block
        from amdisa.codegen.execute.sema_lower import (
            lower_sema_block,
            LoweringContext,
            OperandMap,
            RegClass,
        )

        def _semantic_reg_class(opnd: Operand) -> RegClass:
            if opnd.operand_type.startswith('OPR_ACC'):
                return RegClass.ACC_VGPR
            if opnd.operand_type.startswith('OPR_SRC_ACC'):
                return RegClass.ACC_VGPR
            scalar_operand_types = (
                'OPR_EXEC',
                'OPR_PC',
                'OPR_SDST',
                'OPR_SGPR',
                'OPR_SMEM_OFFSET',
                'OPR_SREG',
                'OPR_SSRC',
                'OPR_VCC',
            )
            if opnd.operand_type.startswith(scalar_operand_types):
                return RegClass.SGPR
            return RegClass.VGPR

        _SEMA_CLASSES = frozenset(
            {
                'scalar_mov',
                'scalar_cmov',
                'scalar_cselect',
                'scalar_cmp',
                'scalar_unary',
                'scalar_binop',
                'scalar_bitcmp',
                'scalar_cvt_pkrtz_f16_f32',
                'scalar_saveexec',
                'scalar_wrexec',
                'scalar_addk',
                'scalar_bfe',
                'pseudo_scalar_unary',
                'vector_swap',
                'vector_mov',
                'vector_binop',
                'vector_ternary',
                'vector_unary',
                'vector_cmp',
                'vector_cndmask',
                'vector_add_co',
            }
        )
        if cls in _SEMA_CLASSES:
            sema_block = derive_sema_block(sem)
            if sema_block is not None and not sema_block.is_empty:
                profile = self.isa_spec.profile
                uses_true16_e32 = bool(
                    getattr(profile, 'uses_packed_16bit_e32_source_selectors', False)
                )
                is_true16_mov = (
                    uses_true16_e32
                    and inst.name == 'V_MOV_B16'
                    and cls == 'vector_mov'
                    and dtype in ('b16', 'u16')
                )
                is_float_op = dtype in ('f16', 'f32', 'f64', 'bf16')
                if (
                    is_vop3
                    and is_float_op
                    and not is_true16_mov
                    and cls != 'pseudo_scalar_unary'
                ):
                    from amdisa.sema_enrich import enrich_block

                    ef = {'neg'}
                    if has_abs:
                        ef.add('abs')
                    inst_fields = getattr(self, '_current_inst_fields', set())
                    if 'clamp' in inst_fields:
                        ef.add('clamp')
                    if 'omod' in inst_fields:
                        ef.add('omod')
                    sema_block = enrich_block(sema_block, enc_field_names=frozenset(ef))
                # Preserve 6470's scalar_saveexec -> b64 dtype fix. Per-operand
                # bit widths (op_widths) subsume the old src_width/dst_width name
                # heuristics: mixed-width instructions (e.g. the f64<->32-bit
                # conversions, frexp_*_f64) bind each operand to its own declared
                # lane width via op.size instead of a single instruction-level
                # dtype width.
                omap_dtype = 'b64' if cls == 'scalar_saveexec' else dtype
                op_widths = {op.name: op.size for op in inst.operands}
                src_reg_classes = {
                    i: _semantic_reg_class(opnd) for i, opnd in enumerate(src_operands)
                }
                dst_reg_classes = {
                    i: _semantic_reg_class(opnd) for i, opnd in enumerate(dst_operands)
                }
                omap = OperandMap.from_operand_names(
                    src_ops,
                    dst_ops,
                    sema_block.pragma,
                    omap_dtype,
                    op_widths,
                    src_reg_classes,
                    dst_reg_classes,
                )
                lctx = LoweringContext(
                    exec_model=sema_block.pragma,
                    operand_map=omap,
                    arch_name=self.isa_spec.arch_name,
                    # Generic generated scalar F16 arithmetic stays out of
                    # FP16_OVFL clamping. Explicit scalar F32->F16 converts are
                    # lowered through sema_lower's mode-aware conversion helper.
                    mode_sensitive_f16_dst=not cls.startswith('scalar_'),
                    mask_result_writer=result_writer,
                )
                if cls == 'vector_cmp':
                    # V_CMP writes a fresh wave mask initialized to zero, so false
                    # lanes can remain clear without emitting redundant bit clears.
                    lctx.clear_false_lane_mask_writes = False
                if is_vop3 and inst.name in (
                    'V_CVT_F32_FP8',
                    'V_CVT_F32_BF8',
                    'V_CVT_F16_FP8',
                    'V_CVT_F16_BF8',
                ):
                    lctx.fp8_byte_select = (
                        '((amdgpu::vop3_opsel(inst_) & 0x1u) << 1) | '
                        '((amdgpu::vop3_opsel(inst_) & 0x2u) >> 1)'
                    )
                if is_vop3 and inst.name in (
                    'V_CVT_F32_FP8',
                    'V_CVT_F16_FP8',
                ):
                    lctx.fp8_decode_e5m3_select = 'amdgpu::vop3_fp8_decode_e5m3(*this)'
                if (
                    sema_block.pragma.name == 'VECTOR'
                    and dst_ops
                    and all(
                        binding.reg_class == RegClass.SGPR
                        for binding in omap.src_bindings.values()
                    )
                    and all(
                        binding.reg_class == RegClass.SGPR
                        for binding in omap.dst_bindings.values()
                    )
                ):
                    lctx.vector_sgpr_once = True
                has_true16_src = any(
                    opnd.is_input and opnd.size == 16 for opnd in src_operands
                )
                has_true16_dst = any(
                    opnd.is_output and opnd.size == 16 for opnd in dst_operands
                )
                true16_vop3_info = self._true16_vop3_info(
                    inst,
                    sem,
                    enc_name,
                    src_operands=src_operands,
                    dst_operands=dst_operands,
                    is_vop3=is_vop3,
                )
                is_true16_vop3 = true16_vop3_info.enabled
                if is_true16_vop3:
                    lctx.vector_preamble.append(
                        '  [[maybe_unused]] uint32_t opsel = amdgpu::vop3_opsel(inst_);'
                    )
                    vop3_opsel = 'opsel'
                    lctx.true16_vop3_opsel = vop3_opsel
                    for src_idx, opnd in enumerate(src_operands):
                        if opnd.is_input and (
                            opnd.size == 16 or true16_vop3_info.force_src
                        ):
                            lctx.true16_src_selects[src_idx] = (
                                f'{vop3_opsel} & 0x{1 << src_idx:x}u'
                            )
                    if true16_vop3_info.has_dst:
                        lctx.true16_dst_select = f'{vop3_opsel} & 0x8u'
                    if inst.name in ('V_CVT_F16_FP8', 'V_CVT_F16_BF8'):
                        lctx.fp8_byte_select = f'({vop3_opsel} & 0x2u) >> 1'
                if cls == 'vector_cndmask' and is_vop3 and len(src_ops) >= 3:
                    selector_read = f'amdgpu::read_wave_mask_scalar({src_ops[2]}, wf)'
                    lctx.vcc_read = selector_read
                    if inst.name == 'V_CNDMASK_B32':
                        return (
                            '  uint64_t exec = wf.exec();\n'
                            '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                            '    if (!(exec & (1ULL << lane)))\n'
                            '      continue;\n'
                            f'    const uint32_t src0_value = amdgpu::apply_vop3_b32_src_mod(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane), inst_.abs, inst_.neg, 0);\n'
                            f'    const uint32_t src1_value = amdgpu::apply_vop3_b32_src_mod(amdgpu::RegisterAccess(wf).read_lane({src_ops[1]}, lane), inst_.abs, inst_.neg, 1);\n'
                            f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, (({selector_read} >> lane) & 1) ? src1_value : src0_value);\n'
                            '  }\n'
                        )
                if (
                    inst.name == 'V_CVT_F32_F16'
                    and is_true16_vop3
                    and src_ops
                    and dst_ops
                ):
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        '  const uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);\n'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'    uint32_t raw = ::rocjitsu::amdgpu::read_vop3_true16_src({src_ops[0]}, wf, lane, opsel, 0);\n'
                        '    float src = util::f16_to_f32(static_cast<uint16_t>(raw));\n'
                        '    if (inst_.abs & (1u << 0))\n'
                        '      src = std::fabs(src);\n'
                        '    if (inst_.neg & (1u << 0))\n'
                        '      src = -src;\n'
                        f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, std::bit_cast<uint32_t>(src));\n'
                        '  }\n'
                    )
                true16_special_vop3_ops = {
                    'V_ASHRREV_I16': (
                        2,
                        ('auto v = static_cast<int16_t>(s1);',),
                        'static_cast<uint32_t>(static_cast<uint16_t>('
                        'v >> (static_cast<int16_t>(s0) & 15u)))',
                    ),
                    'V_LSHLREV_B16': (
                        2,
                        (),
                        '(s1 << (s0 & 15u)) & 0xffffu',
                    ),
                    'V_LSHRREV_B16': (
                        2,
                        (),
                        's1 >> (s0 & 15u)',
                    ),
                    'V_MAD_I16': (
                        3,
                        (
                            'int32_t a = static_cast<int16_t>(s0);',
                            'int32_t b = static_cast<int16_t>(s1);',
                            'int32_t c = static_cast<int16_t>(s2);',
                        ),
                        'static_cast<uint32_t>(static_cast<uint16_t>(a * b + c))',
                    ),
                    'V_MAD_U16': (
                        3,
                        (),
                        '(s0 * s1 + s2) & 0xffffu',
                    ),
                }
                if is_true16_vop3 and inst.name in true16_special_vop3_ops:
                    src_count, setup, result_expr = true16_special_vop3_ops[inst.name]
                    src_lines = ''.join(
                        f'    uint32_t s{i} = ::rocjitsu::amdgpu::read_vop3_true16_src({src_ops[i]}, wf, lane, opsel, {i});\n'
                        for i in range(src_count)
                    )
                    setup_lines = ''.join(f'    {line}\n' for line in setup)
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        '  uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);\n'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'{src_lines}'
                        f'{setup_lines}'
                        f'    uint32_t result = {result_expr};\n'
                        f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst_ops[0]}, wf, lane, opsel, result, true);\n'
                        '  }\n'
                    )
                if (
                    cls == 'vector_ternary'
                    and op == 'fma'
                    and dtype == 'f16'
                    and is_vop3
                ):
                    if is_true16_vop3:
                        src_reads = [
                            f'::rocjitsu::amdgpu::read_vop3_true16_src({src_ops[i]}, wf, lane, opsel, {i})'
                            for i in range(3)
                        ]
                        preamble = '  uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);\n'
                        write_result = f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst_ops[0]}, wf, lane, opsel, result, true);'
                    else:
                        src_reads = [
                            f'amdgpu::RegisterAccess(wf).read_lane({src_ops[i]}, lane)'
                            for i in range(3)
                        ]
                        preamble = ''
                        write_result = f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, result);'
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        f'{preamble}'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'    uint16_t src0_bits = static_cast<uint16_t>({src_reads[0]});\n'
                        f'    uint16_t src1_bits = static_cast<uint16_t>({src_reads[1]});\n'
                        f'    uint16_t src2_bits = static_cast<uint16_t>({src_reads[2]});\n'
                        '    uint32_t omod = amdgpu::fp_mode::effective_f16_omod(\n'
                        '        wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), false, inst_.omod);\n'
                        '    uint16_t result = amdgpu::fp_mode::fma_f16(\n'
                        '        src0_bits, src1_bits, src2_bits, inst_.abs & 1u, inst_.abs & 2u,\n'
                        '        inst_.abs & 4u, inst_.neg & 1u, inst_.neg & 2u, inst_.neg & 4u,\n'
                        '        wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), omod,\n'
                        '        inst_.clamp, wf.fp16_ovfl(), amdgpu::floating_clamp_nan_to_zero(wf));\n'
                        f'{write_result}\n'
                        '  }\n'
                    )
                if (
                    cls == 'vector_ternary'
                    and op == 'fma'
                    and dtype == 'f64'
                    and is_vop3
                ):
                    src_loads = ''.join(
                        f'    double src{i}_value = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src_ops[i]}, lane));\n'
                        for i in range(3)
                    )
                    src_mods = ''.join(
                        f'    if (inst_.abs & (1u << {i})) src{i}_value = std::fabs(src{i}_value);\n'
                        f'    if (inst_.neg & (1u << {i})) src{i}_value = -src{i}_value;\n'
                        for i in range(3)
                    )
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'{src_loads}'
                        f'{src_mods}'
                        '    uint64_t result = amdgpu::fp_mode::fma_f64(\n'
                        '        std::bit_cast<uint64_t>(src0_value), std::bit_cast<uint64_t>(src1_value),\n'
                        '        std::bit_cast<uint64_t>(src2_value), wf.fp_round_mode_f16_f64(),\n'
                        '        wf.fp_denorm_mode_f16_f64());\n'
                        '    uint32_t omod = amdgpu::fp_mode::effective_omod(\n'
                        '        wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), inst_.omod);\n'
                        '    result = amdgpu::fp_mode::finish_f64(\n'
                        '        result, wf.fp_round_mode_f16_f64(), omod, inst_.clamp,\n'
                        '        amdgpu::floating_clamp_nan_to_zero(wf));\n'
                        f'    amdgpu::RegisterAccess(wf).write_lane64({dst_ops[0]}, lane, result);\n'
                        '  }\n'
                    )
                if cls == 'vector_binop' and op == 'fmac' and dtype == 'f64':
                    src_mods = ''
                    finish = ''
                    if is_vop3:
                        src_mods = (
                            '    if (inst_.abs & 1u) src0_value = std::fabs(src0_value);\n'
                            '    if (inst_.abs & 2u) src1_value = std::fabs(src1_value);\n'
                            '    if (inst_.neg & 1u) src0_value = -src0_value;\n'
                            '    if (inst_.neg & 2u) src1_value = -src1_value;\n'
                        )
                        finish = (
                            '    uint32_t omod = amdgpu::fp_mode::effective_omod(\n'
                            '        wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), inst_.omod);\n'
                            '    result = amdgpu::fp_mode::finish_f64(\n'
                            '        result, wf.fp_round_mode_f16_f64(), omod, inst_.clamp,\n'
                            '        amdgpu::floating_clamp_nan_to_zero(wf));\n'
                        )
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'    double src0_value = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src_ops[0]}, lane));\n'
                        f'    double src1_value = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src_ops[1]}, lane));\n'
                        f'{src_mods}'
                        f'    uint64_t accumulator = amdgpu::RegisterAccess(wf).read_lane64({dst_ops[0]}, lane);\n'
                        '    uint64_t result = amdgpu::fp_mode::fma_f64(\n'
                        '        std::bit_cast<uint64_t>(src0_value), std::bit_cast<uint64_t>(src1_value),\n'
                        '        accumulator, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());\n'
                        f'{finish}'
                        f'    amdgpu::RegisterAccess(wf).write_lane64({dst_ops[0]}, lane, result);\n'
                        '  }\n'
                    )
                if cls == 'vector_binop' and op == 'fmac' and dtype == 'f16':
                    if is_true16_vop3:
                        src_reads = [
                            f'::rocjitsu::amdgpu::read_vop3_true16_src({src_ops[i]}, wf, lane, opsel, {i})'
                            for i in range(2)
                        ]
                        accumulator_read = (
                            f'((opsel & 0x8u) != 0 ? '
                            f'(amdgpu::RegisterAccess(wf).read_lane({dst_ops[0]}, lane) >> 16) : '
                            f'amdgpu::RegisterAccess(wf).read_lane({dst_ops[0]}, lane))'
                        )
                        preamble = '  uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);\n'
                        write_result = f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst_ops[0]}, wf, lane, opsel, result, true);'
                    else:
                        src_reads = [
                            f'amdgpu::RegisterAccess(wf).read_lane({src_ops[i]}, lane)'
                            for i in range(2)
                        ]
                        accumulator_read = (
                            f'amdgpu::RegisterAccess(wf).read_lane({dst_ops[0]}, lane)'
                        )
                        preamble = ''
                        write_result = f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, result);'
                    abs0 = 'inst_.abs & 1u' if is_vop3 else 'false'
                    abs1 = 'inst_.abs & 2u' if is_vop3 else 'false'
                    neg0 = 'inst_.neg & 1u' if is_vop3 else 'false'
                    neg1 = 'inst_.neg & 2u' if is_vop3 else 'false'
                    omod = (
                        'amdgpu::fp_mode::effective_f16_omod(\n'
                        '        wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), false, inst_.omod)'
                        if is_vop3
                        else '0u'
                    )
                    clamp = 'inst_.clamp' if is_vop3 else 'false'
                    return (
                        '  uint64_t exec = wf.exec();\n'
                        f'{preamble}'
                        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                        '    if (!(exec & (1ULL << lane)))\n'
                        '      continue;\n'
                        f'    uint16_t src0_bits = static_cast<uint16_t>({src_reads[0]});\n'
                        f'    uint16_t src1_bits = static_cast<uint16_t>({src_reads[1]});\n'
                        f'    uint16_t accumulator = static_cast<uint16_t>({accumulator_read});\n'
                        f'    uint32_t omod = {omod};\n'
                        '    uint16_t result = amdgpu::fp_mode::fma_f16(\n'
                        f'        src0_bits, src1_bits, accumulator, {abs0}, {abs1}, false,\n'
                        f'        {neg0}, {neg1}, false, wf.fp_round_mode_f16_f64(),\n'
                        f'        wf.fp_denorm_mode_f16_f64(), omod, {clamp}, wf.fp16_ovfl(),\n'
                        '        amdgpu::floating_clamp_nan_to_zero(wf));\n'
                        f'{write_result}\n'
                        '  }\n'
                    )
                if cls == 'vector_add_co':
                    if is_vop3 and len(src_ops) >= 3:
                        lctx.vcc_read = (
                            f'amdgpu::read_wave_mask_scalar({src_ops[2]}, wf)'
                        )
                    lctx.vcc_dst = dst_ops[1] if len(dst_ops) > 1 else '__vcc__'
                body = lower_sema_block(sema_block, lctx)
                if (
                    cls == 'scalar_unary'
                    and op is not None
                    and op.startswith('cvt_')
                    and scc == 'none'
                ):
                    body = '  // SOP1 scalar conversions preserve SCC.\n' + body
                return body

        # Try the registry (covers all extracted gen_ functions).
        from amdisa.codegen.execute import ExecuteContext, DISPATCH

        ctx = ExecuteContext(
            inst=inst,
            sem=sem,
            dst_ops=dst_ops,
            src_ops=src_ops,
            profile=profile,
            enc_name=enc_name,
            is_vop3=is_vop3,
            has_abs=has_abs,
            opsel_exprs=self._vop3p_opsel_exprs(),
            op_sel_hi_2_expr=self._op_sel_hi_2_expr(inst.enc_name),
            arch_name=self.isa_spec.arch_name.lower(),
            enc_field_names=getattr(self, '_current_inst_fields', set()),
            encoding_map=self.isa_spec.encoding_map,
            supports_dpp=(
                cls == 'vector_movrel'
                and inst.available_encodings is not None
                and self._instruction_supports_dpp(inst, enc_name)
            ),
            supports_dpp8=(
                cls == 'vector_movrel'
                and inst.available_encodings is not None
                and self._instruction_supports_dpp8(inst, enc_name)
            ),
            supports_sdwa=(
                cls == 'vector_movrel'
                and inst.available_encodings is not None
                and self._instruction_supports_sdwa(inst, enc_name)
            ),
            result_writer=result_writer,
        )
        handler = DISPATCH.get(cls)
        if handler is not None:
            return handler(ctx)

        # Fallback: inline dispatch for classes not yet extracted.
        L = []  # output lines

        if cls == 'true_nop':
            # Sleep has no architectural register effect, but FUNCTIONAL
            # execution must return to the event loop so peer CUs can make
            # progress.
            if sem.name in ('S_SLEEP', 'S_SLEEP_VAR'):
                return self._sleep_body(sem)
            return self._trap_control_body(sem) or '  (void)wf;'

        if cls == 'gpr_idx':
            if op == 'on':
                return (
                    '  uint32_t idx = amdgpu::RegisterAccess(wf).read_scalar(ssrc0) & 0xFF;\n'
                    '  uint32_t mode = amdgpu::RegisterAccess(wf).read_scalar(ssrc1) & 0xF;\n'
                    '  wf.set_m0((wf.m0() & 0xFFFF0F00u) | (mode << 12) | idx);\n'
                    '  wf.set_mode_raw(wf.mode_raw() | Wavefront::GPR_IDX_EN_BIT);'
                )
            if op == 'off':
                return '  wf.set_mode_raw(wf.mode_raw() & ~Wavefront::GPR_IDX_EN_BIT);'
            if op == 'idx':
                return '  wf.set_m0((wf.m0() & 0xFFFFFF00u) | (amdgpu::RegisterAccess(wf).read_scalar(ssrc0) & 0xFF));'
            if op == 'mode':
                return (
                    f'  wf.set_m0((wf.m0() & 0xFFFF0FFFu) '
                    f'| ((amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]}) & 0xF) << 12));'
                )

        if cls == 'set_vgpr_msb':
            L.append(
                f'  wf.set_vgpr_msb_mode(static_cast<uint8_t>({src_ops[0]}.encoding_value_ & 0xffu));'
            )
            return '\n'.join(L)

        if cls == 'nop':
            return '  (void)wf;\n throw util::UnimplementedInst(mnemonic());'

        if cls == 'endpgm':
            # Use end() instead of halt() to drain outstanding memory ops.
            # If all wait counters are zero, end() halts immediately.
            # Otherwise, it transitions to ENDING state and the memory
            # pipeline drain handles the final halt.
            L.append('  wf.end();')
            return '\n'.join(L)

        if cls == 'trap':
            # S_TRAP transfers to the trap handler and RETURNS to the following
            # instruction. This simulator does not configure a trap handler
            # (STATUS.TRAP_EN == 0), the state in which AMD hardware executes
            # S_TRAP as a NOP. Either way the fallthrough is reachable, so S_TRAP
            # is NOT a PROGRAM_TERMINATOR (see the flag-assignment site) and
            # execution simply continues.
            return '  (void)wf;'

        if cls == 'waitcnt':
            L.append(
                f'  uint16_t imm = static_cast<uint16_t>({src_ops[0]}.encoding_value_);'
            )
            wf = self.isa_spec.profile.waitcnt_family
            if wf in ('gfx11', 'gfx12'):
                # GFX11 (RDNA3/3.5) SIMM16 layout. GFX12 uses split S_WAIT_*
                # instructions in the XML, but LLVM still accepts the
                # monolithic S_WAITCNT compatibility opcode with this layout.
                #   expcnt[2:0] = bits [2:0]
                #   lgkmcnt[5:0] = bits [9:4]
                #   vmcnt[5:0] = bits [15:10]
                L.append('  uint8_t exp = imm & 0x7;')
                L.append('  uint8_t lgkm = (imm >> 4) & 0x3F;')
                L.append('  uint8_t vm = (imm >> 10) & 0x3F;')
            else:
                # GFX9 (CDNA1-4) / GFX10 (RDNA1/2) SIMM16 layout:
                #   vmcnt[3:0] = bits [3:0], vmcnt[5:4] = bits [15:14]
                #   expcnt[2:0] = bits [6:4]
                #   lgkmcnt = bits [12:8] (GFX9) or [13:8] (GFX10)
                L.append('  uint8_t vm = (imm & 0xF) | ((imm >> 10) & 0x30);')
                L.append('  uint8_t exp = (imm >> 4) & 0x7;')
                L.append('  uint8_t lgkm = (imm >> 8) & Isa::WAITCNT_LGKMCNT_MASK;')
            L.append('  wf.set_wait_target(vm, lgkm, exp);')
            return '\n'.join(L)

        if cls == 'wait_counter':
            # RDNA4 split-wait instructions: the immediate operand is
            # the counter threshold directly (no bit-packing).
            L.append(
                f'  uint16_t cnt = static_cast<uint16_t>({src_ops[0]}.encoding_value_);'
            )
            if op == 'waitcnt_vscnt':
                L.append('  wf.set_wait_target_vscnt(static_cast<uint8_t>(cnt));')
            elif op == 'waitcnt_vmcnt':
                L.append('  wf.set_wait_target_loadcnt(static_cast<uint8_t>(cnt));')
            elif op == 'waitcnt_expcnt':
                L.append('  wf.set_wait_counter("wait_expcnt", cnt);')
            elif op == 'waitcnt_lgkmcnt':
                L.append('  const auto current_wait = wf.wait_target();')
                L.append(
                    '  wf.set_wait_target(current_wait.vmcnt, static_cast<uint8_t>(cnt), current_wait.expcnt);'
                )
            else:
                L.append(f'  wf.set_wait_counter("{op}", cnt);')
            return '\n'.join(L)

        if cls == 'tensor_load_to_lds':
            return '  amdgpu::execute_tensor_load_to_lds(*this, wf);'

        if cls == 'tensor_store_from_lds':
            return '  amdgpu::execute_tensor_store_from_lds(*this, wf);'

        if cls == 'barrier':
            L.append('  wf.set_state(amdgpu::WfState::BARRIER);')
            return '\n'.join(L)

        if cls == 'scalar_barrier_wait':
            L.append(
                f'  int32_t barrier_id = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append('  wf.barrier_wait(barrier_id);')
            return '\n'.join(L)

        if cls == 'scalar_barrier_leave':
            return '  wf.write_scc(wf.barrier_leave());'

        if cls == 'branch':
            L.append(
                f'  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append('  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;')
            return '\n'.join(L)

        if cls == 'cbranch':
            cond_map = {
                'scc0': '!wf.read_scc()',
                'scc1': 'wf.read_scc()',
                'vccz': 'live_vcc == 0',
                'vccnz': 'live_vcc != 0',
                'execz': 'wf.exec() == 0',
                'execnz': 'wf.exec() != 0',
            }
            if cond in ('vccz', 'vccnz'):
                L.append(
                    '  const uint64_t live_vcc = wf.vcc() & '
                    '(wf.wf_size() >= 64 ? ~0ULL : ((1ULL << wf.wf_size()) - 1ULL));'
                )
            L.append(f'  if ({cond_map[cond]}) {{')
            L.append(
                f'    int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append(
                '    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;'
            )
            L.append('  }')
            return '\n'.join(L)

        # scalar_mov, scalar_cmov, scalar_cselect now handled by SemaAST.

        if cls == 'scalar_movk':
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));'
            )
            return '\n'.join(L)

        if cls == 'scalar_cmovk':
            L.append(
                f'  if (wf.read_scc()) amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));'
            )
            return '\n'.join(L)

        if cls == 'scalar_mulk':
            L.append(
                f'  uint32_t s0 = amdgpu::RegisterAccess(wf).read_scalar({dst_ops[0]});'
            )
            L.append(
                f'  uint32_t imm = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_)));'
            )
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, s0 * imm);'
            )
            return '\n'.join(L)

        if cls == 'scalar_getpc':
            # S_GETPC_B64: returns PC of the instruction FOLLOWING the S_GETPC.
            # At execute() time, wf.pc points to the S_GETPC itself; step() will
            # add size_ afterwards. Write wf.pc + size_ so the net result after
            # post-execute advance is correct (caller sees PC of next instruction).
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, wf.pc + size_);'
            )
            return '\n'.join(L)

        if cls == 'scalar_setpc':
            L.append('  constexpr uint64_t kPcAddressMask = 0x0000FFFFFFFFFFFFULL;')
            L.append('  constexpr uint64_t kPcSignBit = 1ULL << 47;')
            L.append(
                f'  const uint64_t encoded = amdgpu::RegisterAccess(wf).read_scalar64({src_ops[0]});'
            )
            L.append('  uint64_t target = encoded & kPcAddressMask;')
            L.append(
                '  if ((encoded >> 32 == 0x1FFFFu || encoded >> 32 == 0xFFFFFFFFu) && wf.code_load_bias() != 0)'
            )
            L.append(
                '    target = wf.code_load_bias() + static_cast<int32_t>(encoded);'
            )
            L.append('  else if (target & kPcSignBit)')
            L.append('    target |= ~kPcAddressMask;')
            L.append('  wf.pc = target - size_;')
            return '\n'.join(L)

        if cls == 'scalar_swappc':
            # S_SWAPPC_B64: dst = PC of next inst, then jump to src.
            L.append('  constexpr uint64_t kPcAddressMask = 0x0000FFFFFFFFFFFFULL;')
            L.append('  constexpr uint64_t kPcSignBit = 1ULL << 47;')
            L.append(f'  uint64_t next_pc = wf.pc + size_;')
            L.append(
                f'  const uint64_t encoded = amdgpu::RegisterAccess(wf).read_scalar64({src_ops[0]});'
            )
            L.append('  uint64_t target = encoded & kPcAddressMask;')
            L.append(
                '  if ((encoded >> 32 == 0x1FFFFu || encoded >> 32 == 0xFFFFFFFFu) && wf.code_load_bias() != 0)'
            )
            L.append('    target = next_pc - 20 + static_cast<int32_t>(encoded);')
            L.append('  else if (target & kPcSignBit)')
            L.append('    target |= ~kPcAddressMask;')
            L.append('  wf.pc = target - size_;')
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, next_pc);'
            )
            return '\n'.join(L)

        if cls == 'scalar_bitreplicate':
            L.append(
                f'  uint32_t val = amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]});'
            )
            L.append('  uint64_t result = 0;')
            L.append('  for (uint32_t i = 0; i < 32; ++i) {')
            L.append('    uint64_t bit = (static_cast<uint64_t>(val) >> i) & 1ULL;')
            L.append('    result |= bit << (2 * i);')
            L.append('    result |= bit << (2 * i + 1);')
            L.append('  }')
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, result);'
            )
            return '\n'.join(L)

        if cls == 'scalar_addpc':
            L.append(
                f'  wf.pc += amdgpu::RegisterAccess(wf).read_scalar64({src_ops[0]});'
            )
            return '\n'.join(L)

        if cls == 'scalar_shader_cycles':
            L.append('  auto *engine = wf.cu().engine();')
            L.append('  uint64_t cycles = engine ? engine->global_time() : 0;')
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, cycles);'
            )
            return '\n'.join(L)

        if cls == 'scalar_sendmsg_rtn':
            L.append(
                f'  uint32_t msg = static_cast<uint32_t>({src_ops[0]}.encoding_value_);'
            )
            L.append('  uint64_t value = 0;')
            L.append('  switch (msg) {')
            L.append('  case 0x83: {')
            L.append('    auto *engine = wf.cu().engine();')
            L.append('    value = engine ? engine->global_time() : 0;')
            L.append('    break;')
            L.append('  }')
            L.append('  case 0x80:')  # MSG_RTN_GET_DOORBELL
            L.append('  case 0x81:')  # MSG_RTN_GET_DDID
            L.append('  case 0x82:')  # MSG_RTN_GET_TMA
            L.append('  case 0x84:')  # MSG_RTN_SAVE_WAVE
            L.append('  case 0x85:')  # MSG_RTN_GET_TBA
            L.append('  case 0x86:')  # MSG_RTN_GET_TBA_TO_PC
            L.append('  case 0x87:')  # MSG_RTN_GET_SE_AID_ID
            L.append('  case 0x88:')  # MSG_RTN_GET_CLUSTER_BARRIER_STATE
            L.append('  case 0x98:')  # MSG_RTN_SAVE_WAVE_HAS_TDM
            L.append('  default:')
            L.append('    value = 0;')
            L.append('    break;')
            L.append('  }')
            L.append(f'  if ({dst_ops[0]}.size_bits() == 64)')
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, value);'
            )
            L.append('  else')
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, static_cast<uint32_t>(value));'
            )
            return '\n'.join(L)

        if cls == 'scalar_barrier_state':
            L.append(
                f'  uint32_t source = amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]});'
            )
            L.append(
                f'  bool source_is_m0 = '
                f'{src_ops[0]}.encoding_value() == '
                f'OpSelSsrcBarrierId::OPR_SSRC_BARRIER_ID_M0;'
            )
            L.append(
                '  int32_t barrier_id = source_is_m0 ? static_cast<int32_t>(source & 0x1fu) '
                ': static_cast<int32_t>(source);'
            )
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, wf.barrier_state(barrier_id));'
            )
            return '\n'.join(L)

        if cls in (
            'scalar_barrier_signal',
            'scalar_barrier_init',
            'scalar_barrier_join',
        ):
            L.append(
                f'  uint32_t source = amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]});'
            )
            L.append(
                f'  bool source_is_m0 = '
                f'{src_ops[0]}.encoding_value() == '
                f'OpSelSsrcBarrierId::OPR_SSRC_BARRIER_ID_M0;'
            )
            L.append(
                '  int32_t barrier_id = source_is_m0 ? static_cast<int32_t>(source & 0x1fu) '
                ': static_cast<int32_t>(source);'
            )
            if cls == 'scalar_barrier_signal':
                L.append(
                    '  uint32_t member_count = source_is_m0 ? ((source >> 16) & 0x7fu) : 0;'
                )
                if op == 'signal_isfirst':
                    L.append(
                        '  bool barrier_valid = (wf.barrier_state(barrier_id) & 1u) != 0;'
                    )
                    L.append(
                        '  bool is_first = wf.barrier_signal(barrier_id, member_count);'
                    )
                    L.append('  if (barrier_valid) wf.write_scc(is_first);')
                else:
                    L.append('  wf.barrier_signal(barrier_id, member_count);')
            elif cls == 'scalar_barrier_init':
                has_implicit_m0 = any(
                    operand.name == 'm0' and operand.is_input
                    for operand in inst.operands
                )
                member_source = (
                    'wf.m0()'
                    if has_implicit_m0
                    else f'amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]})'
                )
                L.append(f'  uint32_t member_source = {member_source};')
                L.append('  uint32_t member_count = (member_source >> 16) & 0x7fu;')
                L.append('  wf.barrier_init(barrier_id, member_count);')
            else:
                L.append('  wf.barrier_join(barrier_id);')
            return '\n'.join(L)

        if cls == 'scalar_movrel':
            if op == 'src':
                L.append(f'  uint32_t index = wf.m0() & 0xFFu;')
                L.append(
                    f'  uint32_t width_words = '
                    f'static_cast<uint32_t>({src_ops[0]}.size_bits() / 32);'
                )
                L.append(
                    f'  uint32_t src_reg = static_cast<uint32_t>({src_ops[0]}.encoding_value()) + '
                    'index * width_words;'
                )
                L.append(
                    f'  Operand indexed_src({src_ops[0]}.size_bits(), OperandType::OPR_SSRC, '
                    'static_cast<int>(src_reg));'
                )
                L.append('  if (width_words == 2) {')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, amdgpu::RegisterAccess(wf).read_scalar64(indexed_src));'
                )
                L.append('  } else {')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, amdgpu::RegisterAccess(wf).read_scalar(indexed_src));'
                )
                L.append('  }')
                return '\n'.join(L)
            if op == 'dst':
                L.append(f'  uint32_t index = wf.m0() & 0xFFu;')
                L.append(
                    f'  uint32_t width_words = '
                    f'static_cast<uint32_t>({dst_ops[0]}.size_bits() / 32);'
                )
                L.append(
                    f'  uint32_t dst_reg = static_cast<uint32_t>({dst_ops[0]}.encoding_value()) + '
                    'index * width_words;'
                )
                L.append(
                    f'  Operand indexed_dst({dst_ops[0]}.size_bits(), OperandType::OPR_SDST, '
                    'static_cast<int>(dst_reg));'
                )
                L.append('  if (width_words == 2) {')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_scalar64(indexed_dst, amdgpu::RegisterAccess(wf).read_scalar64({src_ops[0]}));'
                )
                L.append('  } else {')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_scalar(indexed_dst, amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]}));'
                )
                L.append('  }')
                return '\n'.join(L)
            if op == 'srcdst2':
                L.append('  uint32_t src_index = wf.m0() & 0xFFu;')
                L.append('  uint32_t dst_index = (wf.m0() >> 8) & 0xFFu;')
                L.append(
                    f'  uint32_t src_reg = static_cast<uint32_t>({src_ops[0]}.encoding_value()) + '
                    'src_index;'
                )
                L.append(
                    f'  uint32_t dst_reg = static_cast<uint32_t>({dst_ops[0]}.encoding_value()) + '
                    'dst_index;'
                )
                L.append(
                    '  Operand indexed_src(32, OperandType::OPR_SSRC, static_cast<int>(src_reg));'
                )
                L.append(
                    '  Operand indexed_dst(32, OperandType::OPR_SDST, static_cast<int>(dst_reg));'
                )
                L.append(
                    '  amdgpu::RegisterAccess(wf).write_scalar(indexed_dst, amdgpu::RegisterAccess(wf).read_scalar(indexed_src));'
                )
                return '\n'.join(L)

        if cls == 'scalar_call':
            # S_CALL_B64: dst = PC of next instruction (return address), then branch.
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar64({dst_ops[0]}, wf.pc + size_);'
            )
            L.append(
                f'  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append('  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;')
            return '\n'.join(L)

        if cls == 'scalar_getreg':
            L.append(f'  uint16_t hwreg = {src_ops[0]}.encoding_value_;')
            L.append('  uint32_t reg_val = 0;')
            L.append('  auto result = amdgpu::read_hwreg_field(wf, hwreg, reg_val);')
            L.append('  if (result != amdgpu::HwregAccessResult::Success)')
            L.append(
                '    util::Logger::warn("s_getreg_b32: ", amdgpu::hwreg_access_result_name(result), '
                '" hwreg=", amdgpu::hwreg_name(wf, hwreg), " id=", amdgpu::hwreg_id(hwreg));'
            )
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, reg_val);'
            )
            return '\n'.join(L)

        if cls == 'scalar_setreg':
            L.append(f'  uint16_t hwreg = {dst_ops[0]}.encoding_value_;')
            L.append(
                f'  uint32_t src = amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]});'
            )
            L.append('  auto result = amdgpu::write_hwreg_field(wf, hwreg, src);')
            L.append('  if (result != amdgpu::HwregAccessResult::Success)')
            L.append(
                '    util::Logger::warn("s_setreg_b32: ", amdgpu::hwreg_access_result_name(result), '
                '" hwreg=", amdgpu::hwreg_name(wf, hwreg), " id=", amdgpu::hwreg_id(hwreg));'
            )
            return '\n'.join(L)

        if cls == 'scalar_setreg_imm':
            L.append(f'  uint16_t hwreg = {dst_ops[0]}.encoding_value_;')
            # S_SETREG_IMM32_B32's source is the extension literal. When it is
            # modeled as a fieldless simm32 operand it appears in src_ops and is
            # read through the observed RegisterAccess facade; when an ISA carries
            # it only in the encoding base's literal_ member (no operand emitted),
            # fall back to that member. Unlike FMAMK/FMAAK, this fallback is
            # retained because the missing operand here is a legitimate,
            # ISA-specific encoding shape rather than a spec omission.
            src_expr = (
                f'amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]})'
                if src_ops
                else 'literal_'
            )
            L.append(f'  uint32_t src = {src_expr};')
            L.append('  auto result = amdgpu::write_hwreg_field(wf, hwreg, src);')
            L.append('  if (result != amdgpu::HwregAccessResult::Success)')
            L.append(
                '    util::Logger::warn("s_setreg_imm32_b32: ", '
                'amdgpu::hwreg_access_result_name(result), " hwreg=", '
                'amdgpu::hwreg_name(wf, hwreg), " id=", amdgpu::hwreg_id(hwreg));'
            )
            return '\n'.join(L)

        if cls == 'vector_readfirstlane':
            L.append('  uint64_t exec = wf.exec();')
            L.append('  uint32_t val = 0;')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (exec & (1ULL << lane)) {')
            L.append(
                f'      val = amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane);'
            )
            L.append('      break;')
            L.append('    }')
            L.append('  }')
            L.append(f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, val);')
            return '\n'.join(L)

        if cls == 'vector_readlane':
            L.append(
                f'  uint32_t lane = amdgpu::RegisterAccess(wf).read_scalar({src_ops[1]});'
            )
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar({dst_ops[0]}, amdgpu::RegisterAccess(wf).read_scalar_selected_lane({src_ops[0]}, lane));'
            )
            return '\n'.join(L)

        if cls == 'vector_writelane':
            L.append(
                f'  uint32_t val = amdgpu::RegisterAccess(wf).read_scalar({src_ops[0]});'
            )
            L.append(
                f'  uint32_t lane = amdgpu::RegisterAccess(wf).read_scalar({src_ops[1]});'
            )
            L.append(
                f'  amdgpu::RegisterAccess(wf).write_scalar_selected_lane({dst_ops[0]}, lane, val);'
            )
            return '\n'.join(L)

        # vector_swap now handled by SemaAST.

        if cls == 'vector_fmamk':
            # D = S0 * K + S2, K is inline constant (second src operand)
            if len(src_ops) < 3:
                raise ValueError(
                    f'{inst.name}: expected fieldless simm32 operand for {cls}'
                )
            k_expr = f'{src_ops[1]}.encoding_value_'
            s2_expr = src_ops[2]
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                if inst.name.startswith('V_FMA'):
                    L.append(
                        f'    uint16_t s0 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane));'
                    )
                    L.append(f'    uint16_t k = static_cast<uint16_t>({k_expr});')
                    L.append(
                        f'    uint16_t s2 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({s2_expr}, lane));'
                    )
                    L.append(
                        f'    uint16_t result = amdgpu::fp_mode::fma_f16(s0, k, s2, false, false, false, false, false, false, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, false, wf.fp16_ovfl(), amdgpu::floating_clamp_nan_to_zero(wf));'
                    )
                    L.append(
                        f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, result);'
                    )
                else:
                    L.append(
                        f'    float s0 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane)));'
                    )
                    L.append(
                        f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));'
                    )
                    L.append(
                        f'    float s2 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({s2_expr}, lane)));'
                    )
                    L.append(
                        f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, util::f32_to_f16_mode(std::fma(s0, k, s2), wf.fp16_ovfl()));'
                    )
            elif dtype == 'f64':
                L.append(
                    f'    uint64_t s0 = amdgpu::RegisterAccess(wf).read_lane64({src_ops[0]}, lane);'
                )
                L.append(
                    f'    uint64_t k = amdgpu::RegisterAccess(wf).read_lane64({src_ops[1]}, lane);'
                )
                L.append(
                    f'    uint64_t s2 = amdgpu::RegisterAccess(wf).read_lane64({s2_expr}, lane);'
                )
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane64({dst_ops[0]}, lane, amdgpu::fp_mode::fma_f64(s0, k, s2, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64()));'
                )
            else:
                L.append(
                    f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane));'
                )
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(
                    f'    float s2 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({s2_expr}, lane));'
                )
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, std::bit_cast<uint32_t>(std::fma(s0, k, s2)));'
                )
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_fmaak':
            # D = S0 * S1 + K, K is inline constant (third src operand)
            if len(src_ops) < 3:
                raise ValueError(
                    f'{inst.name}: expected fieldless simm32 operand for {cls}'
                )
            k_expr = f'{src_ops[2]}.encoding_value_'
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                if inst.name.startswith('V_FMA'):
                    L.append(
                        f'    uint16_t s0 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane));'
                    )
                    L.append(
                        f'    uint16_t s1 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[1]}, lane));'
                    )
                    L.append(f'    uint16_t k = static_cast<uint16_t>({k_expr});')
                    L.append(
                        '    uint16_t result = amdgpu::fp_mode::fma_f16(s0, s1, k, false, false, false, false, false, false, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, false, wf.fp16_ovfl(), amdgpu::floating_clamp_nan_to_zero(wf));'
                    )
                    L.append(
                        f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, result);'
                    )
                else:
                    L.append(
                        f'    float s0 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane)));'
                    )
                    L.append(
                        f'    float s1 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_ops[1]}, lane)));'
                    )
                    L.append(
                        f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));'
                    )
                    L.append(
                        f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, util::f32_to_f16_mode(std::fma(s0, s1, k), wf.fp16_ovfl()));'
                    )
            elif dtype == 'f64':
                L.append(
                    f'    uint64_t s0 = amdgpu::RegisterAccess(wf).read_lane64({src_ops[0]}, lane);'
                )
                L.append(
                    f'    uint64_t s1 = amdgpu::RegisterAccess(wf).read_lane64({src_ops[1]}, lane);'
                )
                L.append(
                    f'    uint64_t k = amdgpu::RegisterAccess(wf).read_lane64({src_ops[2]}, lane);'
                )
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane64({dst_ops[0]}, lane, amdgpu::fp_mode::fma_f64(s0, s1, k, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64()));'
                )
            else:
                L.append(
                    f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src_ops[0]}, lane));'
                )
                L.append(
                    f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src_ops[1]}, lane));'
                )
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane({dst_ops[0]}, lane, std::bit_cast<uint32_t>(std::fma(s0, s1, k)));'
                )
            L.append('  }')
            return '\n'.join(L)

        if cls in (
            'vector_cvt_pk_u8_f32',
            'vector_cvt_pknorm',
            'vector_cvt_pkrtz_f16_f32',
            'vector_cvt_pk',
            'vector_cvt_pk_f16_f32',
            'vector_cvt_pk_bf16_f32',
            'vector_cvt_sr_pk_f16_f32',
            'vector_cvt_sr_pk_bf16_f32',
            'vector_cvt_scale',
            'vector_cvt_sr_f16_f32',
            'vector_cvt_sr_bf16_f32',
            'vector_cvt_sr_fp8_f16',
            'vector_cvt_sr_bf8_f16',
            'vector_pack_b32_f16',
        ):
            if cls == 'vector_cvt_scale':
                return gen_vector_cvt_scale(
                    dst_ops, src_ops, cls, op, self.isa_spec.arch_name
                )
            opsel = '0u'
            if is_vop3:
                inst_fields = getattr(self, '_current_inst_fields', set())
                opsel = 'inst_.opsel' if 'opsel' in inst_fields else 'inst_.op_sel'
            fp8_format_select = (
                'inst_.clamp'
                if (
                    (cls == 'vector_cvt_pk' and op in ('fp8_f32', 'fp8_f16'))
                    or cls == 'vector_cvt_sr_fp8_f16'
                )
                and is_vop3
                and self.isa_spec.arch_name.lower() == 'cdna5'
                else None
            )
            return gen_vector_cvt_pk(
                dst_ops,
                src_ops,
                cls,
                op,
                opsel=opsel,
                dtype=dtype,
                is_vop3=is_vop3,
                fp8_format_select=fp8_format_select,
                arch_name=self.isa_spec.arch_name,
            )

        # ----- VOP3P: packed / dot / mix / MFMA -----
        if cls.startswith('dot2_'):
            return gen_dot2(
                dst_ops, src_ops, cls, opsel_exprs=self._vop3p_opsel_exprs()
            )

        if cls.startswith('dot4_'):
            return gen_dot4(dst_ops, src_ops, cls)

        if cls.startswith('dot8_'):
            return gen_dot8(dst_ops, src_ops, cls)

        if cls == 'smem_load':
            return self._gen_smem_load(dst_ops, src_ops, sem)

        if cls == 'smem_store':
            return self._gen_smem_store(dst_ops, src_ops, sem)

        if cls == 'flat_load':
            return self._gen_flat_load(dst_ops, src_ops, sem)

        if cls == 'flat_store':
            return self._gen_flat_store(dst_ops, src_ops, sem)

        if cls == 'global_load_async_to_lds':
            return self._gen_global_load_async_to_lds(dst_ops, src_ops, sem)

        if cls == 'global_store_async_from_lds':
            return self._gen_global_store_async_from_lds(dst_ops, src_ops, sem)

        if cls == 'global_load_addtid':
            return self._gen_global_load_addtid(dst_ops, src_ops, sem)

        if cls == 'global_store_addtid':
            return self._gen_global_store_addtid(dst_ops, src_ops, sem)

        if cls in ('buffer_load', 'tbuffer_load'):
            return self._gen_buffer_load(dst_ops, src_ops, sem, cls, inst)

        if cls in ('buffer_store', 'tbuffer_store'):
            return self._gen_buffer_store(dst_ops, src_ops, sem, cls)

        if cls in ('buffer_load_format_d16', 'buffer_store_format_d16'):
            # Packed D16 FORMAT execution (two 16-bit components per VGPR) is not
            # modeled by the memory pipeline; the class is metadata-only so the
            # partial-def liveness of D16 FORMAT loads is still emitted.
            return (
                '  (void)wf;\n'
                '  throw util::UnimplementedInst(mnemonic()); '
                '// packed D16 FORMAT execution intentionally unimplemented'
            )

        if cls in (
            'ds_read',
            'ds_read2',
            'ds_write',
            'ds_write2',
            'ds_read_addtid',
            'ds_write_addtid',
            'ds_read_tr_b16',
            'ds_read_tr_b8',
            'ds_read_tr_b4',
            'ds_read_tr_b6',
        ):
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = (
                    '  if (inst_.gds)\n'
                    '    throw util::UnimplementedInst(mnemonic());\n'
                )
            if cls == 'ds_read':
                return gds_guard + self._gen_ds_read(dst_ops, src_ops, sem)
            if cls == 'ds_read2':
                return gds_guard + self._gen_ds_read2(dst_ops, src_ops, sem)
            if cls == 'ds_write':
                return gds_guard + self._gen_ds_write(dst_ops, src_ops, sem)
            if cls == 'ds_write2':
                return gds_guard + self._gen_ds_write2(dst_ops, src_ops, sem)
            if cls == 'ds_read_addtid':
                return gds_guard + self._gen_ds_read_addtid(dst_ops, src_ops, sem)
            if cls == 'ds_write_addtid':
                return gds_guard + self._gen_ds_write_addtid(dst_ops, src_ops, sem)
            if cls.startswith('ds_read_tr_'):
                return gds_guard + self._gen_ds_read_tr(dst_ops, src_ops, sem)
            return gds_guard + self._gen_ds_write2(dst_ops, src_ops, sem)

        if cls == 'dcache_inv':
            return '  wf.cu().l1_scalar().invalidate_all();'

        if cls == 'dcache_wb':
            return '  wf.cu().l1_scalar().writeback_all(wf.process_id());'

        if cls == 'gl1_inv':
            return (
                '  wf.cu().l1_vector().invalidate_all();\n'
                '  if (auto *l2 = wf.cu().l2())\n'
                '    l2->flush_all(wf.process_id());'
            )

        if cls == 'icache_inv':
            return '  wf.cu().instruction_cache().invalidate_all();'

        if cls == 'gl2_wb':
            return (
                '  if (auto *l2 = wf.cu().l2())\n' '    l2->flush_all(wf.process_id());'
            )

        if cls == 'smem_time':
            return (
                '  static thread_local uint64_t counter = 0;\n'
                '  counter += 100;\n'
                '  const uint32_t dst_sel = inst_.sdata;\n'
                '  amdgpu::write_scalar_selector(wf, dst_sel, static_cast<uint32_t>(counter));\n'
                '  amdgpu::write_scalar_selector(wf, dst_sel + 1, static_cast<uint32_t>(counter >> 32));'
            )

        if cls == 'gl1_wbinv':
            return '  wf.cu().l1_vector().flush_all();'

        if cls == 'flat_atomic':
            return self._gen_flat_atomic(dst_ops, src_ops, sem)

        if cls == 'buffer_atomic':
            return self._gen_buffer_atomic(dst_ops, src_ops, sem)

        if cls in ('ds_atomic', 'ds_atomic2'):
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = (
                    '  if (inst_.gds)\n'
                    '    throw util::UnimplementedInst(mnemonic());\n'
                )
            if cls == 'ds_atomic2':
                return gds_guard + self._gen_ds_atomic2(dst_ops, src_ops, sem)
            return gds_guard + self._gen_ds_atomic(dst_ops, src_ops, sem)

        if cls in ('ds_mskor', 'ds_append_consume', 'ds_barrier_arrive'):
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = (
                    '  if (inst_.gds)\n'
                    '    throw util::UnimplementedInst(mnemonic());\n'
                )
            if cls == 'ds_mskor':
                return gds_guard + self._gen_ds_mskor(dst_ops, src_ops, sem)
            if cls == 'ds_append_consume':
                return gds_guard + self._gen_ds_append_consume(dst_ops, src_ops, sem)
            return gds_guard + self._gen_ds_barrier_arrive(dst_ops, src_ops, sem)

        if cls == 'ds_permute':
            is_bpermute = 'BPERMUTE' in sem.name.upper()
            fetch_invalid = 'BPERMUTE_FI' in sem.name.upper()
            L.append(f'  auto &cu = wf.cu();')
            L.append(f'  uint64_t exec = wf.exec();')
            L.append(f'  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);')
            L.append(f'  uint32_t lane_group_width = wf.wf_size();')
            L.append(f'  ::rocjitsu::amdgpu::RegisterAccess regs(wf);')
            L.append(
                f'  if (wf.wf_size() == 64 && (cu.arch() == ROCJITSU_CODE_ARCH_RDNA3 ||'
                f' cu.arch() == ROCJITSU_CODE_ARCH_RDNA3_5))'
            )
            L.append(f'    lane_group_width = 32;')
            L.append(f'  // Pre-read all data0 values from every lane.')
            L.append(f'  uint32_t src_data[64];')
            L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
            L.append(f'    src_data[i] = regs.read_lane(data0, i);')
            if is_bpermute:
                # DS_BPERMUTE_B32 (ISA spec pseudocode, page 476):
                #   tmp[i] = 0 for all lanes
                #   for i in 0..63:
                #     src_lane = (VGPR[i][ADDR] + OFFSET) / 4 % 64
                #     if EXEC[src_lane]: tmp[i] = VGPR[src_lane][DATA0]
                #   for i in 0..63:
                #     if EXEC[i]: VGPR[i][VDST] = tmp[i]
                L.append(f'  uint32_t tmp[64] = {{}};')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    uint32_t addr_val = regs.read_lane(addr, i);')
                L.append(
                    f'    uint32_t group_base = (i / lane_group_width) * lane_group_width;'
                )
                L.append(
                    f'    uint32_t src_lane = group_base + (((addr_val + offset) / 4) % lane_group_width);'
                )
                if fetch_invalid:
                    L.append(f'    tmp[i] = src_data[src_lane];')
                else:
                    L.append(f'    if (exec & (1ULL << src_lane))')
                    L.append(f'      tmp[i] = src_data[src_lane];')
                L.append(f'  }}')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    if (exec & (1ULL << i))')
                L.append(f'      regs.write_lane(vdst, i, tmp[i]);')
                L.append(f'  }}')
            else:
                # DS_PERMUTE_B32 (ISA spec pseudocode, page 475):
                #   tmp[i] = 0 for all lanes
                #   for i in 0..63:
                #     if EXEC[i]:
                #       dst_lane = (VGPR[i][ADDR] + OFFSET) / 4 % 64
                #       tmp[dst_lane] = VGPR[i][DATA0]
                #   for i in 0..63:
                #     if EXEC[i]: VGPR[i][VDST] = tmp[i]
                L.append(f'  uint32_t tmp[64] = {{}};')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    if (!(exec & (1ULL << i))) continue;')
                L.append(f'    uint32_t addr_val = regs.read_lane(addr, i);')
                L.append(
                    f'    uint32_t group_base = (i / lane_group_width) * lane_group_width;'
                )
                L.append(
                    f'    uint32_t dst_lane = group_base + (((addr_val + offset) / 4) % lane_group_width);'
                )
                L.append(f'    tmp[dst_lane] = src_data[i];')
                L.append(f'  }}')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    if (exec & (1ULL << i))')
                L.append(f'      regs.write_lane(vdst, i, tmp[i]);')
                L.append(f'  }}')
            return '\n'.join(L)

        if cls == 'ds_swizzle':
            # DS_SWIZZLE_B32: lane swizzle controlled by offset field.
            # The offset encodes the swizzle pattern. For QDMode (bit 15=1):
            #   for each lane in quad: dst = src[packed_2bit_selector]
            # For BitMode (bit 15=0): swizzle within 32-lane rows via and/or/xor.
            src_field = 'addr'
            L.append(f'  uint64_t exec = wf.exec();')
            L.append(f'  ::rocjitsu::amdgpu::RegisterAccess regs(wf);')
            L.append(f'  uint32_t src_data[64];')
            L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
            L.append(f'    src_data[i] = regs.read_lane({src_field}, i);')
            L.append(f'  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);')
            L.append(f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{')
            L.append(f'    if (!(exec & (1ULL << lane))) continue;')
            L.append(f'    uint32_t src_lane;')
            L.append(f'    if (offset & 0x8000) {{')
            L.append(f'      // QDMode: four packed 2-bit selectors within each quad.')
            L.append(
                f'      src_lane = (lane & ~0x3u) | ((offset >> (2u * (lane & 0x3u))) & 0x3u);'
            )
            L.append(f'    }} else {{')
            L.append(f'      // BitMode: swizzle within 32-lane rows.')
            L.append(f'      uint32_t and_mask = offset & 0x1F;')
            L.append(f'      uint32_t or_mask = (offset >> 5) & 0x1F;')
            L.append(f'      uint32_t xor_mask = (offset >> 10) & 0x1F;')
            L.append(f'      uint32_t row_base = lane & ~0x1Fu;')
            L.append(f'      uint32_t row_lane = lane & 0x1Fu;')
            L.append(
                f'      src_lane = row_base + (((row_lane & and_mask) | or_mask) ^ xor_mask);'
            )
            L.append(f'    }}')
            L.append(f'    if (src_lane < wf.wf_size())')
            L.append(f'      regs.write_lane(vdst, lane, src_data[src_lane]);')
            L.append(f'  }}')
            return '\n'.join(L)

        # ── Image pipeline stubs ──────────────────────────────────────────
        # NOTE for image execution: the image ADDRESS is carried as the
        # fieldless ``vaddr`` operand (OPR_VGPR), currently emitted as an
        # inert placeholder. The real gfx12 address is NSA -- up to
        # 5 non-contiguous VGPRs (vaddr0..vaddr4 in the encoding) that a
        # single base+width Operand cannot express -- so decode it from the
        # machine-inst fields here. ``vdata`` and ``rsrc`` are field-bearing
        # and already modeled.
        if cls == 'image_load':
            # Minimal image load: treat as a flat read from the image resource base address.
            # Full image addressing (texture coordinates, dimensions) not yet implemented.
            L.append('  // Minimal image load stub — not yet implemented.')
            L.append('  (void)wf;')
            return '\n'.join(L)

        if cls == 'image_store':
            L.append('  // Minimal image store stub — not yet implemented.')
            L.append('  (void)wf;')
            return '\n'.join(L)

        if cls in ('image_atomic', 'image_sample', 'image_query', 'image_bvh'):
            L.append('  (void)wf; // Image pipeline not yet implemented.')
            return '\n'.join(L)

        # ── Graphics-only stubs (no-ops in compute simulation) ───────────
        if cls == 'export':
            L.append('  (void)wf; // Export: no-op in compute simulation.')
            return '\n'.join(L)

        if cls in ('interp', 'lds_direct'):
            L.append(
                '  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.'
            )
            return '\n'.join(L)

        return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: {cls}'

    def _gen_smem_load(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        elem_size = sem.elem_size or 4
        nd = sem.num_elems if elem_size == 4 else 1
        L.append(
            f'  auto dst_register = amdgpu::resolve_scalar_register_range(wf, inst_.sdata, {nd}u);'
        )
        L.append('  if (!dst_register) return;')
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append('  d->dst_register = *dst_register;')
        L.append(f'  d->num_dwords = {nd};')
        L.append(f'  d->elem_size = {elem_size};')
        L.append(f'  d->sign_extend = {str(sem.sign_extend).lower()};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'smem_load')
        L.append(f'  d->mtype = {self._mtype_expr(is_smem=True)};')
        if self.isa_spec.profile.smem_address_uses_access_size:
            addr_args = 'inst_, wf, d->elem_size * d->num_dwords'
        else:
            addr_args = 'inst_, wf'
        L.append(f'  auto address = smem_calculate_address({addr_args});')
        L.append('  if (!address) return;')
        L.append('  d->addr = *address;')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_smem_store(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        nd = sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->num_dwords = {nd};')
        if self.isa_spec.profile.smem_address_uses_access_size:
            L.append('  d->elem_size = 4;')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'smem_store')
        L.append(f'  d->mtype = {self._mtype_expr(is_smem=True)};')
        L.append('  const uint32_t sdata_sel = inst_.sdata;')
        L.append(
            f'  auto src_register = amdgpu::resolve_scalar_register_range(wf, sdata_sel, {nd}u);'
        )
        L.append('  if (!src_register) return;')
        L.append(f'  for (uint32_t i = 0; i < {nd}; ++i)')
        L.append(
            '    d->store_data[i] = amdgpu::read_scalar_register(wf, *src_register, i);'
        )
        if self.isa_spec.profile.smem_address_uses_access_size:
            addr_args = 'inst_, wf, d->elem_size * d->num_dwords'
        else:
            addr_args = 'inst_, wf'
        L.append(f'  auto address = smem_calculate_address({addr_args});')
        L.append('  if (!address) return;')
        L.append('  d->addr = *address;')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _vop3p_opsel_exprs(self) -> tuple[str, str]:
        """Return ``(op_sel_expr, op_sel_hi_expr)`` for VOP3P execute() bodies."""
        opsel, opsel_hi = self.isa_spec.profile.vop3p_opsel_fields
        return f'inst_.{opsel}', f'inst_.{opsel_hi}'

    def _coherency_exprs(self) -> tuple[str, str, str]:
        """Return ``(sc0_expr, sc1_expr, nt_expr)`` for execute() body templates.

        Consults the ISA profile so that ISAs with GLC/SLC field names (CDNA1/2,
        RDNA1-3.5) emit ``inst_.glc`` / ``inst_.slc`` instead of the CDNA3/4
        ``inst_.sc0`` / ``inst_.sc1``.  When the profile has no NT field the
        nt_expr is the literal ``0``.
        """
        sc0, sc1, nt = self.isa_spec.profile.coherency_field_names
        sc0_expr = f'inst_.{sc0}'
        sc1_expr = f'inst_.{sc1}'
        nt_expr = f'inst_.{nt}' if nt else '0'
        return sc0_expr, sc1_expr, nt_expr

    def _mtype_expr(self, is_smem: bool = False) -> str:
        """Return the correct ``mtype_from_flags_*()`` call for this ISA.

        For SMEM instructions, the available coherency fields differ from
        vector memory:
        - CDNA1/2: SMEM has only ``glc`` (same as vector).
        - CDNA3/4: SMEM retains ``glc``-only even though vector uses SC0/SC1/NT.
        - RDNA1/2: SMEM has ``glc`` + ``dlc`` but NOT ``slc``.
        - RDNA3/3.5: SMEM has ``glc`` + ``dlc`` but NOT ``slc``.
        - RDNA4: SMEM has ``scope`` + ``th``.

        Args:
            is_smem: True if this is a scalar memory (SMEM) instruction.
        """
        from amdisa.isa_profile import MemoryCoherencyModel

        model = self.isa_spec.profile.coherency_model
        if is_smem:
            # SMEM has limited coherency fields compared to vector memory.
            if model in (
                MemoryCoherencyModel.GFX9_GLC,
                MemoryCoherencyModel.GFX940_SC0_SC1_NT,
            ):
                return 'amdgpu::mtype_from_flags_gfx9(inst_.glc)'
            if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC:
                # SMEM on GFX10 has glc+dlc but no slc.
                return 'amdgpu::mtype_from_flags_gfx10(inst_.glc, inst_.dlc, false)'
            if model == MemoryCoherencyModel.GFX11_SC0_SC1_TH:
                # SMEM on GFX11 has glc+dlc but no slc.
                return 'amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false)'
            if model == MemoryCoherencyModel.GFX12_SCOPE_TH:
                return 'amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th)'
        if model == MemoryCoherencyModel.GFX9_GLC:
            return 'amdgpu::mtype_from_flags_gfx9(inst_.glc)'
        if model == MemoryCoherencyModel.GFX940_SC0_SC1_NT:
            return 'amdgpu::mtype_from_flags_gfx940(inst_.sc0, inst_.sc1, inst_.nt)'
        if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC:
            return 'amdgpu::mtype_from_flags_gfx10(inst_.glc, inst_.dlc, inst_.slc)'
        if model == MemoryCoherencyModel.GFX11_SC0_SC1_TH:
            return 'amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, inst_.slc)'
        if model == MemoryCoherencyModel.GFX12_SCOPE_TH:
            return 'amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th)'
        return 'amdgpu::Mtype::RW'

    def _atomic_return_expr(self, sc0_expr: str) -> str:
        """Return the expression that marks whether an atomic returns old data."""
        from amdisa.isa_profile import MemoryCoherencyModel

        if self.isa_spec.profile.coherency_model == MemoryCoherencyModel.GFX12_SCOPE_TH:
            return 'amdgpu::gfx12_atomic_returns(inst_.th)'
        return f'({sc0_expr} != 0)'

    def _cache_flags_includes(self) -> list[str]:
        """Return cache_flags header path(s) for this ISA's coherency model.

        GFX940 (CDNA3/4) needs both gfx940 (vector memory) and gfx9 (SMEM).
        """
        from amdisa.isa_profile import MemoryCoherencyModel

        model = self.isa_spec.profile.coherency_model
        base = 'rocjitsu/isa/arch/amdgpu/shared'
        if model == MemoryCoherencyModel.GFX940_SC0_SC1_NT:
            return [f'{base}/gfx940_cache_flags.h', f'{base}/gfx9_cache_flags.h']
        _MAP = {
            MemoryCoherencyModel.GFX9_GLC: 'gfx9_cache_flags.h',
            MemoryCoherencyModel.GFX10_GLC_DLC_SLC: 'gfx10_cache_flags.h',
            MemoryCoherencyModel.GFX11_SC0_SC1_TH: 'gfx11_cache_flags.h',
            MemoryCoherencyModel.GFX12_SCOPE_TH: 'gfx12_cache_flags.h',
        }
        return [f'{base}/{_MAP[model]}']

    def _wait_counter_type(self, sem_class: str) -> str | None:
        """Return the WaitCounterType enum for a given memory semantic class.

        Returns None for non-memory instructions. Maps semantic classes to the
        correct counter that must be incremented when the instruction issues.
        """
        from amdisa.isa_profile import MemoryCoherencyModel

        model = self.isa_spec.profile.coherency_model
        is_gfx11_plus = model in (
            MemoryCoherencyModel.GFX11_SC0_SC1_TH,
            MemoryCoherencyModel.GFX12_SCOPE_TH,
        )
        _MAP = {
            'smem_load': (
                'amdgpu::WaitCounterType::KMCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'smem_store': (
                'amdgpu::WaitCounterType::KMCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'flat_load': (
                'amdgpu::WaitCounterType::LOADCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::VMCNT'
            ),
            'flat_store': (
                'amdgpu::WaitCounterType::STORECNT'
                if is_gfx11_plus
                else (
                    'amdgpu::WaitCounterType::VSCNT'
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else 'amdgpu::WaitCounterType::VMCNT'
                )
            ),
            'flat_atomic': (
                'amdgpu::WaitCounterType::LOADCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::VMCNT'
            ),
            'buffer_load': (
                'amdgpu::WaitCounterType::LOADCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::VMCNT'
            ),
            'buffer_store': (
                'amdgpu::WaitCounterType::STORECNT'
                if is_gfx11_plus
                else (
                    'amdgpu::WaitCounterType::VSCNT'
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else 'amdgpu::WaitCounterType::VMCNT'
                )
            ),
            'tbuffer_load': (
                'amdgpu::WaitCounterType::LOADCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::VMCNT'
            ),
            'tbuffer_store': (
                'amdgpu::WaitCounterType::STORECNT'
                if is_gfx11_plus
                else (
                    'amdgpu::WaitCounterType::VSCNT'
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else 'amdgpu::WaitCounterType::VMCNT'
                )
            ),
            'global_load': (
                'amdgpu::WaitCounterType::LOADCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::VMCNT'
            ),
            'global_store': (
                'amdgpu::WaitCounterType::STORECNT'
                if is_gfx11_plus
                else (
                    'amdgpu::WaitCounterType::VSCNT'
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else 'amdgpu::WaitCounterType::VMCNT'
                )
            ),
            'global_load_async_to_lds': 'amdgpu::WaitCounterType::ASYNCCNT',
            'global_store_async_from_lds': 'amdgpu::WaitCounterType::ASYNCCNT',
            'ds_read': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_read2': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_write': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_write2': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_atomic': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_atomic2': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_mskor': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_append_consume': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_barrier_arrive': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_barrier_arrive_async': 'amdgpu::WaitCounterType::ASYNCCNT',
            'ds_read_addtid': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_write_addtid': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_read_tr_b16': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_read_tr_b8': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_read_tr_b4': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
            'ds_read_tr_b6': (
                'amdgpu::WaitCounterType::DSCNT'
                if is_gfx11_plus
                else 'amdgpu::WaitCounterType::LGKMCNT'
            ),
        }
        return _MAP.get(sem_class)

    def _append_wait_counter_type(self, lines: list[str], sem_class: str) -> None:
        counter = self._wait_counter_type(sem_class)
        if counter is not None:
            lines.append(f'  d->wait_counter_type = {counter};')

    @property
    def _acc_vgpr_expr(self) -> str:
        """AccVGPR offset expression for the current encoding.

        When the encoding has an ``acc`` bit field and acc=1, data/vdata/vdst
        references AccVGPRs (physical +256). For encodings without ``acc``
        (CDNA1 DS/FLAT, all RDNA), returns ``0u``.
        """
        if hasattr(self, '_current_inst_fields') and 'acc' in self._current_inst_fields:
            return '(inst_.acc ? 256u : 0u)'
        return '0u'

    def _vgpr_base_expr(
        self,
        opnd_name: str,
        inst_field_name: str | None = None,
        use_acc: bool = True,
        role: str | None = None,
    ) -> str:
        """Return a C++ expression for a physical VGPR base.

        gfx1250 uses MODE-controlled high-bank bits for VGPR operands. The
        generated operand already carries the correct Src/Dst role, so route
        those operands through the ISA resolver. Other profiles preserve the
        existing raw encoding plus optional AccVGPR offset behavior.
        """
        field = inst_field_name or opnd_name
        if self.isa_spec.profile.uses_vgpr_msb_indexing:
            if role is not None:
                return (
                    'wf.vgpr_alloc().base + *Isa::resolved_vgpr_offset(wf, '
                    f'OperandType::OPR_VGPR, inst_.{field}, amdgpu::VgprMsbRole::{role})'
                )
            return (
                'wf.vgpr_alloc().base + *Isa::resolved_vgpr_offset(wf, '
                f'{opnd_name}.opr_type_, {opnd_name}.encoding_value_, '
                f'{opnd_name}.vgpr_msb_role())'
            )

        if use_acc:
            return f'wf.vgpr_alloc().base + {self._acc_vgpr_expr} + inst_.{field}'
        return f'wf.vgpr_alloc().base + inst_.{field}'

    def _gen_flat_load(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'flat_load')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        if sem.d16_hi:
            L.append('  d->d16_hi = true;')
        if sem.d16_lo:
            L.append('  d->d16_lo = true;')
        if getattr(sem, 'transpose_kind', 0):
            L.append(f'  d->transpose = {sem.transpose_kind};')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        if sem.name.startswith('CLUSTER_LOAD_'):
            L.append('  d->request_force_l1_bypass = true;')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_flat_store(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        data_field = self.isa_spec.profile.flat_store_src_field
        data_base = self._vgpr_base_expr(data_field)
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'flat_store')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f'  uint32_t data_base = {data_base};')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz == 4:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {i}, lane);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 4);'
                )
            elif esz == 2:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});'
                )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_global_load_async_to_lds(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        _, _, nt = self._coherency_exprs()
        stride = esz * ne
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'global_load_async_to_lds')
        L.append('  d->lds_dst = true;')
        L.append('  d->lds_per_lane_addr = true;')
        L.append('  d->lds_base = wf.lds_base();')
        if sem.name.startswith('CLUSTER_LOAD_ASYNC_TO_LDS_'):
            L.append('  d->cluster_multicast = true;')
            L.append(
                '  d->cluster_mcast_mask = wf.m0() & amdgpu::kClusterMulticastMask;'
            )
            L.append('  d->request_force_l1_bypass = true;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            '  // CDNA5 ISA 10.8.1 and expressions 95-102 and 106-109 add IOFFSET to global and LDS addresses.'
        )
        L.append(f"  uint32_t lds_addr_base = {self._vgpr_base_expr('vdst')};")
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t lane_lds_addr = amdgpu::RegisterAccess(cu).read_vgpr(lds_addr_base, lane);'
        )
        L.append(
            f'    d->per_lane_lds_addr[lane] = async_lds_lane_address(inst_, wf, lane_lds_addr, {stride});'
        )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_global_store_async_from_lds(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        _, _, nt = self._coherency_exprs()
        stride = esz * ne
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'global_store_async_from_lds')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  const auto &lds = cu.lds();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            '  // CDNA5 ISA 10.8.1 and expressions 95-102 and 106-109 add IOFFSET to global and LDS addresses.'
        )
        L.append(f"  uint32_t lds_addr_base = {self._vgpr_base_expr('vsrc')};")
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t lane_lds_addr = amdgpu::RegisterAccess(cu).read_vgpr(lds_addr_base, lane);'
        )
        L.append(
            f'    uint32_t lds_addr = async_lds_lane_address(inst_, wf, lane_lds_addr, {stride});'
        )
        L.append(
            '    // Out-of-range LDS reads return zero; the global store still issues.'
        )
        L.append(f'    lds.read(lds_addr, &d->store_data[lane * {stride}], {stride});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _append_global_addtid_addresses(self, L: list[str]) -> None:
        L.append('  {')
        L.append('    uint64_t exec = wf.exec();')
        L.append('    d->lane_mask = exec; d->exec_mask = exec;')
        L.append('    d->wf_size = wf.wf_size();')
        L.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        L.append('    uint64_t base = amdgpu::RegisterAccess(wf).read_scalar64(saddr);')
        offset_expr = (
            'signed_ioffset(inst_.ioffset)'
            if self.isa_spec.arch_name == 'cdna5'
            else 'static_cast<int32_t>(inst_.ioffset << 8) >> 8'
        )
        L.append(f'    int64_t offset = static_cast<int64_t>({offset_expr});')
        L.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('      if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '      d->per_lane_addr[lane] = base + static_cast<uint64_t>(offset + static_cast<int64_t>(lane * 4));'
        )
        L.append('    }')
        L.append('  }')

    def _gen_global_load_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'global_load')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append('  d->non_temporal = 0;')
        self._append_global_addtid_addresses(L)
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_global_store_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'global_store')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append('  d->non_temporal = 0;')
        self._append_global_addtid_addresses(L)
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('vsrc')};")
        L.append('  d->store_data.resize(wf.wf_size() * 4);')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t val0 = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
        )
        L.append('    std::memcpy(&d->store_data[lane * 4], &val0, 4);')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    _ATOMIC_OP_ENUM: dict[str, str] = {
        'swap': 'amdgpu::AtomicOp::SWAP',
        'cmpswap': 'amdgpu::AtomicOp::CMPSWAP',
        'mskor': 'amdgpu::AtomicOp::MSKOR',
        'add': 'amdgpu::AtomicOp::ADD',
        'sub': 'amdgpu::AtomicOp::SUB',
        'rsub': 'amdgpu::AtomicOp::RSUB',
        'smin': 'amdgpu::AtomicOp::SMIN',
        'umin': 'amdgpu::AtomicOp::UMIN',
        'smax': 'amdgpu::AtomicOp::SMAX',
        'umax': 'amdgpu::AtomicOp::UMAX',
        'and': 'amdgpu::AtomicOp::AND',
        'or': 'amdgpu::AtomicOp::OR',
        'xor': 'amdgpu::AtomicOp::XOR',
        'inc': 'amdgpu::AtomicOp::INC',
        'dec': 'amdgpu::AtomicOp::DEC',
        'fadd': 'amdgpu::AtomicOp::FADD',
        'fmin': 'amdgpu::AtomicOp::FMIN',
        'fmax': 'amdgpu::AtomicOp::FMAX',
        'append': 'amdgpu::AtomicOp::APPEND',
        'consume': 'amdgpu::AtomicOp::CONSUME',
        'barrier_arrive': 'amdgpu::AtomicOp::BARRIER_ARRIVE',
    }

    def _buffer_payload_exec_expr(self) -> str:
        if self.isa_spec.profile.buffer_payload_reads_use_effective_exec_mask:
            return 'd->exec_mask'
        return 'wf.exec()'

    def _gen_flat_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate flat_atomic execute() body.

        If the operation is recognized, emits a full VectorMemState setup
        with AtomicOp for the pipeline. Unrecognized variants (FP atomics,
        etc.) fall back to a TODO stub.
        """
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled flat_atomic variant ({sem.name})'

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1  # number of dwords of operand data

        L = []
        sc0, sc1, nt = self._coherency_exprs()
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        acc = self._acc_vgpr_expr
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = {self._atomic_return_expr(sc0)};')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'flat_atomic')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        data_field = self.isa_spec.profile.flat_store_src_field
        data_base = self._vgpr_base_expr(data_field)
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f'  uint32_t data_base = {data_base};')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(
                f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);'
            )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate buffer_atomic execute() body (MUBUF encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled buffer_atomic variant ({sem.name})'

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1

        L = []
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdata')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = {self._atomic_return_expr(sc0)};')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'buffer_atomic')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  mubuf_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append(f'  uint64_t exec = {self._buffer_payload_exec_expr()};')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('vdata')};")
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(
                f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);'
            )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_atomic execute() body (DS encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant ({sem.name})'

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1
        returns_data = 'vdst' in dst

        L = []
        is_cmpswap = sem.operation == 'cmpswap'
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        if returns_data:
            L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = {str(returns_data).lower()};')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'ds_atomic')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            f"  uint32_t data_base = {self._vgpr_base_expr('data0', role='Src1')};"
        )
        if is_cmpswap:
            L.append(
                f"  uint32_t data1_base = {self._vgpr_base_expr('data1', role='Src2')};"
            )
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        half = data_dwords // 2
        for i in range(data_dwords):
            if is_cmpswap and i >= half:
                data_source = f'data1_base + {i - half}'
            else:
                data_source = f'data_base + {i}'
            L.append(
                f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr({data_source}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);'
            )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_atomic2(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate a returning two-address LDS exchange."""
        esz = sem.elem_size or 4
        dwords_per_access = esz // 4
        is_stride64 = 'stride64' in sem.name.lower()
        stride_scale = f'{esz * 64}U' if is_stride64 else f'{esz}U'

        L = []
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = true;')
        L.append('  d->atomic_op = amdgpu::AtomicOp::SWAP;')
        self._append_wait_counter_type(L, 'ds_atomic2')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->wf_size = wf.wf_size();')
        L.append('  d->ds2_active = true;')
        L.append(
            f"  d->ds2_dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')} + "
            f'{dwords_per_access};'
        )
        L.append(f'  d->store_data.resize(wf.wf_size() * {esz});')
        L.append(f'  d->ds2_store_data.resize(wf.wf_size() * {esz});')
        L.append(
            f"  uint32_t addr_base = {self._vgpr_base_expr('addr', use_acc=False)};"
        )
        L.append(
            f"  uint32_t data0_base = {self._vgpr_base_expr('data0', role='Src1')};"
        )
        L.append(
            f"  uint32_t data1_base = {self._vgpr_base_expr('data1', role='Src2')};"
        )
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t base = amdgpu::RegisterAccess(cu).read_vgpr(addr_base, lane);'
        )
        L.append(
            f'    d->per_lane_addr[lane] = base + '
            f'static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();'
        )
        L.append(
            f'    d->ds2_per_lane_addr[lane] = base + '
            f'static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();'
        )
        for i in range(dwords_per_access):
            L.append(
                f'    uint32_t val0_{i} = '
                f'amdgpu::RegisterAccess(cu).read_vgpr(data0_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {esz} + {i * 4}], '
                f'&val0_{i}, 4);'
            )
            L.append(
                f'    uint32_t val1_{i} = '
                f'amdgpu::RegisterAccess(cu).read_vgpr(data1_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->ds2_store_data[lane * {esz} + {i * 4}], '
                f'&val1_{i}, 4);'
            )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_mskor(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_mskor execute() body (DS encoding)."""
        op_enum = self._ATOMIC_OP_ENUM['mskor']
        esz = sem.elem_size or 4
        dwords_per_operand = esz // 4
        is_rtn = 'RTN' in sem.name.upper()

        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        if is_rtn:
            L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = {str(is_rtn).lower()};')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'ds_mskor')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            f"  uint32_t mask_base = {self._vgpr_base_expr('data0', role='Src1')};"
        )
        L.append(f"  uint32_t src_base = {self._vgpr_base_expr('data1', role='Src2')};")
        stride = esz * 2
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(dwords_per_operand):
            L.append(
                f'    uint32_t mask{i} = amdgpu::RegisterAccess(cu).read_vgpr(mask_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &mask{i}, 4);'
            )
        for i in range(dwords_per_operand):
            L.append(
                f'    uint32_t src{i} = amdgpu::RegisterAccess(cu).read_vgpr(src_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {esz + i * 4}], &src{i}, 4);'
            )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_append_consume(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_append/ds_consume execute() body (DS encoding)."""
        if sem.operation not in ('append', 'consume'):
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_append_consume variant ({sem.name})'
        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append('  d->elem_size = 4;')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = true;')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'ds_append_consume')
        L.append('  uint64_t exec = wf.exec();')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->wg_id = wf.wg_id();')
        L.append('  d->wf_id = wf.wf_id();')
        L.append('  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);')
        L.append('  uint32_t addr = wf.lds_base() + wf.m0() + offset;')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (exec & (1ULL << lane))')
        L.append('      d->per_lane_addr[lane] = addr;')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_barrier_arrive(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate DS barrier-arrive execute() body (DS encoding)."""
        is_async = sem.operation == 'async_barrier_arrive'
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        if not is_async:
            L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append('  d->elem_size = 8;')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = {str(not is_async).lower()};')
        L.append('  d->atomic_op = amdgpu::AtomicOp::BARRIER_ARRIVE;')
        self._append_wait_counter_type(
            L, 'ds_barrier_arrive_async' if is_async else 'ds_barrier_arrive'
        )
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        if not is_async:
            L.append('  auto &cu = wf.cu();')
            L.append('  uint64_t exec = wf.exec();')
            L.append(
                f"  uint32_t data_base = {self._vgpr_base_expr('data0', role='Src1')};"
            )
            L.append('  d->store_data.resize(wf.wf_size() * 8);')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            L.append(
                '    uint32_t lo = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
            )
            L.append(
                '    uint32_t hi = amdgpu::RegisterAccess(cu).read_vgpr(data_base + 1, lane);'
            )
            L.append('    std::memcpy(&d->store_data[lane * 8], &lo, 4);')
            L.append('    std::memcpy(&d->store_data[lane * 8 + 4], &hi, 4);')
            L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_load(
        self,
        dst: list[str],
        src: list[str],
        sem: InstructionSemantics,
        cls: str = 'buffer_load',
        inst: 'Instruction | None' = None,
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        addr_fn = (
            'mtbuf_calculate_addresses'
            if cls == 'tbuffer_load'
            else 'mubuf_calculate_addresses'
        )
        # Check if this encoding has an 'lds' field in the machine instruction.
        has_lds_field = False
        if inst is not None:
            enc = self.isa_spec.encoding_map.get(inst.enc_name)
            if enc is not None:
                has_lds_field = any(f.name == 'lds' for f in enc.ucode_fields)
        # When the LDS bit is set, the buffer load reads from global memory but
        # writes the result to LDS at M0 + lane_offset instead of to VGPRs.
        # Route through the global memory pipeline for the load, then let the
        # pipeline writeback path detect lds_dst and scatter to LDS.
        if has_lds_field:
            L.append('  if (inst_.lds) {')
            L.append(
                '    auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
            )
            L.append(f'    d->elem_size = {esz};')
            L.append(f'    d->num_elems = {ne};')
            L.append('    d->is_load = true;')
            counter = self._wait_counter_type(cls)
            if counter is not None:
                L.append(f'    d->wait_counter_type = {counter};')
            L.append('    d->lds_dst = true;')
            L.append('    d->lds_base = wf.m0() + inst_.offset + wf.lds_base();')
            L.append(f'    d->mtype = {self._mtype_expr()};')
            L.append(f'    d->non_temporal = {nt};')
            L.append(f'    {addr_fn}(inst_, wf, *d);')
            L.append('    set_data(std::move(d));')
            L.append('    return;')
            L.append('  }')
        acc = self._acc_vgpr_expr
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdata')};")
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, cls)
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        if sem.d16_hi:
            L.append('  d->d16_hi = true;')
        if sem.d16_lo:
            L.append('  d->d16_lo = true;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append(f'  {addr_fn}(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_store(
        self,
        dst: list[str],
        src: list[str],
        sem: InstructionSemantics,
        cls: str = 'buffer_store',
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        addr_fn = (
            'mtbuf_calculate_addresses'
            if cls == 'tbuffer_store'
            else 'mubuf_calculate_addresses'
        )
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, cls)
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append(f'  {addr_fn}(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append(f'  uint64_t exec = {self._buffer_payload_exec_expr()};')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('vdata')};")
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz >= 4:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {i}, lane);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, {esz});'
                )
            elif esz == 2:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});'
                )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'ds_read')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        if sem.d16_hi:
            L.append('  d->d16_hi = true;')
        if sem.d16_lo:
            L.append('  d->d16_lo = true;')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _append_ds_addtid_addresses(self, lines: list[str]) -> None:
        """Emit the profile-specific DS ADDTID per-lane address calculation."""
        lines.append('  {')
        lines.append('    uint64_t exec = wf.exec();')
        lines.append('    d->lane_mask = exec; d->exec_mask = exec;')
        lines.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        lines.append(
            '    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;'
        )
        lines.append('    uint32_t m0 = wf.m0();')
        if self.isa_spec.profile.ds_addtid_uses_m0_byte_base:
            lines.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            lines.append('      if (!(exec & (1ULL << lane))) continue;')
            lines.append('      uint32_t lane_offset = (m0 + lane * 4U) & 0xFFFFFU;')
            lines.append(
                '      d->per_lane_addr[lane] = lane_offset + offset + wf.lds_base();'
            )
        else:
            lines.append('    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;')
            lines.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            lines.append('      if (!(exec & (1ULL << lane))) continue;')
            lines.append(
                '      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();'
            )
        lines.append('    }')
        lines.append('  }')

    def _gen_ds_read_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate profile-specific DS ADDTID load addressing."""
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'ds_read_addtid')
        self._append_ds_addtid_addresses(L)
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate profile-specific DS ADDTID store addressing."""
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'ds_write_addtid')
        self._append_ds_addtid_addresses(L)
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('data0')};")
        L.append(f'  d->store_data.resize(wf.wf_size() * {sem.elem_size});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            f'    uint32_t val0 = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
        )
        L.append(
            f'    std::memcpy(&d->store_data[lane * {sem.elem_size}], &val0, {sem.elem_size});'
        )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read_tr(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """DS read + cross-lane transpose post-processing.

        Uses the standard DS read pipeline (MEMORY_OP). Sets d->transpose to
        signal the memory pipeline to apply the cross-lane shuffle after the
        raw read.
        """
        # amdgpu::TransposeKind: TR_B4=1, TR_B6=2, B64_TR_B8=3,
        # TR16_B128=4, B64_TR_B16=5, WMMA_TR_B8=6,
        # CDNA5_DS_TR_B8=7. Defaults describe the B64 (num_elems=2) forms.
        tr_map = {
            'ds_read_tr_b4': (4, 2, 1),  # elem_size=4, num_elems=2, transpose=1
            'ds_read_tr_b6': (4, 3, 2),  # elem_size=4, num_elems=3, transpose=2
            'ds_read_tr_b8': (4, 2, 3),  # elem_size=4, num_elems=2, transpose=3
            'ds_read_tr_b16': (4, 2, 5),  # elem_size=4, num_elems=2, transpose=5
        }
        default_esz, default_ne, default_tr_kind = tr_map.get(
            sem.semantic_class, (4, 2, 5)
        )
        esz = sem.elem_size if sem.elem_size is not None else default_esz
        ne = sem.num_elems if sem.num_elems is not None else default_ne
        tr_kind = getattr(sem, 'transpose_kind', 0) or default_tr_kind
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, sem.semantic_class)
        L.append(f'  d->transpose = {tr_kind};')
        address_helper = (
            'ds_calculate_addresses_all_lanes'
            if self.isa_spec.profile.ds_transpose_ignores_exec
            else 'ds_calculate_addresses'
        )
        L.append(f'  {address_helper}(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'ds_write')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('data0')};")
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            off = i * esz
            if esz == 8:
                vgpr_base = i * 2
                L.append(
                    f'    uint32_t lo{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {vgpr_base}, lane);'
                )
                L.append(
                    f'    uint32_t hi{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {vgpr_base + 1}, lane);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &lo{i}, 4);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off + 4}], &hi{i}, 4);'
                )
            elif esz == 4:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base + {i}, lane);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 4);'
                )
            elif esz == 2:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(
                    f'    uint32_t val{i} = amdgpu::RegisterAccess(cu).read_vgpr(data_base, lane);'
                )
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    d->store_data[lane * {stride} + {off}] = static_cast<uint8_t>(val{i});'
                )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read2(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_read2 execute body: two independent LDS loads.

        DS_READ2_B32:  vdst[31:0]  = LDS[addr + offset0*4]
                       vdst[63:32] = LDS[addr + offset1*4]
        DS_READ2ST64:  same but offsets scaled by 256 instead of 4.
        B64 variants:  read 8 bytes per access (two dwords each).

        Uses VectorMemState ds2 fields to package both accesses into a
        single pipeline request.
        """
        L = []
        esz = sem.elem_size  # 4 for B32, 8 for B64
        dwords_per_access = esz // 4  # 1 for B32, 2 for B64
        if sem.operation == 'st64':
            stride_scale = f'{esz * 64}U'
        else:
            stride_scale = f'{esz}U'
        acc = self._acc_vgpr_expr
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'ds_read2')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->ds2_active = true;')
        L.append(
            f"  d->ds2_dst_reg_base = {self._vgpr_base_expr('vdst')} + {dwords_per_access};"
        )
        L.append(
            f"  uint32_t addr_base = {self._vgpr_base_expr('addr', use_acc=False)};"
        )
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t base = amdgpu::RegisterAccess(cu).read_vgpr(addr_base, lane);'
        )
        L.append(
            f'    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();'
        )
        L.append(
            f'    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();'
        )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write2(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_write2 execute body: two independent LDS stores.

        DS_WRITE2_B32:  LDS[addr + offset0*4] = data0
                        LDS[addr + offset1*4] = data1
        DS_WRITE2ST64:  same but offsets scaled by 256 instead of 4.
        B64 variants:   write 8 bytes per access (two dwords each).

        Uses VectorMemState ds2 fields to package both accesses into a
        single pipeline request.
        """
        L = []
        esz = sem.elem_size  # 4 for B32, 8 for B64
        dwords_per_access = esz // 4
        if sem.operation == 'st64':
            stride_scale = f'{esz * 64}U'
        else:
            stride_scale = f'{esz}U'
        acc = self._acc_vgpr_expr
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'ds_write2')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->ds2_active = true;')
        L.append(f'  d->store_data.resize(wf.wf_size() * {esz});')
        L.append(f'  d->ds2_store_data.resize(wf.wf_size() * {esz});')
        L.append(
            f"  uint32_t addr_base = {self._vgpr_base_expr('addr', use_acc=False)};"
        )
        L.append(f"  uint32_t data0_base = {self._vgpr_base_expr('data0')};")
        L.append(f"  uint32_t data1_base = {self._vgpr_base_expr('data1')};")
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t base = amdgpu::RegisterAccess(cu).read_vgpr(addr_base, lane);'
        )
        L.append(
            f'    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();'
        )
        L.append(
            f'    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();'
        )
        # Pack data0 into store_data
        for i in range(dwords_per_access):
            L.append(
                f'    uint32_t v0_{i} = amdgpu::RegisterAccess(cu).read_vgpr(data0_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->store_data[lane * {esz} + {i * 4}], &v0_{i}, 4);'
            )
        # Pack data1 into ds2_store_data
        for i in range(dwords_per_access):
            L.append(
                f'    uint32_t v1_{i} = amdgpu::RegisterAccess(cu).read_vgpr(data1_base + {i}, lane);'
            )
            L.append(
                f'    std::memcpy(&d->ds2_store_data[lane * {esz} + {i * 4}], &v1_{i}, 4);'
            )
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _enc_has_field(self, field_name: str) -> bool:
        """Check if the current encoding struct has a named field.

        Uses the struct field names from the machine instruction encoding.
        Falls back to checking _current_inst_fields (ucode_fields + parent
        fields) and the encoding name for known patterns.
        """
        if hasattr(self, '_current_inst_fields') and self._current_inst_fields:
            if field_name in self._current_inst_fields:
                return True
        if field_name == 'gds' and hasattr(self, '_current_enc') and self._current_enc:
            return self._current_enc.enc_name.upper() == 'ENC_DS'
        return False

    def _enc_has_semantics(self, enc: InstEncoding) -> bool:
        """Check if any instruction in this encoding has semantics."""
        if not self.semantics:
            return False
        for inst in enc.insts:
            if inst.name in self.semantics.instructions:
                return True
        return False

    # Semantic classes whose execute() bodies reference ISA-profile-specific
    # code (mtype_from_flags, coherency fields, addr_calc, etc.).
    # These cannot have shared execute templates.
    _NON_SHAREABLE_CLASSES = frozenset(
        {
            # Profile-dependent (ISA-specific coherency/mtype calls):
            'smem_load',
            'smem_store',
            'flat_load',
            'flat_store',
            'flat_atomic',
            'buffer_load',
            'buffer_store',
            'buffer_atomic',
            'tbuffer_load',
            'tbuffer_store',
            'ds_read',
            'ds_read2',
            'ds_write',
            'ds_write2',
            'ds_atomic',
            'ds_atomic2',
            'ds_mskor',
            'ds_append_consume',
            'ds_barrier_arrive',
            'global_load',
            'global_store',
            'global_load_addtid',
            'global_store_addtid',
            'global_load_async_to_lds',
            'global_store_async_from_lds',
            'dcache_inv',
            'dcache_wb',
            'image_load',
            'image_store',
            'image_atomic',
            'image_sample',
            'image_query',
            # Nop/stub bodies don't benefit from sharing:
            'nop',
            # ISA-dependent control flow (reference Isa:: constants or size_):
            'waitcnt',
            'wait_counter',
            'endpgm',
            'branch',
            'cbranch',
            'scalar_getpc',
            'scalar_setpc',
            'scalar_swappc',
            'scalar_addpc',
            'scalar_call',
            'scalar_movrel',
            # HWREG IDs and helper mappings are profile-specific.
            'scalar_getreg',
            'scalar_setreg',
            # S_SETREG_IMM32_B32 shares the profile-specific HWREG handling.
            'scalar_setreg_imm',
            # MFMA/WMMA reference ISA-specific headers:
            'mfma',
            # Interp/export use ISA-specific encoding struct fields:
            'interp',
            'export',
            # AccVGPR read/write use ISA-specific register file:
            'accvgpr_read',
            'accvgpr_write',
            # Vector swap accesses protected inst_ member:
            'vector_swap',
            # Movrel uses ISA-local OperandType/Isa helpers.
            'vector_movrel',
            'vector_swaprel',
            'vector_perm_pk16',
            'vector_qsad',
            # Vector readlane/writelane/readfirstlane access encoding fields:
            'vector_readlane',
            'vector_writelane',
            'vector_readfirstlane',
            # V_PERMLANE op_sel fields are profile-specific (op_sel vs opsel).
            'vector_permlane16',
            'vector_permlanex16',
            # V_CMPX writes VCC+EXEC on CDNA but only EXEC on RDNA:
            'vector_cmpx',
            'vector_cmpx_class',
            # FP8/BF8 conversions access inst_.op_sel (VOP3-only field):
            'cvt_fp8',
            'cvt_scalef32',
            # vector_cvt_pk FP8 pack/unpack uses inst_.op_sel for word sel:
            'vector_cvt_pk',
        }
    )

    def _requires_arch_local_execute(
        self, inst: Instruction | None, enc_name: str | None = None
    ) -> bool:
        if inst is None:
            return False
        profile = getattr(self.isa_spec, 'profile', None)
        uses_true16_e32 = bool(
            getattr(profile, 'uses_packed_16bit_e32_source_selectors', False)
        )
        enc_upper = (enc_name or inst.enc_name).upper()
        if (
            uses_true16_e32
            and enc_upper == 'ENC_VOP1'
            and inst.name == 'V_MOV_B16'
            and any(op.size == 16 for op in inst.operands)
        ):
            return True
        if (
            uses_true16_e32
            and enc_upper in ('ENC_VOP1', 'ENC_VOP2')
            and any(op.is_output and op.size == 16 for op in inst.operands)
        ):
            return True
        if self._uses_true16_vop3_execute(inst, enc_name):
            return True
        return False

    def _uses_true16_vop3_execute(
        self, inst: Instruction | None, enc_name: str | None = None
    ) -> bool:
        if inst is None:
            return False
        profile = getattr(self.isa_spec, 'profile', None)
        if not bool(getattr(profile, 'uses_true16_vop3_opsel', False)):
            return False
        enc_upper = (enc_name or inst.enc_name).upper()
        if enc_upper != 'ENC_VOP3':
            return False
        return any(
            (op.is_input or op.is_output) and op.size == 16 for op in inst.operands
        )

    def _true16_vop3_local_simd_probe(
        self,
        inst: Instruction | None,
        sem: InstructionSemantics | None = None,
        enc_name: str | None = None,
        result_writer: str | None = None,
    ) -> str | None:
        if sem is not None and inst is not None:
            uses_true16_probe = self._true16_vop3_info(
                inst, sem, enc_name or inst.enc_name
            ).enabled
        else:
            uses_true16_probe = self._uses_true16_vop3_execute(inst, enc_name)
        if not uses_true16_probe:
            return None

        from amdisa.codegen.execute.simd_codegen import simd_probe_line

        enc_key = (enc_name or inst.enc_name).lower().replace('enc_', '')
        return simd_probe_line(
            f'{inst.mnemonic}_{enc_key}',
            true16_vop3=True,
            result_writer=result_writer,
        )

    def _renamed_vop3p_local_simd_probe(
        self,
        inst: Instruction | None,
        enc_name: str | None = None,
    ) -> str | None:
        if inst is None:
            return None
        enc_key = (enc_name or inst.enc_name).lower().replace('enc_', '')
        if enc_key != 'vop3p':
            return None

        from amdisa.codegen.execute.simd_codegen import (
            vop3p_local_simd_probe_line,
        )

        return vop3p_local_simd_probe_line(
            f'{inst.mnemonic}_{enc_key}',
            self.isa_spec.profile.vop3p_opsel_fields,
            self._op_sel_hi_2_expr(enc_name or inst.enc_name),
        )

    def _e32_true16_dst_reg_expr(
        self, inst: Instruction | None, enc_name: str | None = None
    ) -> str:
        if inst is None:
            return 'inst_.vdst'
        profile = getattr(self.isa_spec, 'profile', None)
        if not bool(getattr(profile, 'uses_packed_16bit_e32_source_selectors', False)):
            return 'inst_.vdst'
        enc_upper = (enc_name or inst.enc_name).upper()
        if enc_upper not in ('ENC_VOP1', 'ENC_VOP2'):
            return 'inst_.vdst'
        if any(
            op.is_output and op.size == 16 and op.operand_type == 'OPR_VGPR'
            for op in inst.operands
        ):
            return '(inst_.vdst & 0x7fu)'
        return 'inst_.vdst'

    def _shared_execute_key_denied(
        self, mnemonic: str, inst: Instruction | None, enc_name: str | None = None
    ) -> bool:
        enc_key = enc_name or (inst.enc_name if inst else None)
        if enc_key is None:
            return False
        config = getattr(self, 'config', None)
        denied = getattr(config, 'unshared_execute_keys', frozenset())
        return (mnemonic, enc_key) in denied

    def _can_share_execute(
        self,
        mnemonic: str,
        inst: Instruction | None = None,
        enc_name: str | None = None,
    ) -> bool:
        """Check if an instruction's execute() body can be shared across ISAs.

        An instruction is shareable if:
        1. It exists on 2+ ISAs with the same semantic class (family_shared
           or universal in the shared_plan).
        2. Its semantic class is profile-independent (no mtype/coherency calls).
        3. The current ISA is one of the ISAs that share this instruction.
        """
        if self.shared_plan is None:
            return False
        if self._requires_arch_local_execute(inst, enc_name):
            return False
        if self._shared_execute_key_denied(mnemonic, inst, enc_name):
            return False
        arch = self.isa_spec.arch_name
        # Check universal
        if mnemonic in self.shared_plan.universal:
            info = self.shared_plan.universal[mnemonic]
            if info.semantic_class in self._NON_SHAREABLE_CLASSES:
                return False
            return arch in info.isa_names and len(info.isa_names) >= 2
        # Check family_shared — keyed by (mnemonic, encoding_name) tuples.
        # A mnemonic may appear in multiple families and with different
        # encodings. Match the full (mnemonic, encoding) key when the caller
        # specifies an encoding. When enc_name is None the caller does not
        # constrain the encoding, so any entry for the mnemonic is considered.
        for fam_insts in self.shared_plan.family_shared.values():
            for (mn, entry_enc), info in fam_insts.items():
                if mn != mnemonic:
                    continue
                if enc_name is not None and entry_enc != enc_name:
                    continue
                if info.semantic_class in self._NON_SHAREABLE_CLASSES:
                    return False
                if arch in info.isa_names and len(info.isa_names) >= 2:
                    return True
        return False

    def _can_force_shared_simd_probe(
        self, inst: Instruction | None, enc_name: str | None = None
    ) -> bool:
        if not self.config.use_shared_execute_helpers:
            return False
        if inst is None or self._requires_arch_local_execute(inst, enc_name):
            return False
        if self._shared_execute_key_denied(inst.mnemonic, inst, enc_name):
            return False

        from amdisa.codegen.execute.simd_codegen import simd_probe_arch_portable

        enc_key = (enc_name or inst.enc_name).lower().replace('enc_', '')
        return simd_probe_arch_portable(
            f'{inst.mnemonic}_{enc_key}',
            self.isa_spec.profile.vop3p_opsel_fields,
        )

    @staticmethod
    def _fold_acc_bank_selector(ctx: _OperandCtx, expr: str) -> str:
        """Fold separate CDNA accumulator-bank fields into an operand value."""
        enc_name = ctx.enc_name.upper()
        acc_bank_encodings = {
            'ENC_DS',
            'ENC_MUBUF',
            'ENC_MTBUF',
            'ENC_FLAT',
            'ENC_FLAT_GLBL',
            'ENC_FLAT_SCRATCH',
            'ENC_MIMG',
        }
        if (
            enc_name in acc_bank_encodings
            and ctx.has_acc_field
            and ctx.operand_type == 'OPR_VGPR_OR_ACCVGPR'
        ):
            expr = (
                f'({expr} + (reinterpret_cast<const OpEncoding*>(inst)->acc ? '
                'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
            )
        elif enc_name in {'ENC_VOP3P', 'ENC_VOP3P_MFMA'} and ctx.has_acc_cd_field:
            if (
                ctx.opnd_name in {'src0', 'src1'}
                and ctx.operand_type == 'OPR_SRC_VGPR_OR_ACCVGPR'
                and ctx.has_acc_field
            ):
                acc_mask = '0x1u' if ctx.opnd_name == 'src0' else '0x2u'
                expr = (
                    f'({expr} + ((reinterpret_cast<const OpEncoding*>(inst)->acc & '
                    f'{acc_mask}) ? (OpSelSrcVgprOrAccvgpr::'
                    'OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN - '
                    'OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN) : 0))'
                )
            elif ctx.opnd_name == 'vdst' and ctx.operand_type == 'OPR_VGPR_OR_ACCVGPR':
                expr = (
                    f'({expr} + (reinterpret_cast<const OpEncoding*>(inst)->acc_cd ? '
                    'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
                )
            elif (
                ctx.opnd_name == 'src2'
                and ctx.operand_type == 'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST'
            ):
                expr = (
                    f'mfma_src2_encoding({expr}, '
                    'reinterpret_cast<const OpEncoding*>(inst)->acc_cd)'
                )
        # Dedicated AccVGPR types (accvgpr read/write/mov) are always accumulator
        # regs, but their raw field is below the canonical ACC selector range that
        # name()/to_register_ref() match. Canonicalize so those paths see the reg;
        # the value still resolves to the same unified index for execute.
        if ctx.operand_type == 'OPR_ACCVGPR':
            # vdst field holds the accumulator number directly (0..255).
            expr = f'({expr} + OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN)'
        elif ctx.operand_type == 'OPR_SRC_ACCVGPR':
            # 9-bit source field encodes acc N as 256 + N; shift into [768, 1023].
            # Only shift a well-formed field (>= 256); a raw < 256 would escape into
            # [512, 767], which vgpr_index() reads as an out-of-range physical VGPR.
            expr = (
                f'({expr} >= 256 ? '
                f'{expr} + (OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN - 256) : '
                f'{expr})'
            )
        return expr

    @classmethod
    def _operand_encoding_value_expr(cls, ctx: _OperandCtx) -> str:
        """C++ expression for the decoded value passed to an Operand constructor.

        For most operands this is just the value. SMEM SBASE and legacy buffer
        SRSRC fields are encoded in register groups, so this helper scales them
        to the real SGPR index. CDNA memory encodings also carry their AccVGPR
        bank selection in a separate ``acc`` bit, so fold that bit into the
        selector passed to OPR_VGPR_OR_ACCVGPR. MFMA's two ``acc`` bits select
        the AccVGPR bank for its A/B inputs, and ``acc_cd`` selects it for the
        C/D operands. Folding A/B is deliberately gated on ``acc_cd`` so CDNA1,
        which has ``acc`` but not ``acc_cd``, keeps its raw VGPR-only selectors.
        This keeps the operand's register-ref (disassembly, def/use, liveness)
        consistent with execution, which applies these transformations
        independently.
        """
        expr = f'reinterpret_cast<const OpEncoding*>(inst)->{ctx.opnd_name}'
        enc_name = ctx.enc_name.upper()
        if enc_name == 'ENC_SMEM' and ctx.opnd_name == 'sbase':
            expr = f'({expr} * 2)'
        elif enc_name in ('ENC_MUBUF', 'ENC_MTBUF') and ctx.opnd_name == 'srsrc':
            expr = f'({expr} * 4)'
        expr = cls._fold_acc_bank_selector(ctx, expr)
        if ctx.packed_16bit:
            expr = f'static_cast<unsigned short>({expr})'
        return expr

    def _fieldless_canonical_value(self, operand_type: str) -> int:
        """Canonical fixed encoding value for a fieldless operand type.

        Fieldless operands have no encoding field to decode, so they are built
        from a fixed value. That value is the minimum selector value for the
        type (single-valued selectors -> that value; EXEC LO/HI -> LO; register
        ranges -> MIN), which is exactly what the generated ``name()`` /
        ``to_register_ref()`` switch cases gate on (e.g. OPR_PC -> OPR_PC_PC_ALL,
        OPR_VCC -> OPR_VCC_VCC, OPR_SDST_EXEC -> OPR_SDST_EXEC_EXEC_LO). Computed
        from the ISA's own selectors rather than a hand-maintained table; falls
        back to 0 for types without a selector.
        """
        cache = self._fieldless_canon_cache
        if cache is None:
            cache = {}
            for sel in self.isa_spec.opnd_selectors:
                # Selector values are strings; accept decimal and 0x/0o/0b
                # forms, and skip any non-numeric (symbolic) value rather than
                # crashing the whole generator on an unexpected spelling.
                vals = []
                for _, v in sel.op_sel_vals:
                    try:
                        vals.append(int(v, 0))
                    except (TypeError, ValueError):
                        continue
                if vals:
                    cache[sel.operand_type] = min(vals)
            self._fieldless_canon_cache = cache
        return cache.get(operand_type, 0)

    def _operand_selector_intervals(self, operand_type: str) -> list[tuple[int, int]]:
        """Return merged numeric intervals declared for an operand type."""
        cached = self._selector_interval_cache.get(operand_type)
        if cached is not None:
            return cached

        selector = next(
            (
                item
                for item in self.isa_spec.opnd_selectors
                if item.operand_type == operand_type
            ),
            None,
        )
        if selector is None:
            raise ValueError(f'{operand_type}: operand selector is not defined')

        enum_values = {}
        for name, value in selector.op_sel_vals:
            try:
                enum_values[name] = int(value, 0)
            except (TypeError, ValueError):
                continue

        intervals = [(value, value) for value in enum_values.values()]
        for pattern in selector.name_patterns:
            if pattern.kind not in (
                OperandNamePattern.REG_RANGE,
                OperandNamePattern.POS_INT,
                OperandNamePattern.NEG_INT,
            ):
                continue
            lo = enum_values[pattern.min_enum]
            hi = enum_values[pattern.max_enum]
            intervals.append((min(lo, hi), max(lo, hi)))

        merged: list[tuple[int, int]] = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        self._selector_interval_cache[operand_type] = merged
        return merged

    def _validate_shared_scalar_pair_selector_contract(self) -> None:
        """Validate selector values consumed by the shared scalar-pair resolver."""
        selector = next(
            (
                item
                for item in self.isa_spec.opnd_selectors
                if item.operand_type == 'OPR_SSRC'
            ),
            None,
        )
        if selector is None:
            return

        enum_values: dict[str, int] = {}
        for name, value in selector.op_sel_vals:
            try:
                enum_values[name] = int(value, 0)
            except (TypeError, ValueError):
                continue

        arch = self.isa_spec.arch_name

        def require(name: str, expected: int) -> None:
            actual = enum_values.get(name)
            if actual != expected:
                raise ValueError(
                    f'{arch}: shared scalar-pair selector contract requires '
                    f'{name}={expected}, got {actual!r}'
                )

        require('OPR_SSRC_SGPR_MIN', 0)
        require('OPR_SSRC_VCC_LO', 106)
        require('OPR_SSRC_VCC_HI', 107)
        require('OPR_SSRC_EXEC_LO', 126)
        require('OPR_SSRC_EXEC_HI', 127)

        # The shared runtime decoder supports the three layouts present in the
        # MRISA corpus: CDNA1-4 have M0 at 124 and no NULL selector, RDNA1/2
        # have M0 at 124 and NULL at 125, and GFX11+ have the two reversed.
        m0_and_null = (
            enum_values.get('OPR_SSRC_M0'),
            enum_values.get('OPR_SSRC_NULL'),
        )
        if m0_and_null not in ((124, None), (124, 125), (125, 124)):
            raise ValueError(
                f'{arch}: unsupported shared M0/NULL selector layout '
                f'{m0_and_null!r}'
            )

        sgpr_max = enum_values.get('OPR_SSRC_SGPR_MAX')
        if sgpr_max not in (101, 105):
            raise ValueError(
                f'{arch}: shared scalar-pair selector contract requires '
                f'OPR_SSRC_SGPR_MAX to be 101 or 105, got {sgpr_max!r}'
            )
        if sgpr_max == 101:
            require('OPR_SSRC_FLAT_SCRATCH_LO', 102)
            require('OPR_SSRC_FLAT_SCRATCH_HI', 103)
            require('OPR_SSRC_XNACK_MASK_LO', 104)
            require('OPR_SSRC_XNACK_MASK_HI', 105)

        pair_block_starts = (
            'OPR_SSRC_TTMP_MIN',
            'OPR_SSRC_TTMP0',
            'OPR_SSRC_TBA_LO',
        )
        if not any(enum_values.get(name) == 108 for name in pair_block_starts):
            raise ValueError(
                f'{arch}: shared scalar-pair selector contract requires a '
                'TTMP or TBA/TMA register block starting at 108'
            )
        pair_block_ends = ('OPR_SSRC_TTMP_MAX', 'OPR_SSRC_TTMP15')
        if not any(enum_values.get(name) == 123 for name in pair_block_ends):
            raise ValueError(
                f'{arch}: shared scalar-pair selector contract requires the '
                'TTMP register block to end at 123'
            )

        flat_base_names = (
            'OPR_SSRC_SRC_FLAT_SCRATCH_BASE_LO',
            'OPR_SSRC_SRC_FLAT_SCRATCH_BASE_HI',
        )
        if any(name in enum_values for name in flat_base_names):
            require(flat_base_names[0], 230)
            require(flat_base_names[1], 231)

        intervals = self._operand_selector_intervals('OPR_SSRC')
        missing = sorted(
            value
            for value in self._SHARED_SCALAR_PAIR_VALUES
            if not any(lo <= value <= hi for lo, hi in intervals)
        )
        if missing:
            raise ValueError(
                f'{arch}: OPR_SSRC is missing shared scalar-pair selector '
                f'values {missing}'
            )

    def _operand_selector_contains(self, operand_type: str, value: int) -> bool | None:
        """Whether a selector declares ``value``, or None for a non-selector type."""
        try:
            intervals = self._operand_selector_intervals(operand_type)
        except ValueError:
            return None
        return any(lo <= value <= hi for lo, hi in intervals)

    def _instruction_encoding_field_names(self, inst: Instruction) -> set[str]:
        """Return fields visible to the generated constructor for ``inst``."""
        inst_enc = self.isa_spec.encoding_map.get(inst.enc_name)
        if inst_enc is None:
            return set()

        profile = self.isa_spec.profile
        if not profile.is_alt_encoding(inst.enc_name):
            return {field.name for field in inst_enc.ucode_fields}

        parent_name = profile.derive_parent_enc_name(inst.enc_name)
        parent = self.isa_spec.encoding_map.get(parent_name)
        fields = {field.name for field in parent.ucode_fields} if parent else set()
        if not inst.is_implied_literal_enc:
            fields.update(field.name for field in inst_enc.ucode_fields)
        return fields

    def _selector_constructors_use_canonical_values(self, operand_type: str) -> bool:
        """Whether every generated constructor value is in the selector namespace.

        Encoding fields are passed through ``_operand_encoding_value_expr``, which
        maps grouped register fields and accumulator-bank bits to canonical selector
        values. Fieldless operands use a canonical fixed value. A missing,
        non-fieldless operand is initialized with zero until instruction-specific
        code replaces it; such a placeholder is safe to validate only when zero is
        itself declared by the selector.
        """
        if operand_type in self._NON_CANONICAL_SELECTOR_OPERAND_TYPES:
            return False

        zero_is_valid = self._operand_selector_contains(operand_type, 0)
        if zero_is_valid is None:
            return False

        for enc in self.isa_spec.inst_encodings:
            for inst in enc.insts:
                inst_sem = (
                    self.semantics.instructions.get(inst.name)
                    if self.semantics
                    else None
                )
                field_names = self._instruction_encoding_field_names(inst)
                for opnd in inst.operands:
                    if self._constructor_operand_type(inst_sem, opnd) != operand_type:
                        continue
                    if opnd.name in field_names or opnd.fieldless:
                        continue
                    if not zero_is_valid:
                        return False
        return True

    def gen_insts(self) -> None:
        """Generate instruction classes deriving from encoding classes.

        When ``shared_plan`` is set, universal instructions
        are emitted into ``shared/<enc>.h/.cpp`` in the ``rocjitsu::amdgpu``
        namespace.  Per-ISA files include the shared header and emit
        ``using amdgpu::<ClassName>;`` aliases for universals, plus full
        definitions for ISA-exclusive instructions.

        Instructions from alternate sub-encodings (Category 1: VOP3_SDST_ENC,
        VOP3P_MFMA) are generated in the parent encoding's file because
        the C++ build system lists only parent encoding source files.
        The ``encoding_map`` is used to resolve each instruction's actual
        encoding fields for correct struct member access.
        """
        decoder_factories = self._distributed_decoder_factories()

        # Build a mapping of parent encoding names to their child alt
        # encodings that have their own instructions (Category 1 alts).
        profile = self.isa_spec.profile
        child_encs: dict[str, list[InstEncoding]] = {}
        for enc in self.isa_spec.inst_encodings:
            if enc.insts and profile.is_alt_encoding(enc.enc_name):
                parent_name = profile.derive_parent_enc_name(enc.enc_name)
                child_encs.setdefault(parent_name, []).append(enc)

        for enc in self.isa_spec.inst_encodings:
            inst_classes = []
            class_func_impls = _ImplOutputs()
            source_impl_units = _ImplOutputs()
            # Collect instructions from this encoding plus any child
            # alt encodings that contribute to this file.
            all_insts = list(enc.insts)
            for child in child_encs.get(enc.enc_name, []):
                all_insts.extend(child.insts)
            if all_insts and not profile.is_alt_encoding(enc.enc_name):
                enc_field_names = {f.name for f in enc.ucode_fields}
                is_smem = enc.enc_name.upper() == 'ENC_SMEM'
                supports_sdwa_extension = self._supports_sdwa_for_encoding(enc.enc_name)
                has_sem = self._enc_has_semantics(enc)
                # Check child encodings for semantics too.
                for child in child_encs.get(enc.enc_name, []):
                    if self._enc_has_semantics(child):
                        has_sem = True
                for inst in all_insts:
                    inst_sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    # Resolve the instruction's own encoding field names.
                    # Instructions from alternate sub-encodings (e.g.,
                    # VOP3_SDST_ENC under ENC_VOP3) carry their original
                    # enc_name and inherit from the sub-encoding's C++
                    # class whose OpEncoding typedef matches the sub-
                    # encoding's MachineInst struct.  Look up the
                    # instruction's own encoding to get the correct field
                    # set.
                    inst_enc_obj = self.isa_spec.encoding_map.get(inst.enc_name)
                    # Implied-literal encodings reuse their parent VOP layout;
                    # true alternate formats such as VOP3_SDST_ENC retain
                    # their own DPP extension layout.
                    inst_dpp_enc_name = (
                        enc.enc_name if inst.is_implied_literal_enc else inst.enc_name
                    )
                    if (
                        inst_enc_obj is not None
                        and inst_enc_obj is not enc
                        and not inst.is_implied_literal_enc
                    ):
                        inst_field_names = enc_field_names | {
                            f.name for f in inst_enc_obj.ucode_fields
                        }
                    else:
                        inst_field_names = enc_field_names
                    class_members = []
                    public_members = [cgen.Line('public:')]
                    private_members = []
                    opnd_ctor_init = []
                    opnd_body = []
                    conditional_src_body = []
                    conditional_dst_body = []
                    # DPP/SDWA helpers index the architectural input operands
                    # from the front of src_operands_. On those encodings a
                    # read/write output listed before src0 in the XML would
                    # displace src0, so DPP would permute the old destination
                    # instead. Defer read/write outputs to the end there. On
                    # encodings without DPP/SDWA (e.g. SOPK) there is no such
                    # positional dependency, so keep the read/write output at its
                    # XML operand position to preserve the architectural source
                    # order (e.g. s_addk_i32/s_mulk_i32 read sdst as src0).
                    _enc_upper_for_defer = enc.enc_name.upper()
                    # V_PK_FMAC_F16 is the VOP2 exception: its accumulator is
                    # architecturally the first source, and its generated
                    # DPP/SDWA paths operate on the named src0/vsrc1 operands.
                    # Keep vdst first even when a snapshot models it as
                    # output-only and read/write inference supplies the use.
                    defer_readwrite_outputs = inst.name.upper() != 'V_PK_FMAC_F16' and (
                        _enc_upper_for_defer in ('ENC_VOP1', 'ENC_VOP2')
                        or (
                            _enc_upper_for_defer
                            in ('ENC_VOP3', 'ENC_VOP3P', 'VOP3_SDST_ENC')
                            and self._supports_vop_dpp_encoding(_enc_upper_for_defer)
                        )
                    )
                    readwrite_output_sources = []
                    vgpr_msb_role_body = []
                    src_idx = 0
                    dst_idx = 0
                    vgpr_msb_src_role_idx = 0
                    reads_dst = self._dst_is_also_source(inst)
                    # D16(_HI) loads on architectures without SRAM ECC read the
                    # destination they partially write. Model that preservation
                    # as an implicit use so vdst stays out of the printed operand
                    # list (see _d16_load_reads_dst). The override
                    # declaration/definition is emitted below, sharing the path
                    # with generic partial defs (buffer/tbuffer loads name the
                    # dest 'vdata' and are sized as a full 32-bit register, so
                    # _partial_def_outputs does not catch them).
                    d16_implicit_use_opnd = None
                    # Offset (in 32-bit registers) of the one partially-written
                    # register within the destination: only the last register of
                    # an odd-count FORMAT load (e.g. format_d16_xyz) is partial.
                    d16_partial_reg_offset = 0
                    if self._d16_load_reads_dst(inst):
                        # FLAT/GLOBAL/SCRATCH/DS name the dest 'vdst'; MUBUF/MTBUF
                        # (buffer/tbuffer) name it 'vdata'.
                        d16_implicit_use_opnd = next(
                            (
                                o.name
                                for o in inst.operands
                                if o.is_output and o.name in ('vdst', 'sdst', 'vdata')
                            ),
                            None,
                        )
                        # Fail loudly: _d16_load_reads_dst() matched, so a
                        # destination read must be modeled. A None here (e.g. a
                        # future dest rename) would silently skip both the
                        # implicit_uses declaration and definition below, losing
                        # the preserved-destination read from liveness.
                        assert d16_implicit_use_opnd is not None, (
                            f'{inst.name}: D16 load reads its destination, but no '
                            "'vdst'/'sdst'/'vdata' output operand was found to model "
                            'the preserved-destination read'
                        )
                        d16_sem = self.semantics.instructions.get(inst.name)
                        d16_total_bytes = d16_sem.num_elems * d16_sem.elem_size
                        d16_vgpr_count = (d16_total_bytes + 3) // 4  # round up
                        d16_partial_reg_offset = d16_vgpr_count - 1
                    # These gfx1250-only WMMA source-format fields derive the
                    # src0/src1 operand sizes from the instruction shape.
                    cdna5_f8f6f4_shape = self._cdna5_f8f6f4_wmma_shape(inst)
                    cdna4_f8f6f4_shape = self._cdna4_f8f6f4_mfma_shape(inst)
                    cdna5_swmmac_has_modifiers = self._cdna5_swmmac_has_modifiers(inst)
                    operand_size_exprs: dict[str, str] = {}
                    for opnd in inst.operands:
                        opnd_size_expr = self._operand_size_override(
                            enc.enc_name, opnd, inst_sem
                        )
                        if opnd_size_expr is None:
                            opnd_size_expr = self._cdna5_matrix_fmt_operand_size_expr(
                                cdna5_f8f6f4_shape, opnd.name
                            )
                        if opnd_size_expr is None:
                            opnd_size_expr = self._cdna4_matrix_fmt_operand_size_expr(
                                cdna4_f8f6f4_shape, opnd.name
                            )
                        if opnd_size_expr is None:
                            opnd_size_expr = self._buffer_vaddr_operand_size_expr(
                                enc.enc_name, opnd.name
                            )
                        if opnd_size_expr is None:
                            opnd_size_expr = self._vflat_vaddr_operand_size_expr(
                                enc.enc_name, opnd.name
                            )
                        if opnd_size_expr is None:
                            opnd_size_expr = self._mimg_operand_size_expr(
                                enc.enc_name, inst.name, opnd.name
                            )
                        if opnd_size_expr is None:
                            opnd_size_expr = str(opnd.size)
                        # The gfx1250 MRISA describes VOP3 compare masks with
                        # the legacy 64-bit width. gfx1250 is wave32-only: V_CMP
                        # writes one 32-bit SGPR, including when VOP3 selects an
                        # arbitrary SGPR through VDST. Keeping the legacy width
                        # here makes def/use and liveness falsely clobber the
                        # following SGPR.
                        if (
                            self.isa_spec.arch_name == 'cdna5'
                            and inst_sem is not None
                            and inst_sem.semantic_class
                            in ('vector_cmp', 'vector_cmp_class')
                            and opnd.is_output
                            and opnd.operand_type == 'OPR_SREG'
                        ):
                            opnd_size_expr = '32'
                        operand_size_exprs[opnd.name] = opnd_size_expr
                        # Some ISA XMLs describe a buffer atomic's vdata only
                        # as an output even though it always supplies the
                        # atomic payload. Keep the source dependency
                        # independent of whether the old memory value is
                        # returned.
                        _is_buffer_atomic_payload = (
                            inst_sem is not None
                            and inst_sem.semantic_class == 'buffer_atomic'
                            and opnd.name == 'vdata'
                        )
                        _is_optional_vflat_saddr = (
                            enc.enc_name.upper() in ('ENC_VFLAT', 'ENC_VGLOBAL')
                            and opnd.name == 'saddr'
                        )
                        _is_optional_flat_scratch = (
                            enc.enc_name.upper() == 'ENC_FLAT'
                            and opnd.fieldless
                            and opnd.operand_type == 'OPR_FLAT_SCRATCH'
                        )
                        if (opnd.is_input or _is_buffer_atomic_payload) and not (
                            _is_optional_vflat_saddr or _is_optional_flat_scratch
                        ):
                            opnd_body.append(
                                f'src_operands_[{src_idx}] = &{opnd.name};'
                            )
                            src_idx += 1
                        elif opnd.is_input and _is_optional_flat_scratch:
                            # The generic FLAT encoding can select FLAT,
                            # SCRATCH, or GLOBAL at runtime.  GLOBAL does not
                            # consume the flat-scratch base; the public XML's
                            # GPUMEM pseudo-operand remains its sole fieldless
                            # memory source.
                            conditional_src_body.append(
                                'if (inst_.seg != 2) '
                                'src_operands_[num_src_++] = &flat_scratch;'
                            )
                        elif opnd.is_input and _is_optional_vflat_saddr:
                            conditional_src_body.append(
                                f'if (inst_.saddr != {self._saddr_null_expr(enc.enc_name)}) '
                                'src_operands_[num_src_++] = &saddr;'
                            )
                        elif self._output_operand_is_also_source(inst, opnd):
                            # Read/write outputs (FMAC/dot/swap accumulators,
                            # sub-dword partial-def destinations, etc.) are also
                            # sources. On DPP/SDWA-capable encodings, defer them
                            # until every explicit input is registered so the
                            # DPP/SDWA helpers that index src_operands_[0] are not
                            # displaced by an output listed before src0. On other
                            # encodings there is no such positional dependency, so
                            # register the read/write output at its XML position to
                            # preserve architectural source order.
                            if defer_readwrite_outputs:
                                readwrite_output_sources.append(opnd.name)
                            else:
                                opnd_body.append(
                                    f'src_operands_[{src_idx}] = &{opnd.name};'
                                )
                                src_idx += 1
                        _is_optional_atomic_return = (
                            opnd.is_output
                            and inst_sem is not None
                            and (
                                (
                                    inst_sem.semantic_class == 'flat_atomic'
                                    and opnd.name == 'vdst'
                                )
                                or (
                                    inst_sem.semantic_class == 'buffer_atomic'
                                    and opnd.name == 'vdata'
                                )
                            )
                        )
                        # Compare-swap consumes two elements through vdata but
                        # returns only the old element. Keep the destination
                        # metadata on a narrow view of the same encoded VGPR so
                        # liveness does not kill the untouched payload half.
                        _needs_atomic_return_view = (
                            _is_optional_atomic_return
                            and inst_sem.semantic_class == 'buffer_atomic'
                            and inst_sem.operation == 'cmpswap'
                            and opnd.name == 'vdata'
                        )
                        atomic_return_operand = (
                            'vdata_return' if _needs_atomic_return_view else opnd.name
                        )
                        if _is_optional_atomic_return:
                            sc0, _, _ = self._coherency_exprs()
                            conditional_dst_body.append(
                                f'if ({self._atomic_return_expr(sc0)}) '
                                f'dst_operands_[num_dst_++] = &{atomic_return_operand};'
                            )
                        elif opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_[{dst_idx}] = &{opnd.name};'
                            )
                            dst_idx += 1
                        if not opnd.is_input and not opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_[{dst_idx}] = &{opnd.name};'
                            )
                            dst_idx += 1
                        _uses_vgpr_msb_roles = (
                            self.isa_spec.profile.uses_vgpr_msb_indexing
                        )
                        _uses_gpr_idx_roles = (
                            self.isa_spec.profile.supports_gpr_idx
                            and enc.enc_name.upper()
                            in (
                                'ENC_VOP1',
                                'ENC_VOP2',
                                'ENC_VOP3',
                                'ENC_VOP3P',
                                'ENC_VOPC',
                            )
                        )
                        if (
                            _uses_vgpr_msb_roles or _uses_gpr_idx_roles
                        ) and not opnd.fieldless:
                            _role = None
                            if self._operand_can_use_vgpr_msb(opnd):
                                _role = self._fixed_vgpr_msb_role(
                                    enc.enc_name, inst.name, opnd
                                )
                                if _role is None and _uses_vgpr_msb_roles:
                                    if self._vbuffer_store_data_uses_dst_vgpr_msb_role(
                                        enc.enc_name, inst_sem, opnd
                                    ):
                                        _role = 'Dst'
                                    elif opnd.is_output:
                                        _role = 'Dst'
                                    elif opnd.is_input:
                                        if vgpr_msb_src_role_idx < len(
                                            self._VGPR_MSB_SRC_ROLES
                                        ):
                                            _role = self._VGPR_MSB_SRC_ROLES[
                                                vgpr_msb_src_role_idx
                                            ]
                                        vgpr_msb_src_role_idx += 1
                            if _role:
                                vgpr_msb_role_body.append(
                                    f'{opnd.name}.set_vgpr_msb_role(amdgpu::VgprMsbRole::{_role});'
                                )
                        private_members.append(cgen.Statement(f'Operand {opnd.name}'))
                        # SADDR member added after the operand loop below
                        if is_smem and opnd.name == 'soffset':
                            opnd_ctor_init.append(
                                f'{opnd.name}(make_smem_offset('
                                f'reinterpret_cast<const OpEncoding*>(inst)))'
                            )
                        elif opnd.name in inst_field_names:
                            opr_type = self._constructor_operand_type(inst_sem, opnd)
                            packed_16bit_source = (
                                self._operand_uses_packed_16bit_source(
                                    enc.enc_name, opnd, reads_dst=reads_dst
                                )
                            )
                            packed_16bit_dst = self._operand_uses_packed_16bit_dst(
                                enc.enc_name, opnd
                            )
                            packed_16bit_args = ''
                            if packed_16bit_dst:
                                packed_16bit_args = (
                                    f', {str(packed_16bit_source).lower()}, true'
                                )
                            elif packed_16bit_source:
                                packed_16bit_args = ', true'
                            operand_value = self._operand_encoding_value_expr(
                                _OperandCtx(
                                    opnd_name=opnd.name,
                                    enc_name=enc.enc_name,
                                    packed_16bit=(
                                        packed_16bit_source or packed_16bit_dst
                                    ),
                                    operand_type=opr_type,
                                    has_acc_field='acc' in inst_field_names,
                                    has_acc_cd_field='acc_cd' in inst_field_names,
                                )
                            )
                            if (
                                profile.renders_gfx11_image_syntax
                                and enc.enc_name.upper() == 'ENC_MIMG'
                                and opnd.name in ('srsrc', 'ssamp')
                            ):
                                operand_value = f'({operand_value} * 4)'
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd_size_expr}, '
                                f'OperandType::{opr_type}, '
                                f'{operand_value}{packed_16bit_args})'
                            )
                            if _needs_atomic_return_view:
                                private_members.append(
                                    cgen.Statement('Operand vdata_return')
                                )
                                opnd_ctor_init.append(
                                    f'vdata_return({(inst_sem.elem_size or 4) * 8}, '
                                    f'OperandType::{opr_type}, {operand_value})'
                                )
                                if _uses_vgpr_msb_roles or _uses_gpr_idx_roles:
                                    vgpr_msb_role_body.append(
                                        'vdata_return.set_vgpr_msb_role('
                                        'amdgpu::VgprMsbRole::Dst);'
                                    )
                        elif opnd.fieldless:
                            # Fieldless operand: no encoding field to decode, so
                            # construct it from its canonical fixed value. Its
                            # fieldless marker + capability policy are applied
                            # once, after any ctor-body reassignment, by the
                            # apply_fieldless_caps() pass below.
                            canonical = self._fieldless_canonical_value(
                                opnd.operand_type
                            )
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd_size_expr}, '
                                f'OperandType::{opnd.operand_type}, {canonical})'
                            )
                        else:
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd.size}, '
                                f'OperandType::{opnd.operand_type}, 0)'
                            )
                    for opnd_name in readwrite_output_sources:
                        opnd_body.append(f'src_operands_[{src_idx}] = &{opnd_name};')
                        src_idx += 1

                    # For flat encodings with a seg field, add saddr as an
                    # optional operand. Declared and initialized LAST among
                    # operands to avoid reorder warnings with data/addr.
                    _has_flat_saddr = self.isa_spec.profile.mnemonic_rule(
                        enc.enc_name
                    ).use_flat_mnemonic
                    if _has_flat_saddr:
                        saddr_null = self._saddr_null_expr(enc.enc_name)
                        private_members.append(cgen.Statement('Operand saddr'))
                        opnd_ctor_init.append('saddr(0, OperandType::OPR_SREG, 0)')

                    class_ctor_decl = cgen.FunctionDeclaration(
                        cgen.Value('', inst.fmt_name),
                        [cgen.Value('const MachineInst *', 'inst')],
                    )
                    public_members.append(class_ctor_decl)
                    if not inst.model_only:
                        public_members.append(
                            cgen.Statement('void execute_impl(amdgpu::Wavefront &wf)')
                        )
                    # A sub-dword (< 32-bit) destination writes only part of its
                    # 32-bit register lane, so the old value survives and the
                    # register is also a read. Surface these partial defs as
                    # implicit uses. Runtime-sized outputs are never sub-dword,
                    # so a static size check suffices. Immediate/label outputs
                    # (e.g. the S_SETREG hwreg selector) never name a register,
                    # so skip them to avoid dead overrides. Fieldless outputs
                    # (e.g. the 1-bit SCC def on OPR_SSRC_SPECIAL_SCC) are inert
                    # side effects, not partial register writes: their
                    # to_register_ref() is always nullopt, so including them here
                    # would emit a dead override AND classify a def as a use.
                    # Their def/use is owned by the fieldless capability policy.
                    _partial_def_outputs = [
                        o.name
                        for o in inst.operands
                        if o.is_output
                        and not o.fieldless
                        and operand_size_exprs.get(o.name, str(o.size)) == str(o.size)
                        and 0 < o.size < 32
                        and not any(
                            tag in o.operand_type.upper()
                            for tag in ('IMM', 'LABEL', 'CONST')
                        )
                    ]
                    # v_writelane preserves the other lanes of vdst, so it reads
                    # the old value: a lane-partial def, distinct from the
                    # sub-dword partial defs above. Surface it only where the XML
                    # marks vdst output-only (e.g. CDNA4, gfx1250); elsewhere
                    # vdst is already a source. implicit_uses keeps it out of the
                    # printed operand list.
                    _writelane_implicit_use_opnd = None
                    if inst_sem and inst_sem.semantic_class == 'vector_writelane':
                        _writelane_implicit_use_opnd = next(
                            (
                                o.name
                                for o in inst.operands
                                if o.is_output
                                and not o.is_input
                                and o.name in ('vdst', 'sdst')
                            ),
                            None,
                        )
                    _dpp_secondary_mask_preserve_output = bool(
                        self.isa_spec.profile.dpp_bound_ctrl_applies_to_inactive_sources
                        and inst_dpp_enc_name.upper() == 'VOP3_SDST_ENC'
                        and self._supports_dpp_for_instruction(inst, inst_dpp_enc_name)
                        and any(o.name == 'sdst' and o.is_output for o in inst.operands)
                    )
                    if _partial_def_outputs or _dpp_secondary_mask_preserve_output:
                        public_members.append(
                            cgen.Statement(
                                'void implicit_uses(RegisterSet &uses) const override'
                            )
                        )
                        # Gated with the definition below: only gfx1250 has a
                        # consumer for the operand-backed hook.
                        if self.isa_spec.profile.uses_vgpr_msb_indexing:
                            public_members.append(
                                cgen.Statement(
                                    'void implicit_use_operands('
                                    'std::vector<const ::rocjitsu::Operand *> &operands) const '
                                    'override'
                                )
                            )
                    elif d16_implicit_use_opnd:
                        public_members.append(
                            cgen.Statement(
                                'void implicit_uses(RegisterSet &uses) const override'
                            )
                        )
                        # Operand-backed twin for VGPR-MSB banked profiles
                        # (gfx1250), which read banked VGPRs only from
                        # implicit_use_operands(); see _partial_def_outputs above.
                        if self.isa_spec.profile.uses_vgpr_msb_indexing:
                            public_members.append(
                                cgen.Statement(
                                    'void implicit_use_operands('
                                    'std::vector<const ::rocjitsu::Operand *> &operands) const '
                                    'override'
                                )
                            )
                    elif _writelane_implicit_use_opnd:
                        public_members.append(
                            cgen.Statement(
                                'void implicit_uses(RegisterSet &uses) const override'
                            )
                        )
                    if cdna5_f8f6f4_shape is not None or cdna5_swmmac_has_modifiers:
                        public_members.append(
                            cgen.Statement(
                                'void build_modifiers(std::string &out) const override'
                            )
                        )
                    # CFG metadata is emitted on the concrete ISA instruction
                    # class, not inferred by generic analysis from mnemonic
                    # strings. BasicBlock asks the virtual branch_offset_bytes()
                    # for direct branch targets.
                    label_operand = next(
                        (
                            op.name
                            for op in inst.operands
                            if op.operand_type == 'OPR_LABEL'
                        ),
                        None,
                    )
                    branch_offset_operand = None
                    if inst_sem and inst_sem.semantic_class in (
                        'branch',
                        'cbranch',
                        'scalar_call',
                    ):
                        branch_offset_operand = label_operand
                    if branch_offset_operand:
                        public_members.append(
                            cgen.Statement(
                                'std::optional<int64_t> branch_offset_bytes() const override'
                            )
                        )
                    # Embed the full mnemonic (with suffix) as a string literal
                    # so the encoding base gets a string_view to static storage.
                    rule = self.isa_spec.profile.mnemonic_rule(enc.enc_name)
                    full_mnemonic = inst.mnemonic + (rule.suffix or '')
                    rendered_mnemonic = (
                        inst.mnemonic
                        if (
                            inst.name == 'V_SWAP_B16'
                            and profile.uses_packed_16bit_e32_source_selectors
                        )
                        else full_mnemonic
                    )
                    mnemonic_expr = f'"{rendered_mnemonic}"'
                    if inst.name == 'V_SWAP_B16':
                        _, dpp8_struct = self._vop_dpp_struct_names(enc.enc_name)
                        if dpp8_struct is not None:
                            dpp_predicate = (
                                'amdgpu::dpp::is_src_dpp8('
                                'reinterpret_cast<const OpEncoding*>(inst)->src0)'
                            )
                            dpp_mnemonic = inst.mnemonic + '_dpp'
                            mnemonic_expr = (
                                f'{dpp_predicate} ? "{dpp_mnemonic}" : '
                                f'"{rendered_mnemonic}"'
                            )
                    if supports_sdwa_extension and self._instruction_supports_sdwa(
                        inst, enc.enc_name
                    ):
                        assert full_mnemonic.endswith('_e32'), (
                            f'{inst.name}: compact SDWA mnemonic does not end in _e32: '
                            f'{full_mnemonic}'
                        )
                        sdwa_mnemonic = full_mnemonic[:-4] + '_sdwa'
                        mnemonic_expr = (
                            'reinterpret_cast<const OpEncoding*>(inst)->src0 '
                            '== amdgpu::SRC_SDWA ? '
                            f'"{sdwa_mnemonic}" : {mnemonic_expr}'
                        )
                    exec_fn_expr = (
                        'nullptr'
                        if inst.model_only
                        else f'make_exec_fn<{inst.fmt_name}>()'
                    )
                    if profile.split_execution_sources and not inst.model_only:
                        exec_fn_expr = self._split_execute_expr(inst.fmt_name)
                    (
                        _tracks_instruction_literal_support,
                        _supports_simm32_literals,
                        _supports_simm64_literals,
                    ) = self._instruction_literal_support(inst, enc)
                    _base_literal_fields = tuple(
                        field
                        for field in _LITERAL_ENCODING_OPERANDS.get(
                            enc.enc_name.upper(), ('', ())
                        )[1]
                        if field in enc_field_names
                    )
                    literal_policy_args = ''
                    if _tracks_instruction_literal_support and _base_literal_fields:
                        active_selector_fields = self._active_literal_selector_fields(
                            inst, _base_literal_fields
                        )
                        assert (
                            active_selector_fields
                            == _base_literal_fields[: len(active_selector_fields)]
                        ), (
                            f'{inst.name}: non-contiguous encoded source selectors '
                            f'{active_selector_fields}'
                        )
                        literal_support = self._literal_support_enum_name(
                            _supports_simm32_literals, _supports_simm64_literals
                        )
                        if literal_support != 'Both' or len(
                            active_selector_fields
                        ) != len(_base_literal_fields):
                            literal_policy_args = f', LiteralSupport::{literal_support}'
                        if len(active_selector_fields) != len(_base_literal_fields):
                            literal_policy_args += f', {len(active_selector_fields)}'
                    init_list_parts = [
                        f'{inst.fmt_true_enc_name}({mnemonic_expr}, '
                        f'reinterpret_cast<const OpEncoding*>(inst), '
                        f'{exec_fn_expr})'
                    ] + opnd_ctor_init
                    init_list = ', '.join(init_list_parts)
                    # Check if this is a memory instruction to set MEMORY_OP flag
                    _mem_sem = inst_sem
                    _MEM_CLASSES = frozenset(
                        {
                            'smem_load',
                            'smem_store',
                            'flat_load',
                            'flat_store',
                            'flat_atomic',
                            'global_load_async_to_lds',
                            'global_store_async_from_lds',
                            'global_load_addtid',
                            'global_store_addtid',
                            'buffer_load',
                            'buffer_store',
                            'buffer_atomic',
                            'tbuffer_load',
                            'tbuffer_store',
                            'buffer_load_format_d16',
                            'buffer_store_format_d16',
                            'ds_read',
                            'ds_read2',
                            'ds_write',
                            'ds_write2',
                            'ds_atomic',
                            'ds_atomic2',
                            'ds_mskor',
                            'ds_append_consume',
                            'ds_barrier_arrive',
                            'ds_read_addtid',
                            'ds_write_addtid',
                            'ds_read_tr_b16',
                            'ds_read_tr_b8',
                            'ds_read_tr_b4',
                            'ds_read_tr_b6',
                        }
                    )
                    ctor_body_parts = list(opnd_body)
                    if (
                        inst.name == 'V_SWAP_B16'
                        and self.isa_spec.profile.uses_packed_16bit_e32_source_selectors
                    ) or (
                        self.isa_spec.profile.renders_gfx11_image_syntax
                        and (
                            enc.enc_name.upper() == 'ENC_MIMG'
                            or inst.name
                            in self.isa_spec.profile.vop3p_absolute_source_instructions
                        )
                    ):
                        ctor_body_parts.append(
                            'omit_repeated_destination_sources_ = true;'
                        )
                    if inst.required_feature_mask:
                        ctor_body_parts.append(
                            f'required_isa_features_ |= uint32_t{{{inst.required_feature_mask}}};'
                        )
                    fieldless_caps_guards: dict[str, str] = {}
                    factory_validation_parts: list[str] = []
                    factory_op_encoding = f'{inst.fmt_true_enc_name}::OpEncoding'
                    # Guard fieldless def/use operands (pushed positionally) from
                    # silently writing past the fixed-size operand arrays. The
                    # capacities mirror instruction.h (see the class constants).
                    assert src_idx <= self._SRC_OPERANDS_CAPACITY, (
                        f'{inst.name}: {src_idx} src operands exceed '
                        f'src_operands_ capacity {self._SRC_OPERANDS_CAPACITY}; '
                        f'grow the std::array in instruction.h.'
                    )
                    assert dst_idx <= self._DST_OPERANDS_CAPACITY, (
                        f'{inst.name}: {dst_idx} dst operands exceed '
                        f'dst_operands_ capacity {self._DST_OPERANDS_CAPACITY}; '
                        f'grow the std::array in instruction.h.'
                    )
                    ctor_body_parts.append(f'num_src_ = {src_idx};')
                    ctor_body_parts.append(f'num_dst_ = {dst_idx};')
                    ctor_body_parts.extend(conditional_src_body)
                    ctor_body_parts.extend(conditional_dst_body)

                    for opnd in inst.operands:
                        if not self._uses_generic_wmma_accumulator_selector(
                            inst_sem, opnd
                        ):
                            continue
                        intervals = [(0, 124)] + self._operand_selector_intervals(
                            opnd.operand_type
                        )
                        raw_value = (
                            f'reinterpret_cast<const {factory_op_encoding}*>(inst)'
                            f'->{opnd.name}'
                        )
                        valid_expr = ' || '.join(
                            f'({raw_value} >= {lo} && {raw_value} <= {hi})'
                            for lo, hi in intervals
                        )
                        factory_validation_parts.append(
                            f'if (!({valid_expr})) '
                            f'[[unlikely]] return emit_error.emit() << "{inst.name} has an invalid '
                            'accumulator selector";'
                        )

                    # LLVM models pseudo-scalar V_S_* destinations as
                    # SReg_32_XEXEC: selectors 0..125 (SGPRs, TTMPs, VCC,
                    # NULL, and M0) are legal, while EXEC and every larger
                    # selector are excluded. The operand is represented as
                    # OPR_SREG, whose enum stops at NULL, so compare against
                    # the matching SDST_EXEC value for the first excluded
                    # selector immediately above M0.
                    if (
                        inst_sem is not None
                        and inst_sem.semantic_class == 'pseudo_scalar_unary'
                    ):
                        factory_validation_parts.append(
                            f'if (reinterpret_cast<const {factory_op_encoding}*>(inst)->vdst >= '
                            'OpSelSdstExec::OPR_SDST_EXEC_EXEC_LO) '
                            f'[[unlikely]] return emit_error.emit() << "{inst.name} has an invalid '
                            'SReg_32_XEXEC destination";'
                        )

                    # Flat segment-aware operands: adjust addr width and add
                    # saddr for SCRATCH (seg==1) and GLOBAL (seg==2) segments.
                    if rule.use_flat_mnemonic:
                        _has_addr = any(o.name == 'addr' for o in inst.operands)
                        if _has_addr:
                            ctor_body_parts.append('if (inst_.seg == 1) {')
                            ctor_body_parts.append(
                                '  addr = Operand(32, OperandType::OPR_VGPR, '
                                'reinterpret_cast<const OpEncoding*>(&inst_)->addr);'
                            )
                            ctor_body_parts.append(
                                f'  if (inst_.saddr != {saddr_null}) {{'
                            )
                            ctor_body_parts.append(
                                '    saddr = Operand(32, OperandType::OPR_SREG, '
                                'inst_.saddr);'
                            )
                            ctor_body_parts.append(
                                '    src_operands_[num_src_++] = &saddr;'
                            )
                            ctor_body_parts.append('  }')
                            ctor_body_parts.append(
                                '} else if (inst_.seg == 2 && '
                                f'inst_.saddr != {saddr_null}) {{'
                            )
                            ctor_body_parts.append(
                                '  addr = Operand(32, OperandType::OPR_VGPR, '
                                'reinterpret_cast<const OpEncoding*>(&inst_)->addr);'
                            )
                            ctor_body_parts.append(
                                '  saddr = Operand(64, OperandType::OPR_SREG, '
                                'inst_.saddr);'
                            )
                            ctor_body_parts.append(
                                '  src_operands_[num_src_++] = &saddr;'
                            )
                            ctor_body_parts.append('}')

                    # Literal constant fixup: when src0/ssrc0/ssrc1 == 255,
                    # replace the operand with the 32-bit literal from the
                    # extended instruction encoding. When this encoding has a
                    # Literal64 form, selector 254 carries its next two DWORDs.
                    # Operands patched by the generic literal loop below, so the
                    # S_SETREG_IMM32_B32 special-case stays mutually exclusive
                    # with it (no double-patch of one operand).
                    _generic_literal_patched: set[str] = set()
                    _lit_info = self._literal_encoding_info(enc, inst_enc_obj, inst)
                    if _lit_info and self._has_machine_inst_struct(_lit_info[0]):
                        _lit_struct, _lit_fields = _lit_info
                        _lit32_struct = _lit_struct
                        if (
                            inst.is_implied_literal_enc
                            and inst.enc_name.upper().endswith('_INST_LITERAL64')
                        ):
                            _lit32_info = _LITERAL_ENCODING_OPERANDS.get(
                                enc.enc_name.upper()
                            )
                            if _lit32_info:
                                _lit32_struct = _lit32_info[0]
                        _encoded_literal_fields = [
                            opnd.name
                            for opnd in inst.operands
                            if opnd.name in _lit_fields
                            and opnd.name in enc_field_names
                            and opnd.operand_type in _LITERAL_CAPABLE_OPERAND_TYPES
                        ]
                        if (
                            _supports_simm32_literals
                            and _supports_simm64_literals
                            and len(_encoded_literal_fields) > 1
                        ):
                            literal32_selector = ' || '.join(
                                f'reinterpret_cast<const {factory_op_encoding}*>(inst)->'
                                f'{field} == 255'
                                for field in _encoded_literal_fields
                            )
                            literal64_selector = ' || '.join(
                                f'reinterpret_cast<const {factory_op_encoding}*>(inst)->'
                                f'{field} == 254'
                                for field in _encoded_literal_fields
                            )
                            factory_validation_parts.append(
                                f'if (({literal32_selector}) && '
                                f'({literal64_selector})) '
                                f'[[unlikely]] return emit_error.emit() << "{inst.name} may not mix '
                                '32-bit and 64-bit literals";'
                            )
                        for opnd in inst.operands:
                            # The only fieldless operand that needs a fixup
                            # are LITERAL ones.
                            if (
                                opnd.fieldless
                                and fieldless_policy(opnd.operand_type).role
                                != FieldlessCategory.LITERAL
                            ):
                                continue
                            if (
                                opnd.name in _lit_fields
                                and opnd.name in enc_field_names
                            ):
                                if (
                                    opnd.operand_type
                                    not in _LITERAL_CAPABLE_OPERAND_TYPES
                                ):
                                    continue
                                literal_operand_type = self._constructor_operand_type(
                                    inst_sem, opnd
                                )
                                if self._uses_generic_wmma_accumulator_selector(
                                    inst_sem, opnd
                                ):
                                    literal_operand_type = opnd.operand_type
                                accepts_literal32 = self._operand_selector_contains(
                                    literal_operand_type, 255
                                )
                                accepts_literal64 = self._operand_selector_contains(
                                    literal_operand_type, 254
                                )
                                # F16 pseudo-scalar V_S_* instructions always
                                # consume literal bits [15:0]. Unlike generic
                                # true16 VOP3 operands, OPSEL does not select a
                                # literal half for this instruction family.
                                pseudo_scalar_f16 = (
                                    inst_sem is not None
                                    and inst_sem.semantic_class == 'pseudo_scalar_unary'
                                    and inst_sem.data_type == 'f16'
                                )
                                fixup = self._literal_operand_fixup_stmt(
                                    opnd,
                                    _lit32_struct,
                                    operand_size_exprs[opnd.name],
                                    'OPR_SIMM32',
                                    (
                                        _lit_fields.index(opnd.name)
                                        if opnd.size == 16
                                        and self.isa_spec.profile.uses_true16_vop3_opsel
                                        and self.isa_spec.profile.has_src_modifiers(
                                            enc.enc_name
                                        )
                                        and not pseudo_scalar_f16
                                        else None
                                    ),
                                    self.isa_spec.arch_name,
                                    inst.name,
                                    inst.enc_name,
                                )
                                assert fixup is not None
                                if (
                                    _supports_simm32_literals
                                    and accepts_literal32 is not False
                                ):
                                    ctor_body_parts.append(
                                        f'if (reinterpret_cast<const OpEncoding*>(inst)->{opnd.name} == 255) '
                                        f'{fixup}'
                                    )
                                    _generic_literal_patched.add(opnd.name)
                                if (
                                    _supports_simm64_literals
                                    and accepts_literal64 is not False
                                ):
                                    ctor_body_parts.append(
                                        f'if (reinterpret_cast<const OpEncoding*>(inst)->{opnd.name} == 254) {{ '
                                        f'const auto *words = reinterpret_cast<const uint32_t *>(inst); '
                                        f'uint32_t literal_word = sizeof(OpEncoding) / sizeof(uint32_t); '
                                        f'uint64_t literal64 = (static_cast<uint64_t>(words[literal_word + 1]) << 32) | words[literal_word]; '
                                        f'{opnd.name} = Operand({operand_size_exprs[opnd.name]}, OperandType::OPR_SIMM64, literal64, true); }}'
                                    )
                            if opnd.name not in enc_field_names:
                                fixup = self._literal_operand_fixup_stmt(
                                    opnd,
                                    _lit_struct,
                                    operand_size_exprs.get(opnd.name),
                                    arch_name=self.isa_spec.arch_name,
                                    inst_name=inst.name,
                                    enc_name=inst.enc_name,
                                )
                                if fixup:
                                    ctor_body_parts.append(fixup)
                                    _generic_literal_patched.add(opnd.name)

                    # SOPK's S_SETREG_IMM32_B32 carries its extension literal
                    # through the encoding base's literal_ member instead of a
                    # separate *InstLiteralMachineInst struct.
                    if inst.name == 'S_SETREG_IMM32_B32':
                        # Patch the 32-bit immediate operand from the encoding
                        # base's literal_ member so it reads the real value
                        # through the normal accessor (execute reads the operand,
                        # not literal_). This must cover BOTH ways the immediate
                        # is modeled across ISAs: rdna4 et al. emit a fieldless
                        # `simm32`, while gfx1250 emits a field-BEARING `literal`
                        # (OPR_SIMM32). _literal_operand_from_expr_stmt returns
                        # None for any non-literal operand (e.g. the OPR_HWREG
                        # selector), so it is safe to offer it every operand.
                        # Fieldless operands still receive their capability
                        # policy from the apply_fieldless_caps pass above; a
                        # field-bearing literal keeps its default (readable)
                        # caps.
                        for opnd in inst.operands:
                            # Mutually exclusive with the generic literal loop:
                            # skip any operand it already patched (otherwise a
                            # last-wins double-patch on ISAs whose SOPK carries a
                            # literal machine-inst struct).
                            if opnd.name in _generic_literal_patched:
                                continue
                            fixup = self._literal_operand_from_expr_stmt(
                                opnd,
                                'literal_',
                                operand_size_exprs.get(opnd.name),
                                arch_name=self.isa_spec.arch_name,
                                inst_name=inst.name,
                                enc_name=inst.enc_name,
                            )
                            if fixup:
                                ctor_body_parts.append(fixup)

                    # DPP fixup: when src0 == amdgpu::SRC_DPP (DPP marker), replace the
                    # src0 operand with vsrc0 from the DPP extension dword.
                    # This lets the instruction execute normally with the
                    # correct VGPR source. Lane permutation is not yet
                    # applied (identity permutation).
                    # DPP/SDWA: src0 marker values 250 (DPP) and 249 (SDWA)
                    # indicate the real VGPR index is in the extension dword.
                    # CDNA uses VopDpp, RDNA uses VopDpp16 (both have vsrc0).
                    _DPP_ENC_BASES = {
                        'ENC_VOP1': 'Vop1',
                        'ENC_VOP2': 'Vop2',
                        'ENC_VOPC': 'Vop1',
                        'ENC_VOP3': 'Vop3',
                        'ENC_VOP3P': 'Vop3p',
                        'VOP3_SDST_ENC': 'Vop3SdstEnc',
                    }
                    _modifier_enc_name = self._instruction_base_encoding_name(inst)
                    # Alternate instruction encodings are emitted in their
                    # parent source file, but their DPP extension still uses
                    # the alternate encoding's machine-inst layout.  In
                    # particular VOP3_SDST_ENC has an extra scalar destination
                    # and cannot be decoded through ENC_VOP3's DPP struct.
                    _dpp_enc_name = inst_dpp_enc_name
                    _enc_base = _DPP_ENC_BASES.get(_dpp_enc_name.upper())
                    _dpp_struct, _dpp8_struct = self._vop_dpp_struct_names(
                        _dpp_enc_name
                    )
                    _has_sdwa = any(
                        'SDWA' in ie.enc_name for ie in self.isa_spec.inst_encodings
                    )
                    _supports_sdwa_encoding = (
                        _has_sdwa
                        and enc.enc_name.upper()
                        in (
                            'ENC_VOP1',
                            'ENC_VOP2',
                            'ENC_VOPC',
                        )
                        and self._instruction_supports_sdwa(inst, _modifier_enc_name)
                    )
                    _dpp_opcode_rule = self._dpp_opcode_rule(inst, _dpp_enc_name)
                    _supports_dpp_encoding = (
                        _dpp_struct is not None
                        and self._supports_dpp_for_instruction(inst, _dpp_enc_name)
                    )
                    _supports_dpp8 = bool(
                        _dpp8_struct is not None
                        and self._instruction_supports_dpp8(inst, _dpp_enc_name)
                    )
                    _recognizes_dpp8_marker = _dpp8_struct is not None or (
                        enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC')
                        and self._supports_vop_dpp8()
                    )
                    _dpp_struct_has_fi = self._machine_inst_struct_has_field(
                        _dpp_struct, 'fi'
                    )
                    _dpp_fi_ctor_stmt = (
                        ' dpp_fi_ = dp->fi;' if _dpp_struct_has_fi else ''
                    )
                    if _enc_base:
                        # A raw selector owns a DPP extension only when the
                        # instruction has a logical src0. Fieldless instructions
                        # can reuse the same bits without acquiring a modifier.
                        _has_logical_src0 = any(
                            op.is_input and not op.fieldless and op.name == 'src0'
                            for op in inst.operands
                        )
                        _rejected_dpp_markers = []
                        if _has_logical_src0 and (
                            _dpp_opcode_rule is DppOpcodeRule.FORBID
                            or (_dpp_struct is not None and not _supports_dpp_encoding)
                        ):
                            _rejected_dpp_markers.append(
                                (
                                    'reinterpret_cast<const OpEncoding*>(inst)->src0 == '
                                    'amdgpu::SRC_DPP',
                                    'DPP',
                                )
                            )
                        if _has_logical_src0 and (
                            _dpp_opcode_rule is DppOpcodeRule.FORBID
                            or (_recognizes_dpp8_marker and not _supports_dpp8)
                        ):
                            _rejected_dpp_markers.append(
                                (
                                    'amdgpu::dpp::is_src_dpp8('
                                    'reinterpret_cast<const OpEncoding*>(inst)->src0)',
                                    'DPP8',
                                )
                            )
                        if _rejected_dpp_markers:
                            _unsupported_dpp_label = (
                                _rejected_dpp_markers[0][1]
                                if len(_rejected_dpp_markers) == 1
                                else 'DPP'
                            )
                            _qualified_rejected_dpp_markers = [
                                marker.replace('OpEncoding', factory_op_encoding)
                                for marker, _ in _rejected_dpp_markers
                            ]
                            factory_validation_parts.append(
                                f'if ({" || ".join(_qualified_rejected_dpp_markers)}) '
                                f'[[unlikely]] return emit_error.emit() << "{inst.name} does not support '
                                f'{_unsupported_dpp_label}";'
                            )
                        if (
                            self.isa_spec.profile.dpp_requires_opsel_lane_alignment
                            and (_supports_dpp_encoding or _supports_dpp8)
                            and _dpp_enc_name.upper() in ('ENC_VOP3', 'ENC_VOP3P')
                        ):
                            _dpp_marker_expr = (
                                '(reinterpret_cast<const OpEncoding*>(inst)->src0 == '
                                'amdgpu::SRC_DPP || '
                                'amdgpu::dpp::is_src_dpp8('
                                'reinterpret_cast<const OpEncoding*>(inst)->src0))'
                            )
                            _factory_dpp_marker_expr = _dpp_marker_expr.replace(
                                'OpEncoding', factory_op_encoding
                            )
                            if _dpp_enc_name.upper() == 'ENC_VOP3P':
                                _opsel_hi_2_field = self._op_sel_hi_2_field(
                                    _dpp_enc_name
                                )
                                factory_validation_parts.append(
                                    f'if ({_factory_dpp_marker_expr}) {{'
                                    f' auto *op = reinterpret_cast<const {factory_op_encoding}*>(inst);'
                                    ' uint32_t opsel_hi = op->opsel_hi |'
                                    f'     ((op->{_opsel_hi_2_field} & 1u) << 2);'
                                    ' if (op->opsel != 0 || opsel_hi != 0x7)'
                                    f'   [[unlikely]] return emit_error.emit() << "{inst.mnemonic}: DPP requires low/low and high/high OPSEL";'
                                    '}'
                                )
                            else:
                                _vop3_input_mask = sum(
                                    1 << int(op.name[-1])
                                    for op in inst.operands
                                    if op.is_input
                                    and op.name in ('src0', 'src1', 'src2')
                                    and op.size == 16
                                )
                                _vop3_has_16bit_dst = any(
                                    op.is_output and op.size == 16
                                    for op in inst.operands
                                )
                                _vop3_high_pattern = (
                                    _vop3_input_mask | 0x8 if _vop3_has_16bit_dst else 0
                                )
                                if not _vop3_has_16bit_dst:
                                    _vop3_high_pattern = _vop3_input_mask
                                _vop3_opsel_check = f'op->opsel != 0 && op->opsel != 0x{_vop3_high_pattern:X}'
                                factory_validation_parts.append(
                                    f'if ({_factory_dpp_marker_expr}) {{'
                                    f' auto *op = reinterpret_cast<const {factory_op_encoding}*>(inst);'
                                    f' if ({_vop3_opsel_check})'
                                    f'   [[unlikely]] return emit_error.emit() << "{inst.mnemonic}: DPP requires matching OPSEL halves";'
                                    '}'
                                )
                        for opnd in inst.operands:
                            if (
                                opnd.name == 'src0'
                                and opnd.is_input
                                and not opnd.fieldless
                                and opnd.name in inst_field_names
                            ):
                                _packed_dpp_src = (
                                    self._operand_uses_packed_16bit_source(
                                        _dpp_enc_name, opnd
                                    )
                                )
                                _packed_dpp_arg = ', true' if _packed_dpp_src else ''
                                _dpp8_vsrc0 = (
                                    'static_cast<unsigned short>(dp8->vsrc0)'
                                    if _packed_dpp_src
                                    else 'dp8->vsrc0'
                                )
                                _dpp_vsrc0 = (
                                    'static_cast<unsigned short>(dp->vsrc0)'
                                    if _packed_dpp_src
                                    else 'dp->vsrc0'
                                )
                                if _supports_dpp8:
                                    ctor_body_parts.append(
                                        f'if (amdgpu::dpp::is_src_dpp8(reinterpret_cast<const OpEncoding*>(inst)->src0)) {{'
                                        f' auto *dp8 = reinterpret_cast<const {_dpp8_struct}*>(inst);'
                                        f' src0 = Operand({opnd.size}, OperandType::OPR_VGPR, '
                                        f'{_dpp8_vsrc0}{_packed_dpp_arg});'
                                        f' dpp8_lane_sel_ = (dp8->lane_sel_0 << 0) | (dp8->lane_sel_1 << 3) |'
                                        f' (dp8->lane_sel_2 << 6) | (dp8->lane_sel_3 << 9) |'
                                        f' (dp8->lane_sel_4 << 12) | (dp8->lane_sel_5 << 15) |'
                                        f' (dp8->lane_sel_6 << 18) | (dp8->lane_sel_7 << 21);'
                                        f' dpp_fi_ = amdgpu::dpp::src_dpp8_fi(reinterpret_cast<const OpEncoding*>(inst)->src0);'
                                        f'}}'
                                    )
                                # DPP (src0 == amdgpu::SRC_DPP): read vsrc0 and DPP control
                                # fields from the ISA-specific extension dword,
                                # storing them on the Instruction base for
                                # apply_dpp() to use later.
                                if _dpp_struct and _supports_dpp_encoding:
                                    _dpp_feature_mask = self._modifier_feature_mask(
                                        inst, _modifier_enc_name, 'dpp'
                                    )
                                    _dpp_feature_stmt = (
                                        f' required_isa_features_ |= '
                                        f'uint32_t{{{_dpp_feature_mask}}};'
                                        if _dpp_feature_mask
                                        else ''
                                    )
                                    if (
                                        _dpp_opcode_rule
                                        is DppOpcodeRule.ROW_SELECT_ONLY
                                    ):
                                        _factory_src0 = f'reinterpret_cast<const {factory_op_encoding}*>(inst)->src0'
                                        factory_validation_parts.append(
                                            f'if (amdgpu::dpp::is_src_dpp8({_factory_src0})) '
                                            f'[[unlikely]] return emit_error.emit() << "{inst.mnemonic}: only DPP row-select controls 0x150-0x15f are supported";'
                                            f'if ({_factory_src0} == amdgpu::SRC_DPP) {{'
                                            f' auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);'
                                            f' if (dp->dpp_ctrl < amdgpu::dpp::ROW_SELECT_BASE ||'
                                            f'     dp->dpp_ctrl > amdgpu::dpp::ROW_SELECT_MAX)'
                                            f'   [[unlikely]] return emit_error.emit() << "{inst.mnemonic}: only DPP row-select controls 0x150-0x15f are supported";'
                                            f'}}'
                                        )
                                        ctor_body_parts.append(
                                            f'if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_DPP) {{'
                                            f' auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);'
                                            f' src0 = Operand({opnd.size}, OperandType::OPR_VGPR, dp->vsrc0);'
                                            f' dpp_ctrl_ = dp->dpp_ctrl;'
                                            f' dpp_row_mask_ = dp->row_mask;'
                                            f' dpp_bank_mask_ = dp->bank_mask;'
                                            f' dpp_bound_ctrl_ = dp->bound_ctrl;'
                                            f'{_dpp_fi_ctor_stmt}'
                                            f'{_dpp_feature_stmt}'
                                            f'}}'
                                        )
                                unsupported_dpp_markers = []
                                if _dpp_struct and not _supports_dpp_encoding:
                                    unsupported_dpp_markers.append(
                                        (
                                            'reinterpret_cast<const OpEncoding*>(inst)->src0 == '
                                            'amdgpu::SRC_DPP',
                                            'DPP',
                                        )
                                    )
                                if _recognizes_dpp8_marker and not _supports_dpp8:
                                    unsupported_dpp_markers.append(
                                        (
                                            'amdgpu::dpp::is_src_dpp8('
                                            'reinterpret_cast<const OpEncoding*>(inst)->src0)',
                                            'DPP8',
                                        )
                                    )
                                if unsupported_dpp_markers:
                                    # The encoding-wide check above also covers
                                    # instructions with a logical src0. Keep this
                                    # operand-local fallback only for marker
                                    # combinations it did not already reject.
                                    if not _rejected_dpp_markers:
                                        unsupported_dpp_label = (
                                            unsupported_dpp_markers[0][1]
                                            if len(unsupported_dpp_markers) == 1
                                            else 'DPP'
                                        )
                                        qualified_dpp_markers = [
                                            marker.replace(
                                                'OpEncoding', factory_op_encoding
                                            )
                                            for marker, _ in unsupported_dpp_markers
                                        ]
                                        factory_validation_parts.append(
                                            f'if ({" || ".join(qualified_dpp_markers)}) '
                                            f'[[unlikely]] return emit_error.emit() << "{inst.name} does not support '
                                            f'{unsupported_dpp_label}";'
                                        )
                                elif (
                                    _dpp_struct
                                    and _supports_dpp_encoding
                                    and _dpp_opcode_rule
                                    is not DppOpcodeRule.ROW_SELECT_ONLY
                                ):
                                    _allows_wave_controls = str(
                                        profile.dpp_supports_wave_controls
                                    ).lower()
                                    _allows_row_bcast = str(
                                        profile.dpp_supports_row_broadcast_controls
                                    ).lower()
                                    _allows_row_xmask = str(
                                        profile.dpp_supports_row_xmask
                                    ).lower()
                                    _factory_src0 = f'reinterpret_cast<const {factory_op_encoding}*>(inst)->src0'
                                    factory_validation_parts.append(
                                        f'if ({_factory_src0} == amdgpu::SRC_DPP) {{'
                                        f' auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);'
                                        f' if (!amdgpu::dpp::dpp_ctrl_is_valid(dp->dpp_ctrl, '
                                        f'{_allows_wave_controls}, {_allows_row_bcast}, {_allows_row_xmask}))'
                                        f'   [[unlikely]] return emit_error.emit() << "{inst.mnemonic}: reserved DPP control";'
                                        f'}}'
                                    )
                                    ctor_body_parts.append(
                                        f'if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_DPP) {{'
                                        f' auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);'
                                        f' src0 = Operand({opnd.size}, OperandType::OPR_VGPR, '
                                        f'{_dpp_vsrc0}{_packed_dpp_arg});'
                                        f' dpp_ctrl_ = dp->dpp_ctrl;'
                                        f' dpp_row_mask_ = dp->row_mask;'
                                        f' dpp_bank_mask_ = dp->bank_mask;'
                                        f' dpp_bound_ctrl_ = dp->bound_ctrl;'
                                        f'{_dpp_fi_ctor_stmt}'
                                        f'{_dpp_feature_stmt}'
                                        f'}}'
                                    )
                                # SDWA (src0 == amdgpu::SRC_SDWA): CDNA and RDNA1/2 only.
                                if _supports_sdwa_encoding:
                                    if enc.enc_name.upper() == 'ENC_VOPC':
                                        _sdwa_struct = 'VopcVopSdwaSdstEncMachineInst'
                                    else:
                                        _sdwa_struct = f'{_enc_base}VopSdwaMachineInst'
                                    _sdwa_src0_format = (
                                        self._sdwa_source_modifier_format(
                                            inst_sem, 0, opnd
                                        )
                                        if inst_sem
                                        else 'amdgpu::sdwa::SourceModifierFormat::NONE'
                                    )
                                    _sdwa_src1_opnd = next(
                                        (
                                            candidate
                                            for candidate in inst.operands
                                            if candidate.name == 'vsrc1'
                                        ),
                                        None,
                                    )
                                    _sdwa_src1_binding = ''
                                    if _sdwa_src1_opnd is not None:
                                        _sdwa_src1_format = (
                                            self._sdwa_source_modifier_format(
                                                inst_sem, 1, _sdwa_src1_opnd
                                            )
                                            if inst_sem
                                            else 'amdgpu::sdwa::SourceModifierFormat::NONE'
                                        )
                                        _sdwa_src1_binding = (
                                            f' sdwa_src1_operand_ = &{_sdwa_src1_opnd.name};'
                                            f' sdwa_src1_format_ = {_sdwa_src1_format};'
                                        )
                                    _sdwa_s1_code = ''
                                    if enc.enc_name.upper() in (
                                        'ENC_VOP2',
                                        'ENC_VOPC',
                                    ):
                                        _sdwa_s1_code = (
                                            f' if (sw->s1)'
                                            f'   vsrc1 = Operand({opnd.size}, OperandType::OPR_SRC,'
                                            f'     reinterpret_cast<const OpEncoding*>(inst)->vsrc1);'
                                        )
                                    _sdwa_dst_binding = ''
                                    if enc.enc_name.upper() == 'ENC_VOPC':
                                        _sdwa_dst_opnd = next(
                                            (
                                                candidate
                                                for candidate in inst.operands
                                                if candidate.is_output
                                                and candidate.fieldless
                                            ),
                                            None,
                                        )
                                        if _sdwa_dst_opnd is None:
                                            raise ValueError(
                                                f'{inst.name}: SDWA VOPC lacks a '
                                                'fieldless destination'
                                            )
                                        _sdwa_dst_binding = (
                                            f' if (sw->sd) {_sdwa_dst_opnd.name} = '
                                            f'Operand({_sdwa_dst_opnd.size}, '
                                            'OperandType::OPR_SREG, sw->sdst);'
                                        )
                                        fieldless_caps_guards[_sdwa_dst_opnd.name] = (
                                            'reinterpret_cast<const OpEncoding*>'
                                            '(inst)->src0 == amdgpu::SRC_SDWA'
                                        )
                                    ctor_body_parts.append(
                                        f'if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_SDWA) {{'
                                        f' auto *sw = reinterpret_cast<const {_sdwa_struct}*>(inst);'
                                        f' src0 = Operand({opnd.size}, sw->s0 ? OperandType::OPR_SRC : OperandType::OPR_VGPR, sw->vsrc0);'
                                        f' sdwa_src0_sel_ = sw->src0_sel;'
                                        f' sdwa_src0_sext_ = sw->src0_sext;'
                                        f' sdwa_src0_neg_ = sw->src0_neg;'
                                        f' sdwa_src0_abs_ = sw->src0_abs;'
                                        f' sdwa_src1_sel_ = sw->src1_sel;'
                                        f' sdwa_src1_sext_ = sw->src1_sext;'
                                        f' sdwa_src1_neg_ = sw->src1_neg;'
                                        f' sdwa_src1_abs_ = sw->src1_abs;'
                                        f' sdwa_src0_operand_ = &{opnd.name};'
                                        f' sdwa_src0_format_ = {_sdwa_src0_format};'
                                        f'{_sdwa_src1_binding}'
                                        + (
                                            f' sdwa_sdst_ = sw->sdst;'
                                            f' sdwa_sd_ = sw->sd;'
                                            f'{_sdwa_dst_binding}'
                                            if enc.enc_name.upper() == 'ENC_VOPC'
                                            else f' sdwa_dst_sel_ = sw->dst_sel;'
                                            f' sdwa_dst_unused_ = sw->dst_unused;'
                                            f' sdwa_clamp_ = sw->clamp;'
                                            f' sdwa_omod_ = sw->omod;'
                                        )
                                        + f'{_sdwa_s1_code}}}'
                                    )
                                elif _has_sdwa and enc.enc_name.upper() in (
                                    'ENC_VOP1',
                                    'ENC_VOP2',
                                    'ENC_VOPC',
                                ):
                                    factory_validation_parts.append(
                                        f'if (reinterpret_cast<const {factory_op_encoding}*>(inst)->src0 == '
                                        'amdgpu::SRC_SDWA) '
                                        f'[[unlikely]] return emit_error.emit() << "{inst.name} does not support SDWA";'
                                    )

                    if (
                        inst.name == 'V_SWAP_B16'
                        and profile.uses_packed_16bit_e32_source_selectors
                    ):
                        for opnd in inst.operands:
                            if opnd.name == 'src0':
                                ctor_body_parts.append(
                                    ' src0 = Operand(16, OperandType::OPR_VGPR, '
                                    'static_cast<unsigned short>('
                                    'reinterpret_cast<const OpEncoding *>(inst)->src0 & 0xffu), '
                                    'true, true);'
                                )

                    if (
                        profile.renders_gfx11_image_syntax
                        and enc.enc_name.upper() == 'ENC_MIMG'
                    ):
                        ctor_body_parts.append('capture_nsa_words(inst, &vaddr);')

                    # Apply the fieldless-operand capability policy once, after
                    # every ctor-body reassignment. Fieldless operands are built
                    # in the init list with the 3-arg Operand ctor (default
                    # caps), and the implied-literal / S_SETREG patches reassign
                    # a fieldless literal via operator= (which resets caps to
                    # default). Emitting apply_fieldless_caps() here — rather than
                    # at construction — yields exactly one, always-correct caps
                    # call per fieldless operand regardless of any reassignment.
                    for opnd in inst.operands:
                        if opnd.fieldless:
                            caps_stmt = self._fieldless_caps_stmt(
                                opnd.name, opnd.operand_type
                            )
                            guard = fieldless_caps_guards.get(opnd.name)
                            if guard:
                                caps_stmt = f'if (!({guard})) {caps_stmt}'
                            ctor_body_parts.append(caps_stmt)

                    ctor_body_parts.extend(vgpr_msb_role_body)

                    if _mem_sem and _mem_sem.semantic_class in _MEM_CLASSES:
                        ctor_body_parts.append('flags_ |= MEMORY_OP;')
                    # Control-flow flags drive BasicBlock splitting and CFG
                    # edge construction. Keep this metadata generated from the
                    # semantic classification so generic code does not have to
                    # know AMDGPU instruction names or opcode values.
                    if _mem_sem and _mem_sem.semantic_class == 'branch':
                        ctor_body_parts.append('flags_ |= BRANCH;')
                    if _mem_sem and _mem_sem.semantic_class == 'cbranch':
                        ctor_body_parts.append('flags_ |= COND_BRANCH;')
                    if _mem_sem and _mem_sem.semantic_class == 'endpgm':
                        # BasicBlock splitting treats PROGRAM_TERMINATOR as a hard
                        # stop with no fallthrough successor. Only S_ENDPGM ends the
                        # wave. S_TRAP is NOT a terminator: on hardware it transfers
                        # to the trap handler and RETURNS to the next instruction,
                        # and with no handler configured (STATUS.TRAP_EN == 0, the
                        # state this simulator models) it executes as a NOP that
                        # falls through. Marking it PROGRAM_TERMINATOR made the CFG
                        # drop that reachable fallthrough while the executor (a NOP)
                        # kept going, so translated code could run off the end of a
                        # block. Skipped-kernel stubs still halt because they emit an
                        # explicit S_ENDPGM after the trap.
                        ctor_body_parts.append('flags_ |= PROGRAM_TERMINATOR;')
                    if _mem_sem and _mem_sem.semantic_class in (
                        'scalar_setpc',
                        'scalar_addpc',
                    ):
                        ctor_body_parts.append('flags_ |= INDIRECT_BRANCH;')
                    if _mem_sem and _mem_sem.semantic_class in (
                        'scalar_swappc',
                        'scalar_call',
                    ):
                        ctor_body_parts.append('flags_ |= INDIRECT_CALL;')
                    # Conditional scalar moves leave the destination unchanged
                    # when their predicate is false, so liveness cannot treat
                    # them as unconditional kills.
                    if _mem_sem and _mem_sem.semantic_class in (
                        'scalar_cmov',
                        'scalar_cmovk',
                    ):
                        ctor_body_parts.append('flags_ |= PREDICATED_DEF;')

                    _waitcnt_names = {
                        'S_WAITCNT',
                        'S_WAIT_LOADCNT',
                        'S_WAIT_STORECNT',
                        'S_WAIT_XCNT',
                        'S_WAIT_EXPCNT',
                        'S_WAIT_DSCNT',
                        'S_WAIT_KMCNT',
                        'S_WAIT_SAMPLECNT',
                        'S_WAIT_BVHCNT',
                        'S_WAIT_TENSORCNT',
                        'S_WAIT_ASYNCCNT',
                        'S_WAIT_LOADCNT_DSCNT',
                        'S_WAIT_STORECNT_DSCNT',
                        'S_WAIT_IDLE',
                        'S_WAIT_ALU',
                        'S_WAIT_EVENT',
                        'S_WAITCNT_VSCNT',
                        'S_WAITCNT_VMCNT',
                        'S_WAITCNT_LGKMCNT',
                        'S_WAITCNT_EXPCNT',
                        'S_WAITCNT_DEPCTR',
                    }
                    _barrier_names = {
                        'S_BARRIER',
                        'S_BARRIER_SIGNAL',
                        'S_BARRIER_WAIT',
                    }
                    if inst.name in _waitcnt_names:
                        ctor_body_parts.append('flags_ |= WAITCNT;')
                    if inst.name in _barrier_names:
                        ctor_body_parts.append('flags_ |= BARRIER;')

                    if inst.name.startswith('V_MFMA_') or inst.name.startswith(
                        'V_SMFMAC_'
                    ):
                        ctor_body_parts.append('flags_ |= MFMA;')

                    if inst.name in {
                        'V_ACCVGPR_WRITE_B32',
                        'V_ACCVGPR_READ_B32',
                        'V_ACCVGPR_MOV_B32',
                    }:
                        ctor_body_parts.append('flags_ |= ACCVGPR;')

                    # EXEC-mask metadata for liveness/dataflow analyses,
                    # derived from the instruction's semantic AST (see
                    # _exec_mask_flag_stmts).
                    ctor_body_parts.extend(_exec_mask_flag_stmts(_mem_sem))
                    # Result-combinator metadata (RESULT_COPY / RESULT_OR) for
                    # EXEC-state all-ones reasoning.
                    ctor_body_parts.extend(_result_combinator_flag_stmts(_mem_sem))

                    ctor_body = ''.join(ctor_body_parts)
                    class_ctor_impl_str = (
                        f'{inst.fmt_name}::'
                        f'{inst.fmt_name}(const MachineInst *inst) '
                        f': {init_list} '
                        f'{{{ctor_body}}}'
                    )
                    class_ctor_impl = cgen.Line(class_ctor_impl_str)
                    class_members.extend(public_members)
                    class_members.extend(private_members)
                    # Generate execute_impl — non-static member method with
                    # the actual execute logic.  Called via make_exec_fn<>.
                    modifier_exec_impl = None
                    sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    if sem:
                        self._current_inst_fields = inst_field_names
                        self._current_operand_names = set(operand_size_exprs)
                        self._current_enc = enc
                        _mask_result_kind = self._mask_result_kind(
                            inst, sem, enc.enc_name
                        )
                        _mask_result_writer = (
                            'commit_result' if _mask_result_kind is not None else None
                        )
                        body = self._gen_execute_body(
                            inst,
                            sem,
                            enc.enc_name,
                            result_writer=_mask_result_writer,
                        )
                        body_true16_vop3 = self._true16_vop3_info(
                            inst, sem, enc.enc_name
                        ).body_uses_true16
                        # VOP: prepend DPP preamble so the encoding
                        # base's apply_dpp() runs before the ALU logic.
                        _dpp_preamble = ''
                        _enc_upper = enc.enc_name.upper()
                        _modifier_enc_name = self._instruction_base_encoding_name(inst)
                        _dpp_enc_upper = inst_dpp_enc_name.upper()
                        _is_vopc = _enc_upper == 'ENC_VOPC'
                        _is_compare = bool(
                            sem
                            and sem.semantic_class
                            in (
                                'vector_cmp',
                                'vector_cmp_class',
                                'vector_cmpx',
                                'vector_cmpx_class',
                            )
                        )
                        _is_cmpx = bool(
                            sem
                            and sem.semantic_class
                            in ('vector_cmpx', 'vector_cmpx_class')
                        )
                        _uses_compare_result_writer = (
                            _mask_result_kind is _MaskResultKind.COMPARE
                        )
                        _uses_exec_result_writer = (
                            _mask_result_kind is _MaskResultKind.EXEC
                        )
                        _compare_dst_name = (
                            'vdst' if _uses_compare_result_writer else None
                        )
                        _modern_dpp_compare = False
                        _has_sdwa_encoding = _enc_upper in (
                            'ENC_VOP1',
                            'ENC_VOP2',
                            'ENC_VOPC',
                        ) and self._instruction_supports_sdwa(inst, _modifier_enc_name)
                        _dpp_struct, _dpp8_struct = self._vop_dpp_struct_names(
                            _dpp_enc_upper
                        )
                        _dpp_opcode_rule = self._dpp_opcode_rule(inst, _dpp_enc_upper)
                        _supports_dpp_encoding = (
                            _dpp_struct is not None
                            and self._supports_dpp_for_instruction(inst, _dpp_enc_upper)
                        )
                        _supports_dpp8 = bool(
                            _dpp8_struct is not None
                            and self._instruction_supports_dpp8(inst, _dpp_enc_upper)
                        )
                        _has_dpp_encoding = any(
                            opnd.name == 'src0' for opnd in inst.operands
                        ) and (
                            _supports_dpp_encoding
                            or _supports_dpp8
                            or _has_sdwa_encoding
                        )
                        _modifier_markers = []
                        if _has_dpp_encoding and _supports_dpp_encoding:
                            _modifier_markers.append('inst_.src0 == amdgpu::SRC_DPP')
                        if _has_dpp_encoding and _supports_dpp8:
                            _modifier_markers.append(
                                'amdgpu::dpp::is_src_dpp8(inst_.src0)'
                            )
                        if _has_dpp_encoding and _has_sdwa_encoding:
                            _modifier_markers.append('inst_.src0 == amdgpu::SRC_SDWA')
                        _modifier_marker_condition = ' || '.join(_modifier_markers)
                        assert bool(_modifier_marker_condition) == _has_dpp_encoding
                        _modern_dpp_sources = False
                        _secondary_mask_dst_name = next(
                            (
                                o.name
                                for o in inst.operands
                                if o.is_output
                                and o.name == 'sdst'
                                and inst_dpp_enc_name.upper() == 'VOP3_SDST_ENC'
                            ),
                            None,
                        )
                        _uses_secondary_result_writer = (
                            _mask_result_kind is _MaskResultKind.SECONDARY
                        )
                        _uses_implicit_vcc_result_writer = (
                            _mask_result_kind is _MaskResultKind.IMPLICIT_VCC
                        )
                        _dpp_observation_restore = ''
                        if _has_dpp_encoding:
                            # DPP/SDWA permute the field-bearing vector sources
                            # only. A fieldless operand (the FMAMK/MADMK inline
                            # literal simm32, or VCC/M0/... side effects) is never
                            # permuted, and one can occupy an interior source slot
                            # -- FMAMK lays out src_operands_ = {src0, simm32,
                            # vsrc1}, so vsrc1 is NOT at src_operands_[1]. The
                            # permute path below therefore addresses the real
                            # sources by name and never replaces an architectural
                            # src_operands_[] entry.
                            _src_input_ops = [
                                o
                                for o in inst.operands
                                # Read/write destinations such as V_FMAC's
                                # accumulator are semantic inputs, but they are
                                # not the encoded src0/src1 fields modified by
                                # DPP or SDWA.
                                if o.is_input and not o.is_output and not o.fieldless
                            ]
                            _src0_name = (
                                _src_input_ops[0].name if _src_input_ops else None
                            )
                            _src1_name = (
                                _src_input_ops[1].name
                                if len(_src_input_ops) > 1
                                else None
                            )
                            # Tripwire: the permuted sources must be field-bearing,
                            # so a future change to the selection above cannot
                            # silently reintroduce reading a fieldless literal as
                            # a VGPR base.
                            assert all(
                                not o.fieldless for o in _src_input_ops[:2]
                            ), f'{inst.name}: DPP/SDWA permute source is fieldless'
                            _dpp_src0_byte_mask_arg = ''
                            if (
                                body_true16_vop3
                                and _src_input_ops
                                and _src_input_ops[0].size == 16
                            ):
                                _dpp_src0_byte_mask_arg = (
                                    ',\n'
                                    '        amdgpu::dpp::true16_source_byte_mask(\n'
                                    '            amdgpu::vop3_opsel(inst_), 0)'
                                )
                            _is_cmpx_vopc = _is_vopc and _is_cmpx
                            _modern_dpp_compare = bool(
                                _is_compare
                                and _supports_dpp_encoding
                                and self.isa_spec.profile.dpp_suppressed_compare_lanes_zero
                            )
                            _modern_dpp_sources = bool(
                                _supports_dpp_encoding
                                and self.isa_spec.profile.dpp_bound_ctrl_applies_to_inactive_sources
                            )
                            _dst_reg_expr = self._e32_true16_dst_reg_expr(
                                inst, enc.enc_name
                            )
                            _dst_op = next(
                                (
                                    o
                                    for o in inst.operands
                                    if o.is_output and self._operand_can_use_vgpr_msb(o)
                                ),
                                None,
                            )
                            _dst_name = _dst_op.name if _dst_op else None
                            _compare_dst_name = (
                                'vdst' if _is_compare and not _is_vopc else None
                            )
                            _dpp_preamble = ''
                            if _src0_name:
                                _dpp_preamble += (
                                    '  std::optional<StagedOperand> dpp_src0_;\n'
                                )
                            if _has_sdwa_encoding and _src1_name:
                                _dpp_preamble += (
                                    '  std::optional<StagedOperand> dpp_src1_;\n'
                                )
                            if _supports_dpp_encoding:
                                inactive_uses_bound_ctrl = str(
                                    self.isa_spec.profile.dpp_bound_ctrl_applies_to_inactive_sources
                                ).lower()
                                _dpp_preamble += (
                                    '  amdgpu::dpp::DppPlan dpp_plan_;\n'
                                    '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '    dpp_plan_ = amdgpu::dpp::make_dpp_plan(\n'
                                    '        wf.wf_size(), dpp_ctrl_, dpp_row_mask_, dpp_bank_mask_,\n'
                                    '        dpp_bound_ctrl_, dpp_fi_, wf.exec(),\n'
                                    f'        {inactive_uses_bound_ctrl});\n'
                                )

                            def _dpp_mask_lines(
                                row_bank_name: str,
                                source_name: str,
                                *,
                                declare: bool = False,
                            ) -> str:
                                prefix = 'uint64_t ' if declare else ''
                                return (
                                    f'    {prefix}{row_bank_name} = dpp_plan_.row_bank_mask;\n'
                                    f'    {prefix}{source_name} = dpp_plan_.source_write_mask;\n'
                                )

                            def _legacy_dpp_write_mask_lines(
                                var_name: str, *, declare: bool = False
                            ) -> str:
                                prefix = (
                                    f'uint64_t {var_name} = '
                                    if declare
                                    else f'{var_name} = '
                                )
                                return f'    {prefix}dpp_plan_.row_bank_mask & dpp_plan_.source_write_mask;\n'

                            if _is_vopc:
                                if not _modern_dpp_compare:
                                    _dpp_preamble += (
                                        '  uint64_t dpp_old_vcc_ = wf.vcc();\n'
                                    )
                                elif _has_sdwa_encoding:
                                    _dpp_preamble += (
                                        '  uint64_t dpp_old_vcc_ = 0;\n'
                                        '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_sd_)\n'
                                        '    dpp_old_vcc_ = wf.vcc();\n'
                                    )
                                if _supports_dpp_encoding:
                                    if _modern_dpp_sources:
                                        _dpp_preamble += (
                                            '  uint64_t dpp_old_exec_ = wf.exec();\n'
                                            '  uint64_t dpp_row_bank_mask_ = ~0ULL;\n'
                                            '  uint64_t dpp_source_write_mask_ = ~0ULL;\n'
                                            '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                            f'{_dpp_mask_lines("dpp_row_bank_mask_", "dpp_source_write_mask_")}'
                                            '  }\n'
                                        )
                                    else:
                                        _dpp_old_exec_line = (
                                            '  uint64_t dpp_old_exec_ = wf.exec();\n'
                                            if _is_cmpx_vopc
                                            else ''
                                        )
                                        _dpp_preamble += (
                                            f'{_dpp_old_exec_line}'
                                            '  uint64_t dpp_write_mask_ = ~0ULL;\n'
                                            '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                            f'{_legacy_dpp_write_mask_lines("dpp_write_mask_")}'
                                            '  }\n'
                                        )
                                elif _dpp_struct:
                                    _dpp_preamble += (
                                        '  if (inst_.src0 == amdgpu::SRC_DPP || amdgpu::dpp::is_src_dpp8(inst_.src0))\n'
                                        '    throw util::UnimplementedInst(mnemonic());\n'
                                    )
                            elif _modern_dpp_compare:
                                _dpp_preamble += (
                                    '  uint64_t dpp_old_exec_ = wf.exec();\n'
                                    '  uint64_t dpp_row_bank_mask_ = ~0ULL;\n'
                                    '  uint64_t dpp_source_write_mask_ = ~0ULL;\n'
                                    '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                    f'{_dpp_mask_lines("dpp_row_bank_mask_", "dpp_source_write_mask_")}'
                                    '  }\n'
                                )
                            elif not _is_compare:
                                if (
                                    _uses_implicit_vcc_result_writer
                                    and _supports_dpp_encoding
                                ):
                                    _dpp_preamble += (
                                        '  uint64_t dpp_old_exec_ = wf.exec();\n'
                                        '  uint64_t dpp_old_vcc_ = wf.vcc();\n'
                                        '  uint64_t dpp_source_write_mask_ = ~0ULL;\n'
                                        '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                        '    dpp_source_write_mask_ = dpp_plan_.source_write_mask;\n'
                                    )
                                elif _modern_dpp_sources:
                                    _dpp_preamble += '  [[maybe_unused]] uint64_t dpp_old_exec_ = wf.exec();\n'
                                if _secondary_mask_dst_name and _modern_dpp_sources:
                                    _dpp_preamble += (
                                        '  uint64_t dpp_source_write_mask_ = ~0ULL;\n'
                                        '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                        '    dpp_source_write_mask_ = dpp_plan_.source_write_mask;\n'
                                        '  uint64_t dpp_old_secondary_dst_ = 0;\n'
                                        '  if (inst_.src0 == amdgpu::SRC_DPP &&\n'
                                        '      (dpp_old_exec_ & ~dpp_source_write_mask_)) {\n'
                                        f'      dpp_old_secondary_dst_ = amdgpu::read_wave_mask_scalar({_secondary_mask_dst_name}, wf);\n'
                                        '  }\n'
                                    )
                            # DPP/SDWA stage the named architectural source into
                            # an execution-local proxy. Swap-style ops
                            # (vector_swap / permlane*_swap) read every output;
                            # DPP on those lane-crossing operations is not a
                            # modeled/legal combination, so fail loudly.
                            _reads_all_outputs = (
                                sem is not None
                                and sem.semantic_class in self._READS_ALL_OUTPUT_CLASSES
                            )
                            _semantic_stages_src0 = (
                                sem is not None
                                and sem.semantic_class == 'vector_movrel'
                                and sem.operation in ('src', 'srcdst', 'srcdst2')
                            )
                            if _reads_all_outputs and (
                                _supports_dpp_encoding or _supports_dpp8
                            ):
                                _dpp_preamble += (
                                    '  if (inst_.src0 == amdgpu::SRC_DPP ||\n'
                                    '      amdgpu::dpp::is_src_dpp8(inst_.src0))\n'
                                    '    throw util::UnimplementedInst(mnemonic());\n'
                                )
                            elif (
                                not _semantic_stages_src0
                                and _dpp_struct
                                and _supports_dpp_encoding
                                and _src0_name
                            ):
                                _dpp_read_exec = (
                                    'dpp_old_exec_'
                                    if _modern_dpp_sources
                                    else 'wf.exec()'
                                )
                                _dpp_preamble += (
                                    '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    f'    amdgpu::dpp::apply_dpp({_src0_name}, dpp_plan_,\n'
                                    f'        {_dpp_read_exec}, dpp_src0_, wf{_dpp_src0_byte_mask_arg});\n'
                                )
                            if (
                                not _reads_all_outputs
                                and not _semantic_stages_src0
                                and _supports_dpp8
                                and _src0_name
                            ):
                                _dpp_preamble += (
                                    '  if (amdgpu::dpp::is_src_dpp8(inst_.src0))\n'
                                    f'    amdgpu::dpp::apply_dpp8({_src0_name}, dpp8_lane_sel_, dpp_fi_,\n'
                                    f'        dpp_src0_, wf{_dpp_src0_byte_mask_arg});\n'
                                )
                            if (
                                _has_sdwa_encoding
                                and not _semantic_stages_src0
                                and _src0_name
                            ):
                                # src1 references the field-bearing second source
                                # by name (it may not sit at src_operands_[1]; see
                                # FMAMK/MADMK note above). The shared helper owns
                                # byte observation, source selection/modifiers,
                                # and staged storage; generated code supplies only
                                # instruction-specific operands and fields.
                                _sdwa_src0_modifier_format = (
                                    self._sdwa_source_modifier_format(
                                        sem, 0, _src_input_ops[0]
                                    )
                                )
                                _sdwa_src1_block = ''
                                if _src1_name:
                                    _sdwa_src1_modifier_format = (
                                        self._sdwa_source_modifier_format(
                                            sem, 1, _src_input_ops[1]
                                        )
                                    )
                                    _sdwa_src1_block = (
                                        '    if (num_src_ > 1)\n'
                                        f'      amdgpu::sdwa::stage_source({_src1_name}, sdwa_src1_sel_,\n'
                                        '          sdwa_src1_sext_, sdwa_src1_neg_, sdwa_src1_abs_,\n'
                                        f'          {_sdwa_src1_modifier_format}, dpp_src1_, wf);\n'
                                    )
                                _dpp_preamble += (
                                    '  if (inst_.src0 == amdgpu::SRC_SDWA) {\n'
                                    f'    amdgpu::sdwa::stage_source({_src0_name}, sdwa_src0_sel_,\n'
                                    '        sdwa_src0_sext_, sdwa_src0_neg_, sdwa_src0_abs_,\n'
                                    f'        {_sdwa_src0_modifier_format}, dpp_src0_, wf);\n'
                                    + _sdwa_src1_block
                                    + '  }\n'
                                )
                            if not _semantic_stages_src0:
                                _dpp_preamble += (
                                    f'  ScopedOperandDelegate dpp_src0_binding_({_src0_name},\n'
                                    '      dpp_src0_ ? &*dpp_src0_ : nullptr);\n'
                                    if _src0_name
                                    else ''
                                ) + (
                                    f'  ScopedOperandDelegate dpp_src1_binding_({_src1_name},\n'
                                    '      dpp_src1_ ? &*dpp_src1_ : nullptr);\n'
                                    if _has_sdwa_encoding and _src1_name
                                    else ''
                                )
                            if _supports_dpp_encoding and not _is_compare and _dst_name:
                                _dpp_preamble += (
                                    '  amdgpu::dpp::ScopedVgprWriteMask dpp_write_mask_scope_;\n'
                                    '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '    dpp_write_mask_scope_.bind(\n'
                                    '        wf, wf.exec() & dpp_plan_.row_bank_mask &\n'
                                    '                dpp_plan_.source_write_mask);\n'
                                )
                                _dpp_observation_restore = (
                                    '  dpp_write_mask_scope_.restore();\n'
                                )
                        _result_commit_setup = ''
                        _result_shared_arg = ''
                        if _uses_compare_result_writer:
                            if _modern_dpp_compare:
                                _compare_final_result = (
                                    '    if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '      final_result = amdgpu::dpp::dpp_compare_result(\n'
                                    '          raw_result, dpp_old_exec_,\n'
                                    '          dpp_row_bank_mask_, dpp_source_write_mask_);\n'
                                )
                            else:
                                _compare_final_result = ''
                            _result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    uint64_t final_result = raw_result;\n'
                                f'{_compare_final_result}'
                                f'    amdgpu::write_wave_mask_scalar({_compare_dst_name or "vdst"}, wf, final_result);\n'
                                '  };\n'
                            )
                            _result_shared_arg = ', commit_result'
                        elif _uses_exec_result_writer:
                            if _modern_dpp_compare:
                                _exec_final_result = (
                                    '    if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '      final_result = amdgpu::dpp::dpp_compare_result(\n'
                                    '          raw_result, dpp_old_exec_,\n'
                                    '          dpp_row_bank_mask_, dpp_source_write_mask_);\n'
                                )
                            else:
                                _exec_final_result = ''
                            _result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    uint64_t final_result = raw_result;\n'
                                f'{_exec_final_result}'
                                '    wf.set_exec(final_result);\n'
                                '  };\n'
                            )
                            _result_shared_arg = ', commit_result'
                        elif _uses_secondary_result_writer:
                            _secondary_final_result = ''
                            if _modern_dpp_sources:
                                _secondary_final_result = (
                                    '    if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '      final_result = amdgpu::dpp::dpp_source_suppressed_result(\n'
                                    '          raw_result, dpp_old_secondary_dst_, dpp_old_exec_,\n'
                                    '          dpp_source_write_mask_);\n'
                                )
                            _result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    uint64_t final_result = raw_result;\n'
                                f'{_secondary_final_result}'
                                f'    amdgpu::write_wave_mask_scalar({_secondary_mask_dst_name}, wf, final_result);\n'
                                '  };\n'
                            )
                            _result_shared_arg = ', commit_result'
                        elif _uses_implicit_vcc_result_writer:
                            _implicit_vcc_final_result = ''
                            if _supports_dpp_encoding:
                                _implicit_vcc_final_result = (
                                    '    if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                    '      final_result = amdgpu::dpp::dpp_source_suppressed_result(\n'
                                    '          raw_result, dpp_old_vcc_, dpp_old_exec_,\n'
                                    '          dpp_source_write_mask_);\n'
                                )
                            _result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    uint64_t final_result = raw_result;\n'
                                f'{_implicit_vcc_final_result}'
                                '    wf.set_vcc_mask(final_result);\n'
                                '  };\n'
                            )
                            _result_shared_arg = ', commit_result'
                        _ordinary_result_commit_setup = ''
                        if _uses_compare_result_writer:
                            _ordinary_result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                f'    amdgpu::write_wave_mask_scalar({_compare_dst_name or "vdst"}, wf, raw_result);\n'
                                '  };\n'
                            )
                        elif _uses_exec_result_writer:
                            _ordinary_result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    wf.set_exec(raw_result);\n'
                                '  };\n'
                            )
                        elif _uses_secondary_result_writer:
                            _ordinary_result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                f'    amdgpu::write_wave_mask_scalar({_secondary_mask_dst_name}, wf, raw_result);\n'
                                '  };\n'
                            )
                        elif _uses_implicit_vcc_result_writer:
                            _ordinary_result_commit_setup = (
                                '  auto commit_result = [&](uint64_t raw_result) {\n'
                                '    wf.set_vcc_mask(raw_result);\n'
                                '  };\n'
                            )
                        # SDWA postamble: apply dst_sel merge and float clamp after ALU.
                        _sdwa_postamble = ''
                        _dpp_cleanup = ''
                        if _has_dpp_encoding:
                            if (
                                _modern_dpp_compare
                                and not _uses_compare_result_writer
                                and not _uses_exec_result_writer
                            ):
                                _new_result_expr = (
                                    'wf.exec()'
                                    if _is_cmpx
                                    else (
                                        'wf.vcc()'
                                        if _is_vopc
                                        else f'amdgpu::read_wave_mask_scalar({_compare_dst_name}, wf)'
                                    )
                                )
                                if _is_cmpx:
                                    _write_result = '    wf.set_exec(dpp_cmp_result);\n'
                                elif _is_vopc:
                                    _write_result = (
                                        '    wf.set_vcc_mask(dpp_cmp_result);\n'
                                    )
                                else:
                                    _write_result = (
                                        f'    amdgpu::write_wave_mask_scalar({_compare_dst_name}, wf, '
                                        'dpp_cmp_result);\n'
                                    )
                                _dpp_cleanup_mask_lines = (
                                    '    uint64_t dpp_row_bank_mask = dpp_row_bank_mask_;\n'
                                    '    uint64_t dpp_source_write_mask = dpp_source_write_mask_;\n'
                                )
                                _dpp_cleanup += (
                                    '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                    f'{_dpp_cleanup_mask_lines}'
                                    f'    uint64_t dpp_new_result = {_new_result_expr};\n'
                                    '    uint64_t dpp_cmp_result = amdgpu::dpp::dpp_compare_result(\n'
                                    '        dpp_new_result, dpp_old_exec_,\n'
                                    '        dpp_row_bank_mask, dpp_source_write_mask);\n'
                                    f'{_write_result}'
                                    '  }\n'
                                )
                                if _is_vopc and _has_sdwa_encoding:
                                    _dpp_cleanup += (
                                        '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_sd_) {\n'
                                        '    uint64_t cmp_result = wf.vcc();\n'
                                        '    uint32_t sb = wf.sgpr_alloc().base;\n'
                                        '    amdgpu::write_explicit_lane_mask(sb + sdwa_sdst_, wf, cmp_result);\n'
                                        '    wf.set_vcc_raw(dpp_old_vcc_);\n'
                                        '  }\n'
                                    )
                            elif _is_vopc:
                                if _supports_dpp_encoding:
                                    _dpp_cmpx_exec_merge = (
                                        '    uint64_t new_exec = wf.exec();\n'
                                        '    uint64_t merged_exec = (new_exec & dpp_write_mask) |\n'
                                        '                           (dpp_old_exec_ & ~dpp_write_mask);\n'
                                        '    wf.set_exec(merged_exec);\n'
                                        if _is_cmpx_vopc
                                        else ''
                                    )
                                    _dpp_cleanup += (
                                        '  if (inst_.src0 == amdgpu::SRC_DPP && dpp_write_mask_ != ~0ULL) {\n'
                                        '    uint64_t new_vcc = wf.vcc();\n'
                                        '    uint64_t merged = (new_vcc & dpp_write_mask_) | (dpp_old_vcc_ & ~dpp_write_mask_);\n'
                                        '    wf.set_vcc_raw(merged);\n'
                                        f'{_dpp_cmpx_exec_merge.replace("dpp_write_mask", "dpp_write_mask_")}'
                                        '  }\n'
                                    )
                                if _has_sdwa_encoding:
                                    _dpp_cleanup += (
                                        '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_sd_) {\n'
                                        '    uint64_t cmp_result = wf.vcc();\n'
                                        '    uint32_t sb = wf.sgpr_alloc().base;\n'
                                        '    amdgpu::write_explicit_lane_mask(sb + sdwa_sdst_, wf, cmp_result);\n'
                                        '    wf.set_vcc_raw(dpp_old_vcc_);\n'
                                        '  }\n'
                                    )
                        _apply_float_sdwa_clamp = bool(
                            sem and sem.data_type in ('f16', 'f32', 'f64')
                        )
                        _clamp_template_arg = (
                            'true' if _apply_float_sdwa_clamp else 'false'
                        )
                        _local_body = body
                        _local_body = re.sub(
                            r'amdgpu::RegisterAccess\(wf\)\.write_lane\(\s*'
                            r'([A-Za-z_][A-Za-z0-9_]*),\s*lane,\s*',
                            rf'amdgpu::sdwa::write_lane<{_clamp_template_arg}>'
                            r'(*this, wf, \1, lane, ',
                            _local_body,
                        )
                        _local_body = re.sub(
                            r'amdgpu::RegisterAccess\(wf\)\.write_lane64\(\s*'
                            r'([A-Za-z_][A-Za-z0-9_]*),\s*lane,\s*',
                            rf'amdgpu::sdwa::write_lane64<{_clamp_template_arg}>'
                            r'(*this, wf, \1, lane, ',
                            _local_body,
                        )
                        # Skip DPP/SDWA preamble and cleanup for unimplemented
                        # instructions whose body is ONLY a throw — the cleanup
                        # code after the throw would be unreachable. Only match
                        # pure-throw bodies, not bodies with conditional throws.
                        body_stripped = body.strip().rstrip(';').strip()
                        body_throws = (
                            body_stripped.startswith('(void)wf;')
                            and 'throw util::UnimplementedInst' in body_stripped
                            and body_stripped.count('\n') <= 1
                        )
                        can_share = self._can_share_execute(
                            inst.mnemonic, inst, enc.enc_name
                        )
                        # Ops with an arch-portable SIMD fast-path probe must
                        # route through the shared execute template even when
                        # _can_share_execute is False for this ISA: the probe
                        # lives only in the shared kernel (simd_probe_line is
                        # emitted in _write_shared_execute_templates), and
                        # delegating keeps the DPP/SDWA cleanup + postamble
                        # running around the call (an inlined body with the
                        # probe's early `return` would skip them). Without this,
                        # the dst-accumulate v_fmac_f64 / v_fmac_f32 / v_mac_*
                        # family inlines its scalar loop on CDNA4 and the probe
                        # is dead code. Only *arch-portable* probes qualify: the
                        # inline-literal FMA forms (v_fmaak/fmamk/madak/madmk)
                        # read the literal through an ISA-divergent member, so a
                        # single shared body can't serve every ISA — those are
                        # left to the genuine shared plan.
                        _portable_probe = self._can_force_shared_simd_probe(
                            inst, enc.enc_name
                        )
                        _local_true16_probe = self._true16_vop3_local_simd_probe(
                            inst,
                            sem,
                            enc.enc_name,
                            result_writer=_mask_result_writer,
                        )
                        _renamed_vop3p_probe = self._renamed_vop3p_local_simd_probe(
                            inst, enc.enc_name
                        )
                        assert not (_local_true16_probe and _renamed_vop3p_probe)
                        _local_simd_probe = _local_true16_probe or _renamed_vop3p_probe
                        _local_simd_probe_body = ''
                        _local_scalar_body = _local_body
                        _local_execute_body = _local_scalar_body
                        if _local_simd_probe:
                            _local_simd_body = (
                                f'  auto &inst = *this;\n{_local_simd_probe}\n'
                            )
                            if _dpp_cleanup or _sdwa_postamble:
                                # SIMD probe macros return from their containing
                                # function on success. Keep architecture-local
                                # SIMD enabled for DPP/SDWA by containing
                                # the probe and scalar fallback in a lambda; the
                                # return then exits only the lambda, so result
                                # cleanup still runs exactly once afterward.
                                _local_execute_body = (
                                    '  [&]() -> void {\n'
                                    + textwrap.indent(_local_simd_body, '  ')
                                    + textwrap.indent(_local_scalar_body, '  ')
                                    + '\n  }();'
                                )
                            else:
                                _local_simd_probe_body = _local_simd_body
                        _ordinary_local_execute_body = _local_scalar_body
                        if _local_simd_probe:
                            _ordinary_local_execute_body = (
                                _local_simd_body + _local_scalar_body
                            )
                        if _has_dpp_encoding and not body_throws:
                            class_members.extend(
                                [
                                    cgen.Line('private:'),
                                    cgen.Statement(
                                        'void execute_modifier_impl(amdgpu::Wavefront &wf)'
                                    ),
                                ]
                            )
                        if body_throws:
                            exec_impl = cgen.Line(
                                f'void {inst.fmt_name}::execute_impl'
                                f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                            )
                        elif can_share or _portable_probe:
                            enc_key = enc.enc_name.lower().replace('enc_', '')
                            tmpl_name = f'{inst.mnemonic}_{enc_key}'
                            # Tripwire for the hard-coded inline-literal operand
                            # name in the shared SIMD ternary table. If this ISA
                            # reaches the shared SIMD path for an inline-literal
                            # FMA, its instruction must carry the operand the
                            # template reads `k` from (today `simm32`). gfx1250
                            # names its literal `literal` and is kept out of this
                            # path by the sharing preflight; if that ever changes
                            # this fails loudly at generation instead of emitting
                            # `inst.simm32` into a C++ file that will not compile.
                            from amdisa.codegen.execute.simd_codegen import (
                                simd_ternary_literal_operand_name,
                            )

                            _lit_name = simd_ternary_literal_operand_name(tmpl_name)
                            if _lit_name is not None:
                                assert any(
                                    o.name == _lit_name for o in inst.operands
                                ), (
                                    f'{inst.name}: shared SIMD ternary template '
                                    f'{tmpl_name!r} reads its inline literal from '
                                    f'operand {_lit_name!r}, but this instruction '
                                    f'has no such operand (operands: '
                                    f'{[o.name for o in inst.operands]}). An ISA '
                                    f'whose literal is named differently must stay '
                                    f'out of this shared SIMD path.'
                                )
                            _shared_execute_call = f'  amdgpu::execute_{tmpl_name}(*this, wf{_result_shared_arg});\n'
                            if _has_dpp_encoding:
                                exec_impl = cgen.Line(
                                    f'void {inst.fmt_name}::execute_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'  if ({_modifier_marker_condition}) {{\n'
                                    '    execute_modifier_impl(wf);\n'
                                    '    return;\n'
                                    '  }\n'
                                    f'{_ordinary_result_commit_setup}'
                                    f'{_shared_execute_call}'
                                    '}'
                                )
                                modifier_exec_impl = cgen.Line(
                                    f'RJ_NOINLINE void {inst.fmt_name}::execute_modifier_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'{_dpp_preamble}'
                                    f'{_result_commit_setup}'
                                    f'{_shared_execute_call}'
                                    f'{_dpp_observation_restore}'
                                    f'{_dpp_cleanup}'
                                    f'{_sdwa_postamble}}}'
                                )
                            else:
                                exec_impl = cgen.Line(
                                    f'void {inst.fmt_name}::execute_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'{_result_commit_setup}'
                                    f'{_shared_execute_call}'
                                    '}'
                                )
                            # Store the shared template body. First writer wins:
                            # for a can_share op that is its plan owner; for a
                            # force-shared portable probe op (no plan owner on any
                            # ISA) the body is arch-independent, so whichever ISA
                            # writes first is correct. That arch-independence is an
                            # invariant, not a hope: verify it. If a later ISA
                            # produces a DIFFERENT body for the same
                            # (mnemonic, enc) key, the "shared" body is silently
                            # arch-dependent and emitting just the first writer's
                            # version would miscompile the other arch. Assert
                            # instead of discarding.
                            body_key = (inst.mnemonic, enc.enc_name)
                            existing = self._shared_execute_bodies.get(body_key)
                            if existing is None:
                                self._shared_execute_bodies[body_key] = (
                                    inst,
                                    sem,
                                    body,
                                    enc.enc_name,
                                    body_true16_vop3,
                                )
                            elif existing[2] != body:
                                _exist_inst, _, _exist_body, _, _ = existing
                                raise AssertionError(
                                    'shared execute body collision: '
                                    f'mnemonic={inst.mnemonic!r} '
                                    f'enc={enc.enc_name!r} produced two '
                                    'different bodies for the same shared-template '
                                    'key (the body is not arch-independent, so '
                                    'first-writer-wins would miscompile one arch).'
                                    f'\n--- first writer body ---\n{_exist_body}'
                                    f'\n--- this writer body ---\n{body}'
                                )
                            elif existing[4] != body_true16_vop3:
                                raise AssertionError(
                                    'shared execute true16 metadata collision: '
                                    f'mnemonic={inst.mnemonic!r} '
                                    f'enc={enc.enc_name!r} produced the same shared '
                                    'body with different VOP3 true16 SIMD probe '
                                    'requirements.'
                                )
                        else:
                            if _has_dpp_encoding:
                                exec_impl = cgen.Line(
                                    f'void {inst.fmt_name}::execute_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'  if ({_modifier_marker_condition}) {{\n'
                                    '    execute_modifier_impl(wf);\n'
                                    '    return;\n'
                                    '  }\n'
                                    f'{_ordinary_result_commit_setup}'
                                    f'{_ordinary_local_execute_body}\n'
                                    '}'
                                )
                                modifier_exec_impl = cgen.Line(
                                    f'RJ_NOINLINE void {inst.fmt_name}::execute_modifier_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'{_dpp_preamble}'
                                    f'{_result_commit_setup}'
                                    f'{_local_simd_probe_body}'
                                    f'{_local_execute_body}\n'
                                    f'{_dpp_observation_restore}'
                                    f'{_dpp_cleanup}'
                                    f'{_sdwa_postamble}}}'
                                )
                            else:
                                exec_impl = cgen.Line(
                                    f'void {inst.fmt_name}::execute_impl'
                                    f'(amdgpu::Wavefront &wf) {{\n'
                                    f'{_result_commit_setup}'
                                    f'{_local_simd_probe_body}'
                                    f'{_local_execute_body}\n'
                                    '}'
                                )
                    else:
                        exec_impl = cgen.Line(
                            f'void {inst.fmt_name}::execute_impl'
                            f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                        )

                    s = cgen.Struct(
                        f'{inst.fmt_name} : public {inst.fmt_true_enc_name}',
                        class_members,
                    )
                    inst_classes.append(s)
                    inst_impls = [class_ctor_impl]
                    decoder_factory = decoder_factories.get(inst.fmt_name)
                    if decoder_factory is not None:
                        factory_mnemonic_expr = mnemonic_expr.replace(
                            'OpEncoding', factory_op_encoding
                        )
                        factory_validation_body = ''.join(
                            f'  {line}\n' for line in factory_validation_parts
                        )
                        factory_inst_decl = (
                            '  const auto *inst = opcode;\n'
                            if factory_validation_parts
                            or '(inst)' in factory_mnemonic_expr
                            else ''
                        )
                        inst_impls.append(
                            cgen.Line(
                                'namespace detail {\n'
                                f'DecodeResult {decoder_factory}'
                                '(const MachineInst *opcode, const DecodeErrorEmitter &emit_error) {\n'
                                f'{factory_inst_decl}'
                                f'  Result validation = {inst.fmt_true_enc_name}::validate_encoding('
                                f'{factory_mnemonic_expr}, '
                                f'reinterpret_cast<const {factory_op_encoding}*>(opcode), '
                                f'emit_error{literal_policy_args});\n'
                                '  if (validation.failed()) [[unlikely]]\n'
                                '    return Result::failure();\n'
                                f'{factory_validation_body}'
                                f'  return std::make_unique<{inst.fmt_name}>(opcode);\n'
                                '}\n'
                                '} // namespace detail'
                            )
                        )
                    if cdna5_f8f6f4_shape is not None:
                        inst_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::build_modifiers(std::string &out) const {{\n'
                                f'  out += " matrix_a_fmt:";\n'
                                f'  out += cdna5_matrix_fmt_name(inst_.opsel);\n'
                                f'  out += " matrix_b_fmt:";\n'
                                f'  out += cdna5_matrix_fmt_name((inst_.{self._op_sel_hi_2_field("ENC_VOP3P")} << 2) | inst_.opsel_hi);\n'
                                f'}}'
                            )
                        )
                    elif cdna5_swmmac_has_modifiers:
                        inst_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::build_modifiers(std::string &out) const {{\n'
                                f'  if (inst_.opsel & 0x1)\n'
                                f'    out += " index_key:1";\n'
                                f'  if (inst_.opsel & 0x4)\n'
                                f'    out += " matrix_a_reuse";\n'
                                f'  if (inst_.{self._op_sel_hi_2_field("ENC_VOP3P")})\n'
                                f'    out += " matrix_b_reuse";\n'
                                f'}}'
                            )
                        )
                    if branch_offset_operand:
                        inst_impls.append(
                            cgen.Line(
                                f'std::optional<int64_t> '
                                f'{inst.fmt_name}::branch_offset_bytes() const {{\n'
                                f'  // AMDGPU PC-relative branch immediates are signed '
                                f'instruction-count deltas.\n'
                                f'  return static_cast<int64_t>('
                                f'static_cast<int16_t>({branch_offset_operand}.encoding_value_)) * 4;\n'
                                f'}}'
                            )
                        )
                    if _partial_def_outputs or _dpp_secondary_mask_preserve_output:
                        _pd_body = ''.join(
                            f'  if (auto r = {name}.to_register_ref())\n'
                            f'    uses.expand(*r);\n'
                            for name in _partial_def_outputs
                        )
                        if _dpp_secondary_mask_preserve_output:
                            _pd_body += (
                                '  if (inst_.src0 == amdgpu::SRC_DPP && '
                                'dpp_bound_ctrl_ == 0 &&\n'
                                '      (dpp_fi_ == 0 || '
                                'amdgpu::dpp::dpp_ctrl_produces_oob(dpp_ctrl_)))\n'
                                '    if (auto r = sdst.to_register_ref())\n'
                                '      uses.expand(*r);\n'
                            )
                        inst_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::implicit_uses'
                                f'(RegisterSet &uses) const {{\n'
                                f'  {inst.fmt_true_enc_name}::implicit_uses(uses);\n'
                                f'{_pd_body}'
                                f'}}'
                            )
                        )
                        # Operand-backed twin of implicit_uses: report each partial
                        # def as a preserved-read Operand so a caller can resolve it
                        # with its own VGPR-MSB role and width (see
                        # Instruction::implicit_use_operands). Gated to profiles
                        # with VGPR-MSB banking -- InstDefUse only consults this
                        # hook when vgpr_msb != nullptr, so on every other
                        # architecture the override would be dead weight.
                        if self.isa_spec.profile.uses_vgpr_msb_indexing:
                            _pd_operands_body = ''.join(
                                f'  if ({name}.to_register_ref())\n'
                                f'    operands.push_back(&{name});\n'
                                for name in _partial_def_outputs
                            )
                            inst_impls.append(
                                cgen.Line(
                                    f'void {inst.fmt_name}::implicit_use_operands'
                                    f'(std::vector<const ::rocjitsu::Operand *> &operands) '
                                    f'const {{\n'
                                    f'  {inst.fmt_true_enc_name}::implicit_use_operands(operands);\n'
                                    f'{_pd_operands_body}'
                                    f'}}'
                                )
                            )
                    elif d16_implicit_use_opnd:
                        # Single-register loads read the whole destination; a
                        # multi-register FORMAT load only reads its last (partial)
                        # register, so expand just that one.
                        if d16_partial_reg_offset:
                            d16_expand_stmt = (
                                f'uses.expand(RegisterRef{{r->cls, '
                                f'static_cast<uint16_t>(r->index + '
                                f'{d16_partial_reg_offset}), 1}});'
                            )
                        else:
                            d16_expand_stmt = 'uses.expand(*r);'
                        # Older MUBUF encodings redirect the load to LDS when
                        # inst_.lds is set, leaving no VGPR destination; only guard
                        # the preserved-destination read where such a field exists
                        # (newer encodings have no lds field).
                        d16_lds_guarded = 'lds' in inst_field_names
                        d16_guard = '  if (!inst_.lds)\n' if d16_lds_guarded else ''
                        d16_ind = '    ' if d16_lds_guarded else '  '
                        inst_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::implicit_uses'
                                f'(RegisterSet &uses) const {{\n'
                                f'  {inst.fmt_true_enc_name}::implicit_uses(uses);\n'
                                f'{d16_guard}'
                                f'{d16_ind}if (auto r = {d16_implicit_use_opnd}.to_register_ref())\n'
                                f'{d16_ind}  {d16_expand_stmt}\n'
                                f'}}'
                            )
                        )
                        # Operand-backed twin (see header decl): gfx1250 clears
                        # the VGPR class from implicit_uses() and reads banked
                        # VGPRs only from implicit_use_operands(). gfx1250 has
                        # only single-register D16 loads, so the whole operand is
                        # the partial register; the operand hook cannot express a
                        # partial last register of a multi-register def.
                        if self.isa_spec.profile.uses_vgpr_msb_indexing:
                            assert d16_partial_reg_offset == 0, (
                                'multi-register D16 partial def is not '
                                'expressible via implicit_use_operands'
                            )
                            inst_impls.append(
                                cgen.Line(
                                    f'void {inst.fmt_name}::implicit_use_operands'
                                    f'(std::vector<const ::rocjitsu::Operand *> &operands) '
                                    f'const {{\n'
                                    f'  {inst.fmt_true_enc_name}::implicit_use_operands(operands);\n'
                                    f'{d16_guard}'
                                    f'{d16_ind}if ({d16_implicit_use_opnd}.to_register_ref())\n'
                                    f'{d16_ind}  operands.push_back(&{d16_implicit_use_opnd});\n'
                                    f'}}'
                                )
                            )
                    elif _writelane_implicit_use_opnd:
                        inst_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::implicit_uses'
                                f'(RegisterSet &uses) const {{\n'
                                f'  {inst.fmt_true_enc_name}::implicit_uses(uses);\n'
                                f'  if (auto r = '
                                f'{_writelane_implicit_use_opnd}.to_register_ref())\n'
                                f'    uses.expand(*r);\n'
                                f'}}'
                            )
                        )
                    execution_impls = [] if inst.model_only else [exec_impl]
                    if modifier_exec_impl is not None and not inst.model_only:
                        execution_impls.append(modifier_exec_impl)
                    if not profile.split_execution_sources:
                        inst_impls.extend(execution_impls)
                        execution_impls = []
                    class_func_impls.model.extend(inst_impls)
                    class_func_impls.execution.extend(execution_impls)
                    file_stem = profile.source_split_file_stem(
                        enc.enc_name, inst.name, inst_sem
                    )
                    source_impl_units.model.append(
                        _SourceImplUnit(file_stem, inst_impls)
                    )
                    if execution_impls:
                        source_impl_units.execution.append(
                            _SourceImplUnit(file_stem, execution_impls)
                        )

                # Build include lists for .cpp files
                cpp_includes = [
                    (
                        self.config.generated_include(
                            self.generated_dir_name,
                            f'{enc.fmt_enc_name.lower()}.h',
                        ),
                        False,
                    ),
                    ('rocjitsu/isa/arch/amdgpu/shared/simd_glue.h', False),
                    ('rocjitsu/isa/arch/amdgpu/shared/fp_mode.h', False),
                    ('util/except.h', False),
                ]
                _MEM_ENC_NAMES = frozenset(
                    {
                        'ENC_SMEM',
                        'ENC_FLAT',
                        'ENC_MUBUF',
                        'ENC_MTBUF',
                        'ENC_DS',
                        # RDNA4 renamed/new memory encodings
                        'ENC_VFLAT',
                        'ENC_VGLOBAL',
                        'ENC_VSCRATCH',
                        'ENC_VDS',
                        'ENC_VBUFFER',
                    }
                )
                is_mem_enc = enc.enc_name.upper() in _MEM_ENC_NAMES
                if is_mem_enc:
                    cpp_includes.extend(
                        [
                            (
                                self.config.handwritten_include(
                                    self.handwritten_dir_name, 'addr_calc.h'
                                ),
                                False,
                            ),
                            (
                                'rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h',
                                False,
                            ),
                        ]
                    )
                    for cf_inc in self._cache_flags_includes():
                        cpp_includes.append((cf_inc, False))
                    cpp_includes.extend(
                        [
                            ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                            ('rocjitsu/vm/amdgpu/mem_state.h', False),
                            ('cstring', True),
                            ('memory', True),
                        ]
                    )
                has_matrix_exec = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == 'mfma'
                    for i in all_insts
                )
                requires_compute_unit = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class
                    in (
                        'scalar_shader_cycles',
                        'scalar_sendmsg_rtn',
                        'vector_cvt_scale',
                    )
                    for i in all_insts
                )
                if requires_compute_unit:
                    cpp_includes.append(('rocjitsu/vm/amdgpu/compute_unit.h', False))
                if has_matrix_exec:
                    cpp_includes.append(
                        (
                            self.config.handwritten_include(
                                self.handwritten_dir_name, 'mma_exec.h'
                            ),
                            False,
                        )
                    )
                _VOP_ENC_NAMES = frozenset(
                    {
                        'ENC_VOP1',
                        'ENC_VOP2',
                        'ENC_VOP3',
                        'ENC_VOP3P',
                        'ENC_VOPC',
                    }
                )
                if enc.enc_name.upper() in _VOP_ENC_NAMES:
                    cpp_includes.append(
                        (
                            'rocjitsu/isa/arch/amdgpu/shared/transcendental.h',
                            False,
                        )
                    )
                uses_register_access = any(
                    'RegisterAccess' in str(impl)
                    for impl in class_func_impls.model + class_func_impls.execution
                )
                uses_pseudo_scalar = any(
                    'pseudo_scalar::' in str(impl)
                    for impl in class_func_impls.model + class_func_impls.execution
                )
                if uses_pseudo_scalar:
                    cpp_includes.append(
                        (
                            'rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h',
                            False,
                        )
                    )
                if has_sem:
                    cpp_includes.extend(
                        [
                            ('rocjitsu/vm/amdgpu/wavefront.h', False),
                            ('util/data_types.h', False),
                            ('algorithm', True),
                            ('bit', True),
                            ('cmath', True),
                            ('limits', True),
                        ]
                    )
                    if any(
                        'std::optional' in str(impl)
                        for impl in class_func_impls.model + class_func_impls.execution
                    ):
                        cpp_includes.append(('optional', True))
                    if any(
                        'RJ_NOINLINE' in str(impl)
                        for impl in class_func_impls.execution
                    ):
                        cpp_includes.append(('rocjitsu/base/rj_compiler.h', False))
                if uses_register_access:
                    cpp_includes.append(('rocjitsu/vm/amdgpu/register_access.h', False))
                has_tensor_dma = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class
                    in ('tensor_load_to_lds', 'tensor_store_from_lds')
                    for i in all_insts
                )
                if has_tensor_dma:
                    cpp_includes.append(
                        ('rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h', False)
                    )
                # VOP encodings need DPP/SDWA helpers in execute_impl.
                if self._supports_vop_dpp_encoding(
                    enc.enc_name
                ) or enc.enc_name.upper() in (
                    'ENC_VOP1',
                    'ENC_VOP2',
                    'ENC_VOPC',
                ):
                    cpp_includes.append(
                        ('rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h', False)
                    )
                has_saveexec = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == 'scalar_saveexec'
                    for i in all_insts
                )
                if has_saveexec:
                    cpp_includes.extend(
                        [
                            ('util/log.h', False),
                            ('format', True),
                        ]
                    )
                has_hwreg_access = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and (
                        s.semantic_class
                        in (
                            'scalar_getreg',
                            'scalar_setreg',
                            'scalar_setreg_imm',
                        )
                    )
                    for i in all_insts
                )
                if has_hwreg_access and not is_mem_enc:
                    cpp_includes.extend(
                        [
                            ('rocjitsu/vm/amdgpu/hwreg.h', False),
                            ('util/log.h', False),
                        ]
                    )

                # Include the unified shared execute template header when
                # any instruction in this encoding delegates to a template.
                # Portable SIMD probes can delegate even without a shared
                # plan, so this cannot be gated solely on shared_plan.
                def _delegates_to_shared(i: Instruction) -> bool:
                    if not self.semantics or i.name not in self.semantics.instructions:
                        return False
                    return self._can_share_execute(
                        i.mnemonic, i, enc.enc_name
                    ) or self._can_force_shared_simd_probe(i, enc.enc_name)

                has_shared = any(_delegates_to_shared(i) for i in all_insts)
                if has_shared:
                    cpp_includes.append(
                        (
                            self.config.shared_generated_include('execute_shared.h'),
                            False,
                        )
                    )

                # Build per-ISA header includes.
                h_includes = [
                    (
                        self.config.generated_include(
                            self.generated_dir_name, 'encodings.h'
                        ),
                        False,
                    ),
                    (
                        self.config.handwritten_include(
                            self.handwritten_dir_name, 'isa.h'
                        ),
                        False,
                    ),
                    (
                        self.config.generated_include(
                            self.generated_dir_name, 'operand.h'
                        ),
                        False,
                    ),
                ]
                needs_compound_array = (
                    self._supports_cdna_mfma_f8f6f4_vop3px2()
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                )
                if (
                    self._supports_cdna5_scaled_wmma_vop3px2()
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    if not needs_compound_array:
                        h_includes.append(('array', True))
                    cpp_includes.append(('array', True))
                    inst_classes.append(
                        cgen.Line(self._emit_cdna5_scaled_wmma_vop3px2_class())
                    )
                    scaled_outputs = self._emit_cdna5_scaled_wmma_vop3px2_impls()
                    class_func_impls.model.extend(
                        cgen.Line(impl) for impl in scaled_outputs.model
                    )
                    class_func_impls.execution.extend(
                        cgen.Line(impl) for impl in scaled_outputs.execution
                    )

                if (
                    self._supports_cdna_mfma_f8f6f4_vop3px2()
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    h_includes.append(('array', True))
                    inst_classes.append(
                        cgen.Line(self._emit_cdna_mfma_f8f6f4_vop3px2_classes())
                    )
                    scaled_outputs = self._emit_cdna_mfma_f8f6f4_vop3px2_impls()
                    class_func_impls.model.extend(
                        cgen.Line(impl) for impl in scaled_outputs.model
                    )
                    class_func_impls.execution.extend(
                        cgen.Line(impl) for impl in scaled_outputs.execution
                    )

                inst_def_file = CppFile(
                    f'{enc.fmt_enc_name.lower()}',
                    self.out_path,
                    True,
                    h_includes,
                    [],
                    inst_classes,
                    self.cpp_namespace,
                    True,
                    generated_dir_name=self.generated_dir_name,
                )
                # No local f16 helpers needed - using util::f16_to_f32 etc.
                # from data_types.h (included via cpp_includes when has_sem).

                if (
                    self.isa_spec.arch_name.lower() == 'cdna5'
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    matrix_outputs = self._emit_cdna5_matrix_fmt_helpers()
                    class_func_impls.model[0:0] = [
                        cgen.Line(impl) for impl in matrix_outputs.model
                    ]
                    class_func_impls.execution[0:0] = [
                        cgen.Line(impl) for impl in matrix_outputs.execution
                    ]

                if enc.enc_name.upper() in ('ENC_MUBUF', 'ENC_MTBUF'):
                    class_func_impls.model.insert(
                        0,
                        cgen.Line(
                            self._emit_buffer_vaddr_helpers(
                                'buffer_vaddr_bits',
                                'BufferMachineInst',
                                templated=True,
                            )
                        ),
                    )
                elif enc.enc_name.upper() == 'ENC_VBUFFER':
                    class_func_impls.model.insert(
                        0,
                        cgen.Line(
                            self._emit_buffer_vaddr_helpers(
                                'vbuffer_vaddr_bits',
                                'VbufferMachineInst',
                                templated=False,
                            )
                        ),
                    )
                elif enc.enc_name.upper() in ('ENC_VFLAT', 'ENC_VGLOBAL'):
                    class_func_impls.model.insert(
                        0, cgen.Line(self._emit_vflat_helpers())
                    )
                elif (
                    enc.enc_name.upper() == 'ENC_MIMG'
                    and self.isa_spec.profile.renders_gfx11_image_syntax
                ):
                    class_func_impls.model.insert(
                        0, cgen.Line(self._emit_gfx11_mimg_helpers())
                    )

                if (
                    self.isa_spec.arch_name.lower() in {'cdna2', 'cdna3', 'cdna4'}
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    class_func_impls.model.insert(
                        0, cgen.Line(self._emit_mfma_operand_helpers())
                    )
                    if self._supports_cdna_mfma_f8f6f4_vop3px2():
                        class_func_impls.model.insert(
                            0, cgen.Line(self._emit_cdna4_matrix_fmt_helpers())
                        )

                if is_smem:
                    direct_field = self.isa_spec.profile.smem_direct_offset_field
                    if direct_field is None:
                        # CDNA model: soffset_en / imm / soffset three-field logic.
                        smem_body = (
                            'namespace {\n'
                            'Operand make_smem_offset(const Smem::OpEncoding *enc) {\n'
                            '  // SOFFSET_EN and IMM are independent: SOFFSET_EN gates the\n'
                            '  // SGPR field, IMM gates the 21-bit immediate field.\n'
                            '  // When both are set the hardware adds SGPR + immediate;\n'
                            '  // we show the SGPR as the operand and the immediate as\n'
                            '  // an offset modifier.\n'
                            '  if (enc->soffset_en)\n'
                            '    return Operand(32, OperandType::OPR_SMEM_OFFSET, '
                            'static_cast<int>(enc->soffset));\n'
                            '  if (enc->imm)\n'
                            '    return Operand(32, OperandType::OPR_SIMM32, '
                            'static_cast<int>(enc->offset));\n'
                            # IMM=0, SOFFSET_EN=0: OFFSET[6:0] is the offset SGPR.
                            '  return Operand(32, OperandType::OPR_SMEM_OFFSET, '
                            'static_cast<int>(enc->offset & 0x7F));\n'
                            '}\n'
                            '} // namespace'
                        )
                    else:
                        # RDNA model: direct offset field (no soffset_en/imm).
                        smem_body = (
                            'namespace {\n'
                            'Operand make_smem_offset(const Smem::OpEncoding *enc) {\n'
                            f'  return Operand(32, OperandType::OPR_SIMM32, '
                            f'static_cast<int>(enc->{direct_field}));\n'
                            '}\n'
                            '} // namespace'
                        )
                    smem_offset_helper = cgen.Line(smem_body)
                    class_func_impls.model.insert(0, smem_offset_helper)

                # Split profiles get a model-only include surface. Unsplit
                # profiles still emit execution bodies into the model file and
                # therefore retain the full legacy include set.
                model_cpp_includes = cpp_includes
                if profile.split_execution_sources:
                    model_cpp_includes = [
                        cpp_includes[0],
                        (
                            self.config.generated_include(
                                self.generated_dir_name,
                                'execution_backend.h',
                            ),
                            False,
                        ),
                    ]
                    if is_mem_enc:
                        model_cpp_includes.extend(
                            (cache_flags_include, False)
                            for cache_flags_include in self._cache_flags_includes()
                        )
                    if enc.enc_name.upper() in _VOP_ENC_NAMES:
                        model_cpp_includes.append(
                            (
                                'rocjitsu/isa/arch/amdgpu/shared/'
                                'instruction_encoding.h',
                                False,
                            )
                        )
                    model_impl_text = '\n'.join(
                        str(impl) for impl in class_func_impls.model
                    )
                    if 'std::format' in model_impl_text:
                        model_cpp_includes.append(('format', True))
                    if 'std::array' in model_impl_text:
                        model_cpp_includes.append(('array', True))

                model_impl_text = '\n'.join(
                    str(impl) for impl in class_func_impls.model
                )
                if 'std::make_unique' in model_impl_text:
                    model_cpp_includes.append(('memory', True))

                inst_def_file.gen_code()
                self._write_inst_impl_files(
                    enc.enc_name,
                    f'{enc.fmt_enc_name.lower()}',
                    model_cpp_includes,
                    class_func_impls,
                    source_impl_units,
                    execution_cpp_includes=cpp_includes,
                )

        # Generate the umbrella insts.h header that includes all per-
        # encoding instruction headers. Only primary (non-alt) encodings
        # that have instructions generate their own files; alt encoding
        # instructions are merged into their parent's file.
        arch = self.cpp_namespace
        guard = f'ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_INSTS_H_'
        inc_base = self.config.generated_include(self.generated_dir_name)
        insts_h_lines = [
            CppFile._prologue_comment(),
            f'#ifndef {guard}\n#define {guard}\n\n',
        ]
        for enc in self.isa_spec.inst_encodings:
            all_enc_insts = list(enc.insts)
            for child in child_encs.get(enc.enc_name, []):
                all_enc_insts.extend(child.insts)
            if all_enc_insts and not profile.is_alt_encoding(enc.enc_name):
                insts_h_lines.append(
                    f'#include "{inc_base}/{enc.fmt_enc_name.lower()}.h"\n'
                )
        if self._supports_generated_vopd():
            insts_h_lines.append(f'#include "{inc_base}/vopd.h"\n')
        insts_h_lines.append(f'\n#endif // {guard}\n')
        insts_h_path = os.path.join(self.out_path, self.generated_dir_name, 'insts.h')
        with open(insts_h_path, 'w') as f:
            f.write(''.join(insts_h_lines))

        # Shared execute templates are written by the CLI after all ISAs
        # are processed, using the accumulated _shared_execute_bodies dict.
        # Individual ISA codegens just collect; they don't write.

    def _write_inst_impl_files(
        self,
        enc_name: str,
        base_name: str,
        model_cpp_includes: list[tuple[str, bool]],
        class_func_impls: _ImplOutputs,
        source_impl_units: _ImplOutputs | None = None,
        execution_cpp_includes: list[tuple[str, bool]] | None = None,
    ) -> None:
        """Write implementation files for one encoding, splitting when needed.

        Some generated encodings, notably gfx1250 VOP3/VOPC, are large enough
        to trip the repository's added-file size hook. Profiles can set a byte
        limit in ``source_split_max_bytes``. Profiles can also supply logical
        file stems for instructions; this emits ``<base>_<stem>.cpp`` chunks
        that keep related instructions together while removing stale
        split/unsplit files when the split decision changes.
        """
        self._write_inst_impl_file_set(
            enc_name,
            base_name,
            model_cpp_includes,
            class_func_impls.model,
            source_impl_units.model if source_impl_units else None,
        )
        if not self.isa_spec.profile.split_execution_sources:
            return

        execution_source_units = (
            source_impl_units.execution if source_impl_units else None
        )
        if not execution_source_units and source_impl_units:
            # Model and execution fragments use the same logical source stems.
            # Keep those stems available when the execution output becomes
            # empty so stale logical ``*_exec_<stem>.cpp`` chunks are removed.
            execution_source_units = source_impl_units.model
        self._write_inst_impl_file_set(
            enc_name,
            f'{base_name}_exec',
            execution_cpp_includes or model_cpp_includes,
            class_func_impls.execution,
            execution_source_units,
        )

    def _write_inst_impl_file_set(
        self,
        enc_name: str,
        base_name: str,
        cpp_includes: list[tuple[str, bool]],
        class_func_impls: list[object],
        source_impl_units: list[_SourceImplUnit] | None,
    ) -> None:
        """Write one model or execution implementation file set."""

        def includes_for(impls: list[object]) -> list[tuple[str, bool]]:
            uses_optional = any('std::optional' in str(impl) for impl in impls)
            uses_noinline = any('RJ_NOINLINE' in str(impl) for impl in impls)
            return [
                include
                for include in cpp_includes
                if (include != ('optional', True) or uses_optional)
                and (include != ('rocjitsu/base/rj_compiler.h', False) or uses_noinline)
            ]

        max_bytes = self.isa_spec.profile.source_split_max_bytes.get(enc_name.upper())
        arch_dir = os.path.join(self.out_path, self.generated_dir_name)
        if os.path.isdir(arch_dir):
            self._remove_generated_source_split_files(
                arch_dir, base_name, source_impl_units
            )

        if not class_func_impls:
            unsplit_file = os.path.join(arch_dir, f'{base_name}.cpp')
            if os.path.exists(unsplit_file):
                os.remove(unsplit_file)
            return

        def scoped_includes(impls: list[object]) -> list[tuple[str, bool]]:
            impl_text = '\n'.join(str(impl) for impl in impls)
            helper_tokens = {
                'optional': 'std::optional',
                'rocjitsu/base/rj_compiler.h': 'RJ_NOINLINE',
                'rocjitsu/isa/arch/amdgpu/shared/fp_mode.h': 'fp_mode::',
                'rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h': 'pseudo_scalar::',
            }
            result: list[tuple[str, bool]] = []
            for include, system in cpp_includes:
                tokens = helper_tokens.get(include)
                if tokens is None:
                    result.append((include, system))
                    continue
                if isinstance(tokens, str):
                    tokens = (tokens,)
                if any(token in impl_text for token in tokens):
                    result.append((include, system))
            return result

        if not max_bytes:
            CppFile(
                base_name,
                self.out_path,
                False,
                scoped_includes(class_func_impls),
                [],
                class_func_impls,
                self.cpp_namespace,
                generated_dir_name=self.generated_dir_name,
            ).gen_code()
            return

        use_logical_split = bool(
            source_impl_units and any(u.file_stem for u in source_impl_units)
        )

        chunks: list[list[object]] = []
        chunk_file_names: list[str] = []

        # CppFile adds a short prologue, include block, and namespace wrapper.
        # Leave margin under the profile limit so formatting and include growth
        # do not push a chunk over the added-file size hook.
        chunk_overhead = 16 * 1024
        if use_logical_split:
            logical_chunks = self._build_logical_source_chunks(
                base_name,
                class_func_impls,
                source_impl_units or [],
                max_bytes,
                chunk_overhead,
            )
            for file_name, chunk in logical_chunks:
                chunk_file_names.append(file_name)
                chunks.append(chunk)
        else:
            current_chunk: list[object] = []
            current_size = 0
            for impl in class_func_impls:
                impl_size = len(f'{impl}\n\n'.encode())
                if (
                    current_chunk
                    and current_size + impl_size + chunk_overhead > max_bytes
                ):
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                current_chunk.append(impl)
                current_size += impl_size

            if current_chunk:
                chunks.append(current_chunk)

        if len(chunks) > 1:
            unsplit_file = os.path.join(arch_dir, f'{base_name}.cpp')
            if os.path.exists(unsplit_file):
                os.remove(unsplit_file)

        for idx, chunk in enumerate(chunks):
            if len(chunks) == 1:
                file_name = base_name
            elif chunk_file_names:
                file_name = chunk_file_names[idx]
            else:
                file_name = f'{base_name}_part{idx + 1}'
            CppFile(
                file_name,
                self.out_path,
                False,
                scoped_includes(chunk),
                [],
                chunk,
                self.cpp_namespace,
                generated_dir_name=self.generated_dir_name,
            ).gen_code()

    @staticmethod
    def _sanitize_source_split_file_stem(stem: str | None) -> str:
        clean = re.sub(r'[^a-z0-9]+', '_', (stem or 'misc').lower()).strip('_')
        return clean or 'misc'

    @staticmethod
    def _source_impl_size(impls: list[object]) -> int:
        return sum(len(f'{impl}\n\n'.encode()) for impl in impls)

    def _chunk_source_impl_units(
        self,
        units: list[_SourceImplUnit],
        max_bytes: int,
        chunk_overhead: int,
    ) -> list[list[_SourceImplUnit]]:
        chunks: list[list[_SourceImplUnit]] = []
        current_chunk: list[_SourceImplUnit] = []
        current_size = 0
        for unit in units:
            unit_size = self._source_impl_size(unit.impls)
            if current_chunk and current_size + unit_size + chunk_overhead > max_bytes:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(unit)
            current_size += unit_size
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _build_logical_source_chunks(
        self,
        base_name: str,
        class_func_impls: list[object],
        source_impl_units: list[_SourceImplUnit],
        max_bytes: int,
        chunk_overhead: int,
    ) -> list[tuple[str, list[object]]]:
        """Return deterministic logical source chunks for a split encoding."""
        grouped_units: dict[str, list[_SourceImplUnit]] = {}
        unit_impl_ids = {id(impl) for unit in source_impl_units for impl in unit.impls}
        extra_impls = [
            impl for impl in class_func_impls if id(impl) not in unit_impl_ids
        ]

        logical_chunks: list[tuple[str, list[object]]] = []
        if extra_impls:
            extra_units = [_SourceImplUnit('support', [impl]) for impl in extra_impls]
            self._append_logical_source_chunks(
                logical_chunks,
                base_name,
                'support',
                extra_units,
                max_bytes,
                chunk_overhead,
            )

        for unit in source_impl_units:
            stem = self._sanitize_source_split_file_stem(unit.file_stem)
            grouped_units.setdefault(stem, []).append(unit)

        for stem, units in grouped_units.items():
            self._append_logical_source_chunks(
                logical_chunks, base_name, stem, units, max_bytes, chunk_overhead
            )

        self._assert_unique_source_file_names(
            file_name for file_name, _ in logical_chunks
        )
        return logical_chunks

    def _append_logical_source_chunks(
        self,
        logical_chunks: list[tuple[str, list[object]]],
        base_name: str,
        stem: str,
        units: list[_SourceImplUnit],
        max_bytes: int,
        chunk_overhead: int,
    ) -> None:
        unit_chunks = self._chunk_source_impl_units(units, max_bytes, chunk_overhead)
        for idx, unit_chunk in enumerate(unit_chunks):
            chunk_stem = stem if idx == 0 else f'{stem}_{idx + 1}'
            logical_chunks.append(
                (
                    f'{base_name}_{chunk_stem}',
                    [impl for chunk_unit in unit_chunk for impl in chunk_unit.impls],
                )
            )

    @classmethod
    def _is_generated_source_split_file(
        cls,
        base_name: str,
        filename: str,
        source_impl_units: list[_SourceImplUnit] | None,
    ) -> bool:
        if re.fullmatch(rf'{re.escape(base_name)}_part\d+\.cpp', filename):
            return True

        has_logical_stems = any(unit.file_stem for unit in source_impl_units or [])
        stems = {
            cls._sanitize_source_split_file_stem(unit.file_stem)
            for unit in source_impl_units or []
            if unit.file_stem or has_logical_stems
        }
        return any(
            re.fullmatch(
                rf'{re.escape(base_name)}_{re.escape(stem)}(?:_\d+)?\.cpp', filename
            )
            for stem in stems
        )

    @classmethod
    def _remove_generated_source_split_files(
        cls,
        arch_dir: str,
        base_name: str,
        source_impl_units: list[_SourceImplUnit] | None,
    ) -> None:
        for filename in os.listdir(arch_dir):
            if cls._is_generated_source_split_file(
                base_name, filename, source_impl_units
            ):
                os.remove(os.path.join(arch_dir, filename))

    @staticmethod
    def _assert_unique_source_file_names(file_names) -> None:
        seen: set[str] = set()
        for file_name in file_names:
            if file_name in seen:
                raise AssertionError(
                    f'duplicate generated source file name: {file_name}'
                )
            seen.add(file_name)

    def _write_shared_inst_files(
        self,
        shared_by_enc: dict[str, tuple[list, list, list, list]],
    ) -> None:
        """Write shared/<enc>.h for universal instruction classes.

        Universal instruction classes are emitted as header-only definitions
        in the ``rocjitsu::amdgpu`` namespace.  Per-ISA files pull them in
        with ``using amdgpu::ClassName;``.

        Note: the shared classes currently reference per-ISA types
        (``encodings.h``, ``isa.h``, ``operand.h``) from the last-processed
        ISA.  This is correct for compilation because all universal
        instructions have identical encoding layouts, and the per-ISA header
        that includes the shared header provides the correct ISA context.
        A future refactor could introduce shared encoding bases to eliminate this
        dependency.
        """
        shared_dir = os.path.join(self.out_path, 'shared')
        os.makedirs(shared_dir, exist_ok=True)

        for enc_name, (classes, impls, _, _) in sorted(shared_by_enc.items()):
            if not classes:
                continue
            guard = f'ROCJITSU_ISA_AMDGPU_SHARED_{enc_name.upper()}_H_'

            h_path = os.path.join(shared_dir, f'{enc_name}.h')
            with open(h_path, 'w') as f:
                f.write(self.prologue())
                f.write(f'#ifndef {guard}\n#define {guard}\n\n')
                # No #include directives here — this file is included inside
                # a namespace block.  All required headers (wavefront.h,
                # except.h, data_types.h, <cmath>, etc.) must be included
                # by the per-ISA header BEFORE the namespace opens.\n

                # No namespace — this file is included inside per-ISA
                # namespace rocjitsu::<isa> { ... }.

                # Emit class definitions.
                for cls in classes:
                    out = re.sub(r'^struct\s', 'class ', f'{cls}\n\n')
                    f.write(out)

                # Emit inline constructor + execute() bodies.
                for impl in impls:
                    f.write(f'inline {impl}\n\n')

                f.write(f'#endif // {guard}\n')

    def _write_shared_execute_templates(self) -> None:
        """Write shared/execute_<enc>.h with full template execute bodies.

        Each shared instruction gets a template function:
        ``template<typename Inst> inline void execute_<mnemonic>(Inst &inst, Wavefront &wf)``
        with the execute body modified to access operands through ``self.``.
        """
        import re as _re

        entries: list[tuple[str, str, str, bool, str | None]] = []
        for (
            mnemonic,
            enc_name_key,
        ), (
            inst,
            sem,
            body,
            enc_name,
            is_true16_vop3,
        ) in sorted(self._shared_execute_bodies.items()):
            enc_key = enc_name.lower().replace('enc_', '')
            mnemonic = f'{mnemonic}_{enc_key}'
            prefixed_body = body
            for opnd in inst.operands:
                pattern = rf'(?<!\.)(?<!\w){_re.escape(opnd.name)}\.'
                prefixed_body = _re.sub(pattern, f'inst.{opnd.name}.', prefixed_body)
                helper_arg_pattern = (
                    rf'(?<!\.)(?<!\w){_re.escape(opnd.name)}'
                    rf'(?=,\s*wf(?:,\s*lane)?(?:\s*[,)]))'
                )
                prefixed_body = _re.sub(
                    helper_arg_pattern, f'inst.{opnd.name}', prefixed_body
                )
                register_access_arg_pattern = (
                    rf'((?:(?:amdgpu::)?RegisterAccess\(wf\)|regs)\.'
                    rf'(?:read|write)_(?:scalar64|scalar|lane_pair32|lane64|lane|chunk)\()'
                    rf'{_re.escape(opnd.name)}(?=\s*[,)])'
                )
                prefixed_body = _re.sub(
                    register_access_arg_pattern,
                    rf'\1inst.{opnd.name}',
                    prefixed_body,
                )
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)inst_\.', 'inst.inst_.', prefixed_body
            )
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)inst_(?!\w)', 'inst.inst_', prefixed_body
            )
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)set_data\(', 'inst.set_data(', prefixed_body
            )
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)size_(?!\w)', 'inst.size()', prefixed_body
            )
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)mnemonic\(\)', 'inst.mnemonic()', prefixed_body
            )
            prefixed_body = prefixed_body.replace(
                'amdgpu::vop3_fp8_decode_e5m3(*this)',
                'amdgpu::vop3_fp8_decode_e5m3(inst)',
            )
            prefixed_body = _re.sub(
                r'\s*\(void\)wf;\s*(?://[^\n]*)?\n?', '\n', prefixed_body
            )
            clamp_template_arg = (
                'true' if sem.data_type in ('f16', 'f32', 'f64') else 'false'
            )
            prefixed_body = _re.sub(
                r'amdgpu::RegisterAccess\(wf\)\.write_lane\(\s*'
                r'(inst\.[A-Za-z_][A-Za-z0-9_]*),\s*lane,\s*',
                rf'sdwa::write_lane<{clamp_template_arg}>' r'(inst, wf, \1, lane, ',
                prefixed_body,
            )
            prefixed_body = _re.sub(
                r'amdgpu::RegisterAccess\(wf\)\.write_lane64\(\s*'
                r'(inst\.[A-Za-z_][A-Za-z0-9_]*),\s*lane,\s*',
                rf'sdwa::write_lane64<{clamp_template_arg}>' r'(inst, wf, \1, lane, ',
                prefixed_body,
            )
            prefixed_body = prefixed_body.replace(
                '  uint64_t exec = wf.exec();',
                '  uint64_t exec = dpp::execution_lane_mask(inst, wf);',
                1,
            )
            true16_special_ops = {
                # These forms need reversed-source or ternary 16-bit arithmetic
                # that the generic sema lowering cannot yet express without
                # losing the VOP3 true16 op_sel source/destination half rules.
                'v_ashrrev_i16_vop3': (
                    2,
                    ('auto v = static_cast<int16_t>(src1);',),
                    'static_cast<uint32_t>(static_cast<uint16_t>('
                    'v >> (static_cast<int16_t>(src0) & 15u)))',
                ),
                'v_lshlrev_b16_vop3': (
                    2,
                    (),
                    '(src1 << (src0 & 15u)) & 0xffffu',
                ),
                'v_lshrrev_b16_vop3': (
                    2,
                    (),
                    'src1 >> (src0 & 15u)',
                ),
                'v_mad_i16_vop3': (
                    3,
                    (
                        'int32_t a = static_cast<int16_t>(src0);',
                        'int32_t b = static_cast<int16_t>(src1);',
                        'int32_t c = static_cast<int16_t>(src2);',
                    ),
                    'static_cast<uint32_t>(static_cast<uint16_t>(a * b + c))',
                ),
                'v_mad_u16_vop3': (
                    3,
                    (),
                    '(src0 * src1 + src2) & 0xffffu',
                ),
            }
            if mnemonic in true16_special_ops and is_true16_vop3:
                src_count, setup, result_expr = true16_special_ops[mnemonic]
                src_lines = ''.join(
                    f'    uint32_t src{i} = read_vop3_true16_src(inst.src{i}, wf, lane, opsel, {i});\n'
                    for i in range(src_count)
                )
                setup_lines = ''.join(f'    {line}\n' for line in setup)
                prefixed_body = (
                    '  uint64_t exec = dpp::execution_lane_mask(inst, wf);\n'
                    '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {\n'
                    '    if (!(exec & (1ULL << lane)))\n'
                    '      continue;\n'
                    '    uint32_t opsel = vop3_opsel(inst.inst_);\n'
                    f'{src_lines}'
                    f'{setup_lines}'
                    f'    uint32_t result = {result_expr};\n'
                    '    write_vop3_true16_dst(inst.vdst, wf, lane, opsel, result, true);\n'
                    '  }'
                )
            result_kind = self._mask_result_kind(inst, sem, enc_name)
            entries.append(
                (
                    mnemonic,
                    prefixed_body,
                    sem.semantic_class,
                    is_true16_vop3,
                    result_kind,
                )
            )

        shared_dir = os.path.join(self.out_path, 'shared')
        os.makedirs(shared_dir, exist_ok=True)

        from amdisa.codegen.execute.simd_codegen import (
            simd_extra_includes,
            simd_probe_line,
        )

        guard = 'ROCJITSU_ISA_AMDGPU_SHARED_EXECUTE_SHARED_H_'
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f'#ifndef {guard}',
            f'#define {guard}',
            '',
            '#include "rocjitsu/vm/amdgpu/wavefront.h"',
            '#include "rocjitsu/vm/amdgpu/compute_unit.h"',
            '#include "rocjitsu/vm/amdgpu/mem_state.h"',
            '#include "rocjitsu/vm/amdgpu/register_access.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"',
            *simd_extra_includes(),
            '#include "util/data_types.h"',
            '#include "util/except.h"',
            '#include "util/log.h"',
            '#include <algorithm>',
            '#include <bit>',
            '#include <cmath>',
            '#include <functional>',
            '#include <limits>',
            '#include <optional>',
            '',
            'namespace rocjitsu {',
            'namespace amdgpu {',
            '',
        ]

        for (
            mnemonic,
            prefixed_body,
            _sem_class,
            is_true16_vop3,
            result_kind,
        ) in entries:
            uses_result_writer = result_kind is not None
            lines.append(
                'template <typename Inst, typename CommitResult>'
                if uses_result_writer
                else 'template <typename Inst>'
            )
            result_parameter = (
                ', [[maybe_unused]] CommitResult commit_result'
                if uses_result_writer
                else ''
            )
            lines.append(
                f'inline void execute_{mnemonic}('
                f'[[maybe_unused]] Inst &inst, [[maybe_unused]] Wavefront &wf'
                f'{result_parameter}) {{'
            )
            probe = simd_probe_line(
                mnemonic,
                true16_vop3=is_true16_vop3,
                result_writer='commit_result' if uses_result_writer else None,
            )
            alu_classifiers = {
                'v_mul_f32_vop2': 'classify_mul_f32_vop2',
                'v_mul_f32_vop3': 'classify_mul_f32_vop3',
                'v_sqrt_f32_vop1': 'classify_sqrt_f32_vop1',
                'v_sqrt_f32_vop3': 'classify_sqrt_f32_vop3',
                'v_div_fixup_f32_vop3': 'classify_div_fixup_f32_exceptions',
                'v_rcp_iflag_f32_vop1': 'classify_rcp_iflag_f32_exceptions',
                'v_rcp_iflag_f32_vop3': 'classify_rcp_iflag_f32_exceptions',
            }
            classifier = alu_classifiers.get(mnemonic)
            if classifier is not None:
                lines.append(f'  uint32_t alu_causes = {classifier}(inst, wf);')
                # SIMD probes return directly after a successful fast-path
                # execution. Latch the architectural sticky status before any
                # such return; the classifier separately records the transient
                # per-instruction causes used for trap delivery.
                lines.append('  wf.set_trapsts(wf.trapsts() | alu_causes);')
            if probe is not None:
                if classifier is None:
                    lines.append(probe)
                else:
                    lines.append('  if (!alu_exception_trap_enables(wf)) {')
                    lines.append(probe.replace('  ', '    ', 1))
                    lines.append('  }')
            lines.append(prefixed_body)
            lines.append('}')
            lines.append('')

        lines.append('} // namespace amdgpu')
        lines.append('} // namespace rocjitsu')
        lines.append('')
        lines.append(f'#endif // {guard}')
        lines.append('')

        filepath = os.path.join(shared_dir, 'execute_shared.h')
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        import sys

        print(
            f'Generated shared/execute_shared.h with '
            f'{len(entries)} template functions',
            file=sys.stderr,
        )

    @staticmethod
    def _gen_narrow_cvt_header_REMOVED_PLACEHOLDER(out_path: str) -> None:
        return
        content_UNUSED = f"""\
// REMOVED — narrow FP conversions now live in util/data_types.h

#ifndef {guard}
#define {guard}

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace util {{

// LFSR for stochastic rounding seed advancement
inline uint32_t prng_advance(uint32_t seed) {{
  return (seed << 1) ^ ((seed >> 31) ? 197u : 0u);
}}

// --- FP8 E4M3 (OCP E4M3FN) with RNE rounding ---

inline uint8_t f32_to_fp8_e4m3_rne(float val) {{
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  if (std::isnan(val)) return static_cast<uint8_t>(sign | 0x7F);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF) return static_cast<uint8_t>(sign | 0x7E);
  int32_t exp = f_exp - 127 + 7;
  if (exp <= 0) {{
    if (exp < -3) return static_cast<uint8_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 23) return static_cast<uint8_t>(sign);
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 8) {{ result = 4; exp = 1; return static_cast<uint8_t>(sign | (1 << 3) | (result & 0x7)); }}
    return static_cast<uint8_t>(sign | (result & 0x7));
  }}
  if (exp >= 15) {{
    uint32_t mant = (f_mant >> 20) & 0x7;
    if (exp > 15 || (exp == 15 && mant >= 7))
      return static_cast<uint8_t>(sign | 0x7E);
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
  }}
  uint32_t round_bit = (f_mant >> 19) & 1;
  uint32_t sticky = (f_mant & 0x7FFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 20) & 0x7;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x7) {{
    mant = 0;
    exp += 1;
    if (exp >= 15) {{
      if (exp > 15) return static_cast<uint8_t>(sign | 0x7E);
      return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3));
    }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}}

inline uint8_t f32_to_fp8_e4m3_sr(float val, uint32_t seed) {{
  if (std::isnan(val)) return static_cast<uint8_t>((std::bit_cast<uint32_t>(val) >> 24) & 0x80) | 0x7F;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF) return static_cast<uint8_t>(sign | 0x7E);
  int32_t exp = f_exp - 127 + 7;
  if (exp <= 0) return static_cast<uint8_t>(sign);
  if (exp >= 15) return static_cast<uint8_t>(sign | 0x7E);
  uint32_t trunc_bits = f_mant & 0xFFFFF;
  uint32_t random_add = seed >> 12;
  uint32_t mant = (f_mant >> 20) & 0x7;
  if ((trunc_bits + random_add) > 0xFFFFF) {{
    mant += 1;
    if (mant > 0x7) {{ mant = 0; exp += 1; if (exp >= 15) return static_cast<uint8_t>(sign | 0x7E); }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}}

// --- BF8 E5M2 with RNE rounding ---

inline uint8_t f32_to_bf8_e5m2_rne(float val) {{
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  if (std::isnan(val)) return static_cast<uint8_t>(sign | 0x7F);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF) return static_cast<uint8_t>(sign | 0x7C);
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) {{
    if (exp < -1) return static_cast<uint8_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 23) return static_cast<uint8_t>(sign);
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 4) {{ return static_cast<uint8_t>(sign | (1 << 2) | 0); }}
    return static_cast<uint8_t>(sign | (result & 0x3));
  }}
  if (exp >= 31) return static_cast<uint8_t>(sign | 0x7C);
  uint32_t round_bit = (f_mant >> 20) & 1;
  uint32_t sticky = (f_mant & 0xFFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 21) & 0x3;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3) {{
    mant = 0; exp += 1;
    if (exp >= 31) return static_cast<uint8_t>(sign | 0x7C);
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}}

inline uint8_t f32_to_bf8_e5m2_sr(float val, uint32_t seed) {{
  if (std::isnan(val)) return static_cast<uint8_t>((std::bit_cast<uint32_t>(val) >> 24) & 0x80) | 0x7F;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF) return static_cast<uint8_t>(sign | 0x7C);
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) return static_cast<uint8_t>(sign);
  if (exp >= 31) return static_cast<uint8_t>(sign | 0x7C);
  uint32_t trunc_bits = f_mant & 0x1FFFFF;
  uint32_t random_add = seed >> 11;
  uint32_t mant = (f_mant >> 21) & 0x3;
  if ((trunc_bits + random_add) > 0x1FFFFF) {{
    mant += 1;
    if (mant > 0x3) {{ mant = 0; exp += 1; if (exp >= 31) return static_cast<uint8_t>(sign | 0x7C); }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}}

// --- FP4 E2M1 (bias=1, no NaN/Inf, max=6.0) ---

inline float fp4_e2m1_to_f32(uint8_t v) {{
  uint32_t sign = (v >> 3) & 1;
  uint32_t exp = (v >> 1) & 0x3;
  uint32_t mant = v & 0x1;
  if (exp == 0 && mant == 0) return std::bit_cast<float>(sign << 31);
  if (exp == 0) {{
    float result = 0.5f;
    return sign ? -result : result;
  }}
  uint32_t f = (sign << 31) | ((exp + 127 - 1) << 23) | (mant << 22);
  return std::bit_cast<float>(f);
}}

inline uint8_t f32_to_fp4_e2m1_rne(float val) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 28) & 0x8;
  if (std::isinf(val)) return static_cast<uint8_t>(sign | 0x7);
  float absval = std::fabs(val);
  if (absval > 6.0f) return static_cast<uint8_t>(sign | 0x7);
  if (absval < 0.25f) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {{
    uint32_t round_bit = (f_mant >> 22) & 1;
    uint32_t sticky = (f_mant & 0x3FFFFF) ? 1 : 0;
    uint32_t result = round_bit & (sticky | 0);
    return static_cast<uint8_t>(sign | result);
  }}
  if (exp > 3) return static_cast<uint8_t>(sign | 0x7);
  uint32_t round_bit = (f_mant >> 21) & 1;
  uint32_t sticky = (f_mant & 0x1FFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 22) & 0x1;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 1) {{
    mant = 0; exp += 1;
    if (exp > 3) return static_cast<uint8_t>(sign | 0x7);
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | mant);
}}

inline uint8_t f32_to_fp4_e2m1_sr(float val, uint32_t seed) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 28) & 0x8;
  if (std::isinf(val) || std::fabs(val) > 6.0f) return static_cast<uint8_t>(sign | 0x7);
  if (std::fabs(val) < 0.25f) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) return static_cast<uint8_t>(sign);
  if (exp > 3) return static_cast<uint8_t>(sign | 0x7);
  uint32_t trunc_bits = f_mant & 0x3FFFFF;
  uint32_t random_add = seed >> 10;
  uint32_t mant = (f_mant >> 22) & 0x1;
  if ((trunc_bits + random_add) > 0x3FFFFF) {{
    mant += 1;
    if (mant > 1) {{ mant = 0; exp += 1; if (exp > 3) return static_cast<uint8_t>(sign | 0x7); }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | mant);
}}

// --- FP6 E2M3 (bias=1, no NaN/Inf, max=7.5) ---

inline float fp6_e2m3_to_f32(uint8_t v) {{
  uint32_t sign = (v >> 5) & 1;
  uint32_t exp = (v >> 3) & 0x3;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0) return std::bit_cast<float>(sign << 31);
  if (exp == 0) {{
    float result = std::ldexp(static_cast<float>(mant), -3);
    return sign ? -result : result;
  }}
  uint32_t f = (sign << 31) | ((exp + 127 - 1) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}}

inline uint8_t f32_to_fp6_e2m3_rne(float val) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 7.5f) return static_cast<uint8_t>(sign | 0x1F);
  float absval = std::fabs(val);
  if (absval < std::ldexp(1.0f, -4)) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {{
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 23) return static_cast<uint8_t>(sign);
    uint32_t round_bit = (full_mant >> (shift - 1)) & 1;
    uint32_t sticky = (full_mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = full_mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 8) return static_cast<uint8_t>(sign | (1 << 3));
    return static_cast<uint8_t>(sign | (result & 0x7));
  }}
  if (exp > 3) return static_cast<uint8_t>(sign | 0x1F);
  uint32_t round_bit = (f_mant >> 19) & 1;
  uint32_t sticky = (f_mant & 0x7FFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 20) & 0x7;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x7) {{
    mant = 0; exp += 1;
    if (exp > 3) return static_cast<uint8_t>(sign | 0x1F);
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}}

inline uint8_t f32_to_fp6_e2m3_sr(float val, uint32_t seed) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 7.5f) return static_cast<uint8_t>(sign | 0x1F);
  if (std::fabs(val) < std::ldexp(1.0f, -4)) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) return static_cast<uint8_t>(sign);
  if (exp > 3) return static_cast<uint8_t>(sign | 0x1F);
  uint32_t trunc_bits = f_mant & 0xFFFFF;
  uint32_t random_add = seed >> 12;
  uint32_t mant = (f_mant >> 20) & 0x7;
  if ((trunc_bits + random_add) > 0xFFFFF) {{
    mant += 1;
    if (mant > 0x7) {{ mant = 0; exp += 1; if (exp > 3) return static_cast<uint8_t>(sign | 0x1F); }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}}

// --- BF6 E3M2 (bias=3, no NaN/Inf, max=28.0) ---

inline float bf6_e3m2_to_f32(uint8_t v) {{
  uint32_t sign = (v >> 5) & 1;
  uint32_t exp = (v >> 2) & 0x7;
  uint32_t mant = v & 0x3;
  if (exp == 0 && mant == 0) return std::bit_cast<float>(sign << 31);
  if (exp == 0) {{
    float result = std::ldexp(static_cast<float>(mant), -4);
    return sign ? -result : result;
  }}
  uint32_t f = (sign << 31) | ((exp + 127 - 3) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}}

inline uint8_t f32_to_bf6_e3m2_rne(float val) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 28.0f) return static_cast<uint8_t>(sign | 0x1F);
  float absval = std::fabs(val);
  if (absval < std::ldexp(1.0f, -5)) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 3;
  if (exp <= 0) {{
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 23) return static_cast<uint8_t>(sign);
    uint32_t round_bit = (full_mant >> (shift - 1)) & 1;
    uint32_t sticky = (full_mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = full_mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 4) return static_cast<uint8_t>(sign | (1 << 2));
    return static_cast<uint8_t>(sign | (result & 0x3));
  }}
  if (exp > 7) return static_cast<uint8_t>(sign | 0x1F);
  uint32_t round_bit = (f_mant >> 20) & 1;
  uint32_t sticky = (f_mant & 0xFFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 21) & 0x3;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3) {{
    mant = 0; exp += 1;
    if (exp > 7) return static_cast<uint8_t>(sign | 0x1F);
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}}

inline uint8_t f32_to_bf6_e3m2_sr(float val, uint32_t seed) {{
  if (std::isnan(val)) return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 28.0f) return static_cast<uint8_t>(sign | 0x1F);
  if (std::fabs(val) < std::ldexp(1.0f, -5)) return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 3;
  if (exp <= 0) return static_cast<uint8_t>(sign);
  if (exp > 7) return static_cast<uint8_t>(sign | 0x1F);
  uint32_t trunc_bits = f_mant & 0x1FFFFF;
  uint32_t random_add = seed >> 11;
  uint32_t mant = (f_mant >> 21) & 0x3;
  if ((trunc_bits + random_add) > 0x1FFFFF) {{
    mant += 1;
    if (mant > 0x3) {{ mant = 0; exp += 1; if (exp > 7) return static_cast<uint8_t>(sign | 0x1F); }}
  }}
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}}

// --- 6-bit packing/unpacking (32×6-bit values ↔ 6 DWORDs, little-endian) ---

inline void pack_6bit(const uint8_t vals[32], uint32_t dwords[6]) {{
  for (int d = 0; d < 6; ++d) dwords[d] = 0;
  for (int i = 0; i < 32; ++i) {{
    uint64_t bit_offset = static_cast<uint64_t>(i) * 6;
    uint32_t dw_idx = static_cast<uint32_t>(bit_offset / 32);
    uint32_t bit_pos = static_cast<uint32_t>(bit_offset % 32);
    uint32_t val6 = vals[i] & 0x3F;
    dwords[dw_idx] |= val6 << bit_pos;
    if (bit_pos > 26)
      dwords[dw_idx + 1] |= val6 >> (32 - bit_pos);
  }}
}}

inline void unpack_6bit(const uint32_t dwords[6], uint8_t vals[32]) {{
  for (int i = 0; i < 32; ++i) {{
    uint64_t bit_offset = static_cast<uint64_t>(i) * 6;
    uint32_t dw_idx = static_cast<uint32_t>(bit_offset / 32);
    uint32_t bit_pos = static_cast<uint32_t>(bit_offset % 32);
    uint32_t val = (dwords[dw_idx] >> bit_pos) & 0x3F;
    if (bit_pos > 26)
      val |= (dwords[dw_idx + 1] << (32 - bit_pos)) & 0x3F;
    vals[i] = static_cast<uint8_t>(val);
  }}
}}

}} // namespace util

#endif // {guard}
"""
        filepath = os.path.join(shared_dir, 'narrow_cvt.h')
        with open(filepath, 'w') as f:
            f.write(content)

        import sys

        print(f'Generated shared/narrow_cvt.h', file=sys.stderr)

    def gen_operand_types(self) -> None:
        """Generate operand type and OpSel enums."""
        self._validate_shared_scalar_pair_selector_contract()
        code_lines = []
        opnd_type_enum = 'enum class OperandType {'
        for opnd_type in self.isa_spec.operand_types:
            opnd_type_enum += opnd_type + ','
        opnd_type_enum += '};'
        code_lines.append(cgen.Line(opnd_type_enum))

        for opnd_sels in self.isa_spec.opnd_selectors:
            opnd_sel_name = ''.join(
                x.capitalize() for x in opnd_sels.operand_type.split('_')[1:]
            )
            opnd_sel_enum = f'enum OpSel{opnd_sel_name} {{'
            seen_names: set[str] = set()
            for opnd_sel_val in opnd_sels.op_sel_vals:
                if opnd_sel_val[0] not in seen_names:
                    seen_names.add(opnd_sel_val[0])
                    opnd_sel_enum += f'{opnd_sel_val[0]} = {opnd_sel_val[1]},'
            opnd_sel_enum += '};'
            code_lines.append(cgen.Line(opnd_sel_enum))

        # Generate is_vgpr_operand_type() constexpr function.
        vgpr_types = [
            t
            for t in self.isa_spec.operand_types
            if self._operand_type_can_name_vgpr(t)
        ]
        if vgpr_types:
            fn = '[[nodiscard]] constexpr bool is_vgpr_operand_type(OperandType t) {'
            fn += ' switch (t) {'
            for t in vgpr_types:
                fn += f' case OperandType::{t}:'
            fn += ' return true;'
            fn += ' default: return false; } }'
            code_lines.append(cgen.Line(fn))

        # Generate is_immediate_type() constexpr function. Kept here (rather than
        # in a generated .cpp anonymous namespace) so both the model operand.cpp
        # (Operand::const_value) and the execution operand.cpp can use it.
        imm_types = [
            operand_type
            for operand_type in _IMMEDIATE_OPERAND_TYPES
            if operand_type in self.isa_spec.operand_types
        ]
        if imm_types:
            fn = '[[nodiscard]] constexpr bool is_immediate_type(OperandType t) {'
            fn += ' switch (t) {'
            for t in imm_types:
                fn += f' case OperandType::{t}:'
            fn += ' return true;'
            fn += ' default: return false; } }'
            code_lines.append(cgen.Line(fn))

        opnd_type_def_file = CppFile(
            'operand_types',
            self.out_path,
            True,
            [],
            [],
            code_lines,
            self.cpp_namespace,
            generated_dir_name=self.generated_dir_name,
        )
        opnd_type_def_file.gen_code()

    @staticmethod
    def _reg_class_for_prefix(prefix: str) -> str | None:
        """Map an MRISA register-name prefix to an ISA register class."""
        match prefix.lower():
            case 's':
                return 'RegClass::SGPR'
            case 'v':
                return 'RegClass::VGPR'
            case 'acc':
                return 'RegClass::ACC_VGPR'
            case 'ttmp':
                return 'RegClass::TTMP'
            case _:
                return None

    @staticmethod
    def _named_reg_ref(operand_name: str) -> tuple[str, int] | None:
        """Map a named special register to (RegClass, index) for to_register_ref."""
        name = operand_name.upper()
        if name.startswith('TTMP') and name[4:].isdigit():
            index = int(name[4:])
            if 0 <= index < 16:
                return ('RegClass::TTMP', index)
        match name:
            case 'EXEC' | 'EXEC_LO':
                return ('RegClass::EXEC', 0)
            case 'EXEC_HI':
                return ('RegClass::EXEC', 1)
            case _:
                return None

    def gen_operand(self) -> None:
        """Generate the ISA-specific Operand class with name resolution."""
        arch = self.cpp_namespace
        scalar_null_precedes_m0 = self.isa_spec.profile.scalar_null_precedes_m0
        uses_packed_16bit_sources = (
            self.isa_spec.profile.uses_packed_16bit_e32_source_selectors
        )

        switch_cases = []
        ref_switch_cases = []
        opnd_types_with_selectors = set()

        for opnd_sel in self.isa_spec.opnd_selectors:
            opnd_types_with_selectors.add(opnd_sel.operand_type)
            opsel_name = 'OpSel' + ''.join(
                x.capitalize() for x in opnd_sel.operand_type.split('_')[1:]
            )

            case_lines = []
            ref_case_lines = []
            for pattern in opnd_sel.name_patterns:
                if pattern.kind == OperandNamePattern.REG_RANGE:
                    case_lines.append(
                        f'if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && '
                        f'encoding_value_ <= {opsel_name}::{pattern.max_enum}) '
                        f'return reg_name("{pattern.prefix}", '
                        f'encoding_value_ - {opsel_name}::{pattern.min_enum}, size_bits_);'
                    )
                    reg_class = self._reg_class_for_prefix(pattern.prefix)
                    # Only register-file prefixes tracked by RegisterSet become
                    # register refs. Named special registers remain nullopt
                    # until a consumer needs special-register liveness.
                    if reg_class is not None:
                        ref_case_lines.append(
                            f'if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && '
                            f'encoding_value_ <= {opsel_name}::{pattern.max_enum}) '
                            f'return RegisterRef{{{reg_class}, static_cast<uint16_t>('
                            f'encoding_value_ - {opsel_name}::{pattern.min_enum}), reg_width}};'
                        )
                elif pattern.kind == OperandNamePattern.POS_INT:
                    case_lines.append(
                        f'if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && '
                        f'encoding_value_ <= {opsel_name}::{pattern.max_enum}) '
                        f'return std::to_string('
                        f'encoding_value_ - {opsel_name}::{pattern.min_enum});'
                    )
                elif pattern.kind == OperandNamePattern.NEG_INT:
                    case_lines.append(
                        f'if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && '
                        f'encoding_value_ <= {opsel_name}::{pattern.max_enum}) '
                        f'return std::to_string('
                        f'-(encoding_value_ - {opsel_name}::{pattern.min_enum} + 1));'
                    )
                elif pattern.kind == OperandNamePattern.FLOAT_CONST:
                    case_lines.append(
                        f'if (encoding_value_ == {opsel_name}::{pattern.enum_name}) '
                        f'return "{pattern.operand_name}";'
                    )
                elif pattern.kind == OperandNamePattern.NAMED:
                    case_lines.append(
                        f'if (encoding_value_ == {opsel_name}::{pattern.enum_name}) '
                        f'return "{pattern.operand_name}";'
                    )
                    named_ref = self._named_reg_ref(pattern.operand_name)
                    if named_ref is not None:
                        named_class, named_index = named_ref
                        ref_case_lines.append(
                            f'if (encoding_value_ == {opsel_name}::{pattern.enum_name}) '
                            f'return RegisterRef{{{named_class}, {named_index}, reg_width}};'
                        )
                elif pattern.kind == OperandNamePattern.LITERAL:
                    case_lines.append(
                        f'if (encoding_value_ == {opsel_name}::{pattern.enum_name}) '
                        f'return "literal";'
                    )

            case_lines.append('break;')
            case_body = ' '.join(case_lines)
            switch_cases.append(
                f'case OperandType::{opnd_sel.operand_type}: ' f'{{ {case_body} }}'
            )
            ref_case_lines.append('break;')
            ref_case_body = ' '.join(ref_case_lines)
            ref_switch_cases.append(
                f'case OperandType::{opnd_sel.operand_type}: ' f'{{ {ref_case_body} }}'
            )

        no_sel_types = [
            t for t in self.isa_spec.operand_types if t not in opnd_types_with_selectors
        ]
        for t in no_sel_types:
            if t == 'OPR_SIMM32':
                switch_cases.append(
                    f'case OperandType::{t}: '
                    f'return std::format("0x{{:x}}", static_cast<uint32_t>(encoding_value_));'
                )
            elif t == 'OPR_WAITCNT':
                wc = self.isa_spec.profile.waitcnt_decode
                switch_cases.append(
                    f'case OperandType::{t}: {{\n'
                    f'  {wc}'
                    f'  return std::format("vmcnt({{}}) expcnt({{}}) lgkmcnt({{}})", '
                    f'vmcnt, expcnt, lgkmcnt);\n'
                    f'}}'
                )
            elif (
                t == 'OPR_SENDMSG_RTN' and self.isa_spec.profile.sendmsg_return_symbolic
            ):
                switch_cases.append(
                    f'case OperandType::{t}: {{\n'
                    f'  switch (static_cast<uint32_t>(encoding_value_) & 0xFFFF) {{\n'
                    f'  case 128: return "sendmsg(MSG_RTN_GET_DOORBELL)";\n'
                    f'  case 129: return "sendmsg(MSG_RTN_GET_DDID)";\n'
                    f'  case 130: return "sendmsg(MSG_RTN_GET_TMA)";\n'
                    f'  case 131: return "sendmsg(MSG_RTN_GET_REALTIME)";\n'
                    f'  case 132: return "sendmsg(MSG_RTN_SAVE_WAVE)";\n'
                    f'  case 133: return "sendmsg(MSG_RTN_GET_TBA)";\n'
                    f'  case 134: return "sendmsg(MSG_RTN_GET_TBA_TO_PC)";\n'
                    f'  default: break;\n'
                    f'  }}\n'
                    f'  const uint32_t value = static_cast<uint32_t>(encoding_value_) & 0xFFFF;\n'
                    f'  if (value <= 0xFF) return std::format("sendmsg({{}}, 0, 0)", value);\n'
                    f'  return std::to_string(encoding_value_);\n'
                    f'}}'
                )
            else:
                switch_cases.append(
                    f'case OperandType::{t}: return std::to_string(encoding_value_);'
                )

        switch_body = '\n'.join(switch_cases)
        packed_16bit_name_check = ''
        if uses_packed_16bit_sources:
            packed_16bit_name_check = (
                'if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, encoding_value_))\n'
                '  return std::format("v{}.{}", packed->reg, packed->shift ? "h" : "l");\n'
                'if (auto packed = packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_))\n'
                '  return std::format("v{}.{}", packed->reg, packed->shift ? "h" : "l");\n'
            )
        name_impl = (
            f'std::string Operand::name() const {{\n'
            f'if (has_literal64_)\n'
            f'  return std::format("0x{{:x}}", literal64_value_);\n'
            f'if (has_literal16_display_)\n'
            f'  return std::format("0x{{:x}}", literal16_display_value_);\n'
            f'{packed_16bit_name_check}'
            f'switch (opr_type_) {{\n'
            f'{switch_body}\n'
            f'}}\n'
            f'return std::to_string(encoding_value_);\n'
            f'}}'
        )

        ref_switch_body = '\n'.join(ref_switch_cases)
        packed_16bit_ref_check = ''
        if uses_packed_16bit_sources:
            packed_16bit_ref_check = (
                'if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, encoding_value_))\n'
                '  return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(packed->reg), reg_width};\n'
                'if (auto packed = packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_))\n'
                '  return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(packed->reg), reg_width};\n'
            )
        ref_impl = (
            f'std::optional<RegisterRef> Operand::to_register_ref() const {{\n'
            f'if (size_bits_ == 0)\n'
            f'  return std::nullopt;\n'
            f'// A fieldless operand (no MR ISA encoding field: a hardwired\n'
            f'// register/side effect like VCC/EXEC/SCC, or the fieldless image\n'
            f'// address) never denotes a def-use-tracked register: its\n'
            f'// encoding value is a fixed placeholder, not a decoded index, so\n'
            f'// mapping it to a RegisterRef would fabricate a spurious def/use.\n'
            f'// Making this explicit keeps every fieldless operand inert by\n'
            f'// design (not by per-type coincidence) if it is placed in the\n'
            f'// operand arrays.\n'
            f'if (fieldless_)\n'
            f'  return std::nullopt;\n'
            f'// Liveness tracks operands as contiguous 32-bit register lanes.\n'
            f'const auto reg_width = static_cast<uint8_t>(size_bits_ > 32 ? size_bits_ / 32 : 1);\n'
            f'{packed_16bit_ref_check}'
            f'switch (opr_type_) {{\n'
            f'{ref_switch_body}\n'
            f'default:\n'
            f'  break;\n'
            f'}}\n'
            f'return std::nullopt;\n'
            f'}}'
        )

        operand_ctor_decl = (
            '  Operand(int size_bits, OperandType opr_type, int encoding_value,\n'
            '          bool packed_16bit_source = false, bool packed_16bit_dst = false);\n'
            '  Operand(int size_bits, OperandType opr_type, unsigned short encoding_value,\n'
            '          bool packed_16bit_source, bool packed_16bit_dst = false);\n'
            if uses_packed_16bit_sources
            else '  Operand(int size_bits, OperandType opr_type, int encoding_value);\n'
        )
        literal64_decl = (
            '  std::optional<uint64_t> literal64_value() const override;\n'
            '  std::optional<uint64_t> const_value() const override;\n'
        )
        simd_public_decl = ''
        simd_private_decl = ''
        execution_backend_public_decl = ''
        packed_16bit_field = ''
        if uses_packed_16bit_sources or self.isa_spec.profile.split_execution_sources:
            simd_public_decl = '  bool simd_capable() const override;\n'
            simd_private_decl = (
                '  void read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,\n'
                '                       uint32_t *out) const override;\n'
                '  void write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,\n'
                '                        const uint32_t *vals, uint64_t mask) const override;\n'
            )
        if uses_packed_16bit_sources:
            packed_16bit_field = (
                '  bool packed_16bit_source_ = false;\n'
                '  bool packed_16bit_dst_ = false;\n'
            )

        execution_decls = (
            f'{simd_public_decl}'
            'private:\n'
            f'{simd_private_decl}'
            '  uint32_t read_scalar(const amdgpu::Wavefront &wf) const override;\n'
            '  uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
            '  void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const override;\n'
            '  void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const override;\n'
            '  uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
            '  void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const override;\n'
            '  uint64_t read_scalar64(const amdgpu::Wavefront &wf) const override;\n'
            '  void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const override;\n'
        )
        operand_base_decl = 'class Operand : public AmdgpuIsaOperand<Isa> {\n'
        operand_base_init = 'AmdgpuIsaOperand<Isa>'
        execution_backend_ctor_init = ''
        if self.isa_spec.profile.split_execution_sources:
            operand_base_decl = 'class Operand : public IsaOperand<Isa> {\n'
            operand_base_init = 'IsaOperand<Isa>'
            execution_backend_public_decl = (
                '  /// @brief Return the immutable full-simulator operand table.\n'
                '  static const void *full_execution_backend();\n'
                '  /// @brief Validate that every full-simulator operand callback is present.\n'
                '  static bool full_execution_backend_complete();\n'
            )
            execution_backend_ctor_init = (
                ',\n'
                '      execution_backend_(static_cast<const ExecutionBackend *>(\n'
                '          current_isa_operand_backend()))'
            )
            execution_decls += (
                '  std::optional<uint32_t> simd_vgpr_base_impl(const amdgpu::Wavefront &wf) const override;\n'
                '  std::optional<uint32_t> simd_vgpr_base_mut_impl(amdgpu::Wavefront &wf) const override;\n'
                '  amdgpu::ConstVgprStorage simd_vgpr_storage_impl(const amdgpu::Wavefront &wf) const override;\n'
                '  amdgpu::VgprStorage simd_vgpr_storage_mut_impl(amdgpu::Wavefront &wf) const override;\n'
                '  amdgpu::ConstVgprStoragePair64 simd_vgpr_storage64_impl(const amdgpu::Wavefront &wf) const override;\n'
                '  amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut_impl(amdgpu::Wavefront &wf) const override;\n'
                '  void simd_notify_read_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  void simd_notify_read_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  void simd_notify_read64_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  void simd_notify_read64_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  void simd_notify_write_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  void simd_notify_write64_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const override;\n'
                '  /// Same-image dispatch table populated by the execution TU before decode.\n'
                '  /// This is not a registration ABI between independently loaded DSOs.\n'
                '  struct ExecutionBackend {\n'
                '    bool (Operand::*simd_capable)() const = nullptr;\n'
                '    void (Operand::*read_lane_chunk)(const amdgpu::Wavefront &, uint32_t, uint32_t, uint32_t *) const = nullptr;\n'
                '    void (Operand::*write_lane_chunk)(amdgpu::Wavefront &, uint32_t, uint32_t, const uint32_t *, uint64_t) const = nullptr;\n'
                '    uint32_t (Operand::*read_scalar)(const amdgpu::Wavefront &) const = nullptr;\n'
                '    uint32_t (Operand::*read_lane)(const amdgpu::Wavefront &, uint32_t) const = nullptr;\n'
                '    void (Operand::*write_scalar)(amdgpu::Wavefront &, uint32_t) const = nullptr;\n'
                '    void (Operand::*write_lane)(amdgpu::Wavefront &, uint32_t, uint32_t) const = nullptr;\n'
                '    uint64_t (Operand::*read_lane64)(const amdgpu::Wavefront &, uint32_t) const = nullptr;\n'
                '    void (Operand::*write_lane64)(amdgpu::Wavefront &, uint32_t, uint64_t) const = nullptr;\n'
                '    uint64_t (Operand::*read_scalar64)(const amdgpu::Wavefront &) const = nullptr;\n'
                '    void (Operand::*write_scalar64)(amdgpu::Wavefront &, uint64_t) const = nullptr;\n'
                '    std::optional<uint32_t> (Operand::*simd_vgpr_base)(const amdgpu::Wavefront &) const = nullptr;\n'
                '    std::optional<uint32_t> (Operand::*simd_vgpr_base_mut)(amdgpu::Wavefront &) const = nullptr;\n'
                '    amdgpu::ConstVgprStorage (Operand::*simd_vgpr_storage)(const amdgpu::Wavefront &) const = nullptr;\n'
                '    amdgpu::VgprStorage (Operand::*simd_vgpr_storage_mut)(amdgpu::Wavefront &) const = nullptr;\n'
                '    amdgpu::ConstVgprStoragePair64 (Operand::*simd_vgpr_storage64)(const amdgpu::Wavefront &) const = nullptr;\n'
                '    amdgpu::VgprStoragePair64 (Operand::*simd_vgpr_storage64_mut)(amdgpu::Wavefront &) const = nullptr;\n'
                '    void (Operand::*simd_notify_read)(const amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '    void (Operand::*simd_notify_read_mut)(amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '    void (Operand::*simd_notify_read64)(const amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '    void (Operand::*simd_notify_read64_mut)(amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '    void (Operand::*simd_notify_write_mut)(amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '    void (Operand::*simd_notify_write64_mut)(amdgpu::Wavefront &, uint64_t, uint8_t) const = nullptr;\n'
                '  };\n'
                '  const ExecutionBackend *execution_backend_ = nullptr;\n'
                '  bool simd_capable_exec() const;\n'
                '  void read_lane_chunk_exec(const amdgpu::Wavefront &, uint32_t, uint32_t, uint32_t *) const;\n'
                '  void write_lane_chunk_exec(amdgpu::Wavefront &, uint32_t, uint32_t, const uint32_t *, uint64_t) const;\n'
                '  uint32_t read_scalar_exec(const amdgpu::Wavefront &) const;\n'
                '  uint32_t read_lane_exec(const amdgpu::Wavefront &, uint32_t) const;\n'
                '  void write_scalar_exec(amdgpu::Wavefront &, uint32_t) const;\n'
                '  void write_lane_exec(amdgpu::Wavefront &, uint32_t, uint32_t) const;\n'
                '  uint64_t read_lane64_exec(const amdgpu::Wavefront &, uint32_t) const;\n'
                '  void write_lane64_exec(amdgpu::Wavefront &, uint32_t, uint64_t) const;\n'
                '  uint64_t read_scalar64_exec(const amdgpu::Wavefront &) const;\n'
                '  void write_scalar64_exec(amdgpu::Wavefront &, uint64_t) const;\n'
                '  std::optional<uint32_t> simd_vgpr_base_exec(const amdgpu::Wavefront &) const;\n'
                '  std::optional<uint32_t> simd_vgpr_base_mut_exec(amdgpu::Wavefront &) const;\n'
                '  amdgpu::ConstVgprStorage simd_vgpr_storage_exec(const amdgpu::Wavefront &) const;\n'
                '  amdgpu::VgprStorage simd_vgpr_storage_mut_exec(amdgpu::Wavefront &) const;\n'
                '  amdgpu::ConstVgprStoragePair64 simd_vgpr_storage64_exec(const amdgpu::Wavefront &) const;\n'
                '  amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut_exec(amdgpu::Wavefront &) const;\n'
                '  void simd_notify_read_exec(const amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
                '  void simd_notify_read_mut_exec(amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
                '  void simd_notify_read64_exec(const amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
                '  void simd_notify_read64_mut_exec(amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
                '  void simd_notify_write_mut_exec(amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
                '  void simd_notify_write64_mut_exec(amdgpu::Wavefront &, uint64_t, uint8_t) const;\n'
            )

        class_def = [
            cgen.Line(
                f'{operand_base_decl}'
                'public:\n'
                '  enum class Literal32Widening { ZeroExtend, SignExtend, Replicate32, F64HighBits };\n'
                f'{operand_ctor_decl}'
                '  Operand(int size_bits, OperandType opr_type, int encoding_value,\n'
                '          uint16_t literal16_display_value, bool has_literal16_display);\n'
                '  Operand(int size_bits, OperandType opr_type, uint64_t literal64_value, bool is_literal64);\n'
                '  static Operand make_literal32(int size_bits, uint32_t literal_value, Literal32Widening widening);\n'
                '  std::string name() const override;\n'
                f'{literal64_decl}'
                '  std::optional<RegisterRef> to_register_ref() const override;\n'
                f'{execution_backend_public_decl}'
                f'{execution_decls}'
                '  uint64_t widened_literal32_value() const;\n'
                '  uint16_t literal16_display_value_ = 0;\n'
                '  bool has_literal16_display_ = false;\n'
                '  uint64_t literal64_value_ = 0;\n'
                '  bool has_literal64_ = false;\n'
                '  std::optional<Literal32Widening> literal32_widening_;\n'
                f'{packed_16bit_field}'
                '};'
            )
        ]

        operand_ctor_args = (
            'int size_bits, OperandType opr_type, int encoding_value, bool packed_16bit_source, '
            'bool packed_16bit_dst'
            if uses_packed_16bit_sources
            else 'int size_bits, OperandType opr_type, int encoding_value'
        )
        operand_ctor_init = (
            ',\n'
            '      packed_16bit_source_(packed_16bit_source),\n'
            '      packed_16bit_dst_(packed_16bit_dst)'
            if uses_packed_16bit_sources
            else ''
        )
        selector_validation_cases = []
        for selector in sorted(
            self.isa_spec.opnd_selectors, key=lambda item: item.operand_type
        ):
            operand_type = selector.operand_type
            if not self._selector_constructors_use_canonical_values(operand_type):
                continue
            if operand_type == 'OPR_SSRC_LANESEL':
                error = 'InvalidLaneSelector'
            elif operand_type.startswith('OPR_SREG'):
                error = 'InvalidScalarRegisterSelector'
            elif operand_type.startswith('OPR_SDST'):
                error = 'InvalidSelector'
            elif operand_type == 'OPR_EXEC':
                error = 'InvalidExecSelector'
            elif operand_type == 'OPR_SRC_VGPR':
                error = 'InvalidVgprSourceSelector'
            elif operand_type.startswith('OPR_SSRC'):
                error = 'InvalidScalarSourceSelector'
            else:
                error = 'InvalidSelector'
            intervals = self._operand_selector_intervals(operand_type)
            if not intervals:
                continue
            valid_expr = ' || '.join(
                f'(encoding_value >= {lo} && encoding_value <= {hi})'
                for lo, hi in intervals
            )
            selector_validation_cases.append(
                f'  case OperandType::{operand_type}:\n'
                f'    if (!({valid_expr}))\n'
                f'      defer_encoding_error(EncodingError::{error});\n'
                f'    break;\n'
            )
        selector_validation = (
            '  switch (opr_type) {\n'
            + ''.join(selector_validation_cases)
            + '  default:\n'
            + '    break;\n'
            + '  }\n'
        )
        packed_16bit_ctor_impl = []
        if uses_packed_16bit_sources:
            packed_16bit_ctor_impl.append(
                cgen.Line(
                    'Operand::Operand(int size_bits, OperandType opr_type, unsigned short encoding_value,\n'
                    '                 bool packed_16bit_source, bool packed_16bit_dst)\n'
                    '    : Operand(size_bits, opr_type, static_cast<int>(encoding_value),\n'
                    '              packed_16bit_source, packed_16bit_dst) {\n'
                    '}'
                )
            )
        literal64_impl = (
            'std::optional<uint64_t> Operand::literal64_value() const {\n'
            '  if (!has_literal64_)\n'
            '    return std::nullopt;\n'
            '  return literal64_value_;\n'
            '}'
        )
        # Wavefront-free constant value: the register-state-free subset of
        # read_scalar(). Registers are not constants and return nullopt, which
        # also keeps inline-const resolution from misreading a raw register
        # index as a small immediate. This is a model virtual (it participates in
        # the Operand vtable), so it must live in the model translation unit.
        const_value_impl = (
            'std::optional<uint64_t> Operand::const_value() const {\n'
            '  if (has_literal64_)\n'
            '    return literal64_value_;\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint64_t>(static_cast<uint32_t>(encoding_value_));\n'
            '  if (to_register_ref())\n'
            '    return std::nullopt;\n'
            '  return amdgpu::resolve_src_scalar_statically(encoding_value_);\n'
            '}'
        )
        class_impl = _ImplOutputs(
            model=[
                cgen.Line(
                    f'Operand::Operand({operand_ctor_args})\n'
                    f'    : {operand_base_init}(size_bits, opr_type, encoding_value)'
                    f'{execution_backend_ctor_init}'
                    f'{operand_ctor_init} {{\n'
                    f'{selector_validation}'
                    '  is_vgpr_ = is_vgpr_operand_type(opr_type);\n'
                    '}'
                ),
                *packed_16bit_ctor_impl,
                cgen.Line(
                    'Operand::Operand(int size_bits, OperandType opr_type, int encoding_value,\n'
                    '                 uint16_t literal16_display_value, bool has_literal16_display)\n'
                    '    : Operand(size_bits, opr_type, encoding_value) {\n'
                    '  literal16_display_value_ = literal16_display_value;\n'
                    '  has_literal16_display_ = has_literal16_display;\n'
                    '}'
                ),
                cgen.Line(
                    'Operand::Operand(int size_bits, OperandType opr_type, uint64_t literal64_value, bool is_literal64)\n'
                    f'    : {operand_base_init}(size_bits, opr_type, static_cast<int>(literal64_value))'
                    f'{execution_backend_ctor_init},\n'
                    '      literal64_value_(literal64_value), has_literal64_(is_literal64) {\n'
                    '  is_vgpr_ = is_vgpr_operand_type(opr_type);\n'
                    '}'
                ),
                cgen.Line(
                    'Operand Operand::make_literal32(int size_bits, uint32_t literal_value, Literal32Widening widening) {\n'
                    '  Operand operand(size_bits, OperandType::OPR_SIMM32, static_cast<int>(literal_value));\n'
                    '  operand.literal32_widening_ = widening;\n'
                    '  return operand;\n'
                    '}'
                ),
                cgen.Line(
                    'uint64_t Operand::widened_literal32_value() const {\n'
                    '  uint32_t literal_value = static_cast<uint32_t>(encoding_value_);\n'
                    '  switch (*literal32_widening_) {\n'
                    '    case Literal32Widening::ZeroExtend:\n'
                    '      return literal_value;\n'
                    '    case Literal32Widening::SignExtend:\n'
                    '      return static_cast<uint64_t>(static_cast<int64_t>(\n'
                    '          static_cast<int32_t>(literal_value)));\n'
                    '    case Literal32Widening::Replicate32:\n'
                    '      return (static_cast<uint64_t>(literal_value) << 32) | literal_value;\n'
                    '    case Literal32Widening::F64HighBits:\n'
                    '      return static_cast<uint64_t>(literal_value) << 32;\n'
                    '  }\n'
                    '  return literal_value;\n'
                    '}'
                ),
                cgen.Line(literal64_impl),
                cgen.Line(const_value_impl),
                cgen.Line(name_impl),
                cgen.Line(ref_impl),
            ]
        )

        if self.isa_spec.profile.split_execution_sources:
            class_impl.model.append(cgen.Line(textwrap.dedent('''\
                    bool Operand::simd_capable() const {
                      decltype(ExecutionBackend::simd_capable) callback =
                          execution_backend_ ? execution_backend_->simd_capable : nullptr;
                      return callback ? (this->*callback)() : false;
                    }

                    void Operand::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base,
                                                  uint32_t count, uint32_t *out) const {
                      decltype(ExecutionBackend::read_lane_chunk) callback =
                          execution_backend_ ? execution_backend_->read_lane_chunk : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, lane_base, count, out);
                    }

                    void Operand::write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base,
                                                   uint32_t count, const uint32_t *vals,
                                                   uint64_t mask) const {
                      decltype(ExecutionBackend::write_lane_chunk) callback =
                          execution_backend_ ? execution_backend_->write_lane_chunk : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, lane_base, count, vals, mask);
                    }

                    uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::read_scalar) callback =
                          execution_backend_ ? execution_backend_->read_scalar : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      return (this->*callback)(wf);
                    }

                    uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {
                      decltype(ExecutionBackend::read_lane) callback =
                          execution_backend_ ? execution_backend_->read_lane : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      return (this->*callback)(wf, lane);
                    }

                    void Operand::write_scalar(amdgpu::Wavefront &wf, uint32_t val) const {
                      decltype(ExecutionBackend::write_scalar) callback =
                          execution_backend_ ? execution_backend_->write_scalar : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, val);
                    }

                    void Operand::write_lane(amdgpu::Wavefront &wf, uint32_t lane,
                                             uint32_t val) const {
                      decltype(ExecutionBackend::write_lane) callback =
                          execution_backend_ ? execution_backend_->write_lane : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, lane, val);
                    }

                    uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf,
                                                  uint32_t lane) const {
                      decltype(ExecutionBackend::read_lane64) callback =
                          execution_backend_ ? execution_backend_->read_lane64 : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      return (this->*callback)(wf, lane);
                    }

                    void Operand::write_lane64(amdgpu::Wavefront &wf, uint32_t lane,
                                               uint64_t val) const {
                      decltype(ExecutionBackend::write_lane64) callback =
                          execution_backend_ ? execution_backend_->write_lane64 : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, lane, val);
                    }

                    uint64_t Operand::read_scalar64(const amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::read_scalar64) callback =
                          execution_backend_ ? execution_backend_->read_scalar64 : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      return (this->*callback)(wf);
                    }

                    void Operand::write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const {
                      decltype(ExecutionBackend::write_scalar64) callback =
                          execution_backend_ ? execution_backend_->write_scalar64 : nullptr;
                      if (!callback)
                        throw std::logic_error("operand execution backend is not linked");
                      (this->*callback)(wf, val);
                    }

                    std::optional<uint32_t>
                    Operand::simd_vgpr_base_impl(const amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_base) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_base : nullptr;
                      return callback ? (this->*callback)(wf) : std::nullopt;
                    }

                    std::optional<uint32_t>
                    Operand::simd_vgpr_base_mut_impl(amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_base_mut) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_base_mut : nullptr;
                      return callback ? (this->*callback)(wf) : std::nullopt;
                    }

                    amdgpu::ConstVgprStorage
                    Operand::simd_vgpr_storage_impl(const amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_storage) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_storage : nullptr;
                      return callback ? (this->*callback)(wf) : amdgpu::ConstVgprStorage{};
                    }

                    amdgpu::VgprStorage
                    Operand::simd_vgpr_storage_mut_impl(amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_storage_mut) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_storage_mut : nullptr;
                      return callback ? (this->*callback)(wf) : amdgpu::VgprStorage{};
                    }

                    amdgpu::ConstVgprStoragePair64
                    Operand::simd_vgpr_storage64_impl(const amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_storage64) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_storage64 : nullptr;
                      return callback ? (this->*callback)(wf)
                                      : amdgpu::ConstVgprStoragePair64{};
                    }

                    amdgpu::VgprStoragePair64
                    Operand::simd_vgpr_storage64_mut_impl(amdgpu::Wavefront &wf) const {
                      decltype(ExecutionBackend::simd_vgpr_storage64_mut) callback =
                          execution_backend_ ? execution_backend_->simd_vgpr_storage64_mut : nullptr;
                      return callback ? (this->*callback)(wf)
                                      : amdgpu::VgprStoragePair64{};
                    }

                    void Operand::simd_notify_read_impl(const amdgpu::Wavefront &wf,
                                                        uint64_t lane_mask,
                                                        uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_read) callback =
                              execution_backend_ ? execution_backend_->simd_notify_read : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }

                    void Operand::simd_notify_read_mut_impl(amdgpu::Wavefront &wf,
                                                            uint64_t lane_mask,
                                                            uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_read_mut) callback =
                              execution_backend_ ? execution_backend_->simd_notify_read_mut : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }

                    void Operand::simd_notify_read64_impl(const amdgpu::Wavefront &wf,
                                                          uint64_t lane_mask,
                                                          uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_read64) callback =
                              execution_backend_ ? execution_backend_->simd_notify_read64 : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }

                    void Operand::simd_notify_read64_mut_impl(amdgpu::Wavefront &wf,
                                                              uint64_t lane_mask,
                                                              uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_read64_mut) callback =
                              execution_backend_ ? execution_backend_->simd_notify_read64_mut
                                                 : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }

                    void Operand::simd_notify_write_mut_impl(amdgpu::Wavefront &wf,
                                                             uint64_t lane_mask,
                                                             uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_write_mut) callback =
                              execution_backend_ ? execution_backend_->simd_notify_write_mut
                                                 : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }

                    void Operand::simd_notify_write64_mut_impl(amdgpu::Wavefront &wf,
                                                               uint64_t lane_mask,
                                                               uint8_t byte_mask) const {
                      if (decltype(ExecutionBackend::simd_notify_write64_mut) callback =
                              execution_backend_ ? execution_backend_->simd_notify_write64_mut
                                                 : nullptr)
                        (this->*callback)(wf, lane_mask, byte_mask);
                    }
                    ''')))

        packed_16bit_helper = ''
        if uses_packed_16bit_sources:
            packed_16bit_helper = (
                '\n'
                'struct Packed16VgprSource {\n'
                '  uint32_t reg = 0;\n'
                '  uint32_t shift = 0;\n'
                '};\n'
                '\n'
                'std::optional<Packed16VgprSource> packed_16bit_vgpr_source(bool packed_16bit_source, int size_bits,\n'
                '                                                           OperandType opr_type, int ev) {\n'
                '  if (!packed_16bit_source || size_bits != 16)\n'
                '    return std::nullopt;\n'
                '  int selector_base;\n'
                '  if (opr_type == OperandType::OPR_VGPR)\n'
                '    selector_base = 0;\n'
                '  else if (opr_type == OperandType::OPR_SRC)\n'
                '    selector_base = 256;\n'
                '  else\n'
                '    return std::nullopt;\n'
                '  int selector = ev - selector_base;\n'
                '  if (selector >= 0 && selector <= 127)\n'
                '    return Packed16VgprSource{static_cast<uint32_t>(selector), 0};\n'
                '  if (selector >= 128 && selector <= 255)\n'
                '    return Packed16VgprSource{static_cast<uint32_t>(selector - 128), 16};\n'
                '  return std::nullopt;\n'
                '}\n'
                '\n'
                'std::optional<Packed16VgprSource> packed_16bit_vgpr_dst(bool packed_16bit_dst, int size_bits,\n'
                '                                                        OperandType opr_type, int ev) {\n'
                '  if (!packed_16bit_dst || size_bits != 16 || opr_type != OperandType::OPR_VGPR)\n'
                '    return std::nullopt;\n'
                '  if (ev >= 0 && ev <= 127)\n'
                '    return Packed16VgprSource{static_cast<uint32_t>(ev), 0};\n'
                '  if (ev >= 128 && ev <= 255)\n'
                '    return Packed16VgprSource{static_cast<uint32_t>(ev - 128), 16};\n'
                '  return std::nullopt;\n'
                '}\n'
            )
        reg_name_helper = cgen.Line(
            'namespace {\n'
            'std::string reg_name(const char *prefix, int reg_num, int size_bits) {\n'
            '  int count = size_bits / 32;\n'
            '  if (count <= 1)\n'
            '    return prefix + std::to_string(reg_num);\n'
            '  return std::string(prefix) + "[" + std::to_string(reg_num) + ":" +\n'
            '         std::to_string(reg_num + count - 1) + "]";\n'
            '}\n'
            f'{packed_16bit_helper}'
            '} // namespace'
        )
        class_impl.model.insert(0, reg_name_helper)

        # Operand value resolution (consolidated from operand_resolve.cpp)
        # Build ISA-dependent helper function bodies: only reference OperandType
        # values that actually exist in this ISA's generated enum.
        _opr = set(self.isa_spec.operand_types)
        _vgpr_only_parts = ['t == OperandType::OPR_VGPR']
        for _opt in (
            'OPR_VGPR_OR_ACCVGPR',
            'OPR_VGPR_OR_LDS',
            'OPR_SRC_VGPR',
            'OPR_ACCVGPR',
            'OPR_SRC_ACCVGPR',
            'OPR_SRC_VGPR_OR_ACCVGPR',
        ):
            if _opt in _opr:
                _vgpr_only_parts.append(f't == OperandType::{_opt}')
        _is_vgpr_only_body = (
            'bool is_vgpr_only_type(OperandType t) {\n'
            '  return ' + ' ||\n         '.join(_vgpr_only_parts) + ';\n'
            '}'
        )
        # AccVGPR offset within the unified VGPR block.  On CDNA3/4, AccVGPRs
        # live at indices 256-511 within each wavefront's VGPR allocation.  The
        # encoding values differ between destination (512-767) and source
        # (768-1023), but both map to the same physical offset range.
        _ACC_OFFSET = 256

        _vgpr_index_lines = ['uint32_t vgpr_index(OperandType opr_type, int ev) {']
        if 'OPR_VGPR_OR_ACCVGPR' in _opr:
            _vgpr_index_lines.append(
                '  if (opr_type == OperandType::OPR_VGPR || '
                'opr_type == OperandType::OPR_VGPR_OR_ACCVGPR) {\n'
                f'    if (ev >= OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN)\n'
                f'      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN);\n'
                '    return static_cast<uint32_t>(ev);\n'
                '  }'
            )
        else:
            _vgpr_index_lines.append(
                '  if (opr_type == OperandType::OPR_VGPR)\n'
                '    return static_cast<uint32_t>(ev);'
            )
        if 'OPR_ACCVGPR' in _opr:
            _vgpr_index_lines.append(
                '  if (opr_type == OperandType::OPR_ACCVGPR) {\n'
                f'    if (ev >= OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN)\n'
                f'      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN);\n'
                f'    return {_ACC_OFFSET} + static_cast<uint32_t>(ev);\n'
                '  }'
            )
        if 'OPR_SRC_ACCVGPR' in _opr:
            # AccVGPR source: maps to the AccVGPR bank at +256 offset.
            # v_accvgpr_read src0=256 (acc0) → physical index 256.
            # OpSel range 768+ also maps to +256 offset.
            _vgpr_index_lines.append(
                '  if (opr_type == OperandType::OPR_SRC_ACCVGPR) {\n'
                f'    if (ev >= OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN)\n'
                f'      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN);\n'
                f'    if (ev >= 256)\n'
                f'      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - 256);\n'
                f'    return {_ACC_OFFSET} + static_cast<uint32_t>(ev);\n'
                '  }'
            )
        if 'OPR_SRC_VGPR_OR_ACCVGPR' in _opr:
            _vgpr_index_lines.append(
                '  if (opr_type == OperandType::OPR_SRC_VGPR_OR_ACCVGPR) {\n'
                '    if (ev >= OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN)\n'
                f'      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN);\n'
                '    if (ev >= OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN)\n'
                '      return static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN);\n'
                '    return static_cast<uint32_t>(ev);\n'
                '  }'
            )
        _vgpr_index_lines.append('  return static_cast<uint32_t>(ev - 256);')
        _vgpr_index_lines.append('}')
        _vgpr_index_body = '\n'.join(_vgpr_index_lines)

        _read_immediate64_body = (
            'uint64_t read_immediate64(OperandType opr_type, int ev) {\n'
            '  if (opr_type == OperandType::OPR_SIMM32)\n'
            '    return static_cast<uint64_t>(static_cast<uint32_t>(ev));\n'
            '  return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ev)));\n'
            '}'
        )

        # Single source of truth for "does this operand resolve to per-lane VGPR
        # storage, and if so what's the offset within the wavefront's VGPR
        # allocation?". Used by read_lane/read_lane64/write_lane/write_lane64
        # and by every SIMD method below. Adding a new VGPR-bearing operand
        # type means editing this helper, not chasing dispatch through ~8
        # callsites.
        _resolved_vgpr_offset_lines = [
            'std::optional<uint32_t> Isa::resolved_vgpr_offset(OperandType opr_type, int ev) {',
            '  if (is_vgpr_only_type(opr_type))',
            '    return vgpr_index(opr_type, ev);',
            '  if (is_immediate_type(opr_type))',
            '    return std::nullopt;',
            '  if (ev >= 256 && ev <= 511)',
            '    return static_cast<uint32_t>(ev - 256);',
        ]
        if 'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST' in _opr:
            _resolved_vgpr_offset_lines.append(
                '  if (opr_type == OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST &&\n'
                '      ev >= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN &&\n'
                '      ev <= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MAX) {\n'
                f'    return {_ACC_OFFSET} + static_cast<uint32_t>(\n'
                '        ev - OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN);\n'
                '  }'
            )
        _resolved_vgpr_offset_lines.append('  return std::nullopt;')
        _resolved_vgpr_offset_lines.append('}')
        _resolved_vgpr_offset_body = '\n'.join(_resolved_vgpr_offset_lines)

        _resolved_vgpr_offset_with_wf_body = ''
        _resolved_vgpr_read_call = 'Isa::resolved_vgpr_offset(opr_type_, ev)'
        _resolved_vgpr_encoded_call = (
            'Isa::resolved_vgpr_offset(opr_type_, encoding_value_)'
        )
        if self.isa_spec.profile.uses_vgpr_msb_indexing:
            _resolved_vgpr_offset_with_wf_body = (
                '\n\n'
                'std::optional<uint32_t> Isa::resolved_vgpr_offset(const amdgpu::Wavefront &wf,\n'
                '                                                   OperandType opr_type, int ev,\n'
                '                                                   amdgpu::VgprMsbRole role) {\n'
                '  auto off = resolved_vgpr_offset(opr_type, ev);\n'
                '  if (!off)\n'
                '    return std::nullopt;\n'
                '  return *off + (wf.vgpr_msb_for_role(role) << 8);\n'
                '}'
            )
            _resolved_vgpr_read_call = (
                'Isa::resolved_vgpr_offset(wf, opr_type_, ev, vgpr_msb_role())'
            )
            _resolved_vgpr_encoded_call = 'Isa::resolved_vgpr_offset(wf, opr_type_, encoding_value_, vgpr_msb_role())'

        _inert_cond = '!reads_value()'

        def _inert_guard(ret_expr: str = '', cond: str = _inert_cond) -> str:
            # Benign-inert read/write for placeholder/metadata fieldless
            # operands, gated on construction-time capability flags. Read /
            # classify / SIMD paths use the default cond (!reads_value()).
            # Write accessors pass cond='!is_writable()'. Pass ret_expr=''
            # for void methods.
            ret_stmt = f'    return {ret_expr};\n' if ret_expr else '    return;\n'
            return (
                '  // Fieldless operands whose capability policy makes this\n'
                '  // accessor inert (reads yield a benign 0, writes are no-ops).\n'
                '  // Driven by the construction-time capability flags applied\n'
                '  // via apply_fieldless_caps() (see fieldless_policy.py).\n'
                f'  if ({cond})\n' + ret_stmt
            )

        read_lane_lines = [
            'uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {',
            '  if (delegate()) return amdgpu::RegisterAccess(wf).read_lane(*delegate(), lane);',
            *_inert_guard('0u').rstrip('\n').split('\n'),
            '  int ev = encoding_value_;',
        ]
        if uses_packed_16bit_sources:
            read_lane_lines.extend(
                [
                    '  if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, ev)) {',
                    '    uint32_t off = packed->reg + (wf.vgpr_msb_for_role(vgpr_msb_role()) << 8);',
                    '    uint32_t voff = amdgpu::apply_gpr_idx(wf, off, vgpr_msb_role());',
                    '    uint8_t byte_mask = packed->shift ? rocjitsu::ExecutionPlugin::kHighHalfByteMask : rocjitsu::ExecutionPlugin::kLowHalfByteMask;',
                    '    uint32_t raw = amdgpu::RegisterAccess(wf).read_vgpr(wf.vgpr_alloc().base + voff, lane, byte_mask);',
                    '    return (raw >> packed->shift) & 0xffffu;',
                    '  }',
                ]
            )
        read_lane_lines.extend(
            [
                f'  if (auto off = {_resolved_vgpr_read_call}) {{',
                '    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());',
                '    return amdgpu::RegisterAccess(wf).read_vgpr(wf.vgpr_alloc().base + voff, lane);',
                '  }',
                '  if (is_immediate_type(opr_type_))',
                '    return static_cast<uint32_t>(ev);',
                '  if (size_bits_ == 16)',
                '    return amdgpu::resolve_src_scalar16(wf, ev, kM0EncodingValue);',
                '  return amdgpu::resolve_src_scalar(wf, ev, kM0EncodingValue);',
                '}',
            ]
        )
        _read_lane_body = '\n'.join(read_lane_lines)

        _read_lane64_body = (
            'uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
            '  if (delegate()) return amdgpu::RegisterAccess(wf).read_lane64(*delegate(), lane);\n'
            + _inert_guard('0')
            + '  int ev = encoding_value_;\n'
            f'  if (auto off = {_resolved_vgpr_read_call}) {{\n'
            '    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());\n'
            '    uint32_t idx = wf.vgpr_alloc().base + voff;\n'
            '    return amdgpu::RegisterAccess(wf).read_vgpr64(idx, lane);\n'
            '  }\n'
            '  if (literal32_widening_)\n'
            '    return widened_literal32_value();\n'
            '  if (has_literal64_)\n'
            '    return literal64_value_;\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return read_immediate64(opr_type_, ev);\n'
            '  return amdgpu::resolve_src_scalar64(wf, ev, kM0EncodingValue);\n'
            '}'
        )

        simd_methods = ''
        if uses_packed_16bit_sources:
            simd_methods = textwrap.dedent('''\
                bool Operand::simd_capable() const {
                  if (delegate())
                    return delegate()->simd_capable();
                  if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, encoding_value_))
                    return false;
                  if (packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_))
                    return false;
                  return AmdgpuIsaOperand<Isa>::simd_capable();
                }

                void Operand::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                                              uint32_t *out) const {
                  if (delegate()) {
                    amdgpu::RegisterAccess(wf).read_chunk(*delegate(), lane_base, count, out);
                    return;
                  }
                  if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, encoding_value_)) {
                    for (uint32_t i = 0; i < count; ++i)
                      out[i] = read_lane(wf, lane_base + i);
                    return;
                  }
                detail::amdgpu_isa_read_lane_chunk_base(static_cast<const AmdgpuIsaOperand<Isa> &>(*this), wf, lane_base, count, out);
                }

                void Operand::write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                                               const uint32_t *vals, uint64_t mask) const {
                  if (delegate()) {
                    amdgpu::RegisterAccess(wf).write_chunk(*delegate(), lane_base, count, vals, mask);
                    return;
                  }
                  if (packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_)) {
                    for (uint32_t i = 0; i < count; ++i)
                      if (mask & (1ULL << i))
                        write_lane(wf, lane_base + i, vals[i]);
                    return;
                  }
                  detail::amdgpu_isa_write_lane_chunk_base(static_cast<const AmdgpuIsaOperand<Isa> &>(*this), wf, lane_base, count, vals, mask);
                }

                ''')
            if self.isa_spec.profile.split_execution_sources:
                simd_methods = textwrap.dedent('''\
                    bool Operand::simd_capable() const {
                      if (delegate())
                        return delegate()->simd_capable();
                      if (!reads_value())
                        return false;
                      if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_, encoding_value_))
                        return false;
                      if (packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_))
                        return false;
                      return Isa::simd_capable_value(opr_type_, encoding_value_);
                    }

                    void Operand::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base,
                                                  uint32_t count, uint32_t *out) const {
                      if (delegate()) {
                        amdgpu::RegisterAccess(wf).read_chunk(*delegate(), lane_base, count, out);
                        return;
                      }
                      if (!reads_value()) {
                        std::fill_n(out, count, 0u);
                        return;
                      }
                      assert(lane_base <= wf.wf_size());
                      assert(count <= wf.wf_size() - lane_base);
                      if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, opr_type_,
                                                   encoding_value_)) {
                        for (uint32_t i = 0; i < count; ++i)
                          out[i] = read_lane_exec(wf, lane_base + i);
                        return;
                      }
                      if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                        uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                        uint64_t lane_mask = count == 0 ? 0 : util::mask<uint64_t>(static_cast<int>(count)) << lane_base;
                        auto region = amdgpu::RegisterAccess(wf).read_vgpr_region(
                            wf.vgpr_alloc().base + voff, 1, lane_mask);
                        std::copy_n(region.lanes().begin() + lane_base, count, out);
                        return;
                      }
                      std::fill_n(out, count,
                                  Isa::simd_broadcast_value(wf, opr_type_, encoding_value_));
                    }

                    void Operand::write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base,
                                                   uint32_t count, const uint32_t *vals,
                                                   uint64_t mask) const {
                      if (!is_writable())
                        return;
                      assert(lane_base <= wf.wf_size());
                      assert(count <= wf.wf_size() - lane_base);
                      if (packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_,
                                                encoding_value_)) {
                        for (uint32_t i = 0; i < count; ++i)
                          if (mask & (1ULL << i))
                            write_lane_exec(wf, lane_base + i, vals[i]);
                        return;
                      }
                      auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this);
                      if (!off) {
                        for (uint32_t i = 0; i < count; ++i)
                          if (mask & (1ULL << i))
                            write_lane_exec(wf, lane_base + i, vals[i]);
                        return;
                      }
                      amdgpu::VgprMsbRole role =
                          vgpr_msb_role() == amdgpu::VgprMsbRole::None
                              ? amdgpu::VgprMsbRole::Dst
                              : vgpr_msb_role();
                      uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                      uint32_t reg = wf.vgpr_alloc().base + voff;
                      if (!raw_compute_unit(wf.cu()).owns_vgpr_range(wf, reg, 1))
                        return;
                      uint64_t full_mask = util::mask<uint64_t>(static_cast<int>(count));
                      uint8_t *dst = raw_compute_unit(wf.cu()).raw_vgpr_data(reg);
                      if ((mask & full_mask) == full_mask) {
                        std::memcpy(dst + lane_base * sizeof(uint32_t), vals,
                                    count * sizeof(uint32_t));
                        return;
                      }
                      for (uint32_t i = 0; i < count; ++i)
                        if (mask & (1ULL << i))
                          std::memcpy(dst + (lane_base + i) * sizeof(uint32_t), &vals[i],
                                      sizeof(uint32_t));
                    }

                    ''')
        elif self.isa_spec.profile.split_execution_sources:
            simd_methods = textwrap.dedent('''\
                bool Operand::simd_capable() const {
                  if (delegate())
                    return delegate()->simd_capable();
                  if (!reads_value())
                    return false;
                  return Isa::simd_capable_value(opr_type_, encoding_value_);
                }

                void Operand::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base,
                                              uint32_t count, uint32_t *out) const {
                  if (delegate()) {
                    amdgpu::RegisterAccess(wf).read_chunk(*delegate(), lane_base, count, out);
                    return;
                  }
                  if (!reads_value()) {
                    std::fill_n(out, count, 0u);
                    return;
                  }
                  assert(lane_base <= wf.wf_size());
                  assert(count <= wf.wf_size() - lane_base);
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    uint64_t lane_mask =
                        count == 0 ? 0
                                   : util::mask<uint64_t>(static_cast<int>(count)) << lane_base;
                    auto region = amdgpu::RegisterAccess(wf).read_vgpr_region(
                        wf.vgpr_alloc().base + voff, 1, lane_mask);
                    std::copy_n(region.lanes().begin() + lane_base, count, out);
                    return;
                  }
                  std::fill_n(out, count,
                              Isa::simd_broadcast_value(wf, opr_type_, encoding_value_));
                }

                void Operand::write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base,
                                               uint32_t count, const uint32_t *vals,
                                               uint64_t mask) const {
                  if (!is_writable())
                    return;
                  assert(lane_base <= wf.wf_size());
                  assert(count <= wf.wf_size() - lane_base);
                  auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this);
                  if (!off) {
                    for (uint32_t i = 0; i < count; ++i)
                      if (mask & (1ULL << i))
                        write_lane_exec(wf, lane_base + i, vals[i]);
                    return;
                  }
                  amdgpu::VgprMsbRole role =
                      vgpr_msb_role() == amdgpu::VgprMsbRole::None
                          ? amdgpu::VgprMsbRole::Dst
                          : vgpr_msb_role();
                  uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                  uint32_t reg = wf.vgpr_alloc().base + voff;
                  if (!raw_compute_unit(wf.cu()).owns_vgpr_range(wf, reg, 1))
                    return;
                  uint64_t full_mask = util::mask<uint64_t>(static_cast<int>(count));
                  uint8_t *dst = raw_compute_unit(wf.cu()).raw_vgpr_data(reg);
                  if ((mask & full_mask) == full_mask) {
                    std::memcpy(dst + lane_base * sizeof(uint32_t), vals,
                                count * sizeof(uint32_t));
                    return;
                  }
                  for (uint32_t i = 0; i < count; ++i)
                    if (mask & (1ULL << i))
                      std::memcpy(dst + (lane_base + i) * sizeof(uint32_t), &vals[i],
                                  sizeof(uint32_t));
                }

                ''')

        packed_16bit_write_lane_prefix = ''
        if uses_packed_16bit_sources:
            packed_16bit_write_lane_prefix = textwrap.dedent('''\
                  if (auto packed = packed_16bit_vgpr_dst(packed_16bit_dst_, size_bits_, opr_type_, encoding_value_)) {
                    uint32_t off = packed->reg + (wf.vgpr_msb_for_role(vgpr_msb_role()) << 8);
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, off, role);
                    uint32_t idx = wf.vgpr_alloc().base + voff;
                    uint8_t write_byte_mask = packed->shift ? rocjitsu::ExecutionPlugin::kHighHalfByteMask : rocjitsu::ExecutionPlugin::kLowHalfByteMask;
                    uint32_t placed = (val & 0xffffu) << packed->shift;
                    amdgpu::RegisterAccess(wf).write_vgpr(idx, lane, placed, write_byte_mask);
                    return;
                  }
                ''')

        resolve_code = cgen.Line(
            'namespace {\n'
            '\n'
            f'constexpr int kM0EncodingValue = '
            f'{125 if scalar_null_precedes_m0 else 124};\n'
            '\n'
            + _is_vgpr_only_body
            + '\n\n'
            + _vgpr_index_body
            + '\n\n'
            + _read_immediate64_body
            + '\n\n'
            + '\n'
            '} // namespace\n'
            '\n'
            '// ISA-scoped SIMD traits are used directly by split execution output.\n'
            '// Non-split profiles consume them through AmdgpuIsaOperand<Isa>.\n'
            + _resolved_vgpr_offset_body
            + _resolved_vgpr_offset_with_wf_body
            + '\n\n'
            'bool Isa::simd_capable_value(OperandType opr_type, int ev) {\n'
            '  return resolved_vgpr_offset(opr_type, ev).has_value() ||\n'
            '         is_immediate_type(opr_type) || amdgpu::can_resolve_src_scalar(ev, kM0EncodingValue);\n'
            '}\n'
            '\n'
            'uint32_t Isa::simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type,\n'
            '                                   int ev) {\n'
            '  return is_immediate_type(opr_type) ? static_cast<uint32_t>(ev)\n'
            '                                     : amdgpu::resolve_src_scalar(wf, ev, kM0EncodingValue);\n'
            '}\n'
            '\n'
            + simd_methods
            + 'uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {\n'
            '  if (delegate()) return amdgpu::RegisterAccess(wf).read_scalar(*delegate());\n'
            + _inert_guard('0u')
            + '  if (has_literal64_)\n'
            '    return static_cast<uint32_t>(literal64_value_);\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(encoding_value_);\n'
            '  return amdgpu::resolve_src_scalar(wf, encoding_value_, kM0EncodingValue);\n'
            '}\n'
            '\n' + _read_lane_body + '\n\n'
            'void Operand::write_scalar(amdgpu::Wavefront &wf, uint32_t val) const {\n'
            + _inert_guard(cond='!is_writable()')
            + '  amdgpu::resolve_dst_write(wf, encoding_value_, val, kM0EncodingValue);\n'
            '}\n'
            '\n'
            'void Operand::write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const {\n'
            + _inert_guard(cond='!is_writable()')
            + packed_16bit_write_lane_prefix
            + f'  if (auto off = {_resolved_vgpr_encoded_call}) {{\n'
            '    amdgpu::VgprMsbRole role = vgpr_msb_role() == amdgpu::VgprMsbRole::None\n'
            '                                    ? amdgpu::VgprMsbRole::Dst\n'
            '                                    : vgpr_msb_role();\n'
            '    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);\n'
            '    amdgpu::RegisterAccess(wf).write_vgpr(wf.vgpr_alloc().base + voff, lane, val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane called on non-VGPR operand type");\n'
            '}\n'
            '\n' + _read_lane64_body + '\n\n'
            'void Operand::write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const {\n'
            + _inert_guard(cond='!is_writable()')
            + f'  if (auto off = {_resolved_vgpr_encoded_call}) {{\n'
            '    amdgpu::VgprMsbRole role = vgpr_msb_role() == amdgpu::VgprMsbRole::None\n'
            '                                    ? amdgpu::VgprMsbRole::Dst\n'
            '                                    : vgpr_msb_role();\n'
            '    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);\n'
            '    uint32_t idx = wf.vgpr_alloc().base + voff;\n'
            '    amdgpu::RegisterAccess(wf).write_vgpr64(idx, lane, val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane64 called on non-VGPR operand type");\n'
            '}\n'
            '\n'
            'uint64_t Operand::read_scalar64(const amdgpu::Wavefront &wf) const {\n'
            + _inert_guard('0')
            + (
                '  if (literal32_widening_)\n'
                '    return widened_literal32_value();\n'
                '  if (has_literal64_)\n'
                '    return literal64_value_;\n'
                '  if (is_immediate_type(opr_type_))\n'
                '    return read_immediate64(opr_type_, encoding_value_);\n'
                '  return amdgpu::resolve_src_scalar64(wf, encoding_value_, kM0EncodingValue);\n'
                '}\n\n'
            )
            + 'void Operand::write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const {\n'
            + _inert_guard(cond='!is_writable()')
            + '  amdgpu::resolve_dst_write64(wf, encoding_value_, val);\n'
            '}'
        )
        if self.isa_spec.profile.split_execution_sources:
            execution_code = (
                f'namespace {{\n{packed_16bit_helper}\n}} // namespace\n\n'
                f'{resolve_code}'
            )
            for method in (
                'simd_capable',
                'read_lane_chunk',
                'write_lane_chunk',
                'read_scalar',
                'read_lane',
                'write_scalar',
                'write_lane',
                'read_lane64',
                'write_lane64',
                'read_scalar64',
                'write_scalar64',
            ):
                execution_code = execution_code.replace(
                    f'Operand::{method}(', f'Operand::{method}_exec('
                )
            execution_code += '\n\n' + textwrap.dedent('''\
                std::optional<uint32_t>
                Operand::simd_vgpr_base_exec(const amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this))
                    return wf.vgpr_alloc().base +
                           amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                  return std::nullopt;
                }

                std::optional<uint32_t>
                Operand::simd_vgpr_base_mut_exec(amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    return wf.vgpr_alloc().base + amdgpu::apply_gpr_idx(wf, *off, role);
                  }
                  return std::nullopt;
                }

                amdgpu::ConstVgprStorage
                Operand::simd_vgpr_storage_exec(const amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    const auto &cu = raw_compute_unit(wf.cu());
                    return {reinterpret_cast<const uint32_t *>(
                                cu.raw_vgpr_data(wf.vgpr_alloc().base + voff)),
                            cu.vgpr_storage_lane_count()};
                  }
                  return {};
                }

                amdgpu::VgprStorage
                Operand::simd_vgpr_storage_mut_exec(amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                    auto &cu = raw_compute_unit(wf.cu());
                    return {reinterpret_cast<uint32_t *>(
                                cu.raw_vgpr_data(wf.vgpr_alloc().base + voff)),
                            cu.vgpr_storage_lane_count()};
                  }
                  return {};
                }

                amdgpu::ConstVgprStoragePair64
                Operand::simd_vgpr_storage64_exec(const amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    uint32_t reg = wf.vgpr_alloc().base + voff;
                    const auto &cu = raw_compute_unit(wf.cu());
                    return {{reinterpret_cast<const uint32_t *>(cu.raw_vgpr_data(reg)),
                             cu.vgpr_storage_lane_count()},
                            {reinterpret_cast<const uint32_t *>(cu.raw_vgpr_data(reg + 1)),
                             cu.vgpr_storage_lane_count()}};
                  }
                  return {};
                }

                amdgpu::VgprStoragePair64
                Operand::simd_vgpr_storage64_mut_exec(amdgpu::Wavefront &wf) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                    uint32_t reg = wf.vgpr_alloc().base + voff;
                    auto &cu = raw_compute_unit(wf.cu());
                    return {{reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(reg)),
                             cu.vgpr_storage_lane_count()},
                            {reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(reg + 1)),
                             cu.vgpr_storage_lane_count()}};
                  }
                  return {};
                }

                void Operand::simd_notify_read_exec(const amdgpu::Wavefront &wf,
                                                    uint64_t lane_mask,
                                                    uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    raw_compute_unit(wf.cu()).notify_vgpr_read(
                        &wf, wf.vgpr_alloc().base + voff, lane_mask, byte_mask);
                  }
                }

                void Operand::simd_notify_read_mut_exec(amdgpu::Wavefront &wf,
                                                        uint64_t lane_mask,
                                                        uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    raw_compute_unit(wf.cu()).notify_vgpr_read(
                        &wf, wf.vgpr_alloc().base + voff, lane_mask, byte_mask);
                  }
                }

                void Operand::simd_notify_read64_exec(const amdgpu::Wavefront &wf,
                                                      uint64_t lane_mask,
                                                      uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    uint32_t reg = wf.vgpr_alloc().base + voff;
                    raw_compute_unit(wf.cu()).notify_vgpr_read(&wf, reg, lane_mask, byte_mask);
                    raw_compute_unit(wf.cu()).notify_vgpr_read(&wf, reg + 1, lane_mask, byte_mask);
                  }
                }

                void Operand::simd_notify_read64_mut_exec(amdgpu::Wavefront &wf,
                                                          uint64_t lane_mask,
                                                          uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, vgpr_msb_role());
                    uint32_t reg = wf.vgpr_alloc().base + voff;
                    raw_compute_unit(wf.cu()).notify_vgpr_read(&wf, reg, lane_mask, byte_mask);
                    raw_compute_unit(wf.cu()).notify_vgpr_read(&wf, reg + 1, lane_mask, byte_mask);
                  }
                }

                void Operand::simd_notify_write_mut_exec(amdgpu::Wavefront &wf,
                                                         uint64_t lane_mask,
                                                         uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                    raw_compute_unit(wf.cu()).notify_vgpr_write(
                        &wf, wf.vgpr_alloc().base + voff, lane_mask, byte_mask);
                  }
                }

                void Operand::simd_notify_write64_mut_exec(amdgpu::Wavefront &wf,
                                                           uint64_t lane_mask,
                                                           uint8_t byte_mask) const {
                  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
                    amdgpu::VgprMsbRole role =
                        vgpr_msb_role() == amdgpu::VgprMsbRole::None
                            ? amdgpu::VgprMsbRole::Dst
                            : vgpr_msb_role();
                    uint32_t voff = amdgpu::apply_gpr_idx(wf, *off, role);
                    uint32_t reg = wf.vgpr_alloc().base + voff;
                    raw_compute_unit(wf.cu()).notify_vgpr_write(&wf, reg, lane_mask, byte_mask);
                    raw_compute_unit(wf.cu()).notify_vgpr_write(&wf, reg + 1, lane_mask, byte_mask);
                  }
                }

                const void *Operand::full_execution_backend() {
                  static const ExecutionBackend backend{
                      &Operand::simd_capable_exec,
                      &Operand::read_lane_chunk_exec,
                      &Operand::write_lane_chunk_exec,
                      &Operand::read_scalar_exec,
                      &Operand::read_lane_exec,
                      &Operand::write_scalar_exec,
                      &Operand::write_lane_exec,
                      &Operand::read_lane64_exec,
                      &Operand::write_lane64_exec,
                      &Operand::read_scalar64_exec,
                      &Operand::write_scalar64_exec,
                      &Operand::simd_vgpr_base_exec,
                      &Operand::simd_vgpr_base_mut_exec,
                      &Operand::simd_vgpr_storage_exec,
                      &Operand::simd_vgpr_storage_mut_exec,
                      &Operand::simd_vgpr_storage64_exec,
                      &Operand::simd_vgpr_storage64_mut_exec,
                      &Operand::simd_notify_read_exec,
                      &Operand::simd_notify_read_mut_exec,
                      &Operand::simd_notify_read64_exec,
                      &Operand::simd_notify_read64_mut_exec,
                      &Operand::simd_notify_write_mut_exec,
                      &Operand::simd_notify_write64_mut_exec,
                  };
                  return &backend;
                }

                bool Operand::full_execution_backend_complete() {
                  const auto is_complete = [](const ExecutionBackend &backend) {
                    return backend.simd_capable != nullptr &&
                           backend.read_lane_chunk != nullptr &&
                           backend.write_lane_chunk != nullptr &&
                           backend.read_scalar != nullptr &&
                           backend.read_lane != nullptr &&
                           backend.write_scalar != nullptr &&
                           backend.write_lane != nullptr &&
                           backend.read_lane64 != nullptr &&
                           backend.write_lane64 != nullptr &&
                           backend.read_scalar64 != nullptr &&
                           backend.write_scalar64 != nullptr &&
                           backend.simd_vgpr_base != nullptr &&
                           backend.simd_vgpr_base_mut != nullptr &&
                           backend.simd_vgpr_storage != nullptr &&
                           backend.simd_vgpr_storage_mut != nullptr &&
                           backend.simd_vgpr_storage64 != nullptr &&
                           backend.simd_vgpr_storage64_mut != nullptr &&
                           backend.simd_notify_read != nullptr &&
                           backend.simd_notify_read_mut != nullptr &&
                           backend.simd_notify_read64 != nullptr &&
                           backend.simd_notify_read64_mut != nullptr &&
                           backend.simd_notify_write_mut != nullptr &&
                           backend.simd_notify_write64_mut != nullptr;
                  };
                  return is_complete(*static_cast<const ExecutionBackend *>(
                      full_execution_backend()));
                }
                ''')
            execution_code = execution_code.replace(
                'raw_compute_unit(wf.cu())',
                'amdgpu::OperandExecutionAccess::raw_compute_unit(wf.cu())',
            )
            resolve_code = cgen.Line(execution_code)
        class_impl.execution_target(
            self.isa_spec.profile.split_execution_sources
        ).append(resolve_code)

        operand_header_includes = [
            (
                self.config.handwritten_include(self.handwritten_dir_name, 'isa.h'),
                False,
            ),
            (
                self.config.generated_include(
                    self.generated_dir_name, 'operand_types.h'
                ),
                False,
            ),
            ('rocjitsu/isa/operand.h', False),
            ('string', True),
        ]
        if self.isa_spec.profile.split_execution_sources:
            operand_header_includes.append(('rocjitsu/isa/execution_backend.h', False))

        header_file = CppFile(
            'operand',
            self.out_path,
            True,
            operand_header_includes,
            [],
            class_def,
            arch,
            generated_dir_name=self.generated_dir_name,
        )
        header_file.gen_code()
        if self.isa_spec.profile.split_execution_sources:
            CppFile(
                'operand',
                self.out_path,
                False,
                [
                    (
                        self.config.generated_include(
                            self.generated_dir_name, 'operand.h'
                        ),
                        False,
                    ),
                    ('rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h', False),
                    ('format', True),
                    ('optional', True),
                    ('stdexcept', True),
                    ('string', True),
                ],
                [],
                class_impl.model,
                arch,
                generated_dir_name=self.generated_dir_name,
            ).gen_code()
            CppFile(
                'operand_exec',
                self.out_path,
                False,
                [
                    (
                        self.config.generated_include(
                            self.generated_dir_name, 'operand.h'
                        ),
                        False,
                    ),
                    ('rocjitsu/isa/isa_operand_simd_inl.h', False),
                    ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                    ('rocjitsu/vm/amdgpu/operand_execution_access.h', False),
                    ('rocjitsu/vm/amdgpu/register_access.h', False),
                    ('rocjitsu/vm/amdgpu/wavefront.h', False),
                    ('rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h', False),
                    ('format', True),
                    ('optional', True),
                    ('stdexcept', True),
                    ('string', True),
                ],
                [],
                class_impl.execution,
                arch,
                generated_dir_name=self.generated_dir_name,
            ).gen_code()
        else:
            CppFile(
                'operand',
                self.out_path,
                False,
                [
                    (
                        self.config.generated_include(
                            self.generated_dir_name, 'operand.h'
                        ),
                        False,
                    ),
                    ('rocjitsu/isa/isa_operand_simd_inl.h', False),
                    ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                    ('rocjitsu/vm/amdgpu/register_access.h', False),
                    ('rocjitsu/vm/amdgpu/wavefront.h', False),
                    ('rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h', False),
                    ('util/except.h', False),
                    ('format', True),
                    ('optional', True),
                    ('stdexcept', True),
                    ('string', True),
                ],
                [],
                class_impl.model,
                arch,
                generated_dir_name=self.generated_dir_name,
            ).gen_code()

    def _distributed_decoder_factories(self) -> dict[str, str]:
        """Return instruction classes whose trivial factories live with their model.

        Decoder tables contain both dispatch helpers and one-line instruction
        factories.  Keeping the latter in each encoding's existing model source
        distributes code generation without creating more translation units.
        """
        classes = {
            inst.fmt_name for enc in self.isa_spec.inst_encodings for inst in enc.insts
        }
        factory_names: set[str] = set()
        for dte in self.isa_spec.primary_decode_table:
            if dte is None:
                continue
            if dte.is_primary:
                factory_names.add(dte.decode_func)
            elif dte.sub_decode_funcs is not None:
                factory_names.update(dte.sub_decode_funcs)

        return {
            class_name: factory_name
            for class_name in sorted(classes)
            if (factory_name := f'decode{class_name}') in factory_names
        }

    def gen_decoder(self) -> None:
        """Generate decoder lookup tables and decode functions."""
        distributed_factories = self._distributed_decoder_factories()
        distributed_factory_names = set(distributed_factories.values())
        class_def = [
            cgen.Struct(
                'Decoder',
                [
                    cgen.Line('public:'),
                    cgen.Statement(
                        f'static constexpr std::size_t kMaxInstructionWords = {self._max_instruction_word_count()}'
                    ),
                    cgen.FunctionDeclaration(
                        cgen.Value('static DecodeResult', 'decode'),
                        [
                            cgen.Value('const MachineInst *', 'opcode'),
                            cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                        ],
                    ),
                ],
            )
        ]
        class_impl = []
        class_members = [
            cgen.Line('public:'),
            cgen.FunctionDeclaration(
                cgen.Value('static DecodeResult', 'decode'),
                [
                    cgen.Value('const MachineInst *', 'opcode'),
                    cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                ],
            ),
            cgen.Line('private:'),
            cgen.Statement(
                'using DecodeFunc = DecodeResult(*)(const MachineInst *, '
                'const DecodeErrorEmitter &)'
            ),
            cgen.FunctionDeclaration(
                cgen.Value('static DecodeResult', 'decodeInvalid'),
                [
                    cgen.Value('const MachineInst *', 'opcode'),
                    cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                ],
            ),
        ]
        decode_body = []
        if self._supports_cdna5_scaled_wmma_vop3px2():
            class_impl.append(
                cgen.Line(self._emit_cdna5_scaled_wmma_vop3px2_decoder_helpers())
            )
        if self._supports_cdna_mfma_f8f6f4_vop3px2():
            factory_cases = ''.join(
                f'    if (op2 == {spec.opcode})\n'
                f'      return std::make_unique<{spec.class_name}>(opcode);\n'
                for spec in self._cdna_mfma_f8f6f4_vop3px2_specs()
            )
            class_impl.append(
                cgen.Line(self._emit_cdna_mfma_f8f6f4_vop3px2_decoder_helpers())
            )
            decode_body.append(
                cgen.Line(
                    '  if (isMfmaScaleF8f6f4Vop3px2(opcode)) {\n'
                    '    if (!isValidMfmaScaleF8f6f4(opcode))\n'
                    '      return decodeInvalid(opcode, emit_error);\n'
                    '    auto op2 = (opcode[2] >> 16) & 0x7Fu;\n'
                    f'{factory_cases}'
                    '    return decodeInvalid(opcode, emit_error);\n'
                    '  }\n'
                )
            )
        decode_body.extend(
            [
                cgen.Statement(
                    'Sop1MachineInst op = std::bit_cast<decltype(op)>(*opcode)'
                ),
                cgen.Statement(
                    'return primary_decode_table[op.encoding](opcode, emit_error)'
                ),
            ]
        )
        decode_table_funcs = [
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value('DecodeResult', 'DecoderImpl::decode'),
                    [
                        cgen.Value('const MachineInst *', 'opcode'),
                        cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                    ],
                ),
                cgen.Block(decode_body),
            ),
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value(
                        'DecodeResult',
                        'DecoderImpl::decodeInvalid',
                    ),
                    [
                        cgen.Value('const MachineInst *', 'opcode'),
                        cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                    ],
                ),
                cgen.Block(
                    [
                        cgen.Statement(
                            'return emit_error.emit() << "Invalid instruction opcode: " '
                            '<< std::format("{:X}", *opcode)'
                        ),
                    ]
                ),
            ),
        ]
        decode_tables = [
            cgen.Statement(
                f'static const std::array<DecodeFunc, {pow(2, self.isa_spec.profile.max_enc_bits)}> primary_decode_table'
            )
        ]
        decode_table_entries = []
        sub_decode_table_entries = []
        decode_funcs_found = set()
        _custom_decode_bodies: dict[str, object] = {}
        _custom_primary_decode_funcs: dict[int, str] = {}
        _custom_primary_decode_bodies: dict[str, object] = {}

        def _reserve_primary_prefix(prefix: int, prefix_bits: int, fn: str) -> None:
            table_bits = self.isa_spec.profile.max_enc_bits
            if prefix_bits > table_bits:
                raise ValueError(
                    f'Cannot fit {prefix_bits}-bit decode prefix in '
                    f'{table_bits}-bit primary table'
                )
            first = prefix << (table_bits - prefix_bits)
            count = 1 << (table_bits - prefix_bits)
            for index in range(first, first + count):
                if self.isa_spec.primary_decode_table[index] is not None:
                    raise ValueError(
                        f'Custom decoder {fn} conflicts with primary table index {index}'
                    )
                previous = _custom_primary_decode_funcs.setdefault(index, fn)
                if previous != fn:
                    raise ValueError(
                        f'Custom decoders {previous} and {fn} conflict at primary '
                        f'table index {index}'
                    )

        if self._supports_generated_vopd():
            _vopd_fn = 'decodeVopd'
            for _prefix in self.isa_spec.profile.vopd_encoding_prefixes:
                _reserve_primary_prefix(_prefix.prefix, _prefix.prefix_bits, _vopd_fn)
            _custom_primary_decode_bodies[_vopd_fn] = cgen.Block(
                [
                    cgen.Line(
                        '  Result validation = Vopd::validate_encoding(opcode, emit_error);\n'
                        '  if (validation.failed()) [[unlikely]]\n'
                        '    return Result::failure();\n'
                    ),
                    cgen.Statement('return std::make_unique<Vopd>(opcode)'),
                ]
            )

        if self._supports_cdna5_scaled_wmma_vop3px2():
            for _dte in self.isa_spec.primary_decode_table:
                if (
                    _dte is not None
                    and not _dte.is_primary
                    and _dte.sub_decode_funcs is not None
                    and _dte.sub_decode_table
                    and 'vop3p' in _dte.sub_decode_table
                ):
                    _scaled_wmma_fn = 'decodeVWmmaScaleF32Vop3px2'
                    for _opcode in (0x35, 0x3A):
                        if _dte.sub_decode_funcs[_opcode] not in (
                            'decodeInvalid',
                            _scaled_wmma_fn,
                        ):
                            raise ValueError(
                                f'Scaled WMMA decoder conflicts with VOP3P opcode {_opcode}'
                            )
                        _dte.sub_decode_funcs[_opcode] = _scaled_wmma_fn
                    _custom_decode_bodies[_scaled_wmma_fn] = cgen.Block(
                        [
                            cgen.Line(
                                '  if (!isVop3pOp(opcode[2], 0x33) && '
                                '!isVop3pOp(opcode[2], 0x88))\n'
                                '    return decodeInvalid(opcode, emit_error);\n'
                                '  if (!isGfx1250WmmaScalePairValid(opcode))\n'
                                '    return decodeInvalid(opcode, emit_error);\n'
                            ),
                            cgen.Statement(
                                'return std::make_unique<VWmmaScaleF32Vop3px2>(opcode)'
                            ),
                        ]
                    )
                    break
            else:
                raise ValueError('gfx1250 scaled WMMA requires a VOP3P decode table')

        _vop3px2_specs = self._cdna_mfma_f8f6f4_vop3px2_specs()
        if _vop3px2_specs:
            for _dte in self.isa_spec.primary_decode_table:
                if (
                    _dte is not None
                    and not _dte.is_primary
                    and _dte.sub_decode_funcs is not None
                    and _dte.sub_decode_table
                    and 'vop3p' in _dte.sub_decode_table
                ):
                    _pfx = 'decodeVop3pX2Prefix'
                    _prefix_opcode = _vop3px2_specs[0].prefix_opcode
                    if _dte.sub_decode_funcs[_prefix_opcode] not in (
                        'decodeInvalid',
                        _pfx,
                    ):
                        raise ValueError(
                            f'CDNA scaled MFMA prefix conflicts with VOP3P opcode '
                            f'{_prefix_opcode}'
                        )
                    _dte.sub_decode_funcs[_prefix_opcode] = _pfx
                    _suffix_fn = 'decodeCdna4MfmaF8f6f4Suffix'
                    dense_factory_cases = ''.join(
                        f'  if (op.op == {spec.opcode})\n'
                        f'    return std::make_unique<{spec.dense_class_name}>(opcode);\n'
                        for spec in _vop3px2_specs
                    )
                    for spec in _vop3px2_specs:
                        if _dte.sub_decode_funcs[spec.opcode] not in (
                            'decodeInvalid',
                            f'decode{spec.dense_class_name}',
                            _suffix_fn,
                        ):
                            raise ValueError(
                                f'CDNA scaled MFMA suffix conflicts with VOP3P opcode '
                                f'{spec.opcode}'
                            )
                        _dte.sub_decode_funcs[spec.opcode] = _suffix_fn
                    _custom_decode_bodies[_suffix_fn] = cgen.Block(
                        [
                            cgen.Statement(
                                "auto op = *reinterpret_cast<const Vop3pMfma::OpEncoding *>(opcode)"
                            ),
                            cgen.Line(
                                '  if (op.abid != 0u || !isLegalMfmaF8f6f4Format(op.cbsz) ||\n'
                                '      !isLegalMfmaF8f6f4Format(op.blgp))\n'
                                '    return decodeInvalid(opcode, emit_error);\n'
                            ),
                            cgen.Line(dense_factory_cases),
                            cgen.Statement('return decodeInvalid(opcode, emit_error)'),
                        ]
                    )
                    _custom_decode_bodies[_pfx] = cgen.Block(
                        [cgen.Statement('return decodeInvalid(opcode, emit_error)')]
                    )
                    break
            else:
                raise ValueError('CDNA scaled MFMA requires a VOP3P decode table')
        for _fn, _body in _custom_primary_decode_bodies.items():
            _decl = cgen.FunctionDeclaration(
                cgen.Value('static DecodeResult', _fn),
                [
                    cgen.Value('const MachineInst *', 'opcode'),
                    cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                ],
            )
            class_members.append(_decl)
            decode_table_funcs.append(
                cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value('DecodeResult', f'DecoderImpl::{_fn}'),
                        [
                            cgen.Value('const MachineInst *', 'opcode'),
                            cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                        ],
                    ),
                    _body,
                )
            )

        for _primary_index, dte in enumerate(self.isa_spec.primary_decode_table):
            if _primary_index in _custom_primary_decode_funcs:
                decode_table_entries.append(
                    f'&DecoderImpl::{_custom_primary_decode_funcs[_primary_index]},'
                )
                continue
            if dte is not None:
                primary_factory_is_distributed = (
                    dte.is_primary and dte.decode_func in distributed_factory_names
                )
                if primary_factory_is_distributed:
                    decode_table_entries.append(f'&detail::{dte.decode_func},')
                else:
                    decode_table_entries.append(f'&DecoderImpl::{dte.decode_func},')
                if dte.decode_func not in decode_funcs_found:
                    decode_funcs_found.add(dte.decode_func)
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value(
                            'DecodeResult',
                            f'DecoderImpl::{dte.decode_func}',
                        ),
                        [
                            cgen.Value('const MachineInst *', 'opcode'),
                            cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                        ],
                    )
                    sub_decode_func_decls = []
                    if primary_factory_is_distributed:
                        pass
                    elif dte.is_primary:
                        decode_table_funcs.append(
                            cgen.FunctionBody(
                                func_decl,
                                cgen.Block(
                                    [
                                        cgen.Statement(
                                            f'return std::make_unique<{dte.inst_name}>(opcode)'
                                        )
                                    ]
                                ),
                            )
                        )
                    else:
                        decode_table_funcs.append(
                            cgen.FunctionBody(
                                func_decl,
                                cgen.Block(
                                    [
                                        cgen.Value(
                                            f'{dte.enc.fmt_enc_name}::OpEncoding',
                                            'op = *reinterpret_cast<const decltype(op) *>(opcode)',
                                        ),
                                        cgen.Statement(
                                            f'return {dte.sub_decode_table}[op.op](opcode, emit_error)'
                                        ),
                                    ]
                                ),
                            )
                        )
                        decode_tables.append(
                            cgen.Statement(
                                f'static const std::array<DecodeFunc, {len(dte.sub_decode_funcs)}> {dte.sub_decode_table}'
                            )
                        )
                        sub_decode_table_entries.append(
                            cgen.Line(
                                f'const std::array<DecoderImpl::DecodeFunc, {len(dte.sub_decode_funcs)}> DecoderImpl::{dte.sub_decode_table} = {{'
                            )
                        )
                        sub_decode_table_entry_str = []
                        sub_decode_funcs_found = set()
                        for fn in dte.sub_decode_funcs:
                            sub_factory_is_distributed = (
                                fn in distributed_factory_names
                                and fn not in _custom_decode_bodies
                            )
                            if (
                                fn != 'decodeInvalid'
                                and fn not in sub_decode_funcs_found
                            ):
                                sub_decode_funcs_found.add(fn)
                                class_name = fn.removeprefix('decode')
                                if not sub_factory_is_distributed:
                                    sub_decode_func_decls.append(
                                        cgen.FunctionDeclaration(
                                            cgen.Value(
                                                'static DecodeResult',
                                                fn,
                                            ),
                                            [
                                                cgen.Value(
                                                    'const MachineInst *',
                                                    'opcode',
                                                ),
                                                cgen.Value(
                                                    'const DecodeErrorEmitter &',
                                                    'emit_error',
                                                ),
                                            ],
                                        )
                                    )
                                    _fn_body = (
                                        _custom_decode_bodies[fn]
                                        if fn in _custom_decode_bodies
                                        else cgen.Block(
                                            [
                                                cgen.Statement(
                                                    f'return std::make_unique<{class_name}>(opcode)'
                                                )
                                            ]
                                        )
                                    )
                                    decode_table_funcs.append(
                                        cgen.FunctionBody(
                                            cgen.FunctionDeclaration(
                                                cgen.Value(
                                                    'DecodeResult',
                                                    f'DecoderImpl::{fn}',
                                                ),
                                                [
                                                    cgen.Value(
                                                        'const MachineInst *',
                                                        'opcode',
                                                    ),
                                                    cgen.Value(
                                                        'const DecodeErrorEmitter &',
                                                        'emit_error',
                                                    ),
                                                ],
                                            ),
                                            _fn_body,
                                        ),
                                    )
                            if sub_factory_is_distributed:
                                sub_decode_table_entry_str.append(f'&detail::{fn},')
                            else:
                                sub_decode_table_entry_str.append(
                                    f'&DecoderImpl::{fn},'
                                )
                        sub_decode_table_entries.append(
                            cgen.Line(''.join(sub_decode_table_entry_str))
                        )
                        sub_decode_table_entries.append(cgen.Line('};'))
                    if not primary_factory_is_distributed:
                        class_members.append(
                            cgen.FunctionDeclaration(
                                cgen.Value(
                                    'static DecodeResult',
                                    f'{dte.decode_func}',
                                ),
                                [
                                    cgen.Value('const MachineInst *', 'opcode'),
                                    cgen.Value(
                                        'const DecodeErrorEmitter &', 'emit_error'
                                    ),
                                ],
                            )
                        )
                    class_members.extend(sub_decode_func_decls)
            else:
                decode_table_entries.append('&DecoderImpl::decodeInvalid,')
        decode_table_entries = ''.join(decode_table_entries)
        class_members.extend(decode_tables)
        detail_decls = '\n'.join(
            f'DecodeResult {factory_name}'
            '(const MachineInst *opcode, const DecodeErrorEmitter &emit_error);'
            for factory_name in sorted(distributed_factory_names)
        )
        class_impl[0:0] = [
            cgen.Line(
                'namespace detail {\n' f'{detail_decls}\n' '} // namespace detail'
            ),
            cgen.Struct('DecoderImpl', class_members),
        ]
        decode_table_funcs.insert(
            0,
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value('DecodeResult', 'Decoder::decode'),
                    [
                        cgen.Value('const MachineInst *', 'opcode'),
                        cgen.Value('const DecodeErrorEmitter &', 'emit_error'),
                    ],
                ),
                cgen.Block(
                    [cgen.Statement('return DecoderImpl::decode(opcode, emit_error)')]
                ),
            ),
        )
        class_impl.extend(decode_table_funcs)
        class_impl.append(
            cgen.Line(
                f'const std::array<DecoderImpl::DecodeFunc, {pow(2, self.isa_spec.profile.max_enc_bits)}> DecoderImpl::primary_decode_table = {{'
            )
        )
        class_impl.append(cgen.Line(decode_table_entries))
        class_impl.append(cgen.Line('};'))
        class_impl.extend(sub_decode_table_entries)
        class_def_file = CppFile(
            'decoder',
            self.out_path,
            True,
            [
                (
                    self.config.generated_include(
                        self.generated_dir_name, 'machine_insts.h'
                    ),
                    False,
                ),
                ('rocjitsu/isa/decode_result.h', False),
                ('array', True),
                ('cstddef', True),
                ('memory', True),
            ],
            ['Instruction'],
            class_def,
            self.cpp_namespace,
            True,
            generated_dir_name=self.generated_dir_name,
        )
        decoder_impl_includes = [
            (
                self.config.generated_include(self.generated_dir_name, 'decoder.h'),
                False,
            ),
            (
                self.config.generated_include(
                    self.generated_dir_name,
                    'vop3p.h',
                ),
                False,
            ),
        ]
        if self._supports_cdna_mfma_f8f6f4_vop3px2():
            decoder_impl_includes.append(
                (
                    self.config.generated_include(
                        self.generated_dir_name,
                        'opcodes.h',
                    ),
                    False,
                )
            )
        if self._supports_generated_vopd():
            decoder_impl_includes.append(
                (
                    self.config.generated_include(
                        self.generated_dir_name,
                        'vopd.h',
                    ),
                    False,
                )
            )
        decoder_impl_includes.extend([('array', True), ('bit', True), ('format', True)])
        class_impl_file = CppFile(
            'decoder',
            self.out_path,
            False,
            decoder_impl_includes,
            [],
            class_impl,
            self.cpp_namespace,
            generated_dir_name=self.generated_dir_name,
        )
        class_def_file.gen_code()
        class_impl_file.gen_code()

    def _sample_test_encoding_words(
        self, enc: InstEncoding, inst: Instruction
    ) -> tuple[int, int] | None:
        op_field = next((f for f in enc.ucode_fields if f.name == 'op'), None)
        has_encoding_field = any(f.name == 'encoding' for f in enc.ucode_fields)
        ptrs = enc.primary_dt_ptrs
        if not op_field or not has_encoding_field or not ptrs:
            return None
        if inst.opcode >= len(ptrs) or ptrs[inst.opcode] == -1:
            return None

        enc_val = ptrs[inst.opcode]
        word = (enc_val << (32 - self.isa_spec.profile.max_enc_bits)) | (
            inst.opcode << op_field.bit_offset
        )
        for operand in inst.explicit_operands:
            if operand.operand_type != 'OPR_SSRC_BARRIER_ID':
                continue
            operand_field = next(
                (field for field in enc.ucode_fields if field.name == operand.name),
                None,
            )
            if operand_field is not None:
                # The zero selector is reserved for barrier IDs. Encode the
                # inline integer zero instead so the execution smoke fixture
                # remains a valid instruction.
                word |= 128 << operand_field.bit_offset
        return word & 0xFFFFFFFF, (word >> 32) & 0xFFFFFFFF

    def gen_test_encodings(self) -> None:
        """Generate a C++ header with one sample encoding word per instruction.

        Produces ``test_encodings.h`` containing a constexpr array of
        ``{mnemonic, {word0, word1}}`` entries.  The test harness decodes
        each entry and calls ``execute()`` to verify no ``UnimplementedInst``
        is thrown.
        """
        entries: list[str] = []
        profile = self.isa_spec.profile
        # Build child alt mapping (same as gen_insts).
        test_child_encs: dict[str, list[InstEncoding]] = {}
        for enc in self.isa_spec.inst_encodings:
            if enc.insts and profile.is_alt_encoding(enc.enc_name):
                parent_name = profile.derive_parent_enc_name(enc.enc_name)
                test_child_encs.setdefault(parent_name, []).append(enc)

        for enc in self.isa_spec.inst_encodings:
            # Collect instructions from this encoding plus child alts.
            all_test_insts = list(enc.insts)
            for child in test_child_encs.get(enc.enc_name, []):
                all_test_insts.extend(child.insts)
            if not all_test_insts:
                continue
            # Skip alt encodings — their instructions are included via parent.
            if profile.is_alt_encoding(enc.enc_name):
                continue
            for inst in all_test_insts:
                sample = self._sample_test_encoding_words(enc, inst)
                if sample is None:
                    continue
                w0, w1 = sample
                entries.append(
                    f'  {{"{inst.mnemonic}", {{0x{w0:08X}U, 0x{w1:08X}U}}}},'
                )

        arch = self.cpp_namespace
        ns = arch
        guard = f'ROCJITSU_ISA_AMDGPU_{arch.upper()}_TEST_ENCODINGS_H_'
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f'#ifndef {guard}',
            f'#define {guard}',
            '',
            '#include <array>',
            '#include <cstdint>',
            '#include <string_view>',
            '',
            f'namespace rocjitsu::{ns}::test_data {{',
            '',
            'struct TestEncoding {',
            '  std::string_view mnemonic;',
            '  std::array<uint32_t, 2> words;',
            '};',
            '',
            f'inline constexpr TestEncoding ENCODINGS[] = {{',
        ]
        lines.extend(entries)
        lines.append('};')
        lines.append('')
        lines.append(f'inline constexpr size_t NUM_ENCODINGS = {len(entries)};')
        lines.append('')
        lines.append(f'}} // namespace rocjitsu::{ns}::test_data')
        lines.append('')
        lines.append(f'#endif // {guard}')
        lines.append('')

        out_path = os.path.join(
            self.out_path, self.generated_dir_name, 'test_encodings.h'
        )
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))

    def gen_isa_types(self) -> None:
        """Generate an ISA struct wrapping type definitions."""
        isa_typedefs = [
            cgen.Statement('using Decoder = Decoder'),
            cgen.Statement('using OperandType = OperandType'),
        ]
        isa_struct = [cgen.Struct('Isa', isa_typedefs)]
        isa_struct_file = CppFile(
            'isa',
            self.out_path,
            True,
            [
                (
                    self.config.generated_include(self.generated_dir_name, 'decoder.h'),
                    False,
                ),
                (
                    self.config.generated_include(
                        self.generated_dir_name, 'operand_types.h'
                    ),
                    False,
                ),
            ],
            [],
            isa_struct,
            self.cpp_namespace,
            generated_dir_name=self.generated_dir_name,
        )
        isa_struct_file.gen_code()
