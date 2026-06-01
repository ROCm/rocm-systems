# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""C++ code generator for AMD GPU ISA decoders and instruction classes.

Generates the following C++ files from a parsed ``IsaSpec``:

* ``machine_insts.h`` - bitfield structs for each encoding's microcode.
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
import re

from dataclasses import dataclass, field as _field

from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec, OperandNamePattern
from amdisa.semantics import InstructionSemantics, SemanticsSpec

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

    def __init__(
        self,
        isa_spec: IsaSpec,
        out_path: str,
        semantics: SemanticsSpec | None = None,
        config: CodegenConfig | None = None,
        shared_plan: "SharedInstructionPlan | None" = None,
    ) -> None:
        self.isa_spec = isa_spec
        self.out_path = out_path
        self.semantics = semantics
        self.config = config if config is not None else CodegenConfig()
        self.shared_plan = shared_plan
        # Keyed by (mnemonic, enc_name) to avoid cross-encoding conflicts.
        self._shared_execute_bodies: dict[tuple[str, str], tuple] = {}
        self._emitter = _SemanticEmitter(isa_spec, semantics)

    def gen_all(self) -> None:
        """Generate all C++ objects.

        Note: ``gen_isa_types()`` is intentionally excluded because its
        output file (``isa.h``) contains hand-maintained content (ISA
        traits, status register bitfields, etc.) that the generator does
        not produce. Use ``--gen-isa`` only for bootstrapping a new arch.
        """
        self.gen_machine_inst_encodings()
        self.gen_encodings()
        self.gen_operand_types()
        self.gen_operand()
        self.gen_insts()
        self.gen_decoder()
        self.gen_test_encodings()

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

        enc_structs = [cgen.Statement("using MachineInst = uint32_t")]
        for inst_enc in self.isa_spec.inst_encodings:
            struct_name = f"{inst_enc.fmt_enc_name}MachineInst"
            fields = [(x.name, x.bit_cnt) for x in inst_enc.ucode_fields]

            if struct_name in baseline:
                inc_path, expected_fields = baseline[struct_name]
                if fields == expected_fields:
                    shared_includes.add(inc_path)
                    enc_structs.append(
                        cgen.Statement(f"using {struct_name} = amdgpu::{struct_name}")
                    )
                    continue

            s = cgen.Struct(
                struct_name,
                [
                    cgen.Value("uint32_t", f"{x.name} : {x.bit_cnt}")
                    for x in inst_enc.ucode_fields
                ],
            )
            enc_structs.append(s)

        includes: list[tuple[str, bool]] = [("cstdint", True)]
        for inc in sorted(shared_includes):
            includes.append((inc, False))

        cpp_file = CppFile(
            "machine_insts",
            self.out_path,
            True,
            includes,
            [],
            enc_structs,
            self.isa_spec.arch_name,
        )
        cpp_file.gen_code()

    def gen_encodings(self) -> None:
        """Generate encoding classes wrapping raw encoding types."""
        enc_classes = []
        class_func_impls = []
        cond_emitted: set[str] = set()
        for inst_enc in self.isa_spec.inst_encodings:
            if not inst_enc.insts:
                continue
            class_members = []
            public_members = [
                cgen.Line("public:"),
                cgen.FunctionDeclaration(
                    cgen.Value("", f"{inst_enc.fmt_enc_name}"),
                    [
                        cgen.Value("std::string_view", "mnemonic"),
                        cgen.Value(
                            f"const {inst_enc.fmt_enc_name}MachineInst",
                            "*inst",
                        ),
                        cgen.Value("ExecuteFn", "exec_fn"),
                    ],
                ),
            ]
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
            default_cond = dict(inst_enc.enc_conds).get("default_encoding", "true")
            has_real_default_check = inst_enc.bit_cnt < 64 and default_cond != "false"

            if has_real_default_check and inst_enc.has_implied_literal_ops:
                size_condition = "!default_encoding() || hasImpliedLiteral()"
            elif has_real_default_check:
                size_condition = "!default_encoding()"
            elif inst_enc.has_implied_literal_ops:
                size_condition = "hasImpliedLiteral()"
            else:
                size_condition = None

            profile = self.isa_spec.profile
            rule = profile.mnemonic_rule(inst_enc.enc_name)
            if rule.use_flat_mnemonic:
                mnemonic_expr = "flat_mnemonic(mnemonic, inst->seg)"
            else:
                # Suffix is pre-baked into the literal by the instruction
                # constructor, so the encoding base just passes through.
                mnemonic_expr = "mnemonic"

            modifier_lines = ""
            enc_field_names = {f.name for f in inst_enc.ucode_fields}
            for mod in profile.encoding_modifiers(inst_enc.enc_name):
                if not mod.preamble and mod.field not in enc_field_names:
                    continue
                field_ref = mod.field if mod.preamble else f"inst->{mod.field}"
                if mod.preamble:
                    modifier_lines += mod.preamble
                if mod.is_offset:
                    cond = mod.condition if mod.condition else field_ref
                    modifier_lines += (
                        f'if ({cond}) modifiers_ += " offset:"'
                        f" + std::to_string({field_ref});"
                    )
                else:
                    modifier_lines += f'if ({field_ref}) modifiers_ += "{mod.display}";'

            has_op = any(f.name == "op" for f in inst_enc.ucode_fields)
            size_line = (
                " size_ = sizeof(OpEncoding);\n"
                "  raw_encoding_ = reinterpret_cast<const uint32_t *>(&inst_);\n"
                "  encoding_id_ = raw_encoding_[0] >> 23;"
            )
            if has_op:
                size_line += "\n  opcode_ = inst_.op;"
            if size_condition is not None:
                size_line += f" if ({size_condition})" f" size_ += sizeof(MachineInst);"
            if inst_enc.has_implied_literal_ops:
                size_line += (
                    " if (hasImpliedLiteral())"
                    " literal_ = reinterpret_cast<const uint32_t *>(inst)[1];"
                )
            if rule.use_flat_mnemonic:
                # FLAT mnemonics are dynamically constructed ("scratch_*",
                # "global_*"). Store the owned string in a member so the
                # string_view in Instruction doesn't dangle.
                class_ctor_impl = (
                    f"{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}"
                    f"(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst, ExecuteFn exec_fn) "
                    f': IsaInstruction<Isa>("", exec_fn), inst_(*inst), '
                    f"owned_mnemonic_({mnemonic_expr}) "
                    f"{{ mnemonic_ = owned_mnemonic_;{size_line}}}"
                )
            else:
                class_ctor_impl = (
                    f"{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}"
                    f"(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst, ExecuteFn exec_fn) "
                    f": IsaInstruction<Isa>({mnemonic_expr}, exec_fn), inst_(*inst) "
                    f"{{{size_line}}}"
                )
            class_func_impls.append(cgen.Line(class_ctor_impl))

            # Generate build_modifiers() override for encoding bases
            # that have modifier flags (memory instructions). This is
            # called lazily by disassemble() instead of eagerly in the
            # constructor, avoiding string allocation on the hot path.
            if modifier_lines:
                public_members.append(
                    cgen.Line("void build_modifiers(std::string &out) const override;"),
                )
                # The modifier_lines were written for the constructor where
                # they appended to modifiers_ and accessed inst->field.
                # Rewrite to append to 'out' and access via local pointer.
                mod_impl = modifier_lines.replace("modifiers_", "out")
                class_func_impls.append(
                    cgen.Line(
                        f"void {inst_enc.fmt_enc_name}::build_modifiers"
                        f"(std::string &out) const "
                        f"{{ auto *inst = &inst_;(void)inst;"
                        f"{mod_impl}}}"
                    )
                )
            fmt_enc_name = inst_enc.fmt_enc_name
            implicit_uses_impl = self._encoding_implicit_uses_impl(
                inst_enc, enc_field_names
            )
            if implicit_uses_impl:
                public_members.append(
                    cgen.Line("void implicit_uses(RegisterSet &uses) const override;")
                )
                class_func_impls.append(
                    cgen.Line(
                        f"void {fmt_enc_name}::implicit_uses"
                        f"(RegisterSet &uses) const "
                        f"{{ {implicit_uses_impl} }}"
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
                    if enc_cond[0] == "default_encoding" and not has_real_default_check:
                        continue
                    seen_conds.add(enc_cond[0])
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value("bool", f"{enc_cond[0]}"), []
                    )
                    func_body = cgen.FunctionBody(
                        cgen.FunctionDeclaration(
                            cgen.Value("bool", f"{fmt_enc_name}::{enc_cond[0]}"),
                            [],
                        ),
                        cgen.Block([cgen.Statement(f"return {enc_cond[1]}")]),
                    )
                    public_members.append(func_decl)
                    class_func_impls.append(func_body)

            if inst_enc.has_implied_literal_ops:
                func_decl = cgen.FunctionDeclaration(
                    cgen.Value("bool", "hasImpliedLiteral"), []
                )
                implied_literal_cond = " || ".join(
                    f"inst_.op == {op}" for op in inst_enc.implied_literal_ops
                )
                func_body = cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value(
                            "bool",
                            f"{fmt_enc_name}::hasImpliedLiteral",
                        ),
                        [],
                    ),
                    cgen.Block([cgen.Statement(f"return {implied_literal_cond}")]),
                )
                public_members.append(func_decl)
                class_func_impls.append(func_body)

            class_members.extend(public_members)
            class_members.append(
                cgen.Statement(f"using OpEncoding = {inst_enc.fmt_enc_name}MachineInst")
            )
            class_members.append(cgen.Statement("const OpEncoding inst_"))
            if inst_enc.has_implied_literal_ops:
                class_members.append(cgen.Statement("uint32_t literal_ = 0"))
            # FLAT encoding bases need an owned string for the dynamic mnemonic.
            if rule.use_flat_mnemonic:
                class_members.append(cgen.Statement("std::string owned_mnemonic_"))
            # VOP1/VOP2 encoding bases store DPP control fields.
            # apply_dpp() is a free function in dpp_sdwa_ops.h.
            if inst_enc.enc_name.upper() in ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC"):
                class_members.append(cgen.Statement("uint32_t dpp_ctrl_ = 0"))
                class_members.append(cgen.Statement("uint32_t dpp_row_mask_ = 0xF"))
                class_members.append(cgen.Statement("uint32_t dpp_bank_mask_ = 0xF"))
                class_members.append(cgen.Statement("uint32_t dpp_bound_ctrl_ = 0"))
                class_members.append(
                    cgen.Statement("std::unique_ptr<DppOperand> dpp_src0_")
                )
                class_members.append(
                    cgen.Statement("std::unique_ptr<DppOperand> dpp_src1_")
                )
                # SDWA fields (CDNA and RDNA1/2 have hardware SDWA encoding; fields
                # are present on all ISAs for uniform codegen even if unused).
                class_members.append(
                    cgen.Statement("uint32_t sdwa_src0_sel_ = amdgpu::sdwa::DWORD")
                )
                class_members.append(cgen.Statement("bool sdwa_src0_sext_ = false"))
                class_members.append(cgen.Statement("bool sdwa_src0_neg_ = false"))
                class_members.append(cgen.Statement("bool sdwa_src0_abs_ = false"))
                class_members.append(
                    cgen.Statement("uint32_t sdwa_src1_sel_ = amdgpu::sdwa::DWORD")
                )
                class_members.append(cgen.Statement("bool sdwa_src1_sext_ = false"))
                class_members.append(cgen.Statement("bool sdwa_src1_neg_ = false"))
                class_members.append(cgen.Statement("bool sdwa_src1_abs_ = false"))
                if inst_enc.enc_name.upper() != "ENC_VOPC":
                    class_members.append(
                        cgen.Statement("uint32_t sdwa_dst_sel_ = amdgpu::sdwa::DWORD")
                    )
                    class_members.append(
                        cgen.Statement("uint32_t sdwa_dst_unused_ = 0")
                    )
                    class_members.append(cgen.Statement("bool sdwa_clamp_ = false"))
                else:
                    class_members.append(cgen.Statement("uint32_t sdwa_sdst_ = 106"))
                    class_members.append(cgen.Statement("bool sdwa_sd_ = false"))
            s = cgen.Struct(
                f"{inst_enc.fmt_enc_name} : public IsaInstruction<Isa>",
                [x for x in class_members],
            )
            enc_classes.append(s)

        class_def_file = CppFile(
            "encodings",
            self.out_path,
            True,
            [
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/isa.h",
                    False,
                ),
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/machine_insts.h",
                    False,
                ),
                ("rocjitsu/isa/instruction.h", False),
                ("rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h", False),
                ("string", True),
                ("string_view", True),
            ],
            [],
            enc_classes,
            self.isa_spec.arch_name,
            True,
        )
        needs_flat_mnemonic = any(
            self.isa_spec.profile.mnemonic_rule(enc.enc_name).use_flat_mnemonic
            for enc in self.isa_spec.inst_encodings
        )
        if needs_flat_mnemonic:
            flat_mnemonic_helper = cgen.Line(
                "namespace {\n"
                "std::string flat_mnemonic(std::string_view mnemonic, int seg) {\n"
                "  // seg: 0=FLAT, 1=SCRATCH, 2=GLOBAL\n"
                '  if (seg == 1 && mnemonic.substr(0, 5) == "flat_")\n'
                '    return std::string("scratch_").append(mnemonic.substr(5));\n'
                '  if (seg == 2 && mnemonic.substr(0, 5) == "flat_")\n'
                '    return std::string("global_").append(mnemonic.substr(5));\n'
                "  return std::string(mnemonic);\n"
                "}\n"
                "} // namespace"
            )
            class_func_impls.insert(0, flat_mnemonic_helper)

        _enc_cpp_includes = [
            (
                f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/encodings.h",
                False,
            ),
            ("string", True),
        ]
        class_impl_file = CppFile(
            "encodings",
            self.out_path,
            False,
            _enc_cpp_includes,
            [],
            class_func_impls,
            self.isa_spec.arch_name,
        )
        class_def_file.gen_code()
        class_impl_file.gen_code()

    def _encoding_implicit_uses_impl(
        self, inst_enc: InstEncoding, enc_field_names: set[str]
    ) -> str:
        """Return C++ body for hidden register uses on an encoding base."""
        if (
            inst_enc.enc_name.upper() == "ENC_FLAT"
            and {"seg", "saddr"} <= enc_field_names
        ):
            return (
                "if (inst_.saddr == 0x7F) return;"
                "if (inst_.seg == 1) {"
                "uses.expand(RegisterRef{RegClass::SGPR, "
                "static_cast<uint16_t>(inst_.saddr), 1});"
                "} else if (inst_.seg == 2) {"
                "uses.expand(RegisterRef{RegClass::SGPR, "
                "static_cast<uint16_t>(inst_.saddr), 2});"
                "}"
            )
        return ""

    def _enc_field_at_bit(self, enc_name: str, bit_offset: int) -> str | None:
        """Return the field name at a given bit offset in an encoding, or None."""
        for enc in self.isa_spec.inst_encodings:
            if enc.enc_name.upper() == enc_name.upper():
                for f in enc.ucode_fields:
                    if f.bit_offset == bit_offset:
                        return f.name
        return None

    def _op_sel_hi_2_expr(self, enc_name: str) -> str:
        """Return a C++ expression for the op_sel_hi bit for the third source.

        Some ISA specs (e.g. CDNA4) removed the explicit op_sel_hi_2
        field from Vop3pMachineInst and declared the bit as reserved
        (pad_14). The hardware behavior is unchanged - access whatever
        field sits at bit 14.
        """
        field = self._enc_field_at_bit(enc_name, 14)
        if field:
            return f"inst_.{field}"
        return "inst_.op_sel_hi_2"

    # Semantic operations/classes where the destination register is also read.
    # The CDNA4 XML marks these destinations as output-only, but the execute
    # body reads the old value (accumulate, swap, partial write, bitfield set).
    _READS_DST_OPS = frozenset({"fmac", "bitset0", "bitset1"})
    _READS_DST_CLASSES = frozenset(
        {
            "vector_dot",
            "vector_swap",
            "mad_mixlo_f16",
            "mad_mixhi_f16",
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
        if not self.semantics:
            return False
        sem = self.semantics.instructions.get(inst.name)
        if not sem:
            return False
        return (
            sem.operation in self._READS_DST_OPS
            or sem.semantic_class in self._READS_DST_CLASSES
        )

    def _gen_execute_body(
        self, inst: Instruction, sem: InstructionSemantics, enc_name: str = ""
    ) -> str:
        """Generate execute() body from instruction semantics."""
        dst_ops = [op.name for op in inst.operands if not op.is_input]
        src_ops = [op.name for op in inst.operands if op.is_input]
        # Some instructions mark their destination as input (read-modify-write,
        # e.g. S_BITSET0, S_CMOV, V_FMAC, V_SWAP). Recover the destination
        # from src_ops when it looks like one.
        if not dst_ops and src_ops and src_ops[0] in ("sdst", "vdst"):
            dst_ops = [src_ops[0]]
            src_ops = src_ops[1:]
        # Some ISA specs mark swap operands as output-only even though the
        # instruction reads both. Treat the second output as a source.
        if not src_ops and len(dst_ops) >= 2 and sem.semantic_class == "vector_swap":
            src_ops = dst_ops[1:]
            dst_ops = dst_ops[:1]
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
        )

        _SEMA_CLASSES = frozenset(
            {
                "scalar_mov",
                "scalar_cmov",
                "scalar_cselect",
                "scalar_cmp",
                "scalar_unary",
                "scalar_binop",
                "scalar_bitcmp",
                "scalar_saveexec",
                "scalar_bfe",
                "vector_cmp_class",
                "vector_swap",
                "vector_mov",
                "vector_binop",
                "vector_ternary",
                "vector_unary",
                "vector_cmp",
                "vector_cndmask",
                "vector_add_co",
            }
        )
        if cls in _SEMA_CLASSES:
            sema_block = derive_sema_block(sem)
            if sema_block is not None and not sema_block.is_empty:
                is_float_op = (
                    dtype in ("f16", "f32", "f64", "bf16") or cls == "vector_mov"
                )
                if is_vop3 and is_float_op:
                    from amdisa.sema_enrich import enrich_block

                    ef = {"neg"}
                    if has_abs:
                        ef.add("abs")
                    inst_fields = getattr(self, "_current_inst_fields", set())
                    if "clamp" in inst_fields:
                        ef.add("clamp")
                    if "omod" in inst_fields:
                        ef.add("omod")
                    sema_block = enrich_block(sema_block, enc_field_names=frozenset(ef))
                omap_dtype = "b64" if cls == "scalar_saveexec" else dtype
                omap_kwargs = {}
                name_lower = inst.name.lower()
                # AMD naming: V_CVT_DST_SRC — last suffix is source type
                if name_lower.endswith("_f64"):
                    omap_kwargs["src_width"] = 64
                elif "cvt_f64_" in name_lower:
                    omap_kwargs["dst_width"] = 64
                if "frexp_exp_i32_f64" in name_lower:
                    omap_kwargs["src_width"] = 64
                elif "frexp_mant_f64" in name_lower:
                    omap_kwargs["src_width"] = 64
                    omap_kwargs["dst_width"] = 64
                omap = OperandMap.from_operand_names(
                    src_ops, dst_ops, sema_block.pragma, omap_dtype, **omap_kwargs
                )
                lctx = LoweringContext(exec_model=sema_block.pragma, operand_map=omap)
                if cls == "vector_cndmask" and is_vop3 and len(src_ops) >= 3:
                    lctx.vcc_read = f"{src_ops[2]}.read_scalar64(wf)"
                if cls == "vector_add_co":
                    if is_vop3 and len(src_ops) >= 3:
                        lctx.vcc_read = f"{src_ops[2]}.read_scalar64(wf)"
                    lctx.vcc_dst = dst_ops[1] if len(dst_ops) > 1 else "__vcc__"
                return lower_sema_block(sema_block, lctx)

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
            enc_field_names=getattr(self, "_current_inst_fields", set()),
            encoding_map=self.isa_spec.encoding_map,
        )
        handler = DISPATCH.get(cls)
        if handler is not None:
            return handler(ctx)

        # Fallback: inline dispatch for classes not yet extracted.
        L = []  # output lines

        if cls == "true_nop":
            return "  (void)wf;"

        if cls == "nop":
            return "  (void)wf;\n throw util::UnimplementedInst(mnemonic());"

        if cls == "endpgm":
            # Use end() instead of halt() to drain outstanding memory ops.
            # If all wait counters are zero, end() halts immediately.
            # Otherwise, it transitions to ENDING state and the memory
            # pipeline drain handles the final halt.
            L.append("  wf.end();")
            return "\n".join(L)

        if cls == "waitcnt":
            L.append(
                f"  uint16_t imm = static_cast<uint16_t>({src_ops[0]}.encoding_value_);"
            )
            wf = self.isa_spec.profile.waitcnt_family
            if wf == "gfx11":
                # GFX11 (RDNA3/3.5) SIMM16 layout:
                #   expcnt[2:0] = bits [2:0]
                #   lgkmcnt[5:0] = bits [9:4]
                #   vmcnt[5:0] = bits [15:10]
                L.append("  uint8_t exp = imm & 0x7;")
                L.append("  uint8_t lgkm = (imm >> 4) & 0x3F;")
                L.append("  uint8_t vm = (imm >> 10) & 0x3F;")
            else:
                # GFX9 (CDNA1-4) / GFX10 (RDNA1/2) SIMM16 layout:
                #   vmcnt[3:0] = bits [3:0], vmcnt[5:4] = bits [15:14]
                #   expcnt[2:0] = bits [6:4]
                #   lgkmcnt = bits [12:8] (GFX9) or [13:8] (GFX10)
                L.append("  uint8_t vm = (imm & 0xF) | ((imm >> 10) & 0x30);")
                L.append("  uint8_t exp = (imm >> 4) & 0x7;")
                L.append("  uint8_t lgkm = (imm >> 8) & Isa::WAITCNT_LGKMCNT_MASK;")
            L.append("  wf.set_wait_target(vm, lgkm, exp);")
            return "\n".join(L)

        if cls == "wait_counter":
            # RDNA4 split-wait instructions: the immediate operand is
            # the counter threshold directly (no bit-packing).
            L.append(
                f"  uint16_t cnt = static_cast<uint16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append(f'  wf.set_wait_counter("{op}", cnt);')
            return "\n".join(L)

        if cls == "barrier":
            L.append("  wf.set_state(amdgpu::WfState::BARRIER);")
            return "\n".join(L)

        if cls == "branch":
            L.append(
                f"  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append("  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;")
            return "\n".join(L)

        if cls == "cbranch":
            cond_map = {
                "scc0": "!wf.read_scc()",
                "scc1": "wf.read_scc()",
                "vccz": "wf.vcc() == 0",
                "vccnz": "wf.vcc() != 0",
                "execz": "wf.exec() == 0",
                "execnz": "wf.exec() != 0",
            }
            L.append(f"  if ({cond_map[cond]}) {{")
            L.append(
                f"    int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append(
                "    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;"
            )
            L.append("  }")
            return "\n".join(L)

        # scalar_mov, scalar_cmov, scalar_cselect now handled by SemaAST.

        if cls == "scalar_movk":
            L.append(
                f"  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));"
            )
            return "\n".join(L)

        if cls == "scalar_cmovk":
            L.append(
                f"  if (wf.read_scc()) {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));"
            )
            return "\n".join(L)

        if cls == "scalar_addk":
            L.append(
                f"  int32_t s0 = static_cast<int32_t>({dst_ops[0]}.read_scalar(wf));"
            )
            L.append(
                f"  int32_t imm = static_cast<int16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append(
                "  int64_t wide = static_cast<int64_t>(s0) + static_cast<int64_t>(imm);"
            )
            L.append("  int32_t result = static_cast<int32_t>(wide);")
            L.append(f"  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(result));")
            L.append("  wf.write_scc(wide != static_cast<int64_t>(result));")
            return "\n".join(L)

        if cls == "scalar_mulk":
            L.append(
                f"  int32_t s0 = static_cast<int32_t>({dst_ops[0]}.read_scalar(wf));"
            )
            L.append(
                f"  int32_t imm = static_cast<int16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append(
                f"  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(s0 * imm));"
            )
            return "\n".join(L)

        if cls == "scalar_wrexec":
            L.append(f"  uint64_t src = {src_ops[0]}.read_scalar64(wf);")
            if op in ("andn1", "and_not1"):
                # EXEC = SRC & ~EXEC
                L.append("  wf.set_exec(src & ~wf.exec());")
            elif op in ("andn2", "and_not0"):
                # EXEC = EXEC & ~SRC
                L.append("  wf.set_exec(wf.exec() & ~src);")
            else:
                L.append(f"  wf.set_exec(src); // TODO: {op}")
            return "\n".join(L)

        if cls == "scalar_getpc":
            # S_GETPC_B64: returns PC of the instruction FOLLOWING the S_GETPC.
            # At execute() time, wf.pc points to the S_GETPC itself; step() will
            # add size_ afterwards. Write wf.pc + size_ so the net result after
            # post-execute advance is correct (caller sees PC of next instruction).
            L.append(f"  {dst_ops[0]}.write_scalar64(wf, wf.pc + size_);")
            return "\n".join(L)

        if cls == "scalar_setpc":
            L.append(f"  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;")
            return "\n".join(L)

        if cls == "scalar_swappc":
            # S_SWAPPC_B64: dst = PC of next inst, then jump to src.
            L.append(f"  uint64_t next_pc = wf.pc + size_;")
            L.append(f"  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;")
            L.append(f"  {dst_ops[0]}.write_scalar64(wf, next_pc);")
            return "\n".join(L)

        if cls == "scalar_call":
            # S_CALL_B64: dst = PC of next instruction (return address), then branch.
            L.append(f"  {dst_ops[0]}.write_scalar64(wf, wf.pc + size_);")
            L.append(
                f"  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);"
            )
            L.append("  wf.pc = wf.pc + static_cast<int64_t>(offset) * 4 - size_;")
            return "\n".join(L)

        if cls == "scalar_getreg":
            L.append(f"  uint16_t hwreg = {src_ops[0]}.encoding_value_;")
            L.append("  uint32_t reg_id = hwreg & 0x3Fu;")
            L.append("  uint32_t offset = (hwreg >> 6) & 0x1Fu;")
            L.append("  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;")
            L.append("  uint32_t reg_val = 0;")
            L.append("  switch (reg_id) {")
            L.append("  case 1: reg_val = wf.status_raw(); break;")
            L.append("  case 4: reg_val = static_cast<uint32_t>(wf.cu().id()); break;")
            L.append(
                "  case 5: reg_val = static_cast<uint32_t>(wf.cu().id() >> 16); break;"
            )
            L.append(
                "  case 6: reg_val = (wf.sgpr_alloc().count & 0xFFu) | ((wf.sgpr_alloc().base & 0xFFu) << 8); break;"
            )
            L.append(
                "  case 7: reg_val = (wf.vgpr_alloc().count & 0xFFu) | ((wf.vgpr_alloc().base & 0xFFu) << 8); break;"
            )
            L.append(
                '  default: util::Logger::warn("s_getreg_b32: unhandled hwreg id=", reg_id); break;'
            )
            L.append("  }")
            L.append("  if (offset + size > 32) size = 32 - offset;")
            L.append(
                "  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);"
            )
            L.append(f"  {dst_ops[0]}.write_scalar(wf, (reg_val >> offset) & mask);")
            return "\n".join(L)

        if cls == "scalar_setreg":
            L.append(f"  uint16_t hwreg = {dst_ops[0]}.encoding_value_;")
            L.append("  uint32_t reg_id = hwreg & 0x3Fu;")
            L.append("  uint32_t offset = (hwreg >> 6) & 0x1Fu;")
            L.append("  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;")
            L.append("  if (offset + size > 32) size = 32 - offset;")
            L.append(
                "  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);"
            )
            L.append(f"  uint32_t src = {src_ops[0]}.read_scalar(wf);")
            L.append("  switch (reg_id) {")
            L.append("  case 1: {")
            L.append("    uint32_t s = wf.status_raw();")
            L.append("    s = (s & ~(mask << offset)) | ((src & mask) << offset);")
            L.append("    wf.set_status_raw(s);")
            L.append("    break;")
            L.append("  }")
            L.append(
                '  default: util::Logger::warn("s_setreg_b32: unhandled hwreg id=", reg_id); break;'
            )
            L.append("  }")
            return "\n".join(L)

        if cls == "scalar_setreg_imm":
            L.append(f"  uint16_t hwreg = {dst_ops[0]}.encoding_value_;")
            L.append("  uint32_t reg_id = hwreg & 0x3Fu;")
            L.append("  uint32_t offset = (hwreg >> 6) & 0x1Fu;")
            L.append("  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;")
            L.append("  if (offset + size > 32) size = 32 - offset;")
            L.append(
                "  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);"
            )
            L.append("  uint32_t src = inst.literal_;")
            L.append("  switch (reg_id) {")
            L.append("  case 1: {")
            L.append("    uint32_t s = wf.status_raw();")
            L.append("    s = (s & ~(mask << offset)) | ((src & mask) << offset);")
            L.append("    wf.set_status_raw(s);")
            L.append("    break;")
            L.append("  }")
            L.append(
                '  default: util::Logger::warn("s_setreg_imm32_b32: unhandled hwreg id=", reg_id); break;'
            )
            L.append("  }")
            return "\n".join(L)

        if cls == "vector_readfirstlane":
            L.append("  uint64_t exec = wf.exec();")
            L.append("  uint32_t val = 0;")
            L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
            L.append("    if (exec & (1ULL << lane)) {")
            L.append(f"      val = {src_ops[0]}.read_lane(wf, lane);")
            L.append("      break;")
            L.append("    }")
            L.append("  }")
            L.append(f"  {dst_ops[0]}.write_scalar(wf, val);")
            return "\n".join(L)

        if cls == "vector_readlane":
            L.append(f"  uint32_t lane = {src_ops[1]}.read_scalar(wf);")
            L.append(
                f"  {dst_ops[0]}.write_scalar(wf, {src_ops[0]}.read_lane(wf, lane));"
            )
            return "\n".join(L)

        if cls == "vector_writelane":
            L.append(f"  uint32_t val = {src_ops[0]}.read_scalar(wf);")
            L.append(f"  uint32_t lane = {src_ops[1]}.read_scalar(wf);")
            L.append(f"  {dst_ops[0]}.write_lane(wf, lane, val);")
            return "\n".join(L)

        # vector_swap now handled by SemaAST.

        if cls == "vector_fmamk":
            # D = S0 * K + S2, K is inline constant (second src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = f"{src_ops[1]}.encoding_value_" if len(src_ops) >= 3 else "simm32_"
            s2_expr = src_ops[2] if len(src_ops) >= 3 else src_ops[1]
            L.append("  uint64_t exec = wf.exec();")
            L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
            L.append("    if (!(exec & (1ULL << lane))) continue;")
            if dtype == "f16":
                L.append(
                    f"    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));"
                )
                L.append(
                    f"    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));"
                )
                L.append(
                    f"    float s2 = util::f16_to_f32(static_cast<uint16_t>({s2_expr}.read_lane(wf, lane)));"
                )
                L.append(
                    f"    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, k, s2)));"
                )
            else:
                L.append(
                    f"    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));"
                )
                L.append(f"    float k = std::bit_cast<float>({k_expr});")
                L.append(
                    f"    float s2 = std::bit_cast<float>({s2_expr}.read_lane(wf, lane));"
                )
                L.append(
                    f"    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, k, s2)));"
                )
            L.append("  }")
            return "\n".join(L)

        if cls == "vector_fmaak":
            # D = S0 * S1 + K, K is inline constant (third src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = f"{src_ops[2]}.encoding_value_" if len(src_ops) >= 3 else "simm32_"
            L.append("  uint64_t exec = wf.exec();")
            L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
            L.append("    if (!(exec & (1ULL << lane))) continue;")
            if dtype == "f16":
                L.append(
                    f"    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));"
                )
                L.append(
                    f"    float s1 = util::f16_to_f32(static_cast<uint16_t>({src_ops[1]}.read_lane(wf, lane)));"
                )
                L.append(
                    f"    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));"
                )
                L.append(
                    f"    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, s1, k)));"
                )
            else:
                L.append(
                    f"    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));"
                )
                L.append(
                    f"    float s1 = std::bit_cast<float>({src_ops[1]}.read_lane(wf, lane));"
                )
                L.append(f"    float k = std::bit_cast<float>({k_expr});")
                L.append(
                    f"    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, s1, k)));"
                )
            L.append("  }")
            return "\n".join(L)

        if cls in (
            "vector_cvt_pk_u8_f32",
            "vector_cvt_pknorm",
            "vector_cvt_pkrtz_f16_f32",
            "vector_cvt_pk",
            "vector_cvt_pk_f16_f32",
            "vector_cvt_pk_bf16_f32",
            "vector_cvt_sr_f16_f32",
            "vector_cvt_sr_bf16_f32",
            "vector_pack_b32_f16",
        ):
            return gen_vector_cvt_pk(dst_ops, src_ops, cls, op)

        # ----- VOP3P: packed / dot / mix / MFMA -----
        if cls.startswith("dot2_"):
            return gen_dot2(
                dst_ops, src_ops, cls, opsel_exprs=self._vop3p_opsel_exprs()
            )

        if cls.startswith("dot4_"):
            return gen_dot4(dst_ops, src_ops, cls)

        if cls.startswith("dot8_"):
            return gen_dot8(dst_ops, src_ops, cls)

        if cls == "smem_load":
            return self._gen_smem_load(dst_ops, src_ops, sem)

        if cls == "smem_store":
            return self._gen_smem_store(dst_ops, src_ops, sem)

        if cls == "flat_load":
            return self._gen_flat_load(dst_ops, src_ops, sem)

        if cls == "flat_store":
            return self._gen_flat_store(dst_ops, src_ops, sem)

        if cls in ("buffer_load", "tbuffer_load"):
            return self._gen_buffer_load(dst_ops, src_ops, sem, cls, inst)

        if cls in ("buffer_store", "tbuffer_store"):
            return self._gen_buffer_store(dst_ops, src_ops, sem, cls)

        if cls in (
            "ds_read",
            "ds_read2",
            "ds_write",
            "ds_write2",
            "ds_read_addtid",
            "ds_write_addtid",
            "ds_read_tr_b16",
            "ds_read_tr_b8",
            "ds_read_tr_b4",
            "ds_read_tr_b6",
        ):
            gds_guard = ""
            if self._enc_has_field("gds"):
                gds_guard = (
                    "  if (inst_.gds)\n"
                    "    throw util::UnimplementedInst(mnemonic());\n"
                )
            if cls == "ds_read":
                return gds_guard + self._gen_ds_read(dst_ops, src_ops, sem)
            if cls == "ds_read2":
                return gds_guard + self._gen_ds_read2(dst_ops, src_ops, sem)
            if cls == "ds_write":
                return gds_guard + self._gen_ds_write(dst_ops, src_ops, sem)
            if cls == "ds_write2":
                return gds_guard + self._gen_ds_write2(dst_ops, src_ops, sem)
            if cls == "ds_read_addtid":
                return gds_guard + self._gen_ds_read_addtid(dst_ops, src_ops, sem)
            if cls == "ds_write_addtid":
                return gds_guard + self._gen_ds_write_addtid(dst_ops, src_ops, sem)
            if cls.startswith("ds_read_tr_"):
                return gds_guard + self._gen_ds_read_tr(dst_ops, src_ops, sem)
            return gds_guard + self._gen_ds_write2(dst_ops, src_ops, sem)

        if cls == "dcache_inv":
            return "  wf.cu().l1_scalar().invalidate_all();"

        if cls == "dcache_wb":
            return "  wf.cu().l1_scalar().writeback_all(wf.process_id());"

        if cls == "gl1_inv":
            return (
                "  wf.cu().l1_vector().invalidate_all();\n"
                "  if (auto *l2 = wf.cu().l2())\n"
                "    l2->flush_all(wf.process_id());"
            )

        if cls == "gl2_wb":
            return (
                "  if (auto *l2 = wf.cu().l2())\n" "    l2->flush_all(wf.process_id());"
            )

        if cls == "smem_time":
            return (
                "  static thread_local uint64_t counter = 0;\n"
                "  counter += 100;\n"
                "  uint32_t dst = wf.sgpr_alloc().base + inst_.sdata;\n"
                "  wf.cu().write_sgpr(dst, static_cast<uint32_t>(counter));\n"
                "  wf.cu().write_sgpr(dst + 1, static_cast<uint32_t>(counter >> 32));"
            )

        if cls == "flat_atomic":
            return self._gen_flat_atomic(dst_ops, src_ops, sem)

        if cls == "buffer_atomic":
            return self._gen_buffer_atomic(dst_ops, src_ops, sem)

        if cls == "ds_atomic":
            gds_guard = ""
            if self._enc_has_field("gds"):
                gds_guard = (
                    "  if (inst_.gds)\n"
                    "    throw util::UnimplementedInst(mnemonic());\n"
                )
            return gds_guard + self._gen_ds_atomic(dst_ops, src_ops, sem)

        if cls == "ds_permute":
            is_bpermute = "BPERMUTE" in sem.name.upper()
            L.append(f"  auto &cu = wf.cu();")
            L.append(f"  uint64_t exec = wf.exec();")
            L.append(f"  uint32_t vb = wf.vgpr_alloc().base;")
            L.append(f"  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);")
            L.append(f"  // Pre-read all data0 values from every lane.")
            L.append(f"  uint32_t src_data[64];")
            L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i)")
            L.append(f"    src_data[i] = cu.read_vgpr(vb + inst_.data0, i);")
            if is_bpermute:
                # DS_BPERMUTE_B32 (ISA spec pseudocode, page 476):
                #   tmp[i] = 0 for all lanes
                #   for i in 0..63:
                #     src_lane = (VGPR[i][ADDR] + OFFSET) / 4 % 64
                #     if EXEC[src_lane]: tmp[i] = VGPR[src_lane][DATA0]
                #   for i in 0..63:
                #     if EXEC[i]: VGPR[i][VDST] = tmp[i]
                L.append(f"  uint32_t tmp[64] = {{}};")
                L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{")
                L.append(f"    uint32_t addr_val = cu.read_vgpr(vb + inst_.addr, i);")
                L.append(
                    f"    uint32_t src_lane = ((addr_val + offset) / 4) % wf.wf_size();"
                )
                L.append(f"    if (exec & (1ULL << src_lane))")
                L.append(f"      tmp[i] = src_data[src_lane];")
                L.append(f"  }}")
                L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{")
                L.append(f"    if (exec & (1ULL << i))")
                L.append(f"      cu.write_vgpr(vb + inst_.vdst, i, tmp[i]);")
                L.append(f"  }}")
            else:
                # DS_PERMUTE_B32 (ISA spec pseudocode, page 475):
                #   tmp[i] = 0 for all lanes
                #   for i in 0..63:
                #     if EXEC[i]:
                #       dst_lane = (VGPR[i][ADDR] + OFFSET) / 4 % 64
                #       tmp[dst_lane] = VGPR[i][DATA0]
                #   for i in 0..63:
                #     if EXEC[i]: VGPR[i][VDST] = tmp[i]
                L.append(f"  uint32_t tmp[64] = {{}};")
                L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{")
                L.append(f"    if (!(exec & (1ULL << i))) continue;")
                L.append(f"    uint32_t addr_val = cu.read_vgpr(vb + inst_.addr, i);")
                L.append(
                    f"    uint32_t dst_lane = ((addr_val + offset) / 4) % wf.wf_size();"
                )
                L.append(f"    tmp[dst_lane] = src_data[i];")
                L.append(f"  }}")
                L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{")
                L.append(f"    if (exec & (1ULL << i))")
                L.append(f"      cu.write_vgpr(vb + inst_.vdst, i, tmp[i]);")
                L.append(f"  }}")
            return "\n".join(L)

        if cls == "ds_swizzle":
            # DS_SWIZZLE_B32: lane swizzle controlled by offset field.
            # The offset encodes the swizzle pattern. For QDMode (bit 15=1):
            #   for each lane in quad: dst = src[and_mask & or_mask ^ xor_mask]
            # For BitMode (bit 15=0): full-wave swizzle via and/or/xor.
            # Simplified: treat as identity (passthrough) for now.
            L.append(f"  auto &cu = wf.cu();")
            L.append(f"  uint64_t exec = wf.exec();")
            L.append(f"  uint32_t vb = wf.vgpr_alloc().base;")
            L.append(f"  uint32_t src_data[64];")
            L.append(f"  for (uint32_t i = 0; i < wf.wf_size(); ++i)")
            L.append(f"    src_data[i] = cu.read_vgpr(vb + inst_.data0, i);")
            L.append(f"  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);")
            L.append(f"  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{")
            L.append(f"    if (!(exec & (1ULL << lane))) continue;")
            L.append(f"    uint32_t src_lane;")
            L.append(f"    if (offset & 0x8000) {{")
            L.append(f"      // QDMode: swizzle within 4-lane quads.")
            L.append(f"      uint32_t and_mask = offset & 0x1F;")
            L.append(f"      uint32_t or_mask = (offset >> 5) & 0x1F;")
            L.append(f"      uint32_t xor_mask = (offset >> 10) & 0x1F;")
            L.append(f"      src_lane = ((lane & and_mask) | or_mask) ^ xor_mask;")
            L.append(
                f"      src_lane = (lane & ~0x3) | (src_lane & 0x3);  // stay in quad"
            )
            L.append(f"    }} else {{")
            L.append(f"      // BitMode: full-wave swizzle.")
            L.append(f"      uint32_t and_mask = offset & 0x1F;")
            L.append(f"      uint32_t or_mask = (offset >> 5) & 0x1F;")
            L.append(f"      uint32_t xor_mask = (offset >> 10) & 0x1F;")
            L.append(f"      src_lane = ((lane & and_mask) | or_mask) ^ xor_mask;")
            L.append(f"    }}")
            L.append(f"    if (src_lane < wf.wf_size())")
            L.append(f"      cu.write_vgpr(vb + inst_.vdst, lane, src_data[src_lane]);")
            L.append(f"  }}")
            return "\n".join(L)

        # ── Image pipeline stubs ──────────────────────────────────────────
        if cls == "image_load":
            # Minimal image load: treat as a flat read from the image resource base address.
            # Full image addressing (texture coordinates, dimensions) not yet implemented.
            L.append("  // Minimal image load stub — not yet implemented.")
            L.append("  (void)wf;")
            return "\n".join(L)

        if cls == "image_store":
            L.append("  // Minimal image store stub — not yet implemented.")
            L.append("  (void)wf;")
            return "\n".join(L)

        if cls in ("image_atomic", "image_sample", "image_query", "image_bvh"):
            L.append("  (void)wf; // Image pipeline not yet implemented.")
            return "\n".join(L)

        # ── Graphics-only stubs (no-ops in compute simulation) ───────────
        if cls == "export":
            L.append("  (void)wf; // Export: no-op in compute simulation.")
            return "\n".join(L)

        if cls in ("interp", "lds_direct"):
            L.append(
                "  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation."
            )
            return "\n".join(L)

        return f"  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: {cls}"

    def _gen_smem_load(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        nd = sem.num_elems
        L.append("  auto d = std::make_unique<amdgpu::ScalarMemState>();")
        L.append(f"  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;")
        L.append(f"  d->num_dwords = {nd};")
        L.append("  d->is_load = true;")
        L.append(f"  d->mtype = {self._mtype_expr(is_smem=True)};")
        L.append("  d->addr = smem_calculate_address(inst_, wf);")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_smem_store(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        nd = sem.num_elems
        L.append("  auto d = std::make_unique<amdgpu::ScalarMemState>();")
        L.append(f"  d->num_dwords = {nd};")
        L.append("  d->is_load = false;")
        L.append(f"  d->mtype = {self._mtype_expr(is_smem=True)};")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;")
        L.append(f"  for (uint32_t i = 0; i < {nd}; ++i)")
        L.append("    d->store_data[i] = cu.read_sgpr(sdata_base + i);")
        L.append("  d->addr = smem_calculate_address(inst_, wf);")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _vop3p_opsel_exprs(self) -> tuple[str, str]:
        """Return ``(op_sel_expr, op_sel_hi_expr)`` for VOP3P execute() bodies."""
        opsel, opsel_hi = self.isa_spec.profile.vop3p_opsel_fields
        return f"inst_.{opsel}", f"inst_.{opsel_hi}"

    def _coherency_exprs(self) -> tuple[str, str, str]:
        """Return ``(sc0_expr, sc1_expr, nt_expr)`` for execute() body templates.

        Consults the ISA profile so that ISAs with GLC/SLC field names (CDNA1/2,
        RDNA1-3.5) emit ``inst_.glc`` / ``inst_.slc`` instead of the CDNA3/4
        ``inst_.sc0`` / ``inst_.sc1``.  When the profile has no NT field the
        nt_expr is the literal ``0``.
        """
        sc0, sc1, nt = self.isa_spec.profile.coherency_field_names
        sc0_expr = f"inst_.{sc0}"
        sc1_expr = f"inst_.{sc1}"
        nt_expr = f"inst_.{nt}" if nt else "0"
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
                return "amdgpu::mtype_from_flags_gfx9(inst_.glc)"
            if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC:
                # SMEM on GFX10 has glc+dlc but no slc.
                return "amdgpu::mtype_from_flags_gfx10(inst_.glc, inst_.dlc, false)"
            if model == MemoryCoherencyModel.GFX11_SC0_SC1_TH:
                # SMEM on GFX11 has glc+dlc but no slc.
                return "amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false)"
            if model == MemoryCoherencyModel.GFX12_SCOPE_TH:
                return "amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th)"
        if model == MemoryCoherencyModel.GFX9_GLC:
            return "amdgpu::mtype_from_flags_gfx9(inst_.glc)"
        if model == MemoryCoherencyModel.GFX940_SC0_SC1_NT:
            return "amdgpu::mtype_from_flags_gfx940(inst_.sc0, inst_.sc1, inst_.nt)"
        if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC:
            return "amdgpu::mtype_from_flags_gfx10(inst_.glc, inst_.dlc, inst_.slc)"
        if model == MemoryCoherencyModel.GFX11_SC0_SC1_TH:
            return "amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, inst_.slc)"
        if model == MemoryCoherencyModel.GFX12_SCOPE_TH:
            return "amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th)"
        return "amdgpu::Mtype::RW"

    def _cache_flags_includes(self) -> list[str]:
        """Return cache_flags header path(s) for this ISA's coherency model.

        GFX940 (CDNA3/4) needs both gfx940 (vector memory) and gfx9 (SMEM).
        """
        from amdisa.isa_profile import MemoryCoherencyModel

        model = self.isa_spec.profile.coherency_model
        base = "rocjitsu/isa/arch/amdgpu/shared"
        if model == MemoryCoherencyModel.GFX940_SC0_SC1_NT:
            return [f"{base}/gfx940_cache_flags.h", f"{base}/gfx9_cache_flags.h"]
        _MAP = {
            MemoryCoherencyModel.GFX9_GLC: "gfx9_cache_flags.h",
            MemoryCoherencyModel.GFX10_GLC_DLC_SLC: "gfx10_cache_flags.h",
            MemoryCoherencyModel.GFX11_SC0_SC1_TH: "gfx11_cache_flags.h",
            MemoryCoherencyModel.GFX12_SCOPE_TH: "gfx12_cache_flags.h",
        }
        return [f"{base}/{_MAP[model]}"]

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
            "smem_load": (
                "amdgpu::WaitCounterType::KMCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "smem_store": (
                "amdgpu::WaitCounterType::KMCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "flat_load": (
                "amdgpu::WaitCounterType::LOADCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::VMCNT"
            ),
            "flat_store": (
                "amdgpu::WaitCounterType::STORECNT"
                if is_gfx11_plus
                else (
                    "amdgpu::WaitCounterType::VSCNT"
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else "amdgpu::WaitCounterType::VMCNT"
                )
            ),
            "flat_atomic": (
                "amdgpu::WaitCounterType::LOADCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::VMCNT"
            ),
            "buffer_load": (
                "amdgpu::WaitCounterType::LOADCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::VMCNT"
            ),
            "buffer_store": (
                "amdgpu::WaitCounterType::STORECNT"
                if is_gfx11_plus
                else (
                    "amdgpu::WaitCounterType::VSCNT"
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else "amdgpu::WaitCounterType::VMCNT"
                )
            ),
            "tbuffer_load": (
                "amdgpu::WaitCounterType::LOADCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::VMCNT"
            ),
            "tbuffer_store": (
                "amdgpu::WaitCounterType::STORECNT"
                if is_gfx11_plus
                else (
                    "amdgpu::WaitCounterType::VSCNT"
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else "amdgpu::WaitCounterType::VMCNT"
                )
            ),
            "global_load": (
                "amdgpu::WaitCounterType::LOADCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::VMCNT"
            ),
            "global_store": (
                "amdgpu::WaitCounterType::STORECNT"
                if is_gfx11_plus
                else (
                    "amdgpu::WaitCounterType::VSCNT"
                    if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                    else "amdgpu::WaitCounterType::VMCNT"
                )
            ),
            "ds_read": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read2": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_write": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_write2": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_atomic": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read_addtid": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_write_addtid": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read_tr_b16": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read_tr_b8": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read_tr_b4": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
            "ds_read_tr_b6": (
                "amdgpu::WaitCounterType::DSCNT"
                if is_gfx11_plus
                else "amdgpu::WaitCounterType::LGKMCNT"
            ),
        }
        return _MAP.get(sem_class)

    @property
    def _acc_vgpr_expr(self) -> str:
        """AccVGPR offset expression for the current encoding.

        When the encoding has an ``acc`` bit field and acc=1, data/vdata/vdst
        references AccVGPRs (physical +256). For encodings without ``acc``
        (CDNA1 DS/FLAT, all RDNA), returns ``0u``.
        """
        if hasattr(self, "_current_inst_fields") and "acc" in self._current_inst_fields:
            return "(inst_.acc ? 256u : 0u)"
        return "0u"

    def _gen_flat_load(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = true;")
        if sem.sign_extend:
            L.append("  d->sign_extend = true;")
        if sem.d16_hi:
            L.append("  d->d16_hi = true;")
        if sem.d16_lo:
            L.append("  d->d16_lo = true;")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        L.append("  flat_calculate_addresses(inst_, wf, *d);")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_flat_store(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        data_field = self.isa_spec.profile.flat_store_src_field
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = false;")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        L.append("  flat_calculate_addresses(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = esz * ne
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        for i in range(ne):
            if esz == 4:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field} + {i}, lane);"
                )
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 4);"
                )
            elif esz == 2:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field}, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);"
                )
            elif esz == 1:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field}, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});"
                )
        L.append("  }")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    _ATOMIC_OP_ENUM: dict[str, str] = {
        "swap": "amdgpu::AtomicOp::SWAP",
        "cmpswap": "amdgpu::AtomicOp::CMPSWAP",
        "add": "amdgpu::AtomicOp::ADD",
        "sub": "amdgpu::AtomicOp::SUB",
        "rsub": "amdgpu::AtomicOp::RSUB",
        "smin": "amdgpu::AtomicOp::SMIN",
        "umin": "amdgpu::AtomicOp::UMIN",
        "smax": "amdgpu::AtomicOp::SMAX",
        "umax": "amdgpu::AtomicOp::UMAX",
        "and": "amdgpu::AtomicOp::AND",
        "or": "amdgpu::AtomicOp::OR",
        "xor": "amdgpu::AtomicOp::XOR",
        "inc": "amdgpu::AtomicOp::INC",
        "dec": "amdgpu::AtomicOp::DEC",
        "fadd": "amdgpu::AtomicOp::FADD",
        "fmin": "amdgpu::AtomicOp::FMIN",
        "fmax": "amdgpu::AtomicOp::FMAX",
    }

    def _gen_flat_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate flat_atomic execute() body.

        If the operation is recognized, emits a full VectorMemState setup
        with AtomicOp for the pipeline. Unrecognized variants (FP atomics,
        etc.) fall back to a TODO stub.
        """
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f"  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled flat_atomic variant ({sem.name})"

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1  # number of dwords of operand data

        L = []
        sc0, sc1, nt = self._coherency_exprs()
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        acc = self._acc_vgpr_expr
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append("  d->num_elems = 1;")
        L.append(f"  d->is_load = ({sc0} != 0);")
        L.append(f"  d->atomic_op = {op_enum};")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        data_field = self.isa_spec.profile.flat_store_src_field
        L.append("  flat_calculate_addresses(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = data_dwords * 4
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        for i in range(data_dwords):
            L.append(
                f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field} + {i}, lane);"
            )
            L.append(
                f"    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);"
            )
        L.append("  }")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_buffer_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate buffer_atomic execute() body (MUBUF encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f"  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled buffer_atomic variant ({sem.name})"

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1

        L = []
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdata;")
        L.append(f"  d->elem_size = {esz};")
        L.append("  d->num_elems = 1;")
        L.append(f"  d->is_load = ({sc0} != 0);")
        L.append(f"  d->atomic_op = {op_enum};")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        L.append("  mubuf_calculate_addresses(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = data_dwords * 4
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        for i in range(data_dwords):
            L.append(
                f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata + {i}, lane);"
            )
            L.append(
                f"    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);"
            )
        L.append("  }")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_atomic(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """Generate ds_atomic execute() body (DS encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f"  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant ({sem.name})"

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1

        L = []
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append("  d->num_elems = 1;")
        # DS atomics always return the old value (like GLC=1).
        L.append("  d->is_load = true;")
        L.append(f"  d->atomic_op = {op_enum};")
        L.append("  ds_calculate_addresses(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = data_dwords * 4
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        is_cmpswap = sem.operation == "cmpswap"
        half = data_dwords // 2
        for i in range(data_dwords):
            if is_cmpswap and i >= half:
                reg = f"inst_.data1 + {i - half}"
            else:
                reg = f"inst_.data0 + {i}"
            L.append(
                f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + {reg}, lane);"
            )
            L.append(
                f"    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);"
            )
        L.append("  }")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_buffer_load(
        self,
        dst: list[str],
        src: list[str],
        sem: InstructionSemantics,
        cls: str = "buffer_load",
        inst: "Instruction | None" = None,
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        addr_fn = (
            "mtbuf_calculate_addresses"
            if cls == "tbuffer_load"
            else "mubuf_calculate_addresses"
        )
        # Check if this encoding has an 'lds' field in the machine instruction.
        has_lds_field = False
        if inst is not None:
            enc = self.isa_spec.encoding_map.get(inst.enc_name)
            if enc is not None:
                has_lds_field = any(f.name == "lds" for f in enc.ucode_fields)
        # When the LDS bit is set, the buffer load reads from global memory but
        # writes the result to LDS at M0 + lane_offset instead of to VGPRs.
        # Route through the global memory pipeline for the load, then let the
        # pipeline writeback path detect lds_dst and scatter to LDS.
        if has_lds_field:
            L.append("  if (inst_.lds) {")
            L.append(
                "    auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
            )
            L.append(f"    d->elem_size = {esz};")
            L.append(f"    d->num_elems = {ne};")
            L.append("    d->is_load = true;")
            L.append("    d->lds_dst = true;")
            L.append("    d->lds_base = wf.m0() + wf.lds_base();")
            L.append(f"    d->mtype = {self._mtype_expr()};")
            L.append(f"    d->non_temporal = {nt};")
            L.append(f"    {addr_fn}(inst_, wf, *d);")
            L.append("    set_data(std::move(d));")
            L.append("    return;")
            L.append("  }")
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdata;")
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = true;")
        if sem.sign_extend:
            L.append("  d->sign_extend = true;")
        if sem.d16_hi:
            L.append("  d->d16_hi = true;")
        if sem.d16_lo:
            L.append("  d->d16_lo = true;")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        L.append(f"  {addr_fn}(inst_, wf, *d);")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_buffer_store(
        self,
        dst: list[str],
        src: list[str],
        sem: InstructionSemantics,
        cls: str = "buffer_store",
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        addr_fn = (
            "mtbuf_calculate_addresses"
            if cls == "tbuffer_store"
            else "mubuf_calculate_addresses"
        )
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);"
        )
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = false;")
        L.append(f"  d->mtype = {self._mtype_expr()};")
        L.append(f"  d->non_temporal = {nt};")
        L.append(f"  {addr_fn}(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = esz * ne
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        for i in range(ne):
            if esz >= 4:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata + {i}, lane);"
                )
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, {esz});"
                )
            elif esz == 2:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);"
                )
            elif esz == 1:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});"
                )
        L.append("  }")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_read(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = true;")
        if sem.sign_extend:
            L.append("  d->sign_extend = true;")
        if sem.d16_hi:
            L.append("  d->d16_hi = true;")
        if sem.d16_lo:
            L.append("  d->d16_lo = true;")
        L.append("  ds_calculate_addresses(inst_, wf, *d);")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_read_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """ds_read_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;")
        L.append(f"  d->elem_size = {sem.elem_size};")
        L.append(f"  d->num_elems = {sem.num_elems};")
        L.append("  d->is_load = true;")
        L.append("  {")
        L.append("    uint64_t exec = wf.exec();")
        L.append("    d->lane_mask = exec; d->exec_mask = exec;")
        L.append("    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();")
        L.append("    d->cu_path = wf.cu().full_path();")
        L.append(
            "    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;"
        )
        L.append("    uint32_t m0 = wf.m0();")
        L.append("    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;")
        L.append("    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("      if (!(exec & (1ULL << lane))) continue;")
        L.append(
            "      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();"
        )
        L.append("    }")
        L.append("  }")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_write_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """ds_write_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->elem_size = {sem.elem_size};")
        L.append(f"  d->num_elems = {sem.num_elems};")
        L.append("  d->is_load = false;")
        L.append("  {")
        L.append("    uint64_t exec = wf.exec();")
        L.append("    d->lane_mask = exec; d->exec_mask = exec;")
        L.append("    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();")
        L.append("    d->cu_path = wf.cu().full_path();")
        L.append(
            "    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;"
        )
        L.append("    uint32_t m0 = wf.m0();")
        L.append("    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;")
        L.append("    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("      if (!(exec & (1ULL << lane))) continue;")
        L.append(
            "      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();"
        )
        L.append("    }")
        L.append("  }")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        L.append(f"  d->store_data.resize(wf.wf_size() * {sem.elem_size});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        L.append(
            f"    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);"
        )
        L.append(
            f"    std::memcpy(&d->store_data[lane * {sem.elem_size}], &val0, {sem.elem_size});"
        )
        L.append("  }")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_read_tr(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """ds_read_b64_tr_b16 etc: DS read + cross-lane transpose post-processing.

        Uses the standard DS read pipeline (MEMORY_OP) with elem_size=4,
        num_elems=2 (for B64) or 3 (for B96). Sets d->transpose to signal
        the memory pipeline to apply the cross-lane shuffle after the raw read.
        """
        # TR_B4=1, TR_B6=2, TR_B8=3, TR_B16=4
        tr_map = {
            "ds_read_tr_b4": (4, 2, 1),  # elem_size=4, num_elems=2, transpose=1
            "ds_read_tr_b6": (4, 3, 2),  # elem_size=4, num_elems=3, transpose=2
            "ds_read_tr_b8": (4, 2, 3),  # elem_size=4, num_elems=2, transpose=3
            "ds_read_tr_b16": (4, 2, 4),  # elem_size=4, num_elems=2, transpose=4
        }
        esz, ne, tr_kind = tr_map.get(sem.semantic_class, (4, 2, 4))
        L = []
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = true;")
        L.append(f"  d->transpose = {tr_kind};")
        L.append("  ds_calculate_addresses(inst_, wf, *d);")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _gen_ds_write(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->elem_size = {esz};")
        L.append(f"  d->num_elems = {ne};")
        L.append("  d->is_load = false;")
        L.append("  ds_calculate_addresses(inst_, wf, *d);")
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        stride = esz * ne
        L.append(f"  d->store_data.resize(wf.wf_size() * {stride});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        for i in range(ne):
            off = i * esz
            if esz == 8:
                vgpr_base = i * 2
                L.append(
                    f"    uint32_t lo{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {vgpr_base}, lane);"
                )
                L.append(
                    f"    uint32_t hi{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {vgpr_base + 1}, lane);"
                )
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {off}], &lo{i}, 4);"
                )
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {off + 4}], &hi{i}, 4);"
                )
            elif esz == 4:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {i}, lane);"
                )
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 4);"
                )
            elif esz == 2:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 2);"
                )
            elif esz == 1:
                L.append(
                    f"    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0, lane);"
                )
                if sem.d16_hi:
                    L.append(f"    val{i} >>= 16;")
                L.append(
                    f"    d->store_data[lane * {stride} + {off}] = static_cast<uint8_t>(val{i});"
                )
        L.append("  }")
        # Counter increment handled by MemoryPipeline::issue().
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

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
        if sem.operation == "st64":
            stride_scale = f"{esz * 64}U"
        else:
            stride_scale = f"{esz}U"
        acc = self._acc_vgpr_expr
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;")
        L.append(f"  d->elem_size = {esz};")
        L.append("  d->num_elems = 1;")
        L.append("  d->is_load = true;")
        L.append("  d->exec_mask = exec;")
        L.append("  d->lane_mask = exec;")
        L.append("  d->ds2_active = true;")
        L.append(
            f"  d->ds2_dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst + {dwords_per_access};"
        )
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        L.append(
            "    uint32_t base = cu.read_vgpr(wf.vgpr_alloc().base + inst_.addr, lane);"
        )
        L.append(
            f"    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();"
        )
        L.append(
            f"    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();"
        )
        L.append("  }")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

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
        if sem.operation == "st64":
            stride_scale = f"{esz * 64}U"
        else:
            stride_scale = f"{esz}U"
        acc = self._acc_vgpr_expr
        L.append("  auto &cu = wf.cu();")
        L.append("  uint64_t exec = wf.exec();")
        L.append(
            "  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);"
        )
        L.append(f"  d->elem_size = {esz};")
        L.append("  d->num_elems = 1;")
        L.append("  d->is_load = false;")
        L.append("  d->exec_mask = exec;")
        L.append("  d->lane_mask = exec;")
        L.append("  d->ds2_active = true;")
        L.append(f"  d->store_data.resize(wf.wf_size() * {esz});")
        L.append(f"  d->ds2_store_data.resize(wf.wf_size() * {esz});")
        L.append("  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {")
        L.append("    if (!(exec & (1ULL << lane))) continue;")
        L.append(
            "    uint32_t base = cu.read_vgpr(wf.vgpr_alloc().base + inst_.addr, lane);"
        )
        L.append(
            f"    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();"
        )
        L.append(
            f"    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();"
        )
        # Pack data0 into store_data
        for i in range(dwords_per_access):
            L.append(
                f"    uint32_t v0_{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {i}, lane);"
            )
            L.append(
                f"    std::memcpy(&d->store_data[lane * {esz} + {i * 4}], &v0_{i}, 4);"
            )
        # Pack data1 into ds2_store_data
        for i in range(dwords_per_access):
            L.append(
                f"    uint32_t v1_{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data1 + {i}, lane);"
            )
            L.append(
                f"    std::memcpy(&d->ds2_store_data[lane * {esz} + {i * 4}], &v1_{i}, 4);"
            )
        L.append("  }")
        L.append("  set_data(std::move(d));")
        return "\n".join(L)

    def _enc_has_field(self, field_name: str) -> bool:
        """Check if the current encoding struct has a named field.

        Uses the struct field names from the machine instruction encoding.
        Falls back to checking _current_inst_fields (ucode_fields + parent
        fields) and the encoding name for known patterns.
        """
        if hasattr(self, "_current_inst_fields") and self._current_inst_fields:
            if field_name in self._current_inst_fields:
                return True
        if field_name == "gds" and hasattr(self, "_current_enc") and self._current_enc:
            return self._current_enc.enc_name.upper() == "ENC_DS"
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
            "smem_load",
            "smem_store",
            "flat_load",
            "flat_store",
            "flat_atomic",
            "buffer_load",
            "buffer_store",
            "buffer_atomic",
            "tbuffer_load",
            "tbuffer_store",
            "ds_read",
            "ds_read2",
            "ds_write",
            "ds_write2",
            "ds_atomic",
            "global_load",
            "global_store",
            "dcache_inv",
            "dcache_wb",
            "image_load",
            "image_store",
            "image_atomic",
            "image_sample",
            "image_query",
            # Nop/stub bodies don't benefit from sharing:
            "nop",
            # ISA-dependent control flow (reference Isa:: constants or size_):
            "waitcnt",
            "wait_counter",
            "endpgm",
            "branch",
            "cbranch",
            "scalar_getpc",
            "scalar_setpc",
            "scalar_swappc",
            "scalar_call",
            # MFMA/WMMA reference ISA-specific headers:
            "mfma",
            # Interp/export use ISA-specific encoding struct fields:
            "interp",
            "export",
            # AccVGPR read/write use ISA-specific register file:
            "accvgpr_read",
            "accvgpr_write",
            # Vector swap accesses protected inst_ member:
            "vector_swap",
            # Vector readlane/writelane/readfirstlane access encoding fields:
            "vector_readlane",
            "vector_writelane",
            "vector_readfirstlane",
            # V_CMPX writes VCC+EXEC on CDNA but only EXEC on RDNA:
            "vector_cmpx",
            "vector_cmpx_class",
        }
    )

    def _can_share_execute(self, mnemonic: str) -> bool:
        """Check if an instruction's execute() body can be shared across ISAs.

        An instruction is shareable if:
        1. It exists on 2+ ISAs with the same semantic class (family_shared
           or universal in the shared_plan).
        2. Its semantic class is profile-independent (no mtype/coherency calls).
        3. The current ISA is one of the ISAs that share this instruction.
        """
        if self.shared_plan is None:
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
        # encodings. Search all families for any entry matching this mnemonic
        # that includes the current ISA.
        for fam_insts in self.shared_plan.family_shared.values():
            for (mn, enc_name), info in fam_insts.items():
                if mn != mnemonic:
                    continue
                if info.semantic_class in self._NON_SHAREABLE_CLASSES:
                    return False
                if arch in info.isa_names and len(info.isa_names) >= 2:
                    return True
        return False

    def gen_insts(self) -> None:
        """Generate instruction classes deriving from encoding classes.

        When ``shared_plan`` is set (``--multi`` mode), universal instructions
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
            class_func_impls = []
            # Collect instructions from this encoding plus any child
            # alt encodings that contribute to this file.
            all_insts = list(enc.insts)
            for child in child_encs.get(enc.enc_name, []):
                all_insts.extend(child.insts)
            if all_insts and not profile.is_alt_encoding(enc.enc_name):
                enc_field_names = {f.name for f in enc.ucode_fields}
                is_smem = enc.enc_name.upper() == "ENC_SMEM"
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
                    public_members = [cgen.Line("public:")]
                    private_members = []
                    opnd_ctor_init = []
                    opnd_body = []
                    src_idx = 0
                    dst_idx = 0
                    reads_dst = self._dst_is_also_source(inst)
                    for opnd in inst.operands:
                        if opnd.is_input:
                            opnd_body.append(
                                f"src_operands_[{src_idx}] = &{opnd.name};"
                            )
                            src_idx += 1
                        elif (
                            reads_dst
                            and opnd.is_output
                            and opnd.name in ("vdst", "sdst")
                        ):
                            opnd_body.append(
                                f"src_operands_[{src_idx}] = &{opnd.name};"
                            )
                            src_idx += 1
                        if opnd.is_output:
                            opnd_body.append(
                                f"dst_operands_[{dst_idx}] = &{opnd.name};"
                            )
                            dst_idx += 1
                        if not opnd.is_input and not opnd.is_output:
                            opnd_body.append(
                                f"dst_operands_[{dst_idx}] = &{opnd.name};"
                            )
                            dst_idx += 1
                        private_members.append(cgen.Statement(f"Operand {opnd.name}"))
                        # SADDR member added after the operand loop below
                        if is_smem and opnd.name == "soffset":
                            opnd_ctor_init.append(
                                f"{opnd.name}(make_smem_offset("
                                f"reinterpret_cast<const OpEncoding*>(inst)))"
                            )
                        elif opnd.name in inst_field_names:
                            opr_type = opnd.operand_type
                            if inst_sem and inst_sem.accvgpr_srcs and opnd.is_input:
                                opr_type = "OPR_SRC_VGPR_OR_ACCVGPR"
                            opnd_ctor_init.append(
                                f"{opnd.name}({opnd.size}, "
                                f"OperandType::{opr_type}, "
                                f"reinterpret_cast<const OpEncoding*>(inst)"
                                f"->{opnd.name})"
                            )
                        else:
                            opnd_ctor_init.append(
                                f"{opnd.name}({opnd.size}, "
                                f"OperandType::{opnd.operand_type}, 0)"
                            )
                    # For flat encodings with a seg field, add saddr as an
                    # optional operand. Declared and initialized LAST among
                    # operands to avoid reorder warnings with data/addr.
                    _has_flat_saddr = self.isa_spec.profile.mnemonic_rule(
                        enc.enc_name
                    ).use_flat_mnemonic
                    if _has_flat_saddr:
                        private_members.append(cgen.Statement("Operand saddr"))
                        opnd_ctor_init.append("saddr(0, OperandType::OPR_SREG, 0)")

                    class_ctor_decl = cgen.FunctionDeclaration(
                        cgen.Value("", inst.fmt_name),
                        [cgen.Value("const MachineInst *", "inst")],
                    )
                    public_members.append(class_ctor_decl)
                    public_members.append(
                        cgen.Statement("void execute_impl(amdgpu::Wavefront &wf)")
                    )
                    # CFG metadata is emitted on the concrete ISA instruction
                    # class, not inferred by generic analysis from mnemonic
                    # strings. BasicBlock asks the virtual branch_offset_bytes()
                    # for direct branch targets.
                    label_operand = next(
                        (
                            op.name
                            for op in inst.operands
                            if op.operand_type == "OPR_LABEL"
                        ),
                        None,
                    )
                    if (
                        inst_sem
                        and inst_sem.semantic_class in ("branch", "cbranch")
                        and label_operand
                    ):
                        public_members.append(
                            cgen.Statement(
                                "std::optional<int64_t> branch_offset_bytes() const override"
                            )
                        )
                    # Embed the full mnemonic (with suffix) as a string literal
                    # so the encoding base gets a string_view to static storage.
                    rule = self.isa_spec.profile.mnemonic_rule(enc.enc_name)
                    full_mnemonic = inst.mnemonic + (rule.suffix or "")
                    init_list_parts = [
                        f'{inst.fmt_true_enc_name}("{full_mnemonic}", '
                        f"reinterpret_cast<const OpEncoding*>(inst), "
                        f"make_exec_fn<{inst.fmt_name}>())"
                    ] + opnd_ctor_init
                    init_list = ", ".join(init_list_parts)
                    # Check if this is a memory instruction to set MEMORY_OP flag
                    _mem_sem = inst_sem
                    _MEM_CLASSES = frozenset(
                        {
                            "smem_load",
                            "smem_store",
                            "flat_load",
                            "flat_store",
                            "flat_atomic",
                            "buffer_load",
                            "buffer_store",
                            "buffer_atomic",
                            "tbuffer_load",
                            "tbuffer_store",
                            "ds_read",
                            "ds_read2",
                            "ds_write",
                            "ds_write2",
                            "ds_atomic",
                            "ds_read_addtid",
                            "ds_write_addtid",
                            "ds_read_tr_b16",
                            "ds_read_tr_b8",
                            "ds_read_tr_b4",
                            "ds_read_tr_b6",
                        }
                    )
                    ctor_body_parts = list(opnd_body)
                    ctor_body_parts.append(f"num_src_ = {src_idx};")
                    ctor_body_parts.append(f"num_dst_ = {dst_idx};")

                    # Flat segment-aware operands: adjust addr width and add
                    # saddr for SCRATCH (seg==1) and GLOBAL (seg==2) segments.
                    if rule.use_flat_mnemonic:
                        _has_addr = any(o.name == "addr" for o in inst.operands)
                        if _has_addr:
                            ctor_body_parts.append("if (inst_.seg == 1) {")
                            ctor_body_parts.append(
                                "  addr = Operand(32, OperandType::OPR_VGPR, "
                                "reinterpret_cast<const OpEncoding*>(&inst_)->addr);"
                            )
                            ctor_body_parts.append("  if (inst_.saddr != 0x7F) {")
                            ctor_body_parts.append(
                                "    saddr = Operand(32, OperandType::OPR_SREG, "
                                "inst_.saddr);"
                            )
                            ctor_body_parts.append(
                                "    src_operands_[num_src_++] = &saddr;"
                            )
                            ctor_body_parts.append("  }")
                            ctor_body_parts.append(
                                "} else if (inst_.seg == 2 && inst_.saddr != 0x7F) {"
                            )
                            ctor_body_parts.append(
                                "  addr = Operand(32, OperandType::OPR_VGPR, "
                                "reinterpret_cast<const OpEncoding*>(&inst_)->addr);"
                            )
                            ctor_body_parts.append(
                                "  saddr = Operand(64, OperandType::OPR_SREG, "
                                "inst_.saddr);"
                            )
                            ctor_body_parts.append(
                                "  src_operands_[num_src_++] = &saddr;"
                            )
                            ctor_body_parts.append("}")

                    # Literal constant fixup: when src0/ssrc0/ssrc1 == 255,
                    # replace the operand with the 32-bit literal from the
                    # extended instruction encoding.
                    _LIT_ENC_MAP = {
                        "ENC_SOP1": ("Sop1InstLiteralMachineInst", ["ssrc0"]),
                        "ENC_SOP2": ("Sop2InstLiteralMachineInst", ["ssrc0", "ssrc1"]),
                        "ENC_SOPC": ("SopcInstLiteralMachineInst", ["ssrc0", "ssrc1"]),
                        "ENC_VOP1": ("Vop1InstLiteralMachineInst", ["src0"]),
                        "ENC_VOP2": ("Vop2InstLiteralMachineInst", ["src0"]),
                        "ENC_VOPC": ("VopcInstLiteralMachineInst", ["src0"]),
                    }
                    _lit_info = _LIT_ENC_MAP.get(enc.enc_name.upper())
                    if _lit_info:
                        _lit_struct, _lit_fields = _lit_info
                        for opnd in inst.operands:
                            if (
                                opnd.name in _lit_fields
                                and opnd.name in enc_field_names
                            ):
                                ctor_body_parts.append(
                                    f"if (reinterpret_cast<const OpEncoding*>(inst)->{opnd.name} == 255) "
                                    f"{opnd.name} = Operand({opnd.size}, OperandType::OPR_SIMM32, "
                                    f"static_cast<int>(reinterpret_cast<const {_lit_struct}*>(inst)->simm32));"
                                )

                    # DPP fixup: when src0 == amdgpu::SRC_DPP (DPP marker), replace the
                    # src0 operand with vsrc0 from the DPP extension dword.
                    # This lets the instruction execute normally with the
                    # correct VGPR source. Lane permutation is not yet
                    # applied (identity permutation).
                    # DPP/SDWA: src0 marker values 250 (DPP) and 249 (SDWA)
                    # indicate the real VGPR index is in the extension dword.
                    # CDNA uses VopDpp, RDNA uses VopDpp16 (both have vsrc0).
                    _DPP_ENC_BASES = {
                        "ENC_VOP1": "Vop1",
                        "ENC_VOP2": "Vop2",
                        "ENC_VOPC": "Vop1",
                    }
                    _enc_base = _DPP_ENC_BASES.get(enc.enc_name.upper())
                    if _enc_base:
                        # CDNA (GFX9) uses VopDpp; RDNA (GFX10+) uses VopDpp16.
                        _is_rdna = any(
                            ie.enc_name.startswith("VOP1_VOP_DPP16")
                            for ie in self.isa_spec.inst_encodings
                        )
                        _dpp_suffix = "VopDpp16" if _is_rdna else "VopDpp"
                        _dpp_struct = f"{_enc_base}{_dpp_suffix}MachineInst"
                        for opnd in inst.operands:
                            if opnd.name == "src0" and opnd.name in enc_field_names:
                                # DPP (src0 == amdgpu::SRC_DPP): read vsrc0 and DPP control
                                # fields from the ISA-specific extension dword,
                                # storing them on the Instruction base for
                                # apply_dpp() to use later.
                                ctor_body_parts.append(
                                    f"if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_DPP) {{"
                                    f" auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);"
                                    f" src0 = Operand({opnd.size}, OperandType::OPR_VGPR, dp->vsrc0);"
                                    f" dpp_ctrl_ = dp->dpp_ctrl;"
                                    f" dpp_row_mask_ = dp->row_mask;"
                                    f" dpp_bank_mask_ = dp->bank_mask;"
                                    f" dpp_bound_ctrl_ = dp->bound_ctrl;"
                                    f"}}"
                                )
                                # SDWA (src0 == amdgpu::SRC_SDWA): CDNA and RDNA1/2 only.
                                _has_sdwa = any(
                                    "SDWA" in ie.enc_name
                                    for ie in self.isa_spec.inst_encodings
                                )
                                if _has_sdwa:
                                    if enc.enc_name.upper() == "ENC_VOPC":
                                        _sdwa_struct = "VopcVopSdwaSdstEncMachineInst"
                                    else:
                                        _sdwa_struct = f"{_enc_base}VopSdwaMachineInst"
                                    _sdwa_s1_code = ""
                                    if enc.enc_name.upper() in ("ENC_VOP2", "ENC_VOPC"):
                                        _sdwa_s1_code = (
                                            f" if (sw->s1)"
                                            f"   vsrc1 = Operand({opnd.size}, OperandType::OPR_SRC,"
                                            f"     reinterpret_cast<const OpEncoding*>(inst)->vsrc1);"
                                        )
                                    ctor_body_parts.append(
                                        f"if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_SDWA) {{"
                                        f" auto *sw = reinterpret_cast<const {_sdwa_struct}*>(inst);"
                                        f" src0 = Operand({opnd.size}, sw->s0 ? OperandType::OPR_SRC : OperandType::OPR_VGPR, sw->vsrc0);"
                                        f" sdwa_src0_sel_ = sw->src0_sel;"
                                        f" sdwa_src0_sext_ = sw->src0_sext;"
                                        f" sdwa_src0_neg_ = sw->src0_neg;"
                                        f" sdwa_src0_abs_ = sw->src0_abs;"
                                        f" sdwa_src1_sel_ = sw->src1_sel;"
                                        f" sdwa_src1_sext_ = sw->src1_sext;"
                                        f" sdwa_src1_neg_ = sw->src1_neg;"
                                        f" sdwa_src1_abs_ = sw->src1_abs;"
                                        + (
                                            f" sdwa_sdst_ = sw->sdst;"
                                            f" sdwa_sd_ = sw->sd;"
                                            if enc.enc_name.upper() == "ENC_VOPC"
                                            else f" sdwa_dst_sel_ = sw->dst_sel;"
                                            f" sdwa_dst_unused_ = sw->dst_unused;"
                                            f" sdwa_clamp_ = sw->clamp;"
                                        )
                                        + f"{_sdwa_s1_code}}}"
                                    )

                    # Implied literal fixup: FMAMK/FMAAK always carry an
                    # inline 32-bit literal even when the ISA spec omits the
                    # simm32 operand. Add a simm32_ member to hold it.
                    _FMAMK_FMAAK = frozenset(
                        {
                            "vector_fmamk",
                            "vector_fmaak",
                        }
                    )
                    _has_simm32 = any(op.name == "simm32" for op in inst.operands)
                    if (
                        _mem_sem
                        and _mem_sem.semantic_class in _FMAMK_FMAAK
                        and not _has_simm32
                        and _lit_info
                    ):
                        _lit_struct = _lit_info[0]
                        private_members.append(cgen.Statement("uint32_t simm32_"))
                        opnd_ctor_init.append("simm32_(0)")
                        init_list_parts.append("simm32_(0)")
                        init_list = ", ".join(init_list_parts)
                        ctor_body_parts.append(
                            f"simm32_ = reinterpret_cast<const "
                            f"{_lit_struct}*>(inst)->simm32;"
                        )

                    if _mem_sem and _mem_sem.semantic_class in _MEM_CLASSES:
                        ctor_body_parts.append("flags_ |= MEMORY_OP;")
                    # Control-flow flags drive BasicBlock splitting and CFG
                    # edge construction. Keep this metadata generated from the
                    # semantic classification so generic code does not have to
                    # know AMDGPU instruction names or opcode values.
                    if _mem_sem and _mem_sem.semantic_class == "branch":
                        ctor_body_parts.append("flags_ |= BRANCH;")
                    if _mem_sem and _mem_sem.semantic_class == "cbranch":
                        ctor_body_parts.append("flags_ |= COND_BRANCH;")
                    if _mem_sem and _mem_sem.semantic_class == "endpgm":
                        ctor_body_parts.append("flags_ |= PROGRAM_TERMINATOR;")
                    if _mem_sem and _mem_sem.semantic_class == "scalar_setpc":
                        ctor_body_parts.append("flags_ |= INDIRECT_BRANCH;")
                    if _mem_sem and _mem_sem.semantic_class in (
                        "scalar_swappc",
                        "scalar_call",
                    ):
                        ctor_body_parts.append("flags_ |= INDIRECT_CALL;")
                    # Conditional scalar moves leave the destination unchanged
                    # when their predicate is false, so liveness cannot treat
                    # them as unconditional kills.
                    if _mem_sem and _mem_sem.semantic_class in (
                        "scalar_cmov",
                        "scalar_cmovk",
                    ):
                        ctor_body_parts.append("flags_ |= PREDICATED_DEF;")

                    _waitcnt_names = {
                        "S_WAITCNT",
                        "S_WAIT_LOADCNT",
                        "S_WAIT_STORECNT",
                        "S_WAIT_EXPCNT",
                        "S_WAIT_DSCNT",
                        "S_WAIT_KMCNT",
                        "S_WAIT_SAMPLECNT",
                        "S_WAIT_BVHCNT",
                        "S_WAIT_LOADCNT_DSCNT",
                        "S_WAIT_STORECNT_DSCNT",
                        "S_WAIT_IDLE",
                        "S_WAIT_ALU",
                        "S_WAIT_EVENT",
                        "S_WAITCNT_VSCNT",
                        "S_WAITCNT_VMCNT",
                        "S_WAITCNT_LGKMCNT",
                        "S_WAITCNT_EXPCNT",
                        "S_WAITCNT_DEPCTR",
                    }
                    _barrier_names = {
                        "S_BARRIER",
                        "S_BARRIER_SIGNAL",
                        "S_BARRIER_WAIT",
                    }
                    if inst.name in _waitcnt_names:
                        ctor_body_parts.append("flags_ |= WAITCNT;")
                    if inst.name in _barrier_names:
                        ctor_body_parts.append("flags_ |= BARRIER;")

                    if inst.name.startswith("V_MFMA_") or inst.name.startswith(
                        "V_SMFMAC_"
                    ):
                        ctor_body_parts.append("flags_ |= MFMA;")

                    if inst.name in {
                        "V_ACCVGPR_WRITE_B32",
                        "V_ACCVGPR_READ_B32",
                        "V_ACCVGPR_MOV_B32",
                    }:
                        ctor_body_parts.append("flags_ |= ACCVGPR;")

                    # Per-instruction size overrides (e.g., VOP3PX2 128-bit
                    # instructions decoded under 64-bit VOP3P_MFMA).
                    _size_overrides = self.isa_spec.profile.inst_size_overrides
                    if inst.name in _size_overrides:
                        ctor_body_parts.append(f"size_ = {_size_overrides[inst.name]};")

                    class_ctor_impl_str = (
                        f"{inst.fmt_name}::"
                        f"{inst.fmt_name}(const MachineInst *inst) "
                        f": {init_list} "
                        f'{{{"".join(ctor_body_parts)}}}'
                    )
                    class_ctor_impl = cgen.Line(class_ctor_impl_str)
                    class_members.extend(public_members)
                    class_members.extend(private_members)
                    s = cgen.Struct(
                        f"{inst.fmt_name} : public {inst.fmt_true_enc_name}",
                        class_members,
                    )
                    # Generate execute_impl — non-static member method with
                    # the actual execute logic.  Called via make_exec_fn<>.
                    sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    if sem:
                        self._current_inst_fields = inst_field_names
                        self._current_enc = enc
                        body = self._gen_execute_body(inst, sem, enc.enc_name)
                        # VOP1/VOP2: prepend DPP preamble so the encoding
                        # base's apply_dpp() runs before the ALU logic.
                        _dpp_preamble = ""
                        if enc.enc_name.upper() in ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC"):
                            _src0_name = next(
                                (o.name for o in inst.operands if o.is_input), None
                            )
                            _src_inputs = [o.name for o in inst.operands if o.is_input]
                            _src1_name = (
                                _src_inputs[1] if len(_src_inputs) > 1 else None
                            )
                            _is_vopc = enc.enc_name.upper() == "ENC_VOPC"
                            _dpp_preamble = ""
                            if _is_vopc:
                                _dpp_preamble += (
                                    "  uint64_t dpp_old_vcc_ = wf.vcc();\n"
                                    "  uint64_t dpp_write_mask_ = ~0ULL;\n"
                                    "  if (inst_.src0 == amdgpu::SRC_DPP) {\n"
                                    "    dpp_write_mask_ = 0;\n"
                                    "    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n"
                                    "      uint32_t row = ln / 16;\n"
                                    "      uint32_t bank = (ln % 16) / 4;\n"
                                    "      if ((dpp_row_mask_ & (1u << row)) &&\n"
                                    "          (dpp_bank_mask_ & (1u << bank)))\n"
                                    "        dpp_write_mask_ |= (1ULL << ln);\n"
                                    "    }\n"
                                    "  }\n"
                                )
                            elif not _is_vopc:
                                _dpp_preamble += (
                                    "  uint32_t sdwa_old_dst_[64] = {};\n"
                                    "  if (sdwa_dst_sel_ != amdgpu::sdwa::DWORD ||\n"
                                    "      inst_.src0 == amdgpu::SRC_DPP) {\n"
                                    "    uint32_t vb = wf.vgpr_alloc().base;\n"
                                    "    uint64_t ex = wf.exec();\n"
                                    "    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln)\n"
                                    "      if (ex & (1ULL << ln))\n"
                                    "        sdwa_old_dst_[ln] = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n"
                                    "  }\n"
                                )
                            _dpp_preamble += (
                                "  if (inst_.src0 == amdgpu::SRC_DPP)\n"
                                "    amdgpu::dpp::apply_dpp(src_operands_[0], dpp_ctrl_,\n"
                                "        dpp_row_mask_, dpp_bank_mask_, dpp_bound_ctrl_,\n"
                                "        dpp_src0_, wf);\n"
                                "  if (inst_.src0 == amdgpu::SRC_SDWA) {\n"
                                "    auto &cu = wf.cu();\n"
                                "    uint32_t ws = wf.wf_size();\n"
                                "    if (sdwa_src0_sel_ != amdgpu::sdwa::DWORD) {\n"
                                "      uint32_t vb = wf.vgpr_alloc().base + src_operands_[0]->encoding_value_;\n"
                                "      uint32_t result[64];\n"
                                "      for (uint32_t i = 0; i < ws; ++i)\n"
                                "        result[i] = amdgpu::sdwa::sdwa_src_select(\n"
                                "            cu.read_vgpr(vb, i), sdwa_src0_sel_, sdwa_src0_sext_);\n"
                                "      if (sdwa_src0_abs_ || sdwa_src0_neg_) {\n"
                                "        for (uint32_t i = 0; i < ws; ++i) {\n"
                                "          float sv = std::bit_cast<float>(result[i]);\n"
                                "          if (sdwa_src0_abs_) sv = std::fabs(sv);\n"
                                "          if (sdwa_src0_neg_) sv = -sv;\n"
                                "          result[i] = std::bit_cast<uint32_t>(sv);\n"
                                "        }\n"
                                "      }\n"
                                "      dpp_src0_ = std::make_unique<DppOperand>(\n"
                                "          *src_operands_[0], result, static_cast<int>(ws));\n"
                                "      src_operands_[0] = dpp_src0_.get();\n"
                                "    }\n"
                                "    if (sdwa_src1_sel_ != amdgpu::sdwa::DWORD && num_src_ > 1) {\n"
                                "      uint32_t vb = wf.vgpr_alloc().base + src_operands_[1]->encoding_value_;\n"
                                "      uint32_t result1[64];\n"
                                "      for (uint32_t i = 0; i < ws; ++i)\n"
                                "        result1[i] = amdgpu::sdwa::sdwa_src_select(\n"
                                "            cu.read_vgpr(vb, i), sdwa_src1_sel_, sdwa_src1_sext_);\n"
                                "      if (sdwa_src1_abs_ || sdwa_src1_neg_) {\n"
                                "        for (uint32_t i = 0; i < ws; ++i) {\n"
                                "          float sv = std::bit_cast<float>(result1[i]);\n"
                                "          if (sdwa_src1_abs_) sv = std::fabs(sv);\n"
                                "          if (sdwa_src1_neg_) sv = -sv;\n"
                                "          result1[i] = std::bit_cast<uint32_t>(sv);\n"
                                "        }\n"
                                "      }\n"
                                "      dpp_src1_ = std::make_unique<DppOperand>(\n"
                                "          *src_operands_[1], result1, static_cast<int>(ws));\n"
                                "      src_operands_[1] = dpp_src1_.get();\n"
                                "    }\n"
                                "  }\n"
                                + (
                                    f"  if (dpp_src0_) {_src0_name}.set_delegate(dpp_src0_.get());\n"
                                    if _src0_name
                                    else ""
                                )
                                + (
                                    f"  if (dpp_src1_) {_src1_name}.set_delegate(dpp_src1_.get());\n"
                                    if _src1_name
                                    else ""
                                )
                            )
                        # SDWA postamble: apply dst_sel merge and float clamp after ALU.
                        _sdwa_postamble = ""
                        if enc.enc_name.upper() in ("ENC_VOP1", "ENC_VOP2"):
                            is_float_op = sem and sem.data_type in ("f16", "f32", "f64")
                            _sdwa_postamble = (
                                "  if (sdwa_dst_sel_ != amdgpu::sdwa::DWORD) {\n"
                                "    uint64_t ex = wf.exec();\n"
                                "    uint32_t vb = wf.vgpr_alloc().base;\n"
                                "    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n"
                                "      if (!(ex & (1ULL << ln))) continue;\n"
                                "      uint32_t dv = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n"
                                "      dv = amdgpu::sdwa::sdwa_dst_merge(dv, sdwa_old_dst_[ln], sdwa_dst_sel_, sdwa_dst_unused_);\n"
                                "      wf.cu().write_vgpr(vb + inst_.vdst, ln, dv);\n"
                                "    }\n"
                                "  }\n"
                            )
                            if is_float_op:
                                _sdwa_postamble += (
                                    "  if (sdwa_clamp_) {\n"
                                    "    uint64_t ex = wf.exec();\n"
                                    "    uint32_t vb = wf.vgpr_alloc().base;\n"
                                    "    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n"
                                    "      if (!(ex & (1ULL << ln))) continue;\n"
                                    "      uint32_t dv = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n"
                                    "      float fv = std::bit_cast<float>(dv);\n"
                                    "      fv = std::clamp(fv, 0.0f, 1.0f);\n"
                                    "      wf.cu().write_vgpr(vb + inst_.vdst, ln, std::bit_cast<uint32_t>(fv));\n"
                                    "    }\n"
                                    "  }\n"
                                )
                        _dpp_cleanup = ""
                        if enc.enc_name.upper() in ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC"):
                            if _is_vopc:
                                _dpp_cleanup += (
                                    "  if (inst_.src0 == amdgpu::SRC_DPP && dpp_write_mask_ != ~0ULL) {\n"
                                    "    uint64_t new_vcc = wf.vcc();\n"
                                    "    uint64_t merged = (new_vcc & dpp_write_mask_) | (dpp_old_vcc_ & ~dpp_write_mask_);\n"
                                    "    wf.set_vcc(merged);\n"
                                    "  }\n"
                                    "  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_sd_) {\n"
                                    "    uint64_t cmp_result = wf.vcc();\n"
                                    "    uint32_t sb = wf.sgpr_alloc().base;\n"
                                    "    wf.cu().write_sgpr(sb + sdwa_sdst_, static_cast<uint32_t>(cmp_result));\n"
                                    "    wf.cu().write_sgpr(sb + sdwa_sdst_ + 1, static_cast<uint32_t>(cmp_result >> 32));\n"
                                    "    wf.set_vcc(dpp_old_vcc_);\n"
                                    "  }\n"
                                )
                            else:
                                _dpp_cleanup += (
                                    "  if (inst_.src0 == amdgpu::SRC_DPP) {\n"
                                    "    uint64_t dpp_write_mask = 0;\n"
                                    "    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n"
                                    "      uint32_t row = ln / 16;\n"
                                    "      uint32_t bank = (ln % 16) / 4;\n"
                                    "      if ((dpp_row_mask_ & (1u << row)) &&\n"
                                    "          (dpp_bank_mask_ & (1u << bank)))\n"
                                    "        dpp_write_mask |= (1ULL << ln);\n"
                                    "    }\n"
                                    "    if (dpp_write_mask != ~0ULL) {\n"
                                    "      uint64_t ex = wf.exec();\n"
                                    "      uint32_t vb = wf.vgpr_alloc().base;\n"
                                    "      for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n"
                                    "        if ((ex & (1ULL << ln)) && !(dpp_write_mask & (1ULL << ln)))\n"
                                    "          wf.cu().write_vgpr(vb + inst_.vdst, ln,\n"
                                    "              sdwa_old_dst_[ln]);\n"
                                    "      }\n"
                                    "    }\n"
                                    "  }\n"
                                )
                            if _src0_name:
                                _dpp_cleanup += f"  {_src0_name}.clear_delegate();\n"
                            if _src1_name:
                                _dpp_cleanup += f"  {_src1_name}.clear_delegate();\n"
                        # Skip DPP/SDWA preamble and cleanup for unimplemented
                        # instructions whose body is ONLY a throw — the cleanup
                        # code after the throw would be unreachable. Only match
                        # pure-throw bodies, not bodies with conditional throws.
                        body_stripped = body.strip().rstrip(";").strip()
                        body_throws = (
                            body_stripped.startswith("(void)wf;")
                            and "throw util::UnimplementedInst" in body_stripped
                            and body_stripped.count("\n") <= 1
                        )
                        can_share = self._can_share_execute(inst.mnemonic)
                        if body_throws:
                            exec_impl = cgen.Line(
                                f"void {inst.fmt_name}::execute_impl"
                                f"(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}"
                            )
                        elif can_share:
                            enc_key = enc.enc_name.lower().replace("enc_", "")
                            tmpl_name = f"{inst.mnemonic}_{enc_key}"
                            exec_impl = cgen.Line(
                                f"void {inst.fmt_name}::execute_impl"
                                f"(amdgpu::Wavefront &wf) {{\n"
                                f"{_dpp_preamble}"
                                f"  amdgpu::execute_{tmpl_name}(*this, wf);\n"
                                f"{_dpp_cleanup}"
                                f"{_sdwa_postamble}}}"
                            )
                            body_key = (inst.mnemonic, enc.enc_name)
                            self._shared_execute_bodies[body_key] = (
                                inst,
                                sem,
                                body,
                                enc.enc_name,
                            )
                        else:
                            exec_impl = cgen.Line(
                                f"void {inst.fmt_name}::execute_impl"
                                f"(amdgpu::Wavefront &wf) {{\n"
                                f"{_dpp_preamble}"
                                f"{body}\n"
                                f"{_dpp_cleanup}"
                                f"{_sdwa_postamble}}}"
                            )
                    else:
                        exec_impl = cgen.Line(
                            f"void {inst.fmt_name}::execute_impl"
                            f"(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}"
                        )

                    inst_classes.append(s)
                    class_func_impls.append(class_ctor_impl)
                    if (
                        inst_sem
                        and inst_sem.semantic_class in ("branch", "cbranch")
                        and label_operand
                    ):
                        class_func_impls.append(
                            cgen.Line(
                                f"std::optional<int64_t> "
                                f"{inst.fmt_name}::branch_offset_bytes() const {{\n"
                                f"  // AMDGPU direct branch labels are signed "
                                f"instruction-count deltas.\n"
                                f"  return static_cast<int64_t>("
                                f"static_cast<int16_t>({label_operand}.encoding_value_)) * 4;\n"
                                f"}}"
                            )
                        )
                    class_func_impls.append(exec_impl)

                # Build include lists for .cpp files
                cpp_includes = [
                    (
                        f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/"
                        f"{enc.fmt_enc_name.lower()}.h",
                        False,
                    ),
                    ("util/except.h", False),
                ]
                _MEM_ENC_NAMES = frozenset(
                    {
                        "ENC_SMEM",
                        "ENC_FLAT",
                        "ENC_MUBUF",
                        "ENC_MTBUF",
                        "ENC_DS",
                        # RDNA4 renamed/new memory encodings
                        "ENC_VFLAT",
                        "ENC_VGLOBAL",
                        "ENC_VSCRATCH",
                        "ENC_VDS",
                        "ENC_VBUFFER",
                    }
                )
                is_mem_enc = enc.enc_name.upper() in _MEM_ENC_NAMES
                if is_mem_enc:
                    cpp_includes.extend(
                        [
                            (
                                f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/addr_calc.h",
                                False,
                            ),
                        ]
                    )
                    for cf_inc in self._cache_flags_includes():
                        cpp_includes.append((cf_inc, False))
                    cpp_includes.extend(
                        [
                            ("rocjitsu/vm/amdgpu/compute_unit.h", False),
                            ("rocjitsu/vm/amdgpu/mem_state.h", False),
                            ("cstring", True),
                            ("memory", True),
                        ]
                    )
                has_mfma = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == "mfma"
                    for i in all_insts
                )
                if has_mfma:
                    cpp_includes.append(
                        (
                            f"rocjitsu/isa/arch/amdgpu/"
                            f"{self.isa_spec.arch_name}/mfma_exec.h",
                            False,
                        )
                    )
                _VOP_ENC_NAMES = frozenset(
                    {
                        "ENC_VOP1",
                        "ENC_VOP2",
                        "ENC_VOP3",
                        "ENC_VOP3P",
                        "ENC_VOPC",
                    }
                )
                if enc.enc_name.upper() in _VOP_ENC_NAMES:
                    cpp_includes.append(
                        (
                            "rocjitsu/isa/arch/amdgpu/shared/transcendental.h",
                            False,
                        )
                    )
                if has_sem:
                    cpp_includes.extend(
                        [
                            ("rocjitsu/vm/amdgpu/wavefront.h", False),
                            ("util/data_types.h", False),
                            ("algorithm", True),
                            ("bit", True),
                            ("cmath", True),
                            ("limits", True),
                        ]
                    )
                # VOP1/VOP2 need DPP header for apply_dpp() in execute_impl.
                if enc.enc_name.upper() in ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC"):
                    cpp_includes.append(
                        ("rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h", False)
                    )
                has_saveexec = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == "scalar_saveexec"
                    for i in all_insts
                )
                if has_saveexec:
                    cpp_includes.extend(
                        [
                            ("util/log.h", False),
                            ("format", True),
                        ]
                    )
                has_getreg = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class in ("scalar_getreg", "scalar_setreg")
                    for i in all_insts
                )
                if has_getreg and not is_mem_enc:
                    cpp_includes.extend(
                        [
                            ("rocjitsu/vm/amdgpu/compute_unit.h", False),
                            ("util/log.h", False),
                        ]
                    )

                # Include the unified shared execute template header when
                # any instruction in this encoding delegates to a template.
                if self.shared_plan is not None:
                    has_shared = any(
                        self._can_share_execute(i.mnemonic)
                        for i in all_insts
                        if self.semantics and i.name in self.semantics.instructions
                    )
                    if has_shared:
                        cpp_includes.append(
                            (
                                "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h",
                                False,
                            )
                        )

                # Build per-ISA header includes.
                h_includes = [
                    (
                        f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/encodings.h",
                        False,
                    ),
                    (
                        f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/isa.h",
                        False,
                    ),
                    (
                        f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/operand.h",
                        False,
                    ),
                ]

                inst_def_file = CppFile(
                    f"{enc.fmt_enc_name.lower()}",
                    self.out_path,
                    True,
                    h_includes,
                    [],
                    inst_classes,
                    self.isa_spec.arch_name,
                    True,
                )
                inst_impl_file = CppFile(
                    f"{enc.fmt_enc_name.lower()}",
                    self.out_path,
                    False,
                    cpp_includes,
                    [],
                    class_func_impls,
                    self.isa_spec.arch_name,
                )
                # No local f16 helpers needed - using util::f16_to_f32 etc.
                # from data_types.h (included via cpp_includes when has_sem).

                if is_smem:
                    direct_field = self.isa_spec.profile.smem_direct_offset_field
                    if direct_field is None:
                        # CDNA model: soffset_en / imm / soffset three-field logic.
                        smem_body = (
                            "namespace {\n"
                            "Operand make_smem_offset(const Smem::OpEncoding *enc) {\n"
                            "  // SOFFSET_EN and IMM are independent: SOFFSET_EN gates the\n"
                            "  // SGPR field, IMM gates the 21-bit immediate field.\n"
                            "  // When both are set the hardware adds SGPR + immediate;\n"
                            "  // we show the SGPR as the operand and the immediate as\n"
                            "  // an offset modifier.\n"
                            "  if (enc->soffset_en)\n"
                            "    return Operand(32, OperandType::OPR_SMEM_OFFSET, "
                            "static_cast<int>(enc->soffset));\n"
                            "  if (enc->imm)\n"
                            "    return Operand(32, OperandType::OPR_SIMM32, "
                            "static_cast<int>(enc->offset));\n"
                            "  return Operand(32, OperandType::OPR_SIMM32, 0);\n"
                            "}\n"
                            "} // namespace"
                        )
                    else:
                        # RDNA model: direct offset field (no soffset_en/imm).
                        smem_body = (
                            "namespace {\n"
                            "Operand make_smem_offset(const Smem::OpEncoding *enc) {\n"
                            f"  return Operand(32, OperandType::OPR_SIMM32, "
                            f"static_cast<int>(enc->{direct_field}));\n"
                            "}\n"
                            "} // namespace"
                        )
                    smem_offset_helper = cgen.Line(smem_body)
                    class_func_impls.insert(0, smem_offset_helper)

                inst_def_file.gen_code()
                inst_impl_file.gen_code()

        # Generate the umbrella insts.h header that includes all per-
        # encoding instruction headers. Only primary (non-alt) encodings
        # that have instructions generate their own files; alt encoding
        # instructions are merged into their parent's file.
        import os

        arch = self.isa_spec.arch_name
        guard = f"ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_INSTS_H_"
        inc_base = f"rocjitsu/isa/arch/amdgpu/{arch}"
        insts_h_lines = [
            CppFile._prologue_comment(),
            f"#ifndef {guard}\n#define {guard}\n\n",
        ]
        for enc in self.isa_spec.inst_encodings:
            all_enc_insts = list(enc.insts)
            for child in child_encs.get(enc.enc_name, []):
                all_enc_insts.extend(child.insts)
            if all_enc_insts and not profile.is_alt_encoding(enc.enc_name):
                insts_h_lines.append(
                    f'#include "{inc_base}/{enc.fmt_enc_name.lower()}.h"\n'
                )
        insts_h_lines.append(f"\n#endif // {guard}\n")
        insts_h_path = os.path.join(self.out_path, arch, "insts.h")
        with open(insts_h_path, "w") as f:
            f.write("".join(insts_h_lines))

        # Shared execute templates are written by _run_multi after all ISAs
        # are processed, using the accumulated _shared_execute_bodies dict.
        # Individual ISA codegens just collect; they don't write.

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
        import os

        shared_dir = os.path.join(self.out_path, "shared")
        os.makedirs(shared_dir, exist_ok=True)

        for enc_name, (classes, impls, _, _) in sorted(shared_by_enc.items()):
            if not classes:
                continue
            guard = f"ROCJITSU_ISA_AMDGPU_SHARED_{enc_name.upper()}_H_"

            h_path = os.path.join(shared_dir, f"{enc_name}.h")
            with open(h_path, "w") as f:
                f.write(self.prologue())
                f.write(f"#ifndef {guard}\n#define {guard}\n\n")
                # No #include directives here — this file is included inside
                # a namespace block.  All required headers (wavefront.h,
                # except.h, data_types.h, <cmath>, etc.) must be included
                # by the per-ISA header BEFORE the namespace opens.\n

                # No namespace — this file is included inside per-ISA
                # namespace rocjitsu::<isa> { ... }.

                # Emit class definitions.
                for cls in classes:
                    out = re.sub(r"^struct\s", "class ", f"{cls}\n\n")
                    f.write(out)

                # Emit inline constructor + execute() bodies.
                for impl in impls:
                    f.write(f"inline {impl}\n\n")

                f.write(f"#endif // {guard}\n")

    def _write_shared_execute_templates(self) -> None:
        """Write shared/execute_<enc>.h with full template execute bodies.

        Each shared instruction gets a template function:
        ``template<typename Inst> inline void execute_<mnemonic>(Inst &inst, Wavefront &wf)``
        with the execute body modified to access operands through ``self.``.
        """
        import os
        import re as _re

        entries: list[tuple[str, str, str]] = []
        for (mnemonic, enc_name_key), (inst, sem, body, enc_name) in sorted(
            self._shared_execute_bodies.items()
        ):
            enc_key = enc_name.lower().replace("enc_", "")
            mnemonic = f"{mnemonic}_{enc_key}"
            prefixed_body = body
            for opnd in inst.operands:
                pattern = rf"(?<!\.)(?<!\w){_re.escape(opnd.name)}\."
                prefixed_body = _re.sub(pattern, f"inst.{opnd.name}.", prefixed_body)
            prefixed_body = _re.sub(
                r"(?<!\.)(?<!\w)inst_\.", "inst.inst_.", prefixed_body
            )
            prefixed_body = _re.sub(
                r"(?<!\.)(?<!\w)set_data\(", "inst.set_data(", prefixed_body
            )
            prefixed_body = _re.sub(
                r"(?<!\.)(?<!\w)size_(?!\w)", "inst.size()", prefixed_body
            )
            prefixed_body = _re.sub(
                r"(?<!\.)(?<!\w)mnemonic\(\)", "inst.mnemonic()", prefixed_body
            )
            prefixed_body = _re.sub(
                r"(?<!\.)(?<!\w)simm32_(?!\w)", "inst.simm32_", prefixed_body
            )
            prefixed_body = _re.sub(
                r"\s*\(void\)wf;\s*(?://[^\n]*)?\n?", "\n", prefixed_body
            )
            entries.append((mnemonic, prefixed_body, sem.semantic_class))

        shared_dir = os.path.join(self.out_path, "shared")
        os.makedirs(shared_dir, exist_ok=True)

        from amdisa.codegen.execute.simd_codegen import (
            simd_extra_includes,
            simd_probe_line,
        )

        guard = "ROCJITSU_ISA_AMDGPU_SHARED_EXECUTE_SHARED_H_"
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            '#include "rocjitsu/vm/amdgpu/wavefront.h"',
            '#include "rocjitsu/vm/amdgpu/compute_unit.h"',
            '#include "rocjitsu/vm/amdgpu/mem_state.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"',
            *simd_extra_includes(),
            '#include "util/data_types.h"',
            '#include "util/except.h"',
            '#include "util/log.h"',
            "#include <algorithm>",
            "#include <bit>",
            "#include <cmath>",
            "#include <functional>",
            "#include <limits>",
            "",
            "namespace rocjitsu {",
            "namespace amdgpu {",
            "",
        ]

        for mnemonic, prefixed_body, sem_class in entries:
            lines.append("template <typename Inst>")
            lines.append(
                f"inline void execute_{mnemonic}("
                f"[[maybe_unused]] Inst &inst, [[maybe_unused]] Wavefront &wf) {{"
            )
            probe = simd_probe_line(mnemonic)
            if probe is not None:
                lines.append(probe)
            lines.append(prefixed_body)
            lines.append("}")
            lines.append("")

        lines.append("} // namespace amdgpu")
        lines.append("} // namespace rocjitsu")
        lines.append("")
        lines.append(f"#endif // {guard}")
        lines.append("")

        filepath = os.path.join(shared_dir, "execute_shared.h")
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        import sys

        print(
            f"Generated shared/execute_shared.h with "
            f"{len(entries)} template functions",
            file=sys.stderr,
        )

    def gen_operand_types(self) -> None:
        """Generate operand type and OpSel enums."""
        code_lines = []
        opnd_type_enum = "enum class OperandType {"
        for opnd_type in self.isa_spec.operand_types:
            opnd_type_enum += opnd_type + ","
        opnd_type_enum += "};"
        code_lines.append(cgen.Line(opnd_type_enum))

        for opnd_sels in self.isa_spec.opnd_selectors:
            opnd_sel_name = "".join(
                x.capitalize() for x in opnd_sels.operand_type.split("_")[1:]
            )
            opnd_sel_enum = f"enum OpSel{opnd_sel_name} {{"
            seen_names: set[str] = set()
            for opnd_sel_val in opnd_sels.op_sel_vals:
                if opnd_sel_val[0] not in seen_names:
                    seen_names.add(opnd_sel_val[0])
                    opnd_sel_enum += f"{opnd_sel_val[0]} = {opnd_sel_val[1]},"
            opnd_sel_enum += "};"
            code_lines.append(cgen.Line(opnd_sel_enum))

        # Generate is_vgpr_operand_type() constexpr function.
        vgpr_types = [
            t for t in self.isa_spec.operand_types if "VGPR" in t or "ACCVGPR" in t
        ]
        if vgpr_types:
            fn = "[[nodiscard]] constexpr bool is_vgpr_operand_type(OperandType t) {"
            fn += " switch (t) {"
            for t in vgpr_types:
                fn += f" case OperandType::{t}:"
            fn += " return true;"
            fn += " default: return false; } }"
            code_lines.append(cgen.Line(fn))

        opnd_type_def_file = CppFile(
            "operand_types",
            self.out_path,
            True,
            [],
            [],
            code_lines,
            self.isa_spec.arch_name,
        )
        opnd_type_def_file.gen_code()

    @staticmethod
    def _reg_class_for_prefix(prefix: str) -> str | None:
        """Map an MRISA register-name prefix to an ISA register class."""
        match prefix.lower():
            case "s":
                return "RegClass::SGPR"
            case "v":
                return "RegClass::VGPR"
            case "acc":
                return "RegClass::ACC_VGPR"
            case _:
                return None

    def gen_operand(self) -> None:
        """Generate the ISA-specific Operand class with name resolution."""
        arch = self.isa_spec.arch_name

        switch_cases = []
        ref_switch_cases = []
        opnd_types_with_selectors = set()

        for opnd_sel in self.isa_spec.opnd_selectors:
            opnd_types_with_selectors.add(opnd_sel.operand_type)
            opsel_name = "OpSel" + "".join(
                x.capitalize() for x in opnd_sel.operand_type.split("_")[1:]
            )

            case_lines = []
            ref_case_lines = []
            for pattern in opnd_sel.name_patterns:
                if pattern.kind == OperandNamePattern.REG_RANGE:
                    case_lines.append(
                        f"if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && "
                        f"encoding_value_ <= {opsel_name}::{pattern.max_enum}) "
                        f'return reg_name("{pattern.prefix}", '
                        f"encoding_value_ - {opsel_name}::{pattern.min_enum}, size_bits_);"
                    )
                    reg_class = self._reg_class_for_prefix(pattern.prefix)
                    # Only register-file prefixes tracked by RegisterSet become
                    # register refs. Named special registers remain nullopt
                    # until a consumer needs special-register liveness.
                    if reg_class is not None:
                        ref_case_lines.append(
                            f"if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && "
                            f"encoding_value_ <= {opsel_name}::{pattern.max_enum}) "
                            f"return RegisterRef{{{reg_class}, static_cast<uint16_t>("
                            f"encoding_value_ - {opsel_name}::{pattern.min_enum}), reg_width}};"
                        )
                elif pattern.kind == OperandNamePattern.POS_INT:
                    case_lines.append(
                        f"if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && "
                        f"encoding_value_ <= {opsel_name}::{pattern.max_enum}) "
                        f"return std::to_string("
                        f"encoding_value_ - {opsel_name}::{pattern.min_enum});"
                    )
                elif pattern.kind == OperandNamePattern.NEG_INT:
                    case_lines.append(
                        f"if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && "
                        f"encoding_value_ <= {opsel_name}::{pattern.max_enum}) "
                        f"return std::to_string("
                        f"-(encoding_value_ - {opsel_name}::{pattern.min_enum} + 1));"
                    )
                elif pattern.kind == OperandNamePattern.FLOAT_CONST:
                    case_lines.append(
                        f"if (encoding_value_ == {opsel_name}::{pattern.enum_name}) "
                        f'return "{pattern.operand_name}";'
                    )
                elif pattern.kind == OperandNamePattern.NAMED:
                    case_lines.append(
                        f"if (encoding_value_ == {opsel_name}::{pattern.enum_name}) "
                        f'return "{pattern.operand_name}";'
                    )
                elif pattern.kind == OperandNamePattern.LITERAL:
                    case_lines.append(
                        f"if (encoding_value_ == {opsel_name}::{pattern.enum_name}) "
                        f'return "literal";'
                    )

            case_lines.append("break;")
            case_body = " ".join(case_lines)
            switch_cases.append(
                f"case OperandType::{opnd_sel.operand_type}: " f"{{ {case_body} }}"
            )
            ref_case_lines.append("break;")
            ref_case_body = " ".join(ref_case_lines)
            ref_switch_cases.append(
                f"case OperandType::{opnd_sel.operand_type}: " f"{{ {ref_case_body} }}"
            )

        no_sel_types = [
            t for t in self.isa_spec.operand_types if t not in opnd_types_with_selectors
        ]
        for t in no_sel_types:
            if t == "OPR_SIMM32":
                switch_cases.append(
                    f"case OperandType::{t}: "
                    f'return std::format("0x{{:x}}", encoding_value_);'
                )
            elif t == "OPR_WAITCNT":
                wc = self.isa_spec.profile.waitcnt_decode
                switch_cases.append(
                    f"case OperandType::{t}: {{\n"
                    f"  {wc}"
                    f'  return std::format("vmcnt({{}}) expcnt({{}}) lgkmcnt({{}})", '
                    f"vmcnt, expcnt, lgkmcnt);\n"
                    f"}}"
                )
            else:
                switch_cases.append(
                    f"case OperandType::{t}: return std::to_string(encoding_value_);"
                )

        switch_body = "\n".join(switch_cases)
        name_impl = (
            f"std::string Operand::name() const {{\n"
            f"switch (opr_type_) {{\n"
            f"{switch_body}\n"
            f"}}\n"
            f"return std::to_string(encoding_value_);\n"
            f"}}"
        )

        ref_switch_body = "\n".join(ref_switch_cases)
        ref_impl = (
            f"std::optional<RegisterRef> Operand::to_register_ref() const {{\n"
            f"// Liveness tracks operands as contiguous 32-bit register lanes.\n"
            f"const auto reg_width = static_cast<uint8_t>(size_bits_ > 32 ? size_bits_ / 32 : 1);\n"
            f"switch (opr_type_) {{\n"
            f"{ref_switch_body}\n"
            f"default:\n"
            f"  break;\n"
            f"}}\n"
            f"return std::nullopt;\n"
            f"}}"
        )

        class_def = [
            cgen.Line(
                "class Operand : public AmdgpuIsaOperand<Isa> {\n"
                "public:\n"
                "  Operand(int size_bits, OperandType opr_type, int encoding_value);\n"
                "  std::string name() const override;\n"
                "  std::optional<RegisterRef> to_register_ref() const override;\n"
                "  uint32_t read_scalar(const amdgpu::Wavefront &wf) const override;\n"
                "  uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n"
                "  void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const override;\n"
                "  void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const override;\n"
                "  uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n"
                "  void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const override;\n"
                "  uint64_t read_scalar64(const amdgpu::Wavefront &wf) const override;\n"
                "  void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const override;\n"
                "};"
            )
        ]

        class_impl = [
            cgen.Line(
                "Operand::Operand(int size_bits, OperandType opr_type, int encoding_value)\n"
                "    : AmdgpuIsaOperand<Isa>(size_bits, opr_type, encoding_value) {\n"
                "  is_vgpr_ = is_vgpr_operand_type(opr_type);\n"
                "}"
            ),
            cgen.Line(name_impl),
            cgen.Line(ref_impl),
        ]

        reg_name_helper = cgen.Line(
            "namespace {\n"
            "std::string reg_name(const char *prefix, int reg_num, int size_bits) {\n"
            "  int count = size_bits / 32;\n"
            "  if (count <= 1)\n"
            "    return prefix + std::to_string(reg_num);\n"
            '  return std::string(prefix) + "[" + std::to_string(reg_num) + ":" +\n'
            '         std::to_string(reg_num + count - 1) + "]";\n'
            "}\n"
            "} // namespace"
        )
        class_impl.insert(0, reg_name_helper)

        # Operand value resolution (consolidated from operand_resolve.cpp)
        # Build ISA-dependent helper function bodies: only reference OperandType
        # values that actually exist in this ISA's generated enum.
        _opr = set(self.isa_spec.operand_types)
        _vgpr_only_parts = ["t == OperandType::OPR_VGPR"]
        for _opt in (
            "OPR_VGPR_OR_ACCVGPR",
            "OPR_VGPR_OR_LDS",
            "OPR_SRC_VGPR",
            "OPR_ACCVGPR",
            "OPR_SRC_ACCVGPR",
            "OPR_SRC_VGPR_OR_ACCVGPR",
        ):
            if _opt in _opr:
                _vgpr_only_parts.append(f"t == OperandType::{_opt}")
        _is_vgpr_only_body = (
            "bool is_vgpr_only_type(OperandType t) {\n"
            "  return " + " ||\n         ".join(_vgpr_only_parts) + ";\n"
            "}"
        )
        _imm_parts = ["t == OperandType::OPR_SIMM16", "t == OperandType::OPR_SIMM32"]
        for _opt in ("OPR_SIMM4", "OPR_SIMM8"):
            if _opt in _opr:
                _imm_parts.append(f"t == OperandType::{_opt}")
        for _opt in ("OPR_LABEL", "OPR_WAITCNT"):
            if _opt in _opr:
                _imm_parts.append(f"t == OperandType::{_opt}")
        _is_immediate_body = (
            "bool is_immediate_type(OperandType t) {\n"
            "  return " + " ||\n         ".join(_imm_parts) + ";\n"
            "}"
        )
        # AccVGPR offset within the unified VGPR block.  On CDNA3/4, AccVGPRs
        # live at indices 256-511 within each wavefront's VGPR allocation.  The
        # encoding values differ between destination (512-767) and source
        # (768-1023), but both map to the same physical offset range.
        _ACC_OFFSET = 256

        _vgpr_index_lines = ["uint32_t vgpr_index(OperandType opr_type, int ev) {"]
        if "OPR_VGPR_OR_ACCVGPR" in _opr:
            _vgpr_index_lines.append(
                "  if (opr_type == OperandType::OPR_VGPR || "
                "opr_type == OperandType::OPR_VGPR_OR_ACCVGPR) {\n"
                f"    if (ev >= OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN)\n"
                f"      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN);\n"
                "    return static_cast<uint32_t>(ev);\n"
                "  }"
            )
        else:
            _vgpr_index_lines.append(
                "  if (opr_type == OperandType::OPR_VGPR)\n"
                "    return static_cast<uint32_t>(ev);"
            )
        if "OPR_ACCVGPR" in _opr:
            _vgpr_index_lines.append(
                "  if (opr_type == OperandType::OPR_ACCVGPR) {\n"
                f"    if (ev >= OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN)\n"
                f"      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelAccvgpr::OPR_ACCVGPR_ACC_MIN);\n"
                f"    return {_ACC_OFFSET} + static_cast<uint32_t>(ev);\n"
                "  }"
            )
        if "OPR_SRC_ACCVGPR" in _opr:
            # AccVGPR source: maps to the AccVGPR bank at +256 offset.
            # v_accvgpr_read src0=256 (acc0) → physical index 256.
            # OpSel range 768+ also maps to +256 offset.
            _vgpr_index_lines.append(
                "  if (opr_type == OperandType::OPR_SRC_ACCVGPR) {\n"
                f"    if (ev >= OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN)\n"
                f"      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN);\n"
                f"    if (ev >= 256)\n"
                f"      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - 256);\n"
                f"    return {_ACC_OFFSET} + static_cast<uint32_t>(ev);\n"
                "  }"
            )
        if "OPR_SRC_VGPR_OR_ACCVGPR" in _opr:
            _vgpr_index_lines.append(
                "  if (opr_type == OperandType::OPR_SRC_VGPR_OR_ACCVGPR) {\n"
                "    if (ev >= OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN)\n"
                f"      return {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN);\n"
                "    if (ev >= OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN)\n"
                "      return static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN);\n"
                "    return static_cast<uint32_t>(ev);\n"
                "  }"
            )
        _vgpr_index_lines.append("  return static_cast<uint32_t>(ev - 256);")
        _vgpr_index_lines.append("}")
        _vgpr_index_body = "\n".join(_vgpr_index_lines)

        # Single source of truth for "does this operand resolve to per-lane VGPR
        # storage, and if so what's the offset within the wavefront's VGPR
        # allocation?". Used by read_lane/read_lane64/write_lane/write_lane64
        # and by every SIMD method below. Adding a new VGPR-bearing operand
        # type means editing this helper, not chasing dispatch through ~8
        # callsites.
        _resolved_vgpr_offset_lines = [
            "std::optional<uint32_t> Isa::resolved_vgpr_offset(OperandType opr_type, int ev) {",
            "  if (is_vgpr_only_type(opr_type))",
            "    return vgpr_index(opr_type, ev);",
            "  if (is_immediate_type(opr_type))",
            "    return std::nullopt;",
            "  if (ev >= 256 && ev <= 511)",
            "    return static_cast<uint32_t>(ev - 256);",
        ]
        if "OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST" in _opr:
            _resolved_vgpr_offset_lines.append(
                "  if (opr_type == OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST &&\n"
                "      ev >= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN &&\n"
                "      ev <= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MAX) {\n"
                f"    return {_ACC_OFFSET} + static_cast<uint32_t>(\n"
                "        ev - OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN);\n"
                "  }"
            )
        _resolved_vgpr_offset_lines.append("  return std::nullopt;")
        _resolved_vgpr_offset_lines.append("}")
        _resolved_vgpr_offset_body = "\n".join(_resolved_vgpr_offset_lines)

        _read_lane_body = (
            "uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {\n"
            "  if (delegate()) return delegate()->read_lane(wf, lane);\n"
            "  int ev = encoding_value_;\n"
            "  if (auto off = Isa::resolved_vgpr_offset(opr_type_, ev))\n"
            "    return wf.cu().read_vgpr(wf.vgpr_alloc().base + *off, lane);\n"
            "  if (is_immediate_type(opr_type_))\n"
            "    return static_cast<uint32_t>(ev);\n"
            "  return resolve_src_scalar(wf, ev);\n"
            "}"
        )

        _read_lane64_body = (
            "uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const {\n"
            "  if (delegate()) return delegate()->read_lane64(wf, lane);\n"
            "  int ev = encoding_value_;\n"
            "  if (auto off = Isa::resolved_vgpr_offset(opr_type_, ev)) {\n"
            "    uint32_t idx = wf.vgpr_alloc().base + *off;\n"
            "    uint32_t lo = wf.cu().read_vgpr(idx, lane);\n"
            "    uint32_t hi = wf.cu().read_vgpr(idx + 1, lane);\n"
            "    return static_cast<uint64_t>(hi) << 32 | lo;\n"
            "  }\n"
            "  if (is_immediate_type(opr_type_))\n"
            "    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ev)));\n"
            "  return resolve_src_scalar64(wf, ev);\n"
            "}"
        )

        resolve_code = cgen.Line(
            "namespace {\n"
            "\n"
            "uint32_t resolve_src_scalar(const amdgpu::Wavefront &wf, int ev) {\n"
            "  if (ev <= 105)\n"
            "    return wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n"
            "  if (ev == 106)\n"
            "    return static_cast<uint32_t>(wf.vcc());\n"
            "  if (ev == 107)\n"
            "    return static_cast<uint32_t>(wf.vcc() >> 32);\n"
            "  if (ev == 124)\n"
            "    return wf.m0();\n"
            "  if (ev == 126)\n"
            "    return static_cast<uint32_t>(wf.exec());\n"
            "  if (ev == 127)\n"
            "    return static_cast<uint32_t>(wf.exec() >> 32);\n"
            "  if (ev >= 128 && ev <= 192)\n"
            "    return static_cast<uint32_t>(ev - 128);\n"
            "  if (ev >= 193 && ev <= 208)\n"
            "    return static_cast<uint32_t>(static_cast<int32_t>(-(ev - 192)));\n"
            "  if (ev == 240)\n"
            "    return 0x3F000000u; // 0.5f\n"
            "  if (ev == 241)\n"
            "    return 0xBF000000u; // -0.5f\n"
            "  if (ev == 242)\n"
            "    return 0x3F800000u; // 1.0f\n"
            "  if (ev == 243)\n"
            "    return 0xBF800000u; // -1.0f\n"
            "  if (ev == 244)\n"
            "    return 0x40000000u; // 2.0f\n"
            "  if (ev == 245)\n"
            "    return 0xC0000000u; // -2.0f\n"
            "  if (ev == 246)\n"
            "    return 0x40800000u; // 4.0f\n"
            "  if (ev == 247)\n"
            "    return 0xC0800000u; // -4.0f\n"
            "  if (ev == 248)\n"
            "    return 0x3E22F983u; // 1/(2*pi)\n"
            "  if (ev == 235)\n"
            "    return static_cast<uint32_t>(wf.shared_aperture_base() >> 32); // SRC_SHARED_BASE\n"
            "  if (ev == 236)\n"
            "    return static_cast<uint32_t>(wf.shared_aperture_limit() >> 32); // SRC_SHARED_LIMIT\n"
            "  if (ev == 237)\n"
            "    return static_cast<uint32_t>(wf.private_aperture_base() >> 32); // SRC_PRIVATE_BASE\n"
            "  if (ev == 238)\n"
            "    return static_cast<uint32_t>(wf.private_aperture_limit() >> 32); // SRC_PRIVATE_LIMIT\n"
            "  if (ev == 249)\n"
            "    return 0u; // SRC_POPS_EXITING_WAVE_ID (not used in compute)\n"
            "  if (ev == 250)\n"
            "    return 0u; // NULL\n"
            "  if (ev == 251)\n"
            "    return wf.vcc() == 0 ? 1u : 0u; // VCCZ\n"
            "  if (ev == 252)\n"
            "    return wf.exec() == 0 ? 1u : 0u; // EXECZ\n"
            "  if (ev == 253)\n"
            "    return wf.read_scc() ? 1u : 0u; // SCC\n"
            '  throw std::logic_error("Unsupported encoding value for scalar read: " + std::to_string(ev));\n'
            "}\n"
            "\n"
            "// Must stay in sync with resolve_src_scalar above — returns true for\n"
            "// exactly the encoding values that resolve_src_scalar handles without\n"
            "// throwing. Used by Isa::simd_capable_value() to keep the SIMD fast\n"
            "// path off operands whose scalar broadcast would throw at runtime.\n"
            "bool can_resolve_src_scalar(int ev) {\n"
            "  return (ev >= 0 && ev <= 107) || ev == 124 || ev == 126 || ev == 127 ||\n"
            "         (ev >= 128 && ev <= 208) || (ev >= 240 && ev <= 253);\n"
            "}\n"
            "\n"
            "uint64_t resolve_src_scalar64(const amdgpu::Wavefront &wf, int ev) {\n"
            "  if (ev <= 105) {\n"
            "    uint32_t lo = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n"
            "    uint32_t hi = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1));\n"
            "    return static_cast<uint64_t>(hi) << 32 | lo;\n"
            "  }\n"
            "  if (ev == 106)\n"
            "    return wf.vcc();\n"
            "  if (ev == 126)\n"
            "    return wf.exec();\n"
            "  if (ev >= 128 && ev <= 192)\n"
            "    return static_cast<uint64_t>(ev - 128);\n"
            "  if (ev >= 193 && ev <= 208)\n"
            "    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(-(ev - 192))));\n"
            "  if (ev == 240)\n"
            "    return 0x3FE0000000000000ULL; // 0.5\n"
            "  if (ev == 241)\n"
            "    return 0xBFE0000000000000ULL; // -0.5\n"
            "  if (ev == 242)\n"
            "    return 0x3FF0000000000000ULL; // 1.0\n"
            "  if (ev == 243)\n"
            "    return 0xBFF0000000000000ULL; // -1.0\n"
            "  if (ev == 244)\n"
            "    return 0x4000000000000000ULL; // 2.0\n"
            "  if (ev == 245)\n"
            "    return 0xC000000000000000ULL; // -2.0\n"
            "  if (ev == 246)\n"
            "    return 0x4010000000000000ULL; // 4.0\n"
            "  if (ev == 247)\n"
            "    return 0xC010000000000000ULL; // -4.0\n"
            "  if (ev == 248)\n"
            "    return 0x3FC45F306DC9C883ULL; // 1/(2*pi)\n"
            "  if (ev == 235)\n"
            "    return wf.shared_aperture_base(); // SRC_SHARED_BASE\n"
            "  if (ev == 236)\n"
            "    return wf.shared_aperture_limit(); // SRC_SHARED_LIMIT\n"
            "  if (ev == 237)\n"
            "    return wf.private_aperture_base(); // SRC_PRIVATE_BASE\n"
            "  if (ev == 238)\n"
            "    return wf.private_aperture_limit(); // SRC_PRIVATE_LIMIT\n"
            '  throw std::logic_error("Unsupported encoding value for scalar64 read: " + std::to_string(ev));\n'
            "}\n"
            "\n"
            "void resolve_dst_write(amdgpu::Wavefront &wf, int ev, uint32_t val) {\n"
            "  if (ev <= 105) {\n"
            "    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), val);\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 106) {\n"
            "    wf.set_vcc((wf.vcc() & 0xFFFFFFFF00000000ULL) | val);\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 107) {\n"
            "    wf.set_vcc((wf.vcc() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 124) {\n"
            "    wf.set_m0(val);\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 126) {\n"
            "    wf.set_exec((wf.exec() & 0xFFFFFFFF00000000ULL) | val);\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 127) {\n"
            "    wf.set_exec((wf.exec() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));\n"
            "    return;\n"
            "  }\n"
            '  throw std::logic_error("Unsupported encoding value for scalar write: " + std::to_string(ev));\n'
            "}\n"
            "\n"
            "void resolve_dst_write64(amdgpu::Wavefront &wf, int ev, uint64_t val) {\n"
            "  if (ev <= 105) {\n"
            "    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), static_cast<uint32_t>(val));\n"
            "    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1), static_cast<uint32_t>(val >> 32));\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 106) {\n"
            "    wf.set_vcc(val);\n"
            "    return;\n"
            "  }\n"
            "  if (ev == 126) {\n"
            "    wf.set_exec(val);\n"
            "    return;\n"
            "  }\n"
            '  throw std::logic_error("Unsupported encoding value for scalar64 write: " + std::to_string(ev));\n'
            "}\n"
            "\n"
            + _is_vgpr_only_body
            + "\n\n"
            + _is_immediate_body
            + "\n\n"
            + _vgpr_index_body
            + "\n\n"
            + "\n"
            "} // namespace\n"
            "\n"
            "// Isa::-scoped SIMD traits — see rocjitsu/isa/isa_operand_simd_inl.h\n"
            "// for the templated callers in AmdgpuIsaOperand<Isa>.\n"
            + _resolved_vgpr_offset_body
            + "\n\n"
            "bool Isa::simd_capable_value(OperandType opr_type, int ev) {\n"
            "  return resolved_vgpr_offset(opr_type, ev).has_value() ||\n"
            "         is_immediate_type(opr_type) || can_resolve_src_scalar(ev);\n"
            "}\n"
            "\n"
            "uint32_t Isa::simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type,\n"
            "                                   int ev) {\n"
            "  return is_immediate_type(opr_type) ? static_cast<uint32_t>(ev)\n"
            "                                     : resolve_src_scalar(wf, ev);\n"
            "}\n"
            "\n"
            "uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {\n"
            "  if (delegate()) return delegate()->read_scalar(wf);\n"
            "  if (is_immediate_type(opr_type_))\n"
            "    return static_cast<uint32_t>(encoding_value_);\n"
            "  return resolve_src_scalar(wf, encoding_value_);\n"
            "}\n"
            "\n" + _read_lane_body + "\n\n"
            "void Operand::write_scalar(amdgpu::Wavefront &wf, uint32_t val) const {\n"
            "  resolve_dst_write(wf, encoding_value_, val);\n"
            "}\n"
            "\n"
            "void Operand::write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const {\n"
            "  if (auto off = Isa::resolved_vgpr_offset(opr_type_, encoding_value_)) {\n"
            "    wf.cu().write_vgpr(wf.vgpr_alloc().base + *off, lane, val);\n"
            "    return;\n"
            "  }\n"
            '  throw std::logic_error("write_lane called on non-VGPR operand type");\n'
            "}\n"
            "\n" + _read_lane64_body + "\n\n"
            "void Operand::write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const {\n"
            "  if (auto off = Isa::resolved_vgpr_offset(opr_type_, encoding_value_)) {\n"
            "    uint32_t idx = wf.vgpr_alloc().base + *off;\n"
            "    wf.cu().write_vgpr(idx, lane, static_cast<uint32_t>(val));\n"
            "    wf.cu().write_vgpr(idx + 1, lane, static_cast<uint32_t>(val >> 32));\n"
            "    return;\n"
            "  }\n"
            '  throw std::logic_error("write_lane64 called on non-VGPR operand type");\n'
            "}\n"
            "\n"
            "uint64_t Operand::read_scalar64(const amdgpu::Wavefront &wf) const {\n"
            "  if (is_immediate_type(opr_type_))\n"
            "    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(encoding_value_)));\n"
            "  return resolve_src_scalar64(wf, encoding_value_);\n"
            "}\n"
            "\n"
            "void Operand::write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const {\n"
            "  resolve_dst_write64(wf, encoding_value_, val);\n"
            "}"
        )
        class_impl.append(resolve_code)

        header_file = CppFile(
            "operand",
            self.out_path,
            True,
            [
                (f"rocjitsu/isa/arch/amdgpu/{arch}/isa.h", False),
                (f"rocjitsu/isa/arch/amdgpu/{arch}/operand_types.h", False),
                ("rocjitsu/isa/operand.h", False),
                ("string", True),
            ],
            [],
            class_def,
            arch,
        )
        source_file = CppFile(
            "operand",
            self.out_path,
            False,
            [
                (f"rocjitsu/isa/arch/amdgpu/{arch}/operand.h", False),
                ("rocjitsu/isa/isa_operand_simd_inl.h", False),
                ("rocjitsu/vm/amdgpu/compute_unit.h", False),
                ("rocjitsu/vm/amdgpu/wavefront.h", False),
                ("format", True),
                ("optional", True),
                ("stdexcept", True),
                ("string", True),
            ],
            [],
            class_impl,
            arch,
        )
        header_file.gen_code()
        source_file.gen_code()

    def gen_decoder(self) -> None:
        """Generate decoder lookup tables and decode functions."""
        class_def = []
        class_impl = []
        class_members = [
            cgen.Line("public:"),
            cgen.FunctionDeclaration(
                cgen.Value("static std::unique_ptr<Instruction>", "decode"),
                [cgen.Value("const MachineInst *", "opcode")],
            ),
            cgen.Line("private:"),
            cgen.Statement(
                "using DecodeFunc = std::unique_ptr<Instruction>(*)(const MachineInst *)"
            ),
            cgen.FunctionDeclaration(
                cgen.Value("static std::unique_ptr<Instruction>", "decodeInvalid"),
                [cgen.Value("const MachineInst *", "opcode")],
            ),
        ]
        decode_table_funcs = [
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("std::unique_ptr<Instruction>", "Decoder::decode"),
                    [cgen.Value("const MachineInst *", "opcode")],
                ),
                cgen.Block(
                    [
                        cgen.Statement(
                            "Sop1MachineInst op = std::bit_cast<decltype(op)>(*opcode)"
                        ),
                        cgen.Statement(
                            "return primary_decode_table[op.encoding](opcode)"
                        ),
                    ]
                ),
            ),
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value(
                        "std::unique_ptr<Instruction>",
                        "Decoder::decodeInvalid",
                    ),
                    [cgen.Value("const MachineInst *", "opcode")],
                ),
                cgen.Block(
                    [
                        cgen.Statement(
                            'throw util::InvalidInst(std::format("{:X}", *opcode))'
                        ),
                        cgen.Statement("return nullptr"),
                    ]
                ),
            ),
        ]
        decode_tables = [
            cgen.Statement(
                f"static const std::array<DecodeFunc, {pow(2, self.isa_spec.profile.max_enc_bits)}> primary_decode_table"
            )
        ]
        decode_table_entries = []
        sub_decode_table_entries = []
        decode_funcs_found = set()
        for dte in self.isa_spec.primary_decode_table:
            if dte is not None:
                decode_table_entries.append(f"&Decoder::{dte.decode_func},")
                if dte.decode_func not in decode_funcs_found:
                    decode_funcs_found.add(dte.decode_func)
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value(
                            "std::unique_ptr<Instruction>",
                            f"Decoder::{dte.decode_func}",
                        ),
                        [cgen.Value("const MachineInst *", "opcode")],
                    )
                    sub_decode_func_decls = []
                    if dte.is_primary:
                        decode_table_funcs.append(
                            cgen.FunctionBody(
                                func_decl,
                                cgen.Block(
                                    [
                                        cgen.Statement(
                                            f"return std::make_unique<{dte.inst_name}>(opcode)"
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
                                            f"{dte.enc.fmt_enc_name}::OpEncoding",
                                            "op = *reinterpret_cast<const decltype(op) *>(opcode)",
                                        ),
                                        cgen.Statement(
                                            f"return {dte.sub_decode_table}[op.op](opcode)"
                                        ),
                                    ]
                                ),
                            )
                        )
                        decode_tables.append(
                            cgen.Statement(
                                f"static const std::array<DecodeFunc, {len(dte.sub_decode_funcs)}> {dte.sub_decode_table}"
                            )
                        )
                        sub_decode_table_entries.append(
                            cgen.Line(
                                f"const std::array<Decoder::DecodeFunc, {len(dte.sub_decode_funcs)}> Decoder::{dte.sub_decode_table} = {{"
                            )
                        )
                        sub_decode_table_entry_str = []
                        for fn in dte.sub_decode_funcs:
                            if fn != "decodeInvalid":
                                sub_decode_func_decls.append(
                                    cgen.FunctionDeclaration(
                                        cgen.Value(
                                            "static std::unique_ptr<Instruction>",
                                            fn,
                                        ),
                                        [
                                            cgen.Value(
                                                "const MachineInst *",
                                                "opcode",
                                            )
                                        ],
                                    )
                                )
                                decode_table_funcs.append(
                                    cgen.FunctionBody(
                                        cgen.FunctionDeclaration(
                                            cgen.Value(
                                                "std::unique_ptr<Instruction>",
                                                f"Decoder::{fn}",
                                            ),
                                            [
                                                cgen.Value(
                                                    "const MachineInst *",
                                                    "opcode",
                                                )
                                            ],
                                        ),
                                        cgen.Block(
                                            [
                                                cgen.Statement(
                                                    f'return std::make_unique<{fn.removeprefix("decode")}>(opcode)'
                                                )
                                            ]
                                        ),
                                    )
                                )
                            sub_decode_table_entry_str.append(f"&Decoder::{fn},")
                        sub_decode_table_entries.append(
                            cgen.Line("".join(sub_decode_table_entry_str))
                        )
                        sub_decode_table_entries.append(cgen.Line("};"))
                    class_members.append(
                        cgen.FunctionDeclaration(
                            cgen.Value(
                                "static std::unique_ptr<Instruction>",
                                f"{dte.decode_func}",
                            ),
                            [cgen.Value("const MachineInst *", "opcode")],
                        )
                    )
                    class_members.extend(sub_decode_func_decls)
            else:
                decode_table_entries.append("&Decoder::decodeInvalid,")
        decode_table_entries = "".join(decode_table_entries)
        class_members.extend(decode_tables)
        class_def.append(cgen.Struct("Decoder", class_members))
        class_impl.extend(decode_table_funcs)
        class_impl.append(
            cgen.Line(
                f"const std::array<Decoder::DecodeFunc, {pow(2, self.isa_spec.profile.max_enc_bits)}> Decoder::primary_decode_table = {{"
            )
        )
        class_impl.append(cgen.Line(decode_table_entries))
        class_impl.append(cgen.Line("};"))
        class_impl.extend(sub_decode_table_entries)
        class_def_file = CppFile(
            "decoder",
            self.out_path,
            True,
            [
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/machine_insts.h",
                    False,
                ),
                ("array", True),
                ("memory", True),
            ],
            ["Instruction"],
            class_def,
            self.isa_spec.arch_name,
            True,
        )
        class_impl_file = CppFile(
            "decoder",
            self.out_path,
            False,
            [
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/decoder.h",
                    False,
                ),
                ("util/except.h", False),
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/insts.h",
                    False,
                ),
                ("bit", True),
                ("format", True),
            ],
            [],
            class_impl,
            self.isa_spec.arch_name,
        )
        class_def_file.gen_code()
        class_impl_file.gen_code()

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
            op_field = next((f for f in enc.ucode_fields if f.name == "op"), None)
            enc_field = next(
                (f for f in enc.ucode_fields if f.name == "encoding"), None
            )
            ptrs = enc.primary_dt_ptrs
            if not op_field or not enc_field or not ptrs:
                continue
            enc_val = ptrs[0]
            for inst in all_test_insts:
                word = (enc_val << enc_field.bit_offset) | (
                    inst.opcode << op_field.bit_offset
                )
                w0 = word & 0xFFFFFFFF
                w1 = (word >> 32) & 0xFFFFFFFF
                entries.append(
                    f'  {{"{inst.mnemonic}", {{0x{w0:08X}U, 0x{w1:08X}U}}}},'
                )

        arch = self.isa_spec.arch_name
        ns = arch
        guard = f"ROCJITSU_ISA_AMDGPU_{arch.upper()}_TEST_ENCODINGS_H_"
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <array>",
            "#include <cstdint>",
            "#include <string_view>",
            "",
            f"namespace rocjitsu::{ns}::test_data {{",
            "",
            "struct TestEncoding {",
            "  std::string_view mnemonic;",
            "  std::array<uint32_t, 2> words;",
            "};",
            "",
            f"inline constexpr TestEncoding ENCODINGS[] = {{",
        ]
        lines.extend(entries)
        lines.append("};")
        lines.append("")
        lines.append(f"inline constexpr size_t NUM_ENCODINGS = {len(entries)};")
        lines.append("")
        lines.append(f"}} // namespace rocjitsu::{ns}::test_data")
        lines.append("")
        lines.append(f"#endif // {guard}")
        lines.append("")

        import os

        out_path = os.path.join(
            self.out_path, self.isa_spec.arch_name, "test_encodings.h"
        )
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    def gen_isa_types(self) -> None:
        """Generate an ISA struct wrapping type definitions."""
        isa_typedefs = [
            cgen.Statement("using Decoder = Decoder"),
            cgen.Statement("using OperandType = OperandType"),
        ]
        isa_struct = [cgen.Struct("Isa", isa_typedefs)]
        isa_struct_file = CppFile(
            "isa",
            self.out_path,
            True,
            [
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/decoder.h",
                    False,
                ),
                (
                    f"rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/operand_types.h",
                    False,
                ),
            ],
            [],
            isa_struct,
            self.isa_spec.arch_name,
        )
        isa_struct_file.gen_code()
