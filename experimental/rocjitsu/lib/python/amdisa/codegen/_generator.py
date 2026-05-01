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






class _SemanticEmitter:
    """Entry point for execute() body generation.

    Phase 0 introduces this class as a named abstraction; Phase B.9 completes
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
        shared_plan: 'SharedInstructionPlan | None' = None,
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
        baseline = self._shared_baseline() if self.config.use_shared else {}
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
                        cgen.Statement(
                            f'using {struct_name} = amdgpu::{struct_name}'
                        )
                    )
                    continue

            s = cgen.Struct(
                struct_name,
                [
                    cgen.Value('uint32_t', f'{x.name} : {x.bit_cnt}')
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
                cgen.Line('public:'),
                cgen.FunctionDeclaration(
                    cgen.Value('', f'{inst_enc.fmt_enc_name}'),
                    [
                        cgen.Value('std::string_view', 'mnemonic'),
                        cgen.Value(
                            f'const {inst_enc.fmt_enc_name}MachineInst',
                            '*inst',
                        ),
                        cgen.Value('ExecuteFn', 'exec_fn'),
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
            default_cond = dict(inst_enc.enc_conds).get(
                'default_encoding', 'true'
            )
            has_real_default_check = (
                inst_enc.bit_cnt < 64 and default_cond != 'false'
            )

            if has_real_default_check and inst_enc.has_implied_literal_ops:
                size_condition = (
                    '!default_encoding() || hasImpliedLiteral()'
                )
            elif has_real_default_check:
                size_condition = '!default_encoding()'
            elif inst_enc.has_implied_literal_ops:
                size_condition = 'hasImpliedLiteral()'
            else:
                size_condition = None

            profile = self.isa_spec.profile
            rule = profile.mnemonic_rule(inst_enc.enc_name)
            if rule.use_flat_mnemonic:
                mnemonic_expr = 'flat_mnemonic(mnemonic, inst->seg)'
            else:
                # Suffix is pre-baked into the literal by the instruction
                # constructor, so the encoding base just passes through.
                mnemonic_expr = 'mnemonic'

            modifier_lines = ''
            enc_field_names = {f.name for f in inst_enc.ucode_fields}
            for mod in profile.encoding_modifiers(inst_enc.enc_name):
                if not mod.preamble and mod.field not in enc_field_names:
                    continue
                field_ref = (
                    mod.field if mod.preamble else f'inst->{mod.field}'
                )
                if mod.preamble:
                    modifier_lines += mod.preamble
                if mod.is_offset:
                    cond = mod.condition if mod.condition else field_ref
                    modifier_lines += (
                        f'if ({cond}) modifiers_ += " offset:"'
                        f' + std::to_string({field_ref});'
                    )
                else:
                    modifier_lines += (
                        f'if ({field_ref}) modifiers_ += "{mod.display}";'
                    )

            has_op = any(f.name == 'op' for f in inst_enc.ucode_fields)
            size_line = (' size_ = sizeof(OpEncoding);\n'
                        '  raw_encoding_ = reinterpret_cast<const uint32_t *>(&inst_);\n'
                        '  encoding_id_ = raw_encoding_[0] >> 23;')
            if has_op:
                size_line += '\n  opcode_ = inst_.op;'
            if size_condition is not None:
                size_line += (
                    f' if ({size_condition})'
                    f' size_ += sizeof(MachineInst);'
                )
            if rule.use_flat_mnemonic:
                # FLAT mnemonics are dynamically constructed ("scratch_*",
                # "global_*"). Store the owned string in a member so the
                # string_view in Instruction doesn't dangle.
                class_ctor_impl = (
                    f'{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}'
                    f'(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst, ExecuteFn exec_fn) '
                    f': IsaInstruction<Isa>("", exec_fn), inst_(*inst), '
                    f'owned_mnemonic_({mnemonic_expr}) '
                    f'{{ mnemonic_ = owned_mnemonic_;{size_line}}}'
                )
            else:
                class_ctor_impl = (
                    f'{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}'
                    f'(std::string_view mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst, ExecuteFn exec_fn) '
                    f': IsaInstruction<Isa>({mnemonic_expr}, exec_fn), inst_(*inst) '
                    f'{{{size_line}}}'
                )
            class_func_impls.append(cgen.Line(class_ctor_impl))

            # Generate build_modifiers() override for encoding bases
            # that have modifier flags (memory instructions). This is
            # called lazily by disassemble() instead of eagerly in the
            # constructor, avoiding string allocation on the hot path.
            if modifier_lines:
                public_members.append(
                    cgen.Line('void build_modifiers(std::string &out) const override;'),
                )
                # The modifier_lines were written for the constructor where
                # they appended to modifiers_ and accessed inst->field.
                # Rewrite to append to 'out' and access via local pointer.
                mod_impl = modifier_lines.replace('modifiers_', 'out')
                class_func_impls.append(cgen.Line(
                    f'void {inst_enc.fmt_enc_name}::build_modifiers'
                    f'(std::string &out) const '
                    f'{{ auto *inst = &inst_;(void)inst;'
                    f'{mod_impl}}}'
                ))
            fmt_enc_name = inst_enc.fmt_enc_name

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
                    if (
                        enc_cond[0] == 'default_encoding'
                        and not has_real_default_check
                    ):
                        continue
                    seen_conds.add(enc_cond[0])
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value('bool', f'{enc_cond[0]}'), []
                    )
                    func_body = cgen.FunctionBody(
                        cgen.FunctionDeclaration(
                            cgen.Value(
                                'bool', f'{fmt_enc_name}::{enc_cond[0]}'
                            ),
                            [],
                        ),
                        cgen.Block(
                            [cgen.Statement(f'return {enc_cond[1]}')]
                        ),
                    )
                    public_members.append(func_decl)
                    class_func_impls.append(func_body)

            if inst_enc.has_implied_literal_ops:
                func_decl = cgen.FunctionDeclaration(
                    cgen.Value('bool', 'hasImpliedLiteral'), []
                )
                implied_literal_cond = ' || '.join(
                    f'inst_.op == {op}'
                    for op in inst_enc.implied_literal_ops
                )
                func_body = cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value(
                            'bool',
                            f'{fmt_enc_name}::hasImpliedLiteral',
                        ),
                        [],
                    ),
                    cgen.Block(
                        [cgen.Statement(f'return {implied_literal_cond}')]
                    ),
                )
                public_members.append(func_decl)
                class_func_impls.append(func_body)

            class_members.extend(public_members)
            class_members.append(
                cgen.Statement(
                    f'using OpEncoding = {inst_enc.fmt_enc_name}MachineInst'
                )
            )
            class_members.append(
                cgen.Statement(
                    'const OpEncoding inst_'
                )
            )
            # FLAT encoding bases need an owned string for the dynamic mnemonic.
            if rule.use_flat_mnemonic:
                class_members.append(
                    cgen.Statement('std::string owned_mnemonic_')
                )
            # VOP1/VOP2 encoding bases store DPP control fields.
            # apply_dpp() is a free function in dpp_sdwa_ops.h.
            if inst_enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                class_members.append(cgen.Statement('uint32_t dpp_ctrl_ = 0'))
                class_members.append(cgen.Statement('uint32_t dpp_row_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bank_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bound_ctrl_ = 0'))
                class_members.append(cgen.Statement('std::unique_ptr<DppOperand> dpp_src0_'))
                class_members.append(cgen.Statement('std::unique_ptr<DppOperand> dpp_src1_'))
                # SDWA fields (CDNA and RDNA1/2 have hardware SDWA encoding; fields
                # are present on all ISAs for uniform codegen even if unused).
                class_members.append(cgen.Statement('uint32_t sdwa_src0_sel_ = amdgpu::sdwa::DWORD'))
                class_members.append(cgen.Statement('bool sdwa_src0_sext_ = false'))
                class_members.append(cgen.Statement('uint32_t sdwa_src1_sel_ = amdgpu::sdwa::DWORD'))
                class_members.append(cgen.Statement('bool sdwa_src1_sext_ = false'))
                class_members.append(cgen.Statement('uint32_t sdwa_dst_sel_ = amdgpu::sdwa::DWORD'))
                class_members.append(cgen.Statement('uint32_t sdwa_dst_unused_ = 0'))
                class_members.append(cgen.Statement('bool sdwa_clamp_ = false'))
            s = cgen.Struct(
                f'{inst_enc.fmt_enc_name} : public IsaInstruction<Isa>',
                [x for x in class_members],
            )
            enc_classes.append(s)

        class_def_file = CppFile(
            'encodings',
            self.out_path,
            True,
            [
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/isa.h',
                    False,
                ),
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/machine_insts.h',
                    False,
                ),
                ('rocjitsu/isa/instruction.h', False),
                ('rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h', False),
                ('string', True),
                ('string_view', True),
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
                f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/encodings.h',
                False,
            ),
            ('string', True),
        ]
        class_impl_file = CppFile(
            'encodings',
            self.out_path,
            False,
            _enc_cpp_includes,
            [],
            class_func_impls,
            self.isa_spec.arch_name,
        )
        class_def_file.gen_code()
        class_impl_file.gen_code()

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
            return f'inst_.{field}'
        return 'inst_.op_sel_hi_2'

    # Semantic operations/classes where the destination register is also read.
    # The CDNA4 XML marks these destinations as output-only, but the execute
    # body reads the old value (accumulate, swap, partial write, bitfield set).
    _READS_DST_OPS = frozenset({'fmac', 'bitset0', 'bitset1'})
    _READS_DST_CLASSES = frozenset({
        'vector_dot', 'vector_swap',
        'mad_mixlo_f16', 'mad_mixhi_f16',
    })

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
        return (sem.operation in self._READS_DST_OPS
                or sem.semantic_class in self._READS_DST_CLASSES)

    def _gen_execute_body(self, inst: Instruction, sem: InstructionSemantics, enc_name: str = '') -> str:
        """Generate execute() body from instruction semantics."""
        dst_ops = [op.name for op in inst.operands if not op.is_input]
        src_ops = [op.name for op in inst.operands if op.is_input]
        # Some instructions mark their destination as input (read-modify-write,
        # e.g. S_BITSET0, S_CMOV, V_FMAC, V_SWAP). Recover the destination
        # from src_ops when it looks like one.
        if not dst_ops and src_ops and src_ops[0] in ('sdst', 'vdst'):
            dst_ops = [src_ops[0]]
            src_ops = src_ops[1:]
        # Some ISA specs mark swap operands as output-only even though the
        # instruction reads both. Treat the second output as a source.
        if not src_ops and len(dst_ops) >= 2 and sem.semantic_class == 'vector_swap':
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
        # no abs modifier field. Set as instance state so that helper methods
        # (_vop3_src_mod, etc.) can read it without explicit threading.
        self._has_abs = profile.has_abs_modifier(inst.enc_name)
        self._enc_name = enc_name
        L = []  # output lines

        if cls == 'true_nop':
            return '  (void)wf;'

        if cls == 'nop':
            return '  (void)wf;\n throw util::UnimplementedInst(mnemonic());'

        if cls == 'endpgm':
            # Use end() instead of halt() to drain outstanding memory ops.
            # If all wait counters are zero, end() halts immediately.
            # Otherwise, it transitions to ENDING state and the memory
            # pipeline drain handles the final halt.
            L.append('  wf.end();')
            return '\n'.join(L)

        if cls == 'waitcnt':
            L.append(f'  uint16_t imm = static_cast<uint16_t>({src_ops[0]}.encoding_value_);')
            wf = self.isa_spec.profile.waitcnt_family
            if wf == 'gfx11':
                # GFX11 (RDNA3/3.5) SIMM16 layout:
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
            L.append(f'  uint16_t cnt = static_cast<uint16_t>({src_ops[0]}.encoding_value_);')
            L.append(f'  wf.set_wait_counter("{op}", cnt);')
            return '\n'.join(L)

        if cls == 'barrier':
            L.append('  wf.set_state(amdgpu::WfState::BARRIER);')
            return '\n'.join(L)

        if cls == 'branch':
            L.append(f'  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);')
            L.append('  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;')
            return '\n'.join(L)

        if cls == 'cbranch':
            cond_map = {
                'scc0': '!wf.read_scc()',
                'scc1': 'wf.read_scc()',
                'vccz': 'wf.vcc() == 0',
                'vccnz': 'wf.vcc() != 0',
                'execz': 'wf.exec() == 0',
                'execnz': 'wf.exec() != 0',
            }
            L.append(f'  if ({cond_map[cond]}) {{')
            L.append(f'    int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);')
            L.append('    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'scalar_mov':
            if dtype == 'b64':
                L.append(f'  {dst_ops[0]}.write_scalar64(wf, {src_ops[0]}.read_scalar64(wf));')
            else:
                L.append(f'  {dst_ops[0]}.write_scalar(wf, {src_ops[0]}.read_scalar(wf));')
            return '\n'.join(L)

        if cls == 'scalar_movk':
            L.append(f'  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));')
            return '\n'.join(L)

        if cls == 'scalar_cmov':
            if dtype == 'b64':
                L.append(f'  if (wf.read_scc()) {dst_ops[0]}.write_scalar64(wf, {src_ops[0]}.read_scalar64(wf));')
            else:
                L.append(f'  if (wf.read_scc()) {dst_ops[0]}.write_scalar(wf, {src_ops[0]}.read_scalar(wf));')
            return '\n'.join(L)

        if cls == 'scalar_cmovk':
            L.append(f'  if (wf.read_scc()) {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));')
            return '\n'.join(L)

        if cls == 'scalar_cselect':
            if dtype == 'b64':
                L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.read_scc() ? {src_ops[0]}.read_scalar64(wf) : {src_ops[1]}.read_scalar64(wf));')
            else:
                L.append(f'  {dst_ops[0]}.write_scalar(wf, wf.read_scc() ? {src_ops[0]}.read_scalar(wf) : {src_ops[1]}.read_scalar(wf));')
            return '\n'.join(L)

        if cls == 'scalar_unary':
            return self._gen_scalar_unary(dst_ops, src_ops, op, dtype, scc)

        if cls == 'scalar_binop':
            return self._gen_scalar_binop(dst_ops, src_ops, op, dtype, scc)

        if cls == 'scalar_cmp':
            return self._gen_scalar_cmp(src_ops, op, dtype)

        if cls == 'scalar_cmpk':
            return self._gen_scalar_cmpk(dst_ops, src_ops, op, dtype)

        if cls == 'scalar_addk':
            L.append(f'  int32_t s0 = static_cast<int32_t>({dst_ops[0]}.read_scalar(wf));')
            L.append(f'  int32_t imm = static_cast<int16_t>({src_ops[0]}.encoding_value_);')
            L.append('  int64_t wide = static_cast<int64_t>(s0) + static_cast<int64_t>(imm);')
            L.append('  int32_t result = static_cast<int32_t>(wide);')
            L.append(f'  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            L.append('  wf.write_scc(wide != static_cast<int64_t>(result));')
            return '\n'.join(L)

        if cls == 'scalar_mulk':
            L.append(f'  int32_t s0 = static_cast<int32_t>({dst_ops[0]}.read_scalar(wf));')
            L.append(f'  int32_t imm = static_cast<int16_t>({src_ops[0]}.encoding_value_);')
            L.append(f'  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(s0 * imm));')
            return '\n'.join(L)

        if cls == 'scalar_saveexec':
            return self._gen_scalar_saveexec(dst_ops, src_ops, op)

        if cls == 'scalar_wrexec':
            L.append(f'  uint64_t src = {src_ops[0]}.read_scalar64(wf);')
            if op in ('andn1', 'and_not1'):
                # EXEC = SRC & ~EXEC
                L.append('  wf.set_exec(src & ~wf.exec());')
            elif op in ('andn2', 'and_not0'):
                # EXEC = EXEC & ~SRC
                L.append('  wf.set_exec(wf.exec() & ~src);')
            else:
                L.append(f'  wf.set_exec(src); // TODO: {op}')
            return '\n'.join(L)

        if cls == 'scalar_getpc':
            # S_GETPC_B64: returns PC of the instruction FOLLOWING the S_GETPC.
            # At execute() time, wf.pc points to the S_GETPC itself; step() will
            # add size_ afterwards. Write wf.pc + size_ so the net result after
            # post-execute advance is correct (caller sees PC of next instruction).
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.pc + size_);')
            return '\n'.join(L)

        if cls == 'scalar_setpc':
            L.append(f'  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;')
            return '\n'.join(L)

        if cls == 'scalar_swappc':
            # S_SWAPPC_B64: dst = PC of next inst, then jump to src.
            L.append(f'  uint64_t next_pc = wf.pc + size_;')
            L.append(f'  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;')
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, next_pc);')
            return '\n'.join(L)

        if cls == 'scalar_call':
            # S_CALL_B64: dst = PC of next instruction (return address), then branch.
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.pc + size_);')
            L.append(f'  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);')
            L.append('  wf.pc = wf.pc + static_cast<int64_t>(offset) * 4 - size_;')
            return '\n'.join(L)

        if cls == 'scalar_bitcmp':
            return self._gen_scalar_bitcmp(src_ops, op, dtype)

        if cls == 'vector_mov':
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'b64' and is_vop3:
                L.append(f'    double s = std::bit_cast<double>({src_ops[0]}.read_lane64(wf, lane));')
                L.extend(self._vop3_src_mod('s', 0))
                L.extend(self._vop3_dst_mod_f64('s'))
                L.append(f'    {dst_ops[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(s));')
            elif dtype == 'b64':
                L.append(f'    {dst_ops[0]}.write_lane64(wf, lane, {src_ops[0]}.read_lane64(wf, lane));')
            elif is_vop3:
                L.append(f'    float s = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));')
                L.extend(self._vop3_src_mod('s', 0))
                L.extend(self._vop3_dst_mod('s'))
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(s));')
            else:
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, {src_ops[0]}.read_lane(wf, lane));')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_unary':
            return self._gen_vector_unary(dst_ops, src_ops, op, dtype, is_vop3)

        if cls == 'vector_binop':
            return self._gen_vector_binop(dst_ops, src_ops, op, dtype, is_vop3)

        if cls == 'vector_ternary':
            return self._gen_vector_ternary(dst_ops, src_ops, op, dtype, is_vop3)

        if cls == 'vector_cmp':
            return self._gen_vector_cmp(dst_ops, src_ops, op, dtype, is_vop3)

        if cls == 'vector_cmpx':
            return self._gen_vector_cmpx(src_ops, op, dtype, is_vop3, dst_ops)

        if cls == 'vector_cndmask':
            # v_cndmask_b32 is a pure bitwise select — no input/output
            # modifiers on any GFX version.  The VOP3 encoding's abs/neg/
            # omod bits overlap with src2 and must be ignored.
            L.append('  uint64_t exec = wf.exec();')
            if is_vop3:
                L.append(f'  uint64_t cond = {src_ops[2]}.read_scalar64(wf);')
            else:
                L.append('  uint64_t cond = wf.vcc();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if is_vop3:
                L.append(f'    uint32_t val = (cond & (1ULL << lane))')
                L.append(f'        ? {src_ops[1]}.read_lane(wf, lane)')
                L.append(f'        : {src_ops[0]}.read_lane(wf, lane);')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, val);')
            else:
                L.append(f'    uint32_t val = (cond & (1ULL << lane))')
                L.append(f'        ? {src_ops[1]}.read_lane(wf, lane)')
                L.append(f'        : {src_ops[0]}.read_lane(wf, lane);')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, val);')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_add_co':
            return self._gen_vector_add_co(dst_ops, src_ops, op, dtype)

        if cls == 'vector_readfirstlane':
            L.append('  uint64_t exec = wf.exec();')
            L.append('  uint32_t val = 0;')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (exec & (1ULL << lane)) {')
            L.append(f'      val = {src_ops[0]}.read_lane(wf, lane);')
            L.append('      break;')
            L.append('    }')
            L.append('  }')
            L.append(f'  {dst_ops[0]}.write_scalar(wf, val);')
            return '\n'.join(L)

        if cls == 'vector_readlane':
            L.append(f'  uint32_t lane = {src_ops[1]}.read_scalar(wf);')
            L.append(f'  {dst_ops[0]}.write_scalar(wf, {src_ops[0]}.read_lane(wf, lane));')
            return '\n'.join(L)

        if cls == 'vector_writelane':
            L.append(f'  uint32_t val = {src_ops[0]}.read_scalar(wf);')
            L.append(f'  uint32_t lane = {src_ops[1]}.read_scalar(wf);')
            L.append(f'  {dst_ops[0]}.write_lane(wf, lane, val);')
            return '\n'.join(L)

        if cls == 'vector_swap':
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            L.append(f'    uint32_t tmp = {dst_ops[0]}.read_lane(wf, lane);')
            L.append(f'    {dst_ops[0]}.write_lane(wf, lane, {src_ops[0]}.read_lane(wf, lane));')
            L.append(f'    {src_ops[0]}.write_lane(wf, lane, tmp);')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_cmp_class':
            return self._gen_vector_cmp_class(dst_ops, src_ops, dtype, False, is_vop3)

        if cls == 'vector_cmpx_class':
            return self._gen_vector_cmp_class(dst_ops, src_ops, dtype, True, is_vop3)

        if cls == 'vector_fmamk':
            # D = S0 * K + S2, K is inline constant (second src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = (
                f'{src_ops[1]}.encoding_value_'
                if len(src_ops) >= 3
                else 'simm32_'
            )
            s2_expr = src_ops[2] if len(src_ops) >= 3 else src_ops[1]
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                L.append(f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));')
                L.append(f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));')
                L.append(f'    float s2 = util::f16_to_f32(static_cast<uint16_t>({s2_expr}.read_lane(wf, lane)));')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, k, s2)));')
            else:
                L.append(f'    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));')
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(f'    float s2 = std::bit_cast<float>({s2_expr}.read_lane(wf, lane));')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, k, s2)));')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_fmaak':
            # D = S0 * S1 + K, K is inline constant (third src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = (
                f'{src_ops[2]}.encoding_value_'
                if len(src_ops) >= 3
                else 'simm32_'
            )
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                L.append(f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));')
                L.append(f'    float s1 = util::f16_to_f32(static_cast<uint16_t>({src_ops[1]}.read_lane(wf, lane)));')
                L.append(f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, s1, k)));')
            else:
                L.append(f'    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));')
                L.append(f'    float s1 = std::bit_cast<float>({src_ops[1]}.read_lane(wf, lane));')
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, s1, k)));')
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_mbcnt':
            return self._gen_vector_mbcnt(dst_ops, src_ops, op)

        if cls == 'vector_mad_64_32':
            return self._gen_vector_mad_64_32(dst_ops, src_ops, dtype)

        if cls == 'vector_mad_32_16':
            return self._gen_vector_mad_32_16(dst_ops, src_ops, dtype)

        if cls == 'vector_div_fixup':
            return self._gen_vector_div_fixup(dst_ops, src_ops, dtype, is_vop3)

        if cls == 'vector_div_scale':
            return self._gen_vector_div_scale(dst_ops, src_ops, dtype, is_vop3)

        if cls == 'vector_div_fmas':
            return self._gen_vector_div_fmas(dst_ops, src_ops, dtype, is_vop3)

        if cls == 'vector_dot':
            return self._gen_vector_dot(dst_ops, src_ops, op, dtype)

        if cls in ('vector_cvt_pk_u8_f32', 'vector_cvt_pknorm',
                    'vector_cvt_pkrtz_f16_f32', 'vector_cvt_pk',
                    'vector_cvt_pk_f16_f32', 'vector_cvt_pk_bf16_f32',
                    'vector_cvt_sr_f16_f32', 'vector_cvt_sr_bf16_f32',
                    'vector_pack_b32_f16'):
            return self._gen_vector_cvt_pk(dst_ops, src_ops, cls, op)

        if cls == 'vector_dot2c_bf16':
            return self._gen_vector_dot2c_bf16(dst_ops, src_ops)

        if cls == 'vector_bitop3':
            return self._gen_vector_bitop3(dst_ops, src_ops, dtype)

        if cls == 'vector_permlane16_swap':
            return self._gen_vector_permlane_swap(dst_ops, src_ops, stride=16)

        if cls == 'vector_permlane32_swap':
            return self._gen_vector_permlane_swap(dst_ops, src_ops, stride=32)

        if cls == 'vector_permlane16':
            return self._gen_vector_permlane(dst_ops, src_ops, op, cross=False)

        if cls == 'vector_permlanex16':
            return self._gen_vector_permlane(dst_ops, src_ops, op, cross=True)

        if cls == 'vector_permlane64':
            return self._gen_vector_permlane64(dst_ops, src_ops)

        # ----- VOP3P: packed / dot / mix / MFMA -----
        if cls == 'pk_binop':
            return self._gen_pk_binop(dst_ops, src_ops, op, dtype)

        if cls == 'pk_ternary':
            return self._gen_pk_ternary(dst_ops, src_ops, op, dtype)

        if cls == 'pk_binop_f32':
            return self._gen_pk_binop_f32(dst_ops, src_ops, op)

        if cls == 'pk_ternary_f32':
            return self._gen_pk_ternary_f32(dst_ops, src_ops, op)

        if cls == 'pk_mov_b32':
            return self._gen_pk_mov_b32(dst_ops, src_ops)

        if cls == 'mad_mix_f32':
            return self._gen_mad_mix_f32(dst_ops, src_ops)

        if cls == 'mad_mixlo_f16':
            return self._gen_mad_mix_lo_hi(dst_ops, src_ops, is_lo=True)

        if cls == 'mad_mixhi_f16':
            return self._gen_mad_mix_lo_hi(dst_ops, src_ops, is_lo=False)

        if cls.startswith('dot2_'):
            return self._gen_dot2(dst_ops, src_ops, cls)

        if cls.startswith('dot4_'):
            return self._gen_dot4(dst_ops, src_ops, cls)

        if cls.startswith('dot8_'):
            return self._gen_dot8(dst_ops, src_ops, cls)

        if cls == 'accvgpr_read':
            return self._gen_accvgpr_read(dst_ops, src_ops)

        if cls == 'accvgpr_write':
            return self._gen_accvgpr_write(dst_ops, src_ops)

        if cls == 'mfma':
            return self._gen_mfma(inst, dst_ops, src_ops)

        if cls == 'smem_load':
            return self._gen_smem_load(dst_ops, src_ops, sem)

        if cls == 'smem_store':
            return self._gen_smem_store(dst_ops, src_ops, sem)

        if cls == 'flat_load':
            return self._gen_flat_load(dst_ops, src_ops, sem)

        if cls == 'flat_store':
            return self._gen_flat_store(dst_ops, src_ops, sem)

        if cls in ('buffer_load', 'tbuffer_load'):
            return self._gen_buffer_load(dst_ops, src_ops, sem, cls, inst)

        if cls in ('buffer_store', 'tbuffer_store'):
            return self._gen_buffer_store(dst_ops, src_ops, sem, cls)

        if cls in ('ds_read', 'ds_read2', 'ds_write', 'ds_write2',
                   'ds_read_addtid', 'ds_write_addtid',
                   'ds_read_tr_b16', 'ds_read_tr_b8', 'ds_read_tr_b4', 'ds_read_tr_b6'):
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = ('  if (inst_.gds)\n'
                             '    throw util::UnimplementedInst(mnemonic());\n')
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
            return '  wf.cu().l1_scalar().writeback_all();'

        if cls == 'gl1_inv':
            return '  wf.cu().l1_vector().invalidate_all();'

        if cls == 'flat_atomic':
            return self._gen_flat_atomic(dst_ops, src_ops, sem)

        if cls == 'buffer_atomic':
            return self._gen_buffer_atomic(dst_ops, src_ops, sem)

        if cls == 'ds_atomic':
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = ('  if (inst_.gds)\n'
                             '    throw util::UnimplementedInst(mnemonic());\n')
            return gds_guard + self._gen_ds_atomic(dst_ops, src_ops, sem)

        if cls == 'ds_permute':
            is_bpermute = 'BPERMUTE' in sem.name.upper()
            L.append(f'  auto &cu = wf.cu();')
            L.append(f'  uint64_t exec = wf.exec();')
            L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
            L.append(f'  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);')
            L.append(f'  // Pre-read all data0 values from every lane.')
            L.append(f'  uint32_t src_data[64];')
            L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
            L.append(f'    src_data[i] = cu.read_vgpr(vb + inst_.data0, i);')
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
                L.append(f'    uint32_t addr_val = cu.read_vgpr(vb + inst_.addr, i);')
                L.append(f'    uint32_t src_lane = ((addr_val + offset) / 4) % wf.wf_size();')
                L.append(f'    if (exec & (1ULL << src_lane))')
                L.append(f'      tmp[i] = src_data[src_lane];')
                L.append(f'  }}')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    if (exec & (1ULL << i))')
                L.append(f'      cu.write_vgpr(vb + inst_.vdst, i, tmp[i]);')
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
                L.append(f'    uint32_t addr_val = cu.read_vgpr(vb + inst_.addr, i);')
                L.append(f'    uint32_t dst_lane = ((addr_val + offset) / 4) % wf.wf_size();')
                L.append(f'    tmp[dst_lane] = src_data[i];')
                L.append(f'  }}')
                L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i) {{')
                L.append(f'    if (exec & (1ULL << i))')
                L.append(f'      cu.write_vgpr(vb + inst_.vdst, i, tmp[i]);')
                L.append(f'  }}')
            return '\n'.join(L)

        if cls == 'ds_swizzle':
            # DS_SWIZZLE_B32: lane swizzle controlled by offset field.
            # The offset encodes the swizzle pattern. For QDMode (bit 15=1):
            #   for each lane in quad: dst = src[and_mask & or_mask ^ xor_mask]
            # For BitMode (bit 15=0): full-wave swizzle via and/or/xor.
            # Simplified: treat as identity (passthrough) for now.
            L.append(f'  auto &cu = wf.cu();')
            L.append(f'  uint64_t exec = wf.exec();')
            L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
            L.append(f'  uint32_t src_data[64];')
            L.append(f'  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
            L.append(f'    src_data[i] = cu.read_vgpr(vb + inst_.data0, i);')
            L.append(f'  uint32_t offset = inst_.offset0 | (inst_.offset1 << 8);')
            L.append(f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{')
            L.append(f'    if (!(exec & (1ULL << lane))) continue;')
            L.append(f'    uint32_t src_lane;')
            L.append(f'    if (offset & 0x8000) {{')
            L.append(f'      // QDMode: swizzle within 4-lane quads.')
            L.append(f'      uint32_t and_mask = offset & 0x1F;')
            L.append(f'      uint32_t or_mask = (offset >> 5) & 0x1F;')
            L.append(f'      uint32_t xor_mask = (offset >> 10) & 0x1F;')
            L.append(f'      src_lane = ((lane & and_mask) | or_mask) ^ xor_mask;')
            L.append(f'      src_lane = (lane & ~0x3) | (src_lane & 0x3);  // stay in quad')
            L.append(f'    }} else {{')
            L.append(f'      // BitMode: full-wave swizzle.')
            L.append(f'      uint32_t and_mask = offset & 0x1F;')
            L.append(f'      uint32_t or_mask = (offset >> 5) & 0x1F;')
            L.append(f'      uint32_t xor_mask = (offset >> 10) & 0x1F;')
            L.append(f'      src_lane = ((lane & and_mask) | or_mask) ^ xor_mask;')
            L.append(f'    }}')
            L.append(f'    if (src_lane < wf.wf_size())')
            L.append(f'      cu.write_vgpr(vb + inst_.vdst, lane, src_data[src_lane]);')
            L.append(f'  }}')
            return '\n'.join(L)

        # ── Image pipeline stubs ──────────────────────────────────────────
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
            L.append('  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.')
            return '\n'.join(L)

        return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: {cls}'

    def _gen_vector_cmp_class(self, dst: list[str], src: list[str], dtype: str | None, is_cmpx: bool, is_vop3: bool = False) -> str:
        """Generate V_CMP_CLASS / V_CMPX_CLASS body."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        if is_cmpx:
            L.append('  uint64_t result = 0;')
        elif dst:
            # VOP3: initialize from destination register for inactive lanes.
            L.append(f'  uint64_t vcc = {dst[0]}.read_scalar64(wf);')
        else:
            L.append('  uint64_t vcc = wf.vcc();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if dtype == 'f64':
            L.append(f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
            L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
            L.append('    bool match = false;')
            L.append('    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) == 0) match = true;')
            L.append('    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) != 0) match = true;')
            L.append('    if ((mask & 0x004) && std::isinf(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x020) && s0 == 0.0 && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x040) && s0 == 0.0 && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0) match = true;')
            L.append('    if ((mask & 0x200) && std::isinf(s0) && s0 > 0) match = true;')
        elif dtype == 'f16':
            # Read raw f16 bits first for sNaN/qNaN detection (bit 9 is the
            # quiet NaN bit in IEEE 754 binary16), then convert to f32 for
            # the remaining class checks. The f16→f32 conversion may turn
            # sNaN into qNaN, so we cannot rely on the converted value.
            L.append(f'    uint16_t s0_raw = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s0 = util::f16_to_f32(s0_raw);')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
            L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
            L.append('    bool match = false;')
            L.append('    bool is_f16_nan = ((s0_raw & 0x7C00) == 0x7C00) && ((s0_raw & 0x03FF) != 0);')
            L.append('    if ((mask & 0x001) && is_f16_nan && (s0_raw & 0x0200) == 0) match = true;')
            L.append('    if ((mask & 0x002) && is_f16_nan && (s0_raw & 0x0200) != 0) match = true;')
            L.append('    if ((mask & 0x004) && std::isinf(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0) match = true;')
            L.append('    if ((mask & 0x200) && std::isinf(s0) && s0 > 0) match = true;')
        else:
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
            L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
            L.append('    bool match = false;')
            L.append('    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) == 0) match = true;')
            L.append('    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) != 0) match = true;')
            L.append('    if ((mask & 0x004) && std::isinf(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0) match = true;')
            L.append('    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && !std::signbit(s0)) match = true;')
            L.append('    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0) match = true;')
            L.append('    if ((mask & 0x200) && std::isinf(s0) && s0 > 0) match = true;')
        if is_cmpx:
            L.append('    if (match) result |= (1ULL << lane);')
        else:
            L.append('    if (match) vcc |= (1ULL << lane);')
            L.append('    else vcc &= ~(1ULL << lane);')
        L.append('  }')
        if is_cmpx:
            if self.isa_spec.profile.cmpx_writes_vcc:
                L.append('  wf.set_vcc(result);')
            L.append('  wf.set_exec(result);')
        elif dst:
            L.append(f'  {dst[0]}.write_scalar64(wf, vcc);')
        else:
            L.append('  wf.set_vcc(vcc);')
        return '\n'.join(L)

    def _gen_vector_mbcnt(self, dst: list[str], src: list[str], op: str | None) -> str:
        """Generate V_MBCNT_LO/HI_U32_B32 body."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t mask = {src[0]}.read_lane(wf, lane);')
        L.append(f'    uint32_t base = {src[1]}.read_lane(wf, lane);')
        if op == 'lo':
            L.append('    uint32_t thread_mask = lane < 32 ? (1u << lane) - 1 : 0xFFFFFFFFu;')
            L.append('    uint32_t count = std::popcount(mask & thread_mask);')
        else:  # hi
            L.append('    uint32_t shift = lane >= 32 ? lane - 32 : 0;')
            L.append('    uint32_t thread_mask = lane >= 32 ? (1u << shift) - 1 : 0;')
            L.append('    uint32_t count = std::popcount(mask & thread_mask);')
        L.append(f'    {dst[0]}.write_lane(wf, lane, base + count);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_mad_64_32(self, dst: list[str], src: list[str], dtype: str | None) -> str:
        """Generate V_MAD_U64_U32 / V_MAD_I64_I32 body.

        D.i64 = S0.i32 * S1.i32 + S2.i64 (signed)
        D.u64 = S0.u32 * S1.u32 + S2.u64 (unsigned)

        Sources S0 and S1 are 32-bit; the accumulator S2 and result D are
        64-bit VGPR pairs.
        """
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if dtype == 'i64':
            L.append(f'    int64_t s0 = static_cast<int32_t>({src[0]}.read_lane(wf, lane));')
            L.append(f'    int64_t s1 = static_cast<int32_t>({src[1]}.read_lane(wf, lane));')
            L.append(f'    int64_t s2 = static_cast<int64_t>({src[2]}.read_lane64(wf, lane));')
            L.append('    uint64_t result = static_cast<uint64_t>(s0 * s1 + s2);')
        else:
            L.append(f'    uint64_t s0 = {src[0]}.read_lane(wf, lane);')
            L.append(f'    uint64_t s1 = {src[1]}.read_lane(wf, lane);')
            L.append(f'    uint64_t s2 = {src[2]}.read_lane64(wf, lane);')
            L.append('    uint64_t result = s0 * s1 + s2;')
        L.append(f'    {dst[0]}.write_lane64(wf, lane, result);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_mad_32_16(self, dst: list[str], src: list[str], dtype: str | None) -> str:
        """Generate V_MAD_U32_U16 / V_MAD_I32_I16 body.

        D.u32 = S0.u16 * S1.u16 + S2.u32 (unsigned)
        D.i32 = S0.i16 * S1.i16 + S2.i32 (signed)
        """
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if dtype == 'i32':
            L.append(f'    int32_t s0 = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int32_t s1 = static_cast<int16_t>({src[1]}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int32_t s2 = static_cast<int32_t>({src[2]}.read_lane(wf, lane));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(s0 * s1 + s2));')
        else:
            L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane) & 0xFFFFu;')
            L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane) & 0xFFFFu;')
            L.append(f'    uint32_t s2 = {src[2]}.read_lane(wf, lane);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, s0 * s1 + s2);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_div_fixup(self, dst: list[str], src: list[str], dtype: str | None, is_vop3: bool = False) -> str:
        """Generate V_DIV_FIXUP body (corrects division result)."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if dtype == 'f64':
            L.append(f'    double p = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            L.append(f'    double b = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));')
            L.append(f'    double c = std::bit_cast<double>({src[2]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('p', 0))
                L.extend(self._vop3_src_mod('b', 1))
                L.extend(self._vop3_src_mod('c', 2))
            L.append('    double result;')
            L.append('    if (std::isnan(b)) result = b;')
            L.append('    else if (std::isnan(c)) result = c;')
            L.append('    else if (c == 0.0 && b == 0.0) result = std::numeric_limits<double>::quiet_NaN();')
            L.append('    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<double>::quiet_NaN();')
            L.append('    else if (b == 0.0) {')
            L.append('      result = std::copysign(std::numeric_limits<double>::infinity(),')
            L.append('                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));')
            L.append('    }')
            L.append('    else if (c == 0.0) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));')
            L.append('    else if (std::isinf(c)) {')
            L.append('      result = std::copysign(std::numeric_limits<double>::infinity(),')
            L.append('                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));')
            L.append('    }')
            L.append('    else if (std::isinf(b)) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));')
            L.append('    else result = p;')
            if is_vop3:
                L.extend(self._vop3_dst_mod_f64('result'))
            L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
        else:
            L.append(f'    float p = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float b = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    float c = std::bit_cast<float>({src[2]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('p', 0))
                L.extend(self._vop3_src_mod('b', 1))
                L.extend(self._vop3_src_mod('c', 2))
            L.append('    float result;')
            L.append('    if (std::isnan(b)) result = b;')
            L.append('    else if (std::isnan(c)) result = c;')
            L.append('    else if (c == 0.0f && b == 0.0f) result = std::numeric_limits<float>::quiet_NaN();')
            L.append('    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<float>::quiet_NaN();')
            L.append('    else if (b == 0.0f) {')
            L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
            L.append('                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));')
            L.append('    }')
            L.append('    else if (c == 0.0f) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));')
            L.append('    else if (std::isinf(c)) {')
            L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
            L.append('                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));')
            L.append('    }')
            L.append('    else if (std::isinf(b)) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));')
            L.append('    else result = p;')
            if is_vop3:
                L.extend(self._vop3_dst_mod('result'))
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_div_scale(self, dst: list[str], src: list[str], dtype: str | None, is_vop3: bool = False) -> str:
        """Generate V_DIV_SCALE body per ISA pseudocode (CDNA4 p.363-365).

        S1 = denominator, S2 = numerator. S0 selects which to scale
        (S0==S1 → scale denominator, S0==S2 → scale numerator).
        VCC is set when V_DIV_FMAS must apply post-scaling.
        """
        is_f64 = (dtype == 'f64')
        scale_exp = 128 if is_f64 else 64
        exp_threshold = 768 if is_f64 else 96
        tiny_exp = 53 if is_f64 else 23
        fp_type = 'double' if is_f64 else 'float'
        zero = '0.0' if is_f64 else '0.0f'
        read_fn = 'read_lane64' if is_f64 else 'read_lane'
        write_fn = 'write_lane64' if is_f64 else 'write_lane'
        cast_to = 'uint64_t' if is_f64 else 'uint32_t'
        nan_val = 'std::numeric_limits<double>::quiet_NaN()' if is_f64 else 'std::numeric_limits<float>::quiet_NaN()'

        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint64_t vcc = wf.vcc();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    {fp_type} s0 = std::bit_cast<{fp_type}>({src[0]}.{read_fn}(wf, lane));')
        L.append(f'    {fp_type} s1 = std::bit_cast<{fp_type}>({src[1]}.{read_fn}(wf, lane));')
        L.append(f'    {fp_type} s2 = std::bit_cast<{fp_type}>({src[2]}.{read_fn}(wf, lane));')
        if is_vop3:
            L.extend(self._vop3_src_mod('s0', 0))
            L.extend(self._vop3_src_mod('s1', 1))
            L.extend(self._vop3_src_mod('s2', 2))
        L.append(f'    {fp_type} result = s0;')
        L.append('    bool set_vcc = false;')
        L.append(f'    if (s2 == {zero} || s1 == {zero}) {{')
        L.append(f'      // Zero numerator or denominator: pass through s0 unscaled.')
        L.append(f'      // Special-case handling (0/0, 0/x, x/0) is done by v_div_fixup.')
        L.append('    } else {')
        L.append('      int exp1 = 0, exp2 = 0;')
        L.append('      std::frexp(s1, &exp1);')
        L.append('      std::frexp(s2, &exp2);')
        L.append(f'      if (exp2 - exp1 >= {exp_threshold}) {{')
        L.append('        set_vcc = true;')
        L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
        L.append(f'      }} else if (std::fpclassify(s1) == FP_SUBNORMAL) {{')
        L.append(f'        result = std::ldexp(s0, {scale_exp});')
        if is_f64:
            L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL &&')
            L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
        else:
            L.append(f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL &&')
            L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
        L.append('        set_vcc = true;')
        L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
        if is_f64:
            L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL) {{')
        else:
            L.append(f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL) {{')
        L.append(f'        result = std::ldexp(s0, -{scale_exp});')
        L.append(f'      }} else if (std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
        L.append('        set_vcc = true;')
        L.append(f'        if (s0 == s2) result = std::ldexp(s0, {scale_exp});')
        L.append(f'      }} else if (exp2 <= {tiny_exp}) {{')
        L.append(f'        result = std::ldexp(s0, {scale_exp});')
        L.append('      }')
        L.append('    }')
        L.append('    if (set_vcc) vcc |= (1ULL << lane);')
        L.append('    else vcc &= ~(1ULL << lane);')
        L.append(f'    {dst[0]}.{write_fn}(wf, lane, std::bit_cast<{cast_to}>(result));')
        L.append('  }')
        L.append('  wf.set_vcc(vcc);')
        return '\n'.join(L)

    def _gen_vector_div_fmas(self, dst: list[str], src: list[str], dtype: str | None, is_vop3: bool = False) -> str:
        """Generate V_DIV_FMAS body (FMA with scale based on VCC)."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint64_t vcc = wf.vcc();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if dtype == 'f64':
            L.append(f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            L.append(f'    double s1 = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));')
            L.append(f'    double s2 = std::bit_cast<double>({src[2]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
                L.extend(self._vop3_src_mod('s1', 1))
                L.extend(self._vop3_src_mod('s2', 2))
            L.append('    double result = std::fma(s0, s1, s2);')
            L.append('    if (vcc & (1ULL << lane)) {')
            L.append('      result = std::ldexp(result, 64);')
            L.append('    }')
            L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
        else:
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    float s2 = std::bit_cast<float>({src[2]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
                L.extend(self._vop3_src_mod('s1', 1))
                L.extend(self._vop3_src_mod('s2', 2))
            L.append('    float result = std::fma(s0, s1, s2);')
            L.append('    if (vcc & (1ULL << lane)) {')
            L.append('      result = std::ldexp(result, 32);')
            L.append('    }')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_dot(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate V_DOT*C body (dot product accumulate)."""
        L = []
        d = dst[0] if dst else src[0]
        s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
        L.append(f'    int32_t acc = static_cast<int32_t>({d}.read_lane(wf, lane));')
        if op == 'dot4c':
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append('      int8_t ea = static_cast<int8_t>((a >> (i * 8)) & 0xFF);')
            L.append('      int8_t eb = static_cast<int8_t>((b >> (i * 8)) & 0xFF);')
            L.append('      acc += static_cast<int32_t>(ea) * static_cast<int32_t>(eb);')
            L.append('    }')
        elif op == 'dot8c':
            L.append('    for (int i = 0; i < 8; ++i) {')
            L.append('      int32_t ea = static_cast<int32_t>((a >> (i * 4)) & 0xF);')
            L.append('      if (ea & 8) ea |= ~0xF;')
            L.append('      int32_t eb = static_cast<int32_t>((b >> (i * 4)) & 0xF);')
            L.append('      if (eb & 8) eb |= ~0xF;')
            L.append('      acc += ea * eb;')
            L.append('    }')
        elif op == 'dot2c' and dtype == 'f32':
            # V_DOT2C_F32_F16: D.f32 += f16_lo(A)*f16_lo(B) + f16_hi(A)*f16_hi(B)
            L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
            L.append('    float a1 = util::f16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));')
            L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
            L.append('    float b1 = util::f16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));')
            L.append('    float facc = std::bit_cast<float>(static_cast<uint32_t>(acc));')
            L.append('    facc += a0 * b0 + a1 * b1;')
            L.append('    acc = static_cast<int32_t>(std::bit_cast<uint32_t>(facc));')
        elif op == 'dot2c' and dtype == 'i32':
            # V_DOT2C_I32_I16: D.i32 += i16_lo(A)*i16_lo(B) + i16_hi(A)*i16_hi(B)
            L.append('    int16_t a0 = static_cast<int16_t>(a & 0xFFFF);')
            L.append('    int16_t a1 = static_cast<int16_t>((a >> 16) & 0xFFFF);')
            L.append('    int16_t b0 = static_cast<int16_t>(b & 0xFFFF);')
            L.append('    int16_t b1 = static_cast<int16_t>((b >> 16) & 0xFFFF);')
            L.append('    acc += static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1;')
        else:
            L.append(f'    (void)a; (void)b; // unhandled dot variant: {op}/{dtype}')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(acc));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_dot2c_bf16(self, dst: list[str], src: list[str]) -> str:
        """Generate V_DOT2C_F32_BF16 body: D.f32 += A.bf16[0]*B.bf16[0] + A.bf16[1]*B.bf16[1]."""
        L = []
        d = dst[0] if dst else src[0]
        s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
        L.append(f'    float acc = std::bit_cast<float>({d}.read_lane(wf, lane));')
        L.append('    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
        L.append('    float a1 = util::bf16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));')
        L.append('    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
        L.append('    float b1 = util::bf16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));')
        L.append('    acc += a0 * b0 + a1 * b1;')
        L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_bitop3(self, dst: list[str], src: list[str], dtype: str | None) -> str:
        """Generate V_BITOP3_B32/B16 body: 3-input LUT-based bitwise operation.

        The 8-bit truth table is packed into the VOP3 modifier fields:
          truth_table = (omod << 6) | (abs << 3) | neg
        NOT from any source operand value. Source modifiers are not applied.

        Index bit ordering:
          bit 2 = src0, bit 1 = src1, bit 0 = src2
        """
        nbits = '16' if dtype == 'b16' else '32'
        L = []
        L.append('  uint8_t truth_table = static_cast<uint8_t>')
        L.append('      ((inst_.omod << 6) | (inst_.abs << 3) | inst_.neg);')
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t a = {src[0]}.read_lane(wf, lane);')
        L.append(f'    uint32_t b = {src[1]}.read_lane(wf, lane);')
        L.append(f'    uint32_t c = {src[2]}.read_lane(wf, lane);')
        L.append(f'    uint32_t result = 0;')
        L.append(f'    for (int i = 0; i < {nbits}; ++i) {{')
        L.append('      uint32_t idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1);')
        L.append('      result |= ((truth_table >> idx) & 1) << i;')
        L.append('    }')
        L.append(f'    {dst[0]}.write_lane(wf, lane, result);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_permlane_swap(self, dst: list[str], src: list[str],
                                   stride: int) -> str:
        """Generate V_PERMLANE{16,32}_SWAP_B32.

        For each lane N in [0..stride-1]:
          tmp = src0[N]
          src0[N]        ← vdst[N + stride]
          vdst[N+stride] ← tmp
        vdst[0..stride-1] and src0[stride..] are UNCHANGED.
        EXEC mask is IGNORED.
        Both vdst and src0 are outputs (LLVM: returns {vdst_new, src0_new}).
        """
        L = []
        L.append('  uint32_t tmp_dst[64] = {}, tmp_src[64] = {};')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append(f'    tmp_dst[lane] = {dst[0]}.read_lane(wf, lane);')
        L.append(f'    tmp_src[lane] = {dst[1]}.read_lane(wf, lane);')
        L.append('  }')
        L.append(f'  for (uint32_t lane = 0; lane < {stride}; ++lane) {{')
        L.append(f'    if (lane + {stride} >= wf.wf_size()) break;')
        L.append(f'    {dst[1]}.write_lane(wf, lane, tmp_dst[lane + {stride}]);')
        L.append(f'    {dst[0]}.write_lane(wf, lane + {stride}, tmp_src[lane]);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_permlane(self, dst: list[str], src: list[str],
                              op: str | None, cross: bool) -> str:
        """Generate V_PERMLANE16_B32 / V_PERMLANEX16_B32 (imm and var forms).

        For each lane i, read from lane (i & ~0xF) | selector[i & 0xF].
        Immediate form: selector from src1 (low 16 lanes) / src2 (high 16 lanes),
          each is a 4-bit field per sub-lane packed into a scalar.
        Var form: selector from low 4 bits of src2 VGPR per lane.
        For permlanex16 (cross=True), XOR bit 4 into the source lane to
        enable cross-16-group fetches.
        """
        is_var = (op == 'var')
        L = []
        L.append('  constexpr bool fi = false, bound_ctrl = false;')
        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint32_t snap[64];')
        L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
        L.append(f'    snap[i] = {src[0]}.read_lane(wf, i);')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if is_var:
            L.append(f'    uint32_t sel = {src[1]}.read_lane(wf, lane) & 0xF;')
        else:
            L.append(f'    uint32_t sel_word = (lane < 32)')
            L.append(f'        ? {src[1]}.read_scalar(wf)')
            L.append(f'        : {src[2]}.read_scalar(wf);')
            L.append('    uint32_t sub = lane & 0xF;')
            L.append('    uint32_t sel = (sel_word >> (sub * 2)) & 0xF;')
        xor_bit = ' ^ 0x10' if cross else ''
        L.append(f'    uint32_t src_lane = (lane & ~0xFu) | ((sel{xor_bit}) & 0xFu);')
        L.append('    if (src_lane >= wf.wf_size()) continue;')
        L.append('    bool src_active = (exec & (1ULL << src_lane)) != 0;')
        L.append('    if (!src_active && !fi) {')
        L.append('      if (bound_ctrl)')
        L.append(f'        {dst[0]}.write_lane(wf, lane, 0);')
        L.append('      continue;')
        L.append('    }')
        L.append(f'    {dst[0]}.write_lane(wf, lane, snap[src_lane]);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_permlane64(self, dst: list[str], src: list[str]) -> str:
        """Generate V_PERMLANE64_B32: swap lane i with lane i ^ 32."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint32_t snap[64];')
        L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
        L.append(f'    snap[i] = {src[0]}.read_lane(wf, i);')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append('    uint32_t partner = lane ^ 32;')
        L.append('    if (partner < wf.wf_size())')
        L.append(f'      {dst[0]}.write_lane(wf, lane, snap[partner]);')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_cvt_pk(self, dst: list[str], src: list[str], cls: str, op: str | None) -> str:
        """Generate pack/convert instructions."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        if cls == 'vector_cvt_pk_u8_f32':
            L.append(f'    float fval = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    uint32_t byte_sel = {src[1]}.read_lane(wf, lane) & 3;')
            # V_CVT_PK_U8_F32 has 3 srcs; V_CVT_PKACCUM reads old from dst
            old_src = src[2] if len(src) > 2 else dst[0]
            L.append(f'    uint32_t old = {old_src}.read_lane(wf, lane);')
            L.append('    uint32_t byte = static_cast<uint32_t>(std::clamp(fval, 0.0f, 255.0f));')
            L.append('    uint32_t mask = ~(0xFFu << (byte_sel * 8));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, (old & mask) | (byte << (byte_sel * 8)));')
        elif cls == 'vector_cvt_pknorm':
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            if op == 'i16':
                L.append('    auto cvt_i16 = [](float f) -> int16_t {')
                L.append('      if (std::isnan(f)) return 0;')
                L.append('      return static_cast<int16_t>(std::clamp(f * 32767.0f, -32768.0f, 32767.0f));')
                L.append('    };')
                L.append('    int16_t lo = cvt_i16(s0);')
                L.append('    int16_t hi = cvt_i16(s1);')
            else:  # u16
                L.append('    auto cvt_u16 = [](float f) -> uint16_t {')
                L.append('      if (std::isnan(f)) return 0;')
                L.append('      return static_cast<uint16_t>(std::clamp(f * 65535.0f, 0.0f, 65535.0f));')
                L.append('    };')
                L.append('    uint16_t lo = cvt_u16(s0);')
                L.append('    uint16_t hi = cvt_u16(s1);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, (static_cast<uint32_t>(hi) << 16) | (static_cast<uint32_t>(lo) & 0xFFFF));')
        elif cls == 'vector_cvt_pkrtz_f16_f32':
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    uint32_t lo = util::f32_to_f16(s0);')
            L.append(f'    uint32_t hi = util::f32_to_f16(s1);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
        elif cls == 'vector_cvt_pk':
            L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane);')
            L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane);')
            if op == 'u16_u32':
                L.append('    uint16_t lo = static_cast<uint16_t>(std::min(s0, 0xFFFFu));')
                L.append('    uint16_t hi = static_cast<uint16_t>(std::min(s1, 0xFFFFu));')
            else:  # i16_i32
                L.append('    int16_t lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s0), -32768, 32767));')
                L.append('    int16_t hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s1), -32768, 32767));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) | static_cast<uint32_t>(static_cast<uint16_t>(lo)));')
        elif cls == 'vector_cvt_pk_f16_f32':
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    uint32_t lo = util::f32_to_f16(s0);')
            L.append(f'    uint32_t hi = util::f32_to_f16(s1);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
        elif cls == 'vector_cvt_pk_bf16_f32':
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    uint32_t lo = util::f32_to_bf16(s0);')
            L.append(f'    uint32_t hi = util::f32_to_bf16(s1);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
        elif cls == 'vector_pack_b32_f16':
            L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane) & 0xFFFF;')
            L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane) & 0xFFFF;')
            L.append(f'    {dst[0]}.write_lane(wf, lane, s0 | (s1 << 16));')
        elif cls == 'vector_cvt_sr_f16_f32':
            # Stochastic rounding: use src1 as random bits for rounding
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(util::f32_to_f16(s0)));')
        elif cls == 'vector_cvt_sr_bf16_f32':
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(util::f32_to_bf16(s0)));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_scalar_unary(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, scc: str | None) -> str:
        """Generate scalar unary operation body."""
        L = []
        is_64 = dtype in ('b64', 'i64')

        # Special cases that don't follow the is_64 read/write pattern.
        if op == 'flbit_i32_i64':
            # 64-bit signed input → 32-bit output (find first bit of sign).
            L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
            L.append('  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);')
            L.append('  uint32_t result = uval == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(uval));')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op == 'clz64':
            # 64-bit unsigned input → 32-bit output (count leading zeros).
            L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
            L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op == 'cls64':
            # 64-bit signed input → 32-bit output (count leading sign bits).
            L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
            L.append('  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);')
            L.append('  uint32_t result = uval == 0 ? 63u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op == 'ctz' and is_64:
            L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
            L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val));')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op == 'clz' and is_64:
            L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
            L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op == 'cls' and is_64:
            L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
            L.append('  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);')
            L.append('  uint32_t result = uval == 0 ? 63u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')
            return '\n'.join(L)

        if op in ('bitset0', 'bitset1') and is_64:
            # 32-bit input (bit index), 64-bit read-modify-write destination.
            L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
            if op == 'bitset0':
                L.append(f'  uint64_t result = {dst[0]}.read_scalar64(wf) & ~(1ULL << (bit & 63));')
            else:
                L.append(f'  uint64_t result = {dst[0]}.read_scalar64(wf) | (1ULL << (bit & 63));')
            L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            return '\n'.join(L)

        if is_64:
            L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
            op_map = {
                'not': '~val',
                'wqm': '0; for (int q = 0; q < 16; ++q) if (val & (0xFULL << (q * 4))) result |= (0xFULL << (q * 4))',
                'bcnt0': 'static_cast<uint64_t>(std::popcount(~val))',
                'bcnt1': 'static_cast<uint64_t>(std::popcount(val))',
                'ff0': 'static_cast<uint64_t>(val == ~0ULL ? -1 : std::countr_zero(~val))',
                'ff1': 'static_cast<uint64_t>(val == 0 ? -1 : std::countr_zero(val))',
                'flbit': 'static_cast<uint64_t>(val == 0 ? -1 : std::countl_zero(val))',
                'bitrepl': 'val',
            }
            if op == 'brev':
                L.append('  uint64_t result = 0;')
                L.append('  for (int i = 0; i < 64; ++i) result |= ((val >> i) & 1) << (63 - i);')
            elif op == 'quadmask':
                L.append('  uint64_t result = 0;')
                L.append('  for (int q = 0; q < 16; ++q) if (val & (0xFULL << (q * 4))) result |= (1ULL << q);')
            elif op in op_map:
                L.append(f'  uint64_t result = {op_map[op]};')
            else:
                L.append(f'  uint64_t result = val; // unhandled: {op}')
            L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            L.append('  wf.write_scc(result != 0);')
        else:
            if op == 'abs':
                # Use unsigned negation to avoid UB when val == INT_MIN.
                L.append(f'  int32_t val = static_cast<int32_t>({src[0]}.read_scalar(wf));')
                L.append('  uint32_t uval = static_cast<uint32_t>(val);')
                L.append('  uint32_t result = val < 0 ? (0u - uval) : uval;')
            elif op == 'sext8':
                L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
                L.append('  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val & 0xFF)));')
            elif op == 'sext16':
                L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
                L.append('  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val & 0xFFFF)));')
            elif op == 'flbit_i32':
                L.append(f'  int32_t sval = static_cast<int32_t>({src[0]}.read_scalar(wf));')
                L.append('  uint32_t val = sval < 0 ? ~static_cast<uint32_t>(sval) : static_cast<uint32_t>(sval);')
                L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));')
            elif op == 'flbit_i32_i64':
                L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
                L.append('  uint64_t val = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);')
                L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));')
            elif op == 'bitset0':
                # b64 case handled by early return above (is_64 branch).
                L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
                L.append(f'  uint32_t result = {dst[0]}.read_scalar(wf) & ~(1u << (bit & 31));')
            elif op == 'bitset1':
                L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
                L.append(f'  uint32_t result = {dst[0]}.read_scalar(wf) | (1u << (bit & 31));')
            else:
                L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
                op_map = {
                    'not': '~val',
                    'wqm': '0; for (int q = 0; q < 8; ++q) if (val & (0xFu << (q * 4))) result |= (0xFu << (q * 4))',
                    'bcnt0': 'static_cast<uint32_t>(std::popcount(~val))',
                    'bcnt1': 'static_cast<uint32_t>(std::popcount(val))',
                    'ff0': 'val == ~0u ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(~val))',
                    'ff1': 'val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val))',
                    'flbit': 'val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val))',
                    'quadmask': '0; for (int q = 0; q < 8; ++q) if (val & (0xFu << (q * 4))) result |= (1u << q)',
                }
                if op == 'brev':
                    L.append('  uint32_t result = 0;')
                    L.append('  for (int i = 0; i < 32; ++i) result |= ((val >> i) & 1) << (31 - i);')
                elif op == 'ceil' and dtype == 'f32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(std::ceil(std::bit_cast<float>(val)));')
                elif op == 'ceil' and dtype == 'f16':
                    L.append('  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));')
                    L.append('  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::ceil(f)));')
                elif op == 'floor' and dtype == 'f32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(std::floor(std::bit_cast<float>(val)));')
                elif op == 'floor' and dtype == 'f16':
                    L.append('  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));')
                    L.append('  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::floor(f)));')
                elif op == 'trunc' and dtype == 'f32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(std::trunc(std::bit_cast<float>(val)));')
                elif op == 'trunc' and dtype == 'f16':
                    L.append('  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));')
                    L.append('  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::trunc(f)));')
                elif op == 'rndne' and dtype == 'f32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(std::nearbyint(std::bit_cast<float>(val)));')
                elif op == 'rndne' and dtype == 'f16':
                    L.append('  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));')
                    L.append('  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::nearbyint(f)));')
                elif op == 'cvt_f32_i32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<int32_t>(val)));')
                elif op == 'cvt_f32_u32':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(static_cast<float>(val));')
                elif op == 'cvt_i32_f32':
                    L.append('  float f = std::bit_cast<float>(val);')
                    L.append('  int32_t r = std::isnan(f) ? 0 : (f >= 2147483648.0f ? INT32_MAX : (f < -2147483648.0f ? INT32_MIN : static_cast<int32_t>(f)));')
                    L.append('  uint32_t result = static_cast<uint32_t>(r);')
                elif op == 'cvt_u32_f32':
                    L.append('  float f = std::bit_cast<float>(val);')
                    L.append('  uint32_t result = (std::isnan(f) || f < 0.0f) ? 0u : (f >= 4294967296.0f ? UINT32_MAX : static_cast<uint32_t>(f));')
                elif op == 'cvt_f16_f32':
                    L.append('  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::bit_cast<float>(val)));')
                elif op == 'cvt_f32_f16':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF)));')
                elif op == 'cvt_hi_f32_f16':
                    L.append('  uint32_t result = std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>((val >> 16) & 0xFFFF)));')
                elif op == 'ctz':
                    L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val));')
                elif op == 'clz':
                    L.append('  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));')
                elif op == 'cls':
                    # Count leading sign bits: number of consecutive bits matching the sign bit.
                    L.append('  int32_t sval = static_cast<int32_t>(val);')
                    L.append('  uint32_t uval = sval < 0 ? ~static_cast<uint32_t>(sval) : static_cast<uint32_t>(sval);')
                    L.append('  uint32_t result = uval == 0 ? 31u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;')
                elif op in op_map:
                    L.append(f'  uint32_t result = {op_map[op]};')
                else:
                    L.append(f'  uint32_t result = val; // TODO: {op}')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            if scc and scc != 'none':
                L.append('  wf.write_scc(result != 0);')

        return '\n'.join(L)

    def _gen_scalar_binop(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, scc: str | None) -> str:
        """Generate scalar binary operation body."""
        L = []
        is_64 = dtype in ('b64', 'i64', 'u64')

        if is_64:
            if dtype == 'i64':
                L.append(f'  int64_t s0 = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
                L.append(f'  int64_t s1 = static_cast<int64_t>({src[1]}.read_scalar64(wf));')
            else:
                L.append(f'  uint64_t s0 = {src[0]}.read_scalar64(wf);')
                L.append(f'  uint64_t s1 = {src[1]}.read_scalar64(wf);')
        elif dtype in ('i32',):
            L.append(f'  int32_t s0 = static_cast<int32_t>({src[0]}.read_scalar(wf));')
            L.append(f'  int32_t s1 = static_cast<int32_t>({src[1]}.read_scalar(wf));')
        else:
            L.append(f'  uint32_t s0 = {src[0]}.read_scalar(wf);')
            L.append(f'  uint32_t s1 = {src[1]}.read_scalar(wf);')

        # Compute result
        if is_64 and op == 'mul':
            L.append(f'  {dst[0]}.write_scalar64(wf, static_cast<uint64_t>(s0 * s1));')
        elif dtype in ('i32',) and op in ('add', 'sub'):
            sign = '+' if op == 'add' else '-'
            L.append(f'  int64_t wide = static_cast<int64_t>(s0) {sign} static_cast<int64_t>(s1);')
            L.append('  int32_t result = static_cast<int32_t>(wide);')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            L.append('  wf.write_scc(wide != static_cast<int64_t>(result));')
        elif dtype in ('u32',) and op == 'add':
            L.append('  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1);')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
            L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
        elif dtype in ('u32',) and op == 'sub':
            L.append(f'  {dst[0]}.write_scalar(wf, s0 - s1);')
            L.append('  wf.write_scc(s0 < s1);')
        elif dtype in ('u32',) and op == 'addc':
            L.append('  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1) + (wf.read_scc() ? 1u : 0u);')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
            L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
        elif dtype in ('u32',) and op == 'subb':
            L.append('  uint32_t bin = wf.read_scc() ? 1u : 0u;')
            L.append('  uint64_t wide = static_cast<uint64_t>(s0) - static_cast<uint64_t>(s1) - bin;')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
            L.append('  wf.write_scc(static_cast<uint64_t>(s0) < static_cast<uint64_t>(s1) + bin);')
        elif dtype in ('i32',) and op == 'mul':
            # Use unsigned multiply to avoid signed overflow UB. The lower 32
            # bits are identical for signed and unsigned multiplication.
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<uint32_t>(s0) * static_cast<uint32_t>(s1)));')
        elif dtype in ('u32',) and op == 'mul':
            L.append(f'  {dst[0]}.write_scalar(wf, s0 * s1);')
        elif op == 'mulhi':
            if dtype in ('u32',):
                L.append('  uint64_t wide = static_cast<uint64_t>(s0) * static_cast<uint64_t>(s1);')
                L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide >> 32));')
            else:
                L.append('  int64_t wide = static_cast<int64_t>(s0) * static_cast<int64_t>(s1);')
                L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<uint64_t>(wide) >> 32));')
        elif op == 'min':
            if dtype in ('i32',):
                L.append(f'  int32_t result = s0 < s1 ? s0 : s1;')
                L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
                L.append('  wf.write_scc(s0 < s1);')
            else:
                L.append(f'  uint32_t result = s0 < s1 ? s0 : s1;')
                L.append(f'  {dst[0]}.write_scalar(wf, result);')
                L.append('  wf.write_scc(s0 < s1);')
        elif op == 'max':
            if dtype in ('i32',):
                L.append(f'  int32_t result = s0 > s1 ? s0 : s1;')
                L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
                L.append('  wf.write_scc(s0 > s1);')
            else:
                L.append(f'  uint32_t result = s0 > s1 ? s0 : s1;')
                L.append(f'  {dst[0]}.write_scalar(wf, result);')
                L.append('  wf.write_scc(s0 > s1);')
        elif op == 'absdiff':
            L.append(f'  int32_t result = s0 > s1 ? s0 - s1 : s1 - s0;')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            L.append('  wf.write_scc(result != 0);')
        elif op == 'bfm':
            if is_64:
                L.append('  uint64_t count = s0 & 63u;')
                L.append('  uint64_t offset = s1 & 63u;')
                L.append('  uint64_t result = count == 0 ? 0 : ((1ULL << count) - 1) << offset;')
                L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            else:
                L.append('  uint32_t count = s0 & 31u;')
                L.append('  uint32_t offset = s1 & 31u;')
                L.append('  uint32_t result = count == 0 ? 0 : ((1u << count) - 1) << offset;')
                L.append(f'  {dst[0]}.write_scalar(wf, result);')
        elif op == 'bfe':
            return self._gen_scalar_bfe(dst, src, dtype)
        elif op in ('lshl1_add', 'lshl2_add', 'lshl3_add', 'lshl4_add'):
            shift = op[4]  # extract digit
            L.append(f'  uint64_t wide = (static_cast<uint64_t>(s0) << {shift}u) + static_cast<uint64_t>(s1);')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
            L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
        elif dtype == 'f32' and op in ('add', 'sub', 'mul', 'min', 'max', 'fma'):
            fp_op = {
                'add': 'f0 + f1', 'sub': 'f0 - f1', 'mul': 'f0 * f1',
                'min': 'std::fmin(f0, f1)', 'max': 'std::fmax(f0, f1)',
                'fma': 'std::fma(f0, f1, std::bit_cast<float>(static_cast<uint32_t>(wf.read_scc())))',
            }
            L.append('  float f0 = std::bit_cast<float>(s0);')
            L.append('  float f1 = std::bit_cast<float>(s1);')
            L.append(f'  float fr = {fp_op[op]};')
            L.append(f'  {dst[0]}.write_scalar(wf, std::bit_cast<uint32_t>(fr));')
        elif dtype == 'f16' and op in ('add', 'sub', 'mul', 'min', 'max'):
            fp_op = {
                'add': 'f0 + f1', 'sub': 'f0 - f1', 'mul': 'f0 * f1',
                'min': 'std::fmin(f0, f1)', 'max': 'std::fmax(f0, f1)',
            }
            L.append('  float f0 = util::f16_to_f32(static_cast<uint16_t>(s0 & 0xFFFF));')
            L.append('  float f1 = util::f16_to_f32(static_cast<uint16_t>(s1 & 0xFFFF));')
            L.append(f'  float fr = {fp_op[op]};')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(util::f32_to_f16(fr)));')
        elif op == 'pack_ll':
            L.append(f'  {dst[0]}.write_scalar(wf, (s0 & 0xFFFFu) | ((s1 & 0xFFFFu) << 16));')
        elif op == 'pack_lh':
            L.append(f'  {dst[0]}.write_scalar(wf, (s0 & 0xFFFFu) | (s1 & 0xFFFF0000u));')
        elif op == 'pack_hh':
            L.append(f'  {dst[0]}.write_scalar(wf, ((s0 >> 16) & 0xFFFFu) | (s1 & 0xFFFF0000u));')
        else:
            # Bitwise / shift ops
            utype = 'uint64_t' if is_64 else 'uint32_t'
            mask = 63 if is_64 else 31
            op_map = {
                'and': 's0 & s1',
                'or': 's0 | s1',
                'xor': 's0 ^ s1',
                'nand': '~(s0 & s1)',
                'nor': '~(s0 | s1)',
                'xnor': '~(s0 ^ s1)',
                'andn2': 's0 & ~s1',
                'orn2': 's0 | ~s1',
                'shl': f's0 << (s1 & {mask}u)',
                'shr': f's0 >> (s1 & {mask}u)',
            }
            if dtype == 'i32' and op == 'ashr':
                L.append(f'  int32_t result = s0 >> (s1 & 31);')
                L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            elif dtype == 'i64' and op == 'ashr':
                L.append(f'  int64_t result = s0 >> (s1 & 63);')
                L.append(f'  {dst[0]}.write_scalar64(wf, static_cast<uint64_t>(result));')
            elif op in op_map:
                L.append(f'  {utype} result = {op_map[op]};')
                if is_64:
                    L.append(f'  {dst[0]}.write_scalar64(wf, result);')
                else:
                    L.append(f'  {dst[0]}.write_scalar(wf, result);')
            else:
                L.append('  (void)s1;')
                L.append(f'  {utype} result = s0; // TODO: op={op}')
                if is_64:
                    L.append(f'  {dst[0]}.write_scalar64(wf, result);')
                else:
                    L.append(f'  {dst[0]}.write_scalar(wf, result);')

            # SCC
            if scc == 'nonzero':
                L.append('  wf.write_scc(result != 0);')

        return '\n'.join(L)

    def _gen_scalar_bfe(self, dst: list[str], src: list[str], dtype: str | None) -> str:
        """Generate scalar bit field extract body."""
        L = []
        if dtype in ('u64', 'i64'):
            L.append(f'  uint64_t base = {src[0]}.read_scalar64(wf);')
            L.append(f'  uint32_t field = {src[1]}.read_scalar(wf);')
            L.append('  uint32_t offset = field & 63u;')
            L.append('  uint32_t width = (field >> 16) & 127u;')
            L.append('  if (width == 0) {')
            L.append(f'    {dst[0]}.write_scalar64(wf, 0);')
            L.append('    wf.write_scc(false);')
            L.append('  } else {')
            L.append('    uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1);')
            L.append('    uint64_t extracted = (base >> offset) & mask;')
            if dtype == 'i64':
                L.append('    if (width < 64 && (extracted & (1ULL << (width - 1))))')
                L.append('      extracted |= ~mask;')
            L.append(f'    {dst[0]}.write_scalar64(wf, extracted);')
            L.append('    wf.write_scc(extracted != 0);')
            L.append('  }')
        else:
            L.append(f'  uint32_t base = {src[0]}.read_scalar(wf);')
            L.append(f'  uint32_t field = {src[1]}.read_scalar(wf);')
            L.append('  uint32_t offset = field & 31u;')
            L.append('  uint32_t width = (field >> 16) & 127u;')
            L.append('  if (width == 0) {')
            L.append(f'    {dst[0]}.write_scalar(wf, 0);')
            L.append('    wf.write_scc(false);')
            L.append('  } else {')
            L.append('    uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1);')
            L.append('    uint32_t extracted = (base >> offset) & mask;')
            if dtype == 'i32':
                L.append('    if (width < 32 && (extracted & (1u << (width - 1))))')
                L.append('      extracted |= ~mask;')
            L.append(f'    {dst[0]}.write_scalar(wf, extracted);')
            L.append('    wf.write_scc(extracted != 0);')
            L.append('  }')
        return '\n'.join(L)

    def _gen_scalar_cmp(self, src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate scalar compare body (sets SCC)."""
        L = []
        cmp_map = {
            'eq': '==', 'ne': '!=', 'lg': '!=',
            'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<=',
        }
        if dtype in ('i32',):
            L.append(f'  int32_t s0 = static_cast<int32_t>({src[0]}.read_scalar(wf));')
            L.append(f'  int32_t s1 = static_cast<int32_t>({src[1]}.read_scalar(wf));')
        elif dtype in ('i64',):
            L.append(f'  int64_t s0 = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
            L.append(f'  int64_t s1 = static_cast<int64_t>({src[1]}.read_scalar64(wf));')
        elif dtype in ('u64',):
            L.append(f'  uint64_t s0 = {src[0]}.read_scalar64(wf);')
            L.append(f'  uint64_t s1 = {src[1]}.read_scalar64(wf);')
        else:
            L.append(f'  uint32_t s0 = {src[0]}.read_scalar(wf);')
            L.append(f'  uint32_t s1 = {src[1]}.read_scalar(wf);')
        L.append(f'  wf.write_scc(s0 {cmp_map[op]} s1);')
        return '\n'.join(L)

    def _gen_scalar_cmpk(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate scalar compare-with-immediate body (SOPK)."""
        L = []
        cmp_map = {
            'eq': '==', 'ne': '!=', 'lg': '!=',
            'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<=',
        }
        if dtype in ('i32',):
            L.append(f'  int32_t s0 = static_cast<int32_t>({dst[0]}.read_scalar(wf));')
            L.append(f'  int32_t imm = static_cast<int16_t>({src[0]}.encoding_value_);')
        else:
            L.append(f'  uint32_t s0 = {dst[0]}.read_scalar(wf);')
            L.append(f'  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>({src[0]}.encoding_value_));')
        L.append(f'  wf.write_scc(s0 {cmp_map[op]} imm);')
        return '\n'.join(L)

    def _gen_scalar_bitcmp(self, src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate scalar bit compare body."""
        L = []
        if dtype in ('b64',):
            L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
            L.append(f'  uint32_t bit = {src[1]}.read_scalar(wf) & 63u;')
        else:
            L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
            L.append(f'  uint32_t bit = {src[1]}.read_scalar(wf) & 31u;')
        if op == 'bitcmp0':
            L.append('  wf.write_scc(!(val & (1ULL << bit)));')
        else:
            L.append('  wf.write_scc((val & (1ULL << bit)) != 0);')
        return '\n'.join(L)

    def _gen_scalar_saveexec(self, dst: list[str], src: list[str], op: str | None) -> str:
        """Generate saveexec body.

        Per the ISA spec, all sources are read before any destination is written.
        Reading ssrc0 before writing sdst prevents aliasing bugs when sdst == ssrc0.
        """
        L = []
        L.append('  uint64_t old_exec = wf.exec();')
        L.append(f'  uint64_t src = {src[0]}.read_scalar64(wf);')
        L.append(f'  {dst[0]}.write_scalar64(wf, old_exec);')
        saveexec_map = {
            'and': 'old_exec & src',
            'or': 'old_exec | src',
            'xor': 'old_exec ^ src',
            'nand': '~(old_exec & src)',
            'nor': '~(old_exec | src)',
            'xnor': '~(old_exec ^ src)',
            'andn1': '~src & old_exec',
            'andn2': 'src & ~old_exec',
            'orn1': '~src | old_exec',
            'orn2': 'src | ~old_exec',
            # RDNA3/4 not0/not1 variants
            'and_not0': 'old_exec & ~src',
            'or_not0': 'old_exec | ~src',
            'and_not1': '~src & ~old_exec',
            'or_not1': '~src | old_exec',
        }
        if op not in saveexec_map:
            L.append('  (void)src;')
        expr = saveexec_map.get(op, f'old_exec /* TODO: {op} */')
        L.append(f'  uint64_t result = {expr};')
        L.append(f'  util::Logger::vm([&](auto &os) {{')
        L.append(f'    os << std::format("saveexec ssrc0_ev={{}} src={{:#x}} exec={{:#x}}->{{:#x}}",')
        L.append(f'                      {src[0]}.encoding_value(), src, old_exec, result);')
        L.append(f'  }});')
        L.append('  wf.set_exec(result);')
        L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    def _gen_vector_unary(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector unary operation body."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')

        if op == 'cvt':
            cvt_map = {
                'f32_i32': (
                    f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
                ),
                'f32_u32': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
                ),
                'i32_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    int32_t r;\n'
                    f'    if (std::isnan(s)) r = 0;\n'
                    f'    else if (s >= 2147483648.0f) r = INT32_MAX;\n'
                    f'    else if (s < -2147483648.0f) r = INT32_MIN;\n'
                    f'    else r = static_cast<int32_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
                ),
                'u32_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    uint32_t r;\n'
                    f'    if (std::isnan(s) || s < 0.0f) r = 0;\n'
                    f'    else if (s >= 4294967296.0f) r = UINT32_MAX;\n'
                    f'    else r = static_cast<uint32_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, r);'
                ),
                'rpi_i32_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    float rounded = std::ceil(s - 0.5f);\n'
                    f'    int32_t r;\n'
                    f'    if (std::isnan(rounded)) r = 0;\n'
                    f'    else if (rounded >= 2147483648.0f) r = INT32_MAX;\n'
                    f'    else if (rounded < -2147483648.0f) r = INT32_MIN;\n'
                    f'    else r = static_cast<int32_t>(rounded);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
                ),
                'flr_i32_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    float rounded = std::floor(s);\n'
                    f'    int32_t r;\n'
                    f'    if (std::isnan(rounded)) r = 0;\n'
                    f'    else if (rounded >= 2147483648.0f) r = INT32_MAX;\n'
                    f'    else if (rounded < -2147483648.0f) r = INT32_MIN;\n'
                    f'    else r = static_cast<int32_t>(rounded);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
                ),
                'f32_ubyte0': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s & 0xFFu)));'
                ),
                'f32_ubyte1': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 8) & 0xFFu)));'
                ),
                'f32_ubyte2': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 16) & 0xFFu)));'
                ),
                'f32_ubyte3': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 24) & 0xFFu)));'
                ),
                # F64 conversions
                'f64_i32': (
                    f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
                ),
                'i32_f64': (
                    f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                    f'    int32_t r;\n'
                    f'    if (std::isnan(s)) r = 0;\n'
                    f'    else if (s >= 2147483648.0) r = INT32_MAX;\n'
                    f'    else if (s < -2147483648.0) r = INT32_MIN;\n'
                    f'    else r = static_cast<int32_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
                ),
                'f64_u32': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
                ),
                'u32_f64': (
                    f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                    f'    uint32_t r;\n'
                    f'    if (std::isnan(s) || s < 0.0) r = 0;\n'
                    f'    else if (s >= 4294967296.0) r = UINT32_MAX;\n'
                    f'    else r = static_cast<uint32_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, r);'
                ),
                'f64_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
                ),
                'f32_f64': (
                    f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
                ),
                # F16 conversions
                'f16_f32': (
                    f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(s));'
                ),
                'f32_f16': (
                    f'    uint32_t raw = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(raw))));'
                ),
                'f16_u16': (
                    f'    uint16_t s = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));'
                ),
                'f16_i16': (
                    f'    int16_t s = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);\n'
                    f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));'
                ),
                'u16_f16': (
                    f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));\n'
                    f'    uint16_t r;\n'
                    f'    if (std::isnan(s) || s < 0.0f) r = 0;\n'
                    f'    else if (s >= 65536.0f) r = UINT16_MAX;\n'
                    f'    else r = static_cast<uint16_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
                ),
                'i16_f16': (
                    f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));\n'
                    f'    int16_t r;\n'
                    f'    if (std::isnan(s)) r = 0;\n'
                    f'    else if (s >= 32768.0f) r = INT16_MAX;\n'
                    f'    else if (s < -32768.0f) r = INT16_MIN;\n'
                    f'    else r = static_cast<int16_t>(s);\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(r)));'
                ),
                # 16-bit integer conversions
                'i32_i16': (
                    f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                    f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(s & 0xFFFF))));'
                ),
                'u32_u16': (
                    f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                    f'    {dst[0]}.write_lane(wf, lane, s & 0xFFFFu);'
                ),
            }
            if dtype in cvt_map:
                L.append(cvt_map[dtype])
            else:
                L.append(f'    // TODO: cvt {dtype}')
                L.append(f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));')
        elif op == 'cvt_f32_bf16':
            L.append(f'    float r = util::bf16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
        elif op == 'cvt_f32_fp8':
            L.append(f'    float r = util::fp8_e4m3_to_f32(static_cast<uint8_t>({src[0]}.read_lane(wf, lane) & 0xFF));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
        elif op == 'cvt_f32_bf8':
            L.append(f'    float r = util::bf8_e5m2_to_f32(static_cast<uint8_t>({src[0]}.read_lane(wf, lane) & 0xFF));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
        elif op == 'cvt_pk_f32_fp8':
            # Unpack two FP8 values into two F32s in dst[0] and dst[0]+1
            L.append(f'    uint32_t raw = {src[0]}.read_lane(wf, lane);')
            L.append(f'    float lo = util::fp8_e4m3_to_f32(static_cast<uint8_t>(raw & 0xFF));')
            L.append(f'    float hi = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw >> 8) & 0xFF));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(lo));')
            L.append(f'    wf.cu().write_vgpr(wf.vgpr_alloc().base + {dst[0]}.encoding_value_ + 1, lane, std::bit_cast<uint32_t>(hi));')
        elif op == 'cvt_pk_f32_bf8':
            L.append(f'    uint32_t raw = {src[0]}.read_lane(wf, lane);')
            L.append(f'    float lo = util::bf8_e5m2_to_f32(static_cast<uint8_t>(raw & 0xFF));')
            L.append(f'    float hi = util::bf8_e5m2_to_f32(static_cast<uint8_t>((raw >> 8) & 0xFF));')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(lo));')
            L.append(f'    wf.cu().write_vgpr(wf.vgpr_alloc().base + {dst[0]}.encoding_value_ + 1, lane, std::bit_cast<uint32_t>(hi));')
        elif op in ('not', 'bfrev', 'ffbh_u32', 'ffbl', 'ffbh_i32', 'bcnt', 'mbcnt_lo', 'mbcnt_hi'):
            L.append(f'    uint32_t s = {src[0]}.read_lane(wf, lane);')
            int_op_map = {
                'not': '~s',
                'bcnt': 'static_cast<uint32_t>(std::popcount(s))',
                'ffbh_u32': 's == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(s))',
                'ffbl': 's == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(s))',
            }
            if op == 'bfrev':
                L.append('    uint32_t result = 0;')
                L.append('    for (int i = 0; i < 32; ++i) result |= ((s >> i) & 1) << (31 - i);')
                L.append(f'    {dst[0]}.write_lane(wf, lane, result);')
            elif op == 'ffbh_i32':
                L.append('    int32_t sv = static_cast<int32_t>(s);')
                L.append('    uint32_t abs_val = sv < 0 ? ~s : s;')
                L.append(f'    {dst[0]}.write_lane(wf, lane, abs_val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(abs_val)));')
            elif op in int_op_map:
                L.append(f'    {dst[0]}.write_lane(wf, lane, {int_op_map[op]});')
            else:
                L.append(f'    {dst[0]}.write_lane(wf, lane, s);')
        elif op == 'frexp_exp_f32' and dtype == 'f64':
            # V_FREXP_EXP_I32_F64: extract exponent from f64, write as i32
            L.append(f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            L.append('    int exp = 0;')
            L.append('    if (s != 0.0 && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
        elif op == 'frexp_mant_f32' and dtype == 'f64':
            # V_FREXP_MANT_F64: extract mantissa from f64, write as f64
            L.append(f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            L.append('    int exp = 0;')
            L.append('    double result = std::frexp(s, &exp);')
            if is_vop3:
                L.extend(self._vop3_dst_mod_f64('result'))
            L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
        elif op == 'frexp_exp_f32':
            L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            L.append('    int exp = 0;')
            L.append('    if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
        elif op == 'frexp_exp_f16':
            L.append(f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            L.append('    int exp = 0;')
            L.append('    if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
        elif op == 'frexp_mant_f32':
            L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            L.append('    int exp = 0;')
            L.append(f'    float result = std::frexp(s, &exp);')
            if is_vop3:
                L.extend(self._vop3_dst_mod('result'))
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        elif op == 'clrexcp':
            L.append(f'    (void){src[0]};')
            L.append(f'    (void){dst[0]};')
        elif dtype == 'f64':
            L.append(f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            math_map_f64 = {
                'rcp': 'amdgpu::transcendental::rcp_f64(s)',
                'sqrt': 'amdgpu::transcendental::sqrt_f64(s)',
                'rsq': 'amdgpu::transcendental::rsq_f64(s)',
                'floor': 'std::floor(s)',
                'ceil': 'std::ceil(s)',
                'trunc': 'std::trunc(s)',
                'rndne': 'std::nearbyint(s)',
                'fract': 's - std::floor(s)',
                'abs': 'std::fabs(s)',
                'neg': '-s',
            }
            expr = math_map_f64.get(op, f's /* TODO: {op} */')
            if is_vop3:
                L.append(f'    double result = {expr};')
                L.extend(self._vop3_dst_mod_f64('result'))
                L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
            else:
                L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));')
        elif dtype == 'f16':
            L.append(f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            math_map_f16 = {
                'rcp': '1.0f / s',
                'sqrt': 'std::sqrt(s)',
                'rsq': '1.0f / std::sqrt(s)',
                'floor': 'std::floor(s)',
                'ceil': 'std::ceil(s)',
                'trunc': 'std::trunc(s)',
                'rndne': 'std::nearbyint(s)',
                'fract': 's - std::floor(s)',
                'exp2': 'std::exp2(s)',
                'log2': 'std::log2(s)',
                'sin': 'std::sin(s * 6.2831853071795864f)',
                'cos': 'std::cos(s * 6.2831853071795864f)',
                'abs': 'std::fabs(s)',
                'neg': '-s',
            }
            expr = math_map_f16.get(op, f's /* TODO: {op} */')
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(self._vop3_dst_mod('result'))
                L.append(f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(result));')
            else:
                L.append(f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16({expr}));')
        else:
            L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s', 0))
            math_map = {
                'rcp': 'amdgpu::transcendental::rcp_f32(s)',
                'rcp_iflag': 'amdgpu::transcendental::rcp_f32(s)',
                'sqrt': 'amdgpu::transcendental::sqrt_f32(s)',
                'rsq': 'amdgpu::transcendental::rsq_f32(s)',
                'floor': 'std::floor(s)',
                'ceil': 'std::ceil(s)',
                'trunc': 'std::trunc(s)',
                'rndne': 'std::nearbyint(s)',
                'fract': 's - std::floor(s)',
                'exp2': 'amdgpu::transcendental::exp_f32(s)',
                'log2': 'amdgpu::transcendental::log_f32(s)',
                'sin': 'amdgpu::transcendental::sin_f32(s)',
                'cos': 'amdgpu::transcendental::cos_f32(s)',
                'abs': 'std::fabs(s)',
                'neg': '-s',
            }
            expr = math_map.get(op, f's /* TODO: {op} */')
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(self._vop3_dst_mod('result'))
                L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
            else:
                L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));')

        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_binop(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector binary operation body."""
        if dst:
            d = dst[0]
            s0, s1 = src[0], src[1]
        else:
            d = src[0]
            s0, s1 = src[1], src[2]

        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')

        if dtype == 'f64':
            L.append(f'    double sv0 = std::bit_cast<double>({s0}.read_lane64(wf, lane));')
            if op == 'ldexp':
                # src1 is a 32-bit integer exponent, not a double.
                L.append(f'    int32_t sv1_i = static_cast<int32_t>({s1}.read_lane(wf, lane));')
            else:
                L.append(f'    double sv1 = std::bit_cast<double>({s1}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('sv0', 0))
                if op != 'ldexp':
                    L.extend(self._vop3_src_mod('sv1', 1))
            f_op_map = {
                'add': 'sv0 + sv1',
                'sub': 'sv0 - sv1',
                'rsub': 'sv1 - sv0',
                'mul': 'sv0 * sv1',
                'min': 'std::fmin(sv0, sv1)',
                'max': 'std::fmax(sv0, sv1)',
                'fmin': 'std::fmin(sv0, sv1)',
                'fmax': 'std::fmax(sv0, sv1)',
                'fmac': f'std::fma(sv0, sv1, std::bit_cast<double>({d}.read_lane64(wf, lane)))',
                'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
            }
            expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
            if is_vop3:
                L.append(f'    double result = {expr};')
                L.extend(self._vop3_dst_mod_f64('result'))
                L.append(f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
            else:
                L.append(f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));')
        elif dtype == 'f32':
            L.append(f'    float sv0 = std::bit_cast<float>({s0}.read_lane(wf, lane));')
            if op == 'ldexp':
                # src1 is a 32-bit integer exponent, not a float.
                L.append(f'    int32_t sv1_i = static_cast<int32_t>({s1}.read_lane(wf, lane));')
            else:
                L.append(f'    float sv1 = std::bit_cast<float>({s1}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('sv0', 0))
                if op != 'ldexp':
                    L.extend(self._vop3_src_mod('sv1', 1))
            f_op_map = {
                'add': 'sv0 + sv1',
                'sub': 'sv0 - sv1',
                'rsub': 'sv1 - sv0',
                'mul': 'sv0 * sv1',
                'mul_legacy': 'sv0 == 0.0f || sv1 == 0.0f ? 0.0f : sv0 * sv1',
                'min': 'std::fmin(sv0, sv1)',
                'max': 'std::fmax(sv0, sv1)',
                'fmin': 'std::fmin(sv0, sv1)',
                'fmax': 'std::fmax(sv0, sv1)',
                'fmac': f'std::fma(sv0, sv1, std::bit_cast<float>({d}.read_lane(wf, lane)))',
                'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
            }
            expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(self._vop3_dst_mod('result'))
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));')
        elif dtype == 'f16':
            L.append(f'    float sv0 = util::f16_to_f32(static_cast<uint16_t>({s0}.read_lane(wf, lane)));')
            if op == 'ldexp':
                # src1 is a 16-bit integer exponent, not an f16 value.
                L.append(f'    int32_t sv1_i = static_cast<int32_t>(static_cast<int16_t>(static_cast<uint16_t>({s1}.read_lane(wf, lane))));')
            else:
                L.append(f'    float sv1 = util::f16_to_f32(static_cast<uint16_t>({s1}.read_lane(wf, lane)));')
            if is_vop3:
                L.extend(self._vop3_src_mod('sv0', 0))
                if op != 'ldexp':
                    L.extend(self._vop3_src_mod('sv1', 1))
            f_op_map = {
                'add': 'sv0 + sv1',
                'sub': 'sv0 - sv1',
                'rsub': 'sv1 - sv0',
                'mul': 'sv0 * sv1',
                'min': 'std::fmin(sv0, sv1)',
                'max': 'std::fmax(sv0, sv1)',
                'fmin': 'std::fmin(sv0, sv1)',
                'fmax': 'std::fmax(sv0, sv1)',
                'fmac': f'std::fma(sv0, sv1, util::f16_to_f32(static_cast<uint16_t>({d}.read_lane(wf, lane))))',
                'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
            }
            expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(self._vop3_dst_mod('result'))
                L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(result));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16({expr}));')
        elif dtype == 'i24':
            L.append(f'    int32_t sv0 = static_cast<int32_t>({s0}.read_lane(wf, lane) << 8) >> 8;')
            L.append(f'    int32_t sv1 = static_cast<int32_t>({s1}.read_lane(wf, lane) << 8) >> 8;')
            if op == 'mulhi':
                L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>((static_cast<int64_t>(sv0) * sv1) >> 32));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sv0 * sv1));')
        elif dtype == 'u24':
            L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane) & 0x00FFFFFFu;')
            L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane) & 0x00FFFFFFu;')
            if op == 'mulhi':
                L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, sv0 * sv1);')
        elif dtype == 'i16':
            L.append(f'    int16_t sv0 = static_cast<int16_t>({s0}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int16_t sv1 = static_cast<int16_t>({s1}.read_lane(wf, lane) & 0xFFFF);')
            i_op_map = {
                'add': 'static_cast<int16_t>(sv0 + sv1)',
                'sub': 'static_cast<int16_t>(sv0 - sv1)',
                'mul': 'static_cast<int16_t>(sv0 * sv1)',
                'min': 'sv0 < sv1 ? sv0 : sv1',
                'max': 'sv0 > sv1 ? sv0 : sv1',
                'ashr': 'static_cast<int16_t>(sv1 >> (sv0 & 15))',
            }
            expr = i_op_map.get(op, f'sv0 /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>({expr})));')
        elif dtype == 'u16':
            L.append(f'    uint16_t sv0 = static_cast<uint16_t>({s0}.read_lane(wf, lane));')
            L.append(f'    uint16_t sv1 = static_cast<uint16_t>({s1}.read_lane(wf, lane));')
            u_op_map = {
                'add': 'static_cast<uint16_t>(sv0 + sv1)',
                'sub': 'static_cast<uint16_t>(sv0 - sv1)',
                'rsub': 'static_cast<uint16_t>(sv1 - sv0)',
                'mul': 'static_cast<uint16_t>(sv0 * sv1)',
                'min': 'sv0 < sv1 ? sv0 : sv1',
                'max': 'sv0 > sv1 ? sv0 : sv1',
                'shl': 'static_cast<uint16_t>(sv1 << (sv0 & 15u))',
                'shr': 'static_cast<uint16_t>(sv1 >> (sv0 & 15u))',
                'and': 'static_cast<uint16_t>(sv0 & sv1)',
                'or': 'static_cast<uint16_t>(sv0 | sv1)',
                'xor': 'static_cast<uint16_t>(sv0 ^ sv1)',
            }
            expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
        elif dtype in ('i64',):
            if op == 'ashr':
                L.append(f'    int64_t val = static_cast<int64_t>({s1}.read_lane64(wf, lane));')
                L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
                L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(val >> shift));')
            else:
                L.append(f'    int64_t sv0 = static_cast<int64_t>({s0}.read_lane64(wf, lane));')
                L.append(f'    int64_t sv1 = static_cast<int64_t>({s1}.read_lane64(wf, lane));')
                i_op_map = {
                    'add': 'static_cast<uint64_t>(sv0 + sv1)',
                    'sub': 'static_cast<uint64_t>(sv0 - sv1)',
                    'min': 'static_cast<uint64_t>(sv0 < sv1 ? sv0 : sv1)',
                    'max': 'static_cast<uint64_t>(sv0 > sv1 ? sv0 : sv1)',
                }
                expr = i_op_map.get(op, f'static_cast<uint64_t>(sv0) /* TODO: {op} */')
                L.append(f'    {d}.write_lane64(wf, lane, {expr});')
        elif dtype in ('u64', 'b64'):
            if op == 'shl':
                L.append(f'    uint64_t val = {s1}.read_lane64(wf, lane);')
                L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
                L.append(f'    {d}.write_lane64(wf, lane, val << shift);')
            elif op == 'shr':
                L.append(f'    uint64_t val = {s1}.read_lane64(wf, lane);')
                L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
                L.append(f'    {d}.write_lane64(wf, lane, val >> shift);')
            elif op == 'add':
                L.append(f'    uint64_t sv0 = {s0}.read_lane64(wf, lane);')
                L.append(f'    uint64_t sv1 = {s1}.read_lane64(wf, lane);')
                L.append(f'    {d}.write_lane64(wf, lane, sv0 + sv1);')
            else:
                L.append(f'    uint64_t sv0 = {s0}.read_lane64(wf, lane);')
                L.append(f'    uint64_t sv1 = {s1}.read_lane64(wf, lane);')
                u_op_map = {
                    'sub': 'sv0 - sv1',
                    'and': 'sv0 & sv1',
                    'or': 'sv0 | sv1',
                    'xor': 'sv0 ^ sv1',
                    'min': 'sv0 < sv1 ? sv0 : sv1',
                    'max': 'sv0 > sv1 ? sv0 : sv1',
                }
                expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
                L.append(f'    {d}.write_lane64(wf, lane, {expr});')
        elif dtype in ('i32',):
            L.append(f'    int32_t sv0 = static_cast<int32_t>({s0}.read_lane(wf, lane));')
            L.append(f'    int32_t sv1 = static_cast<int32_t>({s1}.read_lane(wf, lane));')
            i_op_map = {
                'add': 'static_cast<uint32_t>(sv0 + sv1)',
                'sub': 'static_cast<uint32_t>(sv0 - sv1)',
                'rsub': 'static_cast<uint32_t>(sv1 - sv0)',
                'min': 'static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1)',
                'max': 'static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1)',
                'ashr': 'static_cast<uint32_t>(static_cast<int32_t>(sv1) >> (sv0 & 31))',
                'mul': 'static_cast<uint32_t>(sv0 * sv1)',
                'mulhi': 'static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(sv0) * sv1) >> 32)',
            }
            expr = i_op_map.get(op, f'static_cast<uint32_t>(sv0) /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, {expr});')
        else:
            L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane);')
            L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane);')
            u_op_map = {
                'add': 'sv0 + sv1',
                'sub': 'sv0 - sv1',
                'rsub': 'sv1 - sv0',
                'mul': 'sv0 * sv1',
                'mulhi': 'static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32)',
                'and': 'sv0 & sv1',
                'or': 'sv0 | sv1',
                'xor': 'sv0 ^ sv1',
                'xnor': '~(sv0 ^ sv1)',
                'shl': 'sv1 << (sv0 & 31u)',
                'shr': 'sv1 >> (sv0 & 31u)',
                'min': 'sv0 < sv1 ? sv0 : sv1',
                'max': 'sv0 > sv1 ? sv0 : sv1',
                'bfm': '(sv0 & 31u) == 0 ? 0u : ((1u << (sv0 & 31u)) - 1u) << (sv1 & 31u)',
            }
            expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, {expr});')

        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_ternary(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector ternary (3-operand) operation body."""
        d = dst[0]
        s0, s1, s2 = src[0], src[1], src[2]

        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')

        # BFE/BFI/PERM always use unsigned raw lane reads regardless of dtype
        if op in ('bfe_u', 'bfe_i', 'perm'):
            L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
            L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
            L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
            if op == 'bfe_u':
                L.append('    uint32_t offset = b & 31;')
                L.append('    uint32_t width = c & 31;')
                L.append('    uint32_t mask = (width == 0 || width >= 32) ? 0u : ((1u << width) - 1);')
                L.append(f'    {d}.write_lane(wf, lane, width == 0 ? 0u : (a >> offset) & mask);')
            elif op == 'bfe_i':
                L.append('    uint32_t offset = b & 31;')
                L.append('    uint32_t width = c & 31;')
                L.append('    int32_t sv = static_cast<int32_t>(a);')
                L.append('    int32_t result_val;')
                L.append('    if (width == 0) result_val = 0;')
                # When offset + width >= 32, the extraction window extends
                # past bit 31. Arithmetic right shift of sv by offset gives
                # the correct sign-extended result without shift UB.
                L.append('    else if (offset + width >= 32) result_val = sv >> offset;')
                L.append('    else result_val = (sv << (32 - offset - width)) >> (32 - width);')
                L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(result_val));')
            else:  # perm
                L.append(f'    uint32_t result = 0;')
                L.append('    uint64_t src = (static_cast<uint64_t>(a) << 32) | b;')
                L.append('    for (int i = 0; i < 4; ++i) {')
                L.append('      uint32_t sel = (c >> (i * 8)) & 0xFF;')
                L.append('      uint32_t byte;')
                L.append('      if (sel <= 7) byte = (src >> (sel * 8)) & 0xFF;')
                # Selector 0x08: constant 0x00
                # Selectors 0x09-0x0C: replicate the MSB (sign bit) of
                # byte 0-3 respectively to all 8 bits (0x00 or 0xFF).
                L.append('      else if (sel >= 0x09 && sel <= 0x0C) {')
                L.append('        uint32_t bi = sel - 0x09;')
                L.append('        byte = ((src >> (bi * 8 + 7)) & 1) ? 0xFF : 0x00;')
                L.append('      }')
                L.append('      else if (sel == 0x0D) byte = 0xFF;')
                # Selectors 0x08, 0x0E-0xFF: constant 0x00
                L.append('      else byte = 0;')
                L.append('      result |= byte << (i * 8);')
                L.append('    }')
                L.append(f'    {d}.write_lane(wf, lane, result);')
            L.append('  }')
            return '\n'.join(L)

        if dtype == 'f32':
            L.append(f'    float a = std::bit_cast<float>({s0}.read_lane(wf, lane));')
            L.append(f'    float b = std::bit_cast<float>({s1}.read_lane(wf, lane));')
            L.append(f'    float c = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('a', 0))
                L.extend(self._vop3_src_mod('b', 1))
                L.extend(self._vop3_src_mod('c', 2))
            f_map = {
                'mad': 'a * b + c',
                'fma': 'std::fma(a, b, c)',
                'min3': 'std::fmin(std::fmin(a, b), c)',
                'max3': 'std::fmax(std::fmax(a, b), c)',
                'minimum3': 'std::fmin(std::fmin(a, b), c)',
                'maximum3': 'std::fmax(std::fmax(a, b), c)',
                'minmax': 'std::fmin(a, std::fmax(b, c))',
                'maxmin': 'std::fmax(a, std::fmin(b, c))',
                'minimummaximum': 'std::fmin(std::fmax(a, b), c)',
                'maximumminimum': 'std::fmax(std::fmin(a, b), c)',
                'minmax_num': 'std::fmin(a, std::fmax(b, c))',
                'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
                'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
            }
            # Cube map operations: inputs are (x, y, z)
            if op == 'cubeid':
                L.append('    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);')
                L.append('    float face;')
                L.append('    if (az >= ax && az >= ay) face = c >= 0 ? 4.0f : 5.0f;')
                L.append('    else if (ay >= ax) face = b >= 0 ? 2.0f : 3.0f;')
                L.append('    else face = a >= 0 ? 0.0f : 1.0f;')
                if is_vop3:
                    L.extend(self._vop3_dst_mod('face'))
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(face));')
            elif op == 'cubesc':
                L.append('    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);')
                L.append('    float sc;')
                L.append('    if (az >= ax && az >= ay) sc = c >= 0 ? a : -a;')
                L.append('    else if (ay >= ax) sc = a;')
                L.append('    else sc = a >= 0 ? -c : c;')
                if is_vop3:
                    L.extend(self._vop3_dst_mod('sc'))
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(sc));')
            elif op == 'cubetc':
                L.append('    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);')
                L.append('    float tc;')
                L.append('    if (az >= ax && az >= ay) tc = -b;')
                L.append('    else if (ay >= ax) tc = b >= 0 ? c : -c;')
                L.append('    else tc = -b;')
                if is_vop3:
                    L.extend(self._vop3_dst_mod('tc'))
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(tc));')
            elif op == 'cubema':
                L.append('    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);')
                L.append('    float ma;')
                L.append('    if (az >= ax && az >= ay) ma = 2.0f * az;')
                L.append('    else if (ay >= ax) ma = 2.0f * ay;')
                L.append('    else ma = 2.0f * ax;')
                if is_vop3:
                    L.extend(self._vop3_dst_mod('ma'))
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(ma));')
            elif op in f_map:
                expr = f_map[op]
                if is_vop3:
                    L.append(f'    float result = {expr};')
                    L.extend(self._vop3_dst_mod('result'))
                    L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
                else:
                    L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(a)); // unhandled: {op}')
        elif dtype == 'f64':
            L.append(f'    double a = std::bit_cast<double>({s0}.read_lane64(wf, lane));')
            L.append(f'    double b = std::bit_cast<double>({s1}.read_lane64(wf, lane));')
            L.append(f'    double c = std::bit_cast<double>({s2}.read_lane64(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('a', 0))
                L.extend(self._vop3_src_mod('b', 1))
                L.extend(self._vop3_src_mod('c', 2))
            f_map = {
                'mad': 'a * b + c',
                'fma': 'std::fma(a, b, c)',
                'min3': 'std::fmin(std::fmin(a, b), c)',
                'max3': 'std::fmax(std::fmax(a, b), c)',
                'minimum3': 'std::fmin(std::fmin(a, b), c)',
                'maximum3': 'std::fmax(std::fmax(a, b), c)',
                'minmax': 'std::fmin(a, std::fmax(b, c))',
                'maxmin': 'std::fmax(a, std::fmin(b, c))',
                'minimummaximum': 'std::fmin(std::fmax(a, b), c)',
                'maximumminimum': 'std::fmax(std::fmin(a, b), c)',
                'minmax_num': 'std::fmin(a, std::fmax(b, c))',
                'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
                'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
            }
            expr = f_map.get(op, f'a /* unhandled: {op} */')
            if is_vop3:
                L.append(f'    double result = {expr};')
                L.extend(self._vop3_dst_mod_f64('result'))
                L.append(f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
            else:
                L.append(f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));')
        elif dtype == 'f16':
            L.append(f'    float a = util::f16_to_f32(static_cast<uint16_t>({s0}.read_lane(wf, lane)));')
            L.append(f'    float b = util::f16_to_f32(static_cast<uint16_t>({s1}.read_lane(wf, lane)));')
            L.append(f'    float c = util::f16_to_f32(static_cast<uint16_t>({s2}.read_lane(wf, lane)));')
            if is_vop3:
                L.extend(self._vop3_src_mod('a', 0))
                L.extend(self._vop3_src_mod('b', 1))
                L.extend(self._vop3_src_mod('c', 2))
            f_map = {
                'mad': 'a * b + c',
                'fma': 'std::fma(a, b, c)',
                'min3': 'std::fmin(std::fmin(a, b), c)',
                'max3': 'std::fmax(std::fmax(a, b), c)',
                'minimum3': 'std::fmin(std::fmin(a, b), c)',
                'maximum3': 'std::fmax(std::fmax(a, b), c)',
                'minmax': 'std::fmin(a, std::fmax(b, c))',
                'maxmin': 'std::fmax(a, std::fmin(b, c))',
                'minimummaximum': 'std::fmin(std::fmax(a, b), c)',
                'maximumminimum': 'std::fmax(std::fmin(a, b), c)',
                'minmax_num': 'std::fmin(a, std::fmax(b, c))',
                'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
                'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
            }
            expr = f_map.get(op, f'a /* unhandled: {op} */')
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(self._vop3_dst_mod('result'))
                L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(result));')
            else:
                L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16({expr}));')
        elif dtype in ('i16',):
            L.append(f'    int16_t a = static_cast<int16_t>({s0}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int16_t b = static_cast<int16_t>({s1}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int16_t c = static_cast<int16_t>({s2}.read_lane(wf, lane) & 0xFFFF);')
            i_map = {
                'min3': 'std::min(std::min(a, b), c)',
                'max3': 'std::max(std::max(a, b), c)',
                'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
                'mad': 'static_cast<int16_t>(a * b + c)',
            }
            expr = i_map.get(op, f'a /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>({expr})));')
        elif dtype in ('u16',):
            L.append(f'    uint16_t a = static_cast<uint16_t>({s0}.read_lane(wf, lane));')
            L.append(f'    uint16_t b = static_cast<uint16_t>({s1}.read_lane(wf, lane));')
            L.append(f'    uint16_t c = static_cast<uint16_t>({s2}.read_lane(wf, lane));')
            u_map = {
                'min3': 'std::min(std::min(a, b), c)',
                'max3': 'std::max(std::max(a, b), c)',
                'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
                'mad': 'static_cast<uint16_t>(a * b + c)',
            }
            expr = u_map.get(op, f'a /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
        elif dtype in ('u64',):
            L.append(f'    uint64_t a = {s0}.read_lane64(wf, lane);')
            L.append(f'    uint64_t b = {s1}.read_lane64(wf, lane);')
            L.append(f'    uint64_t c = {s2}.read_lane64(wf, lane);')
            u_map = {
                'lshl_add': '(a << (b & 63u)) + c',
                'add3': 'a + b + c',
            }
            expr = u_map.get(op, f'a /* unhandled: {op} */')
            L.append(f'    {d}.write_lane64(wf, lane, {expr});')
        elif dtype in ('i24',):
            L.append(f'    int32_t a = static_cast<int32_t>({s0}.read_lane(wf, lane) << 8) >> 8;')
            L.append(f'    int32_t b = static_cast<int32_t>({s1}.read_lane(wf, lane) << 8) >> 8;')
            L.append(f'    int32_t c = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(a * b + c));')
        elif dtype in ('u24',):
            L.append(f'    uint32_t a = {s0}.read_lane(wf, lane) & 0x00FFFFFFu;')
            L.append(f'    uint32_t b = {s1}.read_lane(wf, lane) & 0x00FFFFFFu;')
            L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
            L.append(f'    {d}.write_lane(wf, lane, a * b + c);')
        elif dtype in ('i32',):
            L.append(f'    int32_t a = static_cast<int32_t>({s0}.read_lane(wf, lane));')
            L.append(f'    int32_t b = static_cast<int32_t>({s1}.read_lane(wf, lane));')
            L.append(f'    int32_t c = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            i_map = {
                'min3': 'std::min(std::min(a, b), c)',
                'max3': 'std::max(std::max(a, b), c)',
                'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
                'minmax': 'std::min(a, std::max(b, c))',
                'maxmin': 'std::max(a, std::min(b, c))',
                'minmax_num': 'std::min(a, std::max(b, c))',
                'maxmin_num': 'std::max(a, std::min(b, c))',
                'add_lshl': '(a + b) << (c & 31)',
                'lshl_add': '(a << (b & 31)) + c',
            }
            expr = i_map.get(op, f'a /* TODO: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
        else:
            L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
            L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
            L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
            # SAD and LERP need multi-line code
            if op == 'sad_u8':
                L.append('    uint32_t sum = 0;')
                L.append('    for (int i = 0; i < 4; ++i) {')
                L.append('      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;')
                L.append('      sum += ba > bb ? ba - bb : bb - ba;')
                L.append('    }')
                L.append(f'    {d}.write_lane(wf, lane, sum + c);')
            elif op == 'sad_hi_u8':
                L.append('    uint32_t sum = 0;')
                L.append('    for (int i = 0; i < 4; ++i) {')
                L.append('      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;')
                L.append('      sum += ba > bb ? ba - bb : bb - ba;')
                L.append('    }')
                L.append(f'    {d}.write_lane(wf, lane, (sum << 16) + c);')
            elif op == 'sad_u16':
                L.append('    uint32_t lo_a = a & 0xFFFF, hi_a = a >> 16;')
                L.append('    uint32_t lo_b = b & 0xFFFF, hi_b = b >> 16;')
                L.append('    uint32_t sum = (lo_a > lo_b ? lo_a - lo_b : lo_b - lo_a) + (hi_a > hi_b ? hi_a - hi_b : hi_b - hi_a);')
                L.append(f'    {d}.write_lane(wf, lane, sum + c);')
            elif op == 'sad_u32':
                L.append(f'    {d}.write_lane(wf, lane, (a > b ? a - b : b - a) + c);')
            elif op == 'msad_u8':
                L.append('    uint32_t sum = 0;')
                L.append('    for (int i = 0; i < 4; ++i) {')
                L.append('      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;')
                L.append('      if (ba != 0) sum += ba > bb ? ba - bb : bb - ba;')
                L.append('    }')
                L.append(f'    {d}.write_lane(wf, lane, sum + c);')
            elif op == 'lerp_u8':
                L.append('    uint32_t result = 0;')
                L.append('    for (int i = 0; i < 4; ++i) {')
                L.append('      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;')
                L.append('      uint32_t bc = (c >> (i * 8)) & 1;')
                L.append('      result |= (((ba + bb + bc) >> 1) & 0xFF) << (i * 8);')
                L.append('    }')
                L.append(f'    {d}.write_lane(wf, lane, result);')
            else:
                u_map = {
                    'add3': 'a + b + c',
                    'lshl_or': '(a << (b & 31)) | c',
                    'and_or': '(a & b) | c',
                    'or3': 'a | b | c',
                    'lshl_add': '(a << (b & 31)) + c',
                    'add_lshl': '(a + b) << (c & 31)',
                    'xad': '(a ^ b) + c',
                    'xor3': 'a ^ b ^ c',
                    'min3': 'std::min(std::min(a, b), c)',
                    'max3': 'std::max(std::max(a, b), c)',
                    'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
                    'minmax': 'std::min(a, std::max(b, c))',
                    'maxmin': 'std::max(a, std::min(b, c))',
                    'minmax_num': 'std::min(a, std::max(b, c))',
                    'maxmin_num': 'std::max(a, std::min(b, c))',
                    'bfi': '(a & b) | (~a & c)',
                    'alignbit': 'static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> (c & 31))',
                    'alignbyte': 'static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> ((c & 3) * 8))',
                }
                expr = u_map.get(op, f'a /* unhandled: {op} */')
                L.append(f'    {d}.write_lane(wf, lane, {expr});')

        L.append('  }')
        return '\n'.join(L)

    def _cmp_condition(self, src: list[str], op: str | None, dtype: str | None, is_vop3: bool, L: list[str]) -> str:
        """Emit source reads and return the C++ condition expression.

        For FP types, handles ordered comparisons (eq, lt, le, gt, ge, lg),
        unordered comparisons (neq, nge, ngt, nle, nlt, nlg), and
        ordered/unordered predicates (o, u) per IEEE-754.
        """
        is_fp = dtype in ('f32', 'f64', 'f16')
        if is_fp:
            if dtype == 'f64':
                L.append(f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));')
                L.append(f'    double s1 = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));')
            elif dtype == 'f16':
                L.append(f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));')
                L.append(f'    float s1 = util::f16_to_f32(static_cast<uint16_t>({src[1]}.read_lane(wf, lane)));')
            else:
                L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
                L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
                L.extend(self._vop3_src_mod('s1', 1))
            # Ordered comparisons (false if NaN)
            ordered_map = {
                'eq': 's0 == s1', 'ne': 's0 != s1',
                'lg': 's0 < s1 || s0 > s1',
                'lt': 's0 < s1', 'le': 's0 <= s1',
                'gt': 's0 > s1', 'ge': 's0 >= s1',
            }
            # Unordered comparisons (true if NaN)
            unordered_map = {
                'neq': 's0 != s1 || std::isnan(s0) || std::isnan(s1)',
                'nge': '!(s0 >= s1)',  # true when NaN (IEEE)
                'ngt': '!(s0 > s1)',
                'nle': '!(s0 <= s1)',
                'nlt': '!(s0 < s1)',
                'nlg': '!(s0 < s1 || s0 > s1)',
            }
            if op in ordered_map:
                return ordered_map[op]
            if op in unordered_map:
                return unordered_map[op]
            if op == 'o':
                return '!std::isnan(s0) && !std::isnan(s1)'
            if op == 'u':
                return 'std::isnan(s0) || std::isnan(s1)'
            return f's0 == s1 /* TODO: {op} */'
        elif dtype in ('i64',):
            L.append(f'    int64_t s0 = static_cast<int64_t>({src[0]}.read_lane64(wf, lane));')
            L.append(f'    int64_t s1 = static_cast<int64_t>({src[1]}.read_lane64(wf, lane));')
        elif dtype in ('u64',):
            L.append(f'    uint64_t s0 = {src[0]}.read_lane64(wf, lane);')
            L.append(f'    uint64_t s1 = {src[1]}.read_lane64(wf, lane);')
        elif dtype in ('i16',):
            L.append(f'    int16_t s0 = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);')
            L.append(f'    int16_t s1 = static_cast<int16_t>({src[1]}.read_lane(wf, lane) & 0xFFFF);')
        elif dtype in ('u16',):
            L.append(f'    uint16_t s0 = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));')
            L.append(f'    uint16_t s1 = static_cast<uint16_t>({src[1]}.read_lane(wf, lane));')
        elif dtype in ('i32',):
            L.append(f'    int32_t s0 = static_cast<int32_t>({src[0]}.read_lane(wf, lane));')
            L.append(f'    int32_t s1 = static_cast<int32_t>({src[1]}.read_lane(wf, lane));')
        else:
            L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane);')
            L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane);')
        cmp_map = {
            'eq': '==', 'ne': '!=', 'lg': '!=',
            'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<=',
        }
        cmp_op = cmp_map.get(op, f'== /* TODO: {op} */')
        return f's0 {cmp_op} s1'

    def _gen_vector_cmp(self, dst: list[str], src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector compare body.

        VOPC (VOP2-like): result always goes to VCC.
        VOP3: result goes to dst[0] (explicit SGPR pair, may be VCC or any SGPR).
        Inactive lanes preserve the destination register's existing bits.
        """
        L = []
        L.append('  uint64_t exec = wf.exec();')
        if dst:
            # VOP3: initialize from the destination register so inactive
            # lanes preserve its existing bits (not VCC).
            L.append(f'  uint64_t vcc = {dst[0]}.read_scalar64(wf);')
        else:
            # VOPC: destination is VCC.
            L.append('  uint64_t vcc = wf.vcc();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')

        if op == 'f':
            L.append('    vcc &= ~(1ULL << lane);')
        elif op == 't':
            L.append('    vcc |= (1ULL << lane);')
        else:
            cond = self._cmp_condition(src, op, dtype, is_vop3, L)
            L.append(f'    if ({cond})')
            L.append('      vcc |= (1ULL << lane);')
            L.append('    else')
            L.append('      vcc &= ~(1ULL << lane);')
        L.append('  }')
        if dst:
            # VOP3: write to explicit destination (sdst/vdst SGPR pair).
            L.append(f'  {dst[0]}.write_scalar64(wf, vcc);')
        else:
            # VOPC: write to VCC.
            L.append('  wf.set_vcc(vcc);')
        return '\n'.join(L)

    def _gen_vector_cmpx(self, src: list[str], op: str | None, dtype: str | None,
                         is_vop3: bool = False, dst: list[str] | None = None) -> str:
        """Generate vector compare-and-write-EXEC body.

        On CDNA (GFX9), V_CMPX writes both EXEC and the SDST operand.
        For VOP3 encoding, SDST is the vdst field (which may be VCC or
        another SGPR pair). On RDNA, V_CMPX writes only EXEC.
        """
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint64_t result = 0;')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')

        if op == 'f':
            L.append('    (void)lane;')
        elif op == 't':
            L.append('    result |= (1ULL << lane);')
        else:
            cond = self._cmp_condition(src, op, dtype, is_vop3, L)
            L.append(f'    if ({cond})')
            L.append('      result |= (1ULL << lane);')
        L.append('  }')
        if self.isa_spec.profile.cmpx_writes_vcc:
            if dst and is_vop3:
                L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            else:
                L.append('  wf.set_vcc(result);')
        L.append('  wf.set_exec(result);')
        return '\n'.join(L)

    def _gen_vector_add_co(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate vector add/sub with carry in/out.

        VOP2: carry in/out via VCC (implicit).
        VOP3/VOP3_SDST_ENC: carry-in from src[2] (explicit SGPR pair),
        carry-out to dst[1] (explicit SGPR pair).
        """
        L = []
        d = dst[0]
        s0, s1 = src[0], src[1]
        _is_vop3 = len(src) > 2 or len(dst) > 1

        L.append('  uint64_t exec = wf.exec();')
        if _is_vop3 and op in ('addc', 'subbc', 'subbrevco') and len(src) > 2:
            # VOP3: carry-in from explicit src2 SGPR pair.
            L.append(f'  uint64_t old_vcc = {src[2]}.read_scalar64(wf);')
        elif op in ('addc', 'subbc', 'subbrevco'):
            # VOP2: carry-in from VCC.
            L.append('  uint64_t old_vcc = wf.vcc();')
        L.append('  uint64_t vcc = wf.vcc();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane);')

        if op == 'add':
            L.append('    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1);')
        elif op == 'sub':
            L.append('    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1);')
            L.append('    bool borrow = sv0 < sv1;')
        elif op == 'rsub':
            L.append('    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0);')
            L.append('    bool borrow = sv1 < sv0;')
        elif op == 'addc':
            L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
            L.append('    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1) + cin;')
        elif op == 'subbc':
            L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
            L.append('    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1) - cin;')
            L.append('    bool borrow = static_cast<uint64_t>(sv0) < static_cast<uint64_t>(sv1) + cin;')
        elif op == 'subbrevco':
            L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
            L.append('    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0) - cin;')
            L.append('    bool borrow = static_cast<uint64_t>(sv1) < static_cast<uint64_t>(sv0) + cin;')

        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(wide));')

        if op in ('add', 'addc'):
            L.append('    if (wide > 0xFFFFFFFFULL) vcc |= (1ULL << lane); else vcc &= ~(1ULL << lane);')
        elif op in ('sub', 'rsub', 'subbc', 'subbrevco'):
            L.append('    if (borrow) vcc |= (1ULL << lane); else vcc &= ~(1ULL << lane);')

        L.append('  }')
        if len(dst) > 1:
            # VOP3_SDST_ENC: carry-out goes to sdst (any SGPR pair).
            L.append(f'  {dst[1]}.write_scalar64(wf, vcc);')
        else:
            L.append('  wf.set_vcc(vcc);')
        return '\n'.join(L)

    def _vop3_src_mod(self, varname: str, src_idx: int,
                       indent: str = '    ') -> list[str]:
        """Generate VOP3 input modifier lines (abs then neg) for a float src.

        Uses ``self._has_abs``, which is set by ``_gen_execute_body`` at the
        start of each instruction's code generation.
        """
        lines = []
        if self._has_abs:
            lines.append(
                f'{indent}if (inst_.abs & (1u << {src_idx})) {varname} = std::fabs({varname});')
        lines.append(
            f'{indent}if (inst_.neg & (1u << {src_idx})) {varname} = -{varname};')
        return lines

    def _vop3_dst_mod(self, varname: str,
                       indent: str = '    ') -> list[str]:
        """Generate VOP3 output modifier lines (omod then clamp) for a float result."""
        return [
            f'{indent}if (inst_.omod == 1) {varname} *= 2.0f;',
            f'{indent}else if (inst_.omod == 2) {varname} *= 4.0f;',
            f'{indent}else if (inst_.omod == 3) {varname} *= 0.5f;',
            f'{indent}if (inst_.clamp) {varname} = std::clamp({varname}, 0.0f, 1.0f);',
        ]

    def _vop3_dst_mod_f64(self, varname: str,
                           indent: str = '    ') -> list[str]:
        """Generate VOP3 output modifier lines for a double result."""
        return [
            f'{indent}if (inst_.omod == 1) {varname} *= 2.0;',
            f'{indent}else if (inst_.omod == 2) {varname} *= 4.0;',
            f'{indent}else if (inst_.omod == 3) {varname} *= 0.5;',
            f'{indent}if (inst_.clamp) {varname} = std::clamp({varname}, 0.0, 1.0);',
        ]

    def _gen_pk_binop(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate packed 16-bit binary op (V_PK_ADD_I16, V_PK_MUL_F16, etc.)."""
        d, s0, s1 = dst[0], src[0], src[1]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

        # op_sel: which half of each src for LO result
        # op_sel_hi: which half for HI result (default = hi)
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
        L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
        L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
        L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')

        if dtype == 'f16':
            # FP16: extract as float, operate, pack back
            L.append('    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
            L.append('    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
            L.append('    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
            L.append('    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
            # neg modifiers
            L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
            L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
            L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
            L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
            f_op_map = {
                'add': ('a_lo + b_lo', 'a_hi + b_hi'),
                'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
                'min': ('std::fmin(a_lo, b_lo)', 'std::fmin(a_hi, b_hi)'),
                'max': ('std::fmax(a_lo, b_lo)', 'std::fmax(a_hi, b_hi)'),
            }
            lo_expr, hi_expr = f_op_map[op]
            L.append(f'    float rlo = {lo_expr};')
            L.append(f'    float rhi = {hi_expr};')
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));')
        elif dtype == 'i16':
            L.append('    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            i_op_map = {
                'add': ('a_lo + b_lo', 'a_hi + b_hi'),
                'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
                'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
                'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
                'ashr': ('static_cast<int16_t>(b_lo >> (a_lo & 15))',
                         'static_cast<int16_t>(b_hi >> (a_hi & 15))'),
            }
            lo_expr, hi_expr = i_op_map[op]
            L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
            L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')
        else:  # u16
            L.append('    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            u_op_map = {
                'add': ('a_lo + b_lo', 'a_hi + b_hi'),
                'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
                'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
                'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
                'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
                'shl': ('static_cast<uint16_t>(b_lo << (a_lo & 15u))',
                        'static_cast<uint16_t>(b_hi << (a_hi & 15u))'),
                'shr': ('static_cast<uint16_t>(b_lo >> (a_lo & 15u))',
                        'static_cast<uint16_t>(b_hi >> (a_hi & 15u))'),
            }
            lo_expr, hi_expr = u_op_map[op]
            L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
            L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')

        L.append('  }')
        return '\n'.join(L)

    def _gen_pk_ternary(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate packed 16-bit ternary op (V_PK_FMA_F16, V_PK_MAD_I16, etc.)."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
        L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
        L.append(f'    bool sel2_lo = ({opsel} >> 2) & 1;')
        L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
        L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
        L.append(f'    bool sel2_hi = {self._op_sel_hi_2_expr(self._enc_name)};')

        if dtype == 'f16':
            L.append('    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
            L.append('    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
            L.append('    float c_lo = util::f16_to_f32(static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2));')
            L.append('    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
            L.append('    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
            L.append('    float c_hi = util::f16_to_f32(static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2));')
            L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
            L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
            L.append('    if (inst_.neg & 4) { c_lo = -c_lo; }')
            L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
            L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
            L.append('    if (inst_.neg_hi & 4) { c_hi = -c_hi; }')
            if op == 'fma':
                L.append('    float rlo = std::fma(a_lo, b_lo, c_lo);')
                L.append('    float rhi = std::fma(a_hi, b_hi, c_hi);')
            elif op in ('minimum3', 'min3'):
                L.append('    float rlo = std::fmin(std::fmin(a_lo, b_lo), c_lo);')
                L.append('    float rhi = std::fmin(std::fmin(a_hi, b_hi), c_hi);')
            elif op in ('maximum3', 'max3'):
                L.append('    float rlo = std::fmax(std::fmax(a_lo, b_lo), c_lo);')
                L.append('    float rhi = std::fmax(std::fmax(a_hi, b_hi), c_hi);')
            else:  # mad
                L.append('    float rlo = a_lo * b_lo + c_lo;')
                L.append('    float rhi = a_hi * b_hi + c_hi;')
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));')
        elif dtype == 'i16':
            L.append('    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    int16_t c_lo = static_cast<int16_t>(sel2_lo ? (raw2 >> 16) : raw2);')
            L.append('    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            L.append('    int16_t c_hi = static_cast<int16_t>(sel2_hi ? (raw2 >> 16) : raw2);')
            L.append('    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);')
            L.append('    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')
        else:  # u16
            L.append('    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    uint16_t c_lo = static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2);')
            L.append('    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            L.append('    uint16_t c_hi = static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2);')
            L.append('    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);')
            L.append('    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')

        L.append('  }')
        return '\n'.join(L)

    def _gen_pk_binop_f32(self, dst: list[str], src: list[str], op: str | None) -> str:
        """Generate packed F32 binary op (V_PK_ADD_F32, V_PK_MUL_F32).

        Operands are 64-bit VGPR pairs holding two 32-bit floats.
        Uses op_sel/op_sel_hi to select which 32-bit half feeds each lane,
        and neg/neg_hi for per-lane negation.
        """
        d, s0, s1 = dst[0], src[0], src[1]
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        # Read each source as a pair of F32 values. For VGPR pairs
        # (encoding 256-511), the low register is read_lane and the high
        # register is the next VGPR. For scalar/constant sources, the same
        # 32-bit value applies to both halves.
        for var, src in [('s0', s0), ('s1', s1)]:
            L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
            L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
            L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
            L.append(f'        : {var}_lo_w;')
        L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
        L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
        L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
        L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
        L.append('    float a_lo = std::bit_cast<float>(sel0_lo ? s0_hi_w : s0_lo_w);')
        L.append('    float a_hi = std::bit_cast<float>(sel0_hi ? s0_hi_w : s0_lo_w);')
        L.append('    float b_lo = std::bit_cast<float>(sel1_lo ? s1_hi_w : s1_lo_w);')
        L.append('    float b_hi = std::bit_cast<float>(sel1_hi ? s1_hi_w : s1_lo_w);')
        L.append('    if (inst_.neg & 1) a_lo = -a_lo;')
        L.append('    if (inst_.neg & 2) b_lo = -b_lo;')
        L.append('    if (inst_.neg_hi & 1) a_hi = -a_hi;')
        L.append('    if (inst_.neg_hi & 2) b_hi = -b_hi;')
        f_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
        }
        lo_expr, hi_expr = f_map[op]
        L.append(f'    uint32_t rlo = std::bit_cast<uint32_t>({lo_expr});')
        L.append(f'    uint32_t rhi = std::bit_cast<uint32_t>({hi_expr});')
        L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_pk_ternary_f32(self, dst: list[str], src: list[str], op: str | None) -> str:
        """Generate packed F32 ternary op (V_PK_FMA_F32).

        Uses op_sel/op_sel_hi/op_sel_hi_2 to select which 32-bit half
        of each source feeds the low and high FMA lanes.
        """
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        opsel_hi_2 = self._op_sel_hi_2_expr(self._enc_name)
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for var, src in [('s0', s0), ('s1', s1), ('s2', s2)]:
            L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
            L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
            L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
            L.append(f'        : {var}_lo_w;')
        L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
        L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
        L.append(f'    bool sel2_lo = ({opsel} >> 2) & 1;')
        L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
        L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
        L.append(f'    bool sel2_hi = {opsel_hi_2};')
        L.append('    float a_lo = std::bit_cast<float>(sel0_lo ? s0_hi_w : s0_lo_w);')
        L.append('    float a_hi = std::bit_cast<float>(sel0_hi ? s0_hi_w : s0_lo_w);')
        L.append('    float b_lo = std::bit_cast<float>(sel1_lo ? s1_hi_w : s1_lo_w);')
        L.append('    float b_hi = std::bit_cast<float>(sel1_hi ? s1_hi_w : s1_lo_w);')
        L.append('    float c_lo = std::bit_cast<float>(sel2_lo ? s2_hi_w : s2_lo_w);')
        L.append('    float c_hi = std::bit_cast<float>(sel2_hi ? s2_hi_w : s2_lo_w);')
        L.append('    if (inst_.neg & 1) a_lo = -a_lo;')
        L.append('    if (inst_.neg & 2) b_lo = -b_lo;')
        L.append('    if (inst_.neg & 4) c_lo = -c_lo;')
        L.append('    if (inst_.neg_hi & 1) a_hi = -a_hi;')
        L.append('    if (inst_.neg_hi & 2) b_hi = -b_hi;')
        L.append('    if (inst_.neg_hi & 4) c_hi = -c_hi;')
        L.append('    uint32_t rlo = std::bit_cast<uint32_t>(std::fma(a_lo, b_lo, c_lo));')
        L.append('    uint32_t rhi = std::bit_cast<uint32_t>(std::fma(a_hi, b_hi, c_hi));')
        L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_pk_mov_b32(self, dst: list[str], src: list[str]) -> str:
        """Generate V_PK_MOV_B32: move two 32-bit values based on op_sel."""
        d, s0, s1 = dst[0], src[0], src[1]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for var, src in [('s0', s0), ('s1', s1)]:
            L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
            L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
            L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
            L.append(f'        : {var}_lo_w;')
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L.append(f'    uint32_t lo = ({opsel} & 1) ? s0_hi_w : s0_lo_w;')
        L.append(f'    uint32_t hi = ({opsel_hi} & 2) ? s1_hi_w : s1_lo_w;')
        L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_mad_mix_f32(self, dst: list[str], src: list[str]) -> str:
        """Generate V_MAD_MIX_F32: mixed-precision FMA with op_sel selecting f16/f32 per src."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
        # op_sel_hi selects f16 vs f32 per source (1=f16, 0=f32)
        # When f16: op_sel[i] selects which half (lo=0, hi=1)
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L.append('    float a, b, c;')
        L.append(f'    if ({opsel_hi} & 1) a = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 1) ? (raw0 >> 16) : raw0));')
        L.append('    else a = std::bit_cast<float>(raw0);')
        L.append(f'    if ({opsel_hi} & 2) b = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 2) ? (raw1 >> 16) : raw1));')
        L.append('    else b = std::bit_cast<float>(raw1);')
        L.append(f'    if ({self._op_sel_hi_2_expr(self._enc_name)}) c = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 4) ? (raw2 >> 16) : raw2));')
        L.append('    else c = std::bit_cast<float>(raw2);')
        L.append('    if (inst_.neg & 1) a = -a;')
        L.append('    if (inst_.neg & 2) b = -b;')
        L.append('    if (inst_.neg & 4) c = -c;')
        L.append('    float result = a * b + c;')
        L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
        L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_mad_mix_lo_hi(self, dst: list[str], src: list[str], is_lo: bool) -> str:
        """Generate V_MAD_MIXLO_F16 / V_MAD_MIXHI_F16."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L.append('    float a, b, c;')
        L.append(f'    if ({opsel_hi} & 1) a = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 1) ? (raw0 >> 16) : raw0));')
        L.append('    else a = std::bit_cast<float>(raw0);')
        L.append(f'    if ({opsel_hi} & 2) b = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 2) ? (raw1 >> 16) : raw1));')
        L.append('    else b = std::bit_cast<float>(raw1);')
        L.append(f'    if ({self._op_sel_hi_2_expr(self._enc_name)}) c = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 4) ? (raw2 >> 16) : raw2));')
        L.append('    else c = std::bit_cast<float>(raw2);')
        L.append('    if (inst_.neg & 1) a = -a;')
        L.append('    if (inst_.neg & 2) b = -b;')
        L.append('    if (inst_.neg & 4) c = -c;')
        L.append('    float result = a * b + c;')
        L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
        L.append(f'    uint16_t h = util::f32_to_f16(result);')
        L.append(f'    uint32_t prev = {d}.read_lane(wf, lane);')
        if is_lo:
            L.append(f'    {d}.write_lane(wf, lane, (prev & 0xFFFF0000u) | h);')
        else:
            L.append(f'    {d}.write_lane(wf, lane, (prev & 0x0000FFFFu) | (static_cast<uint32_t>(h) << 16));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_dot2(self, dst: list[str], src: list[str], cls: str) -> str:
        """Generate V_DOT2_F32_F16, V_DOT2_I32_I16, V_DOT2_U32_U16.

        Uses op_sel to select which 16-bit half of each source feeds
        element 0 (low) and element 1 (high) of the dot product.
        neg/neg_hi are split per element.
        """
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        opsel, opsel_hi = self._vop3p_opsel_exprs()
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
        L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
        L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
        L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
        L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')

        if cls == 'dot2_f32_f16':
            L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
            L.append('    float a1 = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
            L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
            L.append('    float b1 = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
            L.append('    if (inst_.neg & 1) a0 = -a0;')
            L.append('    if (inst_.neg & 2) b0 = -b0;')
            L.append('    if (inst_.neg_hi & 1) a1 = -a1;')
            L.append('    if (inst_.neg_hi & 2) b1 = -b1;')
            L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            L.append('    if (inst_.neg & 4) acc = -acc;')
            L.append('    float result = a0 * b0 + a1 * b1 + acc;')
            L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        elif cls == 'dot2_i32_i16':
            L.append('    int16_t a0 = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    int16_t a1 = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    int16_t b0 = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    int16_t b1 = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append('    int32_t result = static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1 + acc;')
            L.append('    if (inst_.clamp) result = std::clamp(result, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(result));')
        else:  # dot2_u32_u16
            L.append('    uint16_t a0 = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t a1 = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
            L.append('    uint16_t b0 = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
            L.append('    uint16_t b1 = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
            L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
            L.append('    uint32_t result = static_cast<uint32_t>(a0) * b0 + static_cast<uint32_t>(a1) * b1 + acc;')
            L.append(f'    {d}.write_lane(wf, lane, result);')

        L.append('  }')
        return '\n'.join(L)

    def _gen_dot4(self, dst: list[str], src: list[str], cls: str) -> str:
        """Generate V_DOT4_I32_I8 / V_DOT4_U32_U8."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

        if cls == 'dot4_i32_i8':
            L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append('    int32_t sum = acc;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append('      int8_t a = static_cast<int8_t>((raw0 >> (i * 8)) & 0xFF);')
            L.append('      int8_t b = static_cast<int8_t>((raw1 >> (i * 8)) & 0xFF);')
            L.append('      sum += static_cast<int32_t>(a) * b;')
            L.append('    }')
            L.append('    if (inst_.clamp) sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sum));')
        elif cls == 'dot4_f32_fp8':
            # FP8 dot product: D.f32 += sum(A.fp8[i] * B.fp8[i]) for i in 0..3
            L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append('      float a = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw0 >> (i * 8)) & 0xFF));')
            L.append('      float b = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw1 >> (i * 8)) & 0xFF));')
            L.append('      acc += a * b;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc));')
        else:  # dot4_u32_u8
            L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
            L.append('    uint32_t sum = acc;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append('      uint8_t a = static_cast<uint8_t>((raw0 >> (i * 8)) & 0xFF);')
            L.append('      uint8_t b = static_cast<uint8_t>((raw1 >> (i * 8)) & 0xFF);')
            L.append('      sum += static_cast<uint32_t>(a) * b;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, sum);')

        L.append('  }')
        return '\n'.join(L)

    def _gen_dot8(self, dst: list[str], src: list[str], cls: str) -> str:
        """Generate V_DOT8_I32_I4 / V_DOT8_U32_U4."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

        if cls == 'dot8_i32_i4':
            L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append('    int32_t sum = acc;')
            L.append('    for (int i = 0; i < 8; ++i) {')
            L.append('      int32_t a = static_cast<int32_t>((raw0 >> (i * 4)) & 0xF);')
            L.append('      if (a & 0x8) a |= ~0xF;')
            L.append('      int32_t b = static_cast<int32_t>((raw1 >> (i * 4)) & 0xF);')
            L.append('      if (b & 0x8) b |= ~0xF;')
            L.append('      sum += a * b;')
            L.append('    }')
            L.append('    if (inst_.clamp) sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sum));')
        else:  # dot8_u32_u4
            L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
            L.append('    uint32_t sum = acc;')
            L.append('    for (int i = 0; i < 8; ++i) {')
            L.append('      uint32_t a = (raw0 >> (i * 4)) & 0xF;')
            L.append('      uint32_t b = (raw1 >> (i * 4)) & 0xF;')
            L.append('      sum += a * b;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, sum);')

        L.append('  }')
        return '\n'.join(L)

    def _gen_accvgpr_read(self, dst: list[str], src: list[str]) -> str:
        """Generate V_ACCVGPR_READ: copy ACCVGPR → VGPR."""
        # In our model, ACCVGPRs are just VGPRs in the accumulator range.
        # The operand resolution already handles the mapping.
        return f'  uint64_t exec = wf.exec();\n' \
               f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n' \
               f'    if (!(exec & (1ULL << lane))) continue;\n' \
               f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n' \
               f'  }}'

    def _gen_accvgpr_write(self, dst: list[str], src: list[str]) -> str:
        """Generate V_ACCVGPR_WRITE: copy VGPR → ACCVGPR."""
        return f'  uint64_t exec = wf.exec();\n' \
               f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n' \
               f'    if (!(exec & (1ULL << lane))) continue;\n' \
               f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n' \
               f'  }}'

    def _gen_mfma(self, inst: Instruction, dst: list[str], src: list[str]) -> str:
        """Generate MFMA / SMFMAC matrix multiply-accumulate.

        Uses the mfma_exec.h helpers which implement the exact GFX9 register
        mapping formulas. The helpers handle cross-lane data movement, WAR
        hazard avoidance (buffered writes), and inline constant accumulator
        initialization without clobbering overlapping source operands.
        """
        name = inst.name
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]

        import re
        m = re.match(
            r'V_(?:S?MFMA[C]?|S?WMMA[C]?)_(F32|I32|F64|F16|BF16|BF8|FP8)_(\d+)X(\d+)X(\d+)'
            r'(?:_\d+B)?_?(F32|XF32|F16|BF16|I8|IU8|IU4|F64|FP8|BF8'
            r'|BF8_BF8|BF8_FP8|FP8_BF8|FP8_FP8'
            r'|F16_FP8|F16_BF8|BF16_FP8|BF16_BF8'
            r'|F8_F6_F4|F8F6F4)?'
            r'(?:_1K)?$',
            name)

        if not m:
            return (f'  // MFMA stub: {name}\n'
                    f'  (void)wf;\n'
                    f'  throw util::UnimplementedInst(mnemonic());')

        result_type = m.group(1)  # F32, I32, F64
        M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
        input_type = m.group(5)   # F32, XF32, F16, BF16, I8, F64, etc.

        dst_bits = inst.operands[0].size if inst.operands else 0
        dst_regs = max(1, dst_bits // 32)

        # SMFMAC: per-lane dot-product functional model (no cross-lane mapping).
        if 'SMFMAC' in name:
            L = []
            L.append(f'  // MFMA: {name} \u2014 {M}x{N}x{K} {input_type}\u2192{result_type}')
            L.append(f'  // D({dst_regs} regs/lane) += A * B, functional model')
            L.append(f'  uint64_t exec = wf.exec();')
            L.append(f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{')
            L.append(f'    if (!(exec & (1ULL << lane)))')
            L.append(f'      continue;')
            L.append(f'    uint32_t a_raw = {s0}.read_lane(wf, lane);')
            L.append(f'    uint32_t b_raw = {s1}.read_lane(wf, lane);')

            if input_type in ('F16', 'BF16'):
                conv = 'util::f16_to_f32' if input_type == 'F16' else 'util::bf16_to_f32'
                L.append(f'    float a0 = {conv}(static_cast<uint16_t>(a_raw));')
                L.append(f'    float a1 = {conv}(static_cast<uint16_t>(a_raw >> 16));')
                L.append(f'    float b0 = {conv}(static_cast<uint16_t>(b_raw));')
                L.append(f'    float b1 = {conv}(static_cast<uint16_t>(b_raw >> 16));')
                L.append(f'    float dot = a0 * b0 + a1 * b1;')
                L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));')
            elif input_type == 'I8':
                L.append(f'    int32_t dot = 0;')
                L.append(f'    for (int k = 0; k < 8; ++k) {{')
                L.append(f'      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);')
                L.append(f'      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);')
                L.append(f'      dot += static_cast<int32_t>(ae) * be;')
                L.append(f'    }}')
                L.append(f'    int32_t acc0 = static_cast<int32_t>({s2}.read_lane(wf, lane));')
                L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));')
                L.append(f'    // Additional result registers would require cross-lane data')
            else:
                # FP8/BF8 variants: input_type is e.g. "BF8_BF8", "BF8_FP8", etc.
                _FP8_CONV = {'BF8': 'util::bf8_e5m2_to_f32', 'FP8': 'util::fp8_e4m3_to_f32'}
                parts = input_type.split('_')
                conv_a = _FP8_CONV.get(parts[0], 'util::fp8_e4m3_to_f32')
                conv_b = _FP8_CONV.get(parts[1], 'util::fp8_e4m3_to_f32')
                L.append(f'    float dot = 0.0f;')
                L.append(f'    for (int k = 0; k < 4; ++k) {{')
                L.append(f'      float ae = {conv_a}(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));')
                L.append(f'      float be = {conv_b}(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));')
                L.append(f'      dot += ae * be;')
                L.append(f'    }}')
                L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
                L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));')

            L.append(f'  }}')
            return '\n'.join(L)

        # Compute number of blocks from output register count and matrix dims.
        if result_type == 'F64':
            B = 64 * (dst_regs // 2) // (M * N)
        else:
            B = 64 * dst_regs // (M * N)

        # Determine input element size in bits and extract functions.
        _INPUT_BITS = {
            'F32': 32, 'XF32': 32, 'F16': 16, 'BF16': 16,
            'I8': 8, 'IU8': 8, 'IU4': 4, 'F64': 64,
            'FP8': 8, 'BF8': 8,
            'FP8_FP8': 8, 'FP8_BF8': 8, 'BF8_FP8': 8, 'BF8_BF8': 8,
            'F16_FP8': 8, 'F16_BF8': 8, 'BF16_FP8': 8, 'BF16_BF8': 8,
            'F8_F6_F4': 8, 'F8F6F4': 8,
        }
        in_bits = _INPUT_BITS.get(input_type, 32)

        # Map input types to amdgpu::extract_* function names.
        _EXTRACT_A = {
            'F32': 'amdgpu::extract_f32', 'XF32': 'amdgpu::extract_f32',
            'F16': 'amdgpu::extract_f16', 'BF16': 'amdgpu::extract_bf16',
            'FP8_FP8': 'amdgpu::extract_fp8', 'FP8_BF8': 'amdgpu::extract_fp8',
            'BF8_FP8': 'amdgpu::extract_bf8', 'BF8_BF8': 'amdgpu::extract_bf8',
            'F8_F6_F4': 'amdgpu::extract_fp8', 'F8F6F4': 'amdgpu::extract_fp8',
        }
        _EXTRACT_B = {
            'F32': 'amdgpu::extract_f32', 'XF32': 'amdgpu::extract_f32',
            'F16': 'amdgpu::extract_f16', 'BF16': 'amdgpu::extract_bf16',
            'FP8_FP8': 'amdgpu::extract_fp8', 'FP8_BF8': 'amdgpu::extract_bf8',
            'BF8_FP8': 'amdgpu::extract_fp8', 'BF8_BF8': 'amdgpu::extract_bf8',
            'F8_F6_F4': 'amdgpu::extract_fp8', 'F8F6F4': 'amdgpu::extract_fp8',
        }

        L = []
        L.append(f'  auto &cu = wf.cu();')
        L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
        # acc_cd field exists in CDNA2/3/4 VOP3P_MFMA encoding (controls
        # AccVGPR bank selection). CDNA1 and RDNA lack this field — default
        # to 1 (always use AccVGPR bank, the CDNA1 behavior).
        arch = self.isa_spec.arch_name.lower()
        has_acc_cd = arch in ('cdna2', 'cdna3', 'cdna4')
        if has_acc_cd:
            L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd);')
        else:
            L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, 1);')
        L.append(f'  uint32_t const_acc;')
        L.append(f'  uint32_t s2 = amdgpu::resolve_acc(vb, dst,')
        L.append(f'      {s2}.encoding_value_, const_acc,'
                 f' [&] {{ return {s2}.read_scalar(wf); }});')

        if result_type == 'F64':
            L.append(f'  amdgpu::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
            L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                 s2, const_acc);')
        elif result_type == 'I32':
            L.append(f'  amdgpu::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
            L.append(f'                     amdgpu::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                     amdgpu::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                     s2, const_acc);')
        else:
            # F32, F16, BF16 result types all use exec_f32 (accumulate in f32,
            # WMMA F16/BF16 results are truncated at writeback — handled by the
            # register layout, not by separate exec functions).
            ea = _EXTRACT_A.get(input_type, 'amdgpu::extract_f32')
            eb = _EXTRACT_B.get(input_type, 'amdgpu::extract_f32')
            # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for
            # A-matrix broadcast and B-matrix lane permutation. RDNA does
            # not have MFMA (only WMMA), so these fields don't exist.
            has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
            L.append(f'  amdgpu::exec_f32(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
            L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
            if has_blgp:
                L.append(f'                 s2, {ea}, {eb}, const_acc,')
                L.append(f'                 inst_.cbsz, inst_.abid, inst_.blgp);')
            else:
                L.append(f'                 s2, {ea}, {eb}, const_acc);')

            if input_type in ('F8_F6_F4', 'F8F6F4'):
                # Rewrite the MFMA call: if ABID[0]=1 (scaling enabled),
                # use exec_f32_scaled which applies per-32-K-block E8M0
                # exponent biases from scale VGPRs in the X2 prefix.
                # Dwords 0-1 of the VOP3PX2 encoding are at inst[-2]/[-1]
                # relative to the MFMA encoding pointer.
                # Dword 1 bits [8:0] = scale_src0, bits [17:9] = scale_src1.
                L_scaled = []
                L_scaled.append('  if (inst_.abid & 1u) {')
                L_scaled.append('    auto *raw = reinterpret_cast<const uint32_t *>(&inst_);')
                L_scaled.append('    uint32_t x2_dw1 = raw[-1];')
                L_scaled.append('    uint32_t scale_src0_enc = x2_dw1 & 0x1FFu;')
                L_scaled.append('    uint32_t scale_src1_enc = (x2_dw1 >> 9) & 0x1FFu;')
                L_scaled.append('    uint32_t sa_base = amdgpu::src_base(vb, scale_src0_enc);')
                L_scaled.append('    uint32_t sb_base = amdgpu::src_base(vb, scale_src1_enc);')
                L_scaled.append(f'    amdgpu::exec_f32_scaled(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
                L_scaled.append(f'        amdgpu::src_base(vb, {s0}.encoding_value_),')
                L_scaled.append(f'        amdgpu::src_base(vb, {s1}.encoding_value_),')
                L_scaled.append(f'        s2, {ea}, {eb}, const_acc,')
                L_scaled.append(f'        inst_.cbsz, inst_.abid, inst_.blgp, sa_base, sb_base);')
                L_scaled.append('  }')
                # Replace the unscaled MFMA call with a conditional:
                # if ABID[0]=1 use scaled path, else the existing unscaled path.
                # Find and wrap the existing exec_f32 call in an else block.
                for i, line in enumerate(L):
                    if 'amdgpu::exec_f32(' in line:
                        L.insert(i, '  if (!(inst_.abid & 1u)) {')
                        # Find the closing semicolon
                        for j in range(i + 1, len(L)):
                            if L[j].rstrip().endswith(';'):
                                L.insert(j + 1, '  } else {')
                                break
                        break
                L.extend(L_scaled)
                L.append('  }')

        return '\n'.join(L)

    def _gen_smem_load(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        nd = sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;')
        L.append(f'  d->num_dwords = {nd};')
        L.append('  d->is_load = true;')
        L.append(f'  d->mtype = {self._mtype_expr(is_smem=True)};')
        L.append('  d->addr = smem_calculate_address(inst_, wf);')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_smem_store(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        nd = sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->num_dwords = {nd};')
        L.append('  d->is_load = false;')
        L.append(f'  d->mtype = {self._mtype_expr(is_smem=True)};')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;')
        L.append(f'  for (uint32_t i = 0; i < {nd}; ++i)')
        L.append('    d->store_data[i] = cu.read_sgpr(sdata_base + i);')
        L.append('  d->addr = smem_calculate_address(inst_, wf);')
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
            if model in (MemoryCoherencyModel.GFX9_GLC,
                         MemoryCoherencyModel.GFX940_SC0_SC1_NT):
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
            'smem_load': 'amdgpu::WaitCounterType::KMCNT' if is_gfx11_plus
                         else 'amdgpu::WaitCounterType::LGKMCNT',
            'smem_store': 'amdgpu::WaitCounterType::KMCNT' if is_gfx11_plus
                          else 'amdgpu::WaitCounterType::LGKMCNT',
            'flat_load': 'amdgpu::WaitCounterType::LOADCNT' if is_gfx11_plus
                         else 'amdgpu::WaitCounterType::VMCNT',
            'flat_store': 'amdgpu::WaitCounterType::STORECNT' if is_gfx11_plus
                          else ('amdgpu::WaitCounterType::VSCNT'
                                if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                                else 'amdgpu::WaitCounterType::VMCNT'),
            'flat_atomic': 'amdgpu::WaitCounterType::LOADCNT' if is_gfx11_plus
                           else 'amdgpu::WaitCounterType::VMCNT',
            'buffer_load': 'amdgpu::WaitCounterType::LOADCNT' if is_gfx11_plus
                           else 'amdgpu::WaitCounterType::VMCNT',
            'buffer_store': 'amdgpu::WaitCounterType::STORECNT' if is_gfx11_plus
                            else ('amdgpu::WaitCounterType::VSCNT'
                                  if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                                  else 'amdgpu::WaitCounterType::VMCNT'),
            'tbuffer_load': 'amdgpu::WaitCounterType::LOADCNT' if is_gfx11_plus
                            else 'amdgpu::WaitCounterType::VMCNT',
            'tbuffer_store': 'amdgpu::WaitCounterType::STORECNT' if is_gfx11_plus
                             else ('amdgpu::WaitCounterType::VSCNT'
                                   if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                                   else 'amdgpu::WaitCounterType::VMCNT'),
            'global_load': 'amdgpu::WaitCounterType::LOADCNT' if is_gfx11_plus
                           else 'amdgpu::WaitCounterType::VMCNT',
            'global_store': 'amdgpu::WaitCounterType::STORECNT' if is_gfx11_plus
                            else ('amdgpu::WaitCounterType::VSCNT'
                                  if model == MemoryCoherencyModel.GFX10_GLC_DLC_SLC
                                  else 'amdgpu::WaitCounterType::VMCNT'),
            'ds_read': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                       else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read2': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                        else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_write': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                        else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_write2': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                         else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_atomic': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                         else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read_addtid': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                              else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_write_addtid': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                               else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read_tr_b16': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                              else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read_tr_b8': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                             else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read_tr_b4': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                             else 'amdgpu::WaitCounterType::LGKMCNT',
            'ds_read_tr_b6': 'amdgpu::WaitCounterType::DSCNT' if is_gfx11_plus
                             else 'amdgpu::WaitCounterType::LGKMCNT',
        }
        return _MAP.get(sem_class)

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

    def _gen_flat_load(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        if sem.d16_hi:
            L.append('  d->d16_hi = true;')
        if sem.d16_lo:
            L.append('  d->d16_lo = true;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_flat_store(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        data_field = self.isa_spec.profile.flat_store_src_field
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz == 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field} + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 4);')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field}, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field}, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    _ATOMIC_OP_ENUM: dict[str, str] = {
        'swap': 'amdgpu::AtomicOp::SWAP',
        'cmpswap': 'amdgpu::AtomicOp::CMPSWAP',
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
    }

    def _gen_flat_atomic(self, dst: list[str], src: list[str],
                         sem: InstructionSemantics) -> str:
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        acc = self._acc_vgpr_expr
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = ({sc0} != 0);')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        data_field = self.isa_spec.profile.flat_store_src_field
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.{data_field} + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_atomic(self, dst: list[str], src: list[str],
                           sem: InstructionSemantics) -> str:
        """Generate buffer_atomic execute() body (MUBUF encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled buffer_atomic variant ({sem.name})'

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1

        L = []
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdata;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append(f'  d->is_load = ({sc0} != 0);')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  mubuf_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_atomic(self, dst: list[str], src: list[str],
                       sem: InstructionSemantics) -> str:
        """Generate ds_atomic execute() body (DS encoding)."""
        if sem.operation is None or sem.operation not in self._ATOMIC_OP_ENUM:
            return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant ({sem.name})'

        op_enum = self._ATOMIC_OP_ENUM[sem.operation]
        esz = sem.elem_size or 4
        data_dwords = sem.num_elems or 1

        L = []
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        # DS atomics always return the old value (like GLC=1).
        L.append('  d->is_load = true;')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_load(self, dst: list[str], src: list[str], sem: InstructionSemantics,
                         cls: str = 'buffer_load', inst: 'Instruction | None' = None) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        addr_fn = 'mtbuf_calculate_addresses' if cls == 'tbuffer_load' else 'mubuf_calculate_addresses'
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
            L.append('    auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
            L.append(f'    d->elem_size = {esz};')
            L.append(f'    d->num_elems = {ne};')
            L.append('    d->is_load = true;')
            L.append('    d->lds_dst = true;')
            L.append('    d->lds_base = wf.m0();')
            L.append(f'    d->mtype = {self._mtype_expr()};')
            L.append(f'    d->non_temporal = {nt};')
            L.append(f'    {addr_fn}(inst_, wf, *d);')
            L.append('    set_data(std::move(d));')
            L.append('    return;')
            L.append('  }')
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdata;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
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

    def _gen_buffer_store(self, dst: list[str], src: list[str], sem: InstructionSemantics, cls: str = 'buffer_store') -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        sc0, sc1, nt = self._coherency_exprs()
        acc = self._acc_vgpr_expr
        addr_fn = 'mtbuf_calculate_addresses' if cls == 'tbuffer_store' else 'mubuf_calculate_addresses'
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append(f'  {addr_fn}(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz >= 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, {esz});')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.vdata, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        if sem.d16_hi:
            L.append('  d->d16_hi = true;')
        if sem.d16_lo:
            L.append('  d->d16_lo = true;')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read_addtid(self, dst: list[str], src: list[str],
                            sem: InstructionSemantics) -> str:
        """ds_read_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = true;')
        L.append('  {')
        L.append('    uint64_t exec = wf.exec();')
        L.append('    d->lane_mask = exec; d->exec_mask = exec;')
        L.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        L.append('    d->cu_path = wf.cu().full_path();')
        L.append('    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;')
        L.append('    uint32_t m0 = wf.m0();')
        L.append('    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;')
        L.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('      if (!(exec & (1ULL << lane))) continue;')
        L.append('      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();')
        L.append('    }')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write_addtid(self, dst: list[str], src: list[str],
                             sem: InstructionSemantics) -> str:
        """ds_write_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = false;')
        L.append('  {')
        L.append('    uint64_t exec = wf.exec();')
        L.append('    d->lane_mask = exec; d->exec_mask = exec;')
        L.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        L.append('    d->cu_path = wf.cu().full_path();')
        L.append('    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;')
        L.append('    uint32_t m0 = wf.m0();')
        L.append('    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;')
        L.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('      if (!(exec & (1ULL << lane))) continue;')
        L.append('      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();')
        L.append('    }')
        L.append('  }')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f'  d->store_data.resize(wf.wf_size() * {sem.elem_size});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);')
        L.append(f'    std::memcpy(&d->store_data[lane * {sem.elem_size}], &val0, {sem.elem_size});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read_tr(self, dst: list[str], src: list[str],
                        sem: InstructionSemantics) -> str:
        """ds_read_b64_tr_b16 etc: DS read + cross-lane transpose post-processing.

        Uses the standard DS read pipeline (MEMORY_OP) with elem_size=4,
        num_elems=2 (for B64) or 3 (for B96). Sets d->transpose to signal
        the memory pipeline to apply the cross-lane shuffle after the raw read.
        """
        # TR_B4=1, TR_B6=2, TR_B8=3, TR_B16=4
        tr_map = {
            'ds_read_tr_b4': (4, 2, 1),   # elem_size=4, num_elems=2, transpose=1
            'ds_read_tr_b6': (4, 3, 2),   # elem_size=4, num_elems=3, transpose=2
            'ds_read_tr_b8': (4, 2, 3),   # elem_size=4, num_elems=2, transpose=3
            'ds_read_tr_b16': (4, 2, 4),  # elem_size=4, num_elems=2, transpose=4
        }
        esz, ne, tr_kind = tr_map.get(sem.semantic_class, (4, 2, 4))
        L = []
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        L.append(f'  d->transpose = {tr_kind};')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        acc = self._acc_vgpr_expr
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            off = i * esz
            if esz == 8:
                vgpr_base = i * 2
                L.append(f'    uint32_t lo{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {vgpr_base}, lane);')
                L.append(f'    uint32_t hi{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {vgpr_base + 1}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &lo{i}, 4);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off + 4}], &hi{i}, 4);')
            elif esz == 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 4);')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(f'    d->store_data[lane * {stride} + {off}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read2(self, dst: list[str], src: list[str],
                      sem: InstructionSemantics) -> str:
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = true;')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->ds2_active = true;')
        L.append(f'  d->ds2_dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst + {dwords_per_access};')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append('    uint32_t base = cu.read_vgpr(wf.vgpr_alloc().base + inst_.addr, lane);')
        L.append(f'    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();')
        L.append(f'    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write2(self, dst: list[str], src: list[str],
                       sem: InstructionSemantics) -> str:
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = false;')
        L.append('  d->exec_mask = exec;')
        L.append('  d->lane_mask = exec;')
        L.append('  d->ds2_active = true;')
        L.append(f'  d->store_data.resize(wf.wf_size() * {esz});')
        L.append(f'  d->ds2_store_data.resize(wf.wf_size() * {esz});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append('    uint32_t base = cu.read_vgpr(wf.vgpr_alloc().base + inst_.addr, lane);')
        L.append(f'    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();')
        L.append(f'    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();')
        # Pack data0 into store_data
        for i in range(dwords_per_access):
            L.append(f'    uint32_t v0_{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data0 + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {esz} + {i * 4}], &v0_{i}, 4);')
        # Pack data1 into ds2_store_data
        for i in range(dwords_per_access):
            L.append(f'    uint32_t v1_{i} = cu.read_vgpr(wf.vgpr_alloc().base + {acc} + inst_.data1 + {i}, lane);')
            L.append(f'    std::memcpy(&d->ds2_store_data[lane * {esz} + {i * 4}], &v1_{i}, 4);')
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
    _NON_SHAREABLE_CLASSES = frozenset({
        # Profile-dependent (ISA-specific coherency/mtype calls):
        'smem_load', 'smem_store',
        'flat_load', 'flat_store', 'flat_atomic',
        'buffer_load', 'buffer_store', 'buffer_atomic',
        'tbuffer_load', 'tbuffer_store',
        'ds_read', 'ds_read2', 'ds_write', 'ds_write2', 'ds_atomic',
        'global_load', 'global_store',
        'dcache_inv', 'dcache_wb',
        'image_load', 'image_store', 'image_atomic', 'image_sample',
        'image_query',
        # Nop/stub bodies don't benefit from sharing:
        'nop',
        # ISA-dependent control flow (reference Isa:: constants or size_):
        'waitcnt', 'wait_counter',
        'endpgm', 'branch', 'cbranch',
        'scalar_getpc', 'scalar_setpc', 'scalar_swappc', 'scalar_call',
        # MFMA/WMMA reference ISA-specific headers:
        'mfma',
        # Interp/export use ISA-specific encoding struct fields:
        'interp', 'export',
        # AccVGPR read/write use ISA-specific register file:
        'accvgpr_read', 'accvgpr_write',
        # Vector swap accesses protected inst_ member:
        'vector_swap',
        # Vector readlane/writelane/readfirstlane access encoding fields:
        'vector_readlane', 'vector_writelane', 'vector_readfirstlane',
        # V_CMPX writes VCC+EXEC on CDNA but only EXEC on RDNA:
        'vector_cmpx', 'vector_cmpx_class',
    })

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
                is_smem = enc.enc_name.upper() == 'ENC_SMEM'
                has_sem = self._enc_has_semantics(enc)
                # Check child encodings for semantics too.
                for child in child_encs.get(enc.enc_name, []):
                    if self._enc_has_semantics(child):
                        has_sem = True
                for inst in all_insts:
                    # Resolve the instruction's own encoding field names.
                    # Instructions from alternate sub-encodings (e.g.,
                    # VOP3_SDST_ENC under ENC_VOP3) carry their original
                    # enc_name and inherit from the sub-encoding's C++
                    # class whose OpEncoding typedef matches the sub-
                    # encoding's MachineInst struct.  Look up the
                    # instruction's own encoding to get the correct field
                    # set.
                    inst_enc_obj = self.isa_spec.encoding_map.get(
                        inst.enc_name
                    )
                    if (
                        inst_enc_obj is not None
                        and inst_enc_obj is not enc
                        and not inst.is_implied_literal_enc
                    ):
                        inst_field_names = (
                            enc_field_names
                            | {f.name for f in inst_enc_obj.ucode_fields}
                        )
                    else:
                        inst_field_names = enc_field_names
                    class_members = []
                    public_members = [cgen.Line('public:')]
                    private_members = []
                    opnd_ctor_init = []
                    opnd_body = []
                    src_idx = 0
                    dst_idx = 0
                    reads_dst = self._dst_is_also_source(inst)
                    for opnd in inst.operands:
                        if opnd.is_input:
                            opnd_body.append(
                                f'src_operands_[{src_idx}] = &{opnd.name};'
                            )
                            src_idx += 1
                        elif (reads_dst and opnd.is_output
                              and opnd.name in ('vdst', 'sdst')):
                            opnd_body.append(
                                f'src_operands_[{src_idx}] = &{opnd.name};'
                            )
                            src_idx += 1
                        if opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_[{dst_idx}] = &{opnd.name};'
                            )
                            dst_idx += 1
                        if not opnd.is_input and not opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_[{dst_idx}] = &{opnd.name};'
                            )
                            dst_idx += 1
                        private_members.append(
                            cgen.Statement(f'Operand {opnd.name}')
                        )
                        if is_smem and opnd.name == 'soffset':
                            opnd_ctor_init.append(
                                f'{opnd.name}(make_smem_offset('
                                f'reinterpret_cast<const OpEncoding*>(inst)))'
                            )
                        elif opnd.name in inst_field_names:
                            opr_type = opnd.operand_type
                            inst_sem = (self.semantics.instructions.get(inst.name)
                                        if self.semantics else None)
                            if (inst_sem and inst_sem.accvgpr_srcs
                                    and opnd.is_input):
                                opr_type = 'OPR_SRC_VGPR_OR_ACCVGPR'
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd.size}, '
                                f'OperandType::{opr_type}, '
                                f'reinterpret_cast<const OpEncoding*>(inst)'
                                f'->{opnd.name})'
                            )
                        else:
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd.size}, '
                                f'OperandType::{opnd.operand_type}, 0)'
                            )
                    class_ctor_decl = cgen.FunctionDeclaration(
                        cgen.Value('', inst.fmt_name),
                        [cgen.Value('const MachineInst *', 'inst')],
                    )
                    public_members.append(class_ctor_decl)
                    public_members.append(
                        cgen.Statement(
                            'void execute_impl(amdgpu::Wavefront &wf)'
                        )
                    )
                    # Embed the full mnemonic (with suffix) as a string literal
                    # so the encoding base gets a string_view to static storage.
                    rule = self.isa_spec.profile.mnemonic_rule(enc.enc_name)
                    full_mnemonic = inst.mnemonic + (rule.suffix or '')
                    init_list_parts = [
                        f'{inst.fmt_true_enc_name}("{full_mnemonic}", '
                        f'reinterpret_cast<const OpEncoding*>(inst), '
                        f'make_exec_fn<{inst.fmt_name}>())'
                    ] + opnd_ctor_init
                    init_list = ', '.join(init_list_parts)
                    # Check if this is a memory instruction to set MEMORY_OP flag
                    _mem_sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics else None
                    )
                    _MEM_CLASSES = frozenset({
                        'smem_load', 'smem_store',
                        'flat_load', 'flat_store', 'flat_atomic',
                        'buffer_load', 'buffer_store', 'buffer_atomic',
                        'tbuffer_load', 'tbuffer_store',
                        'ds_read', 'ds_read2', 'ds_write', 'ds_write2', 'ds_atomic',
                        'ds_read_addtid', 'ds_write_addtid',
                        'ds_read_tr_b16', 'ds_read_tr_b8', 'ds_read_tr_b4', 'ds_read_tr_b6',
                    })
                    ctor_body_parts = list(opnd_body)
                    ctor_body_parts.append(f'num_src_ = {src_idx};')
                    ctor_body_parts.append(f'num_dst_ = {dst_idx};')

                    # Literal constant fixup: when src0/ssrc0/ssrc1 == 255,
                    # replace the operand with the 32-bit literal from the
                    # extended instruction encoding.
                    _LIT_ENC_MAP = {
                        'ENC_SOP1': ('Sop1InstLiteralMachineInst', ['ssrc0']),
                        'ENC_SOP2': ('Sop2InstLiteralMachineInst', ['ssrc0', 'ssrc1']),
                        'ENC_SOPC': ('SopcInstLiteralMachineInst', ['ssrc0', 'ssrc1']),
                        'ENC_VOP1': ('Vop1InstLiteralMachineInst', ['src0']),
                        'ENC_VOP2': ('Vop2InstLiteralMachineInst', ['src0']),
                        'ENC_VOPC': ('VopcInstLiteralMachineInst', ['src0']),
                    }
                    _lit_info = _LIT_ENC_MAP.get(enc.enc_name.upper())
                    if _lit_info:
                        _lit_struct, _lit_fields = _lit_info
                        for opnd in inst.operands:
                            if opnd.name in _lit_fields and opnd.name in enc_field_names:
                                ctor_body_parts.append(
                                    f'if (reinterpret_cast<const OpEncoding*>(inst)->{opnd.name} == 255) '
                                    f'{opnd.name} = Operand({opnd.size}, OperandType::OPR_SIMM32, '
                                    f'static_cast<int>(reinterpret_cast<const {_lit_struct}*>(inst)->simm32));'
                                )

                    # DPP fixup: when src0 == amdgpu::SRC_DPP (DPP marker), replace the
                    # src0 operand with vsrc0 from the DPP extension dword.
                    # This lets the instruction execute normally with the
                    # correct VGPR source. Lane permutation is not yet
                    # applied (identity permutation).
                    # DPP/SDWA: src0 marker values 250 (DPP) and 249 (SDWA)
                    # indicate the real VGPR index is in the extension dword.
                    # CDNA uses VopDpp, RDNA uses VopDpp16 (both have vsrc0).
                    _DPP_ENC_BASES = {'ENC_VOP1': 'Vop1', 'ENC_VOP2': 'Vop2'}
                    _enc_base = _DPP_ENC_BASES.get(enc.enc_name.upper())
                    if _enc_base:
                        # CDNA (GFX9) uses VopDpp; RDNA (GFX10+) uses VopDpp16.
                        _is_rdna = any(
                            ie.enc_name.startswith('VOP1_VOP_DPP16')
                            for ie in self.isa_spec.inst_encodings
                        )
                        _dpp_suffix = 'VopDpp16' if _is_rdna else 'VopDpp'
                        _dpp_struct = f'{_enc_base}{_dpp_suffix}MachineInst'
                        for opnd in inst.operands:
                            if opnd.name == 'src0' and opnd.name in enc_field_names:
                                # DPP (src0 == amdgpu::SRC_DPP): read vsrc0 and DPP control
                                # fields from the ISA-specific extension dword,
                                # storing them on the Instruction base for
                                # apply_dpp() to use later.
                                ctor_body_parts.append(
                                    f'if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_DPP) {{'
                                    f' auto *dp = reinterpret_cast<const {_dpp_struct}*>(inst);'
                                    f' src0 = Operand({opnd.size}, OperandType::OPR_VGPR, dp->vsrc0);'
                                    f' dpp_ctrl_ = dp->dpp_ctrl;'
                                    f' dpp_row_mask_ = dp->row_mask;'
                                    f' dpp_bank_mask_ = dp->bank_mask;'
                                    f' dpp_bound_ctrl_ = dp->bound_ctrl;'
                                    f'}}'
                                )
                                # SDWA (src0 == amdgpu::SRC_SDWA): CDNA and RDNA1/2 only.
                                _has_sdwa = any(
                                    'SDWA' in ie.enc_name
                                    for ie in self.isa_spec.inst_encodings
                                )
                                if _has_sdwa:
                                    _sdwa_struct = f'{_enc_base}VopSdwaMachineInst'
                                    ctor_body_parts.append(
                                        f'if (reinterpret_cast<const OpEncoding*>(inst)->src0 == amdgpu::SRC_SDWA) {{'
                                        f' auto *sw = reinterpret_cast<const {_sdwa_struct}*>(inst);'
                                        f' src0 = Operand({opnd.size}, OperandType::OPR_VGPR, sw->vsrc0);'
                                        f' sdwa_src0_sel_ = sw->src0_sel;'
                                        f' sdwa_src0_sext_ = sw->src0_sext;'
                                        f' sdwa_src1_sel_ = sw->src1_sel;'
                                        f' sdwa_src1_sext_ = sw->src1_sext;'
                                        f' sdwa_dst_sel_ = sw->dst_sel;'
                                        f' sdwa_dst_unused_ = sw->dst_unused;'
                                        f' sdwa_clamp_ = sw->clamp;'
                                        f'}}'
                                    )

                    # Implied literal fixup: FMAMK/FMAAK always carry an
                    # inline 32-bit literal even when the ISA spec omits the
                    # simm32 operand. Add a simm32_ member to hold it.
                    _FMAMK_FMAAK = frozenset({
                        'vector_fmamk', 'vector_fmaak',
                    })
                    _has_simm32 = any(
                        op.name == 'simm32' for op in inst.operands
                    )
                    if (
                        _mem_sem
                        and _mem_sem.semantic_class in _FMAMK_FMAAK
                        and not _has_simm32
                        and _lit_info
                    ):
                        _lit_struct = _lit_info[0]
                        private_members.append(
                            cgen.Statement('uint32_t simm32_')
                        )
                        opnd_ctor_init.append('simm32_(0)')
                        init_list_parts.append('simm32_(0)')
                        init_list = ', '.join(init_list_parts)
                        ctor_body_parts.append(
                            f'simm32_ = reinterpret_cast<const '
                            f'{_lit_struct}*>(inst)->simm32;'
                        )

                    if _mem_sem and _mem_sem.semantic_class in _MEM_CLASSES:
                        ctor_body_parts.append('flags_ |= MEMORY_OP;')

                    _waitcnt_names = {
                        'S_WAITCNT', 'S_WAIT_LOADCNT', 'S_WAIT_STORECNT',
                        'S_WAIT_EXPCNT', 'S_WAIT_DSCNT', 'S_WAIT_KMCNT',
                        'S_WAIT_SAMPLECNT', 'S_WAIT_BVHCNT',
                        'S_WAIT_LOADCNT_DSCNT', 'S_WAIT_STORECNT_DSCNT',
                        'S_WAIT_IDLE', 'S_WAIT_ALU', 'S_WAIT_EVENT',
                        'S_WAITCNT_VSCNT', 'S_WAITCNT_VMCNT',
                        'S_WAITCNT_LGKMCNT', 'S_WAITCNT_EXPCNT',
                        'S_WAITCNT_DEPCTR',
                    }
                    _barrier_names = {
                        'S_BARRIER', 'S_BARRIER_SIGNAL', 'S_BARRIER_WAIT',
                    }
                    if inst.name in _waitcnt_names:
                        ctor_body_parts.append('flags_ |= WAITCNT;')
                    if inst.name in _barrier_names:
                        ctor_body_parts.append('flags_ |= BARRIER;')

                    if (inst.name.startswith('V_MFMA_')
                            or inst.name.startswith('V_SMFMAC_')):
                        ctor_body_parts.append('flags_ |= MFMA;')

                    if inst.name in {'V_ACCVGPR_WRITE_B32',
                                     'V_ACCVGPR_READ_B32',
                                     'V_ACCVGPR_MOV_B32'}:
                        ctor_body_parts.append('flags_ |= ACCVGPR;')

                    # Per-instruction size overrides (e.g., VOP3PX2 128-bit
                    # instructions decoded under 64-bit VOP3P_MFMA).
                    _size_overrides = self.isa_spec.profile.inst_size_overrides
                    if inst.name in _size_overrides:
                        ctor_body_parts.append(
                            f'size_ = {_size_overrides[inst.name]};')

                    class_ctor_impl_str = (
                        f'{inst.fmt_name}::'
                        f'{inst.fmt_name}(const MachineInst *inst) '
                        f': {init_list} '
                        f'{{{"".join(ctor_body_parts)}}}'
                    )
                    class_ctor_impl = cgen.Line(class_ctor_impl_str)
                    class_members.extend(public_members)
                    class_members.extend(private_members)
                    s = cgen.Struct(
                        f'{inst.fmt_name} : public {inst.fmt_true_enc_name}',
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
                        _dpp_preamble = ''
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                            _src0_name = next(
                                (o.name for o in inst.operands if o.is_input),
                                None
                            )
                            _src_inputs = [o.name for o in inst.operands if o.is_input]
                            _src1_name = _src_inputs[1] if len(_src_inputs) > 1 else None
                            _dpp_preamble = (
                                '  uint32_t sdwa_old_dst_[64] = {};\n'
                                '  if (sdwa_dst_sel_ != amdgpu::sdwa::DWORD) {\n'
                                '    uint32_t vb = wf.vgpr_alloc().base;\n'
                                '    uint64_t ex = wf.exec();\n'
                                '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln)\n'
                                '      if (ex & (1ULL << ln))\n'
                                '        sdwa_old_dst_[ln] = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n'
                                '  }\n'
                                '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                '    amdgpu::dpp::apply_dpp(src_operands_[0], dpp_ctrl_,\n'
                                '        dpp_row_mask_, dpp_bank_mask_, dpp_bound_ctrl_,\n'
                                '        dpp_src0_, wf);\n'
                                '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_src0_sel_ != amdgpu::sdwa::DWORD) {\n'
                                '    auto &cu = wf.cu();\n'
                                '    uint32_t ws = wf.wf_size();\n'
                                '    uint32_t vb = wf.vgpr_alloc().base + src_operands_[0]->encoding_value_;\n'
                                '    uint32_t result[64];\n'
                                '    for (uint32_t i = 0; i < ws; ++i)\n'
                                '      result[i] = amdgpu::sdwa::sdwa_src_select(\n'
                                '          cu.read_vgpr(vb, i), sdwa_src0_sel_, sdwa_src0_sext_);\n'
                                '    dpp_src0_ = std::make_unique<DppOperand>(\n'
                                '        *src_operands_[0], result, static_cast<int>(ws));\n'
                                '    src_operands_[0] = dpp_src0_.get();\n'
                                '  }\n'
                                '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_src1_sel_ != amdgpu::sdwa::DWORD && num_src_ > 1) {\n'
                                '    auto &cu = wf.cu();\n'
                                '    uint32_t ws = wf.wf_size();\n'
                                '    uint32_t vb = wf.vgpr_alloc().base + src_operands_[1]->encoding_value_;\n'
                                '    uint32_t result1[64];\n'
                                '    for (uint32_t i = 0; i < ws; ++i)\n'
                                '      result1[i] = amdgpu::sdwa::sdwa_src_select(\n'
                                '          cu.read_vgpr(vb, i), sdwa_src1_sel_, sdwa_src1_sext_);\n'
                                '    dpp_src1_ = std::make_unique<DppOperand>(\n'
                                '        *src_operands_[1], result1, static_cast<int>(ws));\n'
                                '    src_operands_[1] = dpp_src1_.get();\n'
                                '  }\n'
                                + (f'  if (dpp_src0_) {_src0_name}.set_delegate(dpp_src0_.get());\n'
                                   if _src0_name else '')
                                + (f'  if (dpp_src1_) {_src1_name}.set_delegate(dpp_src1_.get());\n'
                                   if _src1_name else '')
                            )
                        # SDWA postamble: apply dst_sel merge and float clamp after ALU.
                        _sdwa_postamble = ''
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                            is_float_op = (sem and sem.data_type in ('f16', 'f32', 'f64'))
                            _sdwa_postamble = (
                                '  if (sdwa_dst_sel_ != amdgpu::sdwa::DWORD) {\n'
                                '    uint64_t ex = wf.exec();\n'
                                '    uint32_t vb = wf.vgpr_alloc().base;\n'
                                '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n'
                                '      if (!(ex & (1ULL << ln))) continue;\n'
                                '      uint32_t dv = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n'
                                '      dv = amdgpu::sdwa::sdwa_dst_merge(dv, sdwa_old_dst_[ln], sdwa_dst_sel_, sdwa_dst_unused_);\n'
                                '      wf.cu().write_vgpr(vb + inst_.vdst, ln, dv);\n'
                                '    }\n'
                                '  }\n'
                            )
                            if is_float_op:
                                _sdwa_postamble += (
                                    '  if (sdwa_clamp_) {\n'
                                    '    uint64_t ex = wf.exec();\n'
                                    '    uint32_t vb = wf.vgpr_alloc().base;\n'
                                    '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n'
                                    '      if (!(ex & (1ULL << ln))) continue;\n'
                                    '      uint32_t dv = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n'
                                    '      float fv = std::bit_cast<float>(dv);\n'
                                    '      fv = std::clamp(fv, 0.0f, 1.0f);\n'
                                    '      wf.cu().write_vgpr(vb + inst_.vdst, ln, std::bit_cast<uint32_t>(fv));\n'
                                    '    }\n'
                                    '  }\n'
                                )
                        _dpp_cleanup = ''
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                            if _src0_name:
                                _dpp_cleanup += f'  {_src0_name}.clear_delegate();\n'
                            if _src1_name:
                                _dpp_cleanup += f'  {_src1_name}.clear_delegate();\n'
                        # Skip DPP/SDWA preamble and cleanup for unimplemented
                        # instructions whose body is ONLY a throw — the cleanup
                        # code after the throw would be unreachable. Only match
                        # pure-throw bodies, not bodies with conditional throws.
                        body_stripped = body.strip().rstrip(';').strip()
                        body_throws = body_stripped.startswith('(void)wf;') and 'throw util::UnimplementedInst' in body_stripped and body_stripped.count('\n') <= 1
                        can_share = self._can_share_execute(inst.mnemonic)
                        if body_throws:
                            exec_impl = cgen.Line(
                                f'void {inst.fmt_name}::execute_impl'
                                f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                            )
                        elif can_share:
                            enc_key = enc.enc_name.lower().replace('enc_', '')
                            tmpl_name = f'{inst.mnemonic}_{enc_key}'
                            exec_impl = cgen.Line(
                                f'void {inst.fmt_name}::execute_impl'
                                f'(amdgpu::Wavefront &wf) {{\n'
                                f'{_dpp_preamble}'
                                f'  amdgpu::execute_{tmpl_name}(*this, wf);\n'
                                f'{_dpp_cleanup}'
                                f'{_sdwa_postamble}}}'
                            )
                            body_key = (inst.mnemonic, enc.enc_name)
                            self._shared_execute_bodies[body_key] = (
                                inst, sem, body, enc.enc_name,
                            )
                        else:
                            exec_impl = cgen.Line(
                                f'void {inst.fmt_name}::execute_impl'
                                f'(amdgpu::Wavefront &wf) {{\n'
                                f'{_dpp_preamble}'
                                f'{body}\n'
                                f'{_dpp_cleanup}'
                                f'{_sdwa_postamble}}}'
                            )
                    else:
                        exec_impl = cgen.Line(
                            f'void {inst.fmt_name}::execute_impl'
                            f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                        )

                    inst_classes.append(s)
                    class_func_impls.append(class_ctor_impl)
                    class_func_impls.append(exec_impl)

                # Build include lists for .cpp files
                cpp_includes = [
                    (
                        f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/'
                        f'{enc.fmt_enc_name.lower()}.h',
                        False,
                    ),
                    ('util/except.h', False),
                ]
                _MEM_ENC_NAMES = frozenset({
                    'ENC_SMEM', 'ENC_FLAT', 'ENC_MUBUF', 'ENC_MTBUF', 'ENC_DS',
                    # RDNA4 renamed/new memory encodings
                    'ENC_VFLAT', 'ENC_VGLOBAL', 'ENC_VSCRATCH',
                    'ENC_VDS', 'ENC_VBUFFER',
                })
                is_mem_enc = enc.enc_name.upper() in _MEM_ENC_NAMES
                if is_mem_enc:
                    cpp_includes.extend([
                        (f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/addr_calc.h', False),
                    ])
                    for cf_inc in self._cache_flags_includes():
                        cpp_includes.append((cf_inc, False))
                    cpp_includes.extend([
                        ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                        ('rocjitsu/vm/amdgpu/mem_state.h', False),
                        ('cstring', True),
                        ('memory', True),
                    ])
                has_mfma = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == 'mfma'
                    for i in all_insts
                )
                if has_mfma:
                    cpp_includes.append((
                        f'rocjitsu/isa/arch/amdgpu/'
                        f'{self.isa_spec.arch_name}/mfma_exec.h',
                        False,
                    ))
                _VOP_ENC_NAMES = frozenset({
                    'ENC_VOP1', 'ENC_VOP2', 'ENC_VOP3', 'ENC_VOP3P', 'ENC_VOPC',
                })
                if enc.enc_name.upper() in _VOP_ENC_NAMES:
                    cpp_includes.append((
                        'rocjitsu/isa/arch/amdgpu/shared/transcendental.h',
                        False,
                    ))
                if has_sem:
                    cpp_includes.extend([
                        ('rocjitsu/vm/amdgpu/wavefront.h', False),
                        ('util/data_types.h', False),
                        ('algorithm', True),
                        ('bit', True),
                        ('cmath', True),
                        ('limits', True),
                    ])
                # VOP1/VOP2 need DPP header for apply_dpp() in execute_impl.
                if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                    cpp_includes.append((
                        'rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h', False
                    ))
                has_saveexec = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == 'scalar_saveexec'
                    for i in all_insts
                )
                if has_saveexec:
                    cpp_includes.extend([
                        ('util/log.h', False),
                        ('format', True),
                    ])

                # Include the unified shared execute template header when
                # any instruction in this encoding delegates to a template.
                if self.shared_plan is not None:
                    has_shared = any(
                        self._can_share_execute(i.mnemonic)
                        for i in all_insts
                        if self.semantics and i.name in self.semantics.instructions
                    )
                    if has_shared:
                        cpp_includes.append((
                            'rocjitsu/isa/arch/amdgpu/shared/execute_shared.h',
                            False,
                        ))

                # Build per-ISA header includes.
                h_includes = [
                    (
                        f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/encodings.h',
                        False,
                    ),
                    (
                        f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/isa.h',
                        False,
                    ),
                    (
                        f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/operand.h',
                        False,
                    ),
                ]

                inst_def_file = CppFile(
                    f'{enc.fmt_enc_name.lower()}',
                    self.out_path,
                    True,
                    h_includes,
                    [],
                    inst_classes,
                    self.isa_spec.arch_name,
                    True,
                )
                inst_impl_file = CppFile(
                    f'{enc.fmt_enc_name.lower()}',
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
                            '  return Operand(32, OperandType::OPR_SIMM32, 0);\n'
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
                    class_func_impls.insert(0, smem_offset_helper)

                inst_def_file.gen_code()
                inst_impl_file.gen_code()

        # Generate the umbrella insts.h header that includes all per-
        # encoding instruction headers. Only primary (non-alt) encodings
        # that have instructions generate their own files; alt encoding
        # instructions are merged into their parent's file.
        import os

        arch = self.isa_spec.arch_name
        guard = f'ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_INSTS_H_'
        inc_base = f'rocjitsu/isa/arch/amdgpu/{arch}'
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
        insts_h_lines.append(f'\n#endif // {guard}\n')
        insts_h_path = os.path.join(self.out_path, arch, 'insts.h')
        with open(insts_h_path, 'w') as f:
            f.write(''.join(insts_h_lines))

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
        Phase F.5 will introduce shared encoding bases to eliminate this
        dependency.
        """
        import os

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
        import os
        import re as _re

        entries: list[tuple[str, str, str]] = []
        for (mnemonic, enc_name_key), (inst, sem, body, enc_name) in sorted(
            self._shared_execute_bodies.items()
        ):
            enc_key = enc_name.lower().replace('enc_', '')
            mnemonic = f'{mnemonic}_{enc_key}'
            prefixed_body = body
            for opnd in inst.operands:
                pattern = rf'(?<!\.)(?<!\w){_re.escape(opnd.name)}\.'
                prefixed_body = _re.sub(pattern, f'inst.{opnd.name}.', prefixed_body)
            prefixed_body = _re.sub(r'(?<!\.)(?<!\w)inst_\.', 'inst.inst_.', prefixed_body)
            prefixed_body = _re.sub(r'(?<!\.)(?<!\w)set_data\(', 'inst.set_data(', prefixed_body)
            prefixed_body = _re.sub(r'(?<!\.)(?<!\w)size_(?!\w)', 'inst.size()', prefixed_body)
            prefixed_body = _re.sub(r'(?<!\.)(?<!\w)mnemonic\(\)', 'inst.mnemonic()', prefixed_body)
            prefixed_body = _re.sub(r'(?<!\.)(?<!\w)simm32_(?!\w)', 'inst.simm32_', prefixed_body)
            prefixed_body = _re.sub(r'\s*\(void\)wf;\s*(?://[^\n]*)?\n?', '\n', prefixed_body)
            entries.append((mnemonic, prefixed_body, sem.semantic_class))

        shared_dir = os.path.join(self.out_path, 'shared')
        os.makedirs(shared_dir, exist_ok=True)

        guard = 'ROCJITSU_ISA_AMDGPU_SHARED_EXECUTE_SHARED_H_'
        lines = CppFile._prologue_comment().splitlines()
        lines += [
            f'#ifndef {guard}',
            f'#define {guard}',
            '',
            '#include "rocjitsu/vm/amdgpu/wavefront.h"',
            '#include "rocjitsu/vm/amdgpu/compute_unit.h"',
            '#include "rocjitsu/vm/amdgpu/mem_state.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"',
            '#include "util/data_types.h"',
            '#include "util/except.h"',
            '#include <algorithm>',
            '#include <bit>',
            '#include <cmath>',
            '#include <limits>',
            '',
            'namespace rocjitsu {',
            'namespace amdgpu {',
            '',
        ]

        for mnemonic, prefixed_body, sem_class in entries:
            lines.append('template <typename Inst>')
            lines.append(
                f'inline void execute_{mnemonic}('
                f'[[maybe_unused]] Inst &inst, [[maybe_unused]] Wavefront &wf) {{'
            )
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
        print(f'Generated shared/execute_shared.h with '
              f'{len(entries)} template functions', file=sys.stderr)

    def gen_operand_types(self) -> None:
        """Generate operand type and OpSel enums."""
        code_lines = []
        opnd_type_enum = 'enum class OperandType {'
        for opnd_type in self.isa_spec.operand_types:
            opnd_type_enum += opnd_type + ','
        opnd_type_enum += '};'
        code_lines.append(cgen.Line(opnd_type_enum))

        for opnd_sels in self.isa_spec.opnd_selectors:
            opnd_sel_name = ''.join(
                x.capitalize()
                for x in opnd_sels.operand_type.split('_')[1:]
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
        vgpr_types = [t for t in self.isa_spec.operand_types
                      if 'VGPR' in t or 'ACCVGPR' in t]
        if vgpr_types:
            fn = '[[nodiscard]] constexpr bool is_vgpr_operand_type(OperandType t) {'
            fn += ' switch (t) {'
            for t in vgpr_types:
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
            self.isa_spec.arch_name,
        )
        opnd_type_def_file.gen_code()

    def gen_operand(self) -> None:
        """Generate the ISA-specific Operand class with name resolution."""
        arch = self.isa_spec.arch_name

        switch_cases = []
        opnd_types_with_selectors = set()

        for opnd_sel in self.isa_spec.opnd_selectors:
            opnd_types_with_selectors.add(opnd_sel.operand_type)
            opsel_name = 'OpSel' + ''.join(
                x.capitalize()
                for x in opnd_sel.operand_type.split('_')[1:]
            )

            case_lines = []
            for pattern in opnd_sel.name_patterns:
                if pattern.kind == OperandNamePattern.REG_RANGE:
                    case_lines.append(
                        f'if (encoding_value_ >= {opsel_name}::{pattern.min_enum} && '
                        f'encoding_value_ <= {opsel_name}::{pattern.max_enum}) '
                        f'return reg_name("{pattern.prefix}", '
                        f'encoding_value_ - {opsel_name}::{pattern.min_enum}, size_bits_);'
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
                elif pattern.kind == OperandNamePattern.LITERAL:
                    case_lines.append(
                        f'if (encoding_value_ == {opsel_name}::{pattern.enum_name}) '
                        f'return "literal";'
                    )

            case_lines.append('break;')
            case_body = ' '.join(case_lines)
            switch_cases.append(
                f'case OperandType::{opnd_sel.operand_type}: '
                f'{{ {case_body} }}'
            )

        no_sel_types = [
            t
            for t in self.isa_spec.operand_types
            if t not in opnd_types_with_selectors
        ]
        for t in no_sel_types:
            if t == 'OPR_SIMM32':
                switch_cases.append(
                    f'case OperandType::{t}: '
                    f'return std::format("0x{{:x}}", encoding_value_);'
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
            else:
                switch_cases.append(
                    f'case OperandType::{t}: return std::to_string(encoding_value_);'
                )

        switch_body = '\n'.join(switch_cases)
        name_impl = (
            f'std::string Operand::name() const {{\n'
            f'switch (opr_type_) {{\n'
            f'{switch_body}\n'
            f'}}\n'
            f'return std::to_string(encoding_value_);\n'
            f'}}'
        )

        class_def = [
            cgen.Line(
                'class Operand : public IsaOperand<Isa> {\n'
                'public:\n'
                'Operand(int size_bits, OperandType opr_type, int encoding_value);\n'
                'std::string name() const override;\n'
                'uint32_t read_scalar(const amdgpu::Wavefront &wf) const override;\n'
                'uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
                'void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const override;\n'
                'void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const override;\n'
                'uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
                'void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const override;\n'
                'uint64_t read_scalar64(const amdgpu::Wavefront &wf) const override;\n'
                'void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const override;\n'
                '};'
            )
        ]

        class_impl = [
            cgen.Line(
                'Operand::Operand(int size_bits, OperandType opr_type, int encoding_value)\n'
                '    : IsaOperand<Isa>(size_bits, opr_type, encoding_value) {\n'
                '  is_vgpr_ = is_vgpr_operand_type(opr_type);\n'
                '}'
            ),
            cgen.Line(name_impl),
        ]

        reg_name_helper = cgen.Line(
            'namespace {\n'
            'std::string reg_name(const char *prefix, int reg_num, int size_bits) {\n'
            '  int count = size_bits / 32;\n'
            '  if (count <= 1)\n'
            '    return prefix + std::to_string(reg_num);\n'
            '  return std::string(prefix) + "[" + std::to_string(reg_num) + ":" +\n'
            '         std::to_string(reg_num + count - 1) + "]";\n'
            '}\n'
            '} // namespace'
        )
        class_impl.insert(0, reg_name_helper)

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
        _imm_parts = ['t == OperandType::OPR_SIMM16', 't == OperandType::OPR_SIMM32']
        for _opt in ('OPR_SIMM4', 'OPR_SIMM8'):
            if _opt in _opr:
                _imm_parts.append(f't == OperandType::{_opt}')
        for _opt in ('OPR_LABEL', 'OPR_WAITCNT'):
            if _opt in _opr:
                _imm_parts.append(f't == OperandType::{_opt}')
        _is_immediate_body = (
            'bool is_immediate_type(OperandType t) {\n'
            '  return ' + ' ||\n         '.join(_imm_parts) + ';\n'
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

        _read_lane_extra = ''
        _read_lane64_extra = ''
        if 'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST' in _opr:
            _read_lane_extra = (
                '  if (opr_type_ == OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST &&\n'
                '      ev >= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN &&\n'
                '      ev <= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MAX) {\n'
                f'    return wf.cu().read_vgpr(\n'
                f'        wf.vgpr_alloc().base + {_ACC_OFFSET} + static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN),\n'
                '        lane);\n'
                '  }\n'
            )
            _read_lane64_extra = (
                '  if (opr_type_ == OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST &&\n'
                '      ev >= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN &&\n'
                '      ev <= OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MAX) {\n'
                f'    uint32_t idx = wf.vgpr_alloc().base + {_ACC_OFFSET} +\n'
                '        static_cast<uint32_t>(ev - OpSelSrcVgprOrAccvgprOrConst::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN);\n'
                '    uint32_t lo = wf.cu().read_vgpr(idx, lane);\n'
                '    uint32_t hi = wf.cu().read_vgpr(idx + 1, lane);\n'
                '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
                '  }\n'
            )

        _read_lane_body = (
            'uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
            '  if (delegate()) return delegate()->read_lane(wf, lane);\n'
            '  int ev = encoding_value_;\n'
            '  if (is_vgpr_only_type(opr_type_))\n'
            '    return wf.cu().read_vgpr(wf.vgpr_alloc().base + vgpr_index(opr_type_, ev), lane);\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(ev);\n'
            '  if (ev >= 256 && ev <= 511)\n'
            '    return wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>(ev - 256), lane);\n'
            f'{_read_lane_extra}'
            '  return resolve_src_scalar(wf, ev);\n'
            '}'
        )

        _read_lane64_body = (
            'uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
            '  if (delegate()) return delegate()->read_lane64(wf, lane);\n'
            '  int ev = encoding_value_;\n'
            '  if (is_vgpr_only_type(opr_type_)) {\n'
            '    uint32_t idx = wf.vgpr_alloc().base + vgpr_index(opr_type_, ev);\n'
            '    uint32_t lo = wf.cu().read_vgpr(idx, lane);\n'
            '    uint32_t hi = wf.cu().read_vgpr(idx + 1, lane);\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            '  if (ev >= 256 && ev <= 511) {\n'
            '    uint32_t idx = wf.vgpr_alloc().base + static_cast<uint32_t>(ev - 256);\n'
            '    uint32_t lo = wf.cu().read_vgpr(idx, lane);\n'
            '    uint32_t hi = wf.cu().read_vgpr(idx + 1, lane);\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            f'{_read_lane64_extra}'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ev)));\n'
            '  return resolve_src_scalar64(wf, ev);\n'
            '}'
        )

        resolve_code = cgen.Line(
            'namespace {\n'
            '\n'
            'uint32_t resolve_src_scalar(const amdgpu::Wavefront &wf, int ev) {\n'
            '  if (ev <= 105)\n'
            '    return wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            '  if (ev == 106)\n'
            '    return static_cast<uint32_t>(wf.vcc());\n'
            '  if (ev == 107)\n'
            '    return static_cast<uint32_t>(wf.vcc() >> 32);\n'
            '  if (ev == 124)\n'
            '    return wf.m0();\n'
            '  if (ev == 126)\n'
            '    return static_cast<uint32_t>(wf.exec());\n'
            '  if (ev == 127)\n'
            '    return static_cast<uint32_t>(wf.exec() >> 32);\n'
            '  if (ev >= 128 && ev <= 192)\n'
            '    return static_cast<uint32_t>(ev - 128);\n'
            '  if (ev >= 193 && ev <= 208)\n'
            '    return static_cast<uint32_t>(static_cast<int32_t>(-(ev - 192)));\n'
            '  if (ev == 240)\n'
            '    return 0x3F000000u; // 0.5f\n'
            '  if (ev == 241)\n'
            '    return 0xBF000000u; // -0.5f\n'
            '  if (ev == 242)\n'
            '    return 0x3F800000u; // 1.0f\n'
            '  if (ev == 243)\n'
            '    return 0xBF800000u; // -1.0f\n'
            '  if (ev == 244)\n'
            '    return 0x40000000u; // 2.0f\n'
            '  if (ev == 245)\n'
            '    return 0xC0000000u; // -2.0f\n'
            '  if (ev == 246)\n'
            '    return 0x40800000u; // 4.0f\n'
            '  if (ev == 247)\n'
            '    return 0xC0800000u; // -4.0f\n'
            '  if (ev == 248)\n'
            '    return 0x3E22F983u; // 1/(2*pi)\n'
            '  if (ev == 249)\n'
            '    return 0u; // SRC_POPS_EXITING_WAVE_ID (not used in compute)\n'
            '  if (ev == 250)\n'
            '    return 0u; // NULL\n'
            '  if (ev == 251)\n'
            '    return wf.vcc() == 0 ? 1u : 0u; // VCCZ\n'
            '  if (ev == 252)\n'
            '    return wf.exec() == 0 ? 1u : 0u; // EXECZ\n'
            '  if (ev == 253)\n'
            '    return wf.read_scc() ? 1u : 0u; // SCC\n'
            '  throw std::logic_error("Unsupported encoding value for scalar read: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            'uint64_t resolve_src_scalar64(const amdgpu::Wavefront &wf, int ev) {\n'
            '  if (ev <= 105) {\n'
            '    uint32_t lo = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            '    uint32_t hi = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1));\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            '  if (ev == 106)\n'
            '    return wf.vcc();\n'
            '  if (ev == 126)\n'
            '    return wf.exec();\n'
            '  if (ev >= 128 && ev <= 192)\n'
            '    return static_cast<uint64_t>(ev - 128);\n'
            '  if (ev >= 193 && ev <= 208)\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(-(ev - 192))));\n'
            '  if (ev == 240)\n'
            '    return 0x3FE0000000000000ULL; // 0.5\n'
            '  if (ev == 241)\n'
            '    return 0xBFE0000000000000ULL; // -0.5\n'
            '  if (ev == 242)\n'
            '    return 0x3FF0000000000000ULL; // 1.0\n'
            '  if (ev == 243)\n'
            '    return 0xBFF0000000000000ULL; // -1.0\n'
            '  if (ev == 244)\n'
            '    return 0x4000000000000000ULL; // 2.0\n'
            '  if (ev == 245)\n'
            '    return 0xC000000000000000ULL; // -2.0\n'
            '  if (ev == 246)\n'
            '    return 0x4010000000000000ULL; // 4.0\n'
            '  if (ev == 247)\n'
            '    return 0xC010000000000000ULL; // -4.0\n'
            '  if (ev == 248)\n'
            '    return 0x3FC45F306DC9C883ULL; // 1/(2*pi)\n'
            '  throw std::logic_error("Unsupported encoding value for scalar64 read: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            'void resolve_dst_write(amdgpu::Wavefront &wf, int ev, uint32_t val) {\n'
            '  if (ev <= 105) {\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 106) {\n'
            '    wf.set_vcc((wf.vcc() & 0xFFFFFFFF00000000ULL) | val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 107) {\n'
            '    wf.set_vcc((wf.vcc() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 124) {\n'
            '    wf.set_m0(val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 126) {\n'
            '    wf.set_exec((wf.exec() & 0xFFFFFFFF00000000ULL) | val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 127) {\n'
            '    wf.set_exec((wf.exec() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("Unsupported encoding value for scalar write: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            'void resolve_dst_write64(amdgpu::Wavefront &wf, int ev, uint64_t val) {\n'
            '  if (ev <= 105) {\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), static_cast<uint32_t>(val));\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1), static_cast<uint32_t>(val >> 32));\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 106) {\n'
            '    wf.set_vcc(val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 126) {\n'
            '    wf.set_exec(val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("Unsupported encoding value for scalar64 write: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            + _is_vgpr_only_body + '\n\n'
            + _is_immediate_body + '\n\n'
            + _vgpr_index_body + '\n\n'
            +
            '\n'
            '} // namespace\n'
            '\n'
            'uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {\n'
            '  if (delegate()) return delegate()->read_scalar(wf);\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(encoding_value_);\n'
            '  return resolve_src_scalar(wf, encoding_value_);\n'
            '}\n'
            '\n'
            + _read_lane_body + '\n\n'
            'void Operand::write_scalar(amdgpu::Wavefront &wf, uint32_t val) const {\n'
            '  resolve_dst_write(wf, encoding_value_, val);\n'
            '}\n'
            '\n'
            'void Operand::write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const {\n'
            '  int ev = encoding_value_;\n'
            '  if (is_vgpr_only_type(opr_type_)) {\n'
            '    wf.cu().write_vgpr(wf.vgpr_alloc().base + vgpr_index(opr_type_, ev), lane, val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane called on non-VGPR operand type");\n'
            '}\n'
            '\n'
            + _read_lane64_body + '\n\n'
            'void Operand::write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const {\n'
            '  int ev = encoding_value_;\n'
            '  if (is_vgpr_only_type(opr_type_)) {\n'
            '    uint32_t idx = wf.vgpr_alloc().base + vgpr_index(opr_type_, ev);\n'
            '    wf.cu().write_vgpr(idx, lane, static_cast<uint32_t>(val));\n'
            '    wf.cu().write_vgpr(idx + 1, lane, static_cast<uint32_t>(val >> 32));\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane64 called on non-VGPR operand type");\n'
            '}\n'
            '\n'
            'uint64_t Operand::read_scalar64(const amdgpu::Wavefront &wf) const {\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(encoding_value_)));\n'
            '  return resolve_src_scalar64(wf, encoding_value_);\n'
            '}\n'
            '\n'
            'void Operand::write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const {\n'
            '  resolve_dst_write64(wf, encoding_value_, val);\n'
            '}'
        )
        class_impl.append(resolve_code)

        header_file = CppFile(
            'operand',
            self.out_path,
            True,
            [
                (f'rocjitsu/isa/arch/amdgpu/{arch}/isa.h', False),
                (f'rocjitsu/isa/arch/amdgpu/{arch}/operand_types.h', False),
                ('rocjitsu/isa/operand.h', False),
                ('string', True),
            ],
            [],
            class_def,
            arch,
        )
        source_file = CppFile(
            'operand',
            self.out_path,
            False,
            [
                (f'rocjitsu/isa/arch/amdgpu/{arch}/operand.h', False),
                ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                        ('rocjitsu/vm/amdgpu/wavefront.h', False),
                ('format', True),
                ('stdexcept', True),
                ('string', True),
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
            cgen.Line('public:'),
            cgen.FunctionDeclaration(
                cgen.Value('static std::unique_ptr<Instruction>', 'decode'),
                [cgen.Value('const MachineInst *', 'opcode')],
            ),
            cgen.Line('private:'),
            cgen.Statement(
                'using DecodeFunc = std::unique_ptr<Instruction>(*)(const MachineInst *)'
            ),
            cgen.FunctionDeclaration(
                cgen.Value(
                    'static std::unique_ptr<Instruction>', 'decodeInvalid'
                ),
                [cgen.Value('const MachineInst *', 'opcode')],
            ),
        ]
        decode_table_funcs = [
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value(
                        'std::unique_ptr<Instruction>', 'Decoder::decode'
                    ),
                    [cgen.Value('const MachineInst *', 'opcode')],
                ),
                cgen.Block(
                    [
                        cgen.Statement(
                            'Sop1MachineInst op = std::bit_cast<decltype(op)>(*opcode)'
                        ),
                        cgen.Statement(
                            'return primary_decode_table[op.encoding](opcode)'
                        ),
                    ]
                ),
            ),
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value(
                        'std::unique_ptr<Instruction>',
                        'Decoder::decodeInvalid',
                    ),
                    [cgen.Value('const MachineInst *', 'opcode')],
                ),
                cgen.Block(
                    [
                        cgen.Statement(
                            'throw util::InvalidInst(std::format("{:X}", *opcode))'
                        ),
                        cgen.Statement('return nullptr'),
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
        for dte in self.isa_spec.primary_decode_table:
            if dte is not None:
                decode_table_entries.append(
                    f'&Decoder::{dte.decode_func},'
                )
                if dte.decode_func not in decode_funcs_found:
                    decode_funcs_found.add(dte.decode_func)
                    func_decl = cgen.FunctionDeclaration(
                        cgen.Value(
                            'std::unique_ptr<Instruction>',
                            f'Decoder::{dte.decode_func}',
                        ),
                        [cgen.Value('const MachineInst *', 'opcode')],
                    )
                    sub_decode_func_decls = []
                    if dte.is_primary:
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
                                            f'return {dte.sub_decode_table}[op.op](opcode)'
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
                                f'const std::array<Decoder::DecodeFunc, {len(dte.sub_decode_funcs)}> Decoder::{dte.sub_decode_table} = {{'
                            )
                        )
                        sub_decode_table_entry_str = []
                        for fn in dte.sub_decode_funcs:
                            if fn != 'decodeInvalid':
                                sub_decode_func_decls.append(
                                    cgen.FunctionDeclaration(
                                        cgen.Value(
                                            'static std::unique_ptr<Instruction>',
                                            fn,
                                        ),
                                        [
                                            cgen.Value(
                                                'const MachineInst *',
                                                'opcode',
                                            )
                                        ],
                                    )
                                )
                                decode_table_funcs.append(
                                    cgen.FunctionBody(
                                        cgen.FunctionDeclaration(
                                            cgen.Value(
                                                'std::unique_ptr<Instruction>',
                                                f'Decoder::{fn}',
                                            ),
                                            [
                                                cgen.Value(
                                                    'const MachineInst *',
                                                    'opcode',
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
                            sub_decode_table_entry_str.append(
                                f'&Decoder::{fn},'
                            )
                        sub_decode_table_entries.append(
                            cgen.Line(
                                ''.join(sub_decode_table_entry_str)
                            )
                        )
                        sub_decode_table_entries.append(cgen.Line('};'))
                    class_members.append(
                        cgen.FunctionDeclaration(
                            cgen.Value(
                                'static std::unique_ptr<Instruction>',
                                f'{dte.decode_func}',
                            ),
                            [cgen.Value('const MachineInst *', 'opcode')],
                        )
                    )
                    class_members.extend(sub_decode_func_decls)
            else:
                decode_table_entries.append('&Decoder::decodeInvalid,')
        decode_table_entries = ''.join(decode_table_entries)
        class_members.extend(decode_tables)
        class_def.append(cgen.Struct('Decoder', class_members))
        class_impl.extend(decode_table_funcs)
        class_impl.append(
            cgen.Line(
                f'const std::array<Decoder::DecodeFunc, {pow(2, self.isa_spec.profile.max_enc_bits)}> Decoder::primary_decode_table = {{'
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
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/machine_insts.h',
                    False,
                ),
                ('array', True),
                ('memory', True),
            ],
            ['Instruction'],
            class_def,
            self.isa_spec.arch_name,
            True,
        )
        class_impl_file = CppFile(
            'decoder',
            self.out_path,
            False,
            [
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/decoder.h',
                    False,
                ),
                ('util/except.h', False),
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/insts.h',
                    False,
                ),
                ('bit', True),
                ('format', True),
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
            op_field = next(
                (f for f in enc.ucode_fields if f.name == 'op'), None
            )
            enc_field = next(
                (f for f in enc.ucode_fields if f.name == 'encoding'), None
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

        import os
        out_path = os.path.join(
            self.out_path, self.isa_spec.arch_name, 'test_encodings.h'
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
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/decoder.h',
                    False,
                ),
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/operand_types.h',
                    False,
                ),
            ],
            [],
            isa_struct,
            self.isa_spec.arch_name,
        )
        isa_struct_file.gen_code()
