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

from datetime import datetime
from collections.abc import Sequence
from typing import TextIO

from amdisa.gpuisa import InstEncoding, Instruction, IsaSpec, OperandNamePattern
from amdisa.semantics import InstructionSemantics, SemanticsSpec


class CppFile:
    """A generated C++ file (.h or .cpp).

    Defines a common structure: copyright, includes, namespace, forward
    declarations, and source code objects.

    Attributes:
        _prologue_comment: Returns boilerplate comment for the top of the file.
        name: File name without extension.
        out_path: Output directory for generated files.
        is_header: True if this is a header file.
        includes: List of (include_name, is_system_header) pairs.
        fwd_decls: Forward declarations within the namespace.
        src_code: Source code objects (strings or cgen objects).
        arch_name: ISA architecture name.
        replace_structs: If True, replace 'struct' with 'class' in output.
        file_name: File name with extension.
    """

    @staticmethod
    def _prologue_comment() -> str:
        year = datetime.now().strftime('%Y')
        return (
            f'// Copyright (c) 2025-{year} Advanced Micro Devices, Inc.'
            f'\n// SPDX-License-Identifier: MIT\n\n'
            f'// This file was automatically generated. Do not modify.\n\n'
        )

    def __init__(
        self,
        name: str,
        out_path: str,
        is_header: bool,
        includes: list[tuple[str, bool]],
        fwd_decls: list[str],
        src_code: Sequence[object],
        arch_name: str,
        replace_structs: bool = False,
    ) -> None:
        self.name = name
        self.out_path = out_path
        self.is_header = is_header
        self.includes = includes
        self.fwd_decls = fwd_decls
        self.src_code = src_code
        self.arch_name = arch_name
        self.replace_structs = replace_structs
        self.file_name = self.name

        if self.is_header:
            self.file_name += '.h'
            include_guard_components = [
                s.upper() for s in self.name.split('_')
            ]
            self.include_guard = (
                f'ROCJITSU_ISA_ARCH_AMDGPU_{self.arch_name.upper()}_'
                + '_'.join(include_guard_components)
                + '_H_'
            )
        else:
            self.file_name += '.cpp'

    def gen_prologue(self, f: TextIO) -> None:
        """Write copyright, include guard, includes, and namespace opener."""
        f.write(CppFile._prologue_comment())
        if self.is_header:
            f.write(f'#ifndef {self.include_guard}\n')
            f.write(f'#define {self.include_guard}\n\n')
        f.writelines(
            [f'{cgen.Include(x[0], x[1])}\n' for x in self.includes]
        )
        f.write('\n')
        f.write('namespace rocjitsu {\n')
        f.writelines(
            [
                f'\n{cgen.Statement("class " + x)}' + '\n\n'
                for x in self.fwd_decls
            ]
        )
        f.write(f'namespace {self.arch_name} {{\n')
        f.write('\n')

    def gen_header(self, f: TextIO) -> None:
        """Write header body and closing guards."""
        if self.replace_structs:
            f.writelines(
                [
                    re.sub(r'^struct\s', 'class ', f'{e}\n\n')
                    for e in self.src_code
                ]
            )
        else:
            f.writelines([f'{e}\n\n' for e in self.src_code])
        f.write(
            f'}} // namespace {self.arch_name}\n}}'
            f' // namespace rocjitsu\n'
        )
        f.write(f'\n#endif // {self.include_guard}\n')

    def gen_cpp(self, f: TextIO) -> None:
        """Write source file body and closing namespaces."""
        f.writelines([f'{e}\n\n' for e in self.src_code])
        f.write(
            f'}} // namespace {self.arch_name}'
            f'\n}} // namespace rocjitsu\n'
        )

    def gen_code(self) -> None:
        """Generate the file (header or source)."""
        with open(
            f'{self.out_path}/{self.arch_name}/{self.file_name}', 'w'
        ) as f:
            self.gen_prologue(f)
            if self.is_header:
                self.gen_header(f)
            else:
                self.gen_cpp(f)


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
    """

    def __init__(
        self,
        isa_spec: IsaSpec,
        out_path: str,
        semantics: SemanticsSpec | None = None,
    ) -> None:
        self.isa_spec = isa_spec
        self.out_path = out_path
        self.semantics = semantics

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

    def gen_machine_inst_encodings(self) -> None:
        """Generate machine instruction encoding structs as bitfields."""
        enc_structs = [cgen.Statement('using MachineInst = uint32_t')]
        for inst_enc in self.isa_spec.inst_encodings:
            s = cgen.Struct(
                f'{inst_enc.fmt_enc_name}MachineInst',
                [
                    cgen.Value('uint32_t', f'{x.name} : {x.bit_cnt}')
                    for x in inst_enc.ucode_fields
                ],
            )
            enc_structs.append(s)
        cpp_file = CppFile(
            'machine_insts',
            self.out_path,
            True,
            [('cstdint', True)],
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
            if inst_enc.is_alt and (
                not inst_enc.has_unique_ops
                or inst_enc.is_implied_literal_enc
            ):
                continue
            class_members = []
            public_members = [
                cgen.Line('public:'),
                cgen.FunctionDeclaration(
                    cgen.Value('', f'{inst_enc.fmt_enc_name}'),
                    [
                        cgen.Value('const std::string', '&mnemonic'),
                        cgen.Value(
                            f'const {inst_enc.fmt_enc_name}MachineInst',
                            '*inst',
                        ),
                    ],
                ),
            ]
            private_members = [cgen.Line('private:')]

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
            elif rule.suffix:
                mnemonic_expr = f'mnemonic + "{rule.suffix}"'
            else:
                mnemonic_expr = 'mnemonic'

            modifier_lines = ''
            for mod in profile.encoding_modifiers(inst_enc.enc_name):
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

            size_line = ' size_ = sizeof(OpEncoding);'
            if size_condition is not None:
                size_line += (
                    f' if ({size_condition})'
                    f' size_ += sizeof(MachineInst);'
                )
            class_ctor_impl = (
                f'{inst_enc.fmt_enc_name}::{inst_enc.fmt_enc_name}'
                f'(const std::string &mnemonic, const {inst_enc.fmt_enc_name}MachineInst *inst) '
                f': IsaInstruction<Isa>({mnemonic_expr}), inst_(*inst) '
                f'{{{size_line}{modifier_lines}}}'
            )
            class_func_impls.append(cgen.Line(class_ctor_impl))
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
                    private_members.append(func_decl)
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
                private_members.append(func_decl)
                class_func_impls.append(func_body)

            class_members.extend(public_members)
            class_members.append(
                cgen.Statement(
                    f'using OpEncoding = {inst_enc.fmt_enc_name}MachineInst'
                )
            )
            # inst_ is protected so derived instruction classes can
            # access encoding fields (e.g. VOP3 neg/abs/clamp/omod).
            protected_members = [cgen.Line('protected:')]
            protected_members.append(
                cgen.Statement(
                    '[[maybe_unused]] const OpEncoding inst_'
                )
            )
            class_members.extend(protected_members)
            class_members.extend(private_members)
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
                ('string', True),
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
                'std::string flat_mnemonic(const std::string &mnemonic, int seg) {\n'
                '  // seg: 0=FLAT, 1=SCRATCH, 2=GLOBAL\n'
                '  if (seg == 1) {\n'
                '    if (mnemonic.substr(0, 5) == "flat_")\n'
                '      return "scratch_" + mnemonic.substr(5);\n'
                '  } else if (seg == 2) {\n'
                '    if (mnemonic.substr(0, 5) == "flat_")\n'
                '      return "global_" + mnemonic.substr(5);\n'
                '  }\n'
                '  return mnemonic;\n'
                '}\n'
                '} // namespace'
            )
            class_func_impls.insert(0, flat_mnemonic_helper)

        class_impl_file = CppFile(
            'encodings',
            self.out_path,
            False,
            [
                (
                    f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/encodings.h',
                    False,
                ),
                ('string', True),
            ],
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

        if cls == 'nop':
            return '  (void)wf;'

        if cls == 'endpgm':
            L.append('  wf.halt();')
            return '\n'.join(L)

        if cls == 'waitcnt':
            lgkm_mask = self.isa_spec.profile.waitcnt_lgkmcnt_mask
            L.append(f'  uint16_t imm = static_cast<uint16_t>({src_ops[0]}.encoding_value_);')
            L.append('  uint8_t vm = (imm & 0xF) | ((imm >> 10) & 0x30);')
            L.append('  uint8_t exp = (imm >> 4) & 0x7;')
            L.append(f'  uint8_t lgkm = (imm >> 8) & {lgkm_mask};')
            L.append('  wf.set_wait_target(vm, lgkm, exp);')
            L.append('  if (!wf.wait_satisfied())')
            L.append('    wf.set_state(amdgpu::WfState::WAITCNT);')
            return '\n'.join(L)

        if cls == 'wait_counter':
            # RDNA4 split-wait instructions: the immediate operand is
            # the counter threshold directly (no bit-packing).
            L.append(f'  uint16_t cnt = static_cast<uint16_t>({src_ops[0]}.encoding_value_);')
            L.append(f'  wf.set_wait_counter("{op}", cnt);')
            L.append('  if (!wf.wait_satisfied())')
            L.append('    wf.set_state(amdgpu::WfState::WAITCNT);')
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
            if op == 'andn1':
                L.append('  wf.set_exec(src & ~wf.exec());')
            elif op == 'andn2':
                L.append('  wf.set_exec(wf.exec() & ~src);')
            else:
                L.append(f'  wf.set_exec(src); // TODO: {op}')
            return '\n'.join(L)

        if cls == 'scalar_getpc':
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.pc);')
            return '\n'.join(L)

        if cls == 'scalar_setpc':
            L.append(f'  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;')
            return '\n'.join(L)

        if cls == 'scalar_swappc':
            L.append('  uint64_t old_pc = wf.pc;')
            L.append(f'  wf.pc = {src_ops[0]}.read_scalar64(wf) - size_;')
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, old_pc);')
            return '\n'.join(L)

        if cls == 'scalar_call':
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.pc);')
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
            return self._gen_vector_cmp(src_ops, op, dtype, is_vop3)

        if cls == 'vector_cmpx':
            return self._gen_vector_cmpx(src_ops, op, dtype, is_vop3)

        if cls == 'vector_cndmask':
            L.append('  uint64_t exec = wf.exec();')
            L.append('  uint64_t vcc = wf.vcc();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if is_vop3:
                L.append(f'    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));')
                L.append(f'    float s1 = std::bit_cast<float>({src_ops[1]}.read_lane(wf, lane));')
                L.extend(self._vop3_src_mod('s0', 0))
                L.extend(self._vop3_src_mod('s1', 1))
                L.append('    float val = (vcc & (1ULL << lane)) ? s1 : s0;')
                L.extend(self._vop3_dst_mod('val'))
                L.append(f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(val));')
            else:
                L.append(f'    uint32_t val = (vcc & (1ULL << lane))')
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
            return self._gen_vector_cmp_class(src_ops, dtype, False, is_vop3)

        if cls == 'vector_cmpx_class':
            return self._gen_vector_cmp_class(src_ops, dtype, True, is_vop3)

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
                    'vector_cvt_pkrtz_f16_f32', 'vector_cvt_pk'):
            return self._gen_vector_cvt_pk(dst_ops, src_ops, cls, op)

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
            return self._gen_buffer_load(dst_ops, src_ops, sem, cls)

        if cls in ('buffer_store', 'tbuffer_store'):
            return self._gen_buffer_store(dst_ops, src_ops, sem, cls)

        if cls == 'ds_read':
            return self._gen_ds_read(dst_ops, src_ops, sem)

        if cls == 'ds_write':
            return self._gen_ds_write(dst_ops, src_ops, sem)

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
            return self._gen_ds_atomic(dst_ops, src_ops, sem)

        return f'  (void)wf;\n  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: {cls}'

    def _gen_vector_cmp_class(self, src: list[str], dtype: str | None, is_cmpx: bool, is_vop3: bool = False) -> str:
        """Generate V_CMP_CLASS / V_CMPX_CLASS body."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
        if is_cmpx:
            L.append('  uint64_t result = 0;')
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
            L.append('  wf.set_exec(result);')
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
            L.append('    else if (c == 0.0) {')
            L.append('      result = std::copysign(std::numeric_limits<double>::infinity(),')
            L.append('                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));')
            L.append('    }')
            L.append('    else if (std::isinf(b)) result = std::copysign(0.0, b);')
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
            L.append('    else if (c == 0.0f) {')
            L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
            L.append('                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));')
            L.append('    }')
            L.append('    else if (std::isinf(b)) result = std::copysign(0.0f, b);')
            L.append('    else result = p;')
            if is_vop3:
                L.extend(self._vop3_dst_mod('result'))
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        L.append('  }')
        return '\n'.join(L)

    def _gen_vector_div_scale(self, dst: list[str], src: list[str], dtype: str | None, is_vop3: bool = False) -> str:
        """Generate V_DIV_SCALE body."""
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
            L.append('    double result = s0;')
            L.append('    bool needs_scale = false;')
            L.append('    if (!std::isnan(s1) && !std::isnan(s2) &&')
            L.append('        !std::isinf(s1) && !std::isinf(s2) &&')
            L.append('        s1 != 0.0 && s2 != 0.0) {')
            L.append('      int exp1, exp2;')
            L.append('      std::frexp(s1, &exp1);')
            L.append('      std::frexp(s2, &exp2);')
            L.append('      needs_scale = std::abs(exp1 - exp2) > 768;')
            L.append('      if (needs_scale) result = std::ldexp(s0, exp2 > exp1 ? 1024 : -1024);')
            L.append('    }')
            L.append('    if (needs_scale) vcc |= (1ULL << lane);')
            L.append('    else vcc &= ~(1ULL << lane);')
            L.append(f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));')
        else:
            L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
            L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
            L.append(f'    float s2 = std::bit_cast<float>({src[2]}.read_lane(wf, lane));')
            if is_vop3:
                L.extend(self._vop3_src_mod('s0', 0))
                L.extend(self._vop3_src_mod('s1', 1))
                L.extend(self._vop3_src_mod('s2', 2))
            L.append('    float result = s0;')
            L.append('    bool needs_scale = false;')
            L.append('    if (!std::isnan(s1) && !std::isnan(s2) &&')
            L.append('        !std::isinf(s1) && !std::isinf(s2) &&')
            L.append('        s1 != 0.0f && s2 != 0.0f) {')
            L.append('      int exp1, exp2;')
            L.append('      std::frexp(s1, &exp1);')
            L.append('      std::frexp(s2, &exp2);')
            L.append('      needs_scale = std::abs(exp1 - exp2) > 100;')
            L.append('      if (needs_scale) result = std::ldexp(s0, exp2 > exp1 ? 128 : -128);')
            L.append('    }')
            L.append('    if (needs_scale) vcc |= (1ULL << lane);')
            L.append('    else vcc &= ~(1ULL << lane);')
            L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
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
            L.append('      result = std::ldexp(result, 1024);')
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
            L.append('      result = std::ldexp(result, 128);')
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
        else:  # dot2c - nop for now (needs F16 or I16)
            L.append(f'    (void)a; (void)b; // dot2c needs F16/I16 support')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(acc));')
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
        is_64 = dtype in ('b64', 'i64')

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
        if dtype in ('i32',) and op in ('add', 'sub'):
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
        """Generate saveexec body."""
        L = []
        L.append('  uint64_t old_exec = wf.exec();')
        L.append(f'  {dst[0]}.write_scalar64(wf, old_exec);')
        L.append(f'  uint64_t src = {src[0]}.read_scalar64(wf);')
        saveexec_map = {
            'and': 'old_exec & src',
            'or': 'old_exec | src',
            'xor': 'old_exec ^ src',
            'nand': '~(old_exec & src)',
            'nor': '~(old_exec | src)',
            'xnor': '~(old_exec ^ src)',
            'andn1': 'src & ~old_exec',
            'andn2': 'old_exec & ~src',
            'orn1': 'src | ~old_exec',
            'orn2': 'old_exec | ~src',
        }
        expr = saveexec_map.get(op, f'old_exec /* TODO: {op} */')
        L.append(f'  uint64_t result = {expr};')
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
                'rcp': '1.0 / s',
                'sqrt': 'std::sqrt(s)',
                'rsq': '1.0 / std::sqrt(s)',
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
                'rcp': '1.0f / s',
                'rcp_iflag': '1.0f / s',
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
                'fmac': f'sv0 * sv1 + std::bit_cast<double>({d}.read_lane64(wf, lane))',
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
                'fmac': f'sv0 * sv1 + std::bit_cast<float>({d}.read_lane(wf, lane))',
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
                'fmac': f'sv0 * sv1 + util::f16_to_f32(static_cast<uint16_t>({d}.read_lane(wf, lane)))',
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
                    'min3': 'std::min(std::min(a, b), c)',
                    'max3': 'std::max(std::max(a, b), c)',
                    'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
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

    def _gen_vector_cmp(self, src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector compare body (sets VCC per lane)."""
        L = []
        L.append('  uint64_t exec = wf.exec();')
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
        L.append('  wf.set_vcc(vcc);')
        return '\n'.join(L)

    def _gen_vector_cmpx(self, src: list[str], op: str | None, dtype: str | None, is_vop3: bool = False) -> str:
        """Generate vector compare-and-write-EXEC body.

        On CDNA (GFX9), V_CMPX writes both EXEC and VCC. On RDNA,
        V_CMPX writes only EXEC.
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
            L.append('  wf.set_vcc(result);')
        L.append('  wf.set_exec(result);')
        return '\n'.join(L)

    def _gen_vector_add_co(self, dst: list[str], src: list[str], op: str | None, dtype: str | None) -> str:
        """Generate vector add/sub with carry out to VCC."""
        L = []
        d = dst[0]
        s0, s1 = src[0], src[1]

        L.append('  uint64_t exec = wf.exec();')
        L.append('  uint64_t vcc = wf.vcc();')
        if op in ('addc', 'subbc', 'subbrevco'):
            L.append('  uint64_t old_vcc = vcc;')
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
        L.append('    bool sel0_lo = (inst_.op_sel >> 0) & 1;')
        L.append('    bool sel1_lo = (inst_.op_sel >> 1) & 1;')
        L.append('    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;')
        L.append('    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;')

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
        L.append('    bool sel0_lo = (inst_.op_sel >> 0) & 1;')
        L.append('    bool sel1_lo = (inst_.op_sel >> 1) & 1;')
        L.append('    bool sel2_lo = (inst_.op_sel >> 2) & 1;')
        L.append('    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;')
        L.append('    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;')
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
        """
        d, s0, s1 = dst[0], src[0], src[1]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint64_t raw0 = {s0}.read_lane64(wf, lane);')
        L.append(f'    uint64_t raw1 = {s1}.read_lane64(wf, lane);')
        L.append('    float a_lo = std::bit_cast<float>(static_cast<uint32_t>(raw0));')
        L.append('    float a_hi = std::bit_cast<float>(static_cast<uint32_t>(raw0 >> 32));')
        L.append('    float b_lo = std::bit_cast<float>(static_cast<uint32_t>(raw1));')
        L.append('    float b_hi = std::bit_cast<float>(static_cast<uint32_t>(raw1 >> 32));')
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; a_hi = -a_hi; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; b_hi = -b_hi; }')
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
        """Generate packed F32 ternary op (V_PK_FMA_F32)."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint64_t raw0 = {s0}.read_lane64(wf, lane);')
        L.append(f'    uint64_t raw1 = {s1}.read_lane64(wf, lane);')
        L.append(f'    uint64_t raw2 = {s2}.read_lane64(wf, lane);')
        L.append('    float a_lo = std::bit_cast<float>(static_cast<uint32_t>(raw0));')
        L.append('    float a_hi = std::bit_cast<float>(static_cast<uint32_t>(raw0 >> 32));')
        L.append('    float b_lo = std::bit_cast<float>(static_cast<uint32_t>(raw1));')
        L.append('    float b_hi = std::bit_cast<float>(static_cast<uint32_t>(raw1 >> 32));')
        L.append('    float c_lo = std::bit_cast<float>(static_cast<uint32_t>(raw2));')
        L.append('    float c_hi = std::bit_cast<float>(static_cast<uint32_t>(raw2 >> 32));')
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; a_hi = -a_hi; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; b_hi = -b_hi; }')
        L.append('    if (inst_.neg & 4) { c_lo = -c_lo; c_hi = -c_hi; }')
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
        L.append(f'    uint64_t raw0 = {s0}.read_lane64(wf, lane);')
        L.append(f'    uint64_t raw1 = {s1}.read_lane64(wf, lane);')
        L.append('    uint32_t lo = (inst_.op_sel & 1) ? static_cast<uint32_t>(raw0 >> 32) : static_cast<uint32_t>(raw0);')
        L.append('    uint32_t hi = (inst_.op_sel & 2) ? static_cast<uint32_t>(raw1 >> 32) : static_cast<uint32_t>(raw1);')
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
        L.append('    float a, b, c;')
        L.append('    if (inst_.op_sel_hi & 1) a = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 1) ? (raw0 >> 16) : raw0));')
        L.append('    else a = std::bit_cast<float>(raw0);')
        L.append('    if (inst_.op_sel_hi & 2) b = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 2) ? (raw1 >> 16) : raw1));')
        L.append('    else b = std::bit_cast<float>(raw1);')
        L.append(f'    if ({self._op_sel_hi_2_expr(self._enc_name)}) c = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 4) ? (raw2 >> 16) : raw2));')
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
        L.append('    float a, b, c;')
        L.append('    if (inst_.op_sel_hi & 1) a = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 1) ? (raw0 >> 16) : raw0));')
        L.append('    else a = std::bit_cast<float>(raw0);')
        L.append('    if (inst_.op_sel_hi & 2) b = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 2) ? (raw1 >> 16) : raw1));')
        L.append('    else b = std::bit_cast<float>(raw1);')
        L.append(f'    if ({self._op_sel_hi_2_expr(self._enc_name)}) c = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 4) ? (raw2 >> 16) : raw2));')
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
        """Generate V_DOT2_F32_F16, V_DOT2_I32_I16, V_DOT2_U32_U16."""
        d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
        L = []
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

        if cls == 'dot2_f32_f16':
            L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(raw0));')
            L.append('    float a1 = util::f16_to_f32(static_cast<uint16_t>(raw0 >> 16));')
            L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(raw1));')
            L.append('    float b1 = util::f16_to_f32(static_cast<uint16_t>(raw1 >> 16));')
            L.append('    if (inst_.neg & 1) { a0 = -a0; a1 = -a1; }')
            L.append('    if (inst_.neg & 2) { b0 = -b0; b1 = -b1; }')
            L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            L.append('    if (inst_.neg & 4) acc = -acc;')
            L.append('    float result = a0 * b0 + a1 * b1 + acc;')
            L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        elif cls == 'dot2_i32_i16':
            L.append('    int16_t a0 = static_cast<int16_t>(raw0);')
            L.append('    int16_t a1 = static_cast<int16_t>(raw0 >> 16);')
            L.append('    int16_t b0 = static_cast<int16_t>(raw1);')
            L.append('    int16_t b1 = static_cast<int16_t>(raw1 >> 16);')
            L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append('    int32_t result = static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1 + acc;')
            L.append('    if (inst_.clamp) result = std::clamp(result, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(result));')
        else:  # dot2_u32_u16
            L.append('    uint16_t a0 = static_cast<uint16_t>(raw0);')
            L.append('    uint16_t a1 = static_cast<uint16_t>(raw0 >> 16);')
            L.append('    uint16_t b0 = static_cast<uint16_t>(raw1);')
            L.append('    uint16_t b1 = static_cast<uint16_t>(raw1 >> 16);')
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
            r'V_(?:S?MFMA[C]?)_(F32|I32|F64)_(\d+)X(\d+)X(\d+)'
            r'(?:_\d+B)?_(F32|XF32|F16|BF16|I8|F64|BF8_BF8|BF8_FP8|FP8_BF8|FP8_FP8)$',
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
            'I8': 8, 'F64': 64,
            'FP8_FP8': 8, 'FP8_BF8': 8, 'BF8_FP8': 8, 'BF8_BF8': 8,
        }
        in_bits = _INPUT_BITS.get(input_type, 32)

        # Map input types to mfma::extract_* function names.
        _EXTRACT_A = {
            'F32': 'mfma::extract_f32', 'XF32': 'mfma::extract_f32',
            'F16': 'mfma::extract_f16', 'BF16': 'mfma::extract_bf16',
            'FP8_FP8': 'mfma::extract_fp8', 'FP8_BF8': 'mfma::extract_fp8',
            'BF8_FP8': 'mfma::extract_bf8', 'BF8_BF8': 'mfma::extract_bf8',
        }
        _EXTRACT_B = {
            'F32': 'mfma::extract_f32', 'XF32': 'mfma::extract_f32',
            'F16': 'mfma::extract_f16', 'BF16': 'mfma::extract_bf16',
            'FP8_FP8': 'mfma::extract_fp8', 'FP8_BF8': 'mfma::extract_bf8',
            'BF8_FP8': 'mfma::extract_fp8', 'BF8_BF8': 'mfma::extract_bf8',
        }

        L = []
        L.append(f'  auto &cu = wf.cu();')
        L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
        L.append(f'  uint32_t dst = mfma::dst_base(vb, {d}.encoding_value_);')
        L.append(f'  uint32_t const_acc;')
        L.append(f'  uint32_t s2 = mfma::resolve_acc(vb, dst,')
        L.append(f'      {s2}.encoding_value_, const_acc,'
                 f' [&] {{ return {s2}.read_scalar(wf); }});')

        if result_type == 'F64':
            L.append(f'  mfma::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
            L.append(f'                 mfma::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 mfma::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                 s2, const_acc);')
        elif result_type == 'I32':
            L.append(f'  mfma::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
            L.append(f'                     mfma::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                     mfma::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                     s2, const_acc);')
        else:
            ea = _EXTRACT_A.get(input_type, 'mfma::extract_f32')
            eb = _EXTRACT_B.get(input_type, 'mfma::extract_f32')
            L.append(f'  mfma::exec_f32(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
            L.append(f'                 mfma::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 mfma::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                 s2, {ea}, {eb}, const_acc);')

        return '\n'.join(L)

    def _gen_smem_load(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        nd = sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;')
        L.append(f'  d->num_dwords = {nd};')
        L.append('  d->is_load = true;')
        L.append('  d->addr = smem_calculate_address(inst_, wf);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_smem_store(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        nd = sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->num_dwords = {nd};')
        L.append('  d->is_load = false;')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;')
        L.append(f'  for (uint32_t i = 0; i < {nd}; ++i)')
        L.append('    d->store_data[i] = cu.read_sgpr(sdata_base + i);')
        L.append('  d->addr = smem_calculate_address(inst_, wf);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_flat_load(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append('  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_flat_store(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append('  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz == 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 4);')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);')
                L.append(f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    _ATOMIC_OP_ENUM: dict[str, str] = {
        'swap': 'amdgpu::AtomicOp::SWAP',
        'cmpswap': 'amdgpu::AtomicOp::CMPSWAP',
        'add': 'amdgpu::AtomicOp::ADD',
        'sub': 'amdgpu::AtomicOp::SUB',
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append('  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = (inst_.sc0 != 0);')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append('  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append('  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        L.append('  d->is_load = (inst_.sc0 != 0);')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append('  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
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
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append('  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        # DS atomics always return the old value (like GLC=1).
        L.append('  d->is_load = true;')
        L.append(f'  d->atomic_op = {op_enum};')
        L.append('  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + {i}, lane);')
            L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_load(self, dst: list[str], src: list[str], sem: InstructionSemantics, cls: str = 'buffer_load') -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        addr_fn = 'mtbuf_calculate_addresses' if cls == 'tbuffer_load' else 'mubuf_calculate_addresses'
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append(f'  {addr_fn}(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_buffer_store(self, dst: list[str], src: list[str], sem: InstructionSemantics, cls: str = 'buffer_store') -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        addr_fn = 'mtbuf_calculate_addresses' if cls == 'tbuffer_store' else 'mubuf_calculate_addresses'
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append('  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);')
        L.append('  d->non_temporal = inst_.nt;')
        L.append(f'  {addr_fn}(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz >= 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, {esz});')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);')
                L.append(f'    d->store_data[lane * {stride} + {i}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_read(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        if sem.sign_extend:
            L.append('  d->sign_extend = true;')
        L.append('  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write(self, dst: list[str], src: list[str], sem: InstructionSemantics) -> str:
        L = []
        esz, ne = sem.elem_size, sem.num_elems
        L.append('  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = false;')
        L.append('  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);')
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
                L.append(f'    uint32_t lo{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + {vgpr_base}, lane);')
                L.append(f'    uint32_t hi{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + {vgpr_base + 1}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &lo{i}, 4);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off + 4}], &hi{i}, 4);')
            elif esz == 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + {i}, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 4);')
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);')
                L.append(f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 2);')
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);')
                L.append(f'    d->store_data[lane * {stride} + {off}] = static_cast<uint8_t>(val{i});')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _enc_has_semantics(self, enc: InstEncoding) -> bool:
        """Check if any instruction in this encoding has semantics."""
        if not self.semantics:
            return False
        for inst in enc.insts:
            if inst.name in self.semantics.instructions:
                return True
        return False

    def gen_insts(self) -> None:
        """Generate instruction classes deriving from encoding classes."""
        for enc in self.isa_spec.inst_encodings:
            inst_classes = []
            class_func_impls = []
            if not enc.is_alt:
                enc_field_names = {f.name for f in enc.ucode_fields}
                is_smem = enc.enc_name.upper() == 'ENC_SMEM'
                has_sem = self._enc_has_semantics(enc)
                for inst in enc.insts:
                    class_members = []
                    public_members = [cgen.Line('public:')]
                    private_members = [cgen.Line('private:')]
                    opnd_ctor_init = []
                    opnd_body = []
                    reads_dst = self._dst_is_also_source(inst)
                    for opnd in inst.operands:
                        if opnd.is_input:
                            opnd_body.append(
                                f'src_operands_.emplace_back(&{opnd.name});'
                            )
                        elif (reads_dst and opnd.is_output
                              and opnd.name in ('vdst', 'sdst')):
                            # Destination is also read (accumulate, swap, etc.)
                            # but the XML only marks it as output.
                            opnd_body.append(
                                f'src_operands_.emplace_back(&{opnd.name});'
                            )
                        if opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_.emplace_back(&{opnd.name});'
                            )
                        if not opnd.is_input and not opnd.is_output:
                            opnd_body.append(
                                f'dst_operands_.emplace_back(&{opnd.name});'
                            )
                        private_members.append(
                            cgen.Statement(f'Operand {opnd.name}')
                        )
                        if is_smem and opnd.name == 'soffset':
                            opnd_ctor_init.append(
                                f'{opnd.name}(make_smem_offset('
                                f'reinterpret_cast<const OpEncoding*>(inst)))'
                            )
                        elif opnd.name in enc_field_names:
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd.size}, '
                                f'OperandType::{opnd.operand_type}, '
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
                            'void execute(amdgpu::Wavefront &wf) override'
                        )
                    )
                    init_list_parts = [
                        f'{inst.fmt_true_enc_name}("{inst.mnemonic}", '
                        f'reinterpret_cast<const OpEncoding*>(inst))'
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
                        'ds_read', 'ds_write', 'ds_atomic',
                    })
                    ctor_body_parts = list(opnd_body)

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
                    inst_classes.append(s)
                    class_func_impls.append(class_ctor_impl)

                    # Generate execute() body
                    sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    if sem:
                        body = self._gen_execute_body(inst, sem, enc.enc_name)
                        class_func_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::execute'
                                f'(amdgpu::Wavefront &wf) {{\n'
                                f'{body}\n}}'
                            )
                        )
                    else:
                        class_func_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::execute'
                                f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                            )
                        )

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
                })
                is_mem_enc = enc.enc_name.upper() in _MEM_ENC_NAMES
                if is_mem_enc:
                    cpp_includes.extend([
                        (f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/addr_calc.h', False),
                        ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                        ('rocjitsu/vm/amdgpu/mem_state.h', False),
                        ('cstring', True),
                        ('memory', True),
                    ])
                has_mfma = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class == 'mfma'
                    for i in enc.insts
                )
                if has_mfma:
                    cpp_includes.append((
                        f'rocjitsu/isa/arch/amdgpu/'
                        f'{self.isa_spec.arch_name}/mfma_exec.h',
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

                inst_def_file = CppFile(
                    f'{enc.fmt_enc_name.lower()}',
                    self.out_path,
                    True,
                    [
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
                    ],
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
                    smem_offset_helper = cgen.Line(
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
                    class_func_impls.insert(0, smem_offset_helper)

                inst_def_file.gen_code()
                inst_impl_file.gen_code()

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
            for opnd_sel_val in opnd_sels.op_sel_vals:
                opnd_sel_enum += f'{opnd_sel_val[0]} = {opnd_sel_val[1]},'
            opnd_sel_enum += '};'
            code_lines.append(cgen.Line(opnd_sel_enum))

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
                '    : IsaOperand<Isa>(size_bits, opr_type, encoding_value) {}'
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
        resolve_code = cgen.Line(
            'namespace {\n'
            '\n'
            'uint32_t resolve_src_scalar(const amdgpu::Wavefront &wf, int ev) {\n'
            '  if (ev <= 101)\n'
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
            '  if (ev <= 101) {\n'
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
            '  if (ev <= 101) {\n'
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
            '  if (ev <= 101) {\n'
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
            'bool is_vgpr_only_type(OperandType t) {\n'
            '  return t == OperandType::OPR_VGPR || t == OperandType::OPR_VGPR_OR_ACCVGPR ||\n'
            '         t == OperandType::OPR_VGPR_OR_LDS || t == OperandType::OPR_SRC_VGPR;\n'
            '}\n'
            '\n'
            'bool is_immediate_type(OperandType t) {\n'
            '  return t == OperandType::OPR_SIMM16 || t == OperandType::OPR_SIMM32 ||\n'
            '         t == OperandType::OPR_SIMM4 || t == OperandType::OPR_SIMM8 ||\n'
            '         t == OperandType::OPR_LABEL || t == OperandType::OPR_WAITCNT;\n'
            '}\n'
            '\n'
            'uint32_t vgpr_index(OperandType opr_type, int ev) {\n'
            '  if (opr_type == OperandType::OPR_VGPR || opr_type == OperandType::OPR_VGPR_OR_ACCVGPR)\n'
            '    return static_cast<uint32_t>(ev);\n'
            '  return static_cast<uint32_t>(ev - 256);\n'
            '}\n'
            '\n'
            '} // namespace\n'
            '\n'
            'uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(encoding_value_);\n'
            '  return resolve_src_scalar(wf, encoding_value_);\n'
            '}\n'
            '\n'
            'uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
            '  int ev = encoding_value_;\n'
            '  if (is_vgpr_only_type(opr_type_))\n'
            '    return wf.cu().read_vgpr(wf.vgpr_alloc().base + vgpr_index(opr_type_, ev), lane);\n'
            '  if (ev >= 256 && ev <= 511)\n'
            '    return wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>(ev - 256), lane);\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(ev);\n'
            '  return resolve_src_scalar(wf, ev);\n'
            '}\n'
            '\n'
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
            'uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
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
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ev)));\n'
            '  return resolve_src_scalar64(wf, ev);\n'
            '}\n'
            '\n'
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
                'using DecodeFunc = std::function<std::unique_ptr<Instruction>(const MachineInst *)>'
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
                ('functional', True),
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
