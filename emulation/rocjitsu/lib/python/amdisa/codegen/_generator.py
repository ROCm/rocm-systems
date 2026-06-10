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
import textwrap
import re

from dataclasses import dataclass, field as _field

from amdisa.gpuisa import (
    InstEncoding,
    Instruction,
    IsaSpec,
    Operand,
    OperandNamePattern,
)
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

_LITERAL_ENCODING_OPERANDS = {
    'ENC_SOP1': ('Sop1InstLiteralMachineInst', ('ssrc0',)),
    'ENC_SOP2': ('Sop2InstLiteralMachineInst', ('ssrc0', 'ssrc1')),
    'ENC_SOPC': ('SopcInstLiteralMachineInst', ('ssrc0', 'ssrc1')),
    'ENC_VOP1': ('Vop1InstLiteralMachineInst', ('src0',)),
    'ENC_VOP2': ('Vop2InstLiteralMachineInst', ('src0',)),
    'ENC_VOPC': ('VopcInstLiteralMachineInst', ('src0',)),
    'ENC_VOP3': ('Vop3InstLiteralMachineInst', ('src0', 'src1', 'src2')),
    'ENC_VOP3P': ('Vop3pInstLiteralMachineInst', ('src0', 'src1', 'src2')),
    'VOP3_SDST_ENC': (
        'Vop3SdstEncInstLiteralMachineInst',
        ('src0', 'src1', 'src2'),
    ),
}


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

    def _supports_simm64_literal_operands(self) -> bool:
        return 'OPR_SIMM64' in self.isa_spec.operand_types

    @staticmethod
    def _literal_encoding_info(
        enc: InstEncoding, inst_enc_obj: InstEncoding | None, inst: Instruction
    ) -> tuple[str, tuple[str, ...]] | None:
        # Implied-literal instructions are parsed from an alternate XML
        # encoding but generated in the parent encoding class.
        lit_enc = enc if inst.is_implied_literal_enc else (inst_enc_obj or enc)
        return _LITERAL_ENCODING_OPERANDS.get(lit_enc.enc_name.upper())

    @staticmethod
    def _literal_operand_fixup_stmt(
        opnd: Operand, lit_struct: str, size_expr: str | None = None
    ) -> str | None:
        if opnd.operand_type not in ('OPR_SIMM16', 'OPR_SIMM32'):
            return None
        literal_expr = f'reinterpret_cast<const {lit_struct} *>(inst)->simm32'
        if opnd.operand_type == 'OPR_SIMM16' or opnd.size == 16:
            literal_expr = f'({literal_expr} & 0xFFFFu)'
        return (
            f'{opnd.name} = Operand({size_expr or opnd.size}, '
            f'OperandType::{opnd.operand_type}, static_cast<int>({literal_expr}));'
        )

    @staticmethod
    def _has_inline_literal_operand(inst: Instruction) -> bool:
        return any(
            opnd.name == 'literal' and opnd.operand_type in ('OPR_SIMM16', 'OPR_SIMM32')
            for opnd in inst.operands
        )

    @staticmethod
    def _with_scalar_literal_fma_operand(inst: Instruction) -> Instruction:
        if inst.name not in ('S_FMAAK_F32', 'S_FMAMK_F32') or any(
            op.name == 'src2' for op in inst.operands
        ):
            return inst
        return Instruction(
            inst.name,
            inst.enc_name,
            inst.opcode,
            [
                *inst.operands,
                Operand(
                    'src2',
                    32,
                    'OPR_SIMM32',
                    is_input=True,
                    is_output=False,
                    is_implicit=False,
                    is_binary_ucode_required=True,
                    order=len(inst.operands),
                ),
            ],
            inst.is_implied_literal_enc,
        )

    def _has_machine_inst_struct(self, struct_name: str) -> bool:
        return struct_name in {
            f'{enc.fmt_enc_name}MachineInst' for enc in self.isa_spec.inst_encodings
        }

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
        # VOPD is generated from a hand-written C++ template because the XML
        # describes the packed dual-slot encoding, while the normal emitters
        # model one instruction and one operand list at a time. Keeping the
        # template here preserves one-step regeneration for gfx1250.
        self.gen_vopd()
        self.gen_insts()
        self.gen_decoder()
        self.gen_test_encodings()

    def _supports_generated_vopd(self) -> bool:
        return self.isa_spec.arch_name == 'gfx1250' and self.isa_spec.profile.has_vopd

    def gen_vopd(self) -> None:
        """Generate gfx1250 VOPD dual-issue decoder/executor files.

        VOPD is skipped by the normal XML instruction generation because it
        uses a dual-slot encoding with bespoke operand packing. Keeping the
        target-specific C++ body here lets the same regeneration path recreate
        the VOPD files and the generated decoder hook together.
        """
        if not self._supports_generated_vopd():
            return

        import os

        arch = self.isa_spec.arch_name
        out_dir = os.path.join(self.out_path, arch)
        os.makedirs(out_dir, exist_ok=True)
        guard = f'ROCJITSU_ISA_ARCH_AMDGPU_{arch.upper()}_VOPD_H_'

        header = textwrap.dedent('''
            // Copyright (c) 2026 Advanced Micro Devices, Inc.
            // SPDX-License-Identifier: MIT
            //
            // AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.
            // See lib/python/amdisa/README.md for regeneration instructions.

            #ifndef @GUARD@
            #define @GUARD@

            #include "rocjitsu/isa/arch/amdgpu/@ARCH@/encodings.h"
            #include "rocjitsu/isa/arch/amdgpu/@ARCH@/operand.h"
            #include <cstdint>
            #include <string>

            namespace rocjitsu {
            namespace @ARCH@ {

            class Vopd : public IsaInstruction<Isa>
            {
              public:
              explicit Vopd(const MachineInst *inst);
              static bool is_vopd(const MachineInst *inst);
              void execute_impl(amdgpu::Wavefront &wf);

              private:
              enum class Format : uint8_t { VopdXy, Vopd3 };

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
              static bool is_float32_op(uint16_t op);
              static uint32_t apply_neg(uint32_t value, uint8_t neg_bits, uint8_t src_idx);
              static uint32_t execute_slot(const Slot &slot, amdgpu::Wavefront &wf,
                                           uint32_t lane);
              static uint32_t bitop2(uint32_t src0, uint32_t src1, uint32_t truth_table);
              std::string format_slot(const Slot &slot) const;
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
            ''').lstrip().replace('@ARCH@', arch).replace('@GUARD@', guard)

        impl = textwrap.dedent('''
            // Copyright (c) 2026 Advanced Micro Devices, Inc.
            // SPDX-License-Identifier: MIT
            //
            // AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.
            // See lib/python/amdisa/README.md for regeneration instructions.

            #include "rocjitsu/isa/arch/amdgpu/@ARCH@/vopd.h"
            #include "util/except.h"
            #include "rocjitsu/vm/amdgpu/wavefront.h"
            #include <algorithm>
            #include <bit>
            #include <cmath>
            #include <format>
            #include <string>

            namespace rocjitsu {
            namespace @ARCH@ {

            namespace {

            Operand make_src0(uint32_t bits, bool vopd3, bool use_literal, uint32_t literal,
                              uint16_t encoded) {
              if (use_literal && encoded == 255)
                return Operand(bits, OperandType::OPR_SIMM32, static_cast<int>(literal));
              return Operand(bits, vopd3 ? OperandType::OPR_SRC_SIMPLE : OperandType::OPR_SRC,
                             encoded);
            }

            std::string operand_list(const Operand &dst, const Operand &src0,
                                     const Operand &src1) {
              return dst.name() + ", " + src0.name() + ", " + src1.name();
            }

            } // namespace

            bool Vopd::is_vopd(const MachineInst *inst) {
              uint32_t word0 = *reinterpret_cast<const uint32_t *>(inst);
              return (word0 >> 24) == 0xCF || (word0 >> 26) == 0x32;
            }

            const char *Vopd::op_name(uint16_t op) {
              switch (op) {
              case 0:
                return "v_dual_fmac_f32";
              case 1:
                return "v_dual_fmaak_f32";
              case 2:
                return "v_dual_fmamk_f32";
              case 3:
                return "v_dual_mul_f32";
              case 4:
                return "v_dual_add_f32";
              case 5:
                return "v_dual_sub_f32";
              case 6:
                return "v_dual_subrev_f32";
              case 7:
                return "v_dual_mul_dx9_zero_f32";
              case 8:
                return "v_dual_mov_b32";
              case 9:
                return "v_dual_cndmask_b32";
              case 10:
                return "v_dual_max_num_f32";
              case 11:
                return "v_dual_min_num_f32";
              case 16:
                return "v_dual_add_nc_u32";
              case 17:
                return "v_dual_lshlrev_b32";
              case 18:
                return "v_dual_bitop2_b32";
              case 19:
                return "v_dual_fma_f32";
              case 20:
                return "v_dual_sub_nc_u32";
              case 21:
                return "v_dual_lshrrev_b32";
              case 22:
                return "v_dual_ashrrev_i32";
              case 23:
                return "v_dual_max_i32";
              case 24:
                return "v_dual_min_i32";
              case 32:
                return "v_dual_add_f64";
              case 33:
                return "v_dual_mul_f64";
              case 34:
                return "v_dual_min_num_f64";
              case 35:
                return "v_dual_max_num_f64";
              case 36:
                return "v_dual_fmac_f64";
              default:
                return "v_dual_unknown";
              }
            }

            bool Vopd::is_float32_op(uint16_t op) {
              switch (op) {
              case 0:
              case 1:
              case 2:
              case 3:
              case 4:
              case 5:
              case 6:
              case 7:
              case 10:
              case 11:
              case 19:
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

            uint32_t Vopd::execute_slot(const Slot &slot, amdgpu::Wavefront &wf,
                                        uint32_t lane) {
              uint32_t src0 = slot.src0->read_lane(wf, lane);
              uint32_t src1 = slot.src1->read_lane(wf, lane);
              uint32_t src2 = slot.has_src2_operand ? slot.src2->read_lane(wf, lane)
                                                     : slot.src2_imm;
              if (is_float32_op(slot.op)) {
                src0 = apply_neg(src0, slot.neg, 0);
                src1 = apply_neg(src1, slot.neg, 1);
                src2 = apply_neg(src2, slot.neg, 2);
              }

              switch (slot.op) {
              case 0: {
                float result = std::fma(std::bit_cast<float>(src0),
                                        std::bit_cast<float>(src1),
                                        std::bit_cast<float>(slot.dst->read_lane(wf, lane)));
                return std::bit_cast<uint32_t>(result);
              }
              case 1: {
                float result = std::fma(std::bit_cast<float>(src0),
                                        std::bit_cast<float>(src1),
                                        std::bit_cast<float>(src2));
                return std::bit_cast<uint32_t>(result);
              }
              case 2: {
                float result = std::fma(std::bit_cast<float>(src0),
                                        std::bit_cast<float>(src2),
                                        std::bit_cast<float>(src1));
                return std::bit_cast<uint32_t>(result);
              }
              case 3: {
                float result = std::bit_cast<float>(src0) * std::bit_cast<float>(src1);
                return std::bit_cast<uint32_t>(result);
              }
              case 7: {
                float lhs = std::bit_cast<float>(src0);
                float rhs = std::bit_cast<float>(src1);
                if (lhs == 0.0f || rhs == 0.0f)
                  return std::bit_cast<uint32_t>(0.0f);
                return std::bit_cast<uint32_t>(lhs * rhs);
              }
              case 4: {
                float result = std::bit_cast<float>(src0) + std::bit_cast<float>(src1);
                return std::bit_cast<uint32_t>(result);
              }
              case 5: {
                float result = std::bit_cast<float>(src0) - std::bit_cast<float>(src1);
                return std::bit_cast<uint32_t>(result);
              }
              case 6: {
                float result = std::bit_cast<float>(src1) - std::bit_cast<float>(src0);
                return std::bit_cast<uint32_t>(result);
              }
              case 8:
                return src0;
              case 9: {
                uint64_t condition = slot.uses_vcc ? wf.vcc() : slot.src2->read_scalar64(wf);
                return ((condition >> lane) & 1u) ? src1 : src0;
              }
              case 10: {
                float result = std::fmax(std::bit_cast<float>(src0),
                                         std::bit_cast<float>(src1));
                return std::bit_cast<uint32_t>(result);
              }
              case 11: {
                float result = std::fmin(std::bit_cast<float>(src0),
                                         std::bit_cast<float>(src1));
                return std::bit_cast<uint32_t>(result);
              }
              case 16:
                return src0 + src1;
              case 17:
                return src1 << (src0 & 31u);
              case 18:
                return bitop2(src0, src1, slot.src2_imm);
              case 19: {
                float result = std::fma(std::bit_cast<float>(src0),
                                        std::bit_cast<float>(src1),
                                        std::bit_cast<float>(src2));
                return std::bit_cast<uint32_t>(result);
              }
              case 20:
                return src0 - src1;
              case 21:
                return src1 >> (src0 & 31u);
              case 22:
                return static_cast<uint32_t>(static_cast<int32_t>(src1) >> (src0 & 31u));
              case 23:
                return static_cast<uint32_t>(std::max(static_cast<int32_t>(src0),
                                                      static_cast<int32_t>(src1)));
              case 24:
                return static_cast<uint32_t>(std::min(static_cast<int32_t>(src0),
                                                      static_cast<int32_t>(src1)));
              default:
                throw util::UnimplementedInst(op_name(slot.op));
              }
            }

            Vopd::Vopd(const MachineInst *inst)
                : IsaInstruction<Isa>("vopd", make_exec_fn<Vopd>()),
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

              if ((word0_ >> 24) == 0xCF) {
                format_ = Format::Vopd3;
                size_ = 12;
                encoding_id_ = 0xCF;
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

                dstx_ = Operand(32, OperandType::OPR_VGPR, vdstx);
                dsty_ = Operand(32, OperandType::OPR_VGPR, vdsty);
                srcx0_ = make_src0(32, true, false, 0, srcx0);
                srcy0_ = make_src0(32, true, false, 0, srcy0);
                srcx1_ = Operand(32, OperandType::OPR_VGPR, vsrcx1);
                srcy1_ = Operand(32, OperandType::OPR_VGPR, vsrcy1);
                srcx2_ = (opx_ == 9) ? Operand(64, OperandType::OPR_SREG, vsrcx2)
                                     : Operand(32, OperandType::OPR_VGPR, vsrcx2);
                srcy2_ = (opy_ == 9) ? Operand(64, OperandType::OPR_SREG, vsrcy2)
                                     : Operand(32, OperandType::OPR_VGPR, vsrcy2);
              } else {
                format_ = Format::VopdXy;
                encoding_id_ = 0x32;
                opx_ = static_cast<uint16_t>((word0_ >> 22) & 0xF);
                opy_ = static_cast<uint16_t>((word0_ >> 17) & 0x1F);
                uint16_t srcx0 = static_cast<uint16_t>(word0_ & 0x1FF);
                uint16_t vsrcx1 = static_cast<uint16_t>((word0_ >> 9) & 0xFF);
                uint16_t srcy0 = static_cast<uint16_t>(word1_ & 0x1FF);
                uint16_t vsrcy1 = static_cast<uint16_t>((word1_ >> 9) & 0xFF);
                uint16_t vdstx = static_cast<uint16_t>((word1_ >> 24) & 0xFF);
                uint16_t vdsty_hi = static_cast<uint16_t>((word1_ >> 17) & 0x7F);
                uint16_t vdsty = static_cast<uint16_t>((vdsty_hi << 1) | ((~vdstx) & 1u));
                has_literal_ = srcx0 == 255 || srcy0 == 255 || opx_ == 1 ||
                               opx_ == 2 || opy_ == 1 || opy_ == 2;
                size_ = has_literal_ ? 12 : 8;
                if (has_literal_) {
                  word2_ = words[2];
                  literal_ = word2_;
                }

                dstx_ = Operand(32, OperandType::OPR_VGPR, vdstx);
                dsty_ = Operand(32, OperandType::OPR_VGPR, vdsty);
                srcx0_ = make_src0(32, false, has_literal_, literal_, srcx0);
                srcy0_ = make_src0(32, false, has_literal_, literal_, srcy0);
                srcx1_ = Operand(32, OperandType::OPR_VGPR, vsrcx1);
                srcy1_ = Operand(32, OperandType::OPR_VGPR, vsrcy1);
              }

              dstx_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);
              dsty_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);
              srcx0_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
              srcy0_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
              srcx1_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);
              srcy1_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);
              srcx2_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);
              srcy2_.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);

              opcode_ = static_cast<uint16_t>((opx_ << 8) | opy_);
              init_operands();
              mnemonic_storage_ = std::string(op_name(opx_)) + " :: " + op_name(opy_);
              mnemonic_ = mnemonic_storage_;
              disassembly_ = format_slot(x_) + " :: " + format_slot(y_);
            }

            void Vopd::init_operands() {
              const bool vopd3 = format_ == Format::Vopd3;
              x_ = Slot{opx_, &dstx_, &srcx0_, &srcx1_, &srcx2_, 0, negx_, false, false,
                        false};
              y_ = Slot{opy_, &dsty_, &srcy0_, &srcy1_, &srcy2_, 0, negy_, false, false,
                        false};

              if (vopd3) {
                x_.has_src2_operand = opx_ == 9 || opx_ == 19;
                y_.has_src2_operand = opy_ == 9 || opy_ == 19;
                x_.src2_is_imm = opx_ == 18;
                y_.src2_is_imm = opy_ == 18;
                x_.src2_imm = static_cast<uint32_t>(srcx2_.encoding_value());
                y_.src2_imm = static_cast<uint32_t>(srcy2_.encoding_value());
              } else {
                x_.uses_vcc = opx_ == 9;
                y_.uses_vcc = opy_ == 9;
                if (opx_ == 1 || opx_ == 2) {
                  x_.src2_is_imm = true;
                  x_.src2_imm = literal_;
                }
                if (opy_ == 1 || opy_ == 2) {
                  y_.src2_is_imm = true;
                  y_.src2_imm = literal_;
                }
              }

              dst_operands_[0] = &dstx_;
              dst_operands_[1] = &dsty_;
              src_operands_[0] = &srcx0_;
              src_operands_[1] = &srcx1_;
              src_operands_[2] = &srcy0_;
              src_operands_[3] = &srcy1_;
              num_dst_ = 2;
              num_src_ = 4;
            }

            std::string Vopd::format_slot(const Slot &slot) const {
              std::string out = op_name(slot.op);
              out += " ";
              switch (slot.op) {
              case 8:
                out += slot.dst->name() + ", " + slot.src0->name();
                break;
              case 9:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                if (!slot.uses_vcc)
                  out += ", " + slot.src2->name();
                break;
              case 18:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                out += std::format(" bitop3:0x{:02x}", slot.src2_imm & 0xFF);
                break;
              case 19:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1) + ", " +
                       slot.src2->name();
                break;
              case 1:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                out += std::format(", 0x{:08x}", slot.src2_imm);
                break;
              case 2:
                out += slot.dst->name() + ", " + slot.src0->name();
                out += std::format(", 0x{:08x}, ", slot.src2_imm);
                out += slot.src1->name();
                break;
              default:
                out += operand_list(*slot.dst, *slot.src0, *slot.src1);
                break;
              }
              return out;
            }

            void Vopd::execute_impl(amdgpu::Wavefront &wf) {
              uint64_t exec = wf.exec();
              for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
                if (!(exec & (1ULL << lane)))
                  continue;
                uint32_t x_result = execute_slot(x_, wf, lane);
                uint32_t y_result = execute_slot(y_, wf, lane);
                x_.dst->write_lane(wf, lane, x_result);
                y_.dst->write_lane(wf, lane, y_result);
              }
            }

            } // namespace @ARCH@
            } // namespace rocjitsu
            ''').lstrip().replace('@ARCH@', arch)

        with open(os.path.join(out_dir, 'vopd.h'), 'w') as f:
            f.write(header)
        with open(os.path.join(out_dir, 'vopd.cpp'), 'w') as f:
            f.write(impl)

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
            default_cond = dict(inst_enc.enc_conds).get('default_encoding', 'true')
            has_real_default_check = inst_enc.bit_cnt < 64 and default_cond != 'false'

            if has_real_default_check and inst_enc.has_implied_literal_ops:
                size_condition = '!default_encoding() || hasImpliedLiteral()'
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

            has_op = any(f.name == 'op' for f in inst_enc.ucode_fields)
            size_line = (
                ' size_ = sizeof(OpEncoding);\n'
                '  raw_encoding_ = reinterpret_cast<const uint32_t *>(&inst_);\n'
                '  encoding_id_ = raw_encoding_[0] >> 23;'
            )
            if has_op:
                size_line += '\n  opcode_ = inst_.op;'
            literal64_conds = [
                name for name, _ in inst_enc.enc_conds if name.startswith('has_lit64')
            ]
            literal64_condition = ' || '.join(f'{name}()' for name in literal64_conds)
            literal32_conds = [
                name
                for name, _ in inst_enc.enc_conds
                if name.startswith('has_lit') and not name.startswith('has_lit64')
            ]
            literal32_condition = ' || '.join(f'{name}()' for name in literal32_conds)
            if literal64_condition and size_condition is not None:
                size_line += (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                    f' else if ({size_condition}) size_ += sizeof(MachineInst);'
                )
            elif literal64_condition and literal32_condition:
                size_line += (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                    f' else if ({literal32_condition}) size_ += sizeof(MachineInst);'
                )
            elif literal64_condition:
                size_line += (
                    f' if ({literal64_condition}) size_ += 2 * sizeof(MachineInst);'
                )
            elif size_condition is not None:
                size_line += f' if ({size_condition})' f' size_ += sizeof(MachineInst);'
            elif literal32_condition:
                size_line += (
                    f' if ({literal32_condition}) size_ += sizeof(MachineInst);'
                )
            if inst_enc.has_implied_literal_ops:
                size_line += (
                    ' if (hasImpliedLiteral())'
                    ' literal_ = reinterpret_cast<const uint32_t *>(inst)[1];'
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
                class_func_impls.append(
                    cgen.Line(
                        f'void {inst_enc.fmt_enc_name}::build_modifiers'
                        f'(std::string &out) const '
                        f'{{ auto *inst = &inst_;(void)inst;'
                        f'{mod_impl}}}'
                    )
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

            class_members.extend(public_members)
            class_members.append(
                cgen.Statement(f'using OpEncoding = {inst_enc.fmt_enc_name}MachineInst')
            )
            class_members.append(cgen.Statement('const OpEncoding inst_'))
            if inst_enc.has_implied_literal_ops:
                class_members.append(cgen.Statement('uint32_t literal_ = 0'))
            # FLAT encoding bases need an owned string for the dynamic mnemonic.
            if rule.use_flat_mnemonic:
                class_members.append(cgen.Statement('std::string owned_mnemonic_'))
            # VOP1/VOP2 encoding bases store DPP control fields.
            # apply_dpp() is a free function in dpp_sdwa_ops.h.
            if inst_enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
                class_members.append(cgen.Statement('uint32_t dpp_ctrl_ = 0'))
                class_members.append(cgen.Statement('uint32_t dpp_row_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bank_mask_ = 0xF'))
                class_members.append(cgen.Statement('uint32_t dpp_bound_ctrl_ = 0'))
                class_members.append(
                    cgen.Statement('std::unique_ptr<DppOperand> dpp_src0_')
                )
                class_members.append(
                    cgen.Statement('std::unique_ptr<DppOperand> dpp_src1_')
                )
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
                if inst_enc.enc_name.upper() != 'ENC_VOPC':
                    class_members.append(
                        cgen.Statement('uint32_t sdwa_dst_sel_ = amdgpu::sdwa::DWORD')
                    )
                    class_members.append(
                        cgen.Statement('uint32_t sdwa_dst_unused_ = 0')
                    )
                    class_members.append(cgen.Statement('bool sdwa_clamp_ = false'))
                else:
                    class_members.append(cgen.Statement('uint32_t sdwa_sdst_ = 106'))
                    class_members.append(cgen.Statement('bool sdwa_sd_ = false'))
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

    def _encoding_implicit_uses_impl(
        self, inst_enc: InstEncoding, enc_field_names: set[str]
    ) -> str:
        """Return C++ body for hidden register uses on an encoding base."""
        if (
            inst_enc.enc_name.upper() == 'ENC_FLAT'
            and {'seg', 'saddr'} <= enc_field_names
        ):
            return (
                'if (inst_.saddr == 0x7F) return;'
                'if (inst_.seg == 1) {'
                'uses.expand(RegisterRef{RegClass::SGPR, '
                'static_cast<uint16_t>(inst_.saddr), 1});'
                '} else if (inst_.seg == 2) {'
                'uses.expand(RegisterRef{RegClass::SGPR, '
                'static_cast<uint16_t>(inst_.saddr), 2});'
                '}'
            )
        return ''

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
    _READS_DST_CLASSES = frozenset(
        {
            'vector_dot',
            'vector_swap',
            'mad_mixlo_f16',
            'mad_mixhi_f16',
            'mad_mixlo_bf16',
            'mad_mixhi_bf16',
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

    _VGPR_MSB_SRC_ROLES = ('Src0', 'Src1', 'Src2')

    @staticmethod
    def _operand_type_can_name_vgpr(opr_type: str) -> bool:
        """Return True if an operand type can resolve to VGPR storage."""
        opr_type = opr_type.upper()
        if opr_type in {'OPR_SRC', 'OPR_SRC_NOINLINE', 'OPR_SRC_SIMPLE'}:
            return True
        return (
            opr_type.startswith('OPR_VGPR')
            or opr_type.startswith('OPR_SRC_VGPR')
            or 'ACCVGPR' in opr_type
        )

    @classmethod
    def _operand_can_use_vgpr_msb(cls, opnd: Operand) -> bool:
        """Return True if an operand can name a VGPR at execution time."""
        return cls._operand_type_can_name_vgpr(opnd.operand_type)

    def _operand_uses_packed_16bit_source(self, enc_name: str, opnd: Operand) -> bool:
        """Return True for gfx1250 E32 16-bit sources with packed-half selectors."""
        profile = self.isa_spec.profile
        if not profile.uses_packed_16bit_e32_source_selectors:
            return False
        if enc_name.upper() not in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
            return False
        if not opnd.is_input or opnd.size != 16:
            return False
        return opnd.operand_type in ('OPR_SRC', 'OPR_SRC_VGPR')

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

    def _supports_gfx1250_scaled_wmma_vop3px2(self) -> bool:
        return (
            self.isa_spec.arch_name.lower() == 'gfx1250'
            and self.isa_spec.profile.generate_scaled_wmma_vop3px2
        )

    def _gfx1250_f8f6f4_wmma_shape(
        self, inst: Instruction
    ) -> tuple[int, int, int] | None:
        if self.isa_spec.arch_name != 'gfx1250' or not inst.name.startswith('V_WMMA_'):
            return None
        m = re.match(
            r'V_WMMA_(?:F32|F16|BF16|I32)_(\d+)X(\d+)X(\d+)_?F8F6F4$',
            inst.name,
        )
        if not m:
            return None
        return tuple(int(x) for x in m.groups())

    def _gfx1250_swmmac_has_modifiers(self, inst: Instruction) -> bool:
        return self.isa_spec.arch_name == 'gfx1250' and inst.name.startswith(
            'V_SWMMAC_'
        )

    def _gfx1250_matrix_fmt_operand_size_expr(
        self, shape: tuple[int, int, int] | None, opnd_name: str
    ) -> str | None:
        if shape is None or opnd_name not in ('src0', 'src1'):
            return None
        m, n, k = shape
        dim = m if opnd_name == 'src0' else n
        if opnd_name == 'src0':
            fmt_expr = 'reinterpret_cast<const OpEncoding *>(inst)->opsel'
        else:
            fmt_expr = (
                '((reinterpret_cast<const OpEncoding *>(inst)->pad_14 << 2) | '
                'reinterpret_cast<const OpEncoding *>(inst)->opsel_hi)'
            )
        return f'gfx1250_matrix_fmt_operand_size_bits({fmt_expr}, {dim}, {k})'

    @staticmethod
    def _emit_gfx1250_matrix_fmt_helpers() -> str:
        """Emit C++ helpers for gfx1250 VOP3P packed and matrix quirks."""
        return (
            textwrap.dedent('''\
            namespace {
            struct PkF32Words {
              uint32_t lo;
              uint32_t hi;
            };

            PkF32Words read_pk_f32_words(const Operand &operand, const amdgpu::Wavefront &wf, uint32_t lane) {
              if (auto literal = operand.literal64_value())
                return {static_cast<uint32_t>(*literal), static_cast<uint32_t>(*literal >> 32)};

              const uint32_t lo = operand.read_lane(wf, lane);
              const auto reg = operand.to_register_ref();
              if (!reg || reg->cls != RegClass::VGPR)
                return {lo, lo};

              const uint64_t raw = operand.read_lane64(wf, lane);
              return {static_cast<uint32_t>(raw), static_cast<uint32_t>(raw >> 32)};
            }
            ''')
            + (
                'const char *gfx1250_matrix_fmt_name(uint32_t fmt) {\n'
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
                'const char *gfx1250_matrix_scale_fmt_name(uint32_t fmt) {\n'
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
                'uint32_t gfx1250_matrix_fmt_element_bits(uint32_t fmt) {\n'
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
                'int gfx1250_matrix_fmt_operand_size_bits(uint32_t fmt, uint32_t dim, uint32_t k) {\n'
                '  return static_cast<int>((dim * k * gfx1250_matrix_fmt_element_bits(fmt)) / 32);\n'
                '}\n'
                '\n'
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
                '  uint32_t raw = src.read_lane(wf, lane);\n'
                '  if (!src_is_f16)\n'
                '    return std::bit_cast<float>(raw);\n'
                '  return util::f16_to_f32(read_fma_mix_f16_bits(raw, src_selector, high_half));\n'
                '}\n'
                '\n'
                'float read_fma_mix_bf16_source_f32(const Operand &src, const amdgpu::Wavefront &wf, uint32_t lane,\n'
                '                                   uint32_t src_selector, bool src_is_bf16, bool high_half) {\n'
                '  uint32_t raw = src.read_lane(wf, lane);\n'
                '  if (!src_is_bf16)\n'
                '    return std::bit_cast<float>(raw);\n'
                '  return util::bf16_to_f32(read_fma_mix_bf16_bits(raw, src_selector, high_half));\n'
                '}\n'
                '} // namespace'
            )
        )

    def _emit_hwreg_helpers(self) -> str:
        """Emit C++ helpers for target-specific generated SOPK hwreg access."""
        profile = self.isa_spec.profile
        mode_id = profile.hwreg_mode_id
        status_id = profile.hwreg_status_id
        wave_sched_id = profile.hwreg_wave_sched_mode_id
        ib_sts2_id = profile.hwreg_ib_sts2_id
        constexprs = []
        if mode_id is not None:
            constexprs.append(f'constexpr uint32_t HW_REG_MODE = {mode_id};')
        constexprs.append(f'constexpr uint32_t HW_REG_STATUS = {status_id};')
        constexprs.extend(
            [
                'constexpr uint32_t HW_REG_HW_ID1 = 4;',
                'constexpr uint32_t HW_REG_HW_ID2 = 5;',
                'constexpr uint32_t HW_REG_GPR_ALLOC = 6;',
                'constexpr uint32_t HW_REG_VGPR_ALLOC = 7;',
            ]
        )
        if wave_sched_id is not None:
            constexprs.append(
                f'constexpr uint32_t HW_REG_WAVE_SCHED_MODE = {wave_sched_id};'
            )
        if ib_sts2_id is not None:
            constexprs.append(f'constexpr uint32_t HW_REG_IB_STS2 = {ib_sts2_id};')

        read_cases = []
        if mode_id is not None:
            read_cases.append(
                '  case HW_REG_MODE:\n'
                '    reg_val = wf.mode_raw();\n'
                '    return true;'
            )
        read_cases.extend(
            [
                '  case HW_REG_STATUS:\n'
                '    reg_val = wf.status_raw();\n'
                '    return true;',
                '  case HW_REG_HW_ID1:\n'
                '    reg_val = static_cast<uint32_t>(wf.cu().id());\n'
                '    return true;',
                '  case HW_REG_HW_ID2:\n'
                '    reg_val = static_cast<uint32_t>(wf.cu().id() >> 16);\n'
                '    return true;',
                '  case HW_REG_GPR_ALLOC:\n'
                '    reg_val = (wf.sgpr_alloc().count & 0xFFu) | ((wf.sgpr_alloc().base & 0xFFu) << 8);\n'
                '    return true;',
                '  case HW_REG_VGPR_ALLOC:\n'
                '    reg_val = (wf.vgpr_alloc().count & 0xFFu) | ((wf.vgpr_alloc().base & 0xFFu) << 8);\n'
                '    return true;',
            ]
        )
        if wave_sched_id is not None:
            read_cases.append(
                '  case HW_REG_WAVE_SCHED_MODE:\n'
                '    reg_val = wf.wave_sched_mode_raw();\n'
                '    return true;'
            )
        if ib_sts2_id is not None:
            read_cases.append(
                '  case HW_REG_IB_STS2:\n' '    reg_val = 0;\n' '    return true;'
            )

        write_cases = []
        if mode_id is not None:
            write_cases.append(
                '  case HW_REG_MODE:\n'
                '    wf.set_mode_raw(insert_hwreg_field(wf.mode_raw(), src, offset, mask));\n'
                '    return true;'
            )
        write_cases.append(
            '  case HW_REG_STATUS:\n'
            '    wf.set_status_raw(insert_hwreg_field(wf.status_raw(), src, offset, mask));\n'
            '    return true;'
        )
        if wave_sched_id is not None:
            write_cases.append(
                '  case HW_REG_WAVE_SCHED_MODE:\n'
                '    wf.set_wave_sched_mode_raw(insert_hwreg_field(wf.wave_sched_mode_raw(), src, offset, mask));\n'
                '    return true;'
            )

        return (
            'namespace {\n' + '\n'.join(constexprs) + '\n\n'
            '[[maybe_unused]] uint32_t insert_hwreg_field(uint32_t reg_val, uint32_t src, uint32_t offset, uint32_t mask) {\n'
            '  return (reg_val & ~(mask << offset)) | ((src & mask) << offset);\n'
            '}\n'
            '\n'
            '[[maybe_unused]] bool read_hwreg(amdgpu::Wavefront &wf, uint32_t reg_id, uint32_t &reg_val) {\n'
            '  switch (reg_id) {\n' + '\n'.join(read_cases) + '\n'
            '  default:\n'
            '    return false;\n'
            '  }\n'
            '}\n'
            '\n'
            '[[maybe_unused]] bool write_hwreg(amdgpu::Wavefront &wf, uint32_t reg_id, uint32_t offset, uint32_t mask,\n'
            '                 uint32_t src) {\n'
            '  switch (reg_id) {\n' + '\n'.join(write_cases) + '\n'
            '  default:\n'
            '    return false;\n'
            '  }\n'
            '}\n'
            '\n'
            '} // namespace'
        )

    @staticmethod
    def _emit_gfx1250_scaled_wmma_vop3px2_class() -> str:
        return textwrap.dedent('''\
            class VWmmaScaleF3216x16x128F8f6f4Vop3px2 : public Vop3p {
            public:
              VWmmaScaleF3216x16x128F8f6f4Vop3px2(const MachineInst *inst);
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

    @staticmethod
    def _emit_gfx1250_scaled_wmma_vop3px2_impls() -> str:
        return textwrap.dedent('''\
            VWmmaScaleF3216x16x128F8f6f4Vop3px2::VWmmaScaleF3216x16x128F8f6f4Vop3px2(const MachineInst *inst)
                : Vop3p("v_wmma_scale_f32_16x16x128_f8f6f4", reinterpret_cast<const OpEncoding *>(inst + 2),
                        make_exec_fn<VWmmaScaleF3216x16x128F8f6f4Vop3px2>()),
                  vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst + 2)->vdst),
                  src0(gfx1250_matrix_fmt_operand_size_bits(
                           reinterpret_cast<const OpEncoding *>(inst + 2)->opsel, 16, 128),
                       OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst + 2)->src0),
                  src1(gfx1250_matrix_fmt_operand_size_bits(
                           ((reinterpret_cast<const OpEncoding *>(inst + 2)->pad_14 << 2) |
                            reinterpret_cast<const OpEncoding *>(inst + 2)->opsel_hi),
                           16, 128),
                       OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst + 2)->src1),
                  src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
                       reinterpret_cast<const OpEncoding *>(inst + 2)->src2),
                  scale_src0(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src0),
                  scale_src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
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

            void VWmmaScaleF3216x16x128F8f6f4Vop3px2::build_modifiers(std::string &out) const {
              const uint32_t matrix_a_fmt = inst_.opsel;
              const uint32_t matrix_b_fmt = (inst_.pad_14 << 2) | inst_.opsel_hi;
              if (matrix_a_fmt != 0) {
                out += " matrix_a_fmt:";
                out += gfx1250_matrix_fmt_name(matrix_a_fmt);
              }
              if (matrix_b_fmt != 0) {
                out += " matrix_b_fmt:";
                out += gfx1250_matrix_fmt_name(matrix_b_fmt);
              }
              if (scale_inst_.opsel & 0x1u)
                out += " matrix_a_scale:MATRIX_SCALE_ROW1";
              if (scale_inst_.opsel_hi & 0x1u)
                out += " matrix_b_scale:MATRIX_SCALE_ROW1";
              const uint32_t matrix_a_scale_fmt = scale_inst_.neg & 0x3u;
              const uint32_t matrix_b_scale_fmt = scale_inst_.neg_hi & 0x3u;
              if (matrix_a_scale_fmt != 0) {
                out += " matrix_a_scale_fmt:";
                out += gfx1250_matrix_scale_fmt_name(matrix_a_scale_fmt);
              }
              if (matrix_b_scale_fmt != 0) {
                out += " matrix_b_scale_fmt:";
                out += gfx1250_matrix_scale_fmt_name(matrix_b_scale_fmt);
              }
              if ((scale_inst_.opsel >> 2) & 0x1u)
                out += " matrix_a_reuse";
              if (scale_inst_.pad_14)
                out += " matrix_b_reuse";
            }

            void VWmmaScaleF3216x16x128F8f6f4Vop3px2::execute_impl(amdgpu::Wavefront &wf) {
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
                const_acc = src2.read_scalar(wf);
              }

              const uint32_t matrix_a_fmt = inst_.opsel;
              const uint32_t matrix_b_fmt = (inst_.pad_14 << 2) | inst_.opsel_hi;
              const uint32_t matrix_a_scale =
                  (scale_inst_.opsel & 0x1u) | (((scale_inst_.opsel >> 2u) & 0x1u) << 1u);
              const uint32_t matrix_b_scale =
                  (scale_inst_.opsel_hi & 0x1u) | ((scale_inst_.pad_14 & 0x1u) << 1u);
              const uint32_t matrix_a_scale_fmt = scale_inst_.neg & 0x3u;
              const uint32_t matrix_b_scale_fmt = scale_inst_.neg_hi & 0x3u;

              bool dispatched = amdgpu::dispatch_matrix_fmt_pair(
                  matrix_a_fmt, matrix_b_fmt,
                  [&](uint32_t a_bits, uint32_t b_bits, auto extract_a, auto extract_b) {
                    amdgpu::exec_wmma_f32_scaled_mixed(
                        cu, 16, 16, 128, a_bits, b_bits, dst, src0_base, src1_base, s2, extract_a, extract_b,
                        const_acc, [&](uint32_t lane) { return scale_src0.read_lane(wf, lane); },
                        [&](uint32_t lane) { return scale_src1.read_lane(wf, lane); }, matrix_a_scale,
                        matrix_b_scale, matrix_a_scale_fmt, matrix_b_scale_fmt);
                  });
              if (!dispatched)
                throw util::UnimplementedInst(mnemonic());
            }
            ''')

    @staticmethod
    def _emit_gfx1250_scaled_wmma_vop3px2_decoder_helpers() -> str:
        return textwrap.dedent('''\
            namespace {

            bool isWmmaScaleF32F8f6f4Vop3px2(const MachineInst *opcode) {
              const auto *low = reinterpret_cast<const Vop3pMachineInst *>(opcode);
              if (low->encoding != 0xcc || low->op != 0x35)
                return false;

              const auto *high = reinterpret_cast<const Vop3pMachineInst *>(opcode + 2);
              return high->encoding == 0xcc && high->op == 0x33;
            }

            } // namespace
            ''')

    def _gen_execute_body(
        self, inst: Instruction, sem: InstructionSemantics, enc_name: str = ''
    ) -> str:
        """Generate execute() body from instruction semantics."""
        dst_operands = [op for op in inst.operands if not op.is_input]
        src_operands = [op for op in inst.operands if op.is_input]
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
            and sem.semantic_class == 'vector_swap'
        ):
            src_operands = dst_operands[1:]
            dst_operands = dst_operands[:1]
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
                'scalar_bfe',
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
                is_gfx1250_true16_mov = (
                    self.isa_spec.arch_name.lower() == 'gfx1250'
                    and inst.name == 'V_MOV_B16'
                    and cls == 'vector_mov'
                    and dtype in ('b16', 'u16')
                )
                is_float_op = dtype in ('f16', 'f32', 'f64', 'bf16')
                if is_vop3 and is_float_op and not is_gfx1250_true16_mov:
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
                lctx = LoweringContext(exec_model=sema_block.pragma, operand_map=omap)
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
                is_gfx1250 = self.isa_spec.arch_name.lower() == 'gfx1250'
                has_true16_vop3_src = any(
                    opnd.is_input and opnd.size == 16 for opnd in src_operands
                )
                has_true16_vop3_dst = any(
                    opnd.is_output and opnd.size == 16 for opnd in dst_operands
                )
                is_true16_vop3 = (
                    is_gfx1250
                    and is_vop3
                    and dst_operands
                    and (has_true16_vop3_src or has_true16_vop3_dst)
                )
                if is_true16_vop3:
                    for src_idx, opnd in enumerate(src_operands):
                        if opnd.is_input and opnd.size == 16:
                            lctx.true16_src_selects[src_idx] = (
                                f'inst_.opsel & 0x{1 << src_idx:x}u'
                            )
                    if has_true16_vop3_dst:
                        lctx.true16_dst_select = 'inst_.opsel & 0x8u'
                elif (
                    is_gfx1250
                    and inst.name == 'V_MOV_B16'
                    and cls == 'vector_mov'
                    and dtype in ('b16', 'u16')
                ):
                    enc_upper = enc_name.upper()
                    if enc_upper == 'ENC_VOP1':
                        lctx.true16_dst_select = 'inst_.vdst & 0x80u'
                        lctx.true16_dst_reg = 'inst_.vdst & 0x7fu'
                        lctx.true16_src_select = (
                            '(inst_.src0 >= 256 && inst_.src0 <= 511) '
                            '? ((inst_.src0 - 256) & 0x80u) : 0u'
                        )
                        lctx.true16_src_raw = (
                            '((inst_.src0 >= 256 && inst_.src0 <= 511) '
                            '? wf.cu().read_vgpr(wf.vgpr_alloc().base + '
                            '((inst_.src0 - 256) & 0x7fu), lane) '
                            ': src0.read_lane(wf, lane))'
                        )
                if cls == 'vector_cndmask' and is_vop3 and len(src_ops) >= 3:
                    lctx.vcc_read = f'{src_ops[2]}.read_scalar64(wf)'
                if cls == 'vector_add_co':
                    if is_vop3 and len(src_ops) >= 3:
                        lctx.vcc_read = f'{src_ops[2]}.read_scalar64(wf)'
                    lctx.vcc_dst = dst_ops[1] if len(dst_ops) > 1 else '__vcc__'
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
            enc_field_names=getattr(self, '_current_inst_fields', set()),
            encoding_map=self.isa_spec.encoding_map,
        )
        handler = DISPATCH.get(cls)
        if handler is not None:
            return handler(ctx)

        # Fallback: inline dispatch for classes not yet extracted.
        L = []  # output lines

        if cls == 'true_nop':
            return '  (void)wf;'

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

        if cls == 'waitcnt':
            L.append(
                f'  uint16_t imm = static_cast<uint16_t>({src_ops[0]}.encoding_value_);'
            )
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
            L.append(
                f'  uint16_t cnt = static_cast<uint16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append(f'  wf.set_wait_counter("{op}", cnt);')
            return '\n'.join(L)

        if cls == 'tensor_load_to_lds':
            return '  amdgpu::execute_tensor_load_to_lds(*this, wf);'

        if cls == 'tensor_store_from_lds':
            return '  amdgpu::execute_tensor_store_from_lds(*this, wf);'

        if cls == 'barrier':
            L.append('  wf.set_state(amdgpu::WfState::BARRIER);')
            return '\n'.join(L)

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
                'vccz': 'wf.vcc() == 0',
                'vccnz': 'wf.vcc() != 0',
                'execz': 'wf.exec() == 0',
                'execnz': 'wf.exec() != 0',
            }
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
                f'  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));'
            )
            return '\n'.join(L)

        if cls == 'scalar_cmovk':
            L.append(
                f'  if (wf.read_scc()) {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_))));'
            )
            return '\n'.join(L)

        if cls == 'scalar_addk':
            L.append(f'  uint32_t s0 = {dst_ops[0]}.read_scalar(wf);')
            L.append(
                f'  uint32_t imm = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({src_ops[0]}.encoding_value_)));'
            )
            L.append(
                '  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(imm);'
            )
            L.append('  uint32_t result = static_cast<uint32_t>(wide);')
            L.append(f'  {dst_ops[0]}.write_scalar(wf, result);')
            L.append('  wf.write_scc(wide > 0xFFFFFFFFu);')
            return '\n'.join(L)

        if cls == 'scalar_mulk':
            L.append(
                f'  int32_t s0 = static_cast<int32_t>({dst_ops[0]}.read_scalar(wf));'
            )
            L.append(
                f'  int32_t imm = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append(
                f'  {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(s0 * imm));'
            )
            return '\n'.join(L)

        if cls == 'scalar_wrexec':
            if dtype == 'b32':
                L.append(f'  uint64_t src = {src_ops[0]}.read_scalar(wf);')
                mask_expr = ' & 0xffffffffULL'
            else:
                L.append(f'  uint64_t src = {src_ops[0]}.read_scalar64(wf);')
                mask_expr = ''
            if op in ('andn1', 'and_not1'):
                # EXEC = SRC & ~EXEC
                L.append(f'  wf.set_exec((src & ~wf.exec()){mask_expr});')
            elif op in ('andn2', 'and_not0'):
                # EXEC = EXEC & ~SRC
                L.append(f'  wf.set_exec((wf.exec() & ~src){mask_expr});')
            else:
                L.append(f'  wf.set_exec(src{mask_expr}); // TODO: {op}')
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

        if cls == 'scalar_bitreplicate':
            L.append(f'  uint32_t val = {src_ops[0]}.read_scalar(wf);')
            L.append('  uint64_t result = 0;')
            L.append('  for (uint32_t i = 0; i < 32; ++i) {')
            L.append('    uint64_t bit = (static_cast<uint64_t>(val) >> i) & 1ULL;')
            L.append('    result |= bit << (2 * i);')
            L.append('    result |= bit << (2 * i + 1);')
            L.append('  }')
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, result);')
            return '\n'.join(L)

        if cls == 'scalar_addpc':
            L.append(
                f'  int64_t offset = static_cast<int64_t>({src_ops[0]}.read_scalar64(wf));'
            )
            L.append(
                '  wf.pc = static_cast<uint64_t>(static_cast<int64_t>(wf.pc) + offset);'
            )
            return '\n'.join(L)

        if cls == 'scalar_shader_cycles':
            L.append('  auto *engine = wf.cu().engine();')
            L.append('  uint64_t cycles = engine ? engine->global_time() : 0;')
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, cycles);')
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
            L.append(f'    {dst_ops[0]}.write_scalar64(wf, value);')
            L.append('  else')
            L.append(
                f'    {dst_ops[0]}.write_scalar(wf, static_cast<uint32_t>(value));'
            )
            return '\n'.join(L)

        if cls == 'scalar_barrier_state':
            L.append(f'  {dst_ops[0]}.write_scalar(wf, 0);')
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
                    f'    {dst_ops[0]}.write_scalar64(wf, indexed_src.read_scalar64(wf));'
                )
                L.append('  } else {')
                L.append(
                    f'    {dst_ops[0]}.write_scalar(wf, indexed_src.read_scalar(wf));'
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
                    f'    indexed_dst.write_scalar64(wf, {src_ops[0]}.read_scalar64(wf));'
                )
                L.append('  } else {')
                L.append(
                    f'    indexed_dst.write_scalar(wf, {src_ops[0]}.read_scalar(wf));'
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
                L.append('  indexed_dst.write_scalar(wf, indexed_src.read_scalar(wf));')
                return '\n'.join(L)

        if cls == 'scalar_call':
            # S_CALL_B64: dst = PC of next instruction (return address), then branch.
            L.append(f'  {dst_ops[0]}.write_scalar64(wf, wf.pc + size_);')
            L.append(
                f'  int16_t offset = static_cast<int16_t>({src_ops[0]}.encoding_value_);'
            )
            L.append('  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;')
            return '\n'.join(L)

        if cls == 'scalar_getreg':
            mode_id = profile.hwreg_mode_id
            status_id = profile.hwreg_status_id
            ib_sts2_id = profile.hwreg_ib_sts2_id
            L.append(f'  uint16_t hwreg = {src_ops[0]}.encoding_value_;')
            L.append('  uint32_t reg_id = hwreg & 0x3Fu;')
            L.append('  uint32_t offset = (hwreg >> 6) & 0x1Fu;')
            L.append('  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;')
            L.append('  uint32_t reg_val = 0;')
            if profile.use_hwreg_helpers:
                L.append('  if (!read_hwreg(wf, reg_id, reg_val))')
                L.append(
                    '    util::Logger::warn("s_getreg_b32: unhandled hwreg id=", reg_id);'
                )
            else:
                L.append('  switch (reg_id) {')
                if mode_id is not None:
                    L.append(f'  case {mode_id}: reg_val = wf.mode_raw(); break;')
                L.append(f'  case {status_id}: reg_val = wf.status_raw(); break;')
                L.append(
                    '  case 4: reg_val = static_cast<uint32_t>(wf.cu().id()); break;'
                )
                L.append(
                    '  case 5: reg_val = static_cast<uint32_t>(wf.cu().id() >> 16); break;'
                )
                L.append(
                    '  case 6: reg_val = (wf.sgpr_alloc().count & 0xFFu) | ((wf.sgpr_alloc().base & 0xFFu) << 8); break;'
                )
                L.append(
                    '  case 7: reg_val = (wf.vgpr_alloc().count & 0xFFu) | ((wf.vgpr_alloc().base & 0xFFu) << 8); break;'
                )
                if ib_sts2_id is not None:
                    L.append(f'  case {ib_sts2_id}: reg_val = 0; break;')
                L.append(
                    '  default: util::Logger::warn("s_getreg_b32: unhandled hwreg id=", reg_id); break;'
                )
                L.append('  }')
            L.append('  if (offset + size > 32) size = 32 - offset;')
            L.append(
                '  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);'
            )
            L.append(f'  {dst_ops[0]}.write_scalar(wf, (reg_val >> offset) & mask);')
            return '\n'.join(L)

        if cls == 'scalar_setreg':
            mode_id = profile.hwreg_mode_id
            status_id = profile.hwreg_status_id
            L.append(f'  uint16_t hwreg = {dst_ops[0]}.encoding_value_;')
            L.append('  uint32_t reg_id = hwreg & 0x3Fu;')
            L.append('  uint32_t offset = (hwreg >> 6) & 0x1Fu;')
            L.append('  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;')
            L.append('  if (offset + size > 32) size = 32 - offset;')
            L.append(
                '  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);'
            )
            L.append(f'  uint32_t src = {src_ops[0]}.read_scalar(wf);')
            if profile.use_hwreg_helpers:
                L.append('  if (!write_hwreg(wf, reg_id, offset, mask, src))')
                L.append(
                    '    util::Logger::warn("s_setreg_b32: unhandled hwreg id=", reg_id);'
                )
            else:
                L.append('  switch (reg_id) {')
                if mode_id is not None:
                    L.append(f'  case {mode_id}: {{')
                    L.append('    uint32_t s = wf.mode_raw();')
                    L.append(
                        '    s = (s & ~(mask << offset)) | ((src & mask) << offset);'
                    )
                    L.append('    wf.set_mode_raw(s);')
                    L.append('    break;')
                    L.append('  }')
                L.append(f'  case {status_id}: {{')
                L.append('    uint32_t s = wf.status_raw();')
                L.append('    s = (s & ~(mask << offset)) | ((src & mask) << offset);')
                L.append('    wf.set_status_raw(s);')
                L.append('    break;')
                L.append('  }')
                L.append(
                    '  default: util::Logger::warn("s_setreg_b32: unhandled hwreg id=", reg_id); break;'
                )
                L.append('  }')
            return '\n'.join(L)

        if cls == 'scalar_setreg_imm':
            mode_id = profile.hwreg_mode_id
            status_id = profile.hwreg_status_id
            L.append(f'  uint16_t hwreg = {dst_ops[0]}.encoding_value_;')
            L.append('  uint32_t reg_id = hwreg & 0x3Fu;')
            L.append('  uint32_t offset = (hwreg >> 6) & 0x1Fu;')
            L.append('  uint32_t size = ((hwreg >> 11) & 0x1Fu) + 1;')
            L.append('  if (offset + size > 32) size = 32 - offset;')
            L.append(
                '  uint32_t mask = (size == 32) ? 0xFFFFFFFFu : ((1u << size) - 1u);'
            )
            L.append('  uint32_t src = literal_;')
            if profile.use_hwreg_helpers:
                L.append('  if (!write_hwreg(wf, reg_id, offset, mask, src))')
                L.append(
                    '    util::Logger::warn("s_setreg_imm32_b32: unhandled hwreg id=", reg_id);'
                )
            else:
                L.append('  switch (reg_id) {')
                if mode_id is not None:
                    L.append(f'  case {mode_id}: {{')
                    L.append('    uint32_t s = wf.mode_raw();')
                    L.append(
                        '    s = (s & ~(mask << offset)) | ((src & mask) << offset);'
                    )
                    L.append('    wf.set_mode_raw(s);')
                    L.append('    break;')
                    L.append('  }')
                L.append(f'  case {status_id}: {{')
                L.append('    uint32_t s = wf.status_raw();')
                L.append('    s = (s & ~(mask << offset)) | ((src & mask) << offset);')
                L.append('    wf.set_status_raw(s);')
                L.append('    break;')
                L.append('  }')
                L.append(
                    '  default: util::Logger::warn("s_setreg_imm32_b32: unhandled hwreg id=", reg_id); break;'
                )
                L.append('  }')
            return '\n'.join(L)

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
            L.append(
                f'  {dst_ops[0]}.write_scalar(wf, {src_ops[0]}.read_lane(wf, lane));'
            )
            return '\n'.join(L)

        if cls == 'vector_writelane':
            L.append(f'  uint32_t val = {src_ops[0]}.read_scalar(wf);')
            L.append(f'  uint32_t lane = {src_ops[1]}.read_scalar(wf);')
            L.append(f'  {dst_ops[0]}.write_lane(wf, lane, val);')
            return '\n'.join(L)

        # vector_swap now handled by SemaAST.

        if cls == 'vector_fmamk':
            # D = S0 * K + S2, K is inline constant (second src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = f'{src_ops[1]}.encoding_value_' if len(src_ops) >= 3 else 'simm32_'
            s2_expr = src_ops[2] if len(src_ops) >= 3 else src_ops[1]
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));'
                )
                L.append(
                    f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));'
                )
                L.append(
                    f'    float s2 = util::f16_to_f32(static_cast<uint16_t>({s2_expr}.read_lane(wf, lane)));'
                )
                L.append(
                    f'    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, k, s2)));'
                )
            else:
                L.append(
                    f'    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));'
                )
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(
                    f'    float s2 = std::bit_cast<float>({s2_expr}.read_lane(wf, lane));'
                )
                L.append(
                    f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, k, s2)));'
                )
            L.append('  }')
            return '\n'.join(L)

        if cls == 'vector_fmaak':
            # D = S0 * S1 + K, K is inline constant (third src operand)
            # Some ISA specs omit the simm32 operand; fall back to the
            # simm32_ member populated in the constructor.
            k_expr = f'{src_ops[2]}.encoding_value_' if len(src_ops) >= 3 else 'simm32_'
            L.append('  uint64_t exec = wf.exec();')
            L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
            L.append('    if (!(exec & (1ULL << lane))) continue;')
            if dtype == 'f16':
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src_ops[0]}.read_lane(wf, lane)));'
                )
                L.append(
                    f'    float s1 = util::f16_to_f32(static_cast<uint16_t>({src_ops[1]}.read_lane(wf, lane)));'
                )
                L.append(
                    f'    float k = util::f16_to_f32(static_cast<uint16_t>({k_expr}));'
                )
                L.append(
                    f'    {dst_ops[0]}.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, s1, k)));'
                )
            else:
                L.append(
                    f'    float s0 = std::bit_cast<float>({src_ops[0]}.read_lane(wf, lane));'
                )
                L.append(
                    f'    float s1 = std::bit_cast<float>({src_ops[1]}.read_lane(wf, lane));'
                )
                L.append(f'    float k = std::bit_cast<float>({k_expr});')
                L.append(
                    f'    {dst_ops[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, s1, k)));'
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
            'vector_cvt_scale',
            'vector_cvt_sr_f16_f32',
            'vector_cvt_sr_bf16_f32',
            'vector_pack_b32_f16',
        ):
            if cls == 'vector_cvt_scale':
                return gen_vector_cvt_scale(dst_ops, src_ops, cls, op)
            return gen_vector_cvt_pk(dst_ops, src_ops, cls, op)

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

        if cls == 'gl2_wb':
            return (
                '  if (auto *l2 = wf.cu().l2())\n' '    l2->flush_all(wf.process_id());'
            )

        if cls == 'smem_time':
            return (
                '  static thread_local uint64_t counter = 0;\n'
                '  counter += 100;\n'
                '  uint32_t dst = wf.sgpr_alloc().base + inst_.sdata;\n'
                '  wf.cu().write_sgpr(dst, static_cast<uint32_t>(counter));\n'
                '  wf.cu().write_sgpr(dst + 1, static_cast<uint32_t>(counter >> 32));'
            )

        if cls == 'gl1_wbinv':
            return '  wf.cu().l1_vector().flush_all();'

        if cls == 'flat_atomic':
            return self._gen_flat_atomic(dst_ops, src_ops, sem)

        if cls == 'buffer_atomic':
            return self._gen_buffer_atomic(dst_ops, src_ops, sem)

        if cls == 'ds_atomic':
            gds_guard = ''
            if self._enc_has_field('gds'):
                gds_guard = (
                    '  if (inst_.gds)\n'
                    '    throw util::UnimplementedInst(mnemonic());\n'
                )
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
                L.append(
                    f'    uint32_t src_lane = ((addr_val + offset) / 4) % wf.wf_size();'
                )
                if fetch_invalid:
                    L.append(f'    tmp[i] = src_data[src_lane];')
                else:
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
                L.append(
                    f'    uint32_t dst_lane = ((addr_val + offset) / 4) % wf.wf_size();'
                )
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
            L.append(
                f'      src_lane = (lane & ~0x3) | (src_lane & 0x3);  // stay in quad'
            )
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
        L.append('  auto d = std::make_unique<amdgpu::ScalarMemState>();')
        L.append(f'  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;')
        L.append(f'  d->num_dwords = {nd};')
        L.append(f'  d->elem_size = {elem_size};')
        L.append(f'  d->sign_extend = {str(sem.sign_extend).lower()};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'smem_load')
        L.append(f'  d->mtype = {self._mtype_expr(is_smem=True)};')
        L.append('  d->addr = smem_calculate_address(inst_, wf);')
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
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'smem_store')
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
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base + {i}, lane);')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 4);'
                )
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
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
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);'
        )
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'global_load_async_to_lds')
        L.append('  d->lds_dst = true;')
        L.append('  d->lds_per_lane_addr = true;')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  flat_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t lds_addr_base = {self._vgpr_base_expr('vdst')};")
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    d->per_lane_lds_addr[lane] = wf.lds_base() + cu.read_vgpr(lds_addr_base, lane);'
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
        L.append(f"  uint32_t lds_addr_base = {self._vgpr_base_expr('vsrc')};")
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    uint32_t lds_addr = wf.lds_base() + cu.read_vgpr(lds_addr_base, lane);'
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
        L.append('    d->cu_path = wf.cu().full_path();')
        L.append('    uint64_t base = saddr.read_scalar64(wf);')
        L.append(
            '    int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst_.ioffset << 8) >> 8);'
        )
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
        L.append('    uint32_t val0 = cu.read_vgpr(data_base, lane);')
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
        L.append(f'  d->is_load = ({sc0} != 0);')
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
            L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base + {i}, lane);')
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
        L.append(f'  d->is_load = ({sc0} != 0);')
        L.append(f'  d->atomic_op = {op_enum};')
        self._append_wait_counter_type(L, 'buffer_atomic')
        L.append(f'  d->mtype = {self._mtype_expr()};')
        L.append(f'  d->non_temporal = {nt};')
        L.append('  mubuf_calculate_addresses(inst_, wf, *d);')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('vdata')};")
        stride = data_dwords * 4
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(data_dwords):
            L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base + {i}, lane);')
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

        L = []
        is_cmpswap = sem.operation == 'cmpswap'
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst', role='Dst')};")
        L.append(f'  d->elem_size = {esz};')
        L.append('  d->num_elems = 1;')
        # DS atomics always return the old value (like GLC=1).
        L.append('  d->is_load = true;')
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
            L.append(f'    uint32_t val{i} = cu.read_vgpr({data_source}, lane);')
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &val{i}, 4);'
            )
        L.append('  }')
        # Counter increment handled by MemoryPipeline::issue().
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
            L.append(f'    uint32_t mask{i} = cu.read_vgpr(mask_base + {i}, lane);')
            L.append(
                f'    std::memcpy(&d->store_data[lane * {stride} + {i * 4}], &mask{i}, 4);'
            )
        for i in range(dwords_per_operand):
            L.append(f'    uint32_t src{i} = cu.read_vgpr(src_base + {i}, lane);')
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
        L.append('  d->cu_path = wf.cu().full_path();')
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
            L.append('    uint32_t lo = cu.read_vgpr(data_base, lane);')
            L.append('    uint32_t hi = cu.read_vgpr(data_base + 1, lane);')
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
            L.append('    d->lds_base = wf.m0() + wf.lds_base();')
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
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('vdata')};")
        stride = esz * ne
        L.append(f'  d->store_data.resize(wf.wf_size() * {stride});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        for i in range(ne):
            if esz >= 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base + {i}, lane);')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, {esz});'
                )
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {i * esz}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
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

    def _gen_ds_read_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """ds_read_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f"  d->dst_reg_base = {self._vgpr_base_expr('vdst')};")
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, 'ds_read_addtid')
        L.append('  {')
        L.append('    uint64_t exec = wf.exec();')
        L.append('    d->lane_mask = exec; d->exec_mask = exec;')
        L.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        L.append('    d->cu_path = wf.cu().full_path();')
        L.append(
            '    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;'
        )
        L.append('    uint32_t m0 = wf.m0();')
        L.append('    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;')
        L.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('      if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();'
        )
        L.append('    }')
        L.append('  }')
        L.append('  set_data(std::move(d));')
        return '\n'.join(L)

    def _gen_ds_write_addtid(
        self, dst: list[str], src: list[str], sem: InstructionSemantics
    ) -> str:
        """ds_write_addtid_b32: addr = thread_id * M0[24:16] * 4 + offset."""
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f'  d->elem_size = {sem.elem_size};')
        L.append(f'  d->num_elems = {sem.num_elems};')
        L.append('  d->is_load = false;')
        self._append_wait_counter_type(L, 'ds_write_addtid')
        L.append('  {')
        L.append('    uint64_t exec = wf.exec();')
        L.append('    d->lane_mask = exec; d->exec_mask = exec;')
        L.append('    d->wg_id = wf.wg_id(); d->wf_id = wf.wf_id();')
        L.append('    d->cu_path = wf.cu().full_path();')
        L.append(
            '    uint32_t offset = (static_cast<uint32_t>(inst_.offset1) << 8) | inst_.offset0;'
        )
        L.append('    uint32_t m0 = wf.m0();')
        L.append('    uint32_t ds_stride_bytes = ((m0 >> 16) & 0x1FF) * 4;')
        L.append('    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('      if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '      d->per_lane_addr[lane] = lane * ds_stride_bytes + offset + wf.lds_base();'
        )
        L.append('    }')
        L.append('  }')
        L.append('  auto &cu = wf.cu();')
        L.append('  uint64_t exec = wf.exec();')
        L.append(f"  uint32_t data_base = {self._vgpr_base_expr('data0')};")
        L.append(f'  d->store_data.resize(wf.wf_size() * {sem.elem_size});')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(f'    uint32_t val0 = cu.read_vgpr(data_base, lane);')
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
        # TR_B4=1, TR_B6=2, TR_B8=3, TR_B16=4
        tr_map = {
            'ds_read_tr_b4': (4, 2, 1),  # elem_size=4, num_elems=2, transpose=1
            'ds_read_tr_b6': (4, 3, 2),  # elem_size=4, num_elems=3, transpose=2
            'ds_read_tr_b8': (4, 2, 3),  # elem_size=4, num_elems=2, transpose=3
            'ds_read_tr_b16': (4, 2, 4),  # elem_size=4, num_elems=2, transpose=4
        }
        default_esz, default_ne, default_tr_kind = tr_map.get(
            sem.semantic_class, (4, 2, 4)
        )
        esz = sem.elem_size if sem.elem_size is not None else default_esz
        ne = sem.num_elems if sem.num_elems is not None else default_ne
        tr_kind = getattr(sem, 'transpose_kind', 0) or default_tr_kind
        acc = self._acc_vgpr_expr
        L = []
        L.append(
            '  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);'
        )
        L.append(f'  d->dst_reg_base = wf.vgpr_alloc().base + {acc} + inst_.vdst;')
        L.append(f'  d->elem_size = {esz};')
        L.append(f'  d->num_elems = {ne};')
        L.append('  d->is_load = true;')
        self._append_wait_counter_type(L, sem.semantic_class)
        L.append(f'  d->transpose = {tr_kind};')
        L.append('  ds_calculate_addresses(inst_, wf, *d);')
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
                    f'    uint32_t lo{i} = cu.read_vgpr(data_base + {vgpr_base}, lane);'
                )
                L.append(
                    f'    uint32_t hi{i} = cu.read_vgpr(data_base + {vgpr_base + 1}, lane);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &lo{i}, 4);'
                )
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off + 4}], &hi{i}, 4);'
                )
            elif esz == 4:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base + {i}, lane);')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 4);'
                )
            elif esz == 2:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
                if sem.d16_hi:
                    L.append(f'    val{i} >>= 16;')
                L.append(
                    f'    std::memcpy(&d->store_data[lane * {stride} + {off}], &val{i}, 2);'
                )
            elif esz == 1:
                L.append(f'    uint32_t val{i} = cu.read_vgpr(data_base, lane);')
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
        L.append('    uint32_t base = cu.read_vgpr(addr_base, lane);')
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
        L.append('    uint32_t base = cu.read_vgpr(addr_base, lane);')
        L.append(
            f'    d->per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset0) * {stride_scale} + wf.lds_base();'
        )
        L.append(
            f'    d->ds2_per_lane_addr[lane] = base + static_cast<uint32_t>(inst_.offset1) * {stride_scale} + wf.lds_base();'
        )
        # Pack data0 into store_data
        for i in range(dwords_per_access):
            L.append(f'    uint32_t v0_{i} = cu.read_vgpr(data0_base + {i}, lane);')
            L.append(
                f'    std::memcpy(&d->store_data[lane * {esz} + {i * 4}], &v0_{i}, 4);'
            )
        # Pack data1 into ds2_store_data
        for i in range(dwords_per_access):
            L.append(f'    uint32_t v1_{i} = cu.read_vgpr(data1_base + {i}, lane);')
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
            # S_SETREG_IMM32_B32 reads the instruction-local literal_ member.
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
        }
    )
    _NON_SHAREABLE_MNEMONICS = frozenset(
        {
            # FMAAK/FMAMK carry a literal addend/multiplicand through
            # ISA-specific constructor operands.
            's_fmaak_f32',
            's_fmamk_f32',
        }
    )

    def _requires_arch_local_execute(
        self, inst: Instruction | None, enc_name: str | None = None
    ) -> bool:
        if inst is None or self.isa_spec.arch_name.lower() != 'gfx1250':
            return False
        enc_upper = (enc_name or inst.enc_name).upper()
        if (
            enc_upper == 'ENC_VOP1'
            and inst.name == 'V_MOV_B16'
            and any(op.size == 16 for op in inst.operands)
        ):
            return True
        if enc_upper == 'ENC_VOP3' and any(
            (op.is_input or op.is_output) and op.size == 16 for op in inst.operands
        ):
            return True
        return False

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
        if mnemonic in self._NON_SHAREABLE_MNEMONICS:
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
                    inst_sem = (
                        self.semantics.instructions.get(inst.name)
                        if self.semantics
                        else None
                    )
                    inst = self._with_scalar_literal_fma_operand(inst)
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
                    public_members = [cgen.Line('public:')]
                    private_members = []
                    opnd_ctor_init = []
                    opnd_body = []
                    vgpr_msb_role_body = []
                    src_idx = 0
                    dst_idx = 0
                    vgpr_msb_src_role_idx = 0
                    reads_dst = self._dst_is_also_source(inst)
                    # These gfx1250-only WMMA source-format fields derive the
                    # src0/src1 operand sizes from the instruction shape.
                    gfx1250_f8f6f4_shape = self._gfx1250_f8f6f4_wmma_shape(inst)
                    gfx1250_swmmac_has_modifiers = self._gfx1250_swmmac_has_modifiers(
                        inst
                    )
                    operand_size_exprs: dict[str, str] = {}
                    for opnd in inst.operands:
                        opnd_size_expr = self._gfx1250_matrix_fmt_operand_size_expr(
                            gfx1250_f8f6f4_shape, opnd.name
                        )
                        if opnd_size_expr is None:
                            opnd_size_expr = str(opnd.size)
                        operand_size_exprs[opnd.name] = opnd_size_expr
                        if opnd.is_input:
                            opnd_body.append(
                                f'src_operands_[{src_idx}] = &{opnd.name};'
                            )
                            src_idx += 1
                        elif (
                            reads_dst
                            and opnd.is_output
                            and opnd.name in ('vdst', 'sdst')
                        ):
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
                        if self.isa_spec.profile.uses_vgpr_msb_indexing:
                            _role = None
                            if self._operand_can_use_vgpr_msb(opnd):
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
                            opr_type = opnd.operand_type
                            if inst_sem and inst_sem.accvgpr_srcs and opnd.is_input:
                                opr_type = 'OPR_SRC_VGPR_OR_ACCVGPR'
                            packed_16bit_source_arg = (
                                ', true'
                                if self._operand_uses_packed_16bit_source(
                                    enc.enc_name, opnd
                                )
                                else ''
                            )
                            operand_value = f'reinterpret_cast<const OpEncoding*>(inst)->{opnd.name}'
                            if packed_16bit_source_arg:
                                operand_value = (
                                    f'static_cast<unsigned short>({operand_value})'
                                )
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd_size_expr}, '
                                f'OperandType::{opr_type}, '
                                f'{operand_value}{packed_16bit_source_arg})'
                            )
                        else:
                            opnd_ctor_init.append(
                                f'{opnd.name}({opnd.size}, '
                                f'OperandType::{opnd.operand_type}, 0)'
                            )
                    # For flat encodings with a seg field, add saddr as an
                    # optional operand. Declared and initialized LAST among
                    # operands to avoid reorder warnings with data/addr.
                    _has_flat_saddr = self.isa_spec.profile.mnemonic_rule(
                        enc.enc_name
                    ).use_flat_mnemonic
                    if _has_flat_saddr:
                        private_members.append(cgen.Statement('Operand saddr'))
                        opnd_ctor_init.append('saddr(0, OperandType::OPR_SREG, 0)')

                    class_ctor_decl = cgen.FunctionDeclaration(
                        cgen.Value('', inst.fmt_name),
                        [cgen.Value('const MachineInst *', 'inst')],
                    )
                    public_members.append(class_ctor_decl)
                    public_members.append(
                        cgen.Statement('void execute_impl(amdgpu::Wavefront &wf)')
                    )
                    if gfx1250_f8f6f4_shape is not None or gfx1250_swmmac_has_modifiers:
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
                    if (
                        inst_sem
                        and inst_sem.semantic_class in ('branch', 'cbranch')
                        and label_operand
                    ):
                        public_members.append(
                            cgen.Statement(
                                'std::optional<int64_t> branch_offset_bytes() const override'
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
                            'ds_read',
                            'ds_read2',
                            'ds_write',
                            'ds_write2',
                            'ds_atomic',
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
                    ctor_body_parts.append(f'num_src_ = {src_idx};')
                    ctor_body_parts.append(f'num_dst_ = {dst_idx};')

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
                            ctor_body_parts.append('  if (inst_.saddr != 0x7F) {')
                            ctor_body_parts.append(
                                '    saddr = Operand(32, OperandType::OPR_SREG, '
                                'inst_.saddr);'
                            )
                            ctor_body_parts.append(
                                '    src_operands_[num_src_++] = &saddr;'
                            )
                            ctor_body_parts.append('  }')
                            ctor_body_parts.append(
                                '} else if (inst_.seg == 2 && inst_.saddr != 0x7F) {'
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
                    # extended instruction encoding. A selector value of 254
                    # carries a 64-bit literal in the next two DWORDs.
                    _lit_info = self._literal_encoding_info(enc, inst_enc_obj, inst)
                    _supports_simm64_literals = self._supports_simm64_literal_operands()
                    if _lit_info and self._has_machine_inst_struct(_lit_info[0]):
                        _lit_struct, _lit_fields = _lit_info
                        for opnd in inst.operands:
                            if (
                                opnd.name in _lit_fields
                                and opnd.name in enc_field_names
                            ):
                                ctor_body_parts.append(
                                    f'if (reinterpret_cast<const OpEncoding*>(inst)->{opnd.name} == 255) '
                                    f'{opnd.name} = Operand({operand_size_exprs[opnd.name]}, OperandType::OPR_SIMM32, '
                                    f'static_cast<int>(reinterpret_cast<const {_lit_struct}*>(inst)->simm32));'
                                )
                                if _supports_simm64_literals:
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
                    }
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
                                    if enc.enc_name.upper() == 'ENC_VOPC':
                                        _sdwa_struct = 'VopcVopSdwaSdstEncMachineInst'
                                    else:
                                        _sdwa_struct = f'{_enc_base}VopSdwaMachineInst'
                                    _sdwa_s1_code = ''
                                    if enc.enc_name.upper() in ('ENC_VOP2', 'ENC_VOPC'):
                                        _sdwa_s1_code = (
                                            f' if (sw->s1)'
                                            f'   vsrc1 = Operand({opnd.size}, OperandType::OPR_SRC,'
                                            f'     reinterpret_cast<const OpEncoding*>(inst)->vsrc1);'
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
                                        + (
                                            f' sdwa_sdst_ = sw->sdst;'
                                            f' sdwa_sd_ = sw->sd;'
                                            if enc.enc_name.upper() == 'ENC_VOPC'
                                            else f' sdwa_dst_sel_ = sw->dst_sel;'
                                            f' sdwa_dst_unused_ = sw->dst_unused;'
                                            f' sdwa_clamp_ = sw->clamp;'
                                        )
                                        + f'{_sdwa_s1_code}}}'
                                    )

                    # Implied literal fixup: FMAMK/FMAAK always carry an
                    # inline 32-bit literal even when the ISA spec omits the
                    # simm32 operand. Add a simm32_ member to hold it.
                    _FMAMK_FMAAK = frozenset(
                        {
                            'vector_fmamk',
                            'vector_fmaak',
                        }
                    )
                    _has_simm32 = any(op.name == 'simm32' for op in inst.operands)
                    if (
                        _mem_sem
                        and _mem_sem.semantic_class in _FMAMK_FMAAK
                        and not _has_simm32
                        and not self._has_inline_literal_operand(inst)
                        and _lit_info
                    ):
                        _lit_struct = _lit_info[0]
                        private_members.append(cgen.Statement('uint32_t simm32_'))
                        opnd_ctor_init.append('simm32_(0)')
                        init_list_parts.append('simm32_(0)')
                        init_list = ', '.join(init_list_parts)
                        ctor_body_parts.append(
                            f'simm32_ = reinterpret_cast<const '
                            f'{_lit_struct}*>(inst)->simm32;'
                        )

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

                    # Per-instruction size overrides (e.g., VOP3PX2 128-bit
                    # instructions decoded under 64-bit VOP3P_MFMA).
                    _size_overrides = self.isa_spec.profile.inst_size_overrides
                    if inst.name in _size_overrides:
                        ctor_body_parts.append(f'size_ = {_size_overrides[inst.name]};')

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
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
                            _src0_name = next(
                                (o.name for o in inst.operands if o.is_input), None
                            )
                            _src_inputs = [o.name for o in inst.operands if o.is_input]
                            _src1_name = (
                                _src_inputs[1] if len(_src_inputs) > 1 else None
                            )
                            _is_vopc = enc.enc_name.upper() == 'ENC_VOPC'
                            _dpp_preamble = ''
                            if _is_vopc:
                                _dpp_preamble += (
                                    '  uint64_t dpp_old_vcc_ = wf.vcc();\n'
                                    '  uint64_t dpp_write_mask_ = ~0ULL;\n'
                                    '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                    '    dpp_write_mask_ = 0;\n'
                                    '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n'
                                    '      uint32_t row = ln / 16;\n'
                                    '      uint32_t bank = (ln % 16) / 4;\n'
                                    '      if ((dpp_row_mask_ & (1u << row)) &&\n'
                                    '          (dpp_bank_mask_ & (1u << bank)))\n'
                                    '        dpp_write_mask_ |= (1ULL << ln);\n'
                                    '    }\n'
                                    '  }\n'
                                )
                            elif not _is_vopc:
                                _dpp_preamble += (
                                    '  uint32_t sdwa_old_dst_[64] = {};\n'
                                    '  if (sdwa_dst_sel_ != amdgpu::sdwa::DWORD ||\n'
                                    '      inst_.src0 == amdgpu::SRC_DPP) {\n'
                                    '    uint32_t vb = wf.vgpr_alloc().base;\n'
                                    '    uint64_t ex = wf.exec();\n'
                                    '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln)\n'
                                    '      if (ex & (1ULL << ln))\n'
                                    '        sdwa_old_dst_[ln] = wf.cu().read_vgpr(vb + inst_.vdst, ln);\n'
                                    '  }\n'
                                )
                            _dpp_preamble += (
                                '  if (inst_.src0 == amdgpu::SRC_DPP)\n'
                                '    amdgpu::dpp::apply_dpp(src_operands_[0], dpp_ctrl_,\n'
                                '        dpp_row_mask_, dpp_bank_mask_, dpp_bound_ctrl_,\n'
                                '        dpp_src0_, wf);\n'
                                '  if (inst_.src0 == amdgpu::SRC_SDWA) {\n'
                                '    auto &cu = wf.cu();\n'
                                '    uint32_t ws = wf.wf_size();\n'
                                '    if (sdwa_src0_sel_ != amdgpu::sdwa::DWORD) {\n'
                                '      uint32_t vb = wf.vgpr_alloc().base + src_operands_[0]->encoding_value_;\n'
                                '      uint32_t result[64];\n'
                                '      for (uint32_t i = 0; i < ws; ++i)\n'
                                '        result[i] = amdgpu::sdwa::sdwa_src_select(\n'
                                '            cu.read_vgpr(vb, i), sdwa_src0_sel_, sdwa_src0_sext_);\n'
                                '      if (sdwa_src0_abs_ || sdwa_src0_neg_) {\n'
                                '        for (uint32_t i = 0; i < ws; ++i) {\n'
                                '          float sv = std::bit_cast<float>(result[i]);\n'
                                '          if (sdwa_src0_abs_) sv = std::fabs(sv);\n'
                                '          if (sdwa_src0_neg_) sv = -sv;\n'
                                '          result[i] = std::bit_cast<uint32_t>(sv);\n'
                                '        }\n'
                                '      }\n'
                                '      dpp_src0_ = std::make_unique<DppOperand>(\n'
                                '          *src_operands_[0], result, static_cast<int>(ws));\n'
                                '      src_operands_[0] = dpp_src0_.get();\n'
                                '    }\n'
                                '    if (sdwa_src1_sel_ != amdgpu::sdwa::DWORD && num_src_ > 1) {\n'
                                '      uint32_t vb = wf.vgpr_alloc().base + src_operands_[1]->encoding_value_;\n'
                                '      uint32_t result1[64];\n'
                                '      for (uint32_t i = 0; i < ws; ++i)\n'
                                '        result1[i] = amdgpu::sdwa::sdwa_src_select(\n'
                                '            cu.read_vgpr(vb, i), sdwa_src1_sel_, sdwa_src1_sext_);\n'
                                '      if (sdwa_src1_abs_ || sdwa_src1_neg_) {\n'
                                '        for (uint32_t i = 0; i < ws; ++i) {\n'
                                '          float sv = std::bit_cast<float>(result1[i]);\n'
                                '          if (sdwa_src1_abs_) sv = std::fabs(sv);\n'
                                '          if (sdwa_src1_neg_) sv = -sv;\n'
                                '          result1[i] = std::bit_cast<uint32_t>(sv);\n'
                                '        }\n'
                                '      }\n'
                                '      dpp_src1_ = std::make_unique<DppOperand>(\n'
                                '          *src_operands_[1], result1, static_cast<int>(ws));\n'
                                '      src_operands_[1] = dpp_src1_.get();\n'
                                '    }\n'
                                '  }\n'
                                + (
                                    f'  if (dpp_src0_) {_src0_name}.set_delegate(dpp_src0_.get());\n'
                                    if _src0_name
                                    else ''
                                )
                                + (
                                    f'  if (dpp_src1_) {_src1_name}.set_delegate(dpp_src1_.get());\n'
                                    if _src1_name
                                    else ''
                                )
                            )
                        # SDWA postamble: apply dst_sel merge and float clamp after ALU.
                        _sdwa_postamble = ''
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2'):
                            is_float_op = sem and sem.data_type in ('f16', 'f32', 'f64')
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
                        if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
                            if _is_vopc:
                                _dpp_cleanup += (
                                    '  if (inst_.src0 == amdgpu::SRC_DPP && dpp_write_mask_ != ~0ULL) {\n'
                                    '    uint64_t new_vcc = wf.vcc();\n'
                                    '    uint64_t merged = (new_vcc & dpp_write_mask_) | (dpp_old_vcc_ & ~dpp_write_mask_);\n'
                                    '    wf.set_vcc(merged);\n'
                                    '  }\n'
                                    '  if (inst_.src0 == amdgpu::SRC_SDWA && sdwa_sd_) {\n'
                                    '    uint64_t cmp_result = wf.vcc();\n'
                                    '    uint32_t sb = wf.sgpr_alloc().base;\n'
                                    '    wf.cu().write_sgpr(sb + sdwa_sdst_, static_cast<uint32_t>(cmp_result));\n'
                                    '    wf.cu().write_sgpr(sb + sdwa_sdst_ + 1, static_cast<uint32_t>(cmp_result >> 32));\n'
                                    '    wf.set_vcc(dpp_old_vcc_);\n'
                                    '  }\n'
                                )
                            else:
                                _dpp_cleanup += (
                                    '  if (inst_.src0 == amdgpu::SRC_DPP) {\n'
                                    '    uint64_t dpp_write_mask = 0;\n'
                                    '    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n'
                                    '      uint32_t row = ln / 16;\n'
                                    '      uint32_t bank = (ln % 16) / 4;\n'
                                    '      if ((dpp_row_mask_ & (1u << row)) &&\n'
                                    '          (dpp_bank_mask_ & (1u << bank)))\n'
                                    '        dpp_write_mask |= (1ULL << ln);\n'
                                    '    }\n'
                                    '    if (dpp_write_mask != ~0ULL) {\n'
                                    '      uint64_t ex = wf.exec();\n'
                                    '      uint32_t vb = wf.vgpr_alloc().base;\n'
                                    '      for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {\n'
                                    '        if ((ex & (1ULL << ln)) && !(dpp_write_mask & (1ULL << ln)))\n'
                                    '          wf.cu().write_vgpr(vb + inst_.vdst, ln,\n'
                                    '              sdwa_old_dst_[ln]);\n'
                                    '      }\n'
                                    '    }\n'
                                    '  }\n'
                                )
                            if _src0_name:
                                _dpp_cleanup += f'  {_src0_name}.clear_delegate();\n'
                            if _src1_name:
                                _dpp_cleanup += f'  {_src1_name}.clear_delegate();\n'
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
                        from amdisa.codegen.execute.simd_codegen import (
                            simd_probe_arch_portable as _simd_probe_arch_portable,
                        )

                        _enc_key_for_probe = enc.enc_name.lower().replace('enc_', '')
                        _portable_probe = _simd_probe_arch_portable(
                            f'{inst.mnemonic}_{_enc_key_for_probe}',
                            self.isa_spec.profile.vop3p_opsel_fields,
                        )
                        if body_throws:
                            exec_impl = cgen.Line(
                                f'void {inst.fmt_name}::execute_impl'
                                f'(amdgpu::Wavefront &wf) {{ (void)wf; throw util::UnimplementedInst(mnemonic()); }}'
                            )
                        elif can_share or _portable_probe:
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
                                )
                            elif existing[2] != body:
                                _exist_inst, _, _exist_body, _ = existing
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
                    if gfx1250_f8f6f4_shape is not None:
                        class_func_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::build_modifiers(std::string &out) const {{\n'
                                f'  out += " matrix_a_fmt:";\n'
                                f'  out += gfx1250_matrix_fmt_name(inst_.opsel);\n'
                                f'  out += " matrix_b_fmt:";\n'
                                f'  out += gfx1250_matrix_fmt_name((inst_.pad_14 << 2) | inst_.opsel_hi);\n'
                                f'}}'
                            )
                        )
                    elif gfx1250_swmmac_has_modifiers:
                        class_func_impls.append(
                            cgen.Line(
                                f'void {inst.fmt_name}::build_modifiers(std::string &out) const {{\n'
                                f'  if (inst_.opsel & 0x1)\n'
                                f'    out += " index_key:1";\n'
                                f'  if (inst_.opsel & 0x4)\n'
                                f'    out += " matrix_a_reuse";\n'
                                f'  if (inst_.pad_14)\n'
                                f'    out += " matrix_b_reuse";\n'
                                f'}}'
                            )
                        )
                    if (
                        inst_sem
                        and inst_sem.semantic_class in ('branch', 'cbranch')
                        and label_operand
                    ):
                        class_func_impls.append(
                            cgen.Line(
                                f'std::optional<int64_t> '
                                f'{inst.fmt_name}::branch_offset_bytes() const {{\n'
                                f'  // AMDGPU direct branch labels are signed '
                                f'instruction-count deltas.\n'
                                f'  return static_cast<int64_t>('
                                f'static_cast<int16_t>({label_operand}.encoding_value_)) * 4;\n'
                                f'}}'
                            )
                        )
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
                                f'rocjitsu/isa/arch/amdgpu/{self.isa_spec.arch_name}/addr_calc.h',
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
                            f'rocjitsu/isa/arch/amdgpu/'
                            f'{self.isa_spec.arch_name}/mma_exec.h',
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
                # VOP1/VOP2 need DPP header for apply_dpp() in execute_impl.
                if enc.enc_name.upper() in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC'):
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
                has_getreg = any(
                    self.semantics
                    and (s := self.semantics.instructions.get(i.name))
                    and s.semantic_class in ('scalar_getreg', 'scalar_setreg')
                    for i in all_insts
                )
                if has_getreg and not is_mem_enc:
                    cpp_includes.extend(
                        [
                            ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                            ('util/log.h', False),
                        ]
                    )

                # Include the unified shared execute template header when
                # any instruction in this encoding delegates to a template.
                if self.shared_plan is not None:
                    from amdisa.codegen.execute.simd_codegen import (
                        simd_probe_arch_portable as _simd_probe_arch_portable,
                    )

                    enc_key = enc.enc_name.lower().replace('enc_', '')

                    def _delegates_to_shared(i: Instruction) -> bool:
                        if (
                            not self.semantics
                            or i.name not in self.semantics.instructions
                        ):
                            return False
                        return self._can_share_execute(
                            i.mnemonic, i, enc.enc_name
                        ) or _simd_probe_arch_portable(
                            f'{i.mnemonic}_{enc_key}',
                            self.isa_spec.profile.vop3p_opsel_fields,
                        )

                    has_shared = any(_delegates_to_shared(i) for i in all_insts)
                    if has_shared:
                        cpp_includes.append(
                            (
                                'rocjitsu/isa/arch/amdgpu/shared/execute_shared.h',
                                False,
                            )
                        )

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
                if (
                    self._supports_gfx1250_scaled_wmma_vop3px2()
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    h_includes.append(('array', True))
                    cpp_includes.append(('array', True))
                    inst_classes.append(
                        cgen.Line(self._emit_gfx1250_scaled_wmma_vop3px2_class())
                    )
                    class_func_impls.append(
                        cgen.Line(self._emit_gfx1250_scaled_wmma_vop3px2_impls())
                    )

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
                # No local f16 helpers needed - using util::f16_to_f32 etc.
                # from data_types.h (included via cpp_includes when has_sem).

                if (
                    self.isa_spec.arch_name.lower() == 'gfx1250'
                    and enc.enc_name.upper() == 'ENC_VOP3P'
                ):
                    class_func_impls.insert(
                        0, cgen.Line(self._emit_gfx1250_matrix_fmt_helpers())
                    )

                if has_getreg and profile.use_hwreg_helpers and not is_mem_enc:
                    class_func_impls.insert(0, cgen.Line(self._emit_hwreg_helpers()))

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
                self._write_inst_impl_files(
                    enc.enc_name,
                    f'{enc.fmt_enc_name.lower()}',
                    cpp_includes,
                    class_func_impls,
                )

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
        if self._supports_generated_vopd():
            insts_h_lines.append(f'#include "{inc_base}/vopd.h"\n')
        insts_h_lines.append(f'\n#endif // {guard}\n')
        insts_h_path = os.path.join(self.out_path, arch, 'insts.h')
        with open(insts_h_path, 'w') as f:
            f.write(''.join(insts_h_lines))

        # Shared execute templates are written by _run_multi after all ISAs
        # are processed, using the accumulated _shared_execute_bodies dict.
        # Individual ISA codegens just collect; they don't write.

    def _write_inst_impl_files(
        self,
        enc_name: str,
        base_name: str,
        cpp_includes: list[tuple[str, bool]],
        class_func_impls: list[object],
    ) -> None:
        """Write implementation files for one encoding, splitting when needed.

        Some generated encodings, notably gfx1250 VOP3/VOPC, are large enough
        to trip the repository's added-file size hook. Profiles can set a byte
        limit in ``source_split_max_bytes``; this emits ``<base>_partN.cpp``
        chunks under that limit and removes stale split/unsplit files when the
        split decision changes.
        """
        max_bytes = self.isa_spec.profile.source_split_max_bytes.get(enc_name.upper())
        if not max_bytes:
            import os

            arch_dir = os.path.join(self.out_path, self.isa_spec.arch_name)
            if os.path.isdir(arch_dir):
                for filename in os.listdir(arch_dir):
                    if re.fullmatch(rf'{re.escape(base_name)}_part\d+\.cpp', filename):
                        os.remove(os.path.join(arch_dir, filename))
            CppFile(
                base_name,
                self.out_path,
                False,
                cpp_includes,
                [],
                class_func_impls,
                self.isa_spec.arch_name,
            ).gen_code()
            return

        import os

        arch_dir = os.path.join(self.out_path, self.isa_spec.arch_name)
        if os.path.isdir(arch_dir):
            for filename in os.listdir(arch_dir):
                if re.fullmatch(rf'{re.escape(base_name)}_part\d+\.cpp', filename):
                    os.remove(os.path.join(arch_dir, filename))

        chunks: list[list[object]] = []
        current_chunk: list[object] = []
        current_size = 0

        # CppFile adds a short prologue, include block, and namespace wrapper.
        # Leave margin under the profile limit so formatting and include growth
        # do not push a chunk over the added-file size hook.
        chunk_overhead = 16 * 1024
        for impl in class_func_impls:
            impl_size = len(f'{impl}\n\n'.encode())
            if current_chunk and current_size + impl_size + chunk_overhead > max_bytes:
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
            file_name = base_name if len(chunks) == 1 else f'{base_name}_part{idx + 1}'
            CppFile(
                file_name,
                self.out_path,
                False,
                cpp_includes,
                [],
                chunk,
                self.isa_spec.arch_name,
            ).gen_code()

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
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)inst_\.', 'inst.inst_.', prefixed_body
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
            prefixed_body = _re.sub(
                r'(?<!\.)(?<!\w)simm32_(?!\w)', 'inst.simm32_', prefixed_body
            )
            prefixed_body = _re.sub(
                r'\s*\(void\)wf;\s*(?://[^\n]*)?\n?', '\n', prefixed_body
            )
            entries.append((mnemonic, prefixed_body, sem.semantic_class))

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
            '#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"',
            '#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"',
            *simd_extra_includes(),
            '#include "util/data_types.h"',
            '#include "util/except.h"',
            '#include "util/log.h"',
            '#include <algorithm>',
            '#include <bit>',
            '#include <cmath>',
            '#include <functional>',
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
            probe = simd_probe_line(mnemonic)
            if probe is not None:
                lines.append(probe)
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
            case _:
                return None

    def gen_operand(self) -> None:
        """Generate the ISA-specific Operand class with name resolution."""
        arch = self.isa_spec.arch_name
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
        packed_16bit_name_check = ''
        if uses_packed_16bit_sources:
            packed_16bit_name_check = (
                'if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, encoding_value_))\n'
                '  return std::format("v{}.{}", packed->reg, packed->shift ? "h" : "l");\n'
            )
        name_impl = (
            f'std::string Operand::name() const {{\n'
            f'if (has_literal64_)\n'
            f'  return std::format("0x{{:x}}", literal64_value_);\n'
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
                'if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, encoding_value_))\n'
                '  return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(packed->reg), reg_width};\n'
            )
        ref_impl = (
            f'std::optional<RegisterRef> Operand::to_register_ref() const {{\n'
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
            '          bool packed_16bit_source = false);\n'
            '  Operand(int size_bits, OperandType opr_type, unsigned short encoding_value,\n'
            '          bool packed_16bit_source);\n'
            if uses_packed_16bit_sources
            else '  Operand(int size_bits, OperandType opr_type, int encoding_value);\n'
        )
        literal64_decl = '  std::optional<uint64_t> literal64_value() const override;\n'
        simd_decl = ''
        packed_16bit_field = ''
        if uses_packed_16bit_sources:
            simd_decl = (
                '  bool simd_capable() const override;\n'
                '  void read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,\n'
                '                       uint32_t *out) const override;\n'
            )
            packed_16bit_field = '  bool packed_16bit_source_ = false;\n'

        class_def = [
            cgen.Line(
                'class Operand : public AmdgpuIsaOperand<Isa> {\n'
                'public:\n'
                f'{operand_ctor_decl}'
                '  Operand(int size_bits, OperandType opr_type, uint64_t literal64_value, bool is_literal64);\n'
                '  std::string name() const override;\n'
                f'{literal64_decl}'
                '  std::optional<RegisterRef> to_register_ref() const override;\n'
                f'{simd_decl}'
                '  uint32_t read_scalar(const amdgpu::Wavefront &wf) const override;\n'
                '  uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
                '  void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const override;\n'
                '  void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const override;\n'
                '  uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const override;\n'
                '  void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const override;\n'
                '  uint64_t read_scalar64(const amdgpu::Wavefront &wf) const override;\n'
                '  void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const override;\n'
                'private:\n'
                '  uint64_t literal64_value_ = 0;\n'
                '  bool has_literal64_ = false;\n'
                f'{packed_16bit_field}'
                '};'
            )
        ]

        operand_ctor_args = (
            'int size_bits, OperandType opr_type, int encoding_value, bool packed_16bit_source'
            if uses_packed_16bit_sources
            else 'int size_bits, OperandType opr_type, int encoding_value'
        )
        operand_ctor_init = (
            ',\n' '      packed_16bit_source_(packed_16bit_source)'
            if uses_packed_16bit_sources
            else ''
        )
        packed_16bit_ctor_impl = []
        if uses_packed_16bit_sources:
            packed_16bit_ctor_impl.append(
                cgen.Line(
                    'Operand::Operand(int size_bits, OperandType opr_type, unsigned short encoding_value,\n'
                    '                 bool packed_16bit_source)\n'
                    '    : Operand(size_bits, opr_type, static_cast<int>(encoding_value),\n'
                    '              packed_16bit_source) {\n'
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
        class_impl = [
            cgen.Line(
                f'Operand::Operand({operand_ctor_args})\n'
                '    : AmdgpuIsaOperand<Isa>(size_bits, opr_type, encoding_value)'
                f'{operand_ctor_init} {{\n'
                '  is_vgpr_ = is_vgpr_operand_type(opr_type);\n'
                '}'
            ),
            *packed_16bit_ctor_impl,
            cgen.Line(
                'Operand::Operand(int size_bits, OperandType opr_type, uint64_t literal64_value, bool is_literal64)\n'
                '    : AmdgpuIsaOperand<Isa>(size_bits, opr_type, static_cast<int>(literal64_value)),\n'
                '      literal64_value_(literal64_value), has_literal64_(is_literal64) {\n'
                '  is_vgpr_ = is_vgpr_operand_type(opr_type);\n'
                '}'
            ),
            cgen.Line(literal64_impl),
            cgen.Line(name_impl),
            cgen.Line(ref_impl),
        ]

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
                '                                                           int ev) {\n'
                '  if (!packed_16bit_source || size_bits != 16)\n'
                '    return std::nullopt;\n'
                '  if (ev >= 384 && ev <= 511)\n'
                '    return Packed16VgprSource{static_cast<uint32_t>(ev - 384), 16};\n'
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
        for _opt in ('OPR_SIMM4', 'OPR_SIMM8', 'OPR_SIMM64'):
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

        read_lane_lines = [
            'uint32_t Operand::read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const {',
            '  if (delegate()) return delegate()->read_lane(wf, lane);',
            '  int ev = encoding_value_;',
        ]
        if uses_packed_16bit_sources:
            read_lane_lines.extend(
                [
                    '  if (auto packed = packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, ev)) {',
                    '    uint32_t off = packed->reg + (wf.vgpr_msb_for_role(vgpr_msb_role()) << 8);',
                    '    uint32_t raw = wf.cu().read_vgpr(wf.vgpr_alloc().base + off, lane);',
                    '    return (raw >> packed->shift) & 0xffffu;',
                    '  }',
                ]
            )
        read_lane_lines.extend(
            [
                f'  if (auto off = {_resolved_vgpr_read_call})',
                '    return wf.cu().read_vgpr(wf.vgpr_alloc().base + *off, lane);',
                '  if (is_immediate_type(opr_type_))',
                '    return static_cast<uint32_t>(ev);',
                '  return resolve_src_scalar(wf, ev);',
                '}',
            ]
        )
        _read_lane_body = '\n'.join(read_lane_lines)

        _read_lane64_body = (
            'uint64_t Operand::read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const {\n'
            '  if (delegate()) return delegate()->read_lane64(wf, lane);\n'
            '  int ev = encoding_value_;\n'
            f'  if (auto off = {_resolved_vgpr_read_call}) {{\n'
            '    uint32_t idx = wf.vgpr_alloc().base + *off;\n'
            '    uint32_t lo = wf.cu().read_vgpr(idx, lane);\n'
            '    uint32_t hi = wf.cu().read_vgpr(idx + 1, lane);\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            '  if (has_literal64_)\n'
            '    return literal64_value_;\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ev)));\n'
            '  return resolve_src_scalar64(wf, ev);\n'
            '}'
        )

        simd_methods = ''
        if uses_packed_16bit_sources:
            simd_methods = textwrap.dedent('''\
                bool Operand::simd_capable() const {
                  if (delegate())
                    return delegate()->simd_capable();
                  if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, encoding_value_))
                    return false;
                  return AmdgpuIsaOperand<Isa>::simd_capable();
                }

                void Operand::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                                              uint32_t *out) const {
                  if (delegate()) {
                    delegate()->read_lane_chunk(wf, lane_base, count, out);
                    return;
                  }
                  if (packed_16bit_vgpr_source(packed_16bit_source_, size_bits_, encoding_value_)) {
                    for (uint32_t i = 0; i < count; ++i)
                      out[i] = read_lane(wf, lane_base + i);
                    return;
                  }
                  AmdgpuIsaOperand<Isa>::read_lane_chunk(wf, lane_base, count, out);
                }

                ''')

        resolve_code = cgen.Line(
            'namespace {\n'
            '\n'
            'uint32_t resolve_src_scalar(const amdgpu::Wavefront &wf, int ev) {\n'
            '  if (ev == 102)\n'
            '    return static_cast<uint32_t>(wf.scratch_base());\n'
            '  if (ev == 103)\n'
            '    return static_cast<uint32_t>(wf.scratch_base() >> 32);\n'
            '  if (ev <= 105)\n'
            '    return wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            '  if (ev == 106)\n'
            '    return static_cast<uint32_t>(wf.vcc());\n'
            '  if (ev == 107)\n'
            '    return static_cast<uint32_t>(wf.vcc() >> 32);\n'
            '  if (ev >= 108 && ev <= 123)\n'
            '    return wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            + (
                '  if (ev == 124)\n'
                '    return 0u; // NULL\n'
                '  if (ev == 125)\n'
                '    return wf.m0();\n'
                if arch in ('rdna4', 'gfx1250')
                else '  if (ev == 124)\n' '    return wf.m0();\n'
            )
            + '  if (ev == 126)\n'
            '    return static_cast<uint32_t>(wf.exec());\n'
            '  if (ev == 127)\n'
            '    return static_cast<uint32_t>(wf.exec() >> 32);\n'
            '  if (ev >= 128 && ev <= 192)\n'
            '    return static_cast<uint32_t>(ev - 128);\n'
            '  if (ev >= 193 && ev <= 208)\n'
            '    return static_cast<uint32_t>(static_cast<int32_t>(-(ev - 192)));\n'
            '  if (ev == 230)\n'
            '    return static_cast<uint32_t>(wf.scratch_base()); // SRC_FLAT_SCRATCH_BASE_LO\n'
            '  if (ev == 231)\n'
            '    return static_cast<uint32_t>(wf.scratch_base() >> 32); // SRC_FLAT_SCRATCH_BASE_HI\n'
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
            '  if (ev == 235)\n'
            '    return static_cast<uint32_t>(wf.shared_aperture_base() >> 32); // SRC_SHARED_BASE\n'
            '  if (ev == 236)\n'
            '    return static_cast<uint32_t>(wf.shared_aperture_limit() >> 32); // SRC_SHARED_LIMIT\n'
            '  if (ev == 237)\n'
            '    return static_cast<uint32_t>(wf.private_aperture_base() >> 32); // SRC_PRIVATE_BASE\n'
            '  if (ev == 238)\n'
            '    return static_cast<uint32_t>(wf.private_aperture_limit() >> 32); // SRC_PRIVATE_LIMIT\n'
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
            '// Must stay in sync with resolve_src_scalar above — returns true for\n'
            '// exactly the encoding values that resolve_src_scalar handles without\n'
            '// throwing. Used by Isa::simd_capable_value() to keep the SIMD fast\n'
            '// path off operands whose scalar broadcast would throw at runtime.\n'
            'bool can_resolve_src_scalar(int ev) {\n'
            + (
                '  return (ev >= 0 && ev <= 107) || (ev >= 108 && ev <= 123) ||\n'
                '         ev == 124 || ev == 125 || ev == 126 || ev == 127 ||\n'
                '         (ev >= 128 && ev <= 208) || ev == 230 || ev == 231 ||\n'
                '         (ev >= 235 && ev <= 238) || (ev >= 240 && ev <= 253);\n'
                if arch in ('rdna4', 'gfx1250')
                else '  return (ev >= 0 && ev <= 107) || (ev >= 108 && ev <= 123) ||\n'
                '         ev == 124 || ev == 126 || ev == 127 ||\n'
                '         (ev >= 128 && ev <= 208) || (ev >= 235 && ev <= 238) ||\n'
                '         (ev >= 240 && ev <= 253);\n'
            )
            + '}\n'
            '\n'
            'uint64_t resolve_src_scalar64(const amdgpu::Wavefront &wf, int ev) {\n'
            '  if (ev == 102)\n'
            '    return wf.scratch_base();\n'
            '  if (ev <= 105) {\n'
            '    uint32_t lo = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            '    uint32_t hi = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1));\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            '  if (ev == 106)\n'
            '    return wf.vcc();\n'
            '  if (ev >= 108 && ev <= 122) {\n'
            '    uint32_t lo = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));\n'
            '    uint32_t hi = wf.cu().read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1));\n'
            '    return static_cast<uint64_t>(hi) << 32 | lo;\n'
            '  }\n'
            + (
                '  if (ev == 124)\n'
                '    return 0u; // NULL\n'
                '  if (ev == 125)\n'
                '    return wf.m0();\n'
                if arch in ('rdna4', 'gfx1250')
                else '  if (ev == 124)\n' '    return wf.m0();\n'
            )
            + '  if (ev == 126)\n'
            '    return wf.exec();\n'
            '  if (ev >= 128 && ev <= 192)\n'
            '    return static_cast<uint64_t>(ev - 128);\n'
            '  if (ev >= 193 && ev <= 208)\n'
            '    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(-(ev - 192))));\n'
            '  if (ev == 230)\n'
            '    return wf.scratch_base(); // SRC_FLAT_SCRATCH_BASE\n'
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
            '  if (ev == 235)\n'
            '    return wf.shared_aperture_base(); // SRC_SHARED_BASE\n'
            '  if (ev == 236)\n'
            '    return wf.shared_aperture_limit(); // SRC_SHARED_LIMIT\n'
            '  if (ev == 237)\n'
            '    return wf.private_aperture_base(); // SRC_PRIVATE_BASE\n'
            '  if (ev == 238)\n'
            '    return wf.private_aperture_limit(); // SRC_PRIVATE_LIMIT\n'
            '  throw std::logic_error("Unsupported encoding value for scalar64 read: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            'void resolve_dst_write(amdgpu::Wavefront &wf, int ev, uint32_t val) {\n'
            '  if (ev == 102) {\n'
            '    uint64_t sb = wf.scratch_base();\n'
            '    wf.set_scratch_base((sb & 0xFFFFFFFF00000000ULL) | val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 103) {\n'
            '    uint64_t sb = wf.scratch_base();\n'
            '    wf.set_scratch_base((sb & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));\n'
            '    return;\n'
            '  }\n'
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
            '  if (ev >= 108 && ev <= 123) {\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), val);\n'
            '    return;\n'
            '  }\n'
            + (
                '  if (ev == 124)\n'
                '    return;\n'
                '  if (ev == 125) {\n'
                '    wf.set_m0(val);\n'
                '    return;\n'
                '  }\n'
                if arch in ('rdna4', 'gfx1250')
                else '  if (ev == 124) {\n'
                '    wf.set_m0(val);\n'
                '    return;\n'
                '  }\n'
            )
            + '  if (ev == 126) {\n'
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
            '  if (ev == 102) {\n'
            '    wf.set_scratch_base(val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev <= 105) {\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), static_cast<uint32_t>(val));\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1), static_cast<uint32_t>(val >> 32));\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 106) {\n'
            '    wf.set_vcc(val);\n'
            '    return;\n'
            '  }\n'
            '  if (ev >= 108 && ev <= 122) {\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), static_cast<uint32_t>(val));\n'
            '    wf.cu().write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev + 1), static_cast<uint32_t>(val >> 32));\n'
            '    return;\n'
            '  }\n'
            '  if (ev == 124)\n'
            '    return;\n'
            '  if (ev == 126) {\n'
            '    wf.set_exec(val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("Unsupported encoding value for scalar64 write: " + std::to_string(ev));\n'
            '}\n'
            '\n'
            + _is_vgpr_only_body
            + '\n\n'
            + _is_immediate_body
            + '\n\n'
            + _vgpr_index_body
            + '\n\n'
            + '\n'
            '} // namespace\n'
            '\n'
            '// Isa::-scoped SIMD traits — see rocjitsu/isa/isa_operand_simd_inl.h\n'
            '// for the templated callers in AmdgpuIsaOperand<Isa>.\n'
            + _resolved_vgpr_offset_body
            + _resolved_vgpr_offset_with_wf_body
            + '\n\n'
            'bool Isa::simd_capable_value(OperandType opr_type, int ev) {\n'
            '  return resolved_vgpr_offset(opr_type, ev).has_value() ||\n'
            '         is_immediate_type(opr_type) || can_resolve_src_scalar(ev);\n'
            '}\n'
            '\n'
            'uint32_t Isa::simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type,\n'
            '                                   int ev) {\n'
            '  return is_immediate_type(opr_type) ? static_cast<uint32_t>(ev)\n'
            '                                     : resolve_src_scalar(wf, ev);\n'
            '}\n'
            '\n'
            + simd_methods
            + 'uint32_t Operand::read_scalar(const amdgpu::Wavefront &wf) const {\n'
            '  if (delegate()) return delegate()->read_scalar(wf);\n'
            '  if (has_literal64_)\n'
            '    return static_cast<uint32_t>(literal64_value_);\n'
            '  if (is_immediate_type(opr_type_))\n'
            '    return static_cast<uint32_t>(encoding_value_);\n'
            '  return resolve_src_scalar(wf, encoding_value_);\n'
            '}\n'
            '\n' + _read_lane_body + '\n\n'
            'void Operand::write_scalar(amdgpu::Wavefront &wf, uint32_t val) const {\n'
            '  resolve_dst_write(wf, encoding_value_, val);\n'
            '}\n'
            '\n'
            'void Operand::write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const {\n'
            f'  if (auto off = {_resolved_vgpr_encoded_call}) {{\n'
            '    wf.cu().write_vgpr(wf.vgpr_alloc().base + *off, lane, val);\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane called on non-VGPR operand type");\n'
            '}\n'
            '\n' + _read_lane64_body + '\n\n'
            'void Operand::write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const {\n'
            f'  if (auto off = {_resolved_vgpr_encoded_call}) {{\n'
            '    uint32_t idx = wf.vgpr_alloc().base + *off;\n'
            '    wf.cu().write_vgpr(idx, lane, static_cast<uint32_t>(val));\n'
            '    wf.cu().write_vgpr(idx + 1, lane, static_cast<uint32_t>(val >> 32));\n'
            '    return;\n'
            '  }\n'
            '  throw std::logic_error("write_lane64 called on non-VGPR operand type");\n'
            '}\n'
            '\n'
            'uint64_t Operand::read_scalar64(const amdgpu::Wavefront &wf) const {\n'
            '  if (has_literal64_)\n'
            '    return literal64_value_;\n'
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
                ('rocjitsu/isa/isa_operand_simd_inl.h', False),
                ('rocjitsu/vm/amdgpu/compute_unit.h', False),
                ('rocjitsu/vm/amdgpu/wavefront.h', False),
                ('format', True),
                ('optional', True),
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
                cgen.Value('static std::unique_ptr<Instruction>', 'decodeInvalid'),
                [cgen.Value('const MachineInst *', 'opcode')],
            ),
        ]
        decode_body = []
        if self._supports_gfx1250_scaled_wmma_vop3px2():
            class_impl.append(
                cgen.Line(self._emit_gfx1250_scaled_wmma_vop3px2_decoder_helpers())
            )
            decode_body.append(
                cgen.Statement(
                    'if (isWmmaScaleF32F8f6f4Vop3px2(opcode)) return std::make_unique<VWmmaScaleF3216x16x128F8f6f4Vop3px2>(opcode)'
                )
            )
        if self._supports_generated_vopd():
            decode_body.append(
                cgen.Statement(
                    'if (Vopd::is_vopd(opcode)) return std::make_unique<Vopd>(opcode)'
                )
            )
        decode_body.extend(
            [
                cgen.Statement(
                    'Sop1MachineInst op = std::bit_cast<decltype(op)>(*opcode)'
                ),
                cgen.Statement('return primary_decode_table[op.encoding](opcode)'),
            ]
        )
        decode_table_funcs = [
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value('std::unique_ptr<Instruction>', 'Decoder::decode'),
                    [cgen.Value('const MachineInst *', 'opcode')],
                ),
                cgen.Block(decode_body),
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
                decode_table_entries.append(f'&Decoder::{dte.decode_func},')
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
                                class_name = fn.removeprefix('decode')
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
                                                    f'return std::make_unique<{class_name}>(opcode)'
                                                )
                                            ]
                                        ),
                                    )
                                )
                            sub_decode_table_entry_str.append(f'&Decoder::{fn},')
                        sub_decode_table_entries.append(
                            cgen.Line(''.join(sub_decode_table_entry_str))
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
