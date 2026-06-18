// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_RISC_V_DECODER_H_
#define ROCJITSU_ISA_RISC_V_DECODER_H_

#include "rocjitsu/isa/instruction.h"

#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace risc_v {

using DecodeFn = std::unique_ptr<Instruction> (*)(uint32_t);

class Decoder {
public:
  static std::unique_ptr<Instruction> decode(uint32_t instr);

private:
  // =========================================================================
  // Dispatch template: extracts bit-field from instr, indexes into subtable,
  // calls the function pointer found there. Returns nullptr for null entries.
  // =========================================================================
  template <const DecodeFn *table, unsigned shift, unsigned num_bits>
  static std::unique_ptr<Instruction> decode_field(uint32_t instr) {
    uint32_t idx = (instr >> shift) & ((1u << num_bits) - 1);
    auto fn = table[idx];
    return fn ? fn(instr) : nullptr;
  }

  // =========================================================================
  // Decode stub declarations
  // =========================================================================

  // Special
  static std::unique_ptr<Instruction> illegal_decode(uint32_t instr);

  // RV64I Base Integer (52)
  static std::unique_ptr<Instruction> lui_decode(uint32_t instr);
  static std::unique_ptr<Instruction> auipc_decode(uint32_t instr);
  static std::unique_ptr<Instruction> jal_decode(uint32_t instr);
  static std::unique_ptr<Instruction> jalr_decode(uint32_t instr);
  static std::unique_ptr<Instruction> beq_decode(uint32_t instr);
  static std::unique_ptr<Instruction> bne_decode(uint32_t instr);
  static std::unique_ptr<Instruction> blt_decode(uint32_t instr);
  static std::unique_ptr<Instruction> bge_decode(uint32_t instr);
  static std::unique_ptr<Instruction> bltu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> bgeu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lb_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lh_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> ld_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lbu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lhu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> lwu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sb_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sh_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sd_decode(uint32_t instr);
  static std::unique_ptr<Instruction> addi_decode(uint32_t instr);
  static std::unique_ptr<Instruction> slti_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sltiu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> xori_decode(uint32_t instr);
  static std::unique_ptr<Instruction> ori_decode(uint32_t instr);
  static std::unique_ptr<Instruction> andi_decode(uint32_t instr);
  static std::unique_ptr<Instruction> slli_decode(uint32_t instr);
  static std::unique_ptr<Instruction> srli_decode(uint32_t instr);
  static std::unique_ptr<Instruction> srai_decode(uint32_t instr);
  static std::unique_ptr<Instruction> add_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sub_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sll_decode(uint32_t instr);
  static std::unique_ptr<Instruction> slt_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sltu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> xor_decode(uint32_t instr);
  static std::unique_ptr<Instruction> srl_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sra_decode(uint32_t instr);
  static std::unique_ptr<Instruction> or_decode(uint32_t instr);
  static std::unique_ptr<Instruction> and_decode(uint32_t instr);
  static std::unique_ptr<Instruction> addiw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> slliw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> srliw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sraiw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> addw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> subw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sllw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> srlw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sraw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> fence_decode(uint32_t instr);
  static std::unique_ptr<Instruction> ecall_decode(uint32_t instr);
  static std::unique_ptr<Instruction> ebreak_decode(uint32_t instr);

  // Privileged (4)
  static std::unique_ptr<Instruction> sret_decode(uint32_t instr);
  static std::unique_ptr<Instruction> mret_decode(uint32_t instr);
  static std::unique_ptr<Instruction> wfi_decode(uint32_t instr);
  static std::unique_ptr<Instruction> sfence_vma_decode(uint32_t instr);

  // Zicsr Extension (6)
  static std::unique_ptr<Instruction> zicsr_csrrw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> zicsr_csrrs_decode(uint32_t instr);
  static std::unique_ptr<Instruction> zicsr_csrrc_decode(uint32_t instr);
  static std::unique_ptr<Instruction> zicsr_csrrwi_decode(uint32_t instr);
  static std::unique_ptr<Instruction> zicsr_csrrsi_decode(uint32_t instr);
  static std::unique_ptr<Instruction> zicsr_csrrci_decode(uint32_t instr);

  // Zifencei Extension (1)
  static std::unique_ptr<Instruction> zifencei_fence_i_decode(uint32_t instr);

  // M Extension (13)
  static std::unique_ptr<Instruction> m_mul_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_mulh_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_mulhsu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_mulhu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_div_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_divu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_rem_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_remu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_mulw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_divw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_divuw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_remw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> m_remuw_decode(uint32_t instr);

  // A Extension (22)
  static std::unique_ptr<Instruction> a_lr_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_sc_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoswap_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoadd_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoxor_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoand_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoor_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomin_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomax_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amominu_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomaxu_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_lr_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_sc_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoswap_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoadd_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoxor_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoand_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amoor_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomin_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomax_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amominu_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> a_amomaxu_d_decode(uint32_t instr);

  // F Extension (30)
  static std::unique_ptr<Instruction> f_flw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsw_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fadd_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsub_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmul_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fdiv_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsqrt_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsgnj_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsgnjn_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fsgnjx_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmin_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmax_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_w_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_wu_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_l_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_lu_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_s_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_s_wu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_s_l_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fcvt_s_lu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmv_x_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fclass_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmv_w_x_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_feq_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_flt_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fle_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmadd_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fmsub_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fnmsub_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> f_fnmadd_s_decode(uint32_t instr);

  // D Extension (32)
  static std::unique_ptr<Instruction> d_fld_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsd_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fadd_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsub_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmul_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fdiv_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsqrt_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsgnj_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsgnjn_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fsgnjx_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmin_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmax_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_w_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_wu_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_l_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_lu_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_d_w_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_d_wu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_d_l_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_d_lu_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_s_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fcvt_d_s_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmv_x_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fclass_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmv_d_x_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_feq_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_flt_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fle_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmadd_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fmsub_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fnmsub_d_decode(uint32_t instr);
  static std::unique_ptr<Instruction> d_fnmadd_d_decode(uint32_t instr);

  // Vendor Custom (4)
  static std::unique_ptr<Instruction> vendor0_ext_decode(uint32_t instr);
  static std::unique_ptr<Instruction> vendor1_ext_decode(uint32_t instr);
  static std::unique_ptr<Instruction> vendor2_ext_decode(uint32_t instr);
  static std::unique_ptr<Instruction> vendor3_ext_decode(uint32_t instr);

  // =========================================================================
  // Decode tables (static, const, directly initialized in .cpp)
  // =========================================================================

  // Level 0: opcode[6:2] (32 entries)
  static const DecodeFn opcode_table_[32];

  // Level 1: funct3 tables (8 entries each)
  static const DecodeFn load_funct3_[8];
  static const DecodeFn load_fp_funct3_[8];
  static const DecodeFn misc_mem_funct3_[8];
  static const DecodeFn op_imm_funct3_[8];
  static const DecodeFn op_imm_32_funct3_[8];
  static const DecodeFn store_funct3_[8];
  static const DecodeFn store_fp_funct3_[8];
  static const DecodeFn amo_funct3_[8];
  static const DecodeFn op_funct3_[8];
  static const DecodeFn op_32_funct3_[8];
  static const DecodeFn branch_funct3_[8];
  static const DecodeFn system_funct3_[8];

  // Level 1: funct2 tables (4 entries each)
  static const DecodeFn madd_fmt_[4];
  static const DecodeFn msub_fmt_[4];
  static const DecodeFn nmsub_fmt_[4];
  static const DecodeFn nmadd_fmt_[4];

  // Level 1: OP-FP funct7 (128 entries)
  static const DecodeFn op_fp_funct7_[128];

  // Level 2: OP-IMM / OP-IMM-32 shift-right bit30 (2 entries each)
  static const DecodeFn op_imm_sri_bit30_[2];
  static const DecodeFn op_imm32_sri_bit30_[2];

  // Level 2: AMO funct5 (32 entries per width)
  static const DecodeFn amo_w_funct5_[32];
  static const DecodeFn amo_d_funct5_[32];

  // Level 2: OP funct7 tables (128 entries x 8 funct3 values)
  static const DecodeFn op_f3_000_funct7_[128];
  static const DecodeFn op_f3_001_funct7_[128];
  static const DecodeFn op_f3_010_funct7_[128];
  static const DecodeFn op_f3_011_funct7_[128];
  static const DecodeFn op_f3_100_funct7_[128];
  static const DecodeFn op_f3_101_funct7_[128];
  static const DecodeFn op_f3_110_funct7_[128];
  static const DecodeFn op_f3_111_funct7_[128];

  // Level 2: OP-32 funct7 tables (128 entries x 8 funct3 values)
  static const DecodeFn op32_f3_000_funct7_[128];
  static const DecodeFn op32_f3_001_funct7_[128];
  static const DecodeFn op32_f3_010_funct7_[128];
  static const DecodeFn op32_f3_011_funct7_[128];
  static const DecodeFn op32_f3_100_funct7_[128];
  static const DecodeFn op32_f3_101_funct7_[128];
  static const DecodeFn op32_f3_110_funct7_[128];
  static const DecodeFn op32_f3_111_funct7_[128];

  // Level 2: OP-FP subtables keyed by funct3 (8 entries each)
  static const DecodeFn fp_fsgnj_s_funct3_[8];
  static const DecodeFn fp_fsgnj_d_funct3_[8];
  static const DecodeFn fp_fminmax_s_funct3_[8];
  static const DecodeFn fp_fminmax_d_funct3_[8];
  static const DecodeFn fp_fcmp_s_funct3_[8];
  static const DecodeFn fp_fcmp_d_funct3_[8];
  static const DecodeFn fp_fmvclass_s_funct3_[8];
  static const DecodeFn fp_fmvclass_d_funct3_[8];

  // Level 2: OP-FP subtables keyed by rs2 (32 entries each)
  static const DecodeFn fp_fcvt_s_d_rs2_[32];
  static const DecodeFn fp_fcvt_d_s_rs2_[32];
  static const DecodeFn fp_fcvt_int_from_s_rs2_[32];
  static const DecodeFn fp_fcvt_int_from_d_rs2_[32];
  static const DecodeFn fp_fcvt_s_from_int_rs2_[32];
  static const DecodeFn fp_fcvt_d_from_int_rs2_[32];

  // Level 2: SYSTEM privileged funct7 (128 entries)
  static const DecodeFn system_priv_funct7_[128];

  // Level 3: SYSTEM privileged rs2 tables (32 entries each)
  static const DecodeFn system_priv_ecall_rs2_[32];
  static const DecodeFn system_priv_sret_rs2_[32];
  static const DecodeFn system_priv_mret_rs2_[32];
};

} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_RISC_V_DECODER_H_
