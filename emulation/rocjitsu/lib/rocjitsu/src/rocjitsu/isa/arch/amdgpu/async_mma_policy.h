// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file async_mma_policy.h
/// @brief Instruction eligibility and register footprints for asynchronous MMA execution.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/async_inline_operations.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <bitset>
#include <optional>

namespace rocjitsu::amdgpu::async_mma_policy {

// CDNA4 dense MFMA handlers touch no wave status register. RDNA4 dense WMMA
// and sparse SWMMAC only read MODE.FP16_OVFL, which stays constant within a
// safe_inline() block and triggers a drain on cross-block mode writes.
// CDNA5 WMMA remains synchronous.
constexpr bool supported(rocjitsu::IsaFamily arch) {
  return arch == rocjitsu::IsaFamily::Cdna4 || arch == rocjitsu::IsaFamily::Rdna4;
}

inline bool is_rdna4_dense_wmma(Operation operation) {
  switch (operation) {
  case Operation::V_WMMA_F32_16X16X16_F16:
  case Operation::V_WMMA_F32_16X16X16_BF16:
  case Operation::V_WMMA_F16_16X16X16_F16:
  case Operation::V_WMMA_BF16_16X16X16_BF16:
  case Operation::V_WMMA_I32_16X16X16_IU8:
  case Operation::V_WMMA_I32_16X16X16_IU4:
  case Operation::V_WMMA_I32_16X16X32_IU4:
  case Operation::V_WMMA_F32_16X16X16_FP8_FP8:
  case Operation::V_WMMA_F32_16X16X16_FP8_BF8:
  case Operation::V_WMMA_F32_16X16X16_BF8_FP8:
  case Operation::V_WMMA_F32_16X16X16_BF8_BF8:
    return true;
  default:
    return false;
  }
}

inline bool is_rdna4_swmmac(Operation operation) {
  switch (operation) {
  case Operation::V_SWMMAC_F32_16X16X32_F16:
  case Operation::V_SWMMAC_F32_16X16X32_BF16:
  case Operation::V_SWMMAC_F16_16X16X32_F16:
  case Operation::V_SWMMAC_BF16_16X16X32_BF16:
  case Operation::V_SWMMAC_I32_16X16X32_IU8:
  case Operation::V_SWMMAC_I32_16X16X32_IU4:
  case Operation::V_SWMMAC_I32_16X16X64_IU4:
  case Operation::V_SWMMAC_F32_16X16X32_FP8_FP8:
  case Operation::V_SWMMAC_F32_16X16X32_FP8_BF8:
  case Operation::V_SWMMAC_F32_16X16X32_BF8_FP8:
  case Operation::V_SWMMAC_F32_16X16X32_BF8_BF8:
    return true;
  default:
    return false;
  }
}

inline bool is_rdna4_wmma(Operation operation) {
  return is_rdna4_dense_wmma(operation) || is_rdna4_swmmac(operation);
}

inline bool is_cdna4_smfmac(Operation operation) {
  switch (operation) {
  case Operation::V_SMFMAC_F32_16X16X64_BF16:
  case Operation::V_SMFMAC_I32_16X16X128_I8:
  case Operation::V_SMFMAC_F32_16X16X128_BF8_BF8:
  case Operation::V_SMFMAC_F32_16X16X128_BF8_FP8:
  case Operation::V_SMFMAC_F32_16X16X128_FP8_BF8:
  case Operation::V_SMFMAC_F32_16X16X128_FP8_FP8:
  case Operation::V_SMFMAC_F32_32X32X32_BF16:
  case Operation::V_SMFMAC_I32_32X32X64_I8:
  case Operation::V_SMFMAC_F32_32X32X64_BF8_BF8:
  case Operation::V_SMFMAC_F32_32X32X64_BF8_FP8:
  case Operation::V_SMFMAC_F32_32X32X64_FP8_BF8:
  case Operation::V_SMFMAC_F32_32X32X64_FP8_FP8:
  case Operation::V_SMFMAC_F32_16X16X64_F16:
  case Operation::V_SMFMAC_F32_32X32X32_F16:
  case Operation::V_SMFMAC_F32_16X16X32_F16:
  case Operation::V_SMFMAC_F32_32X32X16_F16:
  case Operation::V_SMFMAC_F32_16X16X32_BF16:
  case Operation::V_SMFMAC_F32_32X32X16_BF16:
  case Operation::V_SMFMAC_I32_16X16X64_I8:
  case Operation::V_SMFMAC_I32_32X32X32_I8:
  case Operation::V_SMFMAC_F32_16X16X64_BF8_BF8:
  case Operation::V_SMFMAC_F32_16X16X64_BF8_FP8:
  case Operation::V_SMFMAC_F32_16X16X64_FP8_BF8:
  case Operation::V_SMFMAC_F32_16X16X64_FP8_FP8:
  case Operation::V_SMFMAC_F32_32X32X32_BF8_BF8:
  case Operation::V_SMFMAC_F32_32X32X32_BF8_FP8:
  case Operation::V_SMFMAC_F32_32X32X32_FP8_BF8:
  case Operation::V_SMFMAC_F32_32X32X32_FP8_FP8:
    return true;
  default:
    return false;
  }
}

inline bool is_safe_vopd_op(Operation operation) {
  switch (operation) {
  case Operation::V_DUAL_FMAC_F32:
  case Operation::V_DUAL_FMAAK_F32:
  case Operation::V_DUAL_FMAMK_F32:
  case Operation::V_DUAL_MUL_F32:
  case Operation::V_DUAL_ADD_F32:
  case Operation::V_DUAL_SUB_F32:
  case Operation::V_DUAL_SUBREV_F32:
  case Operation::V_DUAL_MUL_DX9_ZERO_F32:
  case Operation::V_DUAL_MOV_B32:
  case Operation::V_DUAL_CNDMASK_B32:
  case Operation::V_DUAL_MAX_NUM_F32:
  case Operation::V_DUAL_MIN_NUM_F32:
  case Operation::V_DUAL_DOT2ACC_F32_F16:
  case Operation::V_DUAL_DOT2ACC_F32_BF16:
  case Operation::V_DUAL_ADD_NC_U32:
  case Operation::V_DUAL_LSHLREV_B32:
  case Operation::V_DUAL_AND_B32:
    return true;
  default:
    return false;
  }
}

inline bool is_safe_vopd(const Instruction &inst) {
  return is_safe_vopd_op(inst.amdgpu_operation()) &&
         is_safe_vopd_op(inst.amdgpu_secondary_operation()) &&
         !(inst.flags() & INVALID_VOPD_OPERANDS);
}

inline bool candidate(const Instruction &inst) {
  const Operation op = inst.amdgpu_operation();
  if (is_rdna4_wmma(op) || is_cdna4_smfmac(op))
    return true;
  switch (op) {
  case Operation::V_MFMA_SCALE_F32_16X16X128_F8F6F4:
  case Operation::V_MFMA_SCALE_F32_32X32X64_F8F6F4:
  case Operation::V_MFMA_F32_16X16X128_F8F6F4:
  case Operation::V_MFMA_F32_32X32X64_F8F6F4:
  case Operation::V_MFMA_F32_16X16X32_FP8_FP8:
  case Operation::V_MFMA_F32_16X16X32_FP8_BF8:
  case Operation::V_MFMA_F32_16X16X32_BF8_FP8:
  case Operation::V_MFMA_F32_16X16X32_BF8_BF8:
  case Operation::V_MFMA_F32_32X32X16_FP8_FP8:
  case Operation::V_MFMA_F32_32X32X16_FP8_BF8:
  case Operation::V_MFMA_F32_32X32X16_BF8_FP8:
  case Operation::V_MFMA_F32_32X32X16_BF8_BF8:
  case Operation::V_MFMA_F32_16X16X32_F16:
  case Operation::V_MFMA_F32_32X32X16_F16:
  case Operation::V_MFMA_F32_16X16X16_F16:
  case Operation::V_MFMA_F32_32X32X8_F16:
  case Operation::V_MFMA_F32_16X16X32_BF16:
  case Operation::V_MFMA_F32_32X32X16_BF16:
  case Operation::V_MFMA_F32_16X16X16_BF16:
  case Operation::V_MFMA_F32_32X32X8_BF16:
  case Operation::V_MFMA_F32_32X32X2_F32:
  case Operation::V_MFMA_F32_16X16X4_F32:
  case Operation::V_MFMA_F64_16X16X4_F64:
  case Operation::V_MFMA_F64_4X4X4_4B_F64:
  case Operation::V_MFMA_I32_16X16X64_I8:
  case Operation::V_MFMA_I32_32X32X32_I8:
  case Operation::V_MFMA_I32_16X16X32_I8:
  case Operation::V_MFMA_I32_32X32X16_I8:
    return true;
  default:
    return false;
  }
}

// Reject non-MFMA/WMMA encodings on CDNA4/RDNA4, and non-candidates on CDNA5.
// This is only a hint: decoded eligibility remains authoritative.
inline bool encoding_may_be_candidate(rocjitsu::IsaFamily arch, uint32_t word) {
  if (arch == rocjitsu::IsaFamily::Cdna4) {
    // VOP3P with a candidate MFMA/SMFMAC opcode, or the scaled-MFMA prefix (44).
    // Other VOP3P (e.g. v_fma_mix_f32) shares the encoding and would only build
    // and look up rejected plans.
    constexpr uint64_t kLow = (uint64_t{1} << 44) | (uint64_t{1} << 45) | (uint64_t{1} << 46) |
                              (uint64_t{1} << 53) | (uint64_t{1} << 54) | (uint64_t{1} << 55) |
                              (uint64_t{1} << 56) | (uint64_t{0x1f} << 57);
    constexpr uint64_t kHigh = (uint64_t{1} << (67 - 64)) | (uint64_t{1} << (68 - 64)) |
                               (uint64_t{1} << (69 - 64)) | (uint64_t{1} << (70 - 64)) |
                               (uint64_t{1} << (71 - 64)) | (uint64_t{1} << (75 - 64)) |
                               (uint64_t{1} << (76 - 64)) | (uint64_t{1} << (77 - 64)) |
                               (uint64_t{1} << (78 - 64)) | (uint64_t{1} << (79 - 64)) |
                               (uint64_t{1} << (83 - 64)) | (uint64_t{1} << (84 - 64)) |
                               (uint64_t{1} << (85 - 64)) | (uint64_t{1} << (86 - 64)) |
                               (uint64_t{1} << (87 - 64)) | (uint64_t{1} << (90 - 64)) |
                               (uint64_t{1} << (91 - 64)) | (uint64_t{1} << (96 - 64)) |
                               (uint64_t{1} << (97 - 64)) | (uint64_t{1} << (98 - 64)) |
                               (uint64_t{1} << (100 - 64)) | (uint64_t{1} << (102 - 64)) |
                               (uint64_t{1} << (104 - 64)) | (uint64_t{1} << (106 - 64)) |
                               (uint64_t{1} << (108 - 64)) | (uint64_t{1} << (110 - 64)) |
                               (uint64_t{1} << (111 - 64)) | (uint64_t{0xffff} << (112 - 64));
    const uint32_t opcode = (word >> 16) & 0x7f;
    return word >> 23 == 423 && ((opcode < 64 ? kLow >> opcode : kHigh >> (opcode - 64)) & 1);
  }
  if (arch == rocjitsu::IsaFamily::Rdna4) {
    // RDNA4 VOP3P encoding byte is 0xcc (primary 408..409); opcodes 64..74 are
    // dense V_WMMA_* and 80..90 are sparse V_SWMMAC_*.
    const uint32_t opcode = (word >> 16) & 0x7f;
    return (word >> 24) == 0xcc && ((opcode >= 64 && opcode <= 74) ||
                                    (opcode >= 80 && opcode <= 90));
  }
  if (arch != rocjitsu::IsaFamily::Cdna5)
    return true;
  const uint32_t opcode = word >> 16;
  constexpr uint32_t encoding = 0xcc00;
  return (opcode >= encoding + cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x64Bf8Bf8Vop3p) ||
         (opcode >= encoding + cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p) ||
         opcode == encoding + cdna5::kVWmmaF3232x16x128F4Vop3p ||
         // The scale prefix needs full decoding to check its paired WMMA.
         opcode == encoding + 0x35;
}

class Access {
public:
  static constexpr uint32_t kRegistersPerBank = 256;
  std::bitset<2 * kRegistersPerBank> reads, writes;
  bool conflicts(const Access &next) const {
    return (writes & (next.reads | next.writes)).any() || (reads & next.writes).any();
  }
};
inline bool ordinary_memory(const Instruction &inst) {
  return inline_operation_kind(inst.amdgpu_operation()) == InlineOperationKind::Memory;
}
// LDS reads and the LDS-free lane permute write only their VGPR destinations,
// which the footprint covers; they execute in program order on the issuer.
// CDNA5 spells the same instructions DS_LOAD_*.
inline bool lds_read(Operation operation) {
  switch (operation) {
  case Operation::DS_READ_B32:
  case Operation::DS_READ_B64:
  case Operation::DS_READ_B96:
  case Operation::DS_READ_B128:
  case Operation::DS_READ2_B32:
  case Operation::DS_READ2_B64:
  case Operation::DS_READ2ST64_B32:
  case Operation::DS_READ2ST64_B64:
  case Operation::DS_READ_I8:
  case Operation::DS_READ_U8:
  case Operation::DS_READ_I16:
  case Operation::DS_READ_U16:
  case Operation::DS_READ_I8_D16:
  case Operation::DS_READ_I8_D16_HI:
  case Operation::DS_READ_U8_D16:
  case Operation::DS_READ_U8_D16_HI:
  case Operation::DS_READ_U16_D16:
  case Operation::DS_READ_U16_D16_HI:
  case Operation::DS_READ_B64_TR_B4:
  case Operation::DS_READ_B64_TR_B8:
  case Operation::DS_READ_B64_TR_B16:
  case Operation::DS_READ_B96_TR_B6:
  case Operation::DS_BPERMUTE_B32:
  case Operation::DS_LOAD_B32:
  case Operation::DS_LOAD_B64:
  case Operation::DS_LOAD_B96:
  case Operation::DS_LOAD_B128:
  case Operation::DS_LOAD_2ADDR_B32:
  case Operation::DS_LOAD_2ADDR_B64:
  case Operation::DS_LOAD_2ADDR_STRIDE64_B32:
  case Operation::DS_LOAD_2ADDR_STRIDE64_B64:
  case Operation::DS_LOAD_I8:
  case Operation::DS_LOAD_U8:
  case Operation::DS_LOAD_I16:
  case Operation::DS_LOAD_U16:
  case Operation::DS_LOAD_I8_D16:
  case Operation::DS_LOAD_I8_D16_HI:
  case Operation::DS_LOAD_U8_D16:
  case Operation::DS_LOAD_U8_D16_HI:
  case Operation::DS_LOAD_U16_D16:
  case Operation::DS_LOAD_U16_D16_HI:
  case Operation::DS_LOAD_TR4_B64:
  case Operation::DS_LOAD_TR6_B96:
  case Operation::DS_LOAD_TR8_B64:
  case Operation::DS_LOAD_TR16_B128:
  case Operation::DS_LOAD_ADDTID_B32:
  case Operation::DS_PERMUTE_B32:
  case Operation::DS_BPERMUTE_FI_B32:
  case Operation::DS_SWIZZLE_B32:
    return true;
  default:
    return false;
  }
}
// Plain LDS stores read only VGPRs (address and data), which the footprint
// covers, and complete at issue; there is no virtual memory behind them.
inline bool lds_write(Operation operation) {
  switch (operation) {
  case Operation::DS_WRITE_B8:
  case Operation::DS_WRITE_B16:
  case Operation::DS_WRITE_B32:
  case Operation::DS_WRITE_B64:
  case Operation::DS_WRITE_B96:
  case Operation::DS_WRITE_B128:
  case Operation::DS_WRITE_B8_D16_HI:
  case Operation::DS_WRITE_B16_D16_HI:
  case Operation::DS_WRITE2_B32:
  case Operation::DS_WRITE2_B64:
  case Operation::DS_WRITE2ST64_B32:
  case Operation::DS_WRITE2ST64_B64:
  case Operation::DS_STORE_B8:
  case Operation::DS_STORE_B16:
  case Operation::DS_STORE_B32:
  case Operation::DS_STORE_B64:
  case Operation::DS_STORE_B96:
  case Operation::DS_STORE_B128:
  case Operation::DS_STORE_B8_D16_HI:
  case Operation::DS_STORE_B16_D16_HI:
  case Operation::DS_STORE_2ADDR_B32:
  case Operation::DS_STORE_2ADDR_B64:
  case Operation::DS_STORE_2ADDR_STRIDE64_B32:
  case Operation::DS_STORE_2ADDR_STRIDE64_B64:
  case Operation::DS_STORE_ADDTID_B32:
    return true;
  default:
    return false;
  }
}
inline bool lds_access(Operation operation) { return lds_read(operation) || lds_write(operation); }
inline bool global_data_share(const Instruction &inst) {
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (!issue)
    return false;
  for (const auto &obligation : issue->counter_obligations())
    if (obligation.completion_class() == MemoryCompletionClass::GDS)
      return true;
  return false;
}
inline constexpr int kCdna4MaxSgprCount = 102;
inline constexpr int kRdna4MaxSgprCount = 106;
inline constexpr int kVccLoEncoding = 106;
inline constexpr int kVccHiEncoding = 107;
inline constexpr int kM0Encoding = 124;
inline constexpr int kRdna4NullEncoding = 124;
inline constexpr int kRdna4M0Encoding = 125;

inline bool safe_inline(const Instruction &inst,
                        rocjitsu::IsaFamily arch = rocjitsu::IsaFamily::Cdna4) {
  const Operation operation = inst.amdgpu_operation();
  if (operation == Operation::Unknown)
    return false;
  if (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR |
                      BARRIER | WRITES_EXEC))
    return false;
  if (inst.amdgpu_secondary_operation() != Operation::Unknown)
    return is_safe_vopd(inst);
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  // DPP execution temporarily narrows the wave's VGPR write mask. Keep that
  // wave-wide state stable while an asynchronous MMA may still consult it.
  if (modifiers.dpp)
    return false;
  if (operation == Operation::S_DELAY_ALU || operation == Operation::S_NOP ||
      operation == Operation::V_NOP || operation == Operation::DS_NOP ||
      operation == Operation::S_SETPRIO || operation == Operation::S_CLAUSE ||
      operation == Operation::S_WAIT_ALU || operation == Operation::S_PREFETCH_DATA ||
      operation == Operation::S_PREFETCH_DATA_PC_REL ||
      operation == Operation::S_PREFETCH_INST ||
      operation == Operation::S_PREFETCH_INST_PC_REL ||
      operation == Operation::S_BUFFER_PREFETCH_DATA)
    return true;
  if (lds_access(operation))
    return !global_data_share(inst);
  if (candidate(inst) || ordinary_memory(inst))
    return true;
  if (inst.is_waitcnt())
    return operation != Operation::S_WAIT_IDLE;
  // Explicitly exclude instructions with hidden EXEC, MODE, cache, scheduling,
  // register-allocation or cross-lane register-index side effects.
  if (inline_operation_kind(operation) != InlineOperationKind::Arithmetic &&
      operation != Operation::S_MOVK_I32 && operation != Operation::S_ADDK_I32)
    return false;
  const bool is_rdna4 = arch == rocjitsu::IsaFamily::Rdna4;
  const int max_sgpr_count = is_rdna4 ? kRdna4MaxSgprCount : kCdna4MaxSgprCount;
  for (int i = 0; i != inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (!op || op->is_vgpr() || op->is_fieldless())
      continue;
    const int enc = op->encoding_value();
    const int dwords = (op->size_bits() + 31) / 32;
    // Ordinary SGPRs (s0..s101 on CDNA4, s0..s105 on RDNA4).
    if (enc >= 0 && enc + dwords <= max_sgpr_count)
      continue;
    // Matrix helpers never read VCC, M0, or RDNA4 NULL.
    if (enc >= kVccLoEncoding && enc + dwords <= kVccHiEncoding + 1)
      continue;
    if (dwords == 1 && (enc == kM0Encoding || (is_rdna4 && (enc == kRdna4NullEncoding ||
                                                            enc == kRdna4M0Encoding))))
      continue;
    return false;
  }
  return true;
}

// RDNA4 WMMA/SWMMAC operands decode with Wave32 bit widths; Wave64 halves the
// VGPR count per operand (rounding up to 1 for 32-bit operands).
inline uint32_t matrix_operand_width(const Instruction &inst, uint32_t width, uint32_t wave_size) {
  if (wave_size == 64 && is_rdna4_wmma(inst.amdgpu_operation()))
    return (width + 1u) / 2u;
  return width;
}

inline std::optional<Access> footprint(const Instruction &inst, uint32_t num_vgprs,
                                       bool has_accvgprs = false, uint32_t wave_size = 32) {
  Access access;
  const bool tied_dst =
      is_rdna4_swmmac(inst.amdgpu_operation()) || is_cdna4_smfmac(inst.amdgpu_operation());
  for (bool dst : {false, true}) {
    const int count = dst ? inst.num_dst_operands() : inst.num_src_operands();
    for (int i = 0; i != count; ++i) {
      const auto *op = dst ? inst.dst_operand(i) : inst.src_operand(i);
      if (!op || !op->is_vgpr())
        continue;
      // is_vgpr() describes selector capability, including scalar and inline
      // encodings. Resolve packed-half aliases to their physical register
      // before constructing the scoreboard footprint.
      const auto ref = op->to_register_ref();
      if (!ref || (ref->cls != RegClass::VGPR && ref->cls != RegClass::ACC_VGPR))
        continue;
      const bool acc = ref->cls == RegClass::ACC_VGPR;
      const uint32_t limit = acc ? (has_accvgprs ? Access::kRegistersPerBank : 0)
                                 : std::min(Access::kRegistersPerBank, num_vgprs);
      const uint32_t index = ref->index;
      const uint32_t width = matrix_operand_width(inst, ref->width, wave_size);
      if (index >= limit || width > limit - index)
        return std::nullopt;
      const uint32_t reg = index + (acc ? Access::kRegistersPerBank : 0);
      for (uint32_t r = reg; r != reg + width; ++r) {
        (dst ? access.writes : access.reads).set(r);
        // A store's encoding can describe its data as a destination operand,
        // and sparse WMMA reads its tied accumulator destination.
        if (inst.is_memory_op() || (dst && tied_dst))
          access.reads.set(r);
      }
    }
  }
  return access;
}
} // namespace rocjitsu::amdgpu::async_mma_policy
