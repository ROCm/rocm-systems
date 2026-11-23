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

// CDNA4 dense MFMA handlers touch no wave status register. They only read the
// issuer-owned instruction error field, which stays clear on valid kernels.
// CDNA5 WMMA remains synchronous.
constexpr bool supported(rocjitsu::IsaFamily arch) { return arch == rocjitsu::IsaFamily::Cdna4; }

inline bool candidate(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
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
    return true;
  default:
    return false;
  }
}

// Reject non-MFMA encodings on CDNA4, and non-candidates in the CDNA5 allowlist.
// This is only a hint: decoded eligibility remains authoritative.
inline bool encoding_may_be_candidate(rocjitsu::IsaFamily arch, uint32_t word) {
  if (arch == rocjitsu::IsaFamily::Cdna4)
    // VOP3P_MFMA, including the scaled-MFMA extension prefix.
    return word >> 23 == 423;
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
    return true;
  default:
    return false;
  }
}
inline bool global_data_share(const Instruction &inst) {
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (!issue)
    return false;
  for (const auto &obligation : issue->counter_obligations())
    if (obligation.completion_class() == MemoryCompletionClass::GDS)
      return true;
  return false;
}
inline constexpr int kM0Encoding = 124;

inline bool safe_inline(const Instruction &inst) {
  const Operation operation = inst.amdgpu_operation();
  if (operation == Operation::Unknown || inst.amdgpu_secondary_operation() != Operation::Unknown)
    return false;
  if (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR |
                      BARRIER | WRITES_EXEC))
    return false;
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  // DPP execution temporarily narrows the wave's VGPR write mask. Keep that
  // wave-wide state stable while an asynchronous MMA may still consult it.
  if (modifiers.dpp)
    return false;
  if (operation == Operation::S_DELAY_ALU || operation == Operation::S_NOP ||
      operation == Operation::V_NOP || operation == Operation::S_SETPRIO)
    return true;
  if (lds_read(operation))
    return !global_data_share(inst);
  if (candidate(inst) || ordinary_memory(inst))
    return true;
  if (inst.is_waitcnt())
    return operation != Operation::S_WAIT_ALU && operation != Operation::S_WAIT_IDLE;
  // Explicitly exclude instructions with hidden EXEC, MODE, cache, scheduling,
  // register-allocation or cross-lane register-index side effects.
  if (inline_operation_kind(operation) != InlineOperationKind::Arithmetic &&
      operation != Operation::S_MOVK_I32)
    return false;
  for (int i = 0; i != inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    // Ordinary SGPRs only. Fieldless SCC/VCC are permitted by the whitelist;
    // WMMA workers do not read them. Explicit special-register writes drain,
    // except an M0 move: matrix helpers never read M0, and the LDS and
    // buffer-to-LDS instructions that do run on the issuer in program order.
    if (op && operation == Operation::S_MOV_B32 && op->encoding_value() == kM0Encoding)
      continue;
    if (op && !op->is_vgpr() && !op->is_fieldless() &&
        (op->encoding_value() < 0 || op->encoding_value() + (op->size_bits() + 31) / 32 > 102))
      return false;
  }
  return true;
}

inline std::optional<Access> footprint(const Instruction &inst, uint32_t num_vgprs,
                                       bool has_accvgprs = false) {
  Access access;
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
      const uint32_t index = ref->index, width = ref->width;
      if (index >= limit || width > limit - index)
        return std::nullopt;
      const uint32_t reg = index + (acc ? Access::kRegistersPerBank : 0);
      for (uint32_t r = reg; r != reg + width; ++r) {
        (dst ? access.writes : access.reads).set(r);
        // A store's encoding can describe its data as a destination operand.
        if (inst.is_memory_op())
          access.reads.set(r);
      }
    }
  }
  return access;
}
} // namespace rocjitsu::amdgpu::async_mma_policy
