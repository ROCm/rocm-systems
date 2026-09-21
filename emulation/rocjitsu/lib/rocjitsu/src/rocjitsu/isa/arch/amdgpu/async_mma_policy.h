// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file async_mma_policy.h
/// @brief Instruction eligibility and register footprints for asynchronous MMA execution.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <bitset>
#include <optional>
#include <string_view>

namespace rocjitsu::amdgpu::async_mma_policy {

// Only the instruction families qualified for the production async adapter.
constexpr bool supported(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

inline bool candidate(std::string_view name) {
  if (((name.starts_with("v_wmma_f32_16x16x64_") || name.starts_with("v_wmma_f32_16x16x128_")) &&
       (name.ends_with("fp8_fp8") || name.ends_with("fp8_bf8") || name.ends_with("bf8_fp8") ||
        name.ends_with("bf8_bf8"))) ||
      name == "v_wmma_f32_32x16x128_f4" || name == "v_wmma_scale_f32_16x16x128_f8f6f4" ||
      name == "v_wmma_f32_16x16x32_f16" || name == "v_wmma_f32_16x16x32_bf16")
    return true;
  if (name == "v_mfma_scale_f32_16x16x128_f8f6f4" || name == "v_mfma_scale_f32_32x32x64_f8f6f4" ||
      name == "v_mfma_f32_16x16x128_f8f6f4" || name == "v_mfma_f32_32x32x64_f8f6f4")
    return true;
  if ((name.starts_with("v_mfma_f32_16x16x32_") || name.starts_with("v_mfma_f32_32x32x16_")) &&
      (name.ends_with("fp8_fp8") || name.ends_with("fp8_bf8") || name.ends_with("bf8_fp8") ||
       name.ends_with("bf8_bf8")))
    return true;
  return name == "v_mfma_f32_32x32x8_f16" || name == "v_mfma_f32_16x16x16_f16" ||
         name == "v_mfma_f32_32x32x16_f16" || name == "v_mfma_f32_16x16x32_f16";
}

// Reject non-MFMA encodings on CDNA4, and non-candidates in the CDNA5 allowlist.
// This is only a hint: decoded eligibility remains authoritative.
inline bool encoding_may_be_candidate(rj_code_arch_t arch, uint32_t word) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    // VOP3P_MFMA, including the scaled-MFMA extension prefix.
    return word >> 23 == 423;
  if (arch != ROCJITSU_CODE_ARCH_CDNA5)
    return true;
  const uint32_t opcode = word >> 16;
  constexpr uint32_t encoding = 0xcc00;
  return (opcode >= encoding + cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x64Bf8Bf8Vop3p) ||
         (opcode >= encoding + cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p) ||
         opcode == encoding + cdna5::kVWmmaF3232x16x128F4Vop3p ||
         // The scale prefix needs full decoding to check its paired WMMA.
         opcode == encoding + 0x35 || opcode == encoding + cdna5::kVWmmaF3216x16x32F16Vop3p ||
         opcode == encoding + cdna5::kVWmmaF3216x16x32Bf16Vop3p;
}

class Access {
public:
  std::bitset<512> reads, writes;
  bool conflicts(const Access &next) const {
    return (writes & (next.reads | next.writes)).any() || (reads & next.writes).any();
  }
};
inline bool ordinary_memory(std::string_view name) {
  return name.starts_with("global_load_") || name.starts_with("global_store_") ||
         name.starts_with("buffer_load_") || name.starts_with("buffer_store_");
}
inline bool safe_inline(const Instruction &inst) {
  if (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR |
                      BARRIER | WRITES_EXEC))
    return false;
  const auto name = inst.mnemonic();
  if (name == "s_delay_alu" || name == "s_nop" || name == "v_nop")
    return true;
  if (candidate(name) || ordinary_memory(name))
    return true;
  if (inst.is_waitcnt())
    return !name.starts_with("s_wait_alu") && name != "s_wait_idle";
  // Explicitly exclude instructions with hidden EXEC, MODE, cache, scheduling,
  // register-allocation or cross-lane register-index side effects.
  constexpr std::string_view prefixes[] = {
      "s_mov_",     "s_add_",      "s_addc_",     "s_sub_",     "s_subb_", "s_mul_",    "s_mad_",
      "s_and_",     "s_or_",       "s_xor_",      "s_lshl_",    "s_lshr_", "s_ashr_",   "s_cmp_",
      "s_cselect_", "v_mov_",      "v_add_",      "v_addc_",    "v_sub_",  "v_subrev_", "v_mul_",
      "v_mad_",     "v_fma_",      "v_lshl",      "v_lshr",     "v_ashr",  "v_and_",    "v_or_",
      "v_xor_",     "v_cvt_",      "v_cmp_",      "v_cndmask_", "v_max_",  "v_min_",    "v_bfe_",
      "v_bfi_",     "v_alignbit_", "v_alignbyte_"};
  if (!std::ranges::any_of(prefixes, [&](auto prefix) { return name.starts_with(prefix); }))
    return false;
  for (int i = 0; i != inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    // Ordinary SGPRs only. Fieldless SCC/VCC are permitted by the whitelist;
    // WMMA workers do not read them. Explicit special-register writes drain.
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
      const uint32_t limit = acc ? (has_accvgprs ? 256 : 0) : std::min(256u, num_vgprs);
      const uint32_t index = ref->index, width = ref->width;
      if (index >= limit || width > limit - index)
        return std::nullopt;
      const uint32_t reg = index + (acc ? 256 : 0);
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
