// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <bitset>
#include <optional>
#include <string_view>

namespace rocjitsu::amdgpu::async_mma_policy {

// Instruction semantics and target eligibility live together; configuration,
// helper ownership and queue scheduling remain in the execution adapter.
struct TargetTraits {
  unsigned wave_size;
  bool has_accvgprs;
};

constexpr TargetTraits target_traits(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return {64, true};
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return {32, false};
  default:
    return {};
  }
}

template <GpuIsa Isa> constexpr bool enabled(int mode, unsigned min_k, unsigned mfma_bits) {
  if constexpr (!HasAsyncMma<Isa>)
    return false;
  else if constexpr (HasLargeWmma<Isa>)
    return mode >= 1 && mode <= 4;
  else if constexpr (Isa::ASYNC_MMA_WAVE_SIZE == 64)
    return mode == 4 && mfma_bits != 0;
  else
    // RDNA4's accepted dense floating-point WMMA shapes have K=16. Its K=32
    // integer and sparse instructions are outside the experimental allowlist.
    return mode == 4 && min_k <= 16;
}

inline bool large_candidate(std::string_view mnemonic) {
  return ((mnemonic.starts_with("v_wmma_f32_16x16x64_") ||
           mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
          (mnemonic.ends_with("fp8_fp8") || mnemonic.ends_with("fp8_bf8") ||
           mnemonic.ends_with("bf8_fp8") || mnemonic.ends_with("bf8_bf8"))) ||
         mnemonic == "v_wmma_f32_32x16x128_f4";
}

inline unsigned wmma_k(std::string_view name) {
  if (name == "v_wmma_f32_16x16x32_f16" || name == "v_wmma_f32_16x16x32_bf16")
    return 32;
  if (name == "v_wmma_f32_16x16x16_f16" || name == "v_wmma_f32_16x16x16_bf16")
    return 16;
  return 0;
}

inline bool mfma_candidate(std::string_view name, unsigned mfma_bits) {
  if ((mfma_bits & 1u) &&
      (name == "v_mfma_f32_32x32x4_2b_f16" || name == "v_mfma_f32_16x16x4_4b_f16" ||
       name == "v_mfma_f32_4x4x4_16b_f16" || name == "v_mfma_f32_32x32x1_2b_f32" ||
       name == "v_mfma_f32_16x16x1_4b_f32"))
    return true;
  if ((mfma_bits & 2u) &&
      (name == "v_mfma_scale_f32_16x16x128_f8f6f4" || name == "v_mfma_scale_f32_32x32x64_f8f6f4"))
    return true;
  return (mfma_bits & 4u) &&
         (name == "v_mfma_f32_32x32x8_f16" || name == "v_mfma_f32_16x16x16_f16" ||
          name == "v_mfma_f32_32x32x16_f16" || name == "v_mfma_f32_16x16x32_f16");
}

inline bool candidate(std::string_view name, unsigned min_k, unsigned mfma_bits) {
  if (large_candidate(name))
    return true;
  if (unsigned k = wmma_k(name))
    return min_k <= k;
  return mfma_candidate(name, mfma_bits);
}

inline bool needs_admission(int mode, std::string_view name) {
  return mode == 3 || name == "v_wmma_f32_16x16x32_f16" || name == "v_wmma_f32_16x16x32_bf16";
}

constexpr bool has_encoding_hint(rj_code_arch_t arch, unsigned min_k) {
  return arch == ROCJITSU_CODE_ARCH_CDNA5 && min_k >= 32;
}

// A rejection is conclusive only for the default CDNA5 dense allowlist. Other
// architectures and experimental K16 paths always require ordinary decoding.
inline bool encoding_may_be_candidate(rj_code_arch_t arch, uint32_t word, unsigned min_k) {
  if (!has_encoding_hint(arch, min_k))
    return true;
  const uint32_t opcode = word >> 16;
  constexpr uint32_t encoding = 0xcc00;
  return (opcode >= encoding + cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x64Bf8Bf8Vop3p) ||
         (opcode >= encoding + cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p &&
          opcode <= encoding + cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p) ||
         opcode == encoding + cdna5::kVWmmaF3232x16x128F4Vop3p ||
         (min_k <= 32 && (opcode == encoding + cdna5::kVWmmaF3216x16x32F16Vop3p ||
                          opcode == encoding + cdna5::kVWmmaF3216x16x32Bf16Vop3p));
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
template <typename IsCandidate>
inline bool safe_inline(const Instruction &inst, IsCandidate is_candidate) {
  if (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR |
                      BARRIER | WRITES_EXEC))
    return false;
  const auto name = inst.mnemonic();
  if (name == "s_delay_alu" || name == "s_nop" || name == "v_nop")
    return true;
  if (is_candidate(name) || ordinary_memory(name))
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

inline bool safe_inline(const Instruction &inst, unsigned min_k, unsigned mfma_bits) {
  return safe_inline(inst,
                     [=](std::string_view name) { return candidate(name, min_k, mfma_bits); });
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
