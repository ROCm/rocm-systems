// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_compare_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

namespace rocjitsu::amdgpu {

/// Evaluate the predicate and INVALID from the same selected source words.
template <typename Fmt, typename Rel, typename Commit>
void execute_compare_with_exceptions(Wavefront &wave, const Operand &lhs, const Operand &rhs,
                                     uint32_t opsel, uint32_t absolute, uint32_t negate,
                                     uint64_t active, bool signal_invalid, bool signaling,
                                     Commit commit) {
  constexpr bool constant =
      std::is_same_v<Rel, comparison::F> || std::is_same_v<Rel, comparison::T>;
  if (constant && !signal_invalid) {
    commit(std::is_same_v<Rel, comparison::T> ? active : 0);
    return;
  }
  const uint64_t selections =
      ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) | ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);
  const auto policy = input_denormal::Policy::make(
      Fmt::kWidth == 32 ? wave.fp_denorm_mode_f32() : wave.fp_denorm_mode_f16_f64());
  uint64_t result = 0;
  bool invalid = false;
  arith::with_compare_inputs<Fmt::kWidth>(
      wave, active, lhs, rhs, false, selections,
      [&](const uint32_t *const *a, const uint32_t *const *b) {
        if constexpr (util::has_stdx_simd) {
          if (active && !util::force_scalar()) {
            detail::record_simd_fast_path(SimdFastPath::VOPC_COMPARE_STATUS);
            using Lane = typename Fmt::Lane;
            constexpr auto width = util::native_width_v<Lane>;
            const uint64_t full = util::mask<uint64_t>(width);
            for (unsigned base = 0; base < wave.wf_size(); base += width) {
              const uint64_t lanes = (active >> base) & full;
              if (!lanes)
                continue;
              const auto read = [&](const uint32_t *const *words, unsigned index) {
                if constexpr (Fmt::kWidth == 64)
                  return util::load64<Lane>(words[0] + base, words[1] + base);
                else {
                  auto bits = util::load<Lane>(words[0] + base);
                  if constexpr (Fmt::kWidth == 16)
                    bits >>= (opsel & (1u << index)) ? 16 : 0;
                  return bits;
                }
              };
              const auto x = read(a, 0);
              const auto y = read(b, 1);
              const auto compare_bits = [&](auto predicate) -> uint64_t {
                if constexpr (Fmt::kWidth == 64)
                  return cmp_bits64<Lane>(x, y, predicate);
                else
                  return util::simd_mask_to_bits(predicate(x, y));
              };
              if (signal_invalid && !invalid) {
                const auto raises = [&](auto value) {
                  const auto nan = comparison::is_nan<Fmt>(value);
                  return signaling ? nan : nan && ((value & Fmt::kQuiet) == 0u);
                };
                const auto exceptions =
                    compare_bits([&](auto lhs, auto rhs) { return raises(lhs) || raises(rhs); });
                invalid = (exceptions & lanes) != 0;
              }
              if constexpr (!constant) {
                const auto predicate = compare_bits([&](auto lhs, auto rhs) {
                  return comparison::evaluate<Fmt, Rel>(lhs, rhs, absolute, negate, policy);
                });
                result |= (predicate & lanes) << base;
              }
            }
            if constexpr (constant)
              result = std::is_same_v<Rel, comparison::T> ? active : 0;
            return RJ_ARITH_SUCCESS;
          }
        }
        for (uint64_t lanes = active; lanes; lanes &= lanes - 1) {
          const unsigned lane = std::countr_zero(lanes);
          const auto read = [&](const uint32_t *const *src, unsigned index) -> typename Fmt::Lane {
            if constexpr (Fmt::kWidth == 64)
              return static_cast<uint64_t>(src[1][lane]) << 32 | src[0][lane];
            if constexpr (Fmt::kWidth == 16)
              return (src[0][lane] >> (((opsel >> index) & 1u) * 16)) & 0xffffu;
            return src[0][lane];
          };
          const auto x = read(a, 0), y = read(b, 1);
          const auto raises_invalid = [signaling](typename Fmt::Lane value) {
            return comparison::is_nan<Fmt>(value) && (signaling || !(value & Fmt::kQuiet));
          };
          invalid |= raises_invalid(x) || raises_invalid(y);
          if (comparison::evaluate<Fmt, Rel>(x, y, absolute, negate, policy))
            result |= uint64_t{1} << lane;
        }
        return RJ_ARITH_SUCCESS;
      });
  if (wave.instruction_execution_failed())
    return;
  commit(result);
  if (signal_invalid && invalid && !wave.instruction_execution_failed()) {
    wave.set_trapsts(wave.trapsts() | RJ_ARITH_EXCEPTION_INVALID);
    wave.raise_alu_causes(RJ_ARITH_EXCEPTION_INVALID);
  }
}
} // namespace rocjitsu::amdgpu
