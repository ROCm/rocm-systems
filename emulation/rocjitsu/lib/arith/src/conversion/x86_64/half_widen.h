// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native FP16 widening with the inherited guest modifiers and NaN policy.
#pragma once

#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd {

// Widen selected halfwords exactly, including subnormals and signaling-NaN payloads.
inline Lanes<32> half_to_float(Lanes<32> raw, uint64_t flags, uint32_t mode, bool quiet_nan) {
  using V = Lanes<32>;
  namespace model = conversion_model;
  namespace fmt = model::fp_format;
  auto policy = conversion_detail::float_policy<fmt::F16, fmt::F32>(flags, mode, quiet_nan);
  raw = (raw >> ((mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0)) & V(0xffffU);
  raw = model::source_modifier::apply<fmt::F16>(raw, 0, policy.abs, policy.neg);
  raw = model::input_denormal::flush_input<fmt::F16>(raw, policy.input);
#if defined(__AVX512F__)
  V result(_mm512_castps_si512(_mm512_cvtph_ps(_mm512_cvtepi32_epi16(raw.bits))));
#else
  // Pack the low halfword of each lane without changing its bit pattern.
  auto packed =
      _mm_packus_epi32(_mm256_castsi256_si128(raw.bits), _mm256_extracti128_si256(raw.bits, 1));
  V result(_mm256_castps_si256(_mm256_cvtph_ps(packed)));
#endif
  if (!quiet_nan) {
    // Native widening quiets sNaNs; restore the source's signaling bit when requested.
    auto signaling = (raw & V(0x7fffU)) > V(0x7c00U) && (raw & V(0x200U)) == V(0U);
    result = model::lane::choose(signaling, result & V(~0x400000U), result);
  }
  return model::output_modifier::apply<fmt::F32>(result, policy.output);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd
