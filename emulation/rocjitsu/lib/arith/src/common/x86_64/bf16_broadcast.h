// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/isa.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Clang 17 crashes when lowering a scalar splat cast to a BF16 vector.
[[gnu::always_inline]] inline __m512i broadcast_bf16_pair(const uint32_t *pair) {
#if defined(__clang__) && __clang_major__ <= 17
  return _mm512_broadcastd_epi32(_mm_loadu_si32(pair));
#else
  return _mm512_set1_epi32(static_cast<int>(*pair));
#endif
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
