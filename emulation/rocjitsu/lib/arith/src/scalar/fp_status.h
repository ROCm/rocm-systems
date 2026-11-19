// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar-specific input/status policy layered on inherited FMA classification.

#pragma once

#include "common/isa.h"
#include "fp/compatibility_fma_status.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Classify one scalar input using the inherited guest flushing policy.
template <bool Half> inline uint32_t scalar_input_causes(uint32_t bits, uint64_t flags) {
  return compatibility_input_causes<Half>(bits, flags);
}

// Scalar operations suppress input-denormal causes for a NaN result; dual MUL does not.
template <bool Half>
inline uint32_t scalar_fma_causes(uint64_t flags, uint32_t a, uint32_t b, uint32_t c,
                                  uint32_t result) {
  uint32_t causes = compatibility_fma_causes<Half>(flags, a, b, c, result);
  constexpr uint32_t magnitude = Half ? 0x7fff : 0x7fffffff;
  constexpr uint32_t infinity = Half ? 0x7c00 : 0x7f800000;
  return (result & magnitude) > infinity ? causes & ~RJ_ARITH_EXCEPTION_INPUT_DENORM : causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
