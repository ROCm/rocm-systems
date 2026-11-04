// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Select the precision-specific input policy for mixed FMA operands.

#pragma once

#include "common/isa.h"
#include "rj_arith/rj_arith.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Select input flushing independently for the source's decoded element type.
static inline bool mixed_fma_flush_input(uint64_t flags, uint32_t mode) {
  const uint64_t half_policy = flags & RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK;
  if ((mode & RJ_ARITH_MIX_F16_A) && half_policy) {
    return half_policy == RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16;
  }
  return flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
