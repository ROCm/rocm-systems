// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar public-API evaluation for decoded conversion binding checks. Numerical
// correctness is tested independently in lib/arith/tests/conversion/.
#pragma once

#include "rj_arith/rj_arith.h"

#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::test {

using UnaryConversion = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                const uint32_t *const *, uint32_t *);

// Evaluate one lane using up to two source/destination words. The caller constructs
// flags independently of the VM translator, so forwarding errors remain observable.
inline uint64_t conversion_binding(UnaryConversion function, uint64_t flags, uint64_t modifiers,
                                   uint64_t raw) {
  uint32_t source[2][32]{}, result[2][32]{};
  source[0][0] = uint32_t(raw);
  source[1][0] = uint32_t(raw >> 32);
  const uint32_t *a[] = {source[0], source[1]};
  uint32_t *d[] = {result[0], result[1]};
  uint32_t status = 0;
  EXPECT_EQ(function(flags, 1, modifiers, d, a, &status), RJ_ARITH_SUCCESS);
  return result[0][0] | (uint64_t(result[1][0]) << 32);
}

} // namespace rocjitsu::test
