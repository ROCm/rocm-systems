// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compile exceptional WMMA dot products with baseline CPU settings.

#include "dot/scalar_repair.h"
#include "dot/dot.h"
#include "dot/gfx11_dot2.h"

namespace rocjitsu::arith {
namespace {

// Select the architecture's dot stage without exposing scalar templates to SIMD.
template <bool Bf16> uint32_t dot(const uint16_t *a, const uint16_t *b, uint32_t c, bool rdna3) {
  if (rdna3) {
    return gfx11_dot2_f32<Bf16>(a[0], b[0], a[1], b[1], c);
  }
  return gfx12_dot_bits<Bf16, 4>({a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}, c);
}

} // namespace

// Return the scalar model's result for an exceptional vector lane.
[[gnu::noinline]] uint32_t repair_wmma_dot(const uint16_t *a, const uint16_t *b, uint32_t c,
                                           bool bf16, bool rdna3) {
  return bf16 ? dot<true>(a, b, c, rdna3) : dot<false>(a, b, c, rdna3);
}

} // namespace rocjitsu::arith
