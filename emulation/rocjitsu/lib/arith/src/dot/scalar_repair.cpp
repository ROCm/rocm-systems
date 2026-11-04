// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compile exceptional WMMA dot products with baseline CPU settings.

#include "dot/scalar_repair.h"
#include "dot/dot.h"
#include "dot/gfx11_dot2.h"

namespace rocjitsu::arith {
namespace {

// Select the architecture's dot stage without exposing scalar templates to SIMD.
template <bool Bf16, bool Packed>
uint32_t dot(const uint16_t *a, const uint16_t *b, uint32_t c, bool rdna3, bool saturate,
             int count) {
  if (rdna3) {
    return gfx11_dot2_bits<Bf16, Packed>(a[0], b[0], a[1], b[1], c, saturate);
  }
  if (count == 2) {
    return gfx12_dot_bits<Bf16, 2, Packed>({a[0], a[1]}, {b[0], b[1]}, c, saturate);
  }
  return gfx12_dot_bits<Bf16, 4, Packed>({a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}, c,
                                         saturate);
}

} // namespace

// Return the scalar model's result for an exceptional vector lane.
[[gnu::noinline]] uint32_t repair_wmma_dot(const uint16_t *a, const uint16_t *b, uint32_t c,
                                           bool bf16, bool rdna3, bool packed, bool saturate,
                                           int count) {
  if (packed) {
    return bf16 ? dot<true, true>(a, b, c, rdna3, saturate, count)
                : dot<false, true>(a, b, c, rdna3, saturate, count);
  }
  return bf16 ? dot<true, false>(a, b, c, rdna3, saturate, count)
              : dot<false, false>(a, b, c, rdna3, saturate, count);
}

} // namespace rocjitsu::arith
