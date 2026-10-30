// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD bit-exact FP16/BF16 DOT2 vector execution and masked result stores.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "dot/x86_64/exact_dot_x86_64.h"
#include "rj_arith/rj_arith.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot {

// DOT2 uses the same integer model as WMMA, with two products per lane.
// All reads for a SIMD block precede its masked store, including aliased C/D.
template <bool Bf16, bool Gfx11, typename V>
void dot2(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
          const uint32_t *b, const uint32_t *c) {
  constexpr int Width = sizeof(V) / sizeof(uint32_t);
  // Select the requested halfword and optionally invert its sign bit.
  auto half = [](V words, bool high, bool negate) {
    return bit_xor(bit_and(shift_right(words, high ? 16 : 0), constant<V>(0xffff)),
                   constant<V>(negate ? 0x8000 : 0));
  };
  for (int lane = 0; lane < 32; lane += Width) {
    auto av = load<V>(a + lane), bv = load<V>(b + lane);
    V left[2] = {half(av, instruction_flags & RJ_ARITH_DOT_LO_A_HIGH,
                      instruction_flags & RJ_ARITH_DOT_NEG_LO_A),
                 half(av, !(instruction_flags & RJ_ARITH_DOT_HI_A_LOW),
                      instruction_flags & RJ_ARITH_DOT_NEG_HI_A)};
    V right[2] = {half(bv, instruction_flags & RJ_ARITH_DOT_LO_B_HIGH,
                       instruction_flags & RJ_ARITH_DOT_NEG_LO_B),
                  half(bv, !(instruction_flags & RJ_ARITH_DOT_HI_B_LOW),
                       instruction_flags & RJ_ARITH_DOT_NEG_HI_B)};
    auto acc =
        bit_xor(bit_and(load<V>(c + lane),
                        constant<V>(instruction_flags & RJ_ARITH_DOT_ABS_C ? 0x7fffffff : -1)),
                constant<V>(instruction_flags & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0));
    auto result = dot<Bf16, Gfx11, false>(left, right, acc, false);

    // The inherited DOT2 model ignores CLAMP, unlike packed WMMA saturation.
    masked_store(d + lane, exec_mask >> lane, result);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot
