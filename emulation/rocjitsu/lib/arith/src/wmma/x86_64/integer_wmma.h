// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared dense INT8/INT4 WMMA algorithm for AVX2 and AVX-512 VNNI backends.
// Owns gfx11/gfx12 register layouts, signedness, and saturation stage ordering;
// the Ops parameter supplies vector operations without changing those rules.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Compute an integer WMMA tile with validated modifiers and the selected
// gfx11/gfx12 layout. Clamp selects staged signed saturation; otherwise D
// wraps modulo 2^32. D may overlap A, B, or C.
template <int Bits, int K, bool Clamp, class Ops, int Lanes = 32, bool Rdna3 = false>
void integer_wmma_impl(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c) {
  using V = typename Ops::V;
  static_assert(!Rdna3 || K == 16);
  constexpr int Groups = Rdna3 ? Lanes / 16 : 1;
  const int sign_a = (modifiers & RJ_ARITH_WMMA_SIGNED_A) ? 1 << (Bits - 1) : 0;
  const int sign_b = (modifiers & RJ_ARITH_WMMA_SIGNED_B) ? 1 << (Bits - 1) : 0;
  // Map logical K to a packed factor word, lane-group offset, and bit shift.
  const auto location = [](int k, int &reg, int &lane, int &shift) {
    int local = Rdna3 ? k : k % 8 + 8 * (k / 16);
    reg = local * Bits / 32;
    lane = Rdna3 ? 0 : 16 * ((k / 8) % 2);
    shift = local * Bits % 32;
  };
  // Widen factors to signed 16-bit pairs, including unsigned bytes (0..255).
  // Byte-pair saturation would lose information before GPU stage saturation.
  uint32_t packed[Groups][2][K / 2][16];
  for (int group = 0; group < Groups; ++group) {
    for (int operand = 0; operand < 2; ++operand) {
      const uint32_t *const *v = operand ? b : a;
      int sign = operand ? sign_b : sign_a;
      for (int k = 0; k < K; k += 2) {
        int reg, lane, shift;
        location(k, reg, lane, shift);
        for (int index = 0; index < 16; index += Ops::width) {
          V words = Ops::load(v[reg] + 16 * group + lane + index);
          // Extract and sign- or zero-extend one factor into each 32-bit lane.
          const auto extend = [&](int offset) {
            V value = Ops::bit_and(Ops::shift_right(words, offset), Ops::splat((1 << Bits) - 1));
            return Ops::sub(Ops::bit_xor(value, Ops::splat(sign)), Ops::splat(sign));
          };
          V pair = Ops::bit_or(Ops::bit_and(extend(shift), Ops::splat(65535)),
                               Ops::shift_left16(extend(shift + Bits)));
          Ops::store(packed[group][operand][k / 2] + index, pair);
        }
      }
    }
  }

  uint32_t result[256 / Lanes][Lanes];
  constexpr bool clamp = Clamp;
  // RDNA3 clamps adjacent pairs (IU8) or quartets (IU4); RDNA4 has
  // two stages, with the physical product order selected below.
  constexpr int stages = !Clamp ? 1 : Rdna3 ? K / (Bits == 8 ? 2 : 4) : 2;
  for (int row = 0; row < 16; ++row) {
    // RDNA3 interleaves output rows across replicated sixteen-lane groups.
    int group = Rdna3 ? row % Groups : 0;
    int reg = Rdna3 ? row / Groups : row % 8;
    int lane = Rdna3 ? 16 * group : 16 * (row / 8);
    V acc[16 / Ops::width];
    for (int chunk = 0; chunk < 16 / Ops::width; ++chunk) {
      acc[chunk] = Ops::load(c[reg] + lane + chunk * Ops::width);
    }
    for (int stage = 0; stage < stages; ++stage) {
      V sums[16 / Ops::width];
      for (auto &sum : sums) {
        sum = Ops::splat(0);
      }
      for (int local = 0; local < K / stages; local += 2) {
        int k = Rdna3   ? stage * (K / stages) + local
                : clamp ? 16 * (local / 8) + 8 * stage + local % 8
                        : local;
        V left = Ops::splat_bits(packed[group][0][k / 2][row]);
        for (int chunk = 0; chunk < 16 / Ops::width; ++chunk) {
          sums[chunk] =
              Ops::dot(sums[chunk], left, Ops::load(packed[group][1][k / 2] + chunk * Ops::width));
        }
      }
      for (int chunk = 0; chunk < 16 / Ops::width; ++chunk) {
        V sum = Ops::add(acc[chunk], sums[chunk]);
        // Each stage's dot fits int32. Saturate its addition to the current
        // accumulator before starting the next hardware stage.
        if (clamp) {
          V overflow = Ops::bit_and(Ops::bit_xor(acc[chunk], sum), Ops::bit_xor(sums[chunk], sum));
          V limit = Ops::bit_xor(Ops::sign(acc[chunk]), Ops::splat(INT32_MAX));
          sum = Ops::select_negative(overflow, limit, sum);
        }
        acc[chunk] = sum;
      }
    }
    for (int chunk = 0; chunk < 16 / Ops::width; ++chunk) {
      Ops::store(result[reg] + lane + chunk * Ops::width, acc[chunk]);
    }
  }
  // All source reads precede writes, including when D shares A/B/C VGPRs.
  for (int reg = 0; reg < 256 / Lanes; ++reg) {
    for (int lane = 0; lane < Lanes; lane += Ops::width) {
      Ops::store(d[reg] + lane, Ops::load(result[reg] + lane));
    }
  }
}

// Compute integer A*B+C with the CLAMP mode in validated modifiers.
// D may alias inputs; signedness and layout follow the template parameters.
template <int Bits, int K, class Ops, int Lanes = 32, bool Rdna3 = false>
void integer_wmma(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                  const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers & RJ_ARITH_WMMA_CLAMP) {
    integer_wmma_impl<Bits, K, true, Ops, Lanes, Rdna3>(modifiers, d, a, b, c);
  } else {
    integer_wmma_impl<Bits, K, false, Ops, Lanes, Rdna3>(modifiers, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
