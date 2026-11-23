// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 dense FP16/BF16 WMMA. Each DOT2 or DOT4 stage reuses its decoded
// A rows and B columns across the full 16x16 tile.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_dot_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v4 {

// Scalar-addressable fields let each row broadcast its decoded A factor.
struct alignas(64) FactorRows {
  int bits[16], units[16], grid[16], sign[16];
};

// Accumulate the sixteen columns of each row, rounding after each DOT stage.
// Wave layouts have already resolved the eight packed K pairs.
template <bool Bf16, bool Rdna3, bool Packed>
void accumulate_tile(uint32_t instruction_flags, bool saturate, __m512i (&acc)[16],
                     const uint32_t *const (&a)[8], const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int Step = Rdna3 ? 2 : 4;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  for (int row = 0; row < 16; ++row) {
    acc[row] = wmma_extend<Packed>(acc[row]);
  }
  for (int step = 0; step < 16; step += Step) {
    __m512i bmax = splat(0), amax = splat(0);
    WmmaFactor right_factors[Step];
    FactorRows left_rows[Step];
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < Step; ++j) {
      int pair = (step + j) / 2;
      auto right = bit_and(shift_right(load<__m512i>(b[pair]), 16 * (j % 2)), splat(0xffff));
      auto negate_b = j % 2 ? RJ_ARITH_WMMA_NEG_HI_B : RJ_ARITH_WMMA_NEG_LO_B;
      right = bit_xor(right, splat(instruction_flags & negate_b ? 0x8000 : 0));
      right_factors[j] = wmma_factor<Bf16, Rdna3>(right, kWmmaGridBias<Bf16, Packed>);
      auto left = bit_and(shift_right(load<__m512i>(a[pair]), 16 * (j % 2)), splat(0xffff));
      auto negate_a = j % 2 ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A;
      left = bit_xor(left, splat(instruction_flags & negate_a ? 0x8000 : 0));
      auto fields = wmma_factor<Bf16, Rdna3>(left);
      _mm512_store_si512(left_rows[j].units, fields.units);
      _mm512_store_si512(left_rows[j].grid, fields.grid);
      _mm512_store_si512(left_rows[j].bits, fields.bits);
      _mm512_store_si512(left_rows[j].sign, fields.sign);
      amax = maximum(amax, bit_and(left, splat(0x7fff)));
      bmax = maximum(bmax, bit_and(right, splat(0x7fff)));
    }
    // Infinite and NaN factors are classified once per column and once per row.
    __mmask16 right_special = _mm512_cmpge_epi32_mask(bmax, splat(Infinity));
    __mmask16 left_special = _mm512_cmpge_epi32_mask(amax, splat(Infinity));
    alignas(64) int row_max[16];
    _mm512_store_si512(row_max, amax);
    for (int row = 0; row < 16; ++row) {
      WmmaFactor left_factors[Step];
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < Step; ++j) {
        const auto &a_row = left_rows[j];
        left_factors[j] = {splat(0), splat(a_row.units[row]), splat(a_row.grid[row]),
                           splat(Rdna3 ? a_row.sign[row] : 0)};
      }
      __mmask16 special;
      auto result =
          wmma_dot<Bf16, Rdna3, Packed>(left_factors, right_factors, acc[row], saturate, special);
      special = __mmask16(special | right_special | (left_special >> row & 1 ? 0xffff : 0));
      if (special) [[unlikely]] {
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < Step; ++j) {
          left_factors[j].bits = splat(left_rows[j].bits[row]);
        }
        result = _mm512_mask_mov_epi32(
            result, special,
            wmma_special<Bf16, Rdna3, Packed>(left_factors, right_factors,
                                              maximum(bmax, splat(row_max[row])), acc[row]));
      }
      acc[row] = result;
    }
  }
  if constexpr (Packed) {
    for (int row = 0; row < 16; ++row) {
      acc[row] = bit_and(acc[row], splat(0xffff));
    }
  }
}

} // namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v4

// Compute bit-exact A*B+C with validated controls. D may alias any input.
// gfx11 packed results preserve the unselected half of each destination.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  using namespace exact_dot;
  using namespace wmma_v4;
  constexpr int Registers = 256 / Lanes,
                DestinationRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  alignas(64) uint32_t result[DestinationRegisters][Lanes] = {};
  const int high = instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
  auto layout = [](int row, int &reg, int &lane) {
    reg = Rdna3 ? row / (Lanes / 16) : row % Registers;
    lane = Rdna3 ? 16 * (row % (Lanes / 16))
                 : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  };
  __m512i acc[16];
  for (int row = 0; row < 16; ++row) {
    int reg, lane;
    layout(row, reg, lane);
    int dest_reg = Packed && !Rdna3 ? reg / 2 : reg;
    auto v = load<__m512i>(c[dest_reg] + lane);
    if constexpr (Packed) {
      int shift = Rdna3 ? high : 16 * (reg % 2);
      v = bit_and(shift_right(v, splat(shift)), splat(0xffff));
    }
    if (instruction_flags & RJ_ARITH_WMMA_ABS_C) {
      v = bit_and(v, splat(Packed ? 0x7fff : 0x7fffffff));
    }
    if (instruction_flags & RJ_ARITH_WMMA_NEG_C) {
      v = bit_xor(v, splat(Packed ? 0x8000 : int(0x80000000U)));
    }
    acc[row] = v;
  }
  // Resolve each packed K pair once. gfx11 replicates A/B in each 16-lane
  // group. gfx12 Wave32 visits K blocks 0,8,4,12, and Wave64 visits 0,4,8,12.
  const uint32_t *left[8], *right[8];
  for (int pair = 0; pair < 8; ++pair) {
    int k = !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * pair) : 2 * pair;
    int reg = Rdna3 ? k / 2 : (k % Registers) / 2;
    int lane = Rdna3 ? 0 : 16 * (k / Registers);
    left[pair] = a[reg] + lane;
    right[pair] = b[reg] + lane;
  }
  accumulate_tile<Bf16, Rdna3, Packed>(instruction_flags, saturate, acc, left, right);
  for (int row = 0; row < 16; ++row) {
    int reg, lane;
    layout(row, reg, lane);
    int dest_reg = Packed && !Rdna3 ? reg / 2 : reg;
    auto v = acc[row];
    if constexpr (Packed) {
      int shift = Rdna3 ? high : 16 * (reg % 2);
      auto old = load<__m512i>((Rdna3 ? d[dest_reg] : result[dest_reg]) + lane);
      v = bit_or(shift_left(v, splat(shift)), andnot(splat(int(0xffffu << shift)), old));
    }
    store(result[dest_reg] + lane, v);
  }
  // Defer destination writes until all source and preserved-half reads are complete.
  for (int reg = 0; reg < DestinationRegisters; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 16) {
      store(d[reg] + lane, load<__m512i>(result[reg] + lane));
    }
  }
}

template void wmma_exact_x86_64_v4<false, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
