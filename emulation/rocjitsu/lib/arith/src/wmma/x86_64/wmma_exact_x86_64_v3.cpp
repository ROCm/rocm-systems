// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Eight-column integer-grid WMMA for the AVX2 backend.
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v3 {

// Decoded A fields are broadcast by row inside the eight-column stage.
struct alignas(32) FactorRows {
  int bits[16], units[16], grid[16], sign[16];
};

// Advance the sixteen rows by one DOT2 or DOT4 at a time. Factors are decoded
// once for the stage. Each eight-column slice shares the decoded A rows.
template <bool Bf16, bool Rdna3, bool Packed>
void accumulate_tile(uint32_t flags, bool saturate, __m256i (&acc)[2][16],
                     const uint32_t *const (&a)[8], const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int N = Rdna3 ? 2 : 4;
  for (int step = 0; step < 16; step += N) {
    FactorRows left[N];
    Factor right_slices[2][N];
    __m256i column_maxima[2] = {splat(0), splat(0)};
    __m256i a_max[2] = {splat(0), splat(0)};
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < N; ++j) {
      int pair = (step + j) / 2, shift = 16 * (j % 2);
      auto negate_a =
          splat((flags & (j % 2 ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A)) ? 0x8000 : 0);
      auto negate_b =
          splat((flags & (j % 2 ? RJ_ARITH_WMMA_NEG_HI_B : RJ_ARITH_WMMA_NEG_LO_B)) ? 0x8000 : 0);
      for (int half = 0; half < 2; ++half) {
        auto bits =
            bit_xor(bit_and(shift_right(load<__m256i>(b[pair] + 8 * half), shift), splat(0xffff)),
                    negate_b);
        right_slices[half][j] = decode<Bf16, Rdna3>(bits, kGridBias<Bf16, Packed>);
        column_maxima[half] = maximum(column_maxima[half], bit_and(bits, splat(0x7fff)));
      }
      for (int k = 0; k < 2; ++k) {
        auto v = bit_xor(bit_and(shift_right(load<__m256i>(a[pair] + 8 * k), shift), splat(0xffff)),
                         negate_a);
        auto f = decode<Bf16, Rdna3>(v);
        store(left[j].bits + 8 * k, v);
        store(left[j].units + 8 * k, f.units);
        store(left[j].grid + 8 * k, f.grid);
        store(left[j].sign + 8 * k, f.sign);
        a_max[k] = maximum(a_max[k], bit_and(v, splat(0x7fff)));
      }
    }
    alignas(32) int row_max[16];
    store(row_max, a_max[0]);
    store(row_max + 8, a_max[1]);
    for (int half = 0; half < 2; ++half) {
      const auto &right = right_slices[half];
      auto b_max = column_maxima[half];
      constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
      int exceptional_rows = lane_mask(greater(a_max[0], splat(Infinity - 1))) |
                             (lane_mask(greater(a_max[1], splat(Infinity - 1))) << 8);
      bool exceptional_columns = lane_mask(greater(b_max, splat(Infinity - 1))) != 0;
      for (int row = 0; row < 16; ++row) {
        Factor factors[N];
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < N; ++j) {
          const auto &f = left[j];
          factors[j] = {splat(0), splat(f.units[row]), splat(f.grid[row]),
                        splat(Rdna3 ? f.sign[row] : 0)};
        }
        auto old = acc[half][row];
        __m256i c_exception;
        auto result = stage<Bf16, Rdna3, Packed>(factors, right, old, saturate, c_exception);
        if (exceptional_columns || ((exceptional_rows >> row) & 1) || lane_mask(c_exception))
            [[unlikely]] {
          RJ_ARITH_UNROLL(4)
          for (int j = 0; j < N; ++j) {
            factors[j].bits = splat(left[j].bits[row]);
          }
          auto factor_max = maximum(b_max, splat(row_max[row]));
          auto mask = bit_or(c_exception, greater(factor_max, splat(Infinity - 1)));
          result =
              choose(mask, special<Bf16, Rdna3, Packed>(factors, right, factor_max, old), result);
        }
        acc[half][row] = result;
      }
    }
  }
}
} // namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v3

// Map validated wave operands into the two eight-column slices. The staged
// destination preserves overlapping inputs and the unused half of a gfx11 D.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v3(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  using namespace exact_dot;
  using namespace wmma_v3;
  constexpr int Registers = 256 / Lanes,
                DestinationRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  alignas(32) uint32_t result[DestinationRegisters][Lanes] = {};
  auto layout = [](int row, int &reg, int &lane) {
    reg = Rdna3 ? row / (Lanes / 16) : row % Registers;
    lane = Rdna3 ? 16 * (row % (Lanes / 16))
                 : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  };
  int high = instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
  // gfx11 requires each 16-lane group to replicate its factors, so group
  // zero supplies the whole tile. gfx12 Wave32 visits K blocks 0, 8, 4, 12.
  const uint32_t *left[8], *right[8];
  for (int pair = 0; pair < 8; ++pair) {
    int k = !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * pair) : 2 * pair;
    int reg = Rdna3 ? k / 2 : (k % Registers) / 2, lane = Rdna3 ? 0 : 16 * (k / Registers);
    left[pair] = a[reg] + lane;
    right[pair] = b[reg] + lane;
  }
  __m256i acc[2][16];
  for (int col = 0; col < 16; col += 8) {
    for (int row = 0; row < 16; ++row) {
      int reg, lane;
      layout(row, reg, lane);
      int dest = Packed && !Rdna3 ? reg / 2 : reg;
      auto v = load<__m256i>(c[dest] + lane + col);
      if constexpr (Packed) {
        v = bit_and(shift_right(v, splat(Rdna3 ? high : 16 * (reg % 2))), splat(0xffff));
      }
      if (instruction_flags & RJ_ARITH_WMMA_ABS_C) {
        v = bit_and(v, splat(Packed ? 0x7fff : 0x7fffffff));
      }
      if (instruction_flags & RJ_ARITH_WMMA_NEG_C) {
        v = bit_xor(v, splat(Packed ? 0x8000 : int(0x80000000U)));
      }
      acc[col / 8][row] = v;
    }
  }
  accumulate_tile<Bf16, Rdna3, Packed>(instruction_flags, saturate, acc, left, right);
  for (int col = 0; col < 16; col += 8) {
    for (int row = 0; row < 16; ++row) {
      int reg, lane;
      layout(row, reg, lane);
      int dest = Packed && !Rdna3 ? reg / 2 : reg;
      auto v = acc[col / 8][row];
      if constexpr (Packed) {
        int shift = Rdna3 ? high : 16 * (reg % 2);
        auto old = load<__m256i>((Rdna3 ? d[dest] : result[dest]) + lane + col);
        v = bit_or(shift_left(v, splat(shift)), andnot(splat(int(0xffffU << shift)), old));
      }
      store(result[dest] + lane + col, v);
    }
  }
  for (int reg = 0; reg < DestinationRegisters; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 8) {
      store(d[reg] + lane, load<__m256i>(result[reg] + lane));
    }
  }
}

template void wmma_exact_x86_64_v3<false, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
