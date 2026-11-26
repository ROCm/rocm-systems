// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 DOT4 over eight output columns using the dense empirical reduction.
#include "wmma/swmmac_exact16.h"
#include "wmma/x86_64/swmmac_exact16_x86_64.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

namespace rocjitsu::arith {
namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half {

template <bool Bf16Param, bool Packed> struct HalfStage {
  using V = __m256i;
  using Mask = V;
  using Factor = wmma_v3::Factor;
  static constexpr bool Bf16 = Bf16Param;
  static constexpr int Inf = Bf16 ? 0x7f80 : 0x7c00;
  static constexpr int Bias = wmma_v3::kGridBias<Bf16, Packed>;

  static Factor factor(V bits, int bias) { return wmma_v3::decode<Bf16, false>(bits, bias); }
  // Zero-extend signed 16-bit Q12 units so VPMADDWD computes exact 32-bit products.
  static V pack_a_unit(V units) { return exact_dot::bit_and(units, wmma_v3::splat(0xffff)); }
  static V initial(V word) { return word; }
  static bool any(Mask mask) { return exact_dot::lane_mask(mask) != 0; }

  static void unpack_c(V c_init, V &c_val, V &c_grid, Mask &cspec) {
    using namespace exact_dot;
    using wmma_v3::splat;
    constexpr int CFrac = Packed ? 7 : 23;
    __m256i ce = bit_and(shift_right(c_init, CFrac), splat(255));
    __m256i cf = shift_left(bit_and(c_init, splat((1 << CFrac) - 1)), 26 - CFrac);
    cspec = equal(ce, splat(255));
    __m256i v_fin = andnot(cspec, c_init);
    __m256i ce_fin = andnot(cspec, ce);
    __m256i cf_fin = andnot(cspec, cf);
    __m256i cv = bit_or(cf_fin, _mm256_sign_epi32(splat(1 << 26), ce_fin));
    __m256i vs = Packed ? shift_left(v_fin, 16) : v_fin;
    c_val = _mm256_sign_epi32(cv, vs);
    c_grid = sub(maximum(ce_fin, splat(1)), splat(153));
  }

  static void dot4_sum(uint32_t au0, int32_t ag0, V bu0, V bg0, uint32_t au1, int32_t ag1, V bu1,
                       V bg1, uint32_t au2, int32_t ag2, V bu2, V bg2, uint32_t au3, int32_t ag3,
                       V bu3, V bg3, V c_val, V c_grid, V &total, V &grid) {
    using namespace exact_dot;
    using wmma_v3::splat;
    __m256i g0 = add(splat(ag0), bg0);
    __m256i v0 = _mm256_madd_epi16(splat(int(au0)), bu0);
    __m256i g1 = add(splat(ag1), bg1);
    __m256i v1 = _mm256_madd_epi16(splat(int(au1)), bu1);
    __m256i pg0 = maximum(g0, g1);
    __m256i p0 = add(_mm256_srav_epi32(v0, sub(pg0, g0)), _mm256_srav_epi32(v1, sub(pg0, g1)));

    __m256i g2 = add(splat(ag2), bg2);
    __m256i v2 = _mm256_madd_epi16(splat(int(au2)), bu2);
    __m256i g3 = add(splat(ag3), bg3);
    __m256i v3 = _mm256_madd_epi16(splat(int(au3)), bu3);
    __m256i pg1 = maximum(g2, g3);
    __m256i p1 = add(_mm256_srav_epi32(v2, sub(pg1, g2)), _mm256_srav_epi32(v3, sub(pg1, g3)));

    grid = maximum(maximum(pg0, pg1), c_grid);
    __m256i cshift = sub(grid, c_grid);
    total = bit_and(greater(splat(27), cshift), _mm256_srav_epi32(c_val, cshift));
    total = add(total, _mm256_srav_epi32(p0, sub(grid, pg0)));
    total = add(total, _mm256_srav_epi32(p1, sub(grid, pg1)));
  }

  static void next_unpacked(V total, V grid, V &c_val, V &c_grid, Mask &bad_acc) {
    using namespace exact_dot;
    using wmma_v3::splat;
    constexpr int FracMaskShl3 = Packed ? (0x7f0000 << 3) : (0x7fffff << 3);
    __m256i z_mask = equal(total, splat(0));
    V ce, raw_cv;
    if constexpr (!Packed) {
      __m256i rounded = _mm256_castps_si256(_mm256_cvtepi32_ps(total));
      __m256i mag = bit_and(rounded, splat(0x7fffffff));
      ce = add(shift_right(mag, 23), grid);
      raw_cv = bit_or(bit_and(shift_left(rounded, 3), splat(FracMaskShl3)), splat(1 << 26));
    } else {
      __m256i magnitude = absolute(total);
      __m256i top = wmma_v3::biased_top_bit(magnitude);
      __m256i exponent = add(top, grid);
      V normalized = shift_left(magnitude, sub(splat(127 + 30), top));
      V parity = bit_and(shift_right(normalized, 7), splat(1));
      V rounded = shift_right(add(normalized, add(splat(63), parity)), 7);
      ce = add(exponent, shift_right(rounded, 24));
      raw_cv = bit_or(bit_and(shift_left(rounded, 3), splat(FracMaskShl3)), splat(1 << 26));
    }
    bad_acc =
        bit_or(bad_acc, andnot(z_mask, bit_or(greater(splat(1), ce), greater(ce, splat(254)))));
    c_val = _mm256_sign_epi32(raw_cv, total);
    c_grid = choose(z_mask, splat(-152), sub(ce, splat(153)));
  }

  static V pack_final(V total, V grid, Mask &bad_acc) {
    using namespace exact_dot;
    using wmma_v3::splat;
    constexpr int Sign = int(0x80000000U);
    __m256i z_mask = equal(total, splat(0));
    V exp, bits, sign_src;
    if constexpr (!Packed) {
      __m256i rounded = _mm256_castps_si256(_mm256_cvtepi32_ps(total));
      __m256i mag = bit_and(rounded, splat(0x7fffffff));
      exp = add(shift_right(mag, 23), grid);
      bits = _mm256_min_epu32(add(mag, shift_left(grid, 23)), splat(0x7f800000));
      sign_src = rounded;
    } else {
      __m256i magnitude = absolute(total);
      __m256i top = wmma_v3::biased_top_bit(magnitude);
      __m256i exponent = add(top, grid);
      V normalized = shift_left(magnitude, sub(splat(127 + 30), top));
      V parity = bit_and(shift_right(normalized, 7), splat(1));
      V rounded = shift_right(add(normalized, add(splat(63), parity)), 7);
      exp = add(exponent, shift_right(rounded, 24));
      V raw = add(shift_left(sub(exponent, splat(1)), 23), rounded);
      bits = _mm256_min_epu32(raw, splat(0x7f800000));
      sign_src = total;
    }
    bad_acc = bit_or(bad_acc, andnot(z_mask, greater(splat(1), exp)));
    __m256i res = andnot(z_mask, bit_or(bits, bit_and(sign_src, splat(Sign))));
    if constexpr (Packed) {
      res = shift_right(res, 16);
    }
    return res;
  }

  static void apply_simple_specials(uint32_t mode, int col, uint32_t c_special_rows,
                                    uint32_t a_nan_rows, const Mask (&c_special_mask)[16],
                                    const uint32_t *const *dest, V (&acc)[16]) {
    using namespace exact_dot;
    using wmma_v3::splat;
    constexpr int CInf = Packed ? Inf : 0x7f800000;
    constexpr int CQuiet = Packed ? (Bf16 ? 0x0040 : 0x0200) : int(gfx12_dot_detail::kQuietNanBit);
    const int FactorNan = Packed ? int(packed16::special<Bf16>(gfx12_dot_detail::kFactorNan))
                                 : int(gfx12_dot_detail::kFactorNan);
    for (uint32_t m = c_special_rows; m; m &= m - 1) {
      int row = __builtin_ctz(m);
      V cv = load_c_row<V, Packed>(mode, col, row, dest);
      __m256i cnan = greater(bit_and(cv, splat(Packed ? 0x7fff : 0x7fffffff)), splat(CInf));
      __m256i resolved = bit_or(cv, bit_and(cnan, splat(CQuiet)));
      acc[row] = choose(c_special_mask[row], resolved, acc[row]);
    }
    for (uint32_t m = a_nan_rows; m; m &= m - 1) {
      acc[__builtin_ctz(m)] = splat(FactorNan);
    }
  }

  static V finite(const Factor (&left)[4], const Factor (&right)[4], V acc, bool saturate,
                  Mask &special) {
    return wmma_v3::stage<Bf16, false, Packed>(left, right, acc, saturate, special);
  }

  static V repair(const Factor (&left)[4], const Factor (&right)[4], V acc, V result,
                  Mask special) {
    using namespace exact_dot;
    auto factor_max = selected_max<V>(left, right);
    special = bit_or(special, greater(factor_max, wmma_v3::splat(Inf - 1)));
    return choose(special, wmma_v3::special<Bf16, false, Packed>(left, right, factor_max, acc),
                  result);
  }
};

} // namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half

template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v3(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index) {
  sparse_half::tile<sparse_half::HalfStage<Bf16, Packed>, WaveSize, Packed>(mode, saturate, dest, a,
                                                                            b, index);
}

template void swmmac_exact16_x86_64_v3<false, 32, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 32, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 64, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 64, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 32, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 32, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 64, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 64, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
