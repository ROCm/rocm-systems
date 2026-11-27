// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 DOT4 over 16 output columns using the dense empirical reduction.
#include "wmma/swmmac_exact16.h"
#include "wmma/x86_64/swmmac_exact16_x86_64.h"
#include "wmma/x86_64/wmma_dot_x86_64_v4.h"

namespace rocjitsu::arith {
namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half {

template <bool Bf16Param, bool Packed> struct HalfStage {
  using V = __m512i;
  using Mask = __mmask16;
  using Factor = wmma_v4::WmmaFactor;
  static constexpr bool Bf16 = Bf16Param;
  static constexpr int Inf = Bf16 ? 0x7f80 : 0x7c00;
  static constexpr int Bias = wmma_v4::kWmmaGridBias<Bf16, Packed>;

  static Factor factor(V bits, int bias) { return wmma_v4::wmma_factor<Bf16, false>(bits, bias); }
  static V pack_a_unit(V units) { return units; }
  static V initial(V word) { return wmma_v4::wmma_extend<Packed>(word); }
  static bool any(Mask mask) { return mask != 0; }

  static void unpack_c(V c_init, V &c_val, V &c_grid, Mask &cspec) {
    using namespace exact_dot;
    using wmma_v4::splat;
    constexpr int CFrac = Packed ? 7 : 23;
    __m512i ext = wmma_v4::wmma_extend<Packed>(c_init);
    __m512i ce = bit_and(shift_right(ext, CFrac), splat(255));
    __m512i cf = bit_and(ext, splat((1 << CFrac) - 1));
    __mmask16 normal = _mm512_test_epi32_mask(ext, splat(255 << CFrac));
    cspec = _mm512_cmpeq_epi32_mask(ce, splat(255));
    __mmask16 finite = __mmask16(~cspec);
    __mmask16 fnormal = __mmask16(normal & finite);
    __m512i cm = _mm512_maskz_mov_epi32(finite, cf);
    cm = _mm512_mask_or_epi32(cm, fnormal, cf, splat(1 << CFrac));
    __m512i cv = shift_left(cm, 26 - CFrac);
    __m512i cg = add(maximum(ce, splat(1)), splat(-153));
    c_val = _mm512_mask_sub_epi32(cv, _mm512_movepi32_mask(ext), splat(0), cv);
    c_grid = _mm512_mask_mov_epi32(splat(-152), finite, cg);
  }

  static void dot4_sum(uint32_t au0, int32_t ag0, V bu0, V bg0, uint32_t au1, int32_t ag1, V bu1,
                       V bg1, uint32_t au2, int32_t ag2, V bu2, V bg2, uint32_t au3, int32_t ag3,
                       V bu3, V bg3, V c_val, V c_grid, V &total, V &grid) {
    using namespace exact_dot;
    using wmma_v4::splat;
    __m512i g0 = add(splat(ag0), bg0);
    __m512i v0 = mul(splat(int(au0)), bu0);
    __m512i g1 = add(splat(ag1), bg1);
    __m512i v1 = mul(splat(int(au1)), bu1);
    __m512i pg0 = maximum(g0, g1);
    __m512i p0 = add(_mm512_srav_epi32(v0, sub(pg0, g0)), _mm512_srav_epi32(v1, sub(pg0, g1)));

    __m512i g2 = add(splat(ag2), bg2);
    __m512i v2 = mul(splat(int(au2)), bu2);
    __m512i g3 = add(splat(ag3), bg3);
    __m512i v3 = mul(splat(int(au3)), bu3);
    __m512i pg1 = maximum(g2, g3);
    __m512i p1 = add(_mm512_srav_epi32(v2, sub(pg1, g2)), _mm512_srav_epi32(v3, sub(pg1, g3)));

    grid = maximum(maximum(pg0, pg1), c_grid);
    __m512i cshift = sub(grid, c_grid);
    total = _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(cshift, splat(27)), c_val, cshift);
    total = add(total, _mm512_srav_epi32(p0, sub(grid, pg0)));
    total = add(total, _mm512_srav_epi32(p1, sub(grid, pg1)));
  }

  static void next_unpacked(V total, V grid, V &c_val, V &c_grid, Mask &bad_acc) {
    using namespace exact_dot;
    using wmma_v4::splat;
    constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    constexpr int FracMaskShl3 = Packed ? (0x7f0000 << 3) : (0x7fffff << 3);
    __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
    __m512i exp = bit_and(shift_right(rounded, 23), splat(255));
    __m512i ce = add(exp, grid);
    __mmask16 nz = _mm512_test_epi32_mask(total, total);
    bad_acc |= _mm512_mask_cmpge_epu32_mask(nz, sub(ce, splat(1)), splat(254));
    __m512i r3 = shift_left(rounded, 3);
    __m512i cv = _mm512_maskz_ternarylogic_epi32(nz, r3, splat(FracMaskShl3), splat(1 << 26), 0xea);
    c_val = _mm512_mask_sub_epi32(cv, _mm512_movepi32_mask(total), splat(0), cv);
    c_grid = _mm512_mask_add_epi32(splat(-152), nz, ce, splat(-153));
  }

  static V pack_final(V total, V grid, Mask &bad_acc) {
    using namespace exact_dot;
    using wmma_v4::splat;
    constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    constexpr int Sign = int(0x80000000u);
    __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
    __m512i mag = bit_and(rounded, splat(0x7fffffff));
    __m512i exp = add(shift_right(mag, 23), grid);
    __mmask16 keep = _mm512_test_epi32_mask(total, total);
    bad_acc |= _mm512_mask_cmple_epi32_mask(keep, exp, splat(0));
    __m512i bits = _mm512_min_epu32(add(mag, shift_left(grid, 23)), splat(0x7f800000));
    bits = _mm512_maskz_ternarylogic_epi32(keep, bits, rounded, splat(Sign), 0xf8);
    if constexpr (Packed) {
      bits = shift_right(bits, 16);
    }
    return bits;
  }

  static void apply_simple_specials(uint32_t mode, int col, uint32_t c_special_rows,
                                    uint32_t a_nan_rows, const Mask (&c_special_mask)[16],
                                    const uint32_t *const *dest, V (&acc)[16]) {
    using namespace exact_dot;
    using wmma_v4::splat;
    constexpr int CInf = Packed ? Inf : 0x7f800000;
    constexpr int CQuiet = Packed ? (Bf16 ? 0x0040 : 0x0200) : int(gfx12_dot_detail::kQuietNanBit);
    const int FactorNan = Packed ? int(packed16::special<Bf16>(gfx12_dot_detail::kFactorNan))
                                 : int(gfx12_dot_detail::kFactorNan);
    for (uint32_t m = c_special_rows; m; m &= m - 1) {
      int row = __builtin_ctz(m);
      V cv = load_c_row<V, Packed>(mode, col, row, dest);
      __mmask16 cnan =
          _mm512_cmpgt_epi32_mask(bit_and(cv, splat(Packed ? 0x7fff : 0x7fffffff)), splat(CInf));
      __m512i resolved = _mm512_mask_or_epi32(cv, cnan, cv, splat(CQuiet));
      acc[row] = _mm512_mask_mov_epi32(acc[row], c_special_mask[row], resolved);
    }
    for (uint32_t m = a_nan_rows; m; m &= m - 1) {
      acc[__builtin_ctz(m)] = splat(FactorNan);
    }
  }

  static V finite(const Factor (&left)[4], const Factor (&right)[4], V acc, bool saturate,
                  Mask &special) {
    return wmma_v4::wmma_dot<Bf16, false, Packed>(left, right, acc, saturate, special);
  }

  static V repair(const Factor (&left)[4], const Factor (&right)[4], V acc, V result,
                  Mask special) {
    auto factor_max = selected_max<V>(left, right);
    special |= _mm512_cmpge_epi32_mask(factor_max, wmma_v4::splat(Inf));
    return _mm512_mask_mov_epi32(
        result, special, wmma_v4::wmma_special<Bf16, false, Packed>(left, right, factor_max, acc));
  }
};

} // namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half

template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index) {
  sparse_half::tile<sparse_half::HalfStage<Bf16, Packed>, WaveSize, Packed>(mode, saturate, dest, a,
                                                                            b, index);
}

template void swmmac_exact16_x86_64_v4<false, 32, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 32, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 64, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 64, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 32, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 32, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 64, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 64, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
