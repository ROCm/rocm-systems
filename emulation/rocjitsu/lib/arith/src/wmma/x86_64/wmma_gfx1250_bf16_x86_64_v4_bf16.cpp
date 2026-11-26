// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 (vdpbf16ps) pair-dot backends for loose gfx1250 BF16, FP8/BF8, and mixed f8f6f4
// WMMA. When any operand fails a VDPBF16PS range, subnormal, signed-zero, or nonfinite guard, an
// out-of-line AVX-512 FP32 FMA fallback reuses the already-loaded/decoded BF16 pairs directly so
// fallback waves avoid redundant operand loads and decodes.
#include "common/compiler.h"
#include "common/x86_64/bf16_broadcast.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_fp8_decode.h"
#include "wmma/x86_64/gfx1250_mixed_fast.h"
#include <immintrin.h>
#include <limits.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

using namespace RJ_ARITH_IMPL_NAMESPACE;

// Return a nonzero mask if any 16-bit BF16 element is subnormal, nonfinite, or outside
// biased exponent interval [96, 159] (zeros are accepted). Within [96, 159], every
// product lies in [2^-62, 2^64] and sums of <=32 products cannot underflow or overflow.
inline __mmask32 bf16_outside_safe_range(__m512i words) {
  const __m512i mag = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
  const __mmask32 nonzero = _mm512_test_epi16_mask(words, _mm512_set1_epi16(0x7fff));
  return _mm512_mask_cmpge_epu16_mask(nonzero, _mm512_sub_epi16(mag, _mm512_set1_epi16(96 << 7)),
                                      _mm512_set1_epi16(64 << 7));
}

// Return a nonzero mask if any FP32 lane is subnormal or -0.0f.
// VDPBF16PS flushes subnormal accumulators and can canonicalize -0.0f sums.
inline __mmask16 f32_needs_fallback(__m512i bits) {
  const __m512i mag = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  const __mmask16 subnormal = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(mag, _mm512_set1_epi32(1)),
                                                      _mm512_set1_epi32(0x7fffff));
  const __mmask16 neg_zero = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(INT32_MIN));
  return subnormal | neg_zero;
}

// Check if any packed byte in 16 dwords is nonfinite (0x7f for FP8, >=0x7c for BF8).
template <bool Bf8> inline __mmask64 fp8_has_nonfinite(__m512i words) {
  const __m512i mag = _mm512_and_si512(words, _mm512_set1_epi8(0x7f));
  if constexpr (Bf8) {
    return _mm512_cmpgt_epi8_mask(mag, _mm512_set1_epi8(0x7b));
  } else {
    return _mm512_cmpeq_epi8_mask(mag, _mm512_set1_epi8(0x7f));
  }
}

// Repair NaN elements of Bf16F32 from already-loaded rotated BF16 pairs (hi = even K, lo = odd K)
// using vectorized AVX-512 NaN-priority FMA across all 16 columns of each NaN-bearing row.
[[gnu::noinline]] void gfx1250_bf16f32_repair_from_pairs(const uint32_t left[16][16],
                                                         const uint32_t right[16][16],
                                                         const uint32_t *const *c,
                                                         uint64_t controls, float result[16][16]) {
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  const __m512i keep = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
  const __m512i negate = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
  const __m512 one = _mm512_set1_ps(1.0f);
  for (int row = 0; row < 16; ++row) {
    const __m512 acc_row = _mm512_load_ps(result[row]);
    const __mmask16 row_nan = _mm512_cmp_ps_mask(acc_row, acc_row, _CMP_UNORD_Q);
    if (!row_nan) {
      continue;
    }
    __m512 dot = _mm512_setzero_ps();
    for (int step = 0; step < 16; ++step) {
      const uint32_t lv = left[step][row];
      const __m512i rw = _mm512_load_si512(right[step]);
      dot = gfx1250_matrix_nan_fma_v4(_mm512_castsi512_ps(_mm512_set1_epi32(int(lv & 0xffff0000U))),
                                      _mm512_castsi512_ps(_mm512_and_si512(rw, hi_mask)), dot);
      dot = gfx1250_matrix_nan_fma_v4(_mm512_castsi512_ps(_mm512_set1_epi32(int(lv << 16))),
                                      _mm512_castsi512_ps(_mm512_slli_epi32(rw, 16)), dot);
    }
    const __m512i c_bits = _mm512_loadu_si512(c[row & 7] + 16 * (row >> 3));
    const __m512 init =
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(c_bits, keep), negate));
    const __m512 repaired = gfx1250_matrix_nan_fma_v4(dot, one, init);
    _mm512_store_ps(result[row], _mm512_mask_blend_ps(row_nan, acc_row, repaired));
  }
}

// Out-of-line AVX-512 FP32 FMA fallback for BF16 Half WMMA, reusing already-loaded rotated pairs
// (high 16 bits = even K, low 16 bits = odd K).
template <Wmma1250Form Form>
[[gnu::noinline]] void
gfx1250_half_bf16_fma_fallback(uint64_t flags, uint64_t controls, uint32_t *const *d,
                               const uint32_t left[16][16], const uint32_t right[16][16],
                               const uint32_t *const *c) {
  constexpr bool DotFirst = Form != Wmma1250Form::Bf16;
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  alignas(64) float left_f[32][16];
  for (int step = 0; step < 16; ++step) {
    const __m512i lw = _mm512_load_si512(left[step]);
    _mm512_store_ps(left_f[2 * step], _mm512_castsi512_ps(_mm512_and_si512(lw, hi_mask)));
    _mm512_store_ps(left_f[2 * step + 1], _mm512_castsi512_ps(_mm512_slli_epi32(lw, 16)));
  }

  __m512 acc[16];
  if constexpr (DotFirst) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_setzero_ps();
    }
  } else {
    gfx1250_load_acc_zmm<Form>(c, controls, acc);
  }

  for (int step = 0; step < 16; ++step) {
    const __m512i rw = _mm512_load_si512(right[step]);
    const __m512 col_even = _mm512_castsi512_ps(_mm512_and_si512(rw, hi_mask));
    const __m512 col_odd = _mm512_castsi512_ps(_mm512_slli_epi32(rw, 16));
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * step][row]), col_even, acc[row]);
    }
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * step + 1][row]), col_odd, acc[row]);
    }
  }

  if constexpr (DotFirst) {
    const __m512i keep = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
    const __m512i negate = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
    __mmask16 any_nan = 0;
    RJ_ARITH_UNROLL(8)
    for (int reg = 0; reg < 8; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        const __m512i bits = _mm512_loadu_si512(c[reg] + 16 * group);
        const __m512 init =
            _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(bits, keep), negate));
        acc[reg + 8 * group] = _mm512_add_ps(acc[reg + 8 * group], init);
        if constexpr (Form == Wmma1250Form::Bf16F32) {
          any_nan |= _mm512_cmp_ps_mask(acc[reg + 8 * group], acc[reg + 8 * group], _CMP_UNORD_Q);
        }
      }
    }
    if constexpr (Form == Wmma1250Form::Bf16F32) {
      if (any_nan) [[unlikely]] {
        alignas(64) float result[16][16];
        for (int row = 0; row < 16; ++row) {
          _mm512_store_ps(result[row], acc[row]);
        }
        gfx1250_bf16f32_repair_from_pairs(left, right, c, controls, result);
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_load_ps(result[row]);
        }
      }
    }
  }
  gfx1250_store_acc_zmm<Form>(flags, d, acc);
}

// Out-of-line AVX-512 FP32 FMA fallback for dense OCP FP8/BF8 WMMA, reusing already-decoded
// rotated BF16 pairs (which exactly preserve every FP8/BF8 subnormal, Inf, and signed QNaN).
template <int K, Wmma1250Form Form>
[[gnu::noinline]] void
gfx1250_fp8_pairs_fma_fallback(uint64_t flags, uint64_t controls, uint32_t *const *d,
                               const uint32_t left[K / 2][16], const uint32_t right[K / 2][16],
                               const uint32_t *const *c) {
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  alignas(64) float left_f[K][16];
  for (int step = 0; step < K / 2; ++step) {
    const __m512i lw = _mm512_load_si512(left[step]);
    _mm512_store_ps(left_f[2 * step], _mm512_castsi512_ps(_mm512_and_si512(lw, hi_mask)));
    _mm512_store_ps(left_f[2 * step + 1], _mm512_castsi512_ps(_mm512_slli_epi32(lw, 16)));
  }

  __m512 acc[16];
  gfx1250_load_acc_zmm<Form>(c, controls, acc);

  for (int step = 0; step < K / 2; ++step) {
    const __m512i rw = _mm512_load_si512(right[step]);
    const __m512 col_even = _mm512_castsi512_ps(_mm512_and_si512(rw, hi_mask));
    const __m512 col_odd = _mm512_castsi512_ps(_mm512_slli_epi32(rw, 16));
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * step][row]), col_even, acc[row]);
    }
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * step + 1][row]), col_odd, acc[row]);
    }
  }
  gfx1250_store_acc_zmm<Form>(flags, d, acc, flags & RJ_ARITH_FP16_OVFL);
}

// Out-of-line AVX-512 FP32 FMA fallback for mixed/scaled f8f6f4 WMMA, reusing already-decoded
// rotated BF16 pairs so nonfinite FP8/BF8 operands are never decoded twice.
template <int Block>
[[gnu::noinline]] void
gfx1250_mixed_pairs_fma_fallback(uint64_t controls, uint32_t *const *d, const uint32_t left[64][16],
                                 const uint32_t right[64][16], const uint32_t *const *c,
                                 const uint32_t *const *scale_a, const uint32_t *const *scale_b) {
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  alignas(64) float left_f[128][16];
  for (int p = 0; p < 64; ++p) {
    const __m512i lw = _mm512_load_si512(left[p]);
    _mm512_store_ps(left_f[2 * p], _mm512_castsi512_ps(_mm512_and_si512(lw, hi_mask)));
    _mm512_store_ps(left_f[2 * p + 1], _mm512_castsi512_ps(_mm512_slli_epi32(lw, 16)));
  }

  alignas(64) float result[16][16];
  gfx1250_mixed_load_c<16>(c, controls, result);

  if constexpr (Block == 0) {
    __m512 acc[16];
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_load_ps(result[row]);
    }
    for (int p = 0; p < 64; ++p) {
      const __m512i rw = _mm512_load_si512(right[p]);
      const __m512 col_even = _mm512_castsi512_ps(_mm512_and_si512(rw, hi_mask));
      const __m512 col_odd = _mm512_castsi512_ps(_mm512_slli_epi32(rw, 16));
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * p][row]), col_even, acc[row]);
      }
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * p + 1][row]), col_odd, acc[row]);
      }
    }
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      _mm512_store_ps(result[row], acc[row]);
    }
  } else {
    constexpr int groups = 128 / Block;
    constexpr int pairs_per_block = Block / 2;
    alignas(64) float sa[groups][16], sb[groups][16];
    const int asf = int((controls >> 8) & 3);
    const int bsf = int((controls >> 10) & 3);
    const int a_lane = 16 * int((controls >> 12) & 1);
    const int b_lane = 16 * int((controls >> 13) & 1);
    for (int blk = 0; blk < groups; ++blk) {
      gfx1250_decode_scales_16(scale_a, a_lane, blk, asf, sa[blk]);
      gfx1250_decode_scales_16(scale_b, b_lane, blk, bsf, sb[blk]);
    }
    for (int blk = 0; blk < groups; ++blk) {
      __m512 acc[16];
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] = _mm512_setzero_ps();
      }
      for (int p = blk * pairs_per_block; p < (blk + 1) * pairs_per_block; ++p) {
        const __m512i rw = _mm512_load_si512(right[p]);
        const __m512 col_even = _mm512_castsi512_ps(_mm512_and_si512(rw, hi_mask));
        const __m512 col_odd = _mm512_castsi512_ps(_mm512_slli_epi32(rw, 16));
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * p][row]), col_even, acc[row]);
        }
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left_f[2 * p + 1][row]), col_odd, acc[row]);
        }
      }
      const __m512 sbv = _mm512_load_ps(sb[blk]);
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        const __m512 scaled =
            _mm512_mul_ps(_mm512_mul_ps(acc[row], _mm512_set1_ps(sa[blk][row])), sbv);
        _mm512_store_ps(result[row], _mm512_add_ps(_mm512_load_ps(result[row]), scaled));
      }
    }
  }
  gfx1250_mixed_store_d<16>(d, result);
}

// Decode a 16-row gfx1250 f8f6f4 operand directly to K-paired BF16 (high half = even K,
// low half = odd K so VDPBF16PS preserves guest K order without extra rotations).
// Returns false if any FP8/BF8 element is nonfinite; FP6/BF6/FP4 are always finite.
bool gfx1250_mixed_decode_bf16_16(const uint32_t *const *input, int format, bool interleave_four,
                                  uint32_t out[64][16]) {
  if (format == RJ_ARITH_MATRIX_FORMAT_FP8 || format == RJ_ARITH_MATRIX_FORMAT_BF8) {
    const bool is_bf8 = format == RJ_ARITH_MATRIX_FORMAT_BF8;
    __mmask64 bad = 0;
    for (int k = 0; k < 128; k += 4) {
      const int reg = 4 * (k / 32) + (k % 16) / 4;
      const int lane = 16 * ((k / 16) % 2);
      const __m512i w = _mm512_loadu_si512(input[reg] + lane);
      if (is_bf8) {
        bad |= fp8_has_nonfinite<true>(w);
        _mm512_store_si512(
            out[k / 2],
            _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<true>(_mm512_cvtepi32_epi16(w)), 16));
        _mm512_store_si512(out[k / 2 + 1],
                           _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<true>(
                                                _mm512_cvtepi32_epi16(_mm512_srli_epi32(w, 16))),
                                            16));
      } else {
        bad |= fp8_has_nonfinite<false>(w);
        _mm512_store_si512(
            out[k / 2],
            _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(w)), 16));
        _mm512_store_si512(out[k / 2 + 1],
                           _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<false>(
                                                _mm512_cvtepi32_epi16(_mm512_srli_epi32(w, 16))),
                                            16));
      }
    }
    return bad == 0;
  }

  if (format == RJ_ARITH_MATRIX_FORMAT_FP4) {
    struct Fp4Bf16 {
      uint16_t bits[32]{};
      constexpr Fp4Bf16() {
        for (int i = 0; i < 16; ++i) {
          bits[i] = uint16_t(
              bit_cast<uint32_t>(matrix_decode_subbyte(i, RJ_ARITH_MATRIX_FORMAT_FP4)) >> 16);
        }
      }
    };
    alignas(64) static constexpr Fp4Bf16 table;
    const __m512i lut = _mm512_load_si512(table.bits);
    const __m512i mask_0f = _mm512_set1_epi32(0x0f);
    const __m512i mask_0f_hi = _mm512_set1_epi32(0x000f0000);
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int v = 0; v < 8; ++v) {
        __m512i w = _mm512_loadu_si512(input[v] + lane);
        const int p0 =
            (interleave_four ? 16 * v + 4 * hw : (v / 4) * 64 + 32 * hw + (v % 4) * 8) / 2;
        const int p1 = interleave_four ? p0 + 4 : p0 + 2;
        auto pair_decode = [&](__m512i cur) {
          const __m512i odd = _mm512_and_si512(_mm512_srli_epi32(cur, 4), mask_0f);
          const __m512i idx =
              _mm512_ternarylogic_epi32(odd, _mm512_slli_epi32(cur, 16), mask_0f_hi, 0xf8);
          return _mm512_permutexvar_epi16(idx, lut);
        };
        _mm512_store_si512(out[p0 + 0], pair_decode(w));
        w = _mm512_srli_epi32(w, 8);
        _mm512_store_si512(out[p0 + 1], pair_decode(w));
        w = _mm512_srli_epi32(w, 8);
        _mm512_store_si512(out[p1 + 0], pair_decode(w));
        w = _mm512_srli_epi32(w, 8);
        _mm512_store_si512(out[p1 + 1], pair_decode(w));
      }
    }
    return true;
  }

  auto decode_6 = [&]<int Fmt>() {
    alignas(64) static constexpr Gfx1250Mag6Table<Fmt> table;
    const __m512i lut = _mm512_load_si512(table.bits);
    const __m512i mask_3f = _mm512_set1_epi32(0x3f);
    const __m512i mask_3f_hi = _mm512_set1_epi32(0x003f0000);
    const __m512i sign_16 = _mm512_set1_epi32(int(0x80008000U));
    auto pair6 = [&](__m512i c_even, __m512i c_odd) {
      const __m512i idx = _mm512_ternarylogic_epi32(
          _mm512_and_si512(c_odd, mask_3f), _mm512_slli_epi32(c_even, 16), mask_3f_hi, 0xf8);
      const __m512i mag = _mm512_permutexvar_epi16(idx, lut);
      const __m512i sgn = _mm512_slli_epi16(idx, 10);
      return _mm512_ternarylogic_epi32(mag, sgn, sign_16, 0xf8);
    };
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int g = 0; g < 4; ++g) {
        const int vb = 3 * g;
        const __m512i w0 = _mm512_loadu_si512(input[vb] + lane);
        const __m512i w1 = _mm512_loadu_si512(input[vb + 1] + lane);
        const __m512i w2 = _mm512_loadu_si512(input[vb + 2] + lane);
        const int p0 =
            (interleave_four ? 32 * g + 4 * hw : (g / 2) * 64 + 32 * hw + (g % 2) * 16) / 2;
        const int pstride = interleave_four ? 4 : 2;
        const int p1 = p0 + pstride, p2 = p0 + 2 * pstride, p3 = p0 + 3 * pstride;
        _mm512_store_si512(out[p0 + 0], pair6(w0, _mm512_srli_epi32(w0, 6)));
        _mm512_store_si512(out[p0 + 1],
                           pair6(_mm512_srli_epi32(w0, 12), _mm512_srli_epi32(w0, 18)));
        _mm512_store_si512(out[p1 + 0], pair6(_mm512_srli_epi32(w0, 24),
                                              _mm512_or_si512(_mm512_srli_epi32(w0, 30),
                                                              _mm512_slli_epi32(w1, 2))));
        _mm512_store_si512(out[p1 + 1], pair6(_mm512_srli_epi32(w1, 4), _mm512_srli_epi32(w1, 10)));
        _mm512_store_si512(out[p2 + 0],
                           pair6(_mm512_srli_epi32(w1, 16), _mm512_srli_epi32(w1, 22)));
        _mm512_store_si512(
            out[p2 + 1], pair6(_mm512_or_si512(_mm512_srli_epi32(w1, 28), _mm512_slli_epi32(w2, 4)),
                               _mm512_srli_epi32(w2, 2)));
        _mm512_store_si512(out[p3 + 0], pair6(_mm512_srli_epi32(w2, 8), _mm512_srli_epi32(w2, 14)));
        _mm512_store_si512(out[p3 + 1],
                           pair6(_mm512_srli_epi32(w2, 20), _mm512_srli_epi32(w2, 26)));
      }
    }
  };
  if (format == RJ_ARITH_MATRIX_FORMAT_FP6) {
    decode_6.template operator()<RJ_ARITH_MATRIX_FORMAT_FP6>();
  } else {
    decode_6.template operator()<RJ_ARITH_MATRIX_FORMAT_BF6>();
  }
  return true;
}

} // namespace

// Loose gfx1250 16x16x32 BF16 WMMA using native VDPBF16PS directly on packed VGPR pairs.
template <Wmma1250Form Form>
void wmma_gfx1250_half_x86_64_v4_bf16(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c) {
  static_assert(Form == Wmma1250Form::F32Bf16 || Form == Wmma1250Form::Bf16 ||
                Form == Wmma1250Form::Bf16F32);
  constexpr bool DotFirst = Form != Wmma1250Form::Bf16;

  // Load A and B once, verifying safe BF16 exponent ranges before touching accumulators.
  alignas(64) uint32_t left[16][16], right[16][16];
  __mmask32 bad = 0;
  for (int step = 0; step < 16; ++step) {
    const int reg = 4 * (step / 8) + (step % 4);
    const int lane = 16 * ((step / 4) % 2);
    const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
    const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
    bad |= bf16_outside_safe_range(aw) | bf16_outside_safe_range(bw);
    // Rotate 16-bit pairs so vdpbf16ps evaluates the low K element before the high K element.
    _mm512_store_si512(left[step], _mm512_rol_epi32(aw, 16));
    _mm512_store_si512(right[step], _mm512_rol_epi32(bw, 16));
  }
  if (bad) [[unlikely]] {
    gfx1250_half_bf16_fma_fallback<Form>(flags, controls, d, left, right, c);
    return;
  }

  __m512 acc[16];
  if constexpr (DotFirst) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_setzero_ps();
    }
  } else if (!(controls & RJ_ARITH_WMMA_INLINE_C)) {
    // Check packed BF16 C directly at 16-bit width (8 ZMMs) while widening into acc[16].
    const __m512i keep16 =
        _mm512_set1_epi16((controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fff : static_cast<int16_t>(0xffff));
    const __m512i neg16 =
        _mm512_set1_epi16((controls & RJ_ARITH_WMMA_NEG_C) ? static_cast<int16_t>(0x8000) : 0);
    const __m512i mask7fff = _mm512_set1_epi16(0x7fff);
    const __m512i min_norm = _mm512_set1_epi16(0x0080);
    const __m512i neg_zero = _mm512_set1_epi16(static_cast<int16_t>(0x8000));
    const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
    __mmask32 bad_c = 0;
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < 4; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int g = 0; g < 2; ++g) {
        const __m512i raw = _mm512_loadu_si512(c[reg] + 16 * g);
        const __m512i w = _mm512_xor_si512(_mm512_and_si512(raw, keep16), neg16);
        const __m512i mag = _mm512_and_si512(w, mask7fff);
        const __mmask32 nz = _mm512_test_epi16_mask(w, mask7fff);
        bad_c |=
            _mm512_mask_cmplt_epu16_mask(nz, mag, min_norm) | _mm512_cmpeq_epi16_mask(w, neg_zero);
        acc[2 * reg + 8 * g] = _mm512_castsi512_ps(_mm512_slli_epi32(w, 16));
        acc[2 * reg + 1 + 8 * g] = _mm512_castsi512_ps(_mm512_and_si512(w, hi_mask));
      }
    }
    if (bad_c) [[unlikely]] {
      gfx1250_half_bf16_fma_fallback<Form>(flags, controls, d, left, right, c);
      return;
    }
  } else {
    gfx1250_load_acc_zmm<Form>(c, controls, acc);
    if (f32_needs_fallback(_mm512_castps_si512(acc[0]))) [[unlikely]] {
      gfx1250_half_bf16_fma_fallback<Form>(flags, controls, d, left, right, c);
      return;
    }
  }

  for (int step = 0; step < 16; ++step) {
    const __m512bh bv = bit_cast<__m512bh>(_mm512_load_si512(right[step]));
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      const __m512bh av = bit_cast<__m512bh>(broadcast_bf16_pair(&left[step][row]));
      acc[row] = _mm512_dpbf16_ps(acc[row], bv, av);
    }
  }

  if constexpr (DotFirst) {
    const __m512i keep = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
    const __m512i negate = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
    __mmask16 nan_c = 0;
    RJ_ARITH_UNROLL(8)
    for (int reg = 0; reg < 8; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        const __m512i bits = _mm512_loadu_si512(c[reg] + 16 * group);
        const __m512i init_bits = _mm512_xor_si512(_mm512_and_si512(bits, keep), negate);
        acc[reg + 8 * group] = _mm512_add_ps(acc[reg + 8 * group], _mm512_castsi512_ps(init_bits));
        if constexpr (Form == Wmma1250Form::Bf16F32) {
          nan_c |= _mm512_cmp_ps_mask(acc[reg + 8 * group], acc[reg + 8 * group], _CMP_UNORD_Q);
        }
      }
    }
    if constexpr (Form == Wmma1250Form::Bf16F32) {
      if (nan_c) [[unlikely]] {
        alignas(64) float result[16][16];
        for (int row = 0; row < 16; ++row) {
          _mm512_store_ps(result[row], acc[row]);
        }
        gfx1250_bf16f32_repair_from_pairs(left, right, c, controls, result);
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_load_ps(result[row]);
        }
      }
    }
  }
  gfx1250_store_acc_zmm<Form>(flags, d, acc);
}

// Loose gfx1250 OCP FP8/BF8 WMMA via 128-entry vpermi2w BF16 LUT + native VDPBF16PS.
// Every finite OCP FP8/BF8 value is exact in BF16 and every finite product is >= 2^-32,
// so finite inputs never underflow or overflow inside VDPBF16PS.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_x86_64_v4_bf16(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                     const uint32_t *const *a, const uint32_t *const *b,
                                     const uint32_t *const *c) {
  constexpr Wmma1250Form form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  // Decode A/B first while all 32 ZMMs are available for the vpermi2w LUT tables.
  alignas(64) uint32_t left[K / 2][16], right[K / 2][16];
  __mmask64 bad_ab = 0;
  for (int k = 0; k < K; k += 4) {
    const int reg = 4 * (k / 32) + (k % 16) / 4;
    const int lane = 16 * ((k / 16) % 2);
    const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
    const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
    bad_ab |= fp8_has_nonfinite<Bf8A>(aw) | fp8_has_nonfinite<Bf8B>(bw);
    _mm512_store_si512(
        left[k / 2],
        _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<Bf8A>(_mm512_cvtepi32_epi16(aw)), 16));
    _mm512_store_si512(
        left[k / 2 + 1],
        _mm512_rol_epi32(
            gfx1250_fp8_to_bf16_pairs<Bf8A>(_mm512_cvtepi32_epi16(_mm512_srli_epi32(aw, 16))), 16));
    _mm512_store_si512(
        right[k / 2],
        _mm512_rol_epi32(gfx1250_fp8_to_bf16_pairs<Bf8B>(_mm512_cvtepi32_epi16(bw)), 16));
    _mm512_store_si512(
        right[k / 2 + 1],
        _mm512_rol_epi32(
            gfx1250_fp8_to_bf16_pairs<Bf8B>(_mm512_cvtepi32_epi16(_mm512_srli_epi32(bw, 16))), 16));
  }
  if (bad_ab) [[unlikely]] {
    gfx1250_fp8_pairs_fma_fallback<K, form>(flags, controls, d, left, right, c);
    return;
  }

  __m512 acc[16];
  gfx1250_load_acc_zmm<form>(c, controls, acc);
  __mmask16 bad_c = 0;
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    bad_c |= f32_needs_fallback(_mm512_castps_si512(acc[row]));
  }
  if (bad_c) [[unlikely]] {
    gfx1250_fp8_pairs_fma_fallback<K, form>(flags, controls, d, left, right, c);
    return;
  }

  for (int step = 0; step < K / 2; ++step) {
    const __m512bh bv = bit_cast<__m512bh>(_mm512_load_si512(right[step]));
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      const __m512bh av = bit_cast<__m512bh>(broadcast_bf16_pair(&left[step][row]));
      acc[row] = _mm512_dpbf16_ps(acc[row], bv, av);
    }
  }
  gfx1250_store_acc_zmm<form>(flags, d, acc, flags & RJ_ARITH_FP16_OVFL);
}

// Loose gfx1250 16x16x128 mixed and block-scaled f8f6f4 WMMA via native VDPBF16PS.
// Every finite value in FP8, BF8, FP6, BF6, and FP4 is exact in BF16 and every finite dot
// stays >= 2^-32 and <= 4.21e11, avoiding VDPBF16PS subnormal/overflow hazards. For unscaled
// Block == 0, seeding each dot accumulator with -0.0f preserves -0.0f when C and all products are
// -0.0f without affecting any other sum (x + (-0.0f) == x). Scales and C (including
// subnormal/Inf/NaN) are applied in exact FP32.
template <int Block>
void wmma_gfx1250_mixed_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, const uint32_t *const *scale_a,
                                       const uint32_t *const *scale_b) {
  const int fa = int(controls & 7);
  const int fb = int((controls >> 3) & 7);
  const bool interleave_four = Block == 0 && fa >= 2 && fb >= 2;

  alignas(64) uint32_t left[64][16], right[64][16];
  const bool ok_a = gfx1250_mixed_decode_bf16_16(a, fa, interleave_four, left);
  const bool ok_b = gfx1250_mixed_decode_bf16_16(b, fb, interleave_four, right);
  if (!ok_a || !ok_b) [[unlikely]] {
    gfx1250_mixed_pairs_fma_fallback<Block>(controls, d, left, right, c, scale_a, scale_b);
    return;
  }

  if constexpr (Block == 0) {
    __m512 acc[16];
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_set1_ps(-0.0f);
    }
    for (int p = 0; p < 64; ++p) {
      const __m512bh bv = bit_cast<__m512bh>(_mm512_load_si512(right[p]));
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] =
            _mm512_dpbf16_ps(acc[row], bv, bit_cast<__m512bh>(broadcast_bf16_pair(&left[p][row])));
      }
    }
    const __m512i kv =
        _mm512_set1_epi32(int((controls & RJ_ARITH_WMMA_MIXED_ABS_C) ? 0x7fffffffU : 0xffffffffU));
    const __m512i nv =
        _mm512_set1_epi32(int((controls & RJ_ARITH_WMMA_MIXED_NEG_C) ? 0x80000000U : 0U));
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      const __m512 c_lo =
          _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c[r]), kv), nv));
      const __m512 c_hi = _mm512_castsi512_ps(
          _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c[r] + 16), kv), nv));
      acc[r] = _mm512_add_ps(c_lo, acc[r]);
      acc[8 + r] = _mm512_add_ps(c_hi, acc[8 + r]);
    }
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      _mm512_storeu_ps(d[r], acc[r]);
      _mm512_storeu_ps(d[r] + 16, acc[8 + r]);
    }
  } else {
    constexpr int groups = 128 / Block;
    constexpr int pairs_per_block = Block / 2;
    alignas(64) float result[16][16];
    gfx1250_mixed_load_c<16>(c, controls, result);

    alignas(64) float sa[groups][16], sb[groups][16];
    const int asf = int((controls >> 8) & 3);
    const int bsf = int((controls >> 10) & 3);
    const int a_lane = 16 * int((controls >> 12) & 1);
    const int b_lane = 16 * int((controls >> 13) & 1);
    for (int blk = 0; blk < groups; ++blk) {
      gfx1250_decode_scales_16(scale_a, a_lane, blk, asf, sa[blk]);
      gfx1250_decode_scales_16(scale_b, b_lane, blk, bsf, sb[blk]);
    }

    for (int blk = 0; blk < groups; ++blk) {
      __m512 acc[16];
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] = _mm512_setzero_ps();
      }
      for (int p = blk * pairs_per_block; p < (blk + 1) * pairs_per_block; ++p) {
        const __m512bh bv = bit_cast<__m512bh>(_mm512_load_si512(right[p]));
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_dpbf16_ps(acc[row], bv,
                                      bit_cast<__m512bh>(broadcast_bf16_pair(&left[p][row])));
        }
      }
      const __m512 sbv = _mm512_load_ps(sb[blk]);
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        const __m512 scaled =
            _mm512_mul_ps(_mm512_mul_ps(acc[row], _mm512_set1_ps(sa[blk][row])), sbv);
        _mm512_store_ps(result[row], _mm512_add_ps(_mm512_load_ps(result[row]), scaled));
      }
    }
    gfx1250_mixed_store_d<16>(d, result);
  }
}

template void wmma_gfx1250_half_x86_64_v4_bf16<Wmma1250Form::F32Bf16>(uint64_t, uint64_t,
                                                                      uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *);
template void wmma_gfx1250_half_x86_64_v4_bf16<Wmma1250Form::Bf16>(uint64_t, uint64_t,
                                                                   uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *);
template void wmma_gfx1250_half_x86_64_v4_bf16<Wmma1250Form::Bf16F32>(uint64_t, uint64_t,
                                                                      uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *,
                                                                      const uint32_t *const *);

#define INSTANTIATE_FP8_BF16(K, A, B, P)                                                           \
  template void wmma_gfx1250_fp8_x86_64_v4_bf16<K, A, B, P>(                                       \
      uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,     \
      const uint32_t *const *)
INSTANTIATE_FP8_BF16(64, false, false, false);
INSTANTIATE_FP8_BF16(64, false, false, true);
INSTANTIATE_FP8_BF16(64, false, true, false);
INSTANTIATE_FP8_BF16(64, false, true, true);
INSTANTIATE_FP8_BF16(64, true, false, false);
INSTANTIATE_FP8_BF16(64, true, false, true);
INSTANTIATE_FP8_BF16(64, true, true, false);
INSTANTIATE_FP8_BF16(64, true, true, true);
INSTANTIATE_FP8_BF16(128, false, false, false);
INSTANTIATE_FP8_BF16(128, false, false, true);
INSTANTIATE_FP8_BF16(128, false, true, false);
INSTANTIATE_FP8_BF16(128, false, true, true);
INSTANTIATE_FP8_BF16(128, true, false, false);
INSTANTIATE_FP8_BF16(128, true, false, true);
INSTANTIATE_FP8_BF16(128, true, true, false);
INSTANTIATE_FP8_BF16(128, true, true, true);
#undef INSTANTIATE_FP8_BF16

template void wmma_gfx1250_mixed_x86_64_v4_bf16<0>(uint64_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void
wmma_gfx1250_mixed_x86_64_v4_bf16<16>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *);
template void
wmma_gfx1250_mixed_x86_64_v4_bf16<32>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
