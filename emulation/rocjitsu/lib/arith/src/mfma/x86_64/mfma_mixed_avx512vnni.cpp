// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 VNNI (vpdpbusd) exact integer-dot backend for CDNA4 unscaled (loose) and scaled
// FP4/FP6 MFMA.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/mixed.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>
#include <immintrin.h>

#if !defined(RJ_ARITH_TARGET_AVX512VNNI)
#error "mfma_mixed_avx512vnni.cpp requires AVX-512 VNNI and AVX-512 BW"
#endif

namespace rocjitsu::arith {

namespace {

// Signed integer values of FP6 (E2M3, bias 1) scaled by 8: exact integers in [-60, +60].
// BiasedA adds +60 to produce unsigned bytes in [0, 120] for vpdpbusd (u8 * s8 -> i32).
template <bool BiasedA> struct Fp6Int8Table {
  uint16_t entries[64]{};
  constexpr Fp6Int8Table() {
    for (int code = 0; code < 64; ++code) {
      const int sign = (code & 32) ? -1 : 1;
      const int mag = code & 31;
      const int exp = mag >> 3;
      const int frac = mag & 7;
      const int val = sign * (exp == 0 ? frac : ((8 + frac) << (exp - 1)));
      const int shifted = BiasedA ? (val + 60) : (val & 0xff);
      entries[code] = uint16_t(shifted & 0xff);
    }
  }
};

// Decode one 32-element K-block of FP4 (4 VGPRs) into 8 K-quad vectors [8][M].
// Each group of 8 K-elements is stored as quad 0 = (k+0, k+2, k+4, k+6) and
// quad 1 = (k+1, k+3, k+5, k+7), requiring zero byte shuffling for FP4.
template <bool BiasedA, int M>
void decode_fp4_vnni_block(const uint32_t *const *input, int base_lane, uint32_t (&out)[8][M]) {
  const __m512i base_tbl = _mm512_broadcast_i32x4(
      _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
  const __m512i tbl = BiasedA ? _mm512_add_epi8(base_tbl, _mm512_set1_epi8(12)) : base_tbl;
  const __m512i mask_nib = _mm512_set1_epi32(0x0f0f0f0f);
  for (int row = 0; row < M; row += 16) {
    const int lane = base_lane + row;
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < 4; ++reg) {
      const __m512i w = _mm512_loadu_si512(input[reg] + lane);
      const __m512i lo_nib = _mm512_and_si512(w, mask_nib);
      const __m512i hi_nib = _mm512_and_si512(_mm512_srli_epi32(w, 4), mask_nib);
      _mm512_store_si512(out[2 * reg + 0] + row, _mm512_shuffle_epi8(tbl, lo_nib));
      _mm512_store_si512(out[2 * reg + 1] + row, _mm512_shuffle_epi8(tbl, hi_nib));
    }
  }
}

// Decode one 32-element K-block of FP6 (6 VGPRs) into 8 K-quad vectors [8][M] using the same
// (k+0, k+2, k+4, k+6) / (k+1, k+3, k+5, k+7) quad ordering as FP4, so all four FP4/FP6
// combinations share the identical vpdpbusd microkernel.
template <bool BiasedA, int M>
void decode_fp6_vnni_block(const uint32_t *const *input, int base_lane, uint32_t (&out)[8][M]) {
  alignas(64) static constexpr Fp6Int8Table<BiasedA> table;
  const __m512i t0 = _mm512_load_si512(table.entries + 0);
  const __m512i t1 = _mm512_load_si512(table.entries + 32);
  const __m512i m63 = _mm512_set1_epi32(63);
  for (int row = 0; row < M; row += 16) {
    const int lane = base_lane + row;
    __m512i w[6];
    for (int r = 0; r < 6; ++r) {
      w[r] = _mm512_loadu_si512(input[r] + lane);
    }
    auto ext6 = [&](int j) {
      const int bit = j * 6;
      const int r = bit / 32;
      const int shift = bit % 32;
      __m512i code = _mm512_srli_epi32(w[r], shift);
      if (shift > 26) {
        code = _mm512_or_si512(code, _mm512_slli_epi32(w[r + 1], 32 - shift));
      }
      return _mm512_and_si512(code, m63);
    };
    auto lookup_pair = [&](__m512i a_idx, __m512i b_idx) {
      const __m512i idx = _mm512_or_si512(a_idx, _mm512_slli_epi32(b_idx, 16));
      return _mm512_permutex2var_epi16(t0, idx, t1);
    };
    const __m512i mask_lo8 = _mm512_set1_epi32(0x00ff00ff);
    for (int g = 0; g < 4; ++g) {
      const int k0 = 8 * g;
      const __m512i e04 = lookup_pair(ext6(k0 + 0), ext6(k0 + 4));
      const __m512i e26 = lookup_pair(ext6(k0 + 2), ext6(k0 + 6));
      const __m512i eq =
          _mm512_or_si512(_mm512_and_si512(e04, mask_lo8), _mm512_slli_epi16(e26, 8));
      const __m512i o15 = lookup_pair(ext6(k0 + 1), ext6(k0 + 5));
      const __m512i o37 = lookup_pair(ext6(k0 + 3), ext6(k0 + 7));
      const __m512i oq =
          _mm512_or_si512(_mm512_and_si512(o15, mask_lo8), _mm512_slli_epi16(o37, 8));
      _mm512_store_si512(out[2 * g + 0] + row, eq);
      _mm512_store_si512(out[2 * g + 1] + row, oq);
    }
  }
}

// Evaluate unscaled (explicit loose) or block-scaled FP4/FP6 combinations with vpdpbusd tiling.
// Even for full K=128 FP6xFP6, |int_sum| <= 128*60*60 = 460800 < 2^19 << 2^24, so the entire
// integer dot converts to FP32 with zero rounding error.
template <int M, int K, bool Fp6A, bool Fp6B, bool Scaled>
void mfma_avx512vnni_impl(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c,
                          const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow) {
  constexpr int blocks = K / 32;
  constexpr int registers = M * M / 64;
  constexpr int col_vecs = M / 16;
  constexpr int bias_a = Fp6A ? 60 : 12;
  constexpr int extra_shift = (Fp6A ? -3 : -1) + (Fp6B ? -3 : -1);

  alignas(64) uint32_t ea[64], eb[64];
  bool plain_scales = false;
  if constexpr (Scaled) {
    plain_scales = mfma_extract_e8m0_scales(controls, scale_a, scale_b, ea, eb, extra_shift);
  }
  const uint32_t abs_mask =
      (Scaled && (controls & RJ_ARITH_MFMA_SCALE_ABS_C)) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t neg_mask = (Scaled && (controls & RJ_ARITH_MFMA_SCALE_NEG_C)) ? 0x80000000U : 0U;

  // Stage all blocks' encoded integer quads before any D writes (arbitrary aliasing safe).
  alignas(64) uint32_t left[blocks][8][M], right[blocks][8][M];
  __m512i neg_bias[blocks][col_vecs];
  const __m512i bias_vec = _mm512_set1_epi8(char(bias_a));
  for (int block = 0; block < blocks; ++block) {
    if constexpr (Fp6A) {
      decode_fp6_vnni_block<true, M>(a, block * M, left[block]);
    } else {
      decode_fp4_vnni_block<true, M>(a, block * M, left[block]);
    }
    if constexpr (Fp6B) {
      decode_fp6_vnni_block<false, M>(b, block * M, right[block]);
    } else {
      decode_fp4_vnni_block<false, M>(b, block * M, right[block]);
    }
    for (int p = 0; p < col_vecs; ++p) {
      __m512i bsum = _mm512_setzero_si512();
      RJ_ARITH_UNROLL(8)
      for (int q = 0; q < 8; ++q) {
        bsum = _mm512_dpbusd_epi32(bsum, bias_vec, _mm512_load_si512(right[block][q] + 16 * p));
      }
      neg_bias[block][p] = _mm512_sub_epi32(_mm512_setzero_si512(), bsum);
    }
  }

  const bool direct = mfma_direct_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];
  const __m512i f32_abs = _mm512_set1_epi32(0x7fffffff);
  const __m512i f32_inf = _mm512_set1_epi32(0x7f800000);

  if constexpr (!Scaled) {
    // Unscaled (explicit loose): accumulate all K blocks in 16 ZMM int32 registers (16x1 or 8x2),
    // then convert and add C once.
    constexpr int row_tile = M == 16 ? 16 : 8;
    __m512i total_bias[col_vecs];
    for (int p = 0; p < col_vecs; ++p) {
      total_bias[p] = neg_bias[0][p];
      for (int block = 1; block < blocks; ++block) {
        total_bias[p] = _mm512_add_epi32(total_bias[p], neg_bias[block][p]);
      }
    }
    const __m512 unit_scale = _mm512_castsi512_ps(_mm512_set1_epi32((127 + extra_shift) << 23));
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512i sum[row_tile][col_vecs];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        for (int p = 0; p < col_vecs; ++p) {
          sum[r][p] = total_bias[p];
        }
      }
      for (int block = 0; block < blocks; ++block) {
        RJ_ARITH_UNROLL(8)
        for (int q = 0; q < 8; ++q) {
          __m512i b_vec[col_vecs];
          for (int p = 0; p < col_vecs; ++p) {
            b_vec[p] = _mm512_load_si512(right[block][q] + 16 * p);
          }
          RJ_ARITH_UNROLL(16)
          for (int r = 0; r < row_tile; ++r) {
            const __m512i a_bcast = _mm512_set1_epi32(int(left[block][q][rb + r]));
            for (int p = 0; p < col_vecs; ++p) {
              sum[r][p] = _mm512_dpbusd_epi32(sum[r][p], a_bcast, b_vec[p]);
            }
          }
        }
      }
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          const __m512 cv =
              _mm512_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane + 16 * p));
          const __m512 val = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum[r][p]), unit_scale, cv);
          if (direct) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), val);
          } else {
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, val);
          }
        }
      }
    }
  } else {
    // Scaled: keep both acc and block sum live in 16 ZMM registers (8x1 for M=16, 4x2 for M=32)
    // to avoid per-block accumulator spills.
    constexpr int row_tile = M == 16 ? 8 : 4;
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512 acc[row_tile][col_vecs];
      __mmask16 c_finite[row_tile][col_vecs];
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] = mfma_mixed_load_c(c[out.reg] + out.lane + 16 * p, abs_mask, neg_mask);
          c_finite[r][p] =
              clamp_overflow
                  ? _mm512_cmplt_epu32_mask(
                        _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf)
                  : __mmask16(0);
        }
      }
      for (int block = 0; block < blocks; ++block) {
        __m512i sum[row_tile][col_vecs];
        for (int r = 0; r < row_tile; ++r) {
          for (int p = 0; p < col_vecs; ++p) {
            sum[r][p] = neg_bias[block][p];
          }
        }
        RJ_ARITH_UNROLL(8)
        for (int q = 0; q < 8; ++q) {
          __m512i b_vec[col_vecs];
          for (int p = 0; p < col_vecs; ++p) {
            b_vec[p] = _mm512_load_si512(right[block][q] + 16 * p);
          }
          RJ_ARITH_UNROLL(8)
          for (int r = 0; r < row_tile; ++r) {
            const __m512i a_bcast = _mm512_set1_epi32(int(left[block][q][rb + r]));
            for (int p = 0; p < col_vecs; ++p) {
              sum[r][p] = _mm512_dpbusd_epi32(sum[r][p], a_bcast, b_vec[p]);
            }
          }
        }
        for (int r = 0; r < row_tile; ++r) {
          const uint32_t a_exp = ea[M * block + rb + r];
          for (int p = 0; p < col_vecs; ++p) {
            const __m512 fdots = _mm512_cvtepi32_ps(sum[r][p]);
            __m512 scaled;
            if (__builtin_expect(plain_scales, 1)) {
              const __m512i eb_vec = _mm512_load_si512(eb + M * block + 16 * p);
              const __m512i exp =
                  _mm512_add_epi32(eb_vec, _mm512_set1_epi32(int(a_exp) - 127 + extra_shift));
              scaled = _mm512_mul_ps(fdots, _mm512_castsi512_ps(_mm512_slli_epi32(exp, 23)));
            } else {
              scaled = mfma_scale_nonplain_vec(fdots, a_exp, eb + M * block + 16 * p, extra_shift);
            }
            acc[r][p] = _mm512_add_ps(acc[r][p], scaled);
          }
        }
      }
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          if (clamp_overflow) {
            const __mmask16 inf_m = _mm512_cmpeq_epi32_mask(
                _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf);
            acc[r][p] = _mm512_castsi512_ps(
                _mm512_mask_sub_epi32(_mm512_castps_si512(acc[r][p]), c_finite[r][p] & inf_m,
                                      _mm512_castps_si512(acc[r][p]), _mm512_set1_epi32(1)));
          }
          if (direct) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), acc[r][p]);
          } else {
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, acc[r][p]);
          }
        }
      }
    }
  }
  if (!direct) {
    for (int reg = 0; reg < registers; ++reg) {
      for (int lane = 0; lane < 64; lane += 16) {
        _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                         _mm512_load_ps(result[reg] + lane));
      }
    }
  }
}

template <int M, int K, bool Scaled>
bool mfma_avx512vnni_dispatch(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              const uint32_t *scale_a, const uint32_t *scale_b,
                              bool clamp_overflow) {
  const uint64_t fa = controls & 7;
  const uint64_t fb = (controls >> RJ_ARITH_MATRIX_FORMAT_B_SHIFT) & 7;
  const bool a_ok = (fa == RJ_ARITH_MATRIX_FORMAT_FP4 || fa == RJ_ARITH_MATRIX_FORMAT_FP6);
  const bool b_ok = (fb == RJ_ARITH_MATRIX_FORMAT_FP4 || fb == RJ_ARITH_MATRIX_FORMAT_FP6);
  if (!a_ok || !b_ok) {
    return false;
  }
  if (fa == RJ_ARITH_MATRIX_FORMAT_FP4 && fb == RJ_ARITH_MATRIX_FORMAT_FP4) {
    mfma_avx512vnni_impl<M, K, false, false, Scaled>(controls, d, a, b, c, scale_a, scale_b,
                                                     clamp_overflow);
  } else if (fa == RJ_ARITH_MATRIX_FORMAT_FP4) {
    mfma_avx512vnni_impl<M, K, false, true, Scaled>(controls, d, a, b, c, scale_a, scale_b,
                                                    clamp_overflow);
  } else if (fb == RJ_ARITH_MATRIX_FORMAT_FP4) {
    mfma_avx512vnni_impl<M, K, true, false, Scaled>(controls, d, a, b, c, scale_a, scale_b,
                                                    clamp_overflow);
  } else {
    mfma_avx512vnni_impl<M, K, true, true, Scaled>(controls, d, a, b, c, scale_a, scale_b,
                                                   clamp_overflow);
  }
  return true;
}

} // namespace

template <int M, int K>
bool mfma_mixed_avx512vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma_avx512vnni_dispatch<M, K, false>(controls, d, a, b, c, nullptr, nullptr, false);
}

template <int M, int K>
bool mfma_scaled_avx512vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *c,
                            const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow) {
  return mfma_avx512vnni_dispatch<M, K, true>(controls, d, a, b, c, scale_a, scale_b,
                                              clamp_overflow);
}

template bool mfma_mixed_avx512vnni<16, 128>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template bool mfma_mixed_avx512vnni<32, 64>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template bool mfma_scaled_avx512vnni<16, 128>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *, const uint32_t *, bool);
template bool mfma_scaled_avx512vnni<32, 64>(uint64_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *, const uint32_t *, bool);

} // namespace rocjitsu::arith
