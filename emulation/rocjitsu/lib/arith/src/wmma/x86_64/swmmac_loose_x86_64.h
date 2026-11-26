// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose sparse float WMMA over the packed architectural operands for AVX2 (v3)
// and AVX-512 (v4). Uses 16-row register-resident accumulators for half inputs
// on AVX-512 and paired F16C/FP8 widening with 4-row register blocking on AVX2.
#pragma once

#include "common/compiler.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/sparse_layout.h"
#include "wmma/x86_64/swmmac_vector.h"
#if defined(RJ_ARITH_TARGET_X86_64_V4)
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"
#else
#include "wmma/x86_64/wmma_loose_x86_64_v3.h"
#endif

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86 {

#if defined(RJ_ARITH_TARGET_X86_64_V4)

template <bool Bf16, bool Packed, bool Saturate>
[[gnu::always_inline]] inline void
swmmac_fma_ck_v4(__m512 (&acc)[16], const float (&left)[16][16], const float (&right)[32][16],
                 const uint32_t (&selected)[16][16], int ck, __m512i b_flip, __m512i bf16_mask,
                 bool round_here) {
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    const __m512 bv =
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_load_si512(right[selected[ck][row]]), b_flip));
    __m512 sum = _mm512_fmadd_ps(_mm512_set1_ps(left[ck][row]), bv, acc[row]);
    if constexpr (Packed) {
      if (round_here) {
        if constexpr (Bf16) {
          sum = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(sum), bf16_mask));
        } else {
          sum = _mm512_cvt_roundph_ps(wmma_narrow_f16<Saturate>(sum), _MM_FROUND_NO_EXC);
        }
      }
    }
    acc[row] = sum;
  }
}

template <bool Bf16, int Wave, bool Packed, bool Saturate>
[[gnu::noinline]] inline void compute_half_v4(uint32_t mode, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *index) {
  alignas(64) float left[16][16];
  alignas(64) float right[32][16];
  alignas(64) uint32_t selected[16][16];
  const int key = SwmmacIndex<16, 32, Wave>::shift(mode);

  const __m512i a_sign = _mm512_set1_epi32((mode & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                                           (mode & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const __m512i mask3 = _mm512_set1_epi32(3);
  for (int p = 0; p < 8; ++p) {
    const int ck = 2 * p;
    const SparseField af = swmmac_a_field<16, Wave>(0, ck);
    const __m512i words = _mm512_xor_si512(_mm512_loadu_si512(a[af.reg] + af.lane), a_sign);
    // wmma_widen_half<false> uses vcvtph2ps {sae}, suppressing invalid exceptions.
    _mm512_store_ps(left[ck], wmma_widen_half<Bf16>(words, 0));
    _mm512_store_ps(left[ck + 1], wmma_widen_half<Bf16>(words, 16));
    const __m512i meta = _mm512_loadu_si512(index[0] + af.lane);
    const __m512i base_k = _mm512_set1_epi32(4 * p);
    const int s0 = swmmac_index_slot<16, Wave>(ck);
    const int s1 = swmmac_index_slot<16, Wave>(ck + 1);
    _mm512_store_si512(
        selected[ck],
        _mm512_add_epi32(base_k, _mm512_and_si512(_mm512_srli_epi32(meta, key + 2 * s0), mask3)));
    _mm512_store_si512(
        selected[ck + 1],
        _mm512_add_epi32(base_k, _mm512_and_si512(_mm512_srli_epi32(meta, key + 2 * s1), mask3)));
  }
  for (int p = 0; p < 16; ++p) {
    const int k = 2 * p;
    const SparseField bf = swmmac_b_field<16, Wave>(0, k);
    const __m512i words = _mm512_loadu_si512(b[bf.reg] + bf.lane);
    _mm512_store_ps(right[k], wmma_widen_half<Bf16>(words, 0));
    _mm512_store_ps(right[k + 1], wmma_widen_half<Bf16>(words, 16));
  }

  // Snapshot all 16 tied C/D rows into ZMM registers before any stores.
  __m512 acc[16];
  const __m512i keep = _mm512_set1_epi32((mode & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
  const __m512i sign = _mm512_set1_epi32((mode & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
  auto apply_c = [&](__m512i bits) -> __m512 {
    return _mm512_castsi512_ps(_mm512_ternarylogic_epi32(bits, keep, sign, 0x6a));
  };
  if constexpr (Packed) {
    RJ_ARITH_UNROLL(8)
    for (int row = 0; row < 16; row += 2) {
      const int reg = (row % 8) / 2;
      const int lane = 16 * (row / 8);
      const __m512i packed = _mm512_loadu_si512(d[reg] + lane);
      acc[row] = apply_c(_mm512_castps_si512(wmma_widen_half<Bf16>(packed, 0)));
      acc[row + 1] = apply_c(_mm512_castps_si512(wmma_widen_half<Bf16>(packed, 16)));
    }
  } else {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      const __m512i bits = _mm512_loadu_si512(d[row % 8] + 16 * (row / 8));
      acc[row] = apply_c(bits);
    }
  }

  const __m512i b_flip_lo = _mm512_set1_epi32(((mode & RJ_ARITH_WMMA_NEG_LO_B)) ? INT32_MIN : 0);
  const __m512i b_flip_hi = _mm512_set1_epi32(((mode & RJ_ARITH_WMMA_NEG_HI_B)) ? INT32_MIN : 0);
  const __m512i bf16_mask = _mm512_set1_epi32(int(0xffff0000U));

  if constexpr (!Packed) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (int pair = 0; pair < 8; ++pair) {
      swmmac_fma_ck_v4<Bf16, false, false>(acc, left, right, selected, 2 * pair, b_flip_lo,
                                           bf16_mask, false);
      swmmac_fma_ck_v4<Bf16, false, false>(acc, left, right, selected, 2 * pair + 1, b_flip_hi,
                                           bf16_mask, false);
    }
  } else {
    // Peel the final stage (no rounding after stage 3).
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (int stage = 0; stage < 3; ++stage) {
      const int ck = 4 * stage;
      swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, ck, b_flip_lo, bf16_mask,
                                             false);
      swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, ck + 1, b_flip_hi,
                                             bf16_mask, false);
      swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, ck + 2, b_flip_lo,
                                             bf16_mask, false);
      swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, ck + 3, b_flip_hi,
                                             bf16_mask, true);
    }
    swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, 12, b_flip_lo, bf16_mask,
                                           false);
    swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, 13, b_flip_hi, bf16_mask,
                                           false);
    swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, 14, b_flip_lo, bf16_mask,
                                           false);
    swmmac_fma_ck_v4<Bf16, true, Saturate>(acc, left, right, selected, 15, b_flip_hi, bf16_mask,
                                           false);
  }

  // All inputs and tied D were read before any stores. Commit directly in order.
  if constexpr (!Packed) {
    for (int reg = 0; reg < 8; ++reg) {
      _mm512_storeu_si512(d[reg], _mm512_castps_si512(acc[reg]));
      _mm512_storeu_si512(d[reg] + 16, _mm512_castps_si512(acc[reg + 8]));
    }
  } else if constexpr (Bf16) {
    for (int reg = 0; reg < 4; ++reg) {
      for (int group = 0; group < 2; ++group) {
        const int row = 2 * reg + 8 * group;
        const __m512i lo = _mm512_srli_epi32(_mm512_castps_si512(acc[row]), 16);
        const __m512i hi = _mm512_castps_si512(acc[row + 1]);
        _mm512_storeu_si512(d[reg] + 16 * group, _mm512_mask_blend_epi16(0xaaaaaaaa, lo, hi));
      }
    }
  } else {
    const __m512i interleave = wmma_f16_interleave_index_v4();
    for (int reg = 0; reg < 4; ++reg) {
      for (int group = 0; group < 2; ++group) {
        const int row = 2 * reg + 8 * group;
        const __m512i lo = _mm512_zextsi256_si512(wmma_narrow_f16<Saturate>(acc[row]));
        const __m512i hi = _mm512_zextsi256_si512(wmma_narrow_f16<Saturate>(acc[row + 1]));
        _mm512_storeu_si512(d[reg] + 16 * group, _mm512_permutex2var_epi16(lo, interleave, hi));
      }
    }
  }
}

#else

// Exception-free software FP16 widen for dense B elements on AVX2 so unused
// signaling NaNs in B never raise FE_INVALID.
[[gnu::always_inline]] inline __m256 widen_f16_safe_v3(U32 x) {
  const U32 m = x & 0x7fffu;
  const U32 sign = (x & 0x8000u) << 16;
  const U32 tiny = bit_cast<U32>(__builtin_convertvector(m & 1023u, F32) * 0x1p-24f);
  const U32 normal = (m << 13) + (112u << 23);
  const U32 special = (m << 13) | 0x7f800000u;
  return bit_cast<__m256>(sign | (m < 1024u ? tiny : m >= 0x7c00u ? special : normal));
}

template <bool Bf8>
[[gnu::always_inline]] inline void store_fp8_word_v3(__m256i w, float *dst0, float *dst1,
                                                     float *dst2, float *dst3) {
  __m256 f0, f1, f2, f3;
  wmma_decode_fp8_word_v3<Bf8, /*CheckNan=*/true>(w, f0, f1, f2, f3);
  if constexpr (!Bf8) {
    const __m256 scale = _mm256_set1_ps(256.0f);
    f0 = _mm256_mul_ps(f0, scale);
    f1 = _mm256_mul_ps(f1, scale);
    f2 = _mm256_mul_ps(f2, scale);
    f3 = _mm256_mul_ps(f3, scale);
  }
  _mm256_store_ps(dst0, f0);
  _mm256_store_ps(dst1, f1);
  _mm256_store_ps(dst2, f2);
  _mm256_store_ps(dst3, f3);
}

template <int Bits, bool BfA, bool BfB, int Wave, bool Packed, bool Saturate>
[[gnu::noinline]] inline void compute_float_v3(uint32_t mode, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               const uint32_t *const *index) {
  alignas(32) float left[16][16];
  alignas(32) float right[32][16];
  alignas(32) uint32_t selected[16][16];
  const int key = SwmmacIndex<Bits, 32, Wave>::shift(mode);

  if constexpr (Bits == 8) {
    const __m256i mask3 = _mm256_set1_epi32(3);
    for (int g = 0; g < 4; ++g) {
      const int ck = 4 * g;
      const SparseField af = swmmac_a_field<8, Wave>(0, ck);
      for (int row = 0; row < 16; row += 8) {
        const __m256i w =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[af.reg] + af.lane + row));
        store_fp8_word_v3<BfA>(w, left[ck] + row, left[ck + 1] + row, left[ck + 2] + row,
                               left[ck + 3] + row);
        const __m256i meta =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index[0] + af.lane + row));
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 4; ++j) {
          const int slot = swmmac_index_slot<8, Wave>(ck + j);
          const __m256i sel =
              _mm256_add_epi32(_mm256_set1_epi32(4 * ((ck + j) / 2)),
                               _mm256_and_si256(_mm256_srli_epi32(meta, key + 2 * slot), mask3));
          _mm256_store_si256(reinterpret_cast<__m256i *>(selected[ck + j] + row), sel);
        }
      }
    }
    for (int g = 0; g < 8; ++g) {
      const int k = 4 * g;
      const SparseField bf = swmmac_b_field<8, Wave>(0, k);
      for (int col = 0; col < 16; col += 8) {
        const __m256i w =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[bf.reg] + bf.lane + col));
        store_fp8_word_v3<BfB>(w, right[k] + col, right[k + 1] + col, right[k + 2] + col,
                               right[k + 3] + col);
      }
    }
  } else {
    const __m256i a_sign = _mm256_set1_epi32((mode & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                                             (mode & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
    const __m256i mask3 = _mm256_set1_epi32(3);
    for (int p = 0; p < 8; ++p) {
      const int ck = 2 * p;
      const SparseField af = swmmac_a_field<16, Wave>(0, ck);
      const __m256i base_k = _mm256_set1_epi32(4 * p);
      const int s0 = swmmac_index_slot<16, Wave>(ck);
      const int s1 = swmmac_index_slot<16, Wave>(ck + 1);
      for (int row = 0; row < 16; row += 8) {
        const __m256i w = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[af.reg] + af.lane + row)),
            a_sign);
        __m256 even, odd;
        wmma_widen_half_pair_v3<BfA>(w, even, odd);
        _mm256_store_ps(left[ck] + row, even);
        _mm256_store_ps(left[ck + 1] + row, odd);
        const __m256i meta =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(index[0] + af.lane + row));
        const __m256i sel0 = _mm256_add_epi32(
            base_k, _mm256_and_si256(_mm256_srli_epi32(meta, key + 2 * s0), mask3));
        const __m256i sel1 = _mm256_add_epi32(
            base_k, _mm256_and_si256(_mm256_srli_epi32(meta, key + 2 * s1), mask3));
        _mm256_store_si256(reinterpret_cast<__m256i *>(selected[ck] + row), sel0);
        _mm256_store_si256(reinterpret_cast<__m256i *>(selected[ck + 1] + row), sel1);
      }
    }
    for (int p = 0; p < 16; ++p) {
      const int k = 2 * p;
      const SparseField bf = swmmac_b_field<16, Wave>(0, k);
      for (int col = 0; col < 16; col += 8) {
        const __m256i w =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[bf.reg] + bf.lane + col));
        if constexpr (BfB) {
          __m256 even, odd;
          wmma_widen_half_pair_v3<true>(w, even, odd);
          _mm256_store_ps(right[k] + col, even);
          _mm256_store_ps(right[k + 1] + col, odd);
        } else {
          const U32 wu = bit_cast<U32>(w);
          _mm256_store_ps(right[k] + col, widen_f16_safe_v3(wu));
          _mm256_store_ps(right[k + 1] + col, widen_f16_safe_v3(wu >> 16));
        }
      }
    }
  }

  const __m256 keep = _mm256_castsi256_ps(
      _mm256_set1_epi32(Bits == 16 && (mode & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1));
  const __m256 sign = _mm256_castsi256_ps(
      _mm256_set1_epi32(Bits == 16 && (mode & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0));
  auto apply_c = [&](__m256 v) {
    if constexpr (Bits == 16) {
      return _mm256_xor_ps(_mm256_and_ps(v, keep), sign);
    } else {
      return v;
    }
  };
  const __m256 b_flip_lo = _mm256_castsi256_ps(
      _mm256_set1_epi32(Bits == 16 && (mode & RJ_ARITH_WMMA_NEG_LO_B) ? INT32_MIN : 0));
  const __m256 b_flip_hi = _mm256_castsi256_ps(
      _mm256_set1_epi32(Bits == 16 && (mode & RJ_ARITH_WMMA_NEG_HI_B) ? INT32_MIN : 0));

  alignas(32) uint32_t result[Packed ? 4 : 8][32];
  constexpr int Rows = 4;

#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (int tile = 0; tile < 16; tile += Rows) {
    __m256 low[Rows], high[Rows];
    if constexpr (Packed) {
      for (int r = 0; r < Rows; r += 2) {
        const int row = tile + r;
        const uint32_t *c_ptr = d[(row % 8) / 2] + 16 * (row / 8);
        const __m256i lo_w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr));
        const __m256i hi_w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8));
        __m256 e0, o0, e1, o1;
        wmma_widen_half_pair_v3<BfA>(lo_w, e0, o0);
        wmma_widen_half_pair_v3<BfA>(hi_w, e1, o1);
        low[r] = apply_c(e0);
        low[r + 1] = apply_c(o0);
        high[r] = apply_c(e1);
        high[r + 1] = apply_c(o1);
      }
    } else {
      for (int r = 0; r < Rows; ++r) {
        const int row = tile + r;
        const uint32_t *c_ptr = d[row % 8] + 16 * (row / 8);
        low[r] = apply_c(
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr))));
        high[r] = apply_c(
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8))));
      }
    }

    auto fma_ck = [&](int ck, __m256 b_flip) {
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < Rows; ++r) {
        const __m256 x = _mm256_broadcast_ss(&left[ck][tile + r]);
        const int k = int(selected[ck][tile + r]);
        __m256 bl = _mm256_load_ps(right[k]);
        __m256 bh = _mm256_load_ps(right[k] + 8);
        if constexpr (Bits == 16) {
          bl = _mm256_xor_ps(bl, b_flip);
          bh = _mm256_xor_ps(bh, b_flip);
        }
        low[r] = _mm256_fmadd_ps(x, bl, low[r]);
        high[r] = _mm256_fmadd_ps(x, bh, high[r]);
      }
    };
    auto fma_round_ck = [&](int ck, __m256 b_flip) {
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < Rows; ++r) {
        const __m256 x = _mm256_broadcast_ss(&left[ck][tile + r]);
        const int k = int(selected[ck][tile + r]);
        __m256 bl = _mm256_load_ps(right[k]);
        __m256 bh = _mm256_load_ps(right[k] + 8);
        if constexpr (Bits == 16) {
          bl = _mm256_xor_ps(bl, b_flip);
          bh = _mm256_xor_ps(bh, b_flip);
        }
        low[r] = wmma_round_stage_v3<BfA, Saturate>(_mm256_fmadd_ps(x, bl, low[r]));
        high[r] = wmma_round_stage_v3<BfA, Saturate>(_mm256_fmadd_ps(x, bh, high[r]));
      }
    };

    if constexpr (!Packed) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
      for (int pair = 0; pair < 8; ++pair) {
        fma_ck(2 * pair, b_flip_lo);
        fma_ck(2 * pair + 1, b_flip_hi);
      }
    } else {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
      for (int stage = 0; stage < 3; ++stage) {
        const int ck = 4 * stage;
        fma_ck(ck, b_flip_lo);
        fma_ck(ck + 1, b_flip_hi);
        fma_ck(ck + 2, b_flip_lo);
        fma_round_ck(ck + 3, b_flip_hi);
      }
      fma_ck(12, b_flip_lo);
      fma_ck(13, b_flip_hi);
      fma_ck(14, b_flip_lo);
      fma_ck(15, b_flip_hi);
    }

    if constexpr (Packed) {
      for (int r = 0; r < Rows; r += 2) {
        const int row = tile + r;
        uint32_t *dst = result[(row % 8) / 2] + 16 * (row / 8);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst),
                           wmma_pack_half_pair_v3<BfA, Saturate>(low[r], low[r + 1]));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + 8),
                           wmma_pack_half_pair_v3<BfA, Saturate>(high[r], high[r + 1]));
      }
    } else {
      for (int r = 0; r < Rows; ++r) {
        const int row = tile + r;
        uint32_t *dst = result[row % 8] + 16 * (row / 8);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst), _mm256_castps_si256(low[r]));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + 8), _mm256_castps_si256(high[r]));
      }
    }
  }

  for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane),
                          _mm256_load_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

#endif

template <int Bits, bool BfA, bool BfB, int Wave, bool Packed>
inline void compute_float(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *index, bool saturate) {
  if constexpr (Packed && !BfA) {
    if (saturate) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
      compute_half_v4<BfA, Wave, Packed, true>(mode, d, a, b, index);
#else
      compute_float_v3<Bits, BfA, BfB, Wave, Packed, true>(mode, d, a, b, index);
#endif
      return;
    }
  }
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  compute_half_v4<BfA, Wave, Packed, false>(mode, d, a, b, index);
#else
  compute_float_v3<Bits, BfA, BfB, Wave, Packed, false>(mode, d, a, b, index);
#endif
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86
