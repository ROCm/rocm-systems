// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 row layouts and packed conversions for Loose WMMA/SWMMAC kernels.
#pragma once

#include "common/bits.h"
#include "common/compiler.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

template <bool Bf16>
[[gnu::always_inline]] inline __m512 wmma_widen_half(__m512i words, int shift) {
  if constexpr (Bf16) {
    return _mm512_castsi512_ps(shift ? _mm512_and_si512(words, _mm512_set1_epi32(int(0xffff0000U)))
                                     : _mm512_slli_epi32(words, 16));
  } else {
    const __m256i packed = _mm512_cvtepi32_epi16(shift ? _mm512_srli_epi32(words, shift) : words);
    return _mm512_cvt_roundph_ps(packed, _MM_FROUND_NO_EXC);
  }
}

// RNE preserves subnormals. Saturating finite overflow replaces a rounded
// infinity with the adjacent maximum finite half of the same sign.
template <bool Saturate> [[gnu::always_inline]] inline __m256i wmma_narrow_f16(__m512 x) {
  auto half = _mm512_cvt_roundps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  if constexpr (Saturate) {
    auto magnitude = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(INT32_MAX));
    auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto inf = _mm256_cmpeq_epi16_mask(_mm256_and_si256(half, _mm256_set1_epi16(0x7fff)),
                                       _mm256_set1_epi16(0x7c00));
    half = _mm256_mask_sub_epi16(half, finite & inf, half, _mm256_set1_epi16(1));
  }
  return half;
}

// Snapshot C before any D store, including the selected gfx11 half.
// For gfx12 packed C, loads each packed register once for both adjacent rows.
// When TrackSubnormal is true, tracks the minimum nonzero FP32 magnitude minus 1
// across all 16 widened rows to detect any subnormal input accumulator.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool TrackSubnormal = false>
[[gnu::always_inline]] inline __m512i wmma_load_acc(__m512 (&acc)[16], uint32_t mods,
                                                    const uint32_t *const *c) {
  const __m512i keep = _mm512_set1_epi32(mods & RJ_ARITH_WMMA_ABS_C ? INT32_MAX : -1);
  const __m512i sign = _mm512_set1_epi32(mods & RJ_ARITH_WMMA_NEG_C ? INT32_MIN : 0);
  const __m512i abs_mask = _mm512_set1_epi32(INT32_MAX);
  const __m512i neg_one = _mm512_set1_epi32(-1);
  __m512i min_mag = neg_one;

  auto finalize = [&](__m512i bits) -> __m512 {
    if constexpr (TrackSubnormal) {
      __m512i mag = _mm512_and_si512(bits, abs_mask);
      min_mag = _mm512_min_epu32(min_mag, _mm512_add_epi32(mag, neg_one));
    }
    // (bits & keep) ^ sign in one instruction (ternary logic 0x6a = C ^ (A & B)).
    return _mm512_castsi512_ps(_mm512_ternarylogic_epi32(bits, keep, sign, 0x6a));
  };

  if constexpr (Packed && !Rdna3) {
    // gfx12 packed C stores rows (row, row+1) in the low and high halves of one word.
    RJ_ARITH_UNROLL(8)
    for (int row = 0; row < 16; row += 2) {
      WmmaRegisterSection src = wmma_half_output<Lanes, false>(row);
      __m512i packed = _mm512_loadu_si512(c[src.reg / 2] + src.lane);
      __m512i lo = _mm512_castps_si512(wmma_widen_half<Bf16>(packed, 0));
      __m512i hi = _mm512_castps_si512(wmma_widen_half<Bf16>(packed, 16));
      acc[row] = finalize(lo);
      acc[row + 1] = finalize(hi);
    }
  } else {
    const int shift = (Packed && Rdna3 && (mods & RJ_ARITH_WMMA_HIGH_C_D)) ? 16 : 0;
    // Constant row indices keep the sixteen accumulators in vector registers.
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      WmmaRegisterSection src = wmma_half_output<Lanes, Rdna3>(row);
      __m512i bits = _mm512_loadu_si512(c[src.reg] + src.lane);
      if constexpr (Packed) {
        bits = _mm512_castps_si512(wmma_widen_half<Bf16>(bits, shift));
      }
      acc[row] = finalize(bits);
    }
  }
  return min_mag;
}

// Interleave index for packing 16 low-half FP16s from an even row and 16
// high-half FP16s from an odd row in one vpermt2w instruction.
[[gnu::always_inline]] inline __m512i wmma_f16_interleave_index_v4() {
  alignas(64) static constexpr uint16_t kInterleave[32] = {
      0, 32, 1, 33, 2,  34, 3,  35, 4,  36, 5,  37, 6,  38, 7,  39,
      8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
  return _mm512_load_si512(kInterleave);
}

// Materialize rows in ascending destination-register order. gfx12 packs
// adjacent rows into a word. gfx11 replaces the same half of every D word.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate>
[[gnu::always_inline]] inline void wmma_store_acc(const __m512 (&acc)[16], uint32_t mods,
                                                  uint32_t *const *d) {
  constexpr int Registers = (256 / Lanes) / (Packed && !Rdna3 ? 2 : 1);
  if constexpr (!Packed) {
    for (int reg = 0; reg < Registers; ++reg) {
      for (int group = 0; group < Lanes / 16; ++group) {
        int row = Rdna3 ? reg * (Lanes / 16) + group : reg + 8 * (group % 2) + 4 * (group / 2);
        _mm512_storeu_si512(d[reg] + 16 * group, _mm512_castps_si512(acc[row]));
      }
    }
  } else if constexpr (Rdna3) {
    const bool high = (mods & RJ_ARITH_WMMA_HIGH_C_D) != 0;
    for (int reg = 0; reg < Registers; ++reg) {
      for (int group = 0; group < Lanes / 16; ++group) {
        int row = reg * (Lanes / 16) + group;
        auto old = _mm512_loadu_si512(d[reg] + 16 * group);
        __m512i value;
        if constexpr (Bf16) {
          __m512i bits = _mm512_castps_si512(acc[row]);
          value = high ? _mm512_mask_blend_epi16(0xaaaaaaaa, old, bits)
                       : _mm512_mask_blend_epi16(0x55555555, old, _mm512_srli_epi32(bits, 16));
        } else {
          __m512i narrowed = _mm512_cvtepu16_epi32(wmma_narrow_f16<Saturate>(acc[row]));
          value = high ? _mm512_mask_blend_epi16(0xaaaaaaaa, old, _mm512_slli_epi32(narrowed, 16))
                       : _mm512_mask_blend_epi16(0x55555555, old, narrowed);
        }
        _mm512_storeu_si512(d[reg] + 16 * group, value);
      }
    }
  } else if constexpr (Bf16) {
    for (int reg = 0; reg < Registers; ++reg) {
      for (int group = 0; group < Lanes / 16; ++group) {
        int row = reg * 2 + 8 * (group % 2) + 4 * (group / 2);
        __m512i lo = _mm512_srli_epi32(_mm512_castps_si512(acc[row]), 16);
        __m512i hi = _mm512_castps_si512(acc[row + 1]);
        _mm512_storeu_si512(d[reg] + 16 * group, _mm512_mask_blend_epi16(0xaaaaaaaa, lo, hi));
      }
    }
  } else {
    const __m512i interleave = wmma_f16_interleave_index_v4();
    for (int reg = 0; reg < Registers; ++reg) {
      for (int group = 0; group < Lanes / 16; ++group) {
        int row = reg * 2 + 8 * (group % 2) + 4 * (group / 2);
        __m512i lo = _mm512_zextsi256_si512(wmma_narrow_f16<Saturate>(acc[row]));
        __m512i hi = _mm512_zextsi256_si512(wmma_narrow_f16<Saturate>(acc[row + 1]));
        _mm512_storeu_si512(d[reg] + 16 * group, _mm512_permutex2var_epi16(lo, interleave, hi));
      }
    }
  }
}

#if defined(RJ_ARITH_TARGET_X86_64_V4_BF16)
// Apply one K-pair DPBF16 step across all 16 output rows.
[[gnu::always_inline]] inline void wmma_bf16_pair_step(__m512 (&sums)[16], const uint32_t *left_row,
                                                       __m512i rhs) {
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    // Keep the vector load to avoid Clang 17's crash when casting a scalar splat to BF16.
    __m512i factor = _mm512_broadcastd_epi32(_mm_loadu_si32(&left_row[row]));
    sums[row] = _mm512_dpbf16_ps(sums[row], bit_cast<__m512bh>(rhs), bit_cast<__m512bh>(factor));
  }
}

// Both inputs contain adjacent BF16 K pairs with their halves swapped:
// DPBF16 consumes the high half first. Keep one accumulator per output row.
template <bool Packed, bool Rdna3>
[[gnu::always_inline]] inline void
wmma_bf16_pairs(__m512 (&sums)[16], const uint32_t (&left)[8][16], const uint32_t (&right)[8][16]) {
  const __m512i bf16_mask = _mm512_set1_epi32(int(0xffff0000U));
  auto trunc_rows = [&]() {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      sums[row] = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(sums[row]), bf16_mask));
    }
  };
  if constexpr (!Packed) {
    // Keep the outer loop rolled so left[step][row] stays in memory and folds
    // directly into vdpbf16ps {1to16} without extra broadcasts or spills.
    RJ_ARITH_UNROLL(1)
    for (int step = 0; step < 8; ++step) {
      wmma_bf16_pair_step(sums, left[step], _mm512_load_si512(right[step]));
    }
  } else if constexpr (Rdna3) {
    // Peel step 7 so no conditional sits inside the unroll(disable) loop.
    RJ_ARITH_UNROLL(1)
    for (int step = 0; step < 7; ++step) {
      wmma_bf16_pair_step(sums, left[step], _mm512_load_si512(right[step]));
      trunc_rows();
    }
    wmma_bf16_pair_step(sums, left[7], _mm512_load_si512(right[7]));
  } else {
    // gfx12 packed narrows every 2 pairs (4 K elements). Packed BF16 stores
    // discard the low 16 bits, so keeping all 4 stages in the rolled loop is
    // exact and avoids peeled register extracts on GCC.
    RJ_ARITH_UNROLL(1)
    for (int stage = 0; stage < 4; ++stage) {
      wmma_bf16_pair_step(sums, left[2 * stage], _mm512_load_si512(right[2 * stage]));
      wmma_bf16_pair_step(sums, left[2 * stage + 1], _mm512_load_si512(right[2 * stage + 1]));
      trunc_rows();
    }
  }
}
#endif

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
