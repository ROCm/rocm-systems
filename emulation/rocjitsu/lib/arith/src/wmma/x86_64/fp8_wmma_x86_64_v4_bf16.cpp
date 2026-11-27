// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native BF16 pair dots for loose gfx12 FP8/BF8 WMMA using AVX-512BW LUT decode.

#include "common/simd.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

namespace rocjitsu::arith {

// Decode every FP8 value into BF16 and reduce with native pair-dot arithmetic.
// Native accumulator flushing is permitted in Loose.
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v4_bf16(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  // Pre-decode A and B into 512B K-pair stack buffers BEFORE loading C so
  // (a) the 4-ZMM LUT is completely dead before acc[16] goes live (zero spills)
  // and (b) the 16 C loads + subnormal check separate left[] stores from
  // vdpbf16ps {1to16} reads (zero STLF stalls). D is untouched on early return.
  alignas(64) uint32_t left[8][16];
  alignas(64) uint32_t right[8][16];
  if constexpr (Bf8A == Bf8B) {
    const WmmaFp8Bf16Lut<Bf8A> lut;
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < 16; k += 4) {
      const int reg = wmma_gfx12_packed_local(k) / 4;
      const int lane = 16 * (k / 8);
      const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
      const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
      _mm512_store_si512(left[k / 2], lut.decode_bf16_pairs(aw, 0));
      _mm512_store_si512(left[k / 2 + 1], lut.decode_bf16_pairs(aw, 1));
      _mm512_store_si512(right[k / 2], lut.decode_bf16_pairs(bw, 0));
      _mm512_store_si512(right[k / 2 + 1], lut.decode_bf16_pairs(bw, 1));
    }
  } else {
    {
      const WmmaFp8Bf16Lut<Bf8A> lut_a;
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 16; k += 4) {
        const int reg = wmma_gfx12_packed_local(k) / 4;
        const int lane = 16 * (k / 8);
        const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
        _mm512_store_si512(left[k / 2], lut_a.decode_bf16_pairs(aw, 0));
        _mm512_store_si512(left[k / 2 + 1], lut_a.decode_bf16_pairs(aw, 1));
      }
    }
    {
      const WmmaFp8Bf16Lut<Bf8B> lut_b;
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 16; k += 4) {
        const int reg = wmma_gfx12_packed_local(k) / 4;
        const int lane = 16 * (k / 8);
        const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
        _mm512_store_si512(right[k / 2], lut_b.decode_bf16_pairs(bw, 0));
        _mm512_store_si512(right[k / 2 + 1], lut_b.decode_bf16_pairs(bw, 1));
      }
    }
  }

  __m512 acc[16];
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    const __m512i bits = _mm512_loadu_si512(c[row % 8] + 16 * (row / 8));
    acc[row] = _mm512_castsi512_ps(bits);
  }

  wmma_bf16_pairs(acc, left, right);
  wmma_store_acc<true, 32, false, false, false>(acc, 0, d);
}

template void fp8_wmma_x86_64_v4_bf16<false, false>(uint32_t *const *, const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void fp8_wmma_x86_64_v4_bf16<false, true>(uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *,
                                                   const uint32_t *const *);
template void fp8_wmma_x86_64_v4_bf16<true, false>(uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *,
                                                   const uint32_t *const *);
template void fp8_wmma_x86_64_v4_bf16<true, true>(uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
