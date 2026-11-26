// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend (vdpbf16ps) for loose CDNA3 FNUZ and CDNA4 OCP FP8/BF8 MFMA.
#include "mfma/backends.h"
#include "mfma/x86_64/bf16_dot.h"
#include "mfma/x86_64/fp8.h"

namespace rocjitsu::arith {

// Every finite FP8/BF8 product (FNUZ and OCP) is an integer multiple of at least 2^-34, so
// nonzero products cannot cancel a normal FP32 accumulator into a subnormal. Only subnormal C
// requires fallback to the general x86-64-v4 FMA backend.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
bool mfma_fp8_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int per_lane = K / (64 / M);
  constexpr int regs = per_lane / 4;
  constexpr int pair_planes = per_lane / 2;
  alignas(64) uint32_t left[pair_planes][64], right[pair_planes][64];
  for (int reg = 0; reg < regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      const __m512i aw = _mm512_loadu_si512(a[reg] + lane);
      const __m512i bw = _mm512_loadu_si512(b[reg] + lane);
      // vdpbf16ps consumes the high half first; swap halves to preserve guest K order.
      const __m512i a0 = mfma_fp8_to_bf16_pairs<Bf8A, Fnuz>(aw, 0);
      const __m512i a1 = mfma_fp8_to_bf16_pairs<Bf8A, Fnuz>(aw, 1);
      const __m512i b0 = mfma_fp8_to_bf16_pairs<Bf8B, Fnuz>(bw, 0);
      const __m512i b1 = mfma_fp8_to_bf16_pairs<Bf8B, Fnuz>(bw, 1);
      _mm512_store_si512(left[2 * reg + 0] + lane, _mm512_rol_epi32(a0, 16));
      _mm512_store_si512(left[2 * reg + 1] + lane, _mm512_rol_epi32(a1, 16));
      _mm512_store_si512(right[2 * reg + 0] + lane, _mm512_rol_epi32(b0, 16));
      _mm512_store_si512(right[2 * reg + 1] + lane, _mm512_rol_epi32(b1, 16));
    }
  }
  return mfma_reduce_bf16_pairs<M, K, 1>(controls, d, left, right, c, false);
}

template bool mfma_fp8_x86_64_v4_bf16<16, 32, false, false, true>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, false, false, false>(uint64_t, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, false, true, true>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, false, true, false>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, true, false, true>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, true, false, false>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, true, true, true>(uint64_t, uint32_t *const *,
                                                                const uint32_t *const *,
                                                                const uint32_t *const *,
                                                                const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<16, 32, true, true, false>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, false, false, true>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, false, false, false>(uint64_t, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, false, true, true>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, false, true, false>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, true, false, true>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, true, false, false>(uint64_t, uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *,
                                                                  const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, true, true, true>(uint64_t, uint32_t *const *,
                                                                const uint32_t *const *,
                                                                const uint32_t *const *,
                                                                const uint32_t *const *);
template bool mfma_fp8_x86_64_v4_bf16<32, 16, true, true, false>(uint64_t, uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *,
                                                                 const uint32_t *const *);

} // namespace rocjitsu::arith
