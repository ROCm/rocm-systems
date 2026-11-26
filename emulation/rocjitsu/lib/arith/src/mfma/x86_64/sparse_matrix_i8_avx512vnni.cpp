// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated AVX-512 VNNI sparse signed-byte SMFMAC entry points.
#include "common/compiler.h"
#include "mfma/backends.h"
#include "mfma/x86_64/sparse_matrix_i8_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace RJ_ARITH_IMPL_NAMESPACE;

// Saturating addition of a signed 24-bit dot product to a signed 32-bit accumulator.
// Sparse 2:4 SMFMAC multiplies at most K/2 <= 64 byte pairs, so |dot| <= 1048576 < 2^23.
inline __m512i saturate_signed_sum_24_sparse(__m512i initial, __m512i dot) {
  const __m512i shuf = _mm512_set4_epi32(0x0e0d800c, 0x0a098008, 0x06058004, 0x02018000);
  const __m512i weights = _mm512_set1_epi32(0x01000001);
  return _mm512_dpwssds_epi32(initial, _mm512_shuffle_epi8(dot, shuf), weights);
}

inline void broadcast_16_lanes_sparse(__m512i v, __m512i (&out)[16]) {
  auto expand4 = [&](int base, __m128i lane4) {
    out[base + 0] = _mm512_broadcastd_epi32(lane4);
    out[base + 1] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0x55));
    out[base + 2] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0xaa));
    out[base + 3] = _mm512_broadcastd_epi32(_mm_shuffle_epi32(lane4, 0xff));
  };
  expand4(0, _mm512_castsi512_si128(v));
  expand4(4, _mm512_extracti32x4_epi32(v, 1));
  expand4(8, _mm512_extracti32x4_epi32(v, 2));
  expand4(12, _mm512_extracti32x4_epi32(v, 3));
}

template <int M, int K, bool Clamp>
void smfmac_i8_kernel_vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *index) {
  static_assert(M == 16);
  constexpr int num_pairs = K / 8;
  constexpr __mmask64 kHiBytesMask = 0xccccccccccccccccULL;

  const int ctrl_shift =
      (M * K < 2048 && (controls & 3) == 0) ? 16 * static_cast<int>((controls >> 3) & 1) : 0;

  const __m512i zero = _mm512_setzero_si512();
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i sign_mask = _mm512_set1_epi8(-128);
  __m512i row_sum = zero;
  for (int p = 0; p < num_pairs; ++p) {
    const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<M, K>(p);
    row_sum = _mm512_dpbusd_epi32(row_sum, ones, _mm512_loadu_si512(a[loc.a_reg] + loc.a_lane));
  }
  const __m512i bias_vec = _mm512_slli_epi32(_mm512_sub_epi32(zero, row_sum), 7);
  __m512i acc[16];
  broadcast_16_lanes_sparse(bias_vec, acc);

  for (int p = 0; p < num_pairs; ++p) {
    const SmfmacI8PairLoc loc = smfmac_i8_pair_loc<M, K>(p);
    const __m512i ub0 = _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg] + loc.b_lane), sign_mask);
    const __m512i ub1 =
        _mm512_xor_si512(_mm512_loadu_si512(b[loc.b_reg + 1] + loc.b_lane), sign_mask);
    const uint32_t *ap = a[loc.a_reg] + loc.a_lane;
    const uint32_t *ip = index[0] + loc.a_lane;
    const int shift = ctrl_shift + loc.nib_shift;
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      const uint32_t sel = (ip[r] >> shift) & 0xffu;
      const __m512i s0 =
          _mm512_shuffle_epi8(ub0, _mm512_load_si512(kSmfmacPairShuf.row[sel & 15u]));
      const __m512i picked = _mm512_mask_shuffle_epi8(
          s0, kHiBytesMask, ub1, _mm512_load_si512(kSmfmacPairShuf.row[sel >> 4]));
      acc[r] = _mm512_dpbusd_epi32(acc[r], picked, _mm512_set1_epi32(static_cast<int>(ap[r])));
    }
  }

  // Consume all initial D accumulators before writing back so arbitrary aliasing is exact.
  RJ_ARITH_UNROLL(16)
  for (int r = 0; r < 16; ++r) {
    const int out_reg = r & 3;
    const int out_lane = (r >> 2) * 16;
    const __m512i initial = _mm512_loadu_si512(d[out_reg] + out_lane);
    if constexpr (Clamp) {
      acc[r] = saturate_signed_sum_24_sparse(initial, acc[r]);
    } else {
      acc[r] = _mm512_add_epi32(initial, acc[r]);
    }
  }
  RJ_ARITH_UNROLL(16)
  for (int r = 0; r < 16; ++r) {
    const int out_reg = r & 3;
    const int out_lane = (r >> 2) * 16;
    _mm512_storeu_si512(d[out_reg] + out_lane, acc[r]);
  }
}

} // namespace

template <int M, int K>
void sparse_matrix_i8_avx512vnni(uint64_t controls, bool clamp, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *index) {
  if (clamp) {
    smfmac_i8_kernel_vnni<M, K, true>(controls, d, a, b, index);
  } else {
    smfmac_i8_kernel_vnni<M, K, false>(controls, d, a, b, index);
  }
}

template void sparse_matrix_i8_avx512vnni<16, 64>(uint64_t, bool, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void sparse_matrix_i8_avx512vnni<16, 128>(uint64_t, bool, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
} // namespace rocjitsu::arith
