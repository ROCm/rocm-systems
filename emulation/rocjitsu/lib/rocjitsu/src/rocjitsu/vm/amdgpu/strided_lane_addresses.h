// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) && (defined(__AVX512F__) || defined(__AVX2__))
#include <immintrin.h>
#endif

namespace rocjitsu::amdgpu::detail {

/// Check a dense window of lane addresses without reading beyond its count.
/// The positive stride and bounded count make the endpoint extent fit uint64_t.
inline bool is_strided_lane_addresses(const uint64_t *addrs, uint32_t count, uint32_t stride) {
  assert(addrs && count >= 1 && count <= 64 && stride != 0);
  if (count == 1)
    return true;
  const uint64_t step = stride;
  const uint64_t extent = (count - 1) * step;
  if (addrs[0] > std::numeric_limits<uint64_t>::max() - extent)
    return false;

  uint32_t index = 0;
#if defined(__x86_64__) && defined(__AVX512F__)
  const __m512i advance = _mm512_set1_epi64(static_cast<int64_t>(step * 8));
  __m512i expected = _mm512_add_epi64(
      _mm512_set1_epi64(static_cast<int64_t>(addrs[0])),
      _mm512_setr_epi64(0, static_cast<int64_t>(step), static_cast<int64_t>(step * 2),
                        static_cast<int64_t>(step * 3), static_cast<int64_t>(step * 4),
                        static_cast<int64_t>(step * 5), static_cast<int64_t>(step * 6),
                        static_cast<int64_t>(step * 7)));
  for (; index + 8 <= count; index += 8) {
    if (_mm512_cmpeq_epi64_mask(_mm512_loadu_si512(addrs + index), expected) != 0xff)
      return false;
    expected = _mm512_add_epi64(expected, advance);
  }
#elif defined(__x86_64__) && defined(__AVX2__)
  const __m256i advance = _mm256_set1_epi64x(static_cast<int64_t>(step * 4));
  __m256i expected = _mm256_add_epi64(_mm256_set1_epi64x(static_cast<int64_t>(addrs[0])),
                                      _mm256_setr_epi64x(0, static_cast<int64_t>(step),
                                                         static_cast<int64_t>(step * 2),
                                                         static_cast<int64_t>(step * 3)));
  for (; index + 4 <= count; index += 4) {
    const __m256i equal = _mm256_cmpeq_epi64(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(addrs + index)), expected);
    if (_mm256_movemask_pd(_mm256_castsi256_pd(equal)) != 15)
      return false;
    expected = _mm256_add_epi64(expected, advance);
  }
#else
  index = 1;
#endif
  for (; index < count; ++index) {
    if (addrs[index] != addrs[0] + index * step)
      return false;
  }
  return true;
}

} // namespace rocjitsu::amdgpu::detail
