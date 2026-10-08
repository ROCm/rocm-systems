// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_CONVERT_X86_F16C_H_
#define UTIL_DETAIL_CONVERT_X86_F16C_H_

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace util::detail {

/// Convert eight half values per iteration; retain the scalar tail in the caller.
inline void f16_to_f32_block_f16c(const uint16_t *src, float *dst, size_t n, size_t &i) {
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(&dst[i],
                     _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
}

} // namespace util::detail

#endif // UTIL_DETAIL_CONVERT_X86_F16C_H_
