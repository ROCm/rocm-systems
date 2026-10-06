// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_CONVERT_SELECT_H_
#define UTIL_DETAIL_CONVERT_SELECT_H_

#include "util/detail/convert/portable.h"
#include "util/detail/target_features.h"

#if defined(UTIL_HAS_X86_F16C)
#include "util/detail/convert/x86_f16c.h"
#if defined(__AVX2__)
#include "util/detail/convert/x86_avx2.h"
#endif
#if defined(__AVX512F__)
#include "util/detail/convert/x86_avx512.h"
#endif
#endif

namespace util::detail {

/// Preserve the existing 16-lane, 8-lane, then scalar-tail conversion order.
/// F16C quiets NaNs in vector chunks; the scalar tail keeps its existing payload policy.
inline void f16_to_f32_block_arch(const uint16_t *src, float *dst, size_t n, size_t &i) {
#if defined(UTIL_HAS_X86_F16C)
#if defined(__AVX512F__)
  f16_to_f32_block_avx512(src, dst, n, i);
#endif
  f16_to_f32_block_f16c(src, dst, n, i);
#else
  f16_to_f32_block_portable(src, dst, n, i);
#endif
}

inline void bf16_to_f32_block_arch(const uint16_t *src, float *dst, size_t n, size_t &i) {
#if defined(UTIL_HAS_X86_F16C)
#if defined(__AVX512F__)
  bf16_to_f32_block_avx512(src, dst, n, i);
#endif
#if defined(__AVX2__)
  bf16_to_f32_block_avx2(src, dst, n, i);
#endif
#endif
  bf16_to_f32_block_portable(src, dst, n, i);
}

inline void i8_to_i32_block_arch(const int8_t *src, int32_t *dst, size_t n, size_t &i) {
#if defined(UTIL_HAS_X86_F16C)
#if defined(__AVX512F__)
  i8_to_i32_block_avx512(src, dst, n, i);
#endif
#if defined(__AVX2__)
  i8_to_i32_block_avx2(src, dst, n, i);
#endif
#endif
  i8_to_i32_block_portable(src, dst, n, i);
}

inline void u8_to_i32_block_arch(const uint8_t *src, int32_t *dst, size_t n, size_t &i) {
#if defined(UTIL_HAS_X86_F16C)
#if defined(__AVX512F__)
  u8_to_i32_block_avx512(src, dst, n, i);
#endif
#if defined(__AVX2__)
  u8_to_i32_block_avx2(src, dst, n, i);
#endif
#endif
  u8_to_i32_block_portable(src, dst, n, i);
}

} // namespace util::detail

#endif // UTIL_DETAIL_CONVERT_SELECT_H_
