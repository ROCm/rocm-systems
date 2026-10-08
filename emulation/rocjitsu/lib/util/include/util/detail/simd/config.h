// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_CONFIG_H_
#define UTIL_DETAIL_SIMD_CONFIG_H_

#include <cstddef>

#if __has_include(<experimental/simd>)
#include <experimental/simd>
#endif

// clang + libstdc++ <experimental/simd> on AVX-512 has produced incorrect
// native 64-bit mask behavior on sanitizer CI hosts. Keep the default
// workaround local to that stack, but leave it overrideable so fixed toolchains
// can force vector coverage with -DUTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS=0.
#ifdef UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS_USER_OVERRIDE 1
#else
#define UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS_USER_OVERRIDE 0
#if defined(__clang__) && defined(__GLIBCXX__) && defined(__AVX512F__)
#define UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS 1
#else
#define UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS 0
#endif
#endif

namespace util {

/// Compile-time switch for `<experimental/simd>` availability. Callers
/// gate SIMD fast paths via `if constexpr (util::has_stdx_simd)` so the
/// branch and all downstream calls compile out when the header is
/// absent (e.g. older libstdc++).
inline constexpr bool has_stdx_simd =
#if __has_include(<experimental/simd>)
    true;
#else
    false;
#endif

} // namespace util

#endif // UTIL_DETAIL_SIMD_CONFIG_H_
