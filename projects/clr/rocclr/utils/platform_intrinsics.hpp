/* Copyright (c) 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef PLATFORM_INTRINSICS_HPP_
#define PLATFORM_INTRINSICS_HPP_

/**
 * @file platform_intrinsics.hpp
 * @brief Platform-specific CPU intrinsics abstraction layer
 *
 * This header provides a unified interface for CPU intrinsics across different
 * platforms (x86/x64, ARM, etc.). For x86/x64, it uses native intrinsics. For
 * ARM and other platforms, it provides equivalent implementations.
 */

#include "top.hpp"  // For ATI_ARCH_X86, ATI_ARCH_ARM definitions

#include <cstdint>
#include <cstring>

// ============================================================================
// x86/x64: Native Intrinsics
// ============================================================================

#if defined(ATI_ARCH_X86)

// Include native x86 intrinsic headers
#if defined(__MINGW64__)
  #include <intrin.h>
#else
  #include <xmmintrin.h>  // SSE  - for _mm_pause, _mm_sfence
  #include <emmintrin.h>  // SSE2 - for _mm_mfence, _mm_stream_si128, _mm_stream_si64, _mm_stream_si32
  #if defined(__AVX__)
    #include <immintrin.h>  // AVX, AVX2, AVX512 - for _mm256_stream_si256, _mm512_stream_si512
  #endif
#endif

// Native x86 intrinsics are used directly - no wrappers needed
// Functions available:
// - _mm_pause()             - CPU pause hint for spin-wait loops
// - _mm_sfence()            - Store memory fence
// - _mm_mfence()            - Full memory fence
// - _mm_stream_si128()      - 128-bit non-temporal store
// - _mm_stream_si64()       - 64-bit non-temporal store
// - _mm_stream_si32()       - 32-bit non-temporal store
// - _mm256_stream_si256()   - 256-bit non-temporal store (requires __AVX__)
// - _mm512_stream_si512()   - 512-bit non-temporal store (requires __AVX512F__)

#else // !ATI_ARCH_X86

// ============================================================================
// ARM and Other Platforms: Portable Implementations
// ============================================================================

// ----------------------------------------------------------------------------
// Type Definitions for SIMD Vectors
// ----------------------------------------------------------------------------

// These type definitions are adapted from SIMDe (https://github.com/simd-everywhere/simde)
// SIMDe is licensed under the MIT License:
//
// Copyright (c) 2017 Evan Nemerson <evan@nemerson.com>
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// 128-bit vector type
#if defined(__GNUC__) || defined(__clang__)
  typedef int64_t __m128i_portable __attribute__((__vector_size__(16), __aligned__(16)));
  #define __m128i __m128i_portable
#else
  struct alignas(16) __m128i {
    int64_t data[2];
  };
#endif

// 256-bit vector type
#if defined(__GNUC__) || defined(__clang__)
  typedef int64_t __m256i_portable __attribute__((__vector_size__(32), __aligned__(32)));
  #define __m256i __m256i_portable
#else
  struct alignas(32) __m256i {
    int64_t data[4];
  };
#endif

// 512-bit vector type
#if defined(__GNUC__) || defined(__clang__)
  typedef int64_t __m512i_portable __attribute__((__vector_size__(64), __aligned__(64)));
  #define __m512i __m512i_portable
#else
  struct alignas(64) __m512i {
    int64_t data[8];
  };
#endif

// ----------------------------------------------------------------------------
// CPU Pause
// ----------------------------------------------------------------------------

static inline void _mm_pause() {
#if defined(ATI_ARCH_ARM)
  // ARM: Use yield instruction (from original CLR code)
  __asm__ __volatile__("yield");
#else
  // Generic fallback: No-op
#endif
}

// ----------------------------------------------------------------------------
// Memory Fences
// ----------------------------------------------------------------------------

// Fence implementations adapted from SIMDe (https://github.com/simd-everywhere/simde)
// Uses C11/C++11 atomics or platform-specific intrinsics

#if !defined(__STDC_NO_ATOMICS__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
  #include <stdatomic.h>
#endif

static inline void _mm_sfence() {
#if defined(ATI_ARCH_ARM)
  // ARM: Data memory barrier - store (from original CLR code)
  __asm__ __volatile__("dmb st" ::: "memory");
#elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
  // GCC 4.7+: Use builtin atomic fence (adapted from SIMDe)
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#elif !defined(__INTEL_COMPILER) && !defined(__STDC_NO_ATOMICS__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
  // C11: Use standard atomic fence (adapted from SIMDe)
  #if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ < 9)
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  #else
    atomic_thread_fence(memory_order_seq_cst);
  #endif
#elif defined(_MSC_VER)
  // MSVC: Use MemoryBarrier (adapted from SIMDe)
  MemoryBarrier();
#elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1))
  // GCC 4.1+: Use sync builtin (adapted from SIMDe)
  __sync_synchronize();
#else
  // Fallback: Compiler barrier
  asm volatile("" ::: "memory");
#endif
}

static inline void _mm_mfence() {
#if defined(ATI_ARCH_ARM)
  // ARM: Data memory barrier - full (from original CLR code)
  __asm__ __volatile__("dmb sy" ::: "memory");
#elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
  // GCC 4.7+: Use builtin atomic fence (adapted from SIMDe)
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#elif !defined(__INTEL_COMPILER) && !defined(__STDC_NO_ATOMICS__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
  // C11: Use standard atomic fence (adapted from SIMDe)
  #if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ < 9)
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  #else
    atomic_thread_fence(memory_order_seq_cst);
  #endif
#elif defined(_MSC_VER)
  // MSVC: Use MemoryBarrier (adapted from SIMDe)
  MemoryBarrier();
#elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1))
  // GCC 4.1+: Use sync builtin (adapted from SIMDe)
  __sync_synchronize();
#else
  // Fallback: Compiler barrier
  asm volatile("" ::: "memory");
#endif
}

// ----------------------------------------------------------------------------
// Non-Temporal Stores
// ----------------------------------------------------------------------------

// These implementations are adapted from SIMDe (https://github.com/simd-everywhere/simde)
// On non-x86 platforms, non-temporal stores fall back to regular stores/memcpy

static inline void _mm_stream_si128(__m128i* mem_addr, __m128i a) {
  // Portable implementation (adapted from SIMDe)
  std::memcpy(mem_addr, &a, sizeof(a));
}

static inline void _mm_stream_si64(int64_t* mem_addr, int64_t a) {
  // Portable implementation (adapted from SIMDe)
  *mem_addr = a;
}

static inline void _mm_stream_si32(int32_t* mem_addr, int32_t a) {
  // Portable implementation (adapted from SIMDe)
  *mem_addr = a;
}

static inline void _mm256_stream_si256(__m256i* mem_addr, __m256i a) {
  // Portable implementation (adapted from SIMDe)
  std::memcpy(mem_addr, &a, sizeof(a));
}

static inline void _mm512_stream_si512(__m512i* mem_addr, __m512i a) {
  // Portable implementation (adapted from SIMDe)
  std::memcpy(mem_addr, &a, sizeof(a));
}

#endif // !ATI_ARCH_X86

#endif // PLATFORM_INTRINSICS_HPP_
