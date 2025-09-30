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

#ifndef CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_
#define CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_

#include <cstdint>
#include <cstring>

#if defined(__AVX__)
#if defined(__MINGW64__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif

namespace amd {

// ================================================================================================
#if IS_LINUX
__attribute__((optimize("unroll-all-loops"), always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
#if defined(ATI_ARCH_X86)
#if defined(__AVX512F__)
  for (auto i = 0u; i != size / sizeof(__m512i); ++i) {
    _mm512_stream_si512(reinterpret_cast<__m512i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m512i* __restrict&>(src)++);
  }
  size = size % sizeof(__m512i);
#endif

#if defined(__AVX__)
  for (auto i = 0u; i != size / sizeof(__m256i); ++i) {
    _mm256_stream_si256(reinterpret_cast<__m256i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m256i* __restrict&>(src)++);
  }
  size = size % sizeof(__m256i);
#endif

  for (auto i = 0u; i != size / sizeof(__m128i); ++i) {
    _mm_stream_si128(reinterpret_cast<__m128i* __restrict&>(dst)++,
                     *(reinterpret_cast<const __m128i* __restrict&>(src)++));
  }
  size = size % sizeof(__m128i);

  for (auto i = 0u; i != size / sizeof(long long); ++i) {
    _mm_stream_si64(reinterpret_cast<long long* __restrict&>(dst)++,
                    *reinterpret_cast<const long long* __restrict&>(src)++);
  }
  size = size % sizeof(long long);

  for (auto i = 0u; i != size / sizeof(int); ++i) {
    _mm_stream_si32(reinterpret_cast<int* __restrict&>(dst)++,
                    *reinterpret_cast<const int* __restrict&>(src)++);
  }

  size = size % sizeof(int);
  // Copy remaining bytes for unaligned size
  std::memcpy(dst, src, size);

  // Add memory fence
  _mm_sfence();
#else
  std::memcpy(dst, src, size);
#endif
}
#else
static inline void nontemporalMemcpy(void* __restrict dst, const void* __restrict src,
                                     size_t size) {
  std::memcpy(dst, src, size);
}
#endif
// ================================================================================================

/// @brief Nontemporal copy for entire AQL packet (64 bytes)
#if IS_LINUX
__attribute__((optimize("unroll-all-loops"), always_inline)) static inline void nontemporalCopyAQL(
    void* __restrict dst, const void* __restrict src) {
#if defined(ATI_ARCH_X86)
// Use nontemporal stores for 64-byte AQL packet copy if enabled, else fallback to memcpy
#if defined(__AVX512F__)
  // Use AVX-512 if available for a single 64-byte store
  __m512i data = _mm512_loadu_si512(src);
  _mm512_stream_si512(dst, data);
#elif defined(__AVX__)
  // Use AVX for two 32-byte stores
  __m256i data0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
  __m256i data1 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(src) + 32));
  _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), data0);
  _mm256_stream_si256(reinterpret_cast<__m256i*>(static_cast<char*>(dst) + 32), data1);
#else
  // Fallback to 4x 16-byte nontemporal stores
  for (int i = 0; i < 4; ++i) {
    __m128i data =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const char*>(src) + i * 16));
    _mm_stream_si128(reinterpret_cast<__m128i*>(static_cast<char*>(dst) + i * 16), data);
  }
#endif
#else
  std::memcpy(dst, src, 64);
#endif
}
#else
static inline void nontemporalCopyAQL(void* __restrict dst, const void* __restrict src) {
  std::memcpy(dst, src, 64);
}
#endif


/// @brief Ensure completion of all pending nontemporal stores (store fence)
#if IS_LINUX
__attribute__((always_inline)) static inline void nontemporalStoreFence() {
#if defined(ATI_ARCH_X86)
  _mm_sfence();
#endif
}
#else
static inline void nontemporalStoreFence() {}
#endif
}  // namespace amd

#endif  // CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_
