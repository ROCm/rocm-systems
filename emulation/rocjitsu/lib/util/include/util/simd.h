// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native-width register transport blocks. These carry raw lane data between
// observed operands and arithmetic calls; they provide no numeric operations.
// Backend-specific types keep vector widths and compiled helper symbols distinct.

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include "util/bit.h"
#include "util/simd_control.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// Dynamic Wave32/Wave64 transport loops already move whole vectors. Prevent
// Clang from adding a four-way unrolled loop and remainder dispatch around them.
#if defined(__clang__)
#define RJ_REGISTER_NO_UNROLL _Pragma("clang loop unroll(disable) interleave(disable)")
#else
#define RJ_REGISTER_NO_UNROLL
#endif

namespace util {

// Keep feature-dependent definitions distinct when baseline and wider-target
// objects are linked together. Each backend owns its vector type and helpers.
#if defined(__AVX512F__)
inline namespace register_transport_avx512 {
#elif defined(__AVX2__)
inline namespace register_transport_avx2 {
#else
inline namespace register_transport_scalar {
#endif

#if defined(__AVX512F__)
inline constexpr std::size_t register_block_bytes = 64;
#else
inline constexpr std::size_t register_block_bytes = 32;
#endif

// A block matches the backend's vector width; Wave32 and Wave64 divide exactly.
// Backend-dependent types must not cross compilation-target boundaries.
// Keep all operations as copies or integer masks; this is not a numeric vector API.
template <class T> class alignas(register_block_bytes) RegisterBlock {
public:
  static_assert(sizeof(T) == 4 || sizeof(T) == 8);
  static constexpr std::size_t kLanes = register_block_bytes / sizeof(T);
  static constexpr std::size_t size() { return kLanes; }
  RegisterBlock() = default;
  explicit RegisterBlock(T value) {
#if defined(__AVX512F__)
    if constexpr (sizeof(T) == 4) {
      bits_ = _mm512_set1_epi32(std::bit_cast<int32_t>(value));
    } else {
      bits_ = _mm512_set1_epi64(std::bit_cast<int64_t>(value));
    }
#elif defined(__AVX2__)
    if constexpr (sizeof(T) == 4) {
      bits_ = _mm256_set1_epi32(std::bit_cast<int32_t>(value));
    } else {
      bits_ = _mm256_set1_epi64x(std::bit_cast<int64_t>(value));
    }
#else
    for (T &lane : lanes_) {
      lane = value;
    }
#endif
  }
  explicit RegisterBlock(const T *values) {
#if defined(__AVX512F__)
    bits_ = _mm512_loadu_si512(values);
#elif defined(__AVX2__)
    bits_ = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
#else
    std::memcpy(lanes_, values, sizeof(lanes_));
#endif
  }
  void copy_to(T *values) const {
#if defined(__AVX512F__)
    _mm512_storeu_si512(values, bits_);
#elif defined(__AVX2__)
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(values), bits_);
#else
    std::memcpy(values, lanes_, sizeof(lanes_));
#endif
  }
  T operator[](std::size_t lane) const {
#if defined(__AVX512F__) || defined(__AVX2__)
    T values[kLanes];
    copy_to(values);
    return values[lane];
#else
    return lanes_[lane];
#endif
  }
  RegisterBlock &operator&=(const RegisterBlock &other)
    requires std::is_integral_v<T>
  {
#if defined(__AVX512F__)
    bits_ = _mm512_and_si512(bits_, other.bits_);
#elif defined(__AVX2__)
    bits_ = _mm256_and_si256(bits_, other.bits_);
#else
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      lanes_[lane] &= other.lanes_[lane];
    }
#endif
    return *this;
  }

private:
#if defined(__AVX512F__)
  __m512i bits_{};
#elif defined(__AVX2__)
  __m256i bits_{};
#else
  T lanes_[kLanes]{};
#endif
};

template <class T> constexpr std::size_t register_block_lanes_v = RegisterBlock<T>::size();
inline constexpr std::size_t register_block_lanes64 = RegisterBlock<uint64_t>::size();

// Reinterpret contiguous words without numeric conversion.
template <class T> RegisterBlock<T> load(const uint32_t *source) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  return std::bit_cast<RegisterBlock<T>>(RegisterBlock<uint32_t>(source));
}

// Repeat a raw word in each lane.
template <class T> RegisterBlock<T> broadcast(uint32_t bits) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  return RegisterBlock<T>(std::bit_cast<T>(bits));
}

// Publish active raw words without writing inactive destination lanes.
template <class T> void masked_store(uint32_t *destination, RegisterBlock<T> value, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
#if defined(__AVX512F__)
  const auto bits = std::bit_cast<__m512i>(value);
  // GCC benefits from an unmasked full block; Clang prefers a uniform masked
  // store in the short, rolled caller loops.
#if !defined(__clang__)
  if ((mask & 0xffffu) == 0xffffu) {
    _mm512_storeu_si512(destination, bits);
    return;
  }
#endif
  _mm512_mask_storeu_epi32(destination, __mmask16(mask), bits);
#else
  auto bits = std::bit_cast<RegisterBlock<uint32_t>>(value);
  if ((mask & util::mask<uint64_t>(RegisterBlock<T>::size())) ==
      util::mask<uint64_t>(RegisterBlock<T>::size())) {
    bits.copy_to(destination);
    return;
  }
  // AVX2 hardware masked stores are slower on Zen4 than selected scalar stores.
  alignas(RegisterBlock<uint32_t>) uint32_t lanes[RegisterBlock<T>::size()];
  bits.copy_to(lanes);
  for (std::size_t lane = 0; lane < RegisterBlock<T>::size(); ++lane) {
    if ((mask >> lane) & 1) {
      destination[lane] = lanes[lane];
    }
  }
#endif
}

// Copy raw words to scratch storage for delegated operand publication.
template <class T>
void blit_to_buffer(uint32_t (&buffer)[RegisterBlock<T>::size()], RegisterBlock<T> value) {
  std::bit_cast<RegisterBlock<uint32_t>>(value).copy_to(buffer);
}

// Join the low and high VGPR of each 64-bit lane.
template <class T> RegisterBlock<T> load64(const uint32_t *low, const uint32_t *high) {
  static_assert(sizeof(T) == sizeof(uint64_t));
#if defined(__AVX512F__)
  const auto lo = _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(low)));
  const auto hi =
      _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(high)));
  return std::bit_cast<RegisterBlock<T>>(_mm512_or_si512(lo, _mm512_slli_epi64(hi, 32)));
#else
  // Keep individual words visible to scalar publication. Eager AVX2 shuffles
  // added work to the partial-mask path instead of improving its packing.
  alignas(RegisterBlock<uint64_t>) uint64_t bits[register_block_lanes64];
  for (std::size_t lane = 0; lane < register_block_lanes64; ++lane) {
    bits[lane] = (uint64_t(high[lane]) << 32) | low[lane];
  }
  return std::bit_cast<RegisterBlock<T>>(RegisterBlock<uint64_t>(bits));
#endif
}

// Repeat a raw pair of words in each lane.
template <class T> RegisterBlock<T> broadcast64(uint64_t bits) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  return RegisterBlock<T>(std::bit_cast<T>(bits));
}

// Split active 64-bit lanes into their two destination VGPRs. Both pointers
// name the same lane offset in whole VGPRs; they may name the same VGPR.
template <class T>
void masked_store64(uint32_t *low, uint32_t *high, RegisterBlock<T> value, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint64_t));
#if defined(__AVX512F__)
  // Integer narrowing stores select one 32-bit word from each packed lane.
  const auto bits = std::bit_cast<__m512i>(value);
  _mm512_mask_cvtepi64_storeu_epi32(low, __mmask8(mask), bits);
  _mm512_mask_cvtepi64_storeu_epi32(high, __mmask8(mask), _mm512_srli_epi64(bits, 32));
#else
  alignas(RegisterBlock<uint64_t>) uint64_t bits[register_block_lanes64];
  std::bit_cast<RegisterBlock<uint64_t>>(value).copy_to(bits);
  for (std::size_t lane = 0; lane < register_block_lanes64; ++lane) {
    if ((mask >> lane) & 1) {
      low[lane] = uint32_t(bits[lane]);
      high[lane] = uint32_t(bits[lane] >> 32);
    }
  }
#endif
}

#if defined(__AVX512F__)
} // namespace register_transport_avx512
#elif defined(__AVX2__)
} // namespace register_transport_avx2
#else
} // namespace register_transport_scalar
#endif

} // namespace util

#endif // UTIL_SIMD_H_
