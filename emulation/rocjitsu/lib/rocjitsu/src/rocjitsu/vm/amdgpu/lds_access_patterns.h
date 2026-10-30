// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_access_patterns.h
/// @brief LDS vector-load kernels chosen by the shape of the lane addresses.
///
/// @details Both entry points fill dst[lane * stride ...] for every active lane
/// with exactly the bytes Lds::vector_load() produces, and leave the bytes of
/// inactive lanes unspecified. That freedom lets a dense mask be read as its
/// whole span of lanes. Lanes are read at their addresses truncated to 32
/// bits, as Lds::load_lanes() does; a range inside the materialized prefix
/// reads exactly those bytes and cannot wrap 2^32. Every other lane is handed
/// to @p slow, which applies the exact capacity and lazy-backing rules.
///
/// Cases, by share of census LDS load calls: one contiguous run (45%, most of
/// them whole wave64) and runs broken at 16-lane boundaries take one block
/// copy per 16-lane segment; at most four active lanes (25%) take one fixed
/// copy each; b64 pairs where lane l + 16 reads 8 bytes after lane l take one
/// 16-byte column per pair; other dense masks take one bounds check and a
/// fixed copy per lane.

#ifndef ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_
#define ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_

#include "rocjitsu/base/rj_compiler.h"
#include "util/host_target.h"

#include <bit>
#include <cstdint>
#include <cstring>

#if RJ_HOST_HAS_AVX2
#include <immintrin.h>
#endif

namespace rocjitsu {
namespace amdgpu {
namespace lds_patterns {

/// @brief The materialized prefix of an LDS backing.
struct Backing {
  const uint8_t *data;
  uint64_t size;
};

#if RJ_HOST_HAS_AVX2
namespace detail {

/// One fixed copy per active lane; lanes not fully backed go to @p slow.
template <uint32_t S, typename Slow>
inline void sparse(Backing backing, const uint64_t *addrs, uint64_t lane_mask, uint8_t *dst,
                   Slow &slow) {
  uint64_t late = 0;
  for (uint64_t lanes = lane_mask; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    if (addrs[lane] + S <= backing.size)
      std::memcpy(dst + lane * S, backing.data + addrs[lane], S);
    else
      late |= uint64_t{1} << lane;
  }
  if (late)
    slow(addrs, late);
}

/// Low dwords of addrs[0..7] in lane order 0, 1, 4, 5, 2, 3, 6, 7 (one shufps).
inline __m256i shuffled_dwords(const uint64_t *addrs) {
  const __m256 lo =
      _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(addrs)));
  const __m256 hi =
      _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(addrs + 4)));
  return _mm256_castps_si256(_mm256_shuffle_ps(lo, hi, 0x88));
}

/// lane * S for lanes base..base+7, in shuffled_dwords() order.
template <uint32_t S> inline __m256i shuffled_lane_offsets(uint32_t base) {
  return _mm256_setr_epi32((base + 0) * S, (base + 1) * S, (base + 4) * S, (base + 5) * S,
                           (base + 2) * S, (base + 3) * S, (base + 6) * S, (base + 7) * S);
}

/// lane * S for lanes base..base+7, in lane order.
template <uint32_t S> inline __m256i lane_offsets(uint32_t base) {
  return _mm256_setr_epi32((base + 0) * S, (base + 1) * S, (base + 2) * S, (base + 3) * S,
                           (base + 4) * S, (base + 5) * S, (base + 6) * S, (base + 7) * S);
}

/// All-ones for the active lanes of an 8-lane chunk, in shuffled_dwords() order.
inline __m256i shuffled_active(uint32_t chunk) {
  const __m256i bits = _mm256_setr_epi32(1, 2, 16, 32, 4, 8, 64, 128);
  return _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(chunk)), bits),
                            bits);
}

/// Whether every active span lane l sits at base + l * S (mod 2^32).
template <uint32_t S, uint32_t Span>
inline bool contiguous(const uint64_t *addrs, uint64_t lane_mask, uint32_t base) {
  const __m256i expect = _mm256_set1_epi32(static_cast<int>(base));
  __m256i diff = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < Span; lane += 8) {
    const __m256i at =
        _mm256_sub_epi32(shuffled_dwords(addrs + lane), shuffled_lane_offsets<S>(lane));
    diff = _mm256_or_si256(
        diff, _mm256_and_si256(_mm256_xor_si256(at, expect),
                               shuffled_active(static_cast<uint32_t>(lane_mask >> lane) & 0xFF)));
  }
  return _mm256_testz_si256(diff, diff);
}

/// Whether each 16-lane segment sits at its first lane's address plus l * S
/// (mod 2^32) and every segment block is backed. Census breaks between runs
/// fall on 16-lane boundaries.
template <uint32_t S, uint32_t Span> inline bool segments(const uint64_t *addrs, uint64_t backed) {
  __m256i diff = _mm256_setzero_si256();
  bool in_backing = true;
#pragma GCC unroll 4
  for (uint32_t lane = 0; lane < Span; lane += 16) {
    const auto start = static_cast<uint32_t>(addrs[lane]);
    in_backing &= uint64_t{start} + 16 * S <= backed;
    const __m256i expect = _mm256_set1_epi32(static_cast<int>(start));
    for (uint32_t at = 0; at < 16; at += 8)
      diff = _mm256_or_si256(diff,
                             _mm256_xor_si256(_mm256_sub_epi32(shuffled_dwords(addrs + lane + at),
                                                               shuffled_lane_offsets<S>(at)),
                                              expect));
  }
  return in_backing && _mm256_testz_si256(diff, diff);
}

/// Whether any span lane, active or not, is not fully backed.
template <uint32_t S, uint32_t Span> inline bool any_late(const uint64_t *addrs, uint64_t backed) {
  const __m256i last_ok = _mm256_set1_epi32(static_cast<int>(backed - S));
  __m256i late = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < Span; lane += 8) {
    const __m256i at = shuffled_dwords(addrs + lane);
    late = _mm256_or_si256(late, _mm256_xor_si256(_mm256_max_epu32(at, last_ok), last_ok));
  }
  return !_mm256_testz_si256(late, late);
}

/// A dense mask over a 32- or 64-lane span. Full means every span lane is active.
template <uint32_t S, uint32_t Span, bool Full, typename Slow>
inline void dense(Backing backing, const uint64_t *addrs, uint64_t lane_mask, uint8_t *dst,
                  Slow &slow) {
  if constexpr (Full) {
    if (segments<S, Span>(addrs, backing.size)) {
      for (uint32_t lane = 0; lane < Span; lane += 16)
        std::memcpy(dst + lane * S, backing.data + static_cast<uint32_t>(addrs[lane]), 16 * S);
      return;
    }
  } else {
    // Contiguous active lanes: one block. A backed block cannot wrap 2^32.
    const uint32_t first = std::countr_zero(lane_mask);
    const uint32_t last = 63 - std::countl_zero(lane_mask);
    const auto start = static_cast<uint32_t>(addrs[first]);
    const uint32_t size = (last - first + 1) * S;
    if (contiguous<S, Span>(addrs, lane_mask, start - first * S) &&
        uint64_t{start} + size <= backing.size) {
      std::memcpy(dst + first * S, backing.data + start, size);
      return;
    }
  }
  if (any_late<S, Span>(addrs, backing.size))
    return sparse<S>(backing, addrs, lane_mask, dst, slow);
  for (uint32_t lane = 0; lane < Span; ++lane)
    std::memcpy(dst + lane * S, backing.data + static_cast<uint32_t>(addrs[lane]), S);
}

template <uint32_t S, typename Slow>
inline void span_by_mask(Backing backing, const uint64_t *addrs, uint64_t lane_mask, uint8_t *dst,
                         Slow &slow) {
  if (std::popcount(lane_mask) <= 4 || backing.size < 16)
    return sparse<S>(backing, addrs, lane_mask, dst, slow);
  if (lane_mask >> 32)
    return lane_mask == ~uint64_t{0} ? dense<S, 64, true>(backing, addrs, lane_mask, dst, slow)
                                     : dense<S, 64, false>(backing, addrs, lane_mask, dst, slow);
  return lane_mask == 0xFFFF'FFFFu ? dense<S, 32, true>(backing, addrs, lane_mask, dst, slow)
                                   : dense<S, 32, false>(backing, addrs, lane_mask, dst, slow);
}

/// Native 32-bit lanes: expand the given lanes to 64-bit addresses for @p slow.
template <typename Slow>
RJ_NOINLINE void slow32(const uint32_t *lanes, uint32_t offset, uint64_t lane_mask, Slow &slow) {
  uint64_t addrs[64];
  for (uint64_t bits = lane_mask; bits; bits &= bits - 1) {
    const uint32_t lane = std::countr_zero(bits);
    addrs[lane] = static_cast<uint32_t>(lanes[lane] + offset);
  }
  slow(addrs, lane_mask);
}

template <uint32_t S, typename Slow>
inline void sparse32(Backing backing, const uint32_t *lanes, uint32_t offset, uint64_t lane_mask,
                     uint8_t *dst, Slow &slow) {
  uint64_t late = 0;
  for (uint64_t bits = lane_mask; bits; bits &= bits - 1) {
    const uint32_t lane = std::countr_zero(bits);
    const uint32_t at = lanes[lane] + offset;
    if (uint64_t{at} + S <= backing.size)
      std::memcpy(dst + lane * S, backing.data + at, S);
    else
      late |= uint64_t{1} << lane;
  }
  if (late)
    slow32(lanes, offset, late, slow);
}

/// Two rows of 16 b64 lanes where lane l + 16 reads the 8 bytes after lane l:
/// load one 16-byte column per pair and unzip it into both rows.
inline bool columns32(Backing backing, const uint32_t *lanes, uint32_t offset, uint8_t *dst) {
  if (backing.size < 16 || lanes[16] != lanes[0] + 8)
    return false;
  const __m256i bias = _mm256_set1_epi32(static_cast<int>(offset));
  __m256i diff = _mm256_setzero_si256(), maximum = _mm256_setzero_si256();
  for (uint32_t lane = 0; lane < 16; lane += 8) {
    const __m256i first = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes + lane));
    const __m256i second = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes + lane + 16));
    diff = _mm256_or_si256(diff,
                           _mm256_xor_si256(second, _mm256_add_epi32(first, _mm256_set1_epi32(8))));
    maximum = _mm256_max_epu32(maximum, _mm256_add_epi32(first, bias));
  }
  const __m256i last_ok = _mm256_set1_epi32(static_cast<int>(backing.size - 16));
  diff = _mm256_or_si256(diff, _mm256_xor_si256(last_ok, _mm256_max_epu32(maximum, last_ok)));
  if (!_mm256_testz_si256(diff, diff))
    return false;
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < 16; lane += 2) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
        backing.data + static_cast<uint32_t>(lanes[lane] + offset)));
    const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
        backing.data + static_cast<uint32_t>(lanes[lane + 1] + offset)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + lane * 8), _mm_unpacklo_epi64(x, y));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + (lane + 16) * 8), _mm_unpackhi_epi64(x, y));
  }
  return true;
}

/// A dense mask over a 16- to 64-lane span of native 32-bit lanes.
template <uint32_t S, uint32_t Span, typename Slow>
inline void dense32(Backing backing, const uint32_t *lanes, uint32_t offset, uint64_t lane_mask,
                    uint8_t *dst, Slow &slow) {
  if constexpr (S == 8 && Span == 32)
    if (columns32(backing, lanes, offset, dst))
      return;
  // Contiguous 16-lane segments. The common offset cancels in the shape proof;
  // it is needed only at the segment bases. The OR of the bases bounds them all.
  __m256i diff = _mm256_setzero_si256();
  uint32_t starts = 0;
#pragma GCC unroll 4
  for (uint32_t lane = 0; lane < Span; lane += 16) {
    starts |= lanes[lane] + offset;
    const __m256i base = _mm256_set1_epi32(static_cast<int>(lanes[lane]));
    for (uint32_t at = 0; at < 16; at += 8) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes + lane + at));
      diff =
          _mm256_or_si256(diff, _mm256_xor_si256(_mm256_sub_epi32(v, lane_offsets<S>(at)), base));
    }
  }
  if (uint64_t{starts} + 16 * S <= backing.size && _mm256_testz_si256(diff, diff)) {
    for (uint32_t lane = 0; lane < Span; lane += 16)
      std::memcpy(dst + lane * S, backing.data + static_cast<uint32_t>(lanes[lane] + offset),
                  16 * S);
    return;
  }
  if (backing.size < S)
    return sparse32<S>(backing, lanes, offset, lane_mask, dst, slow);
  // One bounds reduction over the span, then a fixed copy per lane.
  const __m256i bias = _mm256_set1_epi32(static_cast<int>(offset));
  __m256i maximum = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < Span; lane += 8)
    maximum = _mm256_max_epu32(
        maximum, _mm256_add_epi32(
                     bias, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes + lane))));
  const __m256i last_ok = _mm256_set1_epi32(static_cast<int>(backing.size - S));
  const __m256i late = _mm256_xor_si256(last_ok, _mm256_max_epu32(maximum, last_ok));
  if (!_mm256_testz_si256(late, late))
    return sparse32<S>(backing, lanes, offset, lane_mask, dst, slow);
  for (uint32_t lane = 0; lane < Span; ++lane)
    std::memcpy(dst + lane * S, backing.data + static_cast<uint32_t>(lanes[lane] + offset), S);
}

template <uint32_t S, typename Slow>
inline void span32_by_mask(Backing backing, const uint32_t *lanes, uint32_t offset,
                           uint64_t lane_mask, uint8_t *dst, Slow &slow) {
  if (std::popcount(lane_mask) <= 4)
    return sparse32<S>(backing, lanes, offset, lane_mask, dst, slow);
  if (lane_mask >> 48)
    return dense32<S, 64>(backing, lanes, offset, lane_mask, dst, slow);
  if (lane_mask >> 32)
    return dense32<S, 48>(backing, lanes, offset, lane_mask, dst, slow);
  if (lane_mask >> 16)
    return dense32<S, 32>(backing, lanes, offset, lane_mask, dst, slow);
  return dense32<S, 16>(backing, lanes, offset, lane_mask, dst, slow);
}

} // namespace detail
#endif

/// @brief Load from 64-bit lane addresses.
/// @details Reads the address of every lane of the span (lanes 0-31, or 0-63
/// when a lane above 31 is active), which must be initialized; @p dst must
/// hold that many lanes. @p slow(addrs, lanes) loads the given lanes exactly.
template <typename Slow>
inline void load(Backing backing, const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                 uint32_t num_elems, uint8_t *dst, Slow &&slow) {
#if RJ_HOST_HAS_AVX2
  using namespace detail;
  switch (elem_size * num_elems) {
  case 1:
    return span_by_mask<1>(backing, addrs, lane_mask, dst, slow);
  case 2:
    return span_by_mask<2>(backing, addrs, lane_mask, dst, slow);
  case 4:
    return span_by_mask<4>(backing, addrs, lane_mask, dst, slow);
  case 8:
    return span_by_mask<8>(backing, addrs, lane_mask, dst, slow);
  case 16:
    return span_by_mask<16>(backing, addrs, lane_mask, dst, slow);
  case 12:
    return sparse<12>(backing, addrs, lane_mask, dst, slow);
  default:
    break;
  }
#else
  (void)backing, (void)elem_size, (void)num_elems, (void)dst;
#endif
  slow(addrs, lane_mask);
}

/// @brief Load from 32-bit lane values plus a common offset: lane l reads at
/// (lanes[l] + offset) mod 2^32, as ds_lane_addresses() computes it.
/// @details Reads the lanes of the span (the highest active lane rounded up to a
/// multiple of 16), which must be initialized; @p dst must hold that many lanes.
/// @p slow(addrs, lanes) loads the given lanes exactly from 64-bit addresses.
template <typename Slow>
inline void load32(Backing backing, const uint32_t *lanes, uint32_t offset, uint64_t lane_mask,
                   uint32_t elem_size, uint32_t num_elems, uint8_t *dst, Slow &&slow) {
#if RJ_HOST_HAS_AVX2
  using namespace detail;
  switch (elem_size * num_elems) {
  case 1:
    return span32_by_mask<1>(backing, lanes, offset, lane_mask, dst, slow);
  case 2:
    return span32_by_mask<2>(backing, lanes, offset, lane_mask, dst, slow);
  case 4:
    return span32_by_mask<4>(backing, lanes, offset, lane_mask, dst, slow);
  case 8:
    return span32_by_mask<8>(backing, lanes, offset, lane_mask, dst, slow);
  case 16:
    return span32_by_mask<16>(backing, lanes, offset, lane_mask, dst, slow);
  default:
    return slow32(lanes, offset, lane_mask, slow);
  }
#else
  (void)backing, (void)elem_size, (void)num_elems, (void)dst;
  uint64_t addrs[64];
  for (uint64_t bits = lane_mask; bits; bits &= bits - 1) {
    const uint32_t lane = std::countr_zero(bits);
    addrs[lane] = static_cast<uint32_t>(lanes[lane] + offset);
  }
  slow(addrs, lane_mask);
#endif
}

} // namespace lds_patterns
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_
