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
/// load() cases, by share of census LDS load calls: one contiguous run (45%,
/// most of them whole wave64) and runs broken at 16-lane boundaries take one
/// block copy per 16-lane segment; at most four active lanes (25%) take one
/// fixed copy each; b64 pairs where lane l + 16 reads 8 bytes after lane l
/// take one 16-byte column per pair; other dense masks take one bounds check
/// and a fixed copy per lane. load32() serves the same shapes from native
/// 32-bit lanes, on the shared proofs in lds_lane_shapes.h, and also on hosts
/// without AVX2; its section below reads top-down, dispatch first.

#ifndef ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_
#define ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/lds_lane_shapes.h"
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

} // namespace detail
#endif

// ---------------------------------------------------------------------------
// load32(): native 32-bit lanes. The reference, then the rewrites that make it
// fast. What a load means (Lds::load_lanes, the exact rule):
//
//   for each active lane l:
//     read S bytes at (lanes[l] + offset) mod 2^32 into dst[l],
//       an element past capacity reading 0, bytes past the backing reading 0
//
// Every fast path is that loop rewritten. Each runs only on the plan its proof
// returned (lds_lane_shapes.h, with the materialized size as the limit), and
// the comment on it says why the rewrite equals the loop.
//
//   Level 1  dispatch: by width, by wave, by address shape
//   Level 2  the rewrites
//
// Census vector_load calls by rewrite:
//   blocks 52.9%   lane by lane 24.8%   lanes 13.7%   column pairs 8.5%
// ---------------------------------------------------------------------------

namespace detail {

using lds_shapes::BlockPlan;
using lds_shapes::ColumnPlan;
using lds_shapes::LanesPlan;

/// One DS load, as every rewrite sees it. Everything that takes it is inlined,
/// so it lives in registers. Two layout rules, both measured:
///   - never pass it to an out-of-line call: once its address escapes, every
///     dst store forces its fields to be reloaded (+0.7 ns on runs16);
///   - keep the 32-bit offset after the 64-bit fields: in the call's argument
///     order, clang packs the fields into one ymm at entry and extracts them
///     on use (+0.9 ns on the census mix).
struct Load32 {
  const uint8_t *data;
  uint64_t backed;      // materialized prefix; bytes past it read 0
  const uint32_t *vgpr; // address VGPR: one 32-bit value per lane
  uint8_t *dst;
  uint64_t exec;
  uint32_t offset; // instruction offset, added modulo 2^32

  uint32_t address(uint32_t lane) const { return vgpr[lane] + offset; }
};

// The rewrites (level 2), defined below.
template <uint32_t S, uint32_t Span>
[[gnu::always_inline]] inline void copy_blocks(const Load32 &r, const uint8_t **view,
                                               BlockPlan plan);
[[gnu::always_inline]] inline void copy_column_pairs(const Load32 &r, ColumnPlan plan);
template <uint32_t S, uint32_t Span>
[[gnu::always_inline]] inline void copy_lanes(const Load32 &r, LanesPlan plan);
template <uint32_t S, typename Slow>
[[gnu::always_inline]] inline void lane_by_lane(const Load32 &r, Slow &slow);

// Level 1: dispatch by width (load32() below), by wave, by address shape.

/// By address shape, over a span of more than 4 active lanes. Bytes of
/// inactive lanes are unspecified, so every lane of the span counts as active:
/// unlike stores, no EXEC check in any proof or copy.
template <uint32_t S, uint32_t Span, typename Slow>
[[gnu::always_inline]] inline void by_shape(const Load32 &r, Slow &slow, const uint8_t **view) {
  // The column pattern only occurs in b64 read2 on 32 lanes (8.5% of calls),
  // so no other instantiation pays for its proof.
  if constexpr (S == 8 && Span == 32)
    if (auto plan = lds_shapes::prove_column_pairs(r.vgpr, r.offset, r.backed))
      return copy_column_pairs(r, *plan);
  if (auto plan = lds_shapes::prove_blocks<S, Span>(r.vgpr, r.offset, r.backed))
    return copy_blocks<S, Span>(r, view, *plan);
  if (auto plan = lds_shapes::prove_lanes<S, Span>(r.vgpr, r.offset, r.backed))
    return copy_lanes<S, Span>(r, *plan);
  return lane_by_lane<S>(r, slow); // some lane reaches past the backing
}

/// By wave: how far the active lanes reach.
template <uint32_t S, uint32_t Wave, typename Slow>
[[gnu::always_inline]] inline void load32(const Load32 &r, Slow &slow, const uint8_t **view) {
  // A quarter of all calls, nearly all single-lane: plain per-lane copies beat
  // any setup.
  if (std::popcount(r.exec) <= 4)
    return lane_by_lane<S>(r, slow);
  // Otherwise the span is the 16-lane groups up to the last active lane; the
  // census's address runs break at 16 lanes.
  const uint32_t last = 63 - std::countl_zero(r.exec);
  if constexpr (Wave == 64) {
    if (last >= 48)
      return by_shape<S, 64>(r, slow, view);
    if (last >= 32)
      return by_shape<S, 48>(r, slow, view);
  }
  if (last >= 16)
    return by_shape<S, 32>(r, slow, view);
  return by_shape<S, 16>(r, slow, view);
}

// Level 2: the rewrites.

/// Equals the loop because a group's lanes are consecutive, in order, and all
/// backed: group g (lanes g..g+15) is the 16 * S bytes at lane g's address.
/// One block per 16-lane group rather than per wave: ds_read_b32 often
/// restarts its run every 16 lanes. With a @p view, point at the bytes in place
/// instead of copying them.
template <uint32_t S, uint32_t Span>
[[gnu::always_inline]] inline void copy_blocks(const Load32 &r, const uint8_t **view, BlockPlan) {
#pragma GCC unroll 4
  for (uint32_t g = 0; g < Span; g += 16) {
    if (view)
      view[g / 16] = r.data + r.address(g);
    else
      std::memcpy(r.dst + g * S, r.data + r.address(g), 16 * S);
  }
}

/// Equals the loop because lane i+16's 8 bytes follow lane i's, so one 16-byte
/// read covers both. The surprise in b64 read2: adjacent lanes look scattered
/// (0, 128, 272, ...), but each 16-byte column is [lane i | lane i+16], and
/// two neighbouring columns unzip into two lanes of each output row.
[[gnu::always_inline]] inline void copy_column_pairs(const Load32 &r, ColumnPlan) {
#if RJ_HOST_HAS_AVX2
#pragma GCC unroll 8
  for (uint32_t i = 0; i < 16; i += 2) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r.data + r.address(i)));
    const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r.data + r.address(i + 1)));
    // x = [lane i | lane i+16], y = [lane i+1 | lane i+17]
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r.dst + i * 8), _mm_unpacklo_epi64(x, y));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r.dst + (i + 16) * 8), _mm_unpackhi_epi64(x, y));
  }
#else
  for (uint32_t i = 0; i < 16; ++i) {
    const uint8_t *column = r.data + r.address(i);
    std::memcpy(r.dst + i * 8, column, 8);
    std::memcpy(r.dst + (i + 16) * 8, column + 8, 8);
  }
#endif
}

/// Equals the loop because it is the loop, minus the per-lane bounds check the
/// plan already settled.
template <uint32_t S, uint32_t Span>
[[gnu::always_inline]] inline void copy_lanes(const Load32 &r, LanesPlan) {
#pragma GCC unroll 64
  for (uint32_t lane = 0; lane < Span; ++lane)
    std::memcpy(r.dst + lane * S, r.data + r.address(lane), S);
}

/// The exact rule for the given lanes: @p slow gets their 64-bit addresses.
/// Cold, so out of line, and it takes the load's fields, not the load (see
/// Load32).
template <typename Slow>
RJ_NOINLINE void exact_rule(const uint32_t *vgpr, uint32_t offset, uint64_t lanes, Slow &slow) {
  uint64_t addrs[64];
  for (uint64_t rest = lanes; rest; rest &= rest - 1) {
    const uint32_t lane = std::countr_zero(rest);
    addrs[lane] = static_cast<uint32_t>(vgpr[lane] + offset);
  }
  slow(addrs, lanes);
}

/// The loop itself over the active lanes, each checked; lanes past the backing
/// take the exact rule.
template <uint32_t S, typename Slow>
[[gnu::always_inline]] inline void lane_by_lane(const Load32 &r, Slow &slow) {
  uint64_t late = 0;
  for (uint64_t lanes = r.exec; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    const uint32_t address = r.address(lane);
    if (uint64_t{address} + S <= r.backed)
      std::memcpy(r.dst + lane * S, r.data + address, S);
    else
      late |= uint64_t{1} << lane;
  }
  if (late)
    exact_rule(r.vgpr, r.offset, late, slow);
}

} // namespace detail

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
/// @p lane_mask has no lanes at or past @p Wave.
///
/// With a @p view, the result is read through it instead: lanes 16g..16g+15
/// are the 16 * stride bytes at view[g], which is @p dst + 16g * stride unless
/// those lanes read consecutive LDS bytes, when it points at them in place.
template <uint32_t Wave = 64, typename Slow>
[[gnu::always_inline]] inline void load32(Backing backing, const uint32_t *lanes, uint32_t offset,
                                          uint64_t lane_mask, uint32_t elem_size,
                                          uint32_t num_elems, uint8_t *dst, Slow &&slow,
                                          const uint8_t **view = nullptr) {
  if (view)
    for (uint32_t g = 0; g < Wave / 16; ++g)
      view[g] = dst + g * 16 * elem_size * num_elems;
  const detail::Load32 r{backing.data, backing.size, lanes, dst, lane_mask, offset};
  switch (elem_size * num_elems) {
  case 1:
    return detail::load32<1, Wave>(r, slow, view);
  case 2:
    return detail::load32<2, Wave>(r, slow, view);
  case 4:
    return detail::load32<4, Wave>(r, slow, view);
  case 8:
    return detail::load32<8, Wave>(r, slow, view);
  case 16:
    return detail::load32<16, Wave>(r, slow, view);
  default: // b96 (0.1% of calls): not worth a fast path
    return detail::exact_rule(lanes, offset, lane_mask, slow);
  }
}

} // namespace lds_patterns
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_ACCESS_PATTERNS_H_
