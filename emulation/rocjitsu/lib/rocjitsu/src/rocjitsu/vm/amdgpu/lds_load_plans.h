// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_load_plans.h
/// @brief Plain DS loads: the reference, then the rewrites that make it fast.
///
/// @details What a load means (Lds::load_lanes, exact_rule() below):
///
///   for each active lane l:
///     read Width bytes at (vgpr[l] + offset) mod 2^32 into lane l,
///       an element past capacity reading 0, bytes past the backing reading 0
///
/// Every fast path is that loop rewritten. Each runs only on the plan its proof
/// returned (lds_lane_shapes.h, the proofs stores use too, with the
/// materialized size as the limit), and the comment on it says why the rewrite
/// equals the loop.
///
///   Level 1  dispatch: by width, by wave, by address shape
///   Level 2  the rewrites
///
/// Unlike a store, a load may write any lane of its span: the bytes of
/// inactive lanes are unspecified, and the writeback reads active lanes only.
/// That is what lets a dense mask be proven and copied as a whole span.
///
/// Census loads by rewrite: blocks are the most common by far; lane by lane
/// comes next, nearly all of it single-lane loads; then lanes in any order, and
/// the b64 column pairs of read2.
///
/// Against the run walk this replaces (Lds::vector_load, which compares every
/// lane's address with the previous one to find runs), a load costs several
/// times less on the census mix.

#ifndef ROCJITSU_VM_AMDGPU_LDS_LOAD_PLANS_H_
#define ROCJITSU_VM_AMDGPU_LDS_LOAD_PLANS_H_

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
namespace lds_load {

/// @brief The materialized prefix of an LDS backing.
struct Backing {
  const uint8_t *data;
  uint64_t size;
};

namespace detail {

using lds_shapes::BlockPlan;
using lds_shapes::ColumnPlan;
using lds_shapes::LanesPlan;

/// One DS load, as every rewrite sees it. Everything that takes it is inlined,
/// so it lives in registers. Two layout rules, both measured:
///   - never pass it to an out-of-line call: once its address escapes, every
///     dst store forces its fields to be reloaded;
///   - keep the 32-bit offset after the 64-bit fields: in the call's argument
///     order, clang packs the fields into one ymm at entry and extracts them
///     on use.
struct LoadRequest {
  const uint8_t *data;
  uint64_t backed;      // materialized prefix; bytes past it read 0
  const uint32_t *addr; // address VGPR: one 32-bit value per lane
  uint8_t *dst;
  uint64_t exec;
  uint32_t offset; // instruction offset, added modulo 2^32

  uint32_t address(uint32_t lane) const { return addr[lane] + offset; }
};

// Defined below: the rest of level 1, then the rewrites (level 2).
template <uint32_t Width, uint32_t Wave, typename Slow>
[[gnu::always_inline]] inline void by_wave(const LoadRequest &r, Slow &slow, const uint8_t **view);
template <uint32_t Width, uint32_t Span, typename Slow>
[[gnu::always_inline]] inline void by_shape(const LoadRequest &r, Slow &slow, const uint8_t **view);
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline void copy_blocks(const LoadRequest &r, const uint8_t **view,
                                               BlockPlan plan);
[[gnu::always_inline]] inline void copy_column_pairs(const LoadRequest &r, ColumnPlan plan);
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline void copy_lanes(const LoadRequest &r, LanesPlan plan);
template <uint32_t Width, typename Slow>
[[gnu::always_inline]] inline void lane_by_lane(const LoadRequest &r, Slow &slow);
template <typename Slow>
RJ_NOINLINE void exact_rule(const uint32_t *addr, uint32_t offset, uint64_t lanes, Slow &slow);

} // namespace detail

// ---------------------------------------------------------------------------
// Level 1: dispatch by width, by wave, by address shape.
// ---------------------------------------------------------------------------

/// @brief Load from 32-bit lane values plus a common offset: lane l reads at
/// (lanes[l] + offset) mod 2^32, as ds_lane_addresses() computes it.
/// @details By width: the dispatch the opcode fixes, so it folds away in a
/// handler that inlines it. Reads the lanes of the span (the highest active
/// lane rounded up to a multiple of 16), which must be initialized; @p dst
/// must hold that many lanes. @p slow(addrs, lanes) loads the given lanes
/// exactly from 64-bit addresses. @p lane_mask has no lanes at or past @p Wave.
///
/// With a @p view, the result is read through it instead: lanes 16g..16g+15
/// are the 16 * stride bytes at view[g], which is @p dst + 16g * stride unless
/// those lanes read consecutive LDS bytes, when it points at them in place.
template <uint32_t Wave = 64, typename Slow>
[[gnu::always_inline]] inline void load(Backing backing, const uint32_t *lanes, uint32_t offset,
                                        uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
                                        uint8_t *dst, Slow &&slow, const uint8_t **view = nullptr) {
  if (view)
    for (uint32_t g = 0; g < Wave / 16; ++g)
      view[g] = dst + g * 16 * elem_size * num_elems;
  const detail::LoadRequest r{backing.data, backing.size, lanes, dst, lane_mask, offset};
  switch (elem_size * num_elems) {
  case 1:
    return detail::by_wave<1, Wave>(r, slow, view);
  case 2:
    return detail::by_wave<2, Wave>(r, slow, view);
  case 4:
    return detail::by_wave<4, Wave>(r, slow, view);
  case 8:
    return detail::by_wave<8, Wave>(r, slow, view);
  case 16:
    return detail::by_wave<16, Wave>(r, slow, view);
  default: // b96 is too rare to be worth a fast path
    return detail::exact_rule(lanes, offset, lane_mask, slow);
  }
}

namespace detail {

/// By wave: how far the active lanes reach.
template <uint32_t Width, uint32_t Wave, typename Slow>
[[gnu::always_inline]] inline void by_wave(const LoadRequest &r, Slow &slow, const uint8_t **view) {
  // Many loads, nearly all single-lane: plain per-lane copies beat any setup.
  if (std::popcount(r.exec) <= 4)
    return lane_by_lane<Width>(r, slow);
  // Otherwise the span is the 16-lane groups up to the last active lane; the
  // census's address runs break at 16 lanes.
  const uint32_t last = 63 - std::countl_zero(r.exec);
  if constexpr (Wave == 64) {
    if (last >= 48)
      return by_shape<Width, 64>(r, slow, view);
    if (last >= 32)
      return by_shape<Width, 48>(r, slow, view);
  }
  if (last >= 16)
    return by_shape<Width, 32>(r, slow, view);
  return by_shape<Width, 16>(r, slow, view);
}

/// By address shape, over a span of more than 4 active lanes. Every lane of
/// the span counts as active: unlike stores, no EXEC check in any proof or
/// copy.
template <uint32_t Width, uint32_t Span, typename Slow>
[[gnu::always_inline]] inline void by_shape(const LoadRequest &r, Slow &slow,
                                            const uint8_t **view) {
  // The column pattern only occurs in b64 read2 on 32 lanes, so no other
  // instantiation pays for its proof.
  if constexpr (Width == 8 && Span == 32)
    if (auto plan = lds_shapes::prove_column_pairs(r.addr, r.offset, r.backed))
      return copy_column_pairs(r, *plan);
  if (auto plan = lds_shapes::prove_blocks<Width, Span>(r.addr, r.offset, r.backed))
    return copy_blocks<Width, Span>(r, view, *plan);
  if (auto plan = lds_shapes::prove_lanes<Width, Span>(r.addr, r.offset, r.backed))
    return copy_lanes<Width, Span>(r, *plan);
  return lane_by_lane<Width>(r, slow); // some lane reaches past the backing
}

// ---------------------------------------------------------------------------
// Level 2: the rewrites.
// ---------------------------------------------------------------------------

/// Equals the loop because a group's lanes are consecutive, in order, and all
/// backed: group g (lanes g..g+15) is the 16 * Width bytes at lane g's address.
/// One block per 16-lane group rather than per wave: ds_read_b32 often
/// restarts its run every 16 lanes. With a @p view, point at the bytes in place
/// instead of copying them.
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline void copy_blocks(const LoadRequest &r, const uint8_t **view,
                                               BlockPlan) {
#pragma GCC unroll 4
  for (uint32_t g = 0; g < Span; g += 16) {
    if (view)
      view[g / 16] = r.data + r.address(g);
    else
      std::memcpy(r.dst + g * Width, r.data + r.address(g), 16 * Width);
  }
}

/// Equals the loop because lane i+16's 8 bytes follow lane i's, so one 16-byte
/// read covers both. The surprise in b64 read2: adjacent lanes look scattered
/// (0, 128, 272, ...), but each 16-byte column is [lane i | lane i+16].
[[gnu::always_inline]] inline void copy_column_pairs(const LoadRequest &r, ColumnPlan) {
#if RJ_HOST_HAS_AVX2
  // Two neighbouring columns unzip into two lanes of each output row, so each
  // row gets 16-byte stores instead of 8-byte ones.
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
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline void copy_lanes(const LoadRequest &r, LanesPlan) {
#pragma GCC unroll 64
  for (uint32_t lane = 0; lane < Span; ++lane)
    std::memcpy(r.dst + lane * Width, r.data + r.address(lane), Width);
}

/// The loop itself over the active lanes, each checked; lanes past the backing
/// take the exact rule.
template <uint32_t Width, typename Slow>
[[gnu::always_inline]] inline void lane_by_lane(const LoadRequest &r, Slow &slow) {
  uint64_t late = 0;
  for (uint64_t lanes = r.exec; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    const uint32_t address = r.address(lane);
    if (uint64_t{address} + Width <= r.backed)
      std::memcpy(r.dst + lane * Width, r.data + address, Width);
    else
      late |= uint64_t{1} << lane;
  }
  if (late)
    exact_rule(r.addr, r.offset, late, slow);
}

/// The reference itself for the given lanes: @p slow gets their 64-bit
/// addresses. Cold, so out of line, and it takes the load's fields, not the
/// load (see LoadRequest).
template <typename Slow>
RJ_NOINLINE void exact_rule(const uint32_t *addr, uint32_t offset, uint64_t lanes, Slow &slow) {
  uint64_t addrs[64];
  lds_shapes::exact_addresses(addr, offset, lanes, addrs);
  slow(addrs, lanes);
}

} // namespace detail
} // namespace lds_load
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_LOAD_PLANS_H_
