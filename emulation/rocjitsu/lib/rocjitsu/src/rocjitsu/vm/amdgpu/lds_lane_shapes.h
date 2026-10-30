// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_lane_shapes.h
/// @brief Address-shape proofs shared by DS loads and stores.
///
/// @details Each proof looks at one instruction's lanes and either fails or
/// returns a plan: the facts a fast path needs, already checked. A fast path
/// takes its plan as an argument, so it cannot run unproven, and it reuses what
/// the proof computed instead of recomputing it.
///
/// Loads and stores ask the same questions with a different limit:
///   loads   limit = materialized bytes  (anything past them reads as 0)
///   stores  limit = capacity            (anything past it is dropped; the
///                                        store grows the backing to plan.end)
///
/// Inputs are the address VGPR's raw 32-bit lanes and the instruction offset;
/// lane l is at (vgpr[l] + offset) mod 2^32. Every lane of the span is read,
/// active or not; the caller decides which spans that is allowed for.
///
/// Two facts make the proofs cheap and exact:
///   - the offset is the same for every lane, so it cancels from any shape:
///     shapes are proven on raw VGPR values, with no add per lane;
///   - a range inside the limit can't wrap 2^32, so a shape that holds mod 2^32
///     over an in-limit range touches exactly the bytes lane by lane would.
///
/// Layout: the proofs' logic (early rejects, bounds, plans) is written once, at
/// the bottom. Only the lane loops under it, in `lanes`, have two bodies: AVX2
/// intrinsics, and plain C++ for hosts without AVX2.

#ifndef ROCJITSU_VM_AMDGPU_LDS_LANE_SHAPES_H_
#define ROCJITSU_VM_AMDGPU_LDS_LANE_SHAPES_H_

#include "util/host_target.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#if RJ_HOST_HAS_AVX2
#include <immintrin.h>
#endif

namespace rocjitsu {
namespace amdgpu {
namespace lds_shapes {

/// @brief Every 16-lane group is one run of 16 * Width bytes, all inside the
/// limit.
struct BlockPlan {
  uint64_t end; ///< One past the highest byte any group touches.
};

/// @brief Lane i+16 is the 8 bytes right after lane i (b64, 32 lanes), inside
/// the limit.
struct ColumnPlan {};

/// @brief Every lane has the same address, inside the limit.
struct BroadcastPlan {
  uint64_t end;
};

/// @brief Every lane (any order) inside the limit.
struct LanesPlan {
  uint64_t end;
  bool wraps; ///< Some lane's vgpr + offset wraps 2^32.
};

// ---------------------------------------------------------------------------
// Lane loops: the part with two bodies.
// ---------------------------------------------------------------------------

namespace lanes {

#if RJ_HOST_HAS_AVX2

// Intrinsics: the same instructions the plain bodies compile to on a good day.
// They are here for control: clang's vectorization of the plain bodies broke
// silently four times while they were written (several accumulators, a folded
// lane, an out-of-line proof, lane kernels split out), each costing 5-80% on
// some path. Stores with these bodies: 11.67 ns on the census mix against
// 12.45 with the plain ones (lds_store/mkm3-run7.txt).

[[gnu::always_inline]] inline __m256i lanes8(const uint32_t *vgpr) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vgpr));
}

[[gnu::always_inline]] inline bool none(__m256i v) { return _mm256_testz_si256(v, v); }

/// j*W .. (j+7)*W: where lanes j..j+7 of a group sit relative to its first.
template <uint32_t Width> [[gnu::always_inline]] inline __m256i steps(uint32_t j) {
  return _mm256_setr_epi32(j * Width, (j + 1) * Width, (j + 2) * Width, (j + 3) * Width,
                           (j + 4) * Width, (j + 5) * Width, (j + 6) * Width, (j + 7) * Width);
}

/// Whether lane g+j holds vgpr[g] + j*Width in every 16-lane group.
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline bool ramps(const uint32_t *vgpr) {
  __m256i mismatch = _mm256_setzero_si256();
#pragma GCC unroll 4
  for (uint32_t group = 0; group < Span; group += 16) {
    const __m256i first = _mm256_set1_epi32(static_cast<int>(vgpr[group]));
    for (uint32_t j = 0; j < 16; j += 8)
      mismatch = _mm256_or_si256(
          mismatch,
          _mm256_xor_si256(_mm256_sub_epi32(lanes8(vgpr + group + j), steps<Width>(j)), first));
  }
  return none(mismatch);
}

/// Whether vgpr[i + 16] == vgpr[i] + 8 for every i < 16.
[[gnu::always_inline]] inline bool columns(const uint32_t *vgpr) {
  const __m256i eight = _mm256_set1_epi32(8);
  __m256i mismatch = _mm256_setzero_si256();
  for (uint32_t lane = 0; lane < 16; lane += 8)
    mismatch =
        _mm256_or_si256(mismatch, _mm256_xor_si256(lanes8(vgpr + lane + 16),
                                                   _mm256_add_epi32(lanes8(vgpr + lane), eight)));
  return none(mismatch);
}

/// Whether every lane of the span equals lane 0.
template <uint32_t Span> [[gnu::always_inline]] inline bool all_same(const uint32_t *vgpr) {
  const __m256i base = _mm256_set1_epi32(static_cast<int>(vgpr[0]));
  __m256i mismatch = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < Span; lane += 8)
    mismatch = _mm256_or_si256(mismatch, _mm256_xor_si256(lanes8(vgpr + lane), base));
  return none(mismatch);
}

/// max over the first N lanes of (vgpr[l] + offset) mod 2^32.
template <uint32_t N>
[[gnu::always_inline]] inline uint32_t max_lane(const uint32_t *vgpr, uint32_t offset) {
  const __m256i bias = _mm256_set1_epi32(static_cast<int>(offset));
  __m256i highest = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < N; lane += 8)
    highest = _mm256_max_epu32(highest, _mm256_add_epi32(lanes8(vgpr + lane), bias));
  __m128i x = _mm_max_epu32(_mm256_castsi256_si128(highest), _mm256_extracti128_si256(highest, 1));
  x = _mm_max_epu32(x, _mm_shuffle_epi32(x, 0x4E));
  x = _mm_max_epu32(x, _mm_shuffle_epi32(x, 0xB1));
  return static_cast<uint32_t>(_mm_cvtsi128_si32(x));
}

/// Whether (vgpr[l] + offset) mod 2^32 <= last for each of the first N lanes.
/// Tested in-register: folding to one max first, as max_lane does, cost loads
/// 0.25 ns on the census mix (12.23 -> 12.47, lds_load/mkm3-scalar-intrinsics.txt).
template <uint32_t N>
[[gnu::always_inline]] inline bool all_at_most(const uint32_t *vgpr, uint32_t offset,
                                               uint32_t last) {
  const __m256i bias = _mm256_set1_epi32(static_cast<int>(offset));
  const __m256i bound = _mm256_set1_epi32(static_cast<int>(last));
  __m256i highest = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < N; lane += 8)
    highest = _mm256_max_epu32(highest, _mm256_add_epi32(lanes8(vgpr + lane), bias));
  return none(_mm256_xor_si256(_mm256_max_epu32(highest, bound), bound));
}

#else

// Plain C++: each a flat loop over lanes into one accumulator, the form clang
// still turns into vectors where it can (SSE, NEON). Nested group/lane loops
// or several accumulators make it shuffle lanes instead: loads 13.92 ->
// 12.44 ns on the census mix once every proof took this form
// (lds_load/mkm3-scalar-final.txt).

/// x, hidden from the optimizer. Without it v[0] ^ v[0] folds to 0 and the
/// remaining 31 or 63 lanes get shuffled into vectors: broadcast b128 store
/// 15.2 -> 10.3 cycles with it (lds_store --drive 1172).
[[gnu::always_inline]] inline uint32_t opaque(uint32_t x) {
  asm("" : "+r"(x));
  return x;
}

template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline bool ramps(const uint32_t *vgpr) {
  uint32_t first[Span / 16];
  for (uint32_t group = 0; group < Span / 16; ++group)
    first[group] = vgpr[group * 16];
  uint32_t mismatch = 0;
  for (uint32_t lane = 0; lane < Span; ++lane)
    mismatch |= (vgpr[lane] - (lane % 16) * Width) ^ first[lane / 16];
  return mismatch == 0;
}

[[gnu::always_inline]] inline bool columns(const uint32_t *vgpr) {
  uint32_t mismatch = 0;
  for (uint32_t lane = 0; lane < 16; ++lane)
    mismatch |= vgpr[lane + 16] ^ (vgpr[lane] + 8);
  return mismatch == 0;
}

template <uint32_t Span> [[gnu::always_inline]] inline bool all_same(const uint32_t *vgpr) {
  const uint32_t base = opaque(vgpr[0]);
  uint32_t mismatch = 0;
  for (uint32_t lane = 0; lane < Span; ++lane)
    mismatch |= vgpr[lane] ^ base;
  return mismatch == 0;
}

template <uint32_t N>
[[gnu::always_inline]] inline uint32_t max_lane(const uint32_t *vgpr, uint32_t offset) {
  uint32_t highest = 0;
  for (uint32_t lane = 0; lane < N; ++lane)
    highest = std::max(highest, vgpr[lane] + offset);
  return highest;
}

template <uint32_t N>
[[gnu::always_inline]] inline bool all_at_most(const uint32_t *vgpr, uint32_t offset,
                                               uint32_t last) {
  return max_lane<N>(vgpr, offset) <= last;
}

#endif

} // namespace lanes

// ---------------------------------------------------------------------------
// The proofs. Forced inline: they are building blocks, and an out-of-line
// proof returning an optional put a call on the scattered store path (store
// broadcast 14.4 vs 10.8 cycles, b128 scattered 65 vs 50).
// ---------------------------------------------------------------------------

/// The highest start address whose Width bytes end inside the limit, clamped
/// to 32 bits (an LDS limit is far below 2^32). Requires limit >= Width.
[[gnu::always_inline]] inline uint32_t last_start(uint64_t limit, uint32_t width) {
  return static_cast<uint32_t>(std::min<uint64_t>(limit - width, UINT32_MAX));
}

/// Accepts: every 16-lane group is its own ramp.
///   ▁▂▃▄▅▆▇█ ▁▂▃▄▅▆▇█ ▁▂▃▄▅▆▇█ ▁▂▃▄▅▆▇█    (each ramp starts at its own base)
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline std::optional<BlockPlan>
prove_blocks(const uint32_t *vgpr, uint32_t offset, uint64_t limit) {
  // Every other shape fails here and skips the full proof: load census mix
  // 12.44 -> 12.14 ns (lds_load/mkm3-scalar-final.txt).
  if (vgpr[1] != vgpr[0] + Width)
    return std::nullopt;
  // The OR is >= every base, so one compare bounds all groups. It can reject a
  // valid set, never accept one past the limit.
  uint32_t bases = 0, highest = 0;
  for (uint32_t group = 0; group < Span; group += 16) {
    bases |= vgpr[group] + offset;
    highest = std::max(highest, vgpr[group] + offset);
  }
  if (uint64_t{bases} + 16 * Width > limit || !lanes::ramps<Width, Span>(vgpr))
    return std::nullopt;
  return BlockPlan{uint64_t{highest} + 16 * Width};
}

/// Accepts: lanes 16-31 repeat lanes 0-15's shape, 8 bytes higher.
///   ▃▁▆▂█▄▇▅ ▃▁▆▂█▄▇▅                      (scattered, but in matching pairs)
/// b64 read2 looks scattered lane to lane, but this holds for 54.5% of read2.
[[gnu::always_inline]] inline std::optional<ColumnPlan>
prove_column_pairs(const uint32_t *vgpr, uint32_t offset, uint64_t limit) {
  // Most b64 loads already fail on lane 0: no loop for them.
  if (limit < 16 || vgpr[16] != vgpr[0] + 8 || !lanes::columns(vgpr) ||
      !lanes::all_at_most<16>(vgpr, offset, last_start(limit, 16)))
    return std::nullopt;
  return ColumnPlan{};
}

/// Accepts: every lane the same address.
///   ▄▄▄▄▄▄▄▄
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline std::optional<BroadcastPlan>
prove_broadcast(const uint32_t *vgpr, uint32_t offset, uint64_t limit) {
  const uint64_t end = uint64_t{uint32_t(vgpr[0] + offset)} + Width;
  if (!lanes::all_same<Span>(vgpr) || end > limit)
    return std::nullopt;
  return BroadcastPlan{end};
}

/// Accepts: any order, all under the line.
///   ──────── limit - Width
///   ▅▁▇▂▃▆▄▂
/// One bounds pass, so the copies after it need no check per lane. Checking
/// each lane instead cost b64 loads 15.1 -> 21.6 ns (lds_load/mkm3-scalar-1.txt).
template <uint32_t Width, uint32_t Span>
[[gnu::always_inline]] inline std::optional<LanesPlan>
prove_lanes(const uint32_t *vgpr, uint32_t offset, uint64_t limit) {
  if (limit < Width || !lanes::all_at_most<Span>(vgpr, offset, last_start(limit, Width)))
    return std::nullopt;
  // Loads read neither field, so for them both passes below compile away.
  return LanesPlan{uint64_t{lanes::max_lane<Span>(vgpr, offset)} + Width,
                   uint64_t{lanes::max_lane<Span>(vgpr, 0)} + offset > UINT32_MAX};
}

} // namespace lds_shapes
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_LANE_SHAPES_H_
