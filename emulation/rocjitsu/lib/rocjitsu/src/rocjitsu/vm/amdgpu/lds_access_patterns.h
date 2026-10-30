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
/// 32-bit lanes; its section below reads top-down, decision first.

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

// ---------------------------------------------------------------------------
// load32(): native 32-bit lanes. Read top-down, in three levels:
//
//   Level 1  the decision: which Strategy serves this load
//   Level 2  the strategies: what each one copies, and why that is exact
//   Level 3  the proofs: the AVX2 checks behind each decision
//
// Census vector_load calls by the strategy that serves them:
//
//   Strategy       share  when
//   Blocks         52.9%  every 16-lane group reads consecutive bytes
//   CheckedLanes   24.8%  at most 4 active lanes (almost always 1)
//   LaneCopies     13.7%  dense, scattered, and every lane is backed
//   ColumnPairs     8.5%  b64 read2: lane i+16 reads the 8 bytes after lane i
//   (exact rule)    0.1%  b96, and any lane reaching past the backing
// ---------------------------------------------------------------------------

/// One DS load, as every load32 strategy sees it. Everything that takes it is
/// inlined, so it lives in registers. Two layout rules, both measured:
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

/// How a dense load (more than 4 active lanes) is served.
enum class Strategy {
  Blocks,       // one memcpy per 16-lane group
  ColumnPairs,  // b64: 16-byte columns, unzipped into two output rows
  LaneCopies,   // one fixed-size copy per lane, no bounds or EXEC checks
  CheckedLanes, // one bounds-checked copy per active lane
};

// The strategies (level 2) and proofs (level 3), defined below.
template <uint32_t S, uint32_t Span> void copy_blocks(const Load32 &r);
inline void copy_column_pairs(const Load32 &r);
template <uint32_t S, uint32_t Span> void copy_lanes(const Load32 &r);
template <uint32_t S, typename Slow> void checked_lanes(const Load32 &r, Slow &slow);

template <uint32_t S, uint32_t Span> bool is_blocks(const Load32 &r);
inline bool is_column_pairs(const Load32 &r);
template <uint32_t S, uint32_t Span> bool all_backed(const Load32 &r);

// Level 1: the decision.

/// Pick the cheapest strategy that is exact for this load. Order matters only
/// for speed: each check is cheaper to fail than the next strategy is to run.
template <uint32_t S, uint32_t Span> Strategy choose(const Load32 &r) {
  if constexpr (S == 8 && Span == 32)
    if (is_column_pairs(r))
      return Strategy::ColumnPairs;
  if (is_blocks<S, Span>(r))
    return Strategy::Blocks;
  if (all_backed<S, Span>(r))
    return Strategy::LaneCopies;
  return Strategy::CheckedLanes; // some lane reaches past the backing
}

/// More than 4 active lanes. Bytes of inactive lanes are unspecified, so the
/// dense strategies treat every lane of the span as active: EXEC drops out.
template <uint32_t S, uint32_t Span, typename Slow> void dense32(const Load32 &r, Slow &slow) {
  switch (choose<S, Span>(r)) {
  case Strategy::Blocks:
    return copy_blocks<S, Span>(r);
  case Strategy::ColumnPairs:
    return copy_column_pairs(r);
  case Strategy::LaneCopies:
    return copy_lanes<S, Span>(r);
  case Strategy::CheckedLanes:
    return checked_lanes<S>(r, slow);
  }
}

template <uint32_t S, typename Slow> void load32(const Load32 &r, Slow &slow) {
  // A few lanes: plain per-lane copies beat any setup.
  if (std::popcount(r.exec) <= 4)
    return checked_lanes<S>(r, slow);
  // Otherwise the span is the 16-lane groups up to the last active lane.
  const uint32_t last = 63 - std::countl_zero(r.exec);
  if (last >= 48)
    return dense32<S, 64>(r, slow);
  if (last >= 32)
    return dense32<S, 48>(r, slow);
  if (last >= 16)
    return dense32<S, 32>(r, slow);
  return dense32<S, 16>(r, slow);
}

// Level 2: the strategies. Each runs only after its proof (level 3) passed.

/// Group g (lanes g..g+15) is the 16 * S bytes at lane g's address.
template <uint32_t S, uint32_t Span> void copy_blocks(const Load32 &r) {
#pragma GCC unroll 4
  for (uint32_t g = 0; g < Span; g += 16)
    std::memcpy(r.dst + g * S, r.data + r.address(g), 16 * S);
}

/// The surprise in b64 read2: adjacent lanes look scattered (0, 128, 272, ...),
/// but lane i+16 always reads the 8 bytes right after lane i. So one 16-byte
/// load fetches a column [lane i | lane i+16], and two neighbouring columns
/// unzip into two lanes of each output row.
inline void copy_column_pairs(const Load32 &r) {
#pragma GCC unroll 8
  for (uint32_t i = 0; i < 16; i += 2) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r.data + r.address(i)));
    const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r.data + r.address(i + 1)));
    // x = [lane i | lane i+16], y = [lane i+1 | lane i+17]
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r.dst + i * 8), _mm_unpacklo_epi64(x, y));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r.dst + (i + 16) * 8), _mm_unpackhi_epi64(x, y));
  }
}

/// Every lane of the span is backed, so each is a plain fixed-size copy.
template <uint32_t S, uint32_t Span> void copy_lanes(const Load32 &r) {
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

/// Active lanes only, each bounds-checked; misses take the exact rule.
template <uint32_t S, typename Slow> void checked_lanes(const Load32 &r, Slow &slow) {
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

// Level 3: the proofs. Eight lanes per AVX2 op; all run on 32-bit values.
//
// Two tricks keep them cheap:
//   - the offset is the same for every lane, so it cancels from any shape
//     check: shapes are proven on the raw VGPR values, with no add per lane;
//   - a backed range cannot wrap 2^32, so "the shape holds mod 2^32 and the
//     range is backed" means each lane reads exactly the bytes we copy.

/// Lanes at..at+7 of the address VGPR.
inline __m256i lanes8(const uint32_t *vgpr) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vgpr));
}

/// j*S .. (j+7)*S: where lanes j..j+7 of a group sit relative to its first lane.
template <uint32_t S> inline __m256i steps(uint32_t j) {
  return _mm256_setr_epi32(j * S, (j + 1) * S, (j + 2) * S, (j + 3) * S, (j + 4) * S, (j + 5) * S,
                           (j + 6) * S, (j + 7) * S);
}

/// Nonzero exactly in the lanes where v > limit (unsigned).
inline __m256i above(__m256i v, uint64_t limit) {
  const __m256i l = _mm256_set1_epi32(static_cast<int>(limit));
  return _mm256_xor_si256(_mm256_max_epu32(v, l), l);
}

inline bool none(__m256i v) { return _mm256_testz_si256(v, v); }

/// Blocks: lane g+j holds vgpr[g] + j*S in every group, and every group's
/// 16 * S bytes are backed.
template <uint32_t S, uint32_t Span> bool is_blocks(const Load32 &r) {
  __m256i mismatch = _mm256_setzero_si256();
  uint32_t bases = 0;
#pragma GCC unroll 4
  for (uint32_t g = 0; g < Span; g += 16) {
    bases |= r.address(g);
    const __m256i first = _mm256_set1_epi32(static_cast<int>(r.vgpr[g]));
    for (uint32_t j = 0; j < 16; j += 8)
      mismatch = _mm256_or_si256(
          mismatch, _mm256_xor_si256(_mm256_sub_epi32(lanes8(r.vgpr + g + j), steps<S>(j)), first));
  }
  // The OR of the bases is >= every base, so one compare bounds all groups.
  // It may reject a valid set, never accept one reading past the backing.
  return uint64_t{bases} + 16 * S <= r.backed && none(mismatch);
}

/// ColumnPairs: b64 over 32 lanes, lane i+16 holds vgpr[i] + 8 for every
/// i < 16, and every 16-byte column is backed.
inline bool is_column_pairs(const Load32 &r) {
  // Cheap reject first: most b64 loads already fail on lane 0.
  if (r.backed < 16 || r.vgpr[16] != r.vgpr[0] + 8)
    return false;
  const __m256i offset = _mm256_set1_epi32(static_cast<int>(r.offset));
  const __m256i eight = _mm256_set1_epi32(8);
  __m256i mismatch = _mm256_setzero_si256(), highest = _mm256_setzero_si256();
  for (uint32_t i = 0; i < 16; i += 8) {
    const __m256i first_row = lanes8(r.vgpr + i), second_row = lanes8(r.vgpr + i + 16);
    mismatch =
        _mm256_or_si256(mismatch, _mm256_xor_si256(second_row, _mm256_add_epi32(first_row, eight)));
    highest = _mm256_max_epu32(highest, _mm256_add_epi32(first_row, offset));
  }
  return none(_mm256_or_si256(mismatch, above(highest, r.backed - 16)));
}

/// LaneCopies: every lane of the span (active or not) is backed.
template <uint32_t S, uint32_t Span> bool all_backed(const Load32 &r) {
  if (r.backed < S)
    return false;
  const __m256i offset = _mm256_set1_epi32(static_cast<int>(r.offset));
  __m256i highest = _mm256_setzero_si256();
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < Span; lane += 8)
    highest = _mm256_max_epu32(highest, _mm256_add_epi32(lanes8(r.vgpr + lane), offset));
  return none(above(highest, r.backed - S));
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
  const detail::Load32 r{backing.data, backing.size, lanes, dst, lane_mask, offset};
  switch (elem_size * num_elems) {
  case 1:
    return detail::load32<1>(r, slow);
  case 2:
    return detail::load32<2>(r, slow);
  case 4:
    return detail::load32<4>(r, slow);
  case 8:
    return detail::load32<8>(r, slow);
  case 16:
    return detail::load32<16>(r, slow);
  default: // b96 (0.1% of calls): not worth a fast path
    return detail::exact_rule(lanes, offset, lane_mask, slow);
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
