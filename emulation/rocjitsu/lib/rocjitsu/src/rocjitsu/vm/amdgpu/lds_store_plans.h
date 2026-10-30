// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_store_plans.h
/// @brief Plain DS stores: the reference, then the rewrites that make it fast.
///
/// @details What a store means (Lds::store_lanes, exact_rule() below):
///
///   for each active lane l, lowest first:
///     write payload(l) at (vgpr[l] + offset) mod 2^32,
///       dropping any element past capacity, growing the backing to cover it
///
/// Every fast path is that loop rewritten. Each runs only on the plan its proof
/// returned (lds_lane_shapes.h, the proofs loads use too, with capacity as the
/// limit), and the comment on it says why the rewrite equals the loop.
///
///   Level 1  dispatch: by opcode, by EXEC, by address shape
///   Level 2  the rewrites
///
/// Census stores by rewrite (2.72B instructions):
///   blocks 75.1%   lane by lane 12.8%   lanes 7.6%   broadcast 4.4%
///
/// Standalone comparison against the run walk this replaces, on the census's
/// store instances (~/rj-lds-patterns/microbench/lds_store, Ryzen 9 9950X), ns
/// per DS store, census mix / runs16: 91.6 / 87.2 -> 11.7 / 6.1.

#ifndef ROCJITSU_VM_AMDGPU_LDS_STORE_PLANS_H_
#define ROCJITSU_VM_AMDGPU_LDS_STORE_PLANS_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/lds_lane_shapes.h"

#include <bit>
#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {
namespace lds_store {

/// @brief What a DS store takes from its data VGPRs.
struct StoreFormat {
  uint8_t elem_size = 4;
  uint8_t num_elems = 1;
  bool d16_hi = false; ///< Narrow stores take the upper 16 bits.
};

template <StoreFormat F> constexpr uint32_t kWidth = F.elem_size * F.num_elems;

/// @brief One plain DS store as the writers see it: register rows, not values.
struct StoreRequest {
  const uint32_t *addr;    ///< Address VGPR, one 32-bit value per lane.
  const uint32_t *data[4]; ///< Payload VGPRs (data_vgpr + i), one row each.
  uint64_t exec;
  uint32_t offset; ///< Byte offset, LDS base included.
};

/// @brief The Lds internals the writers use.
struct Access {
  static uint64_t capacity(const Lds &lds) { return lds.capacity_bytes_; }
  /// Grow the backing to cover [0, end); returns its data, which growing moves.
  static uint8_t *materialize(Lds &lds, uint64_t end) {
    lds.ensure_materialized(end);
    return lds.data_.data();
  }
  static void store_lanes(Lds &lds, const uint64_t *addrs, uint64_t lanes, uint32_t elem_size,
                          uint32_t num_elems, const uint8_t *src) {
    lds.store_lanes(addrs, lanes, elem_size, num_elems, src);
  }
};

namespace detail {

using lds_shapes::BlockPlan;
using lds_shapes::BroadcastPlan;
using lds_shapes::LanesPlan;

constexpr uint64_t prefix(uint32_t lanes) {
  return lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
}

template <StoreFormat F> RJ_NOINLINE void lane_by_lane(Lds &lds, const StoreRequest &r);
template <StoreFormat F, uint32_t Span> RJ_NOINLINE void by_shape(Lds &lds, const StoreRequest &r);

} // namespace detail

// ---------------------------------------------------------------------------
// Level 1: dispatch by opcode, by EXEC, by address shape.
// ---------------------------------------------------------------------------

/// @brief Store a plain DS store's payload rows to LDS.
/// @details By opcode: F is the opcode's payload format. By EXEC: the dense
/// rewrites write every lane of their span, so they only run when EXEC covers
/// it all; unlike loads, a store may not touch inactive lanes. Inlined into
/// the store's handler; everything behind it is out of line, so the 12.8% of
/// stores with at most 4 lanes pay no large frame (a single-lane store went
/// 14.2 -> 4.4 cycles, lds_store_bench --drive 2730).
template <StoreFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void store(Lds &lds, const StoreRequest &r) {
  using namespace detail;
  if (std::popcount(r.exec) <= 4) // 12.8% of stores, nearly all EXEC = 1
    return lane_by_lane<F>(lds, r);
  if constexpr (Wave == 64) {
    if (r.exec == prefix(64))
      return by_shape<F, 64>(lds, r);
    if (r.exec == prefix(48)) // 5.6% of stores: ds_write_b32 on lanes 0-47
      return by_shape<F, 48>(lds, r);
  }
  if (r.exec == prefix(32))
    return by_shape<F, 32>(lds, r);
  if (r.exec == prefix(16))
    return by_shape<F, 16>(lds, r);
  return lane_by_lane<F>(lds, r); // EXEC with holes: 0.2% of stores
}

namespace detail {

template <StoreFormat F, uint32_t Span>
void write_blocks(Lds &lds, const StoreRequest &r, BlockPlan plan);
template <StoreFormat F, uint32_t Span>
void write_highest_lane(Lds &lds, const StoreRequest &r, BroadcastPlan plan);
template <StoreFormat F, uint32_t Span>
void write_lanes(Lds &lds, const StoreRequest &r, LanesPlan plan);
template <StoreFormat F, uint32_t Span> void exact_span(Lds &lds, const StoreRequest &r);

/// By address shape. Out of line, one per format and span: the staging buffer
/// for wide payloads would otherwise sit in every caller's frame, single-lane
/// stores included.
template <StoreFormat F, uint32_t Span> void by_shape(Lds &lds, const StoreRequest &r) {
  constexpr uint32_t Width = kWidth<F>;
  const uint64_t capacity = Access::capacity(lds);
  // [[likely]] keeps the block store on the fall-through path. Without it:
  // b32 wave64 blocks (48% of all stores) 16.5 -> 17.0 cycles, the 48-lane
  // prefix 12.3 -> 13.3, broadcast 10.7 -> 13.1 (--drive 2610 / 2740 / 1172).
  if (auto plan = lds_shapes::prove_blocks<Width, Span>(r.addr, r.offset, capacity)) [[likely]]
    return write_blocks<F, Span>(lds, r, *plan);
  if (auto plan = lds_shapes::prove_broadcast<Width, Span>(r.addr, r.offset, capacity))
    return write_highest_lane<F, Span>(lds, r, *plan);
  if (auto plan = lds_shapes::prove_lanes<Width, Span>(r.addr, r.offset, capacity))
    return write_lanes<F, Span>(lds, r, *plan);
  return exact_span<F, Span>(lds, r); // some lane reaches past capacity
}

// ---------------------------------------------------------------------------
// Level 2: the rewrites.
// ---------------------------------------------------------------------------

/// Lane-major payload of the first Span lanes.
///   b32 (76% of stores): none. A VGPR row already holds lane 0, lane 1, ... in
///     order, so the row itself is the payload. Gathering it lane by lane is
///     what made a per-lane writer slower than the run walk (116 vs 93 ns).
///   b8 / b16 / d16_hi: a block store costs the same as b32's (16.3 vs 18.1
///     cycles, --drive 1214 / 2610).
///   b64 / b96 / b128: interleaved from the rows; clang leaves this scalar, and
///     an AVX2 zip / transpose is worth at most ~2% of store time.
template <StoreFormat F, uint32_t Span>
[[gnu::always_inline]] inline const uint8_t *payload(const StoreRequest &r, uint8_t *staging) {
  constexpr uint32_t Width = kWidth<F>;
  if constexpr (Width == 4) {
    return reinterpret_cast<const uint8_t *>(r.data[0]);
  } else if constexpr (Width < 4) {
    for (uint32_t lane = 0; lane < Span; ++lane) {
      const uint32_t value = F.d16_hi ? r.data[0][lane] >> 16 : r.data[0][lane];
      std::memcpy(staging + lane * Width, &value, Width);
    }
    return staging;
  } else {
    for (uint32_t lane = 0; lane < Span; ++lane)
      for (uint32_t i = 0; i < Width / 4; ++i)
        std::memcpy(staging + lane * Width + 4 * i, &r.data[i][lane], 4);
    return staging;
  }
}

/// One lane's payload bytes, straight from the data rows.
template <StoreFormat F>
[[gnu::always_inline]] inline void lane_payload(const StoreRequest &r, uint32_t lane,
                                                uint8_t *bytes) {
  if constexpr (kWidth<F> >= 4) {
    for (uint32_t i = 0; i < kWidth<F> / 4; ++i)
      std::memcpy(bytes + 4 * i, &r.data[i][lane], 4);
  } else {
    const uint32_t value = F.d16_hi ? r.data[0][lane] >> 16 : r.data[0][lane];
    std::memcpy(bytes, &value, kWidth<F>);
  }
}

/// The reference itself, for lanes nothing else may write. Out of line: only
/// lanes reaching past capacity get here.
template <StoreFormat F>
RJ_NOINLINE void exact_rule(Lds &lds, const StoreRequest &r, uint64_t lanes, const uint8_t *src) {
  uint64_t addrs[64];
  for (uint64_t rest = lanes; rest; rest &= rest - 1) {
    const uint32_t lane = std::countr_zero(rest);
    addrs[lane] = static_cast<uint32_t>(r.addr[lane] + r.offset);
  }
  Access::store_lanes(lds, addrs, lanes, F.elem_size, F.num_elems, src);
}

/// The reference for one lane. Its lane-major buffer lives here, out of the hot
/// frames.
template <StoreFormat F>
RJ_NOINLINE void exact_lane(Lds &lds, const StoreRequest &r, uint32_t lane) {
  alignas(16) uint8_t staged[64 * 16];
  lane_payload<F>(r, lane, staged + lane * kWidth<F>);
  exact_rule<F>(lds, r, uint64_t{1} << lane, staged);
}

template <StoreFormat F, uint32_t Span> void exact_span(Lds &lds, const StoreRequest &r) {
  alignas(64) uint8_t staging[64 * 16];
  exact_rule<F>(lds, r, r.exec, payload<F, Span>(r, staging));
}

/// Equals the loop because a group's lanes are consecutive and in order, so
/// none overlaps another lane of its group, and groups go lowest first. All of
/// it lies inside capacity, so nothing is dropped, and growing to plan.end
/// once equals growing as the loop goes: growth depends only on the highest
/// byte.
template <StoreFormat F, uint32_t Span>
[[gnu::always_inline]] inline void write_blocks(Lds &lds, const StoreRequest &r, BlockPlan plan) {
  constexpr uint32_t Width = kWidth<F>;
  uint8_t *data = Access::materialize(lds, plan.end);
  alignas(64) uint8_t staging[64 * 16];
  const uint8_t *src = payload<F, Span>(r, staging);
  const uint32_t offset = r.offset;
  for (uint32_t group = 0; group < Span; group += 16)
    std::memcpy(data + static_cast<uint32_t>(r.addr[group] + offset), src + group * Width,
                16 * Width);
}

/// Equals the loop because every lane writes the same bytes in order, so the
/// last, highest lane's bytes are what remain.
template <StoreFormat F, uint32_t Span>
[[gnu::always_inline]] inline void write_highest_lane(Lds &lds, const StoreRequest &r,
                                                      BroadcastPlan plan) {
  uint8_t bytes[16];
  lane_payload<F>(r, Span - 1, bytes);
  uint8_t *data = Access::materialize(lds, plan.end);
  std::memcpy(data + (plan.end - kWidth<F>), bytes, kWidth<F>);
}

/// Equals the loop because it is the loop, minus the per-lane bounds check
/// and growth the plan already settled.
template <StoreFormat F, uint32_t Span>
[[gnu::always_inline]] inline void write_lanes(Lds &lds, const StoreRequest &r, LanesPlan plan) {
  constexpr uint32_t Width = kWidth<F>;
  uint8_t *data = Access::materialize(lds, plan.end);
  alignas(64) uint8_t staging[64 * 16];
  const uint8_t *src = payload<F, Span>(r, staging);
  // Locals: stores through uint8_t* may alias r, and r.offset was reloaded
  // for every lane.
  const uint32_t offset = r.offset;
  const uint32_t *v = r.addr;
  // No lane wraps 2^32, so the offset folds into the base: three instructions
  // a lane instead of four. b32 wave64 88.7 -> 66.2 cycles (--drive 2663).
  // Wider lanes lost with it (b128 81 -> 135 cycles, --drive 1174), so b32
  // and narrower only.
  if (Width <= 4 && !plan.wraps) {
    uint8_t *base = data + offset;
#pragma GCC unroll 64
    for (uint32_t lane = 0; lane < Span; ++lane)
      std::memcpy(base + v[lane], src + lane * Width, Width);
  } else {
#pragma GCC unroll 64
    for (uint32_t lane = 0; lane < Span; ++lane)
      std::memcpy(data + static_cast<uint32_t>(v[lane] + offset), src + lane * Width, Width);
  }
}

/// The loop itself over the active lanes, for few lanes or EXEC with holes.
/// Each lane's check is its own one-lane plan.
template <StoreFormat F> void lane_by_lane(Lds &lds, const StoreRequest &r) {
  constexpr uint32_t Width = kWidth<F>;
  const uint32_t offset = r.offset; // see write_lanes
  const uint32_t *v = r.addr;
  const uint64_t capacity = Access::capacity(lds);
  for (uint64_t lanes = r.exec; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    const uint32_t at = v[lane] + offset;
    if (uint64_t{at} + Width > capacity) {
      exact_lane<F>(lds, r, lane);
      continue;
    }
    uint8_t bytes[16];
    lane_payload<F>(r, lane, bytes);
    uint8_t *data = Access::materialize(lds, uint64_t{at} + Width);
    std::memcpy(data + at, bytes, Width);
  }
}

} // namespace detail
} // namespace lds_store
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_STORE_PLANS_H_
