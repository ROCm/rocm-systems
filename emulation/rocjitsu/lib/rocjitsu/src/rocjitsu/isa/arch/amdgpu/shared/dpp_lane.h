// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "util/bit.h"
#include <array>

namespace rocjitsu::amdgpu::dpp {

/// Row size for DPP operations (16 lanes per row).
constexpr int ROW_SIZE = 16;
/// Number of banks per row (4 banks of 4 lanes each).
constexpr int NUM_BANKS = 4;

/// @brief Compute the source lane index for a DPP permutation.
///
/// @param dpp_ctrl 9-bit DPP control value.
/// @param lane Current lane index (0..wf_size-1).
/// @param wf_size Wavefront size (32 or 64).
/// @param[out] out_of_bounds Set to true if the source lane is invalid.
/// @returns Source lane to read from.
inline int dpp_permute(uint32_t dpp_ctrl, int lane, int wf_size, bool &out_of_bounds) {
  out_of_bounds = false;
  int row_num = lane / ROW_SIZE;
  int row_off = lane % ROW_SIZE;

  if (dpp_ctrl <= QUAD_PERM_MAX) {
    // Quad permute: 4 lanes per quad, 2-bit selector per lane position.
    int quad_base = lane & ~3;
    int quad_idx = lane & 3;
    int new_idx = (dpp_ctrl >> (2 * quad_idx)) & 3;
    return quad_base | new_idx;
  }

  if (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) {
    // row_shl N: data shifts left (toward lower lane indices).
    // Lane K reads from lane K+N (higher index).
    int shift = dpp_ctrl - ROW_SHL1 + 1;
    int new_off = row_off + shift;
    if (new_off >= ROW_SIZE) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) {
    // row_shr N: data shifts right (toward higher lane indices).
    // Lane K reads from lane K-N (lower index).
    int shift = dpp_ctrl - ROW_SHR1 + 1;
    int new_off = row_off - shift;
    if (new_off < 0) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl >= ROW_ROR1 && dpp_ctrl <= ROW_ROR_MAX) {
    // row_ror N: data rotates right within the row.
    // Lane K reads from lane (K-N+ROW_SIZE) % ROW_SIZE.
    int rot = dpp_ctrl - ROW_ROR1 + 1;
    int new_off = (row_off - rot + ROW_SIZE) % ROW_SIZE;
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl == WF_SHL1) {
    // Wave shift left 1: lane K reads from lane K+1.
    int src = lane + 1;
    if (src >= wf_size)
      out_of_bounds = true;
    return src < wf_size ? src : lane;
  }

  if (dpp_ctrl == WF_ROL1) {
    // Wave rotate left 1: lane K reads from lane (K+1) % wf_size.
    return (lane + 1) % wf_size;
  }

  if (dpp_ctrl == WF_SRL1) {
    // Wave shift right 1: lane K reads from lane K-1.
    int src = lane - 1;
    if (src < 0)
      out_of_bounds = true;
    return src >= 0 ? src : lane;
  }

  if (dpp_ctrl == WF_ROR1) {
    // Wave rotate right 1: lane K reads from lane (K-1+wf_size) % wf_size.
    return (lane - 1 + wf_size) % wf_size;
  }

  if (dpp_ctrl == ROW_MIRROR) {
    return row_num * ROW_SIZE + (ROW_SIZE - 1 - row_off);
  }

  if (dpp_ctrl == ROW_HALF_MIRROR) {
    int half_base = lane & ~7;
    int half_off = lane & 7;
    return half_base | (7 - half_off);
  }

  if (dpp_ctrl == ROW_BCAST15) {
    // Broadcast lane 15 of each row to the following row. The first row
    // retains each lane's source.
    if (lane < ROW_SIZE)
      return lane;
    return row_num * ROW_SIZE - 1;
  }

  if (dpp_ctrl == ROW_BCAST31) {
    // Broadcast lane 31 to lanes 32-63. The first half retains its sources.
    if (lane < 32 || wf_size <= 32)
      return lane;
    return 31;
  }

  if (dpp_ctrl >= ROW_SHARE_BASE && dpp_ctrl <= ROW_SHARE_MAX) {
    // row_share/row_newbcast: broadcast one selected source lane within the
    // destination row.
    int lane_sel = dpp_ctrl - ROW_SHARE_BASE;
    return row_num * ROW_SIZE + lane_sel;
  }

  if (dpp_ctrl >= ROW_XMASK_BASE && dpp_ctrl <= ROW_XMASK_MAX) {
    // row_xmask: XOR the lane offset within the row with a 4-bit mask.
    int mask = dpp_ctrl - ROW_XMASK_BASE;
    int new_off = row_off ^ mask;
    if (new_off >= ROW_SIZE) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  // Unknown dpp_ctrl — identity.
  return lane;
}

/// @brief Check if a lane is disabled by DPP row/bank masks.
///
/// @param lane Lane index.
/// @param row_mask 4-bit row mask (bit N enables row N, 16 lanes/row).
///        For wave32, only bits 0-1 are meaningful (rows 0-1 cover lanes 0-31);
///        bits 2-3 have no effect since no lanes map to rows 2-3.
/// @param bank_mask 4-bit bank mask (bit N enables bank N, 4 lanes/bank).
/// @returns True if the lane is disabled (should not be written).
inline bool dpp_lane_masked(int lane, uint32_t row_mask, uint32_t bank_mask) {
  int row = lane / ROW_SIZE;
  int bank = (lane % ROW_SIZE) / NUM_BANKS;
  return ((row_mask & (1u << row)) == 0) || ((bank_mask & (1u << bank)) == 0);
}

/// @brief Check if a DPP instruction writes the destination lane.
///
/// Row/bank masks always disable writes. When the DPP permutation has invalid
/// shared data, BOUND_CTRL=0 disables the write and BOUND_CTRL=1 writes using a
/// zero source value.
inline bool dpp_lane_write_enabled(int lane, int wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                                   uint32_t bank_mask, uint32_t bound_ctrl) {
  if (dpp_lane_masked(lane, row_mask, bank_mask))
    return false;

  bool oob = false;
  (void)dpp_permute(dpp_ctrl, lane, wf_size, oob);
  return !oob || bound_ctrl != 0;
}

/// @brief Execution-local DPP source and destination analysis.
///
/// Keeping these masks separate is essential: row/bank masking is a
/// destination rule, while BOUND_CTRL decides whether an invalid source writes
/// a zero-derived result or suppresses the write.
struct DppPlan {
  static constexpr uint8_t INVALID_LANE = 0xFF;

  std::array<uint8_t, 64> source_lanes{};
  uint64_t physical_read_dest_mask = 0;
  uint64_t zero_source_mask = 0;
  uint64_t source_write_mask = 0;
  uint64_t row_bank_mask = 0;
};

inline uint64_t dpp_row_bank_mask(uint32_t wf_size, uint32_t row_mask, uint32_t bank_mask) {
  uint64_t mask = 0;
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    if (!dpp_lane_masked(static_cast<int>(lane), row_mask, bank_mask))
      mask |= uint64_t{1} << lane;
  return mask;
}

inline DppPlan compute_dpp_plan(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                                uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                                uint64_t exec_mask, bool inactive_uses_bound_ctrl) {
  DppPlan plan;
  plan.row_bank_mask = dpp_row_bank_mask(wf_size, row_mask, bank_mask);
  plan.source_lanes.fill(DppPlan::INVALID_LANE);
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    bool out_of_bounds = false;
    const int source_lane =
        dpp_permute(dpp_ctrl, static_cast<int>(lane), static_cast<int>(wf_size), out_of_bounds);
    // FI applies only to inactive in-range source lanes. Out-of-range sources
    // are governed by BOUND_CTRL alone, irrespective of FI.
    const bool inactive = !out_of_bounds && !fi && (exec_mask & (uint64_t{1} << source_lane)) == 0;
    const uint64_t lane_bit = uint64_t{1} << lane;
    if (!out_of_bounds)
      plan.source_lanes[lane] = static_cast<uint8_t>(source_lane);
    if (!out_of_bounds && !inactive)
      plan.physical_read_dest_mask |= lane_bit;
    else if (bound_ctrl || (inactive && !inactive_uses_bound_ctrl))
      plan.zero_source_mask |= lane_bit;
    if (bound_ctrl || (!out_of_bounds && !(inactive && inactive_uses_bound_ctrl)))
      plan.source_write_mask |= lane_bit;
  }
  return plan;
}

inline DppPlan make_dpp_plan(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                             uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                             uint64_t exec_mask, bool inactive_uses_bound_ctrl) {
  const uint64_t full_wave = util::mask<uint64_t>(wf_size);
  if ((exec_mask & full_wave) != full_wave || row_mask != 0xf || bank_mask != 0xf)
    return compute_dpp_plan(wf_size, dpp_ctrl, row_mask, bank_mask, bound_ctrl, fi, exec_mask,
                            inactive_uses_bound_ctrl);

  // With every source active, only the permutation, the wave width and
  // BOUND_CTRL affect the plan. Lane values are still read for each issue.
  struct Entry {
    uint32_t width = 0, control = 0, bound = 0;
    DppPlan plan;
  };
  static thread_local std::array<Entry, 16> entries{};
  auto &entry = entries[(dpp_ctrl ^ (dpp_ctrl >> 4)) % entries.size()];
  if (entry.width != wf_size || entry.control != dpp_ctrl || entry.bound != bound_ctrl) {
    entry = {wf_size, dpp_ctrl, bound_ctrl,
             compute_dpp_plan(wf_size, dpp_ctrl, row_mask, bank_mask, bound_ctrl, fi, exec_mask,
                              inactive_uses_bound_ctrl)};
  }
  return entry.plan;
}

inline uint64_t dpp_physical_source_mask(const DppPlan &plan, uint64_t destination_mask,
                                         uint32_t wf_size) {
  uint64_t source_mask = 0;
  const uint64_t read_destinations = destination_mask & plan.physical_read_dest_mask;
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    if (read_destinations & (uint64_t{1} << lane))
      source_mask |= uint64_t{1} << plan.source_lanes[lane];
  return source_mask;
}

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel) {
  uint32_t sel = (lane_sel >> ((lane & 7u) * 3u)) & 7u;
  return (lane & ~7u) | sel;
}

inline DppPlan make_dpp8_plan(uint32_t wf_size, uint32_t lane_sel, uint32_t fi,
                              uint64_t exec_mask) {
  DppPlan plan;
  plan.source_lanes.fill(DppPlan::INVALID_LANE);
  plan.row_bank_mask = ~0ULL;
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    const uint32_t source_lane = dpp8_src_lane(lane, lane_sel);
    const uint64_t lane_bit = uint64_t{1} << lane;
    if (source_lane < wf_size)
      plan.source_lanes[lane] = static_cast<uint8_t>(source_lane);
    if (source_lane < wf_size && (fi || (exec_mask & (uint64_t{1} << source_lane))))
      plan.physical_read_dest_mask |= lane_bit;
    else
      plan.zero_source_mask |= lane_bit;
    plan.source_write_mask |= lane_bit;
  }
  return plan;
}

} // namespace rocjitsu::amdgpu::dpp
