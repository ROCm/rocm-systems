// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_DISPATCH_ENTRY_H_
#define ROCJITSU_VM_AMDGPU_DISPATCH_ENTRY_H_

/// @file dispatch_entry.h
/// @brief Per-dispatch tracking entry for the command processor pipeline.
///
/// @details Analogous to gem5's HSAQueueEntry. Each kernel dispatch gets a
/// unique dispatch_id and tracks workgroup lifecycle (dispatched vs completed)
/// independently. Completion signals fire when all WGs of a dispatch finish,
/// in per-queue submission order.

#include <cassert>
#include <cstdint>
#include <deque>

namespace rocjitsu {
namespace amdgpu {

struct WorkgroupCoord {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
};

struct ClusterDispatchShape {
  uint32_t count_x = 0;
  uint32_t count_y = 0;
  uint32_t count_z = 0;
  uint32_t size_x = 1;
  uint32_t size_y = 1;
  uint32_t size_z = 1;
};

/// @brief Per-dispatch tracking entry created by the AQL Packet Processor.
struct DispatchEntry {
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;
  uint32_t process_id = 0;

  uint64_t kernel_entry_pc = 0;
  uint32_t wfs_per_workgroup = 1;
  uint32_t sgprs_per_wf = 104;
  uint32_t vgprs_per_wf = 256;
  uint64_t kernarg_addr = 0;
  uint32_t kernarg_size = 0;
  uint32_t num_user_sgprs = 2;
  uint32_t kernel_code_properties = 0;
  uint16_t kernarg_preload = 0;
  uint64_t dispatch_ptr = 0;
  uint64_t queue_ptr = 0;
  uint32_t workgroup_id_offset = 0;
  uint32_t grid_wgs_x = 0;
  uint32_t grid_wgs_y = 1;
  uint32_t grid_wgs_z = 1;
  uint32_t cluster_count_x = 0;
  uint32_t cluster_count_y = 0;
  uint32_t cluster_count_z = 0;
  uint32_t cluster_size_x = 1;
  uint32_t cluster_size_y = 1;
  uint32_t cluster_size_z = 1;
  bool enable_wg_id_x = true;
  bool enable_wg_id_y = false;
  bool enable_wg_id_z = false;
  uint8_t enable_vgpr_workitem_id = 0;
  uint16_t workgroup_size_x = 64;
  uint16_t workgroup_size_y = 1;
  uint16_t workgroup_size_z = 1;
  uint64_t scratch_backing_addr = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t group_segment_fixed_size = 0;

  uint32_t total_wgs = 0;
  uint32_t dispatched_wgs = 0;
  uint32_t completed_wgs = 0;

  uint64_t completion_signal = 0;
  bool host_signal = false;
  bool barrier_bit = false;

  bool fully_dispatched() const { return dispatched_wgs >= total_wgs; }
  bool fully_completed() const { return completed_wgs >= total_wgs; }
  bool is_non_kernel() const { return total_wgs == 0; }

  uint32_t cluster_size() const { return cluster_size_x * cluster_size_y * cluster_size_z; }
  bool has_workgroup_clusters() const { return cluster_size() > 1; }
  bool cluster_grid_is_complete() const {
    return static_cast<uint64_t>(cluster_count_x) * cluster_size_x == grid_wgs_x &&
           static_cast<uint64_t>(cluster_count_y) * cluster_size_y == grid_wgs_y &&
           static_cast<uint64_t>(cluster_count_z) * cluster_size_z == grid_wgs_z;
  }

  WorkgroupCoord local_wg_coord(uint32_t local_wg_id) const {
    uint32_t gx = grid_wgs_x == 0 ? 1 : grid_wgs_x;
    uint32_t gy = grid_wgs_y == 0 ? 1 : grid_wgs_y;
    return {local_wg_id % gx, (local_wg_id / gx) % gy, local_wg_id / (gx * gy)};
  }

  uint32_t flatten_local_wg_coord(WorkgroupCoord coord) const {
    uint32_t gx = grid_wgs_x == 0 ? 1 : grid_wgs_x;
    uint32_t gy = grid_wgs_y == 0 ? 1 : grid_wgs_y;
    return coord.x + gx * (coord.y + gy * coord.z);
  }

  uint32_t cluster_rank_for_local_wg(uint32_t local_wg_id) const {
    WorkgroupCoord coord = local_wg_coord(local_wg_id);
    uint32_t sx = cluster_size_x == 0 ? 1 : cluster_size_x;
    uint32_t sy = cluster_size_y == 0 ? 1 : cluster_size_y;
    uint32_t sz = cluster_size_z == 0 ? 1 : cluster_size_z;
    uint32_t local_x = coord.x % sx;
    uint32_t local_y = coord.y % sy;
    uint32_t local_z = coord.z % sz;
    return local_x + sx * (local_y + sy * local_z);
  }

  uint32_t cluster_base_local_wg_id(uint32_t local_wg_id) const {
    WorkgroupCoord coord = local_wg_coord(local_wg_id);
    uint32_t sx = cluster_size_x == 0 ? 1 : cluster_size_x;
    uint32_t sy = cluster_size_y == 0 ? 1 : cluster_size_y;
    uint32_t sz = cluster_size_z == 0 ? 1 : cluster_size_z;
    coord.x -= coord.x % sx;
    coord.y -= coord.y % sy;
    coord.z -= coord.z % sz;
    return flatten_local_wg_coord(coord);
  }

  uint32_t cluster_base_local_wg_id_for_ordinal(uint32_t cluster_ordinal) const {
    uint32_t cx = cluster_count_x == 0 ? 1 : cluster_count_x;
    uint32_t cy = cluster_count_y == 0 ? 1 : cluster_count_y;
    WorkgroupCoord cluster_coord{};
    cluster_coord.x = cluster_ordinal % cx;
    cluster_coord.y = (cluster_ordinal / cx) % cy;
    cluster_coord.z = cluster_ordinal / (cx * cy);
    WorkgroupCoord base{};
    base.x = cluster_coord.x * cluster_size_x;
    base.y = cluster_coord.y * cluster_size_y;
    base.z = cluster_coord.z * cluster_size_z;
    return flatten_local_wg_coord(base);
  }

  uint32_t cluster_peer_local_wg_id(uint32_t local_wg_id, uint32_t rank) const {
    assert(cluster_grid_is_complete() &&
           "cluster peer math requires full clusters in every grid dimension");
    WorkgroupCoord base = local_wg_coord(cluster_base_local_wg_id(local_wg_id));
    uint32_t sx = cluster_size_x == 0 ? 1 : cluster_size_x;
    uint32_t sy = cluster_size_y == 0 ? 1 : cluster_size_y;
    WorkgroupCoord peer{};
    peer.x = base.x + rank % sx;
    peer.y = base.y + (rank / sx) % sy;
    peer.z = base.z + rank / (sx * sy);
    return flatten_local_wg_coord(peer);
  }
};

inline uint32_t dispatch_workgroup_size(const DispatchEntry &entry) {
  auto dim = [](uint16_t value) -> uint32_t { return value == 0 ? 1u : value; };
  return dim(entry.workgroup_size_x) * dim(entry.workgroup_size_y) * dim(entry.workgroup_size_z);
}

inline uint64_t wave_lane_mask(uint32_t wave_size) {
  if (wave_size >= 64)
    return ~0ULL;
  if (wave_size == 0)
    return 0;
  return (1ULL << wave_size) - 1;
}

inline uint64_t initial_exec_mask_for_wave(const DispatchEntry &entry, uint32_t wf_index_in_wg,
                                           uint32_t wave_size) {
  uint32_t wg_size = dispatch_workgroup_size(entry);
  uint32_t first_lane = wf_index_in_wg * wave_size;
  if (first_lane >= wg_size)
    return 0;
  uint32_t remaining = wg_size - first_lane;
  uint32_t active_lanes = remaining < wave_size ? remaining : wave_size;
  return wave_lane_mask(active_lanes);
}

/// @brief Per-queue state for the command processor.
///
/// @details Each HW queue has its own ordered deque of dispatch entries.
/// Entries complete in submission order (in-order retirement per queue).
struct HwQueueState {
  enum class Status { IDLE, ACTIVE, BLOCKED };

  Status status = Status::IDLE;
  std::deque<DispatchEntry> entries;
  bool implicit_barrier_next = false;
  size_t next_dispatch_idx = 0;
  uint64_t queue_desc_va = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_DISPATCH_ENTRY_H_
