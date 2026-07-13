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
  uint32_t wave_size = 64;
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
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 1;
  uint32_t grid_size_z = 1;
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
  /// GFX10+ COMPUTE_PGM_RSRC1.WGP_MODE. When set, the workgroup is placed on
  /// a sibling-CU pair and draws from the pair's aggregate LDS pool. A single
  /// workgroup remains limited to one CU's addressable LDS size.
  bool wgp_mode = false;

  uint32_t total_wgs = 0;
  uint32_t dispatched_wgs = 0;
  uint32_t completed_wgs = 0;

  uint64_t completion_signal = 0;
  /// @brief HSA-system-clock tick captured when the CP accepted this dispatch.
  ///
  /// @details ROCR's `hsa_amd_profiling_get_dispatch_time` reads dispatch
  /// timestamps from the completion signal after the packet retires. Real CP
  /// firmware writes those fields only for profiled queues; rocjitsu records the
  /// timestamp on every kernel dispatch because non-profiled signals ignore the
  /// fields, and this keeps HIP/MIOpen event timing consistent for guest queues.
  uint64_t profiling_start_timestamp = 0;
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

inline uint32_t dispatch_dim(uint32_t value) { return value == 0 ? 1u : value; }

inline uint32_t dispatch_wg_dim(uint16_t value) { return value == 0 ? 1u : value; }

inline uint32_t dispatch_workgroup_size(const DispatchEntry &entry) {
  return dispatch_wg_dim(entry.workgroup_size_x) * dispatch_wg_dim(entry.workgroup_size_y) *
         dispatch_wg_dim(entry.workgroup_size_z);
}

inline uint32_t active_dim_workitems(uint32_t grid_size, uint32_t wg_id, uint32_t wg_size) {
  if (grid_size == 0)
    return 0;
  const uint64_t start = static_cast<uint64_t>(wg_id) * wg_size;
  if (start >= grid_size)
    return 0;
  const uint64_t remaining = grid_size - start;
  return static_cast<uint32_t>(remaining < wg_size ? remaining : wg_size);
}

inline uint32_t active_workitems_in_workgroup(const DispatchEntry &entry, uint32_t global_wg_id) {
  const uint32_t relative_wg_id = global_wg_id >= entry.workgroup_id_offset
                                      ? global_wg_id - entry.workgroup_id_offset
                                      : global_wg_id;
  const uint32_t grid_wgs_x = dispatch_dim(entry.grid_wgs_x);
  const uint32_t grid_wgs_y = dispatch_dim(entry.grid_wgs_y);
  const uint32_t wg_x = relative_wg_id % grid_wgs_x;
  const uint32_t wg_y = (relative_wg_id / grid_wgs_x) % grid_wgs_y;
  const uint32_t wg_z = relative_wg_id / (grid_wgs_x * grid_wgs_y);
  return active_dim_workitems(entry.grid_size_x, wg_x, dispatch_wg_dim(entry.workgroup_size_x)) *
         active_dim_workitems(entry.grid_size_y, wg_y, dispatch_wg_dim(entry.workgroup_size_y)) *
         active_dim_workitems(entry.grid_size_z, wg_z, dispatch_wg_dim(entry.workgroup_size_z));
}

inline uint64_t wave_lane_mask(uint32_t wave_size) {
  if (wave_size >= 64)
    return ~0ULL;
  if (wave_size == 0)
    return 0;
  return (1ULL << wave_size) - 1;
}

inline uint64_t initial_exec_mask_for_wave(const DispatchEntry &entry, uint32_t global_wg_id,
                                           uint32_t wf_index_in_wg, uint32_t wave_size) {
  const uint32_t relative_wg_id = global_wg_id >= entry.workgroup_id_offset
                                      ? global_wg_id - entry.workgroup_id_offset
                                      : global_wg_id;
  const uint32_t grid_wgs_x = dispatch_dim(entry.grid_wgs_x);
  const uint32_t grid_wgs_y = dispatch_dim(entry.grid_wgs_y);
  const uint32_t wg_x = relative_wg_id % grid_wgs_x;
  const uint32_t wg_y = (relative_wg_id / grid_wgs_x) % grid_wgs_y;
  const uint32_t wg_z = relative_wg_id / (grid_wgs_x * grid_wgs_y);
  const uint32_t workgroup_size_x = dispatch_wg_dim(entry.workgroup_size_x);
  const uint32_t workgroup_size_y = dispatch_wg_dim(entry.workgroup_size_y);
  const uint32_t workgroup_size_z = dispatch_wg_dim(entry.workgroup_size_z);
  const uint32_t workgroup_xy = workgroup_size_x * workgroup_size_y;

  uint64_t mask = 0;
  const uint32_t lanes = wave_size > 64 ? 64 : wave_size;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t flat_id = wf_index_in_wg * wave_size + lane;
    const uint32_t id_z = flat_id / workgroup_xy;
    if (id_z >= workgroup_size_z)
      continue;
    const uint32_t id_y = (flat_id / workgroup_size_x) % workgroup_size_y;
    const uint32_t id_x = flat_id % workgroup_size_x;
    const uint64_t global_x = static_cast<uint64_t>(wg_x) * workgroup_size_x + id_x;
    const uint64_t global_y = static_cast<uint64_t>(wg_y) * workgroup_size_y + id_y;
    const uint64_t global_z = static_cast<uint64_t>(wg_z) * workgroup_size_z + id_z;
    if (global_x < entry.grid_size_x && global_y < entry.grid_size_y &&
        global_z < entry.grid_size_z)
      mask |= 1ULL << lane;
  }
  return mask;
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
