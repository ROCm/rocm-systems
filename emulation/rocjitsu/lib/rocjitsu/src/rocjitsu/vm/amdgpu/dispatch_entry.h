// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file dispatch_entry.h
/// @brief Per-dispatch tracking entry for the command processor pipeline.
///
/// @details Analogous to gem5's HSAQueueEntry. Each kernel dispatch gets a
/// unique dispatch_id and tracks workgroup lifecycle (dispatched vs completed)
/// independently. Completion signals fire when all WGs of a dispatch finish,
/// in per-queue submission order.

#include "rocjitsu/vm/amdgpu/aql_packet_types.h"
#include "rocjitsu/vm/amdgpu/consumer_cursor_journal.h"
#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/queue_doorbell.h"
#include "rocjitsu/vm/amdgpu/xcd_shard.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

class GpuVmAccess;

/// @brief Validate the packet storage shared by AQL registration and update paths.
[[nodiscard]] inline constexpr bool valid_aql_packet_ring(uint64_t base_address,
                                                          uint32_t size_bytes) {
  return base_address != 0 && (base_address % kAqlPacketBytes) == 0 &&
         size_bytes >= kAqlPacketBytes && (size_bytes % kAqlPacketBytes) == 0;
}

/// @brief Validate the complete guest-visible AQL ring layout.
[[nodiscard]] inline constexpr bool valid_aql_queue_layout(uint64_t base_address,
                                                           uint32_t size_bytes,
                                                           uint64_t consumer_pointer_address,
                                                           uint64_t producer_pointer_address) {
  return valid_aql_packet_ring(base_address, size_bytes) && consumer_pointer_address != 0 &&
         (consumer_pointer_address % alignof(uint64_t)) == 0 && producer_pointer_address != 0 &&
         (producer_pointer_address % alignof(uint64_t)) == 0;
}

/// @brief Configuration supplied when registering an AQL queue with the CP.
struct AqlQueueConfig {
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink{};
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint64_t ring_base_va = 0;
  uint32_t ring_size = 0;
  uint64_t read_ptr_va = 0;
  uint64_t write_ptr_va = 0;
  uint32_t doorbell_offset = 0;
  void *doorbell_base = nullptr;
  uint64_t doorbell_va = 0;
  uint64_t last_doorbell = 0;
  QueueDoorbellMode doorbell_mode = QueueDoorbellMode::Explicit;
  bool uses_kfd_queue_abi = false;
  uint64_t queue_desc_va = 0;
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  /// Spread each of this queue's dispatches over every XCD of the SoC, the way a
  /// multi-XCD part does when it runs as a single partition. Set by the queue
  /// creation path that models such a device; registering the queue replicates it
  /// onto the peer XCDs.
  bool xcd_fanout = false;
};

struct WorkgroupCoord {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
};

struct WorkitemCoord {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
};

/// @brief What a queue entry carries, which decides how it retires.
enum class DispatchPacketKind : uint8_t {
  /// @brief Never a valid kind for an entry that has been queued.
  ///
  /// @details The default exists so that forgetting to set the kind trips the
  /// assertion in @ref DispatchEntry::is_non_kernel rather than silently
  /// retiring a kernel as though it were a barrier.
  Unset = 0,
  /// @brief An AQL kernel dispatch, even when this entry's share of it is empty.
  Kernel,
  /// @brief A packet that runs no shader: barrier, barrier-value, PM4 IB.
  NonKernel,
};

/// @brief Durable stages of one dispatch's guest-visible retirement.
///
/// @details A translated backing may temporarily report Unavailable after an earlier
/// publication step has already committed.  Keeping the exact next stage on
/// the dispatch prevents a retry from repeating the signal decrement or a
/// plugin callback.
enum class CompletionPublicationPhase : uint8_t {
  ExecutionEnd,
  CaptureAccess,
  ReadMailboxPointer,
  ReadEventId,
  StoreStartTimestamp,
  StoreEndTimestamp,
  DecrementSignal,
  StoreMailbox,
  DeliverInterrupt,
  Complete,
};

/// @brief Durable state for publishing one dispatch completion.
class CompletionPublicationState {
public:
  CompletionPublicationPhase phase = CompletionPublicationPhase::ExecutionEnd;
  std::shared_ptr<GpuVmAccess> access;
  uint64_t start_timestamp = 0;
  uint64_t end_timestamp = 0;
  uint64_t mailbox_pointer = 0;
  uint64_t compare_expected = 0;
  uint64_t signal_old_value = 0;
  uint64_t signal_new_value = 0;
  uint32_t event_id = 0;
};

/// @brief Durable stages of the queue-inactive notification emitted after the
/// final dispatch retires.
enum class QueueIdlePublicationPhase : uint8_t {
  Inactive,
  CaptureAccess,
  ReadSignalHandle,
  StoreIdleStatus,
  ReadMailboxPointer,
  ReadEventId,
  StoreMailbox,
  DeliverEventInterrupt,
  DeliverGenericInterrupt,
  Complete,
};

/// @brief Durable state for publishing one queue-inactive notification.
class QueueIdlePublicationState {
public:
  QueueIdlePublicationPhase phase = QueueIdlePublicationPhase::Inactive;
  std::shared_ptr<GpuVmAccess> access;
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink;
  uint64_t queue_desc_va = 0;
  uint64_t signal_address = 0;
  uint64_t mailbox_pointer = 0;
  uint32_t event_id = 0;
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  /// @brief Queue activity generation for which this empty transition was observed.
  uint64_t activity_generation = 0;

  [[nodiscard]] bool active() const {
    return phase != QueueIdlePublicationPhase::Inactive &&
           phase != QueueIdlePublicationPhase::Complete;
  }

  void reset() { *this = {}; }
};

/// @brief Grid-wide retirement state shared by every shard of one dispatch.
///
/// @details When a dispatch is fanned out, each participating XCD retires its own
/// share independently, but the dispatch's completion signal must fire once, after
/// the last workgroup anywhere on the device. Each XCD flushes its own caches and
/// then publishes its share here; the owning XCD fires the signal only once the
/// published total covers the grid. Publishing releases and the owner's check
/// acquires, so every XCD's flush is visible before the signal is written.
struct GridCompletion {
  /// Workgroups retired across all participating XCDs.
  std::atomic<uint32_t> completed_wgs{0};
  /// Workgroups in the whole grid.
  uint32_t grid_wgs = 0;
  /// Cleared by the first XCD to place a workgroup, so the dispatch reports that
  /// it began exactly once however the grid is split. An XCD whose share is empty
  /// never places anything, so this cannot be tied to the XCD that read the packet.
  std::atomic_flag execution_begun{};

  /// @brief Add one XCD's retired workgroups to the grid total.
  /// @returns True for the single caller whose contribution completed the grid, so
  /// the follow-on wake happens exactly once rather than once per racing XCD.
  [[nodiscard]] bool publish_share(uint32_t wgs) {
    uint32_t before = completed_wgs.fetch_add(wgs, std::memory_order_acq_rel);
    return before < grid_wgs && before + wgs >= grid_wgs;
  }

  /// @returns True once every XCD has published its share.
  [[nodiscard]] bool grid_retired() const {
    return completed_wgs.load(std::memory_order_acquire) >= grid_wgs;
  }

  /// @returns True for the first caller only.
  [[nodiscard]] bool claim_execution_begin() {
    return !execution_begun.test_and_set(std::memory_order_relaxed);
  }

  /// @brief Publish a terminal fault shared by every XCD shard.
  void mark_faulted() { terminal_faulted.store(true, std::memory_order_release); }

  /// @brief Whether any shard reported a terminal fault for this grid.
  [[nodiscard]] bool faulted() const { return terminal_faulted.load(std::memory_order_acquire); }

  std::atomic<bool> terminal_faulted{false};
};

/// @brief Per-dispatch tracking entry created by the AQL Packet Processor.
struct DispatchEntry {
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;
  uint32_t queue_packet_id = 0;
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink;
  uint32_t process_id = 0;

  /// AQL ring packet id (queue read index at which this dispatch's packet was
  /// fetched). Used only for rocm-dbgapi wave/dispatch correlation.
  uint32_t aql_packet_id = 0;

  uint64_t kernel_entry_pc = 0;
  uint64_t code_load_bias = 0;
  uint32_t wfs_per_workgroup = 1;
  uint32_t sgprs_per_wf = 104;
  uint32_t vgprs_per_wf = 256;
  uint64_t kernarg_addr = 0;
  uint32_t kernarg_size = 0;
  uint32_t num_user_sgprs = 2;
  uint32_t kernel_code_properties = 0;
  uint32_t num_named_barriers = 0;
  /// Architectural wave size selected by the kernel descriptor and used by
  /// the dispatched wavefront.
  uint8_t kernel_wave_size = 0;
  uint16_t kernarg_preload = 0;
  uint32_t initial_mode_raw = 0;
  uint64_t dispatch_ptr = 0;
  uint64_t queue_ptr = 0;
  uint32_t workgroup_id_offset = 0;
  // Actual AQL grid extents.
  uint32_t grid_size_x = 1;
  uint32_t grid_size_y = 1;
  uint32_t grid_size_z = 1;
  uint32_t grid_wgs_x = 0;
  uint32_t grid_wgs_y = 1;
  uint32_t grid_wgs_z = 1;
  bool grid_yz_valid = false;
  uint32_t cluster_count_x = 0;
  uint32_t cluster_count_y = 0;
  uint32_t cluster_count_z = 0;
  uint32_t cluster_size_x = 1;
  uint32_t cluster_size_y = 1;
  uint32_t cluster_size_z = 1;
  /// RDNA COMPUTE_PGM_RSRC1.WGP_MODE. When set, the workgroup is placed on
  /// one of a sibling CU pair and uses their shared WGP LDS backing.
  bool wgp_mode = false;
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

  /// This entry's share of the grid. The default owns all of it.
  XcdShard shard{};

  /// Workgroups this entry is responsible for: the whole grid for an unsharded
  /// entry, otherwise this shard's share. dispatched_wgs and completed_wgs count
  /// against it, so an entry retires when its own share is done.
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
  /// What this entry carries. Every site that queues an entry must set it.
  DispatchPacketKind kind = DispatchPacketKind::Unset;
  bool host_signal = false;
  /// Header barrier bit: this packet waits for all earlier packets in its queue.
  bool wait_for_predecessors = false;
  /// Packet-type ordering: following packets cannot pass this packet.
  bool blocks_following = false;
  /// Completion hooks and signal have already been delivered.
  bool completion_notified = false;
  bool execution_begun = false;
  /// The packet carried an acquire fence of at least agent scope. Recorded on the
  /// entry so that every XCD running part of the grid can invalidate its own caches
  /// on its own thread, not just the one that read the packet.
  bool acquire_invalidate = false;

  /// Grid-wide retirement state, shared by every shard of a fanned-out dispatch.
  /// Null when this entry owns the whole grid by itself.
  std::shared_ptr<GridCompletion> grid_completion{};
  /// True on the shards handed to peer XCDs. The shard left on the XCD that read
  /// the packet keeps the completion signal and the dispatch-level callbacks.
  bool fanout_peer = false;
  /// Set once this shard has published its retired workgroups to grid_completion,
  /// so a repeated drain cannot double-count them.
  bool grid_share_published = false;
  /// @brief Terminal execution fault; suppresses normal retirement publication.
  bool terminal_faulted = false;
  /// Guest-visible completion publication retained across transient VM stalls.
  CompletionPublicationState completion_publication{};

  bool fully_dispatched() const { return dispatched_wgs >= total_wgs; }
  bool fully_completed() const { return completed_wgs >= total_wgs; }

  /// @returns Workgroups in the whole grid, as opposed to total_wgs which after
  /// a fan-out counts only this XCD's share. Anything sized or indexed against the
  /// grid -- scratch backing, whose per-wave offset comes from the grid-wide
  /// workgroup id -- must use this and not total_wgs.
  [[nodiscard]] uint32_t grid_total_wgs() const {
    return grid_completion ? grid_completion->grid_wgs : total_wgs;
  }

  /// @returns True when the dispatch has finished everywhere, not merely on this
  /// XCD. The AQL barrier bit means "no later packet starts until every preceding
  /// packet has completed", which for a fanned-out dispatch is a property of the
  /// whole grid: this XCD finishing its share says nothing about the others.
  [[nodiscard]] bool grid_fully_completed() const {
    if (grid_completion)
      return grid_completion->grid_retired();
    return fully_completed();
  }

  [[nodiscard]] bool grid_faulted() const {
    return terminal_faulted || (grid_completion && grid_completion->faulted());
  }

  /// @returns True for packets that run no shader at all (barrier, barrier-value,
  /// PM4 IB). Deliberately a stored kind rather than `total_wgs == 0`: a kernel
  /// dispatch split across XCDs can legitimately leave one XCD an empty share,
  /// and inferring the kind from the count would retire that kernel as though it
  /// were a barrier and fire its completion signal early.
  bool is_non_kernel() const {
    assert(kind != DispatchPacketKind::Unset && "queued entry never had its packet kind set");
    return kind != DispatchPacketKind::Kernel;
  }

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

  WorkgroupCoord cluster_local_wg_coord_for_flat_wg_id(uint32_t flat_wg_id) const {
    WorkgroupCoord coord = local_wg_coord(flat_wg_id);
    uint32_t sx = cluster_size_x == 0 ? 1 : cluster_size_x;
    uint32_t sy = cluster_size_y == 0 ? 1 : cluster_size_y;
    uint32_t sz = cluster_size_z == 0 ? 1 : cluster_size_z;
    return {coord.x % sx, coord.y % sy, coord.z % sz};
  }

  uint32_t cluster_rank_for_flat_wg_id(uint32_t flat_wg_id) const {
    WorkgroupCoord local = cluster_local_wg_coord_for_flat_wg_id(flat_wg_id);
    uint32_t sx = cluster_size_x == 0 ? 1 : cluster_size_x;
    uint32_t sy = cluster_size_y == 0 ? 1 : cluster_size_y;
    return local.x + sx * (local.y + sy * local.z);
  }

  uint64_t cluster_rank_period() const {
    assert(cluster_grid_is_complete() && "cluster rank period requires a complete cluster grid");
    // A complete grid makes the highest active axis period a multiple of all
    // lower-axis periods, so one alignment check preserves the full rank.
    uint64_t period = cluster_size_x;
    if (cluster_size_y > 1 || cluster_size_z > 1)
      period = static_cast<uint64_t>(grid_wgs_x) * cluster_size_y;
    if (cluster_size_z > 1)
      period = static_cast<uint64_t>(grid_wgs_x) * grid_wgs_y * cluster_size_z;
    return period;
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

  /// @brief Chunk the grid walk advances by: a cluster, else a single workgroup.
  [[nodiscard]] uint32_t dispatch_chunk_wgs() const {
    return has_workgroup_clusters() ? cluster_size() : 1u;
  }

  /// @brief Grid-wide chunk ordinal for this entry's @p shard_chunk_index -th chunk.
  ///
  /// @details The input is shard-local -- an index into this entry's own share,
  /// counted from zero on every XCD -- and the result is grid-wide. The shard
  /// cannot check the bound itself, since it knows the stride but not this
  /// entry's share size, so an off-by-one or a workgroup-versus-cluster unit
  /// mix-up would otherwise return an ordinal outside the grid and place a
  /// workgroup that does not exist. Bound it here, where the share size is known.
  /// @param shard_chunk_index Zero-based chunk index within this entry's share.
  /// @returns The chunk's ordinal in the whole grid.
  [[nodiscard]] uint32_t chunk_ordinal_for(uint32_t shard_chunk_index) const {
    assert(static_cast<uint64_t>(shard_chunk_index) * dispatch_chunk_wgs() < total_wgs &&
           "chunk index is shard-local and must lie within this entry's share");
    return shard.nth_owned_chunk(shard_chunk_index);
  }

  /// @brief Narrow this entry to one XCD's share of the grid it currently holds.
  ///
  /// @details Reads the grid size from this entry's own total_wgs and rewrites it
  /// to the share @p xcd_shard owns, so the existing dispatch and retirement
  /// bookkeeping operates on the share. The grid size is not passed in: a caller
  /// supplying a mismatched value would silently desynchronize total_wgs from the
  /// coordinate extents.
  ///
  /// Retirement contract: once a packet is split, per-entry retirement is NOT
  /// packet retirement. An entry retiring means only that one XCD finished its
  /// share, so the code that splits a packet is responsible for aggregating the
  /// shares and for leaving completion_signal on exactly one entry -- otherwise
  /// the signal fires once per share, and fires when the first XCD finishes.
  /// The dispatch-level callbacks are the same problem in a second place: every
  /// share reaches drain_completions() and would emit its own execution-end,
  /// while a share with no workgroups never emits a matching begin. One matched
  /// begin/end pair is owed per packet, not per share.
  /// Nothing in this commit splits a packet; the fan-out change adds both.
  ///
  /// @param xcd_shard The shard to apply. Applying twice is rejected: the second
  /// call would re-shard an already narrowed entry.
  void apply_shard(XcdShard xcd_shard) {
    assert(dispatched_wgs == 0 && "shard must be applied before dispatching");
    assert(shard.is_unsharded() && "entry has already been sharded");
    const uint32_t grid_wgs = total_wgs;
    const uint32_t chunk = dispatch_chunk_wgs();
    assert(chunk == 1 || (cluster_grid_is_complete() && grid_wgs % chunk == 0 &&
                          "a clustered grid must tile exactly, or the truncated tail is dropped "
                          "from every shard and the grid can never retire"));
    shard = xcd_shard;
    total_wgs = xcd_shard.owned_chunks(grid_wgs / chunk) * chunk;
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

template <typename T> [[nodiscard]] inline constexpr uint32_t nonzero_dim(T value) {
  return value == 0 ? 1u : static_cast<uint32_t>(value);
}

inline constexpr uint32_t kPackedTidMask = 0x3FFu;
inline constexpr uint32_t kPackedTidYShift = 10;
inline constexpr uint32_t kPackedTidZShift = 20;

[[nodiscard]] inline constexpr uint32_t pack_workitem_id(WorkitemCoord id,
                                                         uint32_t component_count) {
  uint32_t packed = id.x & kPackedTidMask;
  if (component_count >= 1)
    packed |= (id.y & kPackedTidMask) << kPackedTidYShift;
  if (component_count >= 2)
    packed |= (id.z & kPackedTidMask) << kPackedTidZShift;
  return packed;
}

[[nodiscard]] inline WorkitemCoord workitem_local_coord(const DispatchEntry &entry,
                                                        uint32_t wf_index_in_wg, uint32_t lane,
                                                        uint32_t wave_size) {
  const uint32_t workgroup_size_x = nonzero_dim(entry.workgroup_size_x);
  const uint32_t workgroup_size_y = nonzero_dim(entry.workgroup_size_y);
  const uint32_t flat_id = wf_index_in_wg * wave_size + lane;
  return {flat_id % workgroup_size_x, (flat_id / workgroup_size_x) % workgroup_size_y,
          flat_id / (workgroup_size_x * workgroup_size_y)};
}

[[nodiscard]] inline uint64_t initial_exec_mask_for_wave(const DispatchEntry &entry,
                                                         uint32_t global_wg_id,
                                                         uint32_t wf_index_in_wg,
                                                         uint32_t wave_size) {
  assert(wave_size <= 64 && "AMDGPU wave size must not exceed 64 lanes");
  assert(entry.grid_size_x != 0 && entry.grid_size_y != 0 && entry.grid_size_z != 0 &&
         "kernel dispatch grid dimensions must be nonzero");
  const uint32_t relative_wg_id = global_wg_id >= entry.workgroup_id_offset
                                      ? global_wg_id - entry.workgroup_id_offset
                                      : global_wg_id;
  const uint32_t grid_wgs_x = nonzero_dim(entry.grid_wgs_x);
  const uint32_t grid_wgs_y = nonzero_dim(entry.grid_wgs_y);
  const uint32_t wg_x = relative_wg_id % grid_wgs_x;
  const uint32_t wg_y = (relative_wg_id / grid_wgs_x) % grid_wgs_y;
  const uint32_t wg_z = relative_wg_id / (grid_wgs_x * grid_wgs_y);
  const uint32_t workgroup_size_x = nonzero_dim(entry.workgroup_size_x);
  const uint32_t workgroup_size_y = nonzero_dim(entry.workgroup_size_y);
  const uint32_t workgroup_size_z = nonzero_dim(entry.workgroup_size_z);

  uint64_t mask = 0;
  const uint32_t lanes = wave_size > 64 ? 64 : wave_size;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const WorkitemCoord id = workitem_local_coord(entry, wf_index_in_wg, lane, wave_size);
    if (id.z >= workgroup_size_z)
      continue;
    const uint64_t global_x = static_cast<uint64_t>(wg_x) * workgroup_size_x + id.x;
    const uint64_t global_y = static_cast<uint64_t>(wg_y) * workgroup_size_y + id.y;
    const uint64_t global_z = static_cast<uint64_t>(wg_z) * workgroup_size_z + id.z;
    if (global_x < entry.grid_size_x && global_y < entry.grid_size_y &&
        global_z < entry.grid_size_z)
      mask |= 1ULL << lane;
  }
  return mask;
}

/// @brief Configuration and runtime state for one CP-registered AQL queue.
///
/// @details Keeping the registration configuration and dispatch runtime in one
/// record prevents queue mutation and teardown from desynchronizing parallel
/// containers. Entries complete in submission order (in-order retirement per
/// queue).
struct AqlQueueRecord : AqlQueueConfig {
  AqlQueueRecord() : read_pointer_journal(read_ptr_va) {}
  explicit AqlQueueRecord(AqlQueueConfig config)
      : AqlQueueConfig(std::move(config)), read_pointer_journal(read_ptr_va) {}

  enum class Status { Idle, Active, Blocked };

  /// @brief CP-local lifetime identity; never reused during this CP lifetime.
  uint64_t registration_id = 0;
  /// Prevent address-space teardown while this CP may take future snapshots.
  GpuVmBindingLease address_space_lease;
  /// @brief Set when a packet faulted; the queue stops until it is torn down.
  /// @details A faulted packet is retired rather than retried, because its
  /// endpoint will never resolve. Continuing the scan would then run the FENCE
  /// or signal packet behind it and publish completion for work that never
  /// happened, which is the same lie the faulted copy was stopped from telling.
  /// The model therefore contains the failure to this queue until its owner
  /// tears it down; fault notification is handled separately.
  bool faulted = false;
  bool debug_suspended = false;
  bool runtime_suspended = false;
  /// A command-processor pass observed this queue while its debugger gate was closed.
  /// Cleared on resume after scheduling one pass to process the deferred work.
  bool debug_work_deferred = false;
  /// CP-private monotonic fetch cursor: the next ring index to fetch. Normally
  /// tracks read_ptr_va exactly, but stays ahead of it while the debugger holds
  /// the queue's read_dispatch_id at a trapped dispatch (so packets are not
  /// re-fetched). See fetch_from_queue and serialize_queue_debug_waves.
  uint64_t fetch_cursor = 0;
  /// Set on the replicas that xcd_fanout creates. A replica never reads the ring
  /// and never polls a doorbell; work reaches it as dispatch shards from the XCD
  /// that owns the queue.
  bool fanout_replica = false;
  Status status = Status::Idle;
  std::deque<DispatchEntry> entries;
  /// @brief Total entries accepted by this queue, for tests.
  /// @details A replica's entry list is otherwise unobservable after its packets
  /// retire. Recording acceptance at the queue's single ordered push site lets a
  /// test verify delivery AFTER the run without depending on how many entries
  /// happened to coexist in the queue or reaching into a running peer CP.
  size_t accepted_entries = 0;
  /// @brief Kinds of the first two accepted entries, for tests.
  /// @details Bounded because the replication-order test needs only the submitted
  /// kernel and the non-kernel packet behind it; production queues must not retain
  /// an unbounded history after entries retire.
  std::array<DispatchPacketKind, 2> first_accepted_entry_kinds{};
  AddressSpaceHandle first_accepted_address_space;
  bool implicit_barrier_next = false;
  size_t next_dispatch_idx = 0;
  /// Incremented at the single entry-admission point. Idle publication may
  /// only continue while this generation still describes an empty queue.
  uint64_t activity_generation = 0;
  /// Queue-inactive publication survives after the last entry has been popped.
  QueueIdlePublicationState idle_publication{};
  /// A guest-visible completion or idle publication is waiting for its backing
  /// transport. The CP may keep servicing other queues, but must not admit or
  /// advance this queue until the journal can resume in order.
  bool publication_retry_pending = false;
  /// A retained cursor or completion publication failed terminally. Graceful
  /// removal reports the failure instead of discarding partially committed state.
  bool publication_faulted = false;
  /// Read-pointer retirement retained with the exact VM snapshot that observed
  /// the consumed packets, so a retry cannot publish through a replacement root.
  ConsumerCursorJournal read_pointer_journal;

  /// @brief Append an entry, maintaining accepted_entries.
  /// @details The single ordered push site, so the acceptance count tracks the number
  /// of pushed entries. Callers already hold the CP's queue mutex.
  void push_entry(DispatchEntry entry) {
    ++activity_generation;
    if (activity_generation == 0)
      ++activity_generation;
    if (accepted_entries < first_accepted_entry_kinds.size())
      first_accepted_entry_kinds[accepted_entries] = entry.kind;
    if (accepted_entries == 0)
      first_accepted_address_space = entry.address_space;
    entries.push_back(std::move(entry));
    ++accepted_entries;
  }
};

} // namespace amdgpu
} // namespace rocjitsu
