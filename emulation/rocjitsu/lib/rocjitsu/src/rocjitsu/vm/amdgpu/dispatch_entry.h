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

#include <cstdint>
#include <deque>

namespace rocjitsu {
namespace amdgpu {

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
};

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
