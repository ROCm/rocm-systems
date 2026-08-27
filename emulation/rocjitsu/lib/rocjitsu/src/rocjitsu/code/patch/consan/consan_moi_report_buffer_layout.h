// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_abi.h"

#include <cstddef>
#include <cstdint>

namespace rocjitsu {

/// Concrete, address-free byte geometry of one MOI report allocation.
///
/// Capacities state how many ABI records each region can hold; offsets name
/// their aligned locations relative to the beginning of the allocation.
/// Unused regions alias the end of the allocation and have zero capacity. The
/// engine is retained so a layout cannot be rebound to another evidence
/// protocol. This value owns no allocation, device address, generation, or
/// lifetime.
struct ConSanMoiReportBufferLayout {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint32_t access_record_capacity = 0;
  uint32_t record_replay_logical_access_range_count = 0;
  uint32_t record_replay_dispatch_token_capacity = 0;
  uint32_t record_replay_access_dispatch_bank_count = 1;
  uint32_t record_replay_access_owner_bank_count = 1;
  uint32_t record_replay_address_group_headroom = 1;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_exact_dispatch_bank_count = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t inline_compact_token_mapping_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  size_t record_replay_dispatch_tokens_offset = sizeof(ConSanMoiReportHeader);
  size_t access_records_offset = sizeof(ConSanMoiReportHeader);
  size_t barrier_records_offset = sizeof(ConSanMoiReportHeader);
  size_t atomic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t fence_records_offset = sizeof(ConSanMoiReportHeader);
  size_t diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t exact_shadow_entries_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_atomic_release_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_acquired_epoch_token_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_causal_snapshots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_compact_token_mappings_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_watchpoints_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_causal_windows_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_sync_metadata_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_pending_acquires_offset = sizeof(ConSanMoiReportHeader);
  size_t required_bytes = sizeof(ConSanMoiReportHeader);
  bool valid = false;

  bool operator==(const ConSanMoiReportBufferLayout &) const = default;
};

} // namespace rocjitsu
