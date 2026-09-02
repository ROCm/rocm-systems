// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report_test_support.h
/// @brief Test-only convenience construction for explicit report fixtures.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu {

/// Build deliberately heterogeneous fixture headers without giving production
/// a second field-by-field construction path beside its typed layout product.
[[nodiscard]] constexpr ConSanMoiReportHeader make_consan_moi_report_header(
    uint64_t generation, uint64_t dispatch_id, uint32_t access_record_capacity,
    uint32_t diagnostic_capacity, uint32_t exact_shadow_entry_capacity,
    uint32_t sampled_watchpoint_capacity, uint32_t barrier_record_capacity = 0,
    uint32_t atomic_record_capacity = 0, uint32_t inline_atomic_release_capacity = 0,
    uint32_t fence_record_capacity = 0, uint32_t inline_acquired_epoch_token_capacity = 0,
    uint32_t inline_causal_snapshot_capacity = 0,
    ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.engine = static_cast<uint32_t>(engine);
  header.access_record_capacity = access_record_capacity;
  header.barrier_record_capacity = barrier_record_capacity;
  header.atomic_record_capacity = atomic_record_capacity;
  header.diagnostic_capacity = diagnostic_capacity;
  header.exact_shadow_entry_capacity = exact_shadow_entry_capacity;
  header.sampled_watchpoint_capacity = sampled_watchpoint_capacity;
  header.inline_atomic_release_capacity = inline_atomic_release_capacity;
  header.inline_acquired_epoch_token_capacity = inline_acquired_epoch_token_capacity;
  header.inline_causal_snapshot_capacity = inline_causal_snapshot_capacity;
  header.sampled_causal_window_capacity = sampled_watchpoint_capacity;
  header.sampled_sync_metadata_capacity = sampled_watchpoint_capacity;
  header.sampled_pending_acquire_capacity = sampled_watchpoint_capacity;
  header.fence_record_capacity = fence_record_capacity;
  if (engine == ConSanMoiEngine::InlineShadow)
    header.layout_flags = kConSanMoiReportKnownLayoutFlags;
  return header;
}

} // namespace rocjitsu
