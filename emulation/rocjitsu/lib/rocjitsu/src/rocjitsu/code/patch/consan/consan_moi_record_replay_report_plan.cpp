// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"

#include "util/bit.h"

#include <algorithm>
#include <bit>

namespace rocjitsu::consan_moi_impl {

bool plan_record_replay_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                      ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  if (!checked_moi_report_capacity(inventory.access_range_count,
                                   layout.record_replay_logical_access_range_count)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  if (inventory.access_range_count == 0u) {
    layout.record_replay_access_dispatch_bank_count = 1u;
    layout.record_replay_access_owner_bank_count = 1u;
    layout.record_replay_address_group_headroom = 1u;
  } else {
    if (inventory.record_replay_dispatch_token_capacity == 0u ||
        inventory.record_replay_dispatch_token_capacity >
            kConSanMoiRecordReplayMaximumDispatchTokenCount ||
        (inventory.record_replay_dispatch_token_capacity &
         (inventory.record_replay_dispatch_token_capacity - 1u)) != 0u ||
        inventory.record_replay_access_dispatch_bank_count == 0u ||
        inventory.record_replay_access_dispatch_bank_count >
            kConSanMoiRecordReplayMaximumDispatchBankCount ||
        (inventory.record_replay_access_dispatch_bank_count &
         (inventory.record_replay_access_dispatch_bank_count - 1u)) != 0u ||
        inventory.record_replay_access_owner_bank_count == 0u ||
        inventory.record_replay_access_owner_bank_count >
            kConSanMoiRecordReplayMaximumOwnerBankCount ||
        (inventory.record_replay_access_owner_bank_count &
         (inventory.record_replay_access_owner_bank_count - 1u)) != 0u ||
        inventory.record_replay_address_group_headroom == 0u ||
        inventory.record_replay_address_group_headroom >
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
        (inventory.record_replay_address_group_headroom &
         (inventory.record_replay_address_group_headroom - 1u)) != 0u) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
    if (!checked_moi_report_capacity(inventory.record_replay_dispatch_token_capacity,
                                     layout.record_replay_dispatch_token_capacity) ||
        !checked_moi_report_capacity(inventory.record_replay_access_dispatch_bank_count,
                                     layout.record_replay_access_dispatch_bank_count) ||
        !checked_moi_report_capacity(inventory.record_replay_access_owner_bank_count,
                                     layout.record_replay_access_owner_bank_count) ||
        !checked_moi_report_capacity(inventory.record_replay_address_group_headroom,
                                     layout.record_replay_address_group_headroom)) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
  }
  const auto dispatch_records =
      util::checked_mul(inventory.access_range_count,
                        static_cast<uint64_t>(layout.record_replay_access_dispatch_bank_count));
  const auto owner_records =
      dispatch_records
          ? util::checked_mul(*dispatch_records,
                              static_cast<uint64_t>(layout.record_replay_access_owner_bank_count))
          : std::nullopt;
  const auto address_group_records =
      owner_records
          ? util::checked_mul(*owner_records, inventory.record_replay_address_group_headroom)
          : std::nullopt;
  const auto hash_records =
      address_group_records ? util::checked_mul(*address_group_records,
                                                uint64_t{kConSanMoiRecordReplayHashTableHeadroom})
                            : std::nullopt;
  if (!hash_records) {
    plan.reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  uint64_t access_record_count = *hash_records;
  if (access_record_count != 0u) {
    if (access_record_count > uint64_t{1} << 31u) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow;
      return false;
    }
    access_record_count = std::bit_ceil(access_record_count);
  }
  if (!checked_moi_report_capacity(access_record_count, layout.access_record_capacity) ||
      !checked_moi_report_capacity(inventory.barrier_event_count, layout.barrier_record_capacity) ||
      !checked_moi_report_capacity(inventory.atomic_event_count, layout.atomic_record_capacity) ||
      !checked_moi_report_capacity(inventory.fence_event_count, layout.fence_record_capacity) ||
      !checked_moi_report_capacity(inventory.diagnostic_count, layout.diagnostic_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_moi_report_region(layout.record_replay_dispatch_token_capacity, sizeof(uint64_t),
                                  alignof(uint64_t), cursor,
                                  layout.record_replay_dispatch_tokens_offset) &&
         append_moi_report_region(access_record_count, sizeof(ConSanMoiAccessRecord),
                                  alignof(ConSanMoiAccessRecord), cursor,
                                  layout.access_records_offset) &&
         append_moi_report_region(inventory.barrier_event_count, sizeof(ConSanMoiBarrierRecord),
                                  alignof(ConSanMoiBarrierRecord), cursor,
                                  layout.barrier_records_offset) &&
         append_moi_report_region(inventory.atomic_event_count, sizeof(ConSanMoiAtomicRecord),
                                  alignof(ConSanMoiAtomicRecord), cursor,
                                  layout.atomic_records_offset) &&
         append_moi_report_region(inventory.fence_event_count, sizeof(ConSanMoiFenceRecord),
                                  alignof(ConSanMoiFenceRecord), cursor,
                                  layout.fence_records_offset) &&
         append_moi_report_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                                  alignof(ConSanMoiDiagnosticRecord), cursor,
                                  layout.diagnostic_records_offset);
}

std::optional<ConSanMoiAutoReportInventory>
reconstruct_record_replay_report_inventory(const ConSanMoiReportBufferLayout &candidate) {
  const bool empty_access_layout = candidate.access_record_capacity == 0u &&
                                   candidate.record_replay_logical_access_range_count == 0u &&
                                   candidate.record_replay_dispatch_token_capacity == 0u &&
                                   candidate.record_replay_access_dispatch_bank_count == 1u &&
                                   candidate.record_replay_access_owner_bank_count == 1u &&
                                   candidate.record_replay_address_group_headroom == 1u;
  if ((!empty_access_layout && (candidate.record_replay_dispatch_token_capacity == 0u ||
                                candidate.record_replay_dispatch_token_capacity >
                                    kConSanMoiRecordReplayMaximumDispatchTokenCount ||
                                (candidate.record_replay_dispatch_token_capacity &
                                 (candidate.record_replay_dispatch_token_capacity - 1u)) != 0u)) ||
      candidate.record_replay_access_dispatch_bank_count == 0u ||
      candidate.record_replay_access_dispatch_bank_count >
          kConSanMoiRecordReplayMaximumDispatchBankCount ||
      (candidate.record_replay_access_dispatch_bank_count &
       (candidate.record_replay_access_dispatch_bank_count - 1u)) != 0u ||
      candidate.record_replay_access_owner_bank_count == 0u ||
      candidate.record_replay_access_owner_bank_count >
          kConSanMoiRecordReplayMaximumOwnerBankCount ||
      (candidate.record_replay_access_owner_bank_count &
       (candidate.record_replay_access_owner_bank_count - 1u)) != 0u ||
      candidate.record_replay_address_group_headroom == 0u ||
      candidate.record_replay_address_group_headroom >
          kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
      (candidate.record_replay_address_group_headroom &
       (candidate.record_replay_address_group_headroom - 1u)) != 0u) {
    return std::nullopt;
  }
  const uint64_t bank_count =
      static_cast<uint64_t>(candidate.record_replay_access_dispatch_bank_count) *
      candidate.record_replay_access_owner_bank_count;
  const auto minimum_access_capacity = util::checked_mul(
      static_cast<uint64_t>(candidate.record_replay_logical_access_range_count), bank_count);
  const auto grouped_access_capacity =
      minimum_access_capacity
          ? util::checked_mul(*minimum_access_capacity,
                              static_cast<uint64_t>(candidate.record_replay_address_group_headroom))
          : std::nullopt;
  if (!grouped_access_capacity || *grouped_access_capacity > candidate.access_record_capacity)
    return std::nullopt;

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = candidate.record_replay_logical_access_range_count;
  inventory.record_replay_dispatch_token_capacity = candidate.record_replay_dispatch_token_capacity;
  inventory.record_replay_access_dispatch_bank_count =
      candidate.record_replay_access_dispatch_bank_count;
  inventory.record_replay_access_owner_bank_count = candidate.record_replay_access_owner_bank_count;
  inventory.record_replay_address_group_headroom = candidate.record_replay_address_group_headroom;
  inventory.barrier_event_count = candidate.barrier_record_capacity;
  inventory.atomic_event_count = candidate.atomic_record_capacity;
  inventory.fence_event_count = candidate.fence_record_capacity;
  inventory.diagnostic_count = candidate.diagnostic_capacity;
  return inventory;
}

} // namespace rocjitsu::consan_moi_impl

namespace rocjitsu {

ConSanMoiAutoReportInventory
fit_consan_moi_record_replay_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                                   uint64_t caller_ceiling_bytes) {
  if (inventory.engine != ConSanMoiEngine::RecordReplay)
    return inventory;

  const uint64_t static_barriers = inventory.barrier_event_count;
  const uint64_t static_atomics = inventory.atomic_event_count;
  const uint64_t static_fences = inventory.fence_event_count;
  const uint64_t static_diagnostics = inventory.diagnostic_count;
  const auto expanded_count = [](uint64_t count, uint64_t headroom) {
    return util::saturating_mul(count, headroom);
  };
  const auto expanded_candidate = [&](uint64_t lane_headroom) {
    ConSanMoiAutoReportInventory candidate = inventory;
    const uint64_t barrier_headroom = std::max<uint64_t>(lane_headroom / 64u, 1u);
    candidate.barrier_event_count = expanded_count(static_barriers, barrier_headroom);
    candidate.atomic_event_count = expanded_count(static_atomics, lane_headroom);
    candidate.fence_event_count = expanded_count(static_fences, lane_headroom);
    if (candidate.record_replay_bank_count_adaptive) {
      uint64_t diagnostic_count = util::saturating_mul(
          candidate.access_range_count, candidate.record_replay_access_dispatch_bank_count);
      diagnostic_count =
          util::saturating_mul(diagnostic_count, candidate.record_replay_access_owner_bank_count);
      diagnostic_count =
          util::saturating_mul(diagnostic_count, candidate.record_replay_address_group_headroom);
      candidate.diagnostic_count = std::max(static_diagnostics, diagnostic_count);
    }
    return candidate;
  };
  for (;;) {
    uint64_t headroom = kConSanMoiRecordReplayDynamicLaneEventHeadroom;
    for (;;) {
      ConSanMoiAutoReportInventory candidate = expanded_candidate(headroom);
      const ConSanMoiAutoReportPlan plan =
          plan_consan_moi_auto_report(candidate, caller_ceiling_bytes);
      if (plan.complete())
        return candidate;
      const bool capacity_limited =
          plan.outcome == ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity &&
          plan.reason == ConSanMoiAutoReportPlanReason::PerBufferCeiling;
      const bool adaptive_abi_capacity_limited =
          plan.outcome == ConSanMoiAutoReportPlanOutcome::Overflow &&
          plan.reason == ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow &&
          inventory.record_replay_bank_count_adaptive;
      if (!capacity_limited && !adaptive_abi_capacity_limited)
        return candidate;
      if (headroom == 1u)
        break;
      headroom = std::max<uint64_t>(headroom / 2u, 1u);
    }

    if (!inventory.record_replay_bank_count_adaptive)
      return expanded_candidate(1u);
    if (inventory.record_replay_access_dispatch_bank_count != 1u ||
        inventory.record_replay_access_owner_bank_count != 1u) {
      inventory.record_replay_access_dispatch_bank_count =
          std::max<uint64_t>(inventory.record_replay_access_dispatch_bank_count / 2u, 1u);
      inventory.record_replay_access_owner_bank_count =
          std::max<uint64_t>(inventory.record_replay_access_owner_bank_count / 2u, 1u);
      continue;
    }
    if (inventory.record_replay_address_group_headroom != 1u) {
      inventory.record_replay_address_group_headroom =
          std::max<uint64_t>(inventory.record_replay_address_group_headroom / 2u, 1u);
      continue;
    }
    return expanded_candidate(1u);
  }
}

} // namespace rocjitsu
