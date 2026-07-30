// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace rocjitsu {
namespace {

[[nodiscard]] std::optional<size_t> claim_record_replay_dispatch_bank(std::span<uint64_t> slots,
                                                                      uint64_t dispatch_id) {
  const uint64_t token = consan_moi_record_replay_claim_token(dispatch_id);
  if (token == 0u || slots.empty() || (slots.size() & (slots.size() - 1u)) != 0u)
    return std::nullopt;
  // Host oracle for the instruction sequence in
  // build_first_light_access_record_words().
  size_t bank = consan_moi_record_replay_fold_dispatch_token(token) & (slots.size() - 1u);
  const size_t probe_limit = std::min<size_t>(slots.size(), kConSanMoiRecordReplayProbeLimit);
  for (size_t probe = 0; probe < probe_limit; ++probe) {
    if (slots[bank] == 0u) {
      slots[bank] = token;
      return bank;
    }
    if (slots[bank] == token)
      return bank;
    bank = consan_moi_record_replay_advance_probe(static_cast<uint32_t>(bank),
                                                  static_cast<uint32_t>(probe + 1u),
                                                  static_cast<uint32_t>(slots.size()));
  }
  return std::nullopt;
}

struct RecordReplayAccessIdentity {
  uint32_t dispatch_bank = 0;
  uint32_t site_token = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;

  bool operator==(const RecordReplayAccessIdentity &) const = default;
};

[[nodiscard]] uint32_t
record_replay_access_identity_hash(const RecordReplayAccessIdentity &identity) {
  uint32_t hash = identity.wave_id;
  for (uint32_t coordinate : {identity.workgroup_x, identity.workgroup_y, identity.workgroup_z})
    hash = consan_moi_record_replay_mix_identity(hash, coordinate);
  hash = consan_moi_record_replay_mix_identity(hash, identity.dispatch_bank);
  return consan_moi_record_replay_mix_identity(hash, identity.site_token);
}

[[nodiscard]] std::optional<size_t>
claim_record_replay_access_identity(std::span<std::optional<RecordReplayAccessIdentity>> slots,
                                    const RecordReplayAccessIdentity &identity) {
  if (slots.empty() || (slots.size() & (slots.size() - 1u)) != 0u)
    return std::nullopt;
  // Host oracle for the instruction sequence in
  // build_first_light_access_record_words().
  size_t bank = record_replay_access_identity_hash(identity) & (slots.size() - 1u);
  const size_t probe_limit = std::min<size_t>(slots.size(), kConSanMoiRecordReplayProbeLimit);
  for (size_t probe = 0; probe < probe_limit; ++probe) {
    if (!slots[bank]) {
      slots[bank] = identity;
      return bank;
    }
    if (*slots[bank] == identity)
      return bank;
    bank = consan_moi_record_replay_advance_probe(static_cast<uint32_t>(bank),
                                                  static_cast<uint32_t>(probe + 1u),
                                                  static_cast<uint32_t>(slots.size()));
  }
  return std::nullopt;
}

TEST(ConSanMoiAutoReportPlan, EmptyRecordReplayIsExactlyOneHeader) {
  const auto plan = plan_consan_moi_auto_report({});
  EXPECT_TRUE(plan.complete());
  EXPECT_EQ(plan.outcome, ConSanMoiAutoReportPlanOutcome::Complete);
  EXPECT_EQ(plan.reason, ConSanMoiAutoReportPlanReason::None);
  EXPECT_EQ(plan.required_bytes, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(plan.layout.required_bytes, sizeof(ConSanMoiReportHeader));
  EXPECT_TRUE(plan.layout.valid);
  EXPECT_EQ(consan_moi_auto_report_plan_outcome_name(plan.outcome), "complete");
}

TEST(ConSanMoiAutoReportPlan, RecordReplayUsesIndependentExactRegionCounts) {
  const ConSanMoiAutoReportInventory inventory{.engine = ConSanMoiEngine::RecordReplay,
                                               .access_range_count = 3,
                                               .barrier_event_count = 5,
                                               .atomic_event_count = 7,
                                               .fence_event_count = 11,
                                               .diagnostic_count = 13};
  const auto plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.record_replay_access_dispatch_bank_count,
            kConSanMoiRecordReplayMaximumDispatchBankCount);
  EXPECT_EQ(plan.layout.record_replay_access_owner_bank_count,
            kConSanMoiRecordReplayMaximumOwnerBankCount);
  EXPECT_EQ(plan.layout.record_replay_logical_access_range_count, 3u);
  EXPECT_EQ(plan.layout.access_record_capacity, 131072u);
  EXPECT_EQ(plan.layout.barrier_record_capacity, 5u);
  EXPECT_EQ(plan.layout.atomic_record_capacity, 7u);
  EXPECT_EQ(plan.layout.fence_record_capacity, 11u);
  EXPECT_EQ(plan.layout.diagnostic_capacity, 13u);
  EXPECT_EQ(plan.layout.record_replay_dispatch_token_capacity,
            kConSanMoiRecordReplayMaximumDispatchTokenCount);
  EXPECT_EQ(plan.layout.record_replay_dispatch_token_capacity,
            kConSanMoiRecordReplayAnticipatedDispatchCount *
                kConSanMoiRecordReplayHashTableHeadroom);
  EXPECT_EQ(plan.layout.record_replay_dispatch_tokens_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(plan.layout.access_records_offset,
            sizeof(ConSanMoiReportHeader) +
                kConSanMoiRecordReplayMaximumDispatchTokenCount * sizeof(uint64_t));
  EXPECT_EQ(plan.layout.barrier_records_offset,
            plan.layout.access_records_offset +
                plan.layout.access_record_capacity * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(plan.layout.atomic_records_offset,
            plan.layout.barrier_records_offset + 5 * sizeof(ConSanMoiBarrierRecord));
  EXPECT_EQ(plan.layout.fence_records_offset,
            plan.layout.atomic_records_offset + 7 * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(plan.layout.diagnostic_records_offset,
            plan.layout.fence_records_offset + 11 * sizeof(ConSanMoiFenceRecord));
  EXPECT_EQ(plan.required_bytes,
            plan.layout.diagnostic_records_offset + 13 * sizeof(ConSanMoiDiagnosticRecord));
}

TEST(ConSanMoiAutoReportPlan, RecordReplayAccessTableMakesTheSizingIncreaseExplicit) {
  constexpr uint64_t kHistoricalAccessRecordBytes = 64u;
  constexpr uint64_t kFormerlyFittingLogicalRanges = 4096u;
  constexpr uint64_t kBankCount =
      static_cast<uint64_t>(kConSanMoiRecordReplayMaximumDispatchBankCount) *
      kConSanMoiRecordReplayMaximumOwnerBankCount;
  static_assert(sizeof(ConSanMoiAccessRecord) == 80u);
  static_assert(sizeof(ConSanMoiReportHeader) +
                    kFormerlyFittingLogicalRanges * kHistoricalAccessRecordBytes <
                kConSanMoiAutoReportBufferCeilingBytes);

  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::RecordReplay,
      .access_range_count = kFormerlyFittingLogicalRanges,
  };
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);

  EXPECT_EQ(plan.outcome, ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(plan.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
  const uint64_t expected_access_capacity = std::bit_ceil(
      kFormerlyFittingLogicalRanges * kBankCount *
      kConSanMoiRecordReplayMaximumAddressGroupsPerWave * kConSanMoiRecordReplayHashTableHeadroom);
  EXPECT_EQ(plan.layout.access_record_capacity, expected_access_capacity);
  EXPECT_EQ(plan.required_bytes,
            sizeof(ConSanMoiReportHeader) +
                kConSanMoiRecordReplayMaximumDispatchTokenCount * sizeof(uint64_t) +
                expected_access_capacity * sizeof(ConSanMoiAccessRecord));
  EXPECT_GT(plan.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);
}

TEST(ConSanMoiAutoReportPlan, RecordReplayOverrideRoundTripsIdentityTableLayout) {
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::RecordReplay,
      .access_range_count = 3,
      .barrier_event_count = 5,
      .atomic_event_count = 7,
      .fence_event_count = 11,
      .diagnostic_count = 13,
  };
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  const auto override_layout = consan_moi_auto_report_layout_override(plan);
  ASSERT_TRUE(override_layout);

  const ConSanMoiReportBufferLayout restored = consan_moi_report_layout_from_override(
      *override_layout, ConSanMoiEngine::RecordReplay, plan.required_bytes);
  EXPECT_TRUE(restored.valid);
  EXPECT_EQ(restored.record_replay_access_dispatch_bank_count,
            kConSanMoiRecordReplayMaximumDispatchBankCount);
  EXPECT_EQ(restored.record_replay_dispatch_token_capacity,
            kConSanMoiRecordReplayMaximumDispatchTokenCount);
  EXPECT_EQ(restored.record_replay_logical_access_range_count, inventory.access_range_count);
  EXPECT_EQ(restored.record_replay_access_owner_bank_count,
            kConSanMoiRecordReplayMaximumOwnerBankCount);
  EXPECT_EQ(restored.access_record_capacity, plan.layout.access_record_capacity);
  EXPECT_EQ(restored.required_bytes, plan.required_bytes);

  auto corrupt = *override_layout;
  --corrupt.record_replay_dispatch_token_capacity;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::RecordReplay,
                                                      plan.required_bytes)
                   .valid);
  corrupt = *override_layout;
  corrupt.record_replay_access_owner_bank_count = 3u;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::RecordReplay,
                                                      plan.required_bytes)
                   .valid);
  corrupt = *override_layout;
  --corrupt.access_record_capacity;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::RecordReplay,
                                                      plan.required_bytes)
                   .valid);
}

TEST(ConSanMoiAutoReportPlan, RecordReplayClaimTokenPreservesDispatchIdentityAndZeroSentinel) {
  constexpr uint64_t dispatch_id = 0x123456789abcdef0ull;
  constexpr uint64_t token = consan_moi_record_replay_claim_token(dispatch_id);
  EXPECT_NE(token, 0u);
  EXPECT_EQ(consan_moi_record_replay_claim_token(token), dispatch_id);
  EXPECT_EQ(consan_moi_record_replay_claim_token(kConSanMoiRecordReplayClaimTokenXorMask), 0u);
}

TEST(ConSanMoiAutoReportPlan, RecordReplayDispatchTableResolvesHashCollisionsUntilFull) {
  std::array<uint64_t, 4> slots{};
  constexpr uint64_t first_dispatch = 1u;
  constexpr uint64_t colliding_dispatch = uint64_t{1} << 32u;
  static_assert(
      (static_cast<uint32_t>(consan_moi_record_replay_claim_token(first_dispatch)) ^
       static_cast<uint32_t>(consan_moi_record_replay_claim_token(first_dispatch) >> 32u)) ==
      (static_cast<uint32_t>(consan_moi_record_replay_claim_token(colliding_dispatch)) ^
       static_cast<uint32_t>(consan_moi_record_replay_claim_token(colliding_dispatch) >> 32u)));

  const auto first = claim_record_replay_dispatch_bank(slots, first_dispatch);
  const auto collision = claim_record_replay_dispatch_bank(slots, colliding_dispatch);
  ASSERT_TRUE(first);
  ASSERT_TRUE(collision);
  EXPECT_NE(*first, *collision);
  EXPECT_EQ(claim_record_replay_dispatch_bank(slots, first_dispatch), first);
  EXPECT_TRUE(claim_record_replay_dispatch_bank(slots, 2u));
  EXPECT_TRUE(claim_record_replay_dispatch_bank(slots, 3u));
  EXPECT_FALSE(claim_record_replay_dispatch_bank(slots, 4u));
}

TEST(ConSanMoiAutoReportPlan, RecordReplayAccessTableResolvesHashCollisionsUntilFull) {
  std::array<std::optional<RecordReplayAccessIdentity>, 4> slots{};
  const std::array<RecordReplayAccessIdentity, 5> same_home_identities = {{
      {.workgroup_x = 0u, .wave_id = 0u},
      {.workgroup_x = 1u, .wave_id = 1u},
      {.workgroup_x = 2u, .wave_id = 2u},
      {.workgroup_x = 3u, .wave_id = 3u},
      {.workgroup_x = 4u, .wave_id = 4u},
  }};
  for (const RecordReplayAccessIdentity &identity : same_home_identities)
    ASSERT_EQ(record_replay_access_identity_hash(identity) & (slots.size() - 1u), 0u);

  std::array<size_t, 4> claimed_banks{};
  for (size_t index = 0; index < claimed_banks.size(); ++index) {
    const auto bank = claim_record_replay_access_identity(slots, same_home_identities[index]);
    ASSERT_TRUE(bank);
    claimed_banks[index] = *bank;
  }
  EXPECT_EQ(claimed_banks, (std::array<size_t, 4>{0u, 1u, 3u, 2u}));
  EXPECT_EQ(claim_record_replay_access_identity(slots, same_home_identities.front()), 0u);
  EXPECT_FALSE(claim_record_replay_access_identity(slots, same_home_identities.back()));
}

TEST(ConSanMoiAutoReportPlan, RecordReplayAccessIdentityIncludesDispatchAndSite) {
  std::array<std::optional<RecordReplayAccessIdentity>, 8> slots{};
  const RecordReplayAccessIdentity first{
      .dispatch_bank = 1u, .site_token = 7u, .workgroup_x = 3u, .wave_id = 2u};
  RecordReplayAccessIdentity other_dispatch = first;
  ++other_dispatch.dispatch_bank;
  RecordReplayAccessIdentity other_site = first;
  ++other_site.site_token;

  const auto first_bank = claim_record_replay_access_identity(slots, first);
  const auto dispatch_bank = claim_record_replay_access_identity(slots, other_dispatch);
  const auto site_bank = claim_record_replay_access_identity(slots, other_site);
  ASSERT_TRUE(first_bank);
  ASSERT_TRUE(dispatch_bank);
  ASSERT_TRUE(site_bank);
  EXPECT_NE(*first_bank, *dispatch_bank);
  EXPECT_NE(*first_bank, *site_bank);
  EXPECT_EQ(claim_record_replay_access_identity(slots, first), first_bank);
}

TEST(ConSanMoiAutoReportPlan, RecordReplayCapacityValidationRejectsMalformedHashTables) {
  const auto plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(plan.complete());
  const auto valid = plan.layout;
  ASSERT_TRUE(consan_moi_report_layout_has_required_capacities(valid, ConSanMoiEngine::RecordReplay,
                                                               /*track_barriers=*/false,
                                                               /*track_atomics=*/false));

  auto malformed = valid;
  malformed.record_replay_dispatch_token_capacity = 3u;
  EXPECT_FALSE(consan_moi_report_layout_has_required_capacities(
      malformed, ConSanMoiEngine::RecordReplay, false, false));

  malformed = valid;
  malformed.record_replay_dispatch_token_capacity =
      kConSanMoiRecordReplayMaximumDispatchTokenCount * 2u;
  EXPECT_FALSE(consan_moi_report_layout_has_required_capacities(
      malformed, ConSanMoiEngine::RecordReplay, false, false));

  malformed = valid;
  malformed.record_replay_logical_access_range_count = 0u;
  EXPECT_FALSE(consan_moi_report_layout_has_required_capacities(
      malformed, ConSanMoiEngine::RecordReplay, false, false));

  malformed = valid;
  malformed.record_replay_logical_access_range_count =
      malformed.access_record_capacity / (malformed.record_replay_access_dispatch_bank_count *
                                          malformed.record_replay_access_owner_bank_count) +
      1u;
  EXPECT_FALSE(consan_moi_report_layout_has_required_capacities(
      malformed, ConSanMoiEngine::RecordReplay, false, false));

  malformed = valid;
  --malformed.access_record_capacity;
  EXPECT_FALSE(consan_moi_report_layout_has_required_capacities(
      malformed, ConSanMoiEngine::RecordReplay, false, false));
}

TEST(ConSanMoiAutoReportPlan, SampledSeparatesBanksFromMultiCellWatchpoints) {
  const ConSanMoiAutoReportInventory inventory{.engine = ConSanMoiEngine::Sampled,
                                               .diagnostic_count = 4,
                                               .sampled_range_bank_count = 17,
                                               .sampled_watchpoint_count = 53};
  const auto plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.diagnostic_capacity, 4u);
  EXPECT_EQ(plan.layout.sampled_causal_window_capacity, 17u);
  EXPECT_EQ(plan.layout.sampled_watchpoint_capacity, 53u);
  EXPECT_EQ(plan.layout.sampled_sync_metadata_capacity, 17u);
  EXPECT_EQ(plan.layout.sampled_pending_acquire_capacity, 17u);
  EXPECT_EQ(plan.layout.diagnostic_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(plan.layout.sampled_causal_windows_offset,
            plan.layout.diagnostic_records_offset + 4 * sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(plan.layout.sampled_watchpoints_offset,
            plan.layout.sampled_causal_windows_offset + 17 * sizeof(ConSanMoiSampledCausalWindow));
  EXPECT_EQ(plan.layout.sampled_sync_metadata_offset,
            plan.layout.sampled_watchpoints_offset + 53 * sizeof(uint64_t));
  EXPECT_EQ(plan.layout.sampled_pending_acquires_offset,
            plan.layout.sampled_sync_metadata_offset +
                17 * sizeof(ConSanMoiSampledSyncMetadataPacked));
  EXPECT_EQ(plan.required_bytes, plan.layout.sampled_pending_acquires_offset +
                                     17 * sizeof(ConSanMoiSampledPendingAcquireSlot));
}

TEST(ConSanMoiAutoReportPlan, InlineRoundsDeclaredLdsAndKeepsOrderingTablesIndependent) {
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .diagnostic_count = 4,
      .inline_lds_bytes = 4097,
      .inline_atomic_release_count = 3,
      .inline_causal_snapshot_count = 5,
      .inline_acquired_epoch_token_count = 7,
      .inline_compact_token_mapping_count = 6,
  };
  const auto plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  const uint32_t expected_dispatch_banks =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes);
  const uint64_t expected_exact_shadow_entries = 1025u * expected_dispatch_banks;
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count, expected_dispatch_banks);
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity, expected_exact_shadow_entries);
  EXPECT_EQ(plan.layout.inline_atomic_release_capacity, 3u);
  EXPECT_EQ(plan.layout.inline_causal_snapshot_capacity, 5u);
  EXPECT_EQ(plan.layout.inline_acquired_epoch_token_capacity, 7u);
  EXPECT_EQ(plan.layout.inline_compact_token_mapping_capacity, 6u);
  EXPECT_EQ(plan.layout.exact_shadow_entries_offset,
            sizeof(ConSanMoiReportHeader) + 4 * sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(plan.layout.inline_atomic_release_slots_offset,
            plan.layout.exact_shadow_entries_offset +
                expected_exact_shadow_entries * sizeof(ConSanMoiInlineExactShadowSlot));
  EXPECT_EQ(plan.layout.inline_causal_snapshots_offset,
            plan.layout.inline_atomic_release_slots_offset +
                3 * sizeof(ConSanMoiInlineAtomicReleaseSlot));
  EXPECT_EQ(plan.layout.inline_compact_token_mappings_offset,
            plan.layout.inline_causal_snapshots_offset + 5 * sizeof(ConSanMoiInlineCausalSnapshot));
  EXPECT_EQ(plan.layout.inline_acquired_epoch_token_slots_offset,
            plan.layout.inline_compact_token_mappings_offset +
                6 * sizeof(ConSanMoiCompactDiagnosticTokenMapping));
  EXPECT_EQ(plan.required_bytes, plan.layout.inline_acquired_epoch_token_slots_offset +
                                     7 * sizeof(ConSanMoiInlineAcquiredEpochTokenSlot));
}

TEST(ConSanMoiAutoReportPlan, InlineLdsUsesPerLayoutDispatchBanking) {
  constexpr uint64_t kFullLdsBytes = 64u * 1024u;
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .inline_lds_bytes = kFullLdsBytes,
  };

  const auto plan = plan_consan_moi_auto_report(inventory);

  ASSERT_TRUE(plan.complete());
  const uint32_t expected_dispatch_banks =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(kFullLdsBytes);
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count, expected_dispatch_banks);
  EXPECT_EQ(expected_dispatch_banks, 128u);
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            (kFullLdsBytes / consan_moi_exact_shadow::granule_bytes) * expected_dispatch_banks);
  EXPECT_LE(plan.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);
}

TEST(ConSanMoiAutoReportPlan, InlineExactOverrideRoundTripsDispatchBankedLayout) {
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .diagnostic_count = 4,
      .inline_lds_bytes = 4097,
      .inline_atomic_release_count = 3,
      .inline_causal_snapshot_count = 5,
      .inline_acquired_epoch_token_count = 7,
      .inline_compact_token_mapping_count = 6,
  };
  const auto plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  const auto override_layout = consan_moi_auto_report_layout_override(plan);
  ASSERT_TRUE(override_layout);

  const auto restored = consan_moi_report_layout_from_override(
      *override_layout, ConSanMoiEngine::InlineShadow, plan.required_bytes);
  EXPECT_TRUE(restored.valid);
  EXPECT_EQ(restored.exact_shadow_entry_capacity, plan.layout.exact_shadow_entry_capacity);
  EXPECT_EQ(restored.inline_exact_dispatch_bank_count,
            plan.layout.inline_exact_dispatch_bank_count);
  EXPECT_EQ(restored.diagnostic_capacity, plan.layout.diagnostic_capacity);
  EXPECT_EQ(restored.inline_compact_token_mapping_capacity, 6u);
  EXPECT_EQ(restored.inline_compact_token_mappings_offset,
            plan.layout.inline_compact_token_mappings_offset);
  EXPECT_EQ(restored.required_bytes, plan.required_bytes);

  auto corrupt = *override_layout;
  --corrupt.exact_shadow_entry_capacity;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::InlineShadow,
                                                      plan.required_bytes)
                   .valid);
  corrupt = *override_layout;
  corrupt.inline_exact_dispatch_bank_count /= 2u;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::InlineShadow,
                                                      plan.required_bytes)
                   .valid);
  corrupt = *override_layout;
  ++corrupt.inline_compact_token_mappings_offset;
  EXPECT_FALSE(consan_moi_report_layout_from_override(corrupt, ConSanMoiEngine::InlineShadow,
                                                      plan.required_bytes)
                   .valid);
}

TEST(ConSanMoiAutoReportPlan, PerBufferCeilingIsInclusiveAndRetainsRequiredBytes) {
  // Solve the heterogeneous boundary without assuming that one record stride
  // divides the payload. An empty access inventory deliberately omits the
  // automatic dispatch directory.
  uint64_t fence_count = 0;
  uint64_t diagnostic_count = 0;
  for (uint64_t candidate_fences = 0; candidate_fences < 20u; ++candidate_fences) {
    const uint64_t fence_bytes = candidate_fences * sizeof(ConSanMoiFenceRecord);
    const uint64_t remaining =
        kConSanMoiAutoReportBufferCeilingBytes - sizeof(ConSanMoiReportHeader) - fence_bytes;
    if (remaining % sizeof(ConSanMoiDiagnosticRecord) == 0u) {
      fence_count = candidate_fences;
      diagnostic_count = remaining / sizeof(ConSanMoiDiagnosticRecord);
      break;
    }
  }
  ASSERT_NE(diagnostic_count, 0u);
  const ConSanMoiAutoReportInventory fitting{
      .engine = ConSanMoiEngine::RecordReplay,
      .fence_event_count = fence_count,
      .diagnostic_count = diagnostic_count,
  };
  const auto accepted = plan_consan_moi_auto_report(fitting);
  ASSERT_TRUE(accepted.complete());
  EXPECT_EQ(accepted.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);

  auto too_large = fitting;
  ++too_large.diagnostic_count;
  const auto rejected = plan_consan_moi_auto_report(too_large);
  EXPECT_FALSE(rejected.complete());
  EXPECT_EQ(rejected.outcome, ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(rejected.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
  EXPECT_GT(rejected.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);
  EXPECT_FALSE(rejected.layout.valid);
  EXPECT_EQ(rejected.required_bytes,
            kConSanMoiAutoReportBufferCeilingBytes + sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(consan_moi_auto_report_plan_outcome_name(rejected.outcome),
            "insufficient_report_capacity");
}

TEST(ConSanMoiAutoReportPlan, SampledBoundaryIsExactAndOneWatchpointFails) {
  ASSERT_EQ((kConSanMoiAutoReportBufferCeilingBytes - sizeof(ConSanMoiReportHeader)) %
                sizeof(uint64_t),
            0u);
  const uint64_t watchpoint_count =
      (kConSanMoiAutoReportBufferCeilingBytes - sizeof(ConSanMoiReportHeader)) / sizeof(uint64_t);
  ConSanMoiAutoReportInventory inventory{.engine = ConSanMoiEngine::Sampled,
                                         .sampled_watchpoint_count = watchpoint_count};
  const auto accepted = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(accepted.complete());
  EXPECT_EQ(accepted.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);

  ++inventory.sampled_watchpoint_count;
  const auto rejected = plan_consan_moi_auto_report(inventory);
  EXPECT_EQ(rejected.outcome, ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(rejected.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
  EXPECT_EQ(rejected.required_bytes, kConSanMoiAutoReportBufferCeilingBytes + sizeof(uint64_t));
}

TEST(ConSanMoiAutoReportPlan, AdaptiveSampledBanksFitWithoutDroppingLogicalRanges) {
  constexpr uint64_t kLogicalRanges = 135610u;
  const ConSanMoiAutoReportInventory requested{
      .engine = ConSanMoiEngine::Sampled,
      .access_range_count = kLogicalRanges,
      .diagnostic_count = kLogicalRanges,
      .sampled_range_bank_count = 8u * kLogicalRanges,
      .sampled_watchpoint_count = 8u * kLogicalRanges,
      .sampled_bank_count_adaptive = true,
  };
  ASSERT_EQ(plan_consan_moi_auto_report(requested).outcome,
            ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);

  const auto fitted = fit_consan_moi_sampled_auto_report_inventory(requested);
  EXPECT_EQ(fitted.access_range_count, kLogicalRanges);
  EXPECT_EQ(fitted.diagnostic_count, kLogicalRanges);
  EXPECT_EQ(fitted.sampled_range_bank_count, 4u * kLogicalRanges);
  EXPECT_EQ(fitted.sampled_watchpoint_count, 4u * kLogicalRanges);
  EXPECT_TRUE(plan_consan_moi_auto_report(fitted).complete());

  auto exact = requested;
  exact.sampled_bank_count_adaptive = false;
  EXPECT_EQ(fit_consan_moi_sampled_auto_report_inventory(exact).sampled_range_bank_count,
            8u * kLogicalRanges);
}

TEST(ConSanMoiAutoReportPlan, InlineBoundaryChangesOnlyAtWholeLdsCells) {
  ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .inline_lds_bytes = 4096u,
  };
  const auto aligned = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(aligned.complete());

  ++inventory.inline_lds_bytes;
  const auto next_cell = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(next_cell.complete());
  EXPECT_EQ(next_cell.layout.inline_exact_dispatch_bank_count,
            aligned.layout.inline_exact_dispatch_bank_count);
  EXPECT_EQ(next_cell.layout.exact_shadow_entry_capacity,
            aligned.layout.exact_shadow_entry_capacity +
                aligned.layout.inline_exact_dispatch_bank_count);

  inventory.inline_lds_bytes += consan_moi_exact_shadow::granule_bytes - 1u;
  const auto same_cell = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(same_cell.complete());
  EXPECT_EQ(same_cell.layout.exact_shadow_entry_capacity,
            next_cell.layout.exact_shadow_entry_capacity);
}

TEST(ConSanMoiAutoReportPlan, EveryAbiCapacityRejectsOnePastUint32) {
  constexpr uint64_t overflow = uint64_t{std::numeric_limits<uint32_t>::max()} + 1u;
  const ConSanMoiAutoReportInventory inventories[] = {
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = overflow},
      {.engine = ConSanMoiEngine::RecordReplay, .barrier_event_count = overflow},
      {.engine = ConSanMoiEngine::RecordReplay, .atomic_event_count = overflow},
      {.engine = ConSanMoiEngine::RecordReplay, .fence_event_count = overflow},
      {.engine = ConSanMoiEngine::RecordReplay, .diagnostic_count = overflow},
      {.engine = ConSanMoiEngine::Sampled, .sampled_range_bank_count = overflow},
      {.engine = ConSanMoiEngine::Sampled, .sampled_watchpoint_count = overflow},
      {.engine = ConSanMoiEngine::InlineShadow, .inline_atomic_release_count = overflow},
      {.engine = ConSanMoiEngine::InlineShadow, .inline_causal_snapshot_count = overflow},
      {.engine = ConSanMoiEngine::InlineShadow, .inline_acquired_epoch_token_count = overflow},
      {.engine = ConSanMoiEngine::InlineShadow,
       .inline_lds_bytes = overflow * consan_moi_exact_shadow::granule_bytes},
  };
  for (const auto &inventory : inventories) {
    const auto plan = plan_consan_moi_auto_report(inventory);
    EXPECT_EQ(plan.outcome, ConSanMoiAutoReportPlanOutcome::Overflow);
    EXPECT_EQ(plan.reason, ConSanMoiAutoReportPlanReason::AbiCapacityOverflow);
    EXPECT_EQ(plan.required_bytes, 0u);
    EXPECT_FALSE(plan.layout.valid);
  }
}

TEST(ConSanMoiAutoReportPlan, InlineLdsRoundingOverflowIsTyped) {
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .inline_lds_bytes = std::numeric_limits<uint64_t>::max(),
  };
  const auto plan = plan_consan_moi_auto_report(inventory);
  EXPECT_EQ(plan.outcome, ConSanMoiAutoReportPlanOutcome::Overflow);
  EXPECT_EQ(plan.reason, ConSanMoiAutoReportPlanReason::ByteSizeOverflow);
  EXPECT_EQ(consan_moi_auto_report_plan_outcome_name(plan.outcome), "overflow");
  EXPECT_FALSE(plan.layout.valid);
}

TEST(ConSanMoiAutoReportPlan, RepresentableHugeCountsAreCapacityInsufficientNotOverflow) {
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::Sampled,
      .sampled_range_bank_count = std::numeric_limits<uint32_t>::max(),
      .sampled_watchpoint_count = std::numeric_limits<uint32_t>::max(),
  };
  const auto plan = plan_consan_moi_auto_report(inventory);
  EXPECT_EQ(plan.outcome, ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(plan.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
  EXPECT_GT(plan.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);
  EXPECT_FALSE(plan.layout.valid);
}

TEST(ConSanMoiAutoReportPlan, FrozenSafetyCeilingsRemainDistinct) {
  EXPECT_EQ(kConSanMoiAutoReportBufferCeilingBytes, 128u * 1024u * 1024u);
  EXPECT_EQ(kConSanMoiAutoReportProcessCeilingBytes, 256u * 1024u * 1024u);
  EXPECT_GT(kConSanMoiAutoReportProcessCeilingBytes, kConSanMoiAutoReportBufferCeilingBytes);
  EXPECT_EQ(
      consan_moi_auto_report_plan_reason_name(ConSanMoiAutoReportPlanReason::PerBufferCeiling),
      "per_buffer_ceiling");
}

TEST(ConSanMoiAutoReportPlan, ProcessBudgetIsInclusiveAcrossObjectsAndReleaseIsChecked) {
  ConSanMoiAutoReportProcessBudget budget;
  constexpr uint64_t first = 16u * 1024u * 1024u;
  ASSERT_TRUE(reserve_consan_moi_auto_report_bytes(budget, first));
  ASSERT_TRUE(reserve_consan_moi_auto_report_bytes(budget, kConSanMoiAutoReportProcessCeilingBytes -
                                                               first));
  EXPECT_EQ(budget.current_live_bytes, kConSanMoiAutoReportProcessCeilingBytes);
  EXPECT_EQ(budget.peak_live_bytes, kConSanMoiAutoReportProcessCeilingBytes);
  EXPECT_FALSE(reserve_consan_moi_auto_report_bytes(budget, 1u));
  EXPECT_FALSE(
      release_consan_moi_auto_report_bytes(budget, kConSanMoiAutoReportProcessCeilingBytes + 1u));
  ASSERT_TRUE(release_consan_moi_auto_report_bytes(budget, first));
  ASSERT_TRUE(release_consan_moi_auto_report_bytes(budget, kConSanMoiAutoReportProcessCeilingBytes -
                                                               first));
  EXPECT_EQ(budget.current_live_bytes, 0u);
  EXPECT_EQ(budget.peak_live_bytes, kConSanMoiAutoReportProcessCeilingBytes);
}

TEST(ConSanMoiAutoReportPlan, ExactOverrideRoundTripsHeterogeneousSampledLayout) {
  const ConSanMoiAutoReportInventory inventory{.engine = ConSanMoiEngine::Sampled,
                                               .diagnostic_count = 3,
                                               .sampled_range_bank_count = 5,
                                               .sampled_watchpoint_count = 17};
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  const auto override_layout = consan_moi_auto_report_layout_override(plan);
  ASSERT_TRUE(override_layout);
  EXPECT_EQ(override_layout->sampled_causal_window_capacity, 5u);
  EXPECT_EQ(override_layout->sampled_watchpoint_capacity, 17u);

  const ConSanMoiReportBufferLayout resolved = consan_moi_report_layout_from_override(
      *override_layout, ConSanMoiEngine::Sampled, plan.required_bytes);
  EXPECT_TRUE(resolved.valid);
  EXPECT_EQ(resolved.required_bytes, plan.layout.required_bytes);
  EXPECT_EQ(resolved.diagnostic_records_offset, plan.layout.diagnostic_records_offset);
  EXPECT_EQ(resolved.sampled_causal_windows_offset, plan.layout.sampled_causal_windows_offset);
  EXPECT_EQ(resolved.sampled_watchpoints_offset, plan.layout.sampled_watchpoints_offset);
  EXPECT_EQ(resolved.sampled_pending_acquires_offset, plan.layout.sampled_pending_acquires_offset);
  const ConSanMoiReportHeader header = make_consan_moi_report_header_for_layout(
      /*generation=*/7, /*dispatch_id=*/9, resolved, ConSanMoiEngine::Sampled);
  EXPECT_EQ(header.sampled_watchpoint_capacity, 17u);
  EXPECT_EQ(header.sampled_causal_window_capacity, 5u);
  EXPECT_EQ(header.sampled_sync_metadata_capacity, 5u);
  EXPECT_EQ(header.sampled_pending_acquire_capacity, 5u);
  EXPECT_TRUE(consan_moi_report_layout_matches_header(header, resolved, ConSanMoiEngine::Sampled,
                                                      plan.required_bytes));
}

TEST(ConSanMoiAutoReportPlan, ExactOverrideRejectsCorruptOffsetWrongEngineAndShortAllocation) {
  const ConSanMoiAutoReportPlan plan =
      plan_consan_moi_auto_report({.engine = ConSanMoiEngine::RecordReplay,
                                   .access_range_count = 2,
                                   .barrier_event_count = 3,
                                   .atomic_event_count = 1,
                                   .fence_event_count = 4,
                                   .diagnostic_count = 2});
  ASSERT_TRUE(plan.complete());
  auto override_layout = consan_moi_auto_report_layout_override(plan);
  ASSERT_TRUE(override_layout);

  ++override_layout->atomic_records_offset;
  EXPECT_FALSE(consan_moi_report_layout_from_override(
                   *override_layout, ConSanMoiEngine::RecordReplay, plan.required_bytes)
                   .valid);
  --override_layout->atomic_records_offset;
  EXPECT_FALSE(consan_moi_report_layout_from_override(
                   *override_layout, ConSanMoiEngine::InlineShadow, plan.required_bytes)
                   .valid);
  EXPECT_FALSE(consan_moi_report_layout_from_override(
                   *override_layout, ConSanMoiEngine::RecordReplay, plan.required_bytes - 1u)
                   .valid);
}

TEST(ConSanMoiAutoReportPlan, IncompletePlanCannotProduceAnOverride) {
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::Sampled,
       .sampled_range_bank_count = kConSanMoiAutoReportBufferCeilingBytes});
  ASSERT_FALSE(plan.complete());
  EXPECT_FALSE(consan_moi_auto_report_layout_override(plan));
}

} // namespace
} // namespace rocjitsu
