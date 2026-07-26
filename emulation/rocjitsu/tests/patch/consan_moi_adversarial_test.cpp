// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

namespace rocjitsu {
namespace {

constexpr ConSanMoiInlineAcquiredEpochTokenSlot make_token(uint64_t dispatch_id,
                                                           uint32_t consumer_owner_id,
                                                           uint32_t producer_owner_id,
                                                           uint32_t epoch_plus_one = 12) {
  return {.consumer_owner_id = consumer_owner_id,
          .producer_owner_id = producer_owner_id,
          .producer_epoch_plus_one = epoch_plus_one,
          .workgroup_key = 19,
          .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
          .dispatch_id = dispatch_id,
          .source_release_address = 0x4000,
          .source_release_version = 2,
          .consumer_epoch_plus_one = 1};
}

constexpr ConSanMoiInlineCausalTokenView make_token_view(uint64_t dispatch_id,
                                                         uint32_t consumer_owner_id,
                                                         uint32_t producer_owner_id,
                                                         uint32_t epoch_plus_one = 12) {
  return {.version_before = 2,
          .version_after = 2,
          .dispatch_id = dispatch_id,
          .workgroup_key = 19,
          .consumer_owner_id = consumer_owner_id,
          .producer_owner_id = producer_owner_id,
          .producer_epoch_plus_one = epoch_plus_one,
          .kind = ConSanMoiInlineTokenEvidenceKind::Direct,
          .source_release_address = 0x4000,
          .source_release_version = 2,
          .consumer_epoch_plus_one = 1};
}

TEST(ConSanMoiAdversarial, MultipleProducerClaimLossRollsBackAndDispatchReuseCollides) {
  constexpr uint64_t first_dispatch = 0x1000000000000001ull;
  constexpr uint64_t second_dispatch = 0x1000000000000002ull;
  std::array<ConSanMoiInlineAcquiredEpochTokenSlot, 64> table{};
  const std::array desired = {make_token(first_dispatch, 7, 3), make_token(first_dispatch, 7, 5)};
  ASSERT_NE(consan_moi_inline_acquired_epoch_token_slot_index(19, 7, 3, table.size()),
            consan_moi_inline_acquired_epoch_token_slot_index(19, 7, 5, table.size()));

  const auto pristine = table;
  constexpr std::array claim_loss = {true, false};
  const auto failed = consan_moi_inline_publish_acquired_token_transaction(
      table, desired, std::span<const bool>(claim_loss));
  EXPECT_TRUE(failed.claim_failed);
  EXPECT_FALSE(failed.committed);
  EXPECT_EQ(table, pristine);

  const auto committed = consan_moi_inline_publish_acquired_token_transaction(table, desired);
  ASSERT_TRUE(committed.committed);
  const auto after_first_dispatch = table;
  const std::array repeated_dispatch = {make_token(second_dispatch, 7, 3),
                                        make_token(second_dispatch, 7, 5)};
  const auto rejected =
      consan_moi_inline_publish_acquired_token_transaction(table, repeated_dispatch);
  EXPECT_TRUE(rejected.collision);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(table, after_first_dispatch)
      << "a reused direct-map slot may not authorize a different hardware dispatch";
}

TEST(ConSanMoiAdversarial, TransitiveImportUsesImmutableBeforeReleaseAncestry) {
  constexpr uint64_t dispatch = 0x2000000000000001ull;
  constexpr ConSanMoiInlineVersionedReleaseIdentity producer_identity{dispatch, 0x4000, 19};
  constexpr ConSanMoiInlineStableReleaseSnapshot producer_release{
      2, 2, producer_identity, /*releaser_owner_id=*/3, /*epoch_plus_one=*/12, {}};
  const auto middle_import =
      consan_moi_inline_plan_causal_import(producer_release, producer_identity, 7);
  ASSERT_TRUE(middle_import.authoritative());

  std::array middle_tokens = {make_token_view(dispatch, /*consumer=*/7, /*producer=*/3,
                                              middle_import.entries[0].producer_epoch_plus_one)};
  const auto before_release =
      consan_moi_inline_capture_causal_snapshot(middle_tokens, dispatch, 19, 7);
  middle_tokens[0].producer_epoch_plus_one = 22;
  const auto after_release =
      consan_moi_inline_capture_causal_snapshot(middle_tokens, dispatch, 19, 7);
  ASSERT_EQ(before_release.entries[0].ancestor_epoch_plus_one, 12u);
  ASSERT_EQ(after_release.entries[0].ancestor_epoch_plus_one, 22u);

  constexpr ConSanMoiInlineVersionedReleaseIdentity middle_identity{dispatch, 0x5000, 19};
  ConSanMoiInlineStableReleaseSnapshot middle_release{
      4, 4, middle_identity, /*releaser_owner_id=*/7, /*epoch_plus_one=*/30, before_release};
  const auto consumer_import =
      consan_moi_inline_plan_causal_import(middle_release, middle_identity, 9);
  ASSERT_TRUE(consumer_import.authoritative());
  ASSERT_EQ(consumer_import.entry_count, 2u);
  EXPECT_EQ(consumer_import.entries[0].producer_owner_id, 7u);
  EXPECT_EQ(consumer_import.entries[0].producer_epoch_plus_one, 30u);
  EXPECT_EQ(consumer_import.entries[1].producer_owner_id, 3u);
  EXPECT_EQ(consumer_import.entries[1].producer_epoch_plus_one, 12u);

  middle_release.version_before = 5;
  middle_release.version_after = 5;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(middle_release, middle_identity, 9).status,
            ConSanMoiInlineCausalImportStatus::UnstableRelease);
  middle_release.version_before = 4;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(middle_release, middle_identity, 9).status,
            ConSanMoiInlineCausalImportStatus::UnstableRelease);
}

TEST(ConSanMoiAdversarial, SnapshotOverflowPublicationDriftAndDuplicateFailClosed) {
  constexpr uint64_t dispatch = 0x3000000000000001ull;
  std::array<ConSanMoiInlineCausalTokenView, 5> fan_in{};
  for (uint32_t i = 0; i < fan_in.size(); ++i)
    fan_in[i] = make_token_view(dispatch, 7, i + 1u, /*epoch_plus_one=*/1023);
  const auto overflow = consan_moi_inline_capture_causal_snapshot(fan_in, dispatch, 19, 7);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(overflow, 7),
            ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow);

  std::array unstable = {fan_in[0]};
  unstable[0].version_after = 4;
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(
                consan_moi_inline_capture_causal_snapshot(unstable, dispatch, 19, 7), 7),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  unstable[0] = fan_in[0];
  unstable[0].version_before = unstable[0].version_after = 3;
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(
                consan_moi_inline_capture_causal_snapshot(unstable, dispatch, 19, 7), 7),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  const std::array duplicate = {fan_in[0], fan_in[0]};
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(
                consan_moi_inline_capture_causal_snapshot(duplicate, dispatch, 19, 7), 7),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  std::array<ConSanMoiInlineAcquiredEpochTokenSlot, 64> table{};
  const std::array duplicate_destination = {make_token(dispatch, 7, 3), make_token(dispatch, 7, 3)};
  const auto transaction =
      consan_moi_inline_publish_acquired_token_transaction(table, duplicate_destination);
  EXPECT_TRUE(transaction.duplicate_destination);
  EXPECT_FALSE(transaction.committed);
  EXPECT_EQ(table, decltype(table){});
}

TEST(ConSanMoiAdversarial, DispatchIdSgprCheckpointRestoresEveryShiftedGuestRegister) {
  for (const auto [user_count, system_count, prefix] :
       std::array<std::array<uint16_t, 3>, 4>{{{8, 4, 0}, {12, 2, 4}, {14, 0, 10}, {10, 6, 8}}}) {
    SCOPED_TRACE(user_count);
    const auto plan = consan_moi_plan_dispatch_id_preload(user_count, system_count, prefix,
                                                          /*dispatch_id_already_enabled=*/false);
    ASSERT_TRUE(plan.supported());
    ASSERT_TRUE(plan.descriptor_change_required());
    EXPECT_EQ(plan.dispatch_id_sgpr, prefix);
    for (uint16_t destination = 0; destination < plan.required_sgpr_count - 2u; ++destination) {
      const auto source = consan_moi_dispatch_id_restore_source(plan, destination);
      if (destination < prefix) {
        EXPECT_FALSE(source.has_value());
      } else {
        ASSERT_TRUE(source.has_value());
        EXPECT_EQ(*source, destination + 2u);
      }
    }
  }
  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(15, 0, 4, false).support,
            ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit);
  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(14, 4, 4, false, /*sgpr_limit=*/19).support,
            ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit);
}

TEST(ConSanMoiAdversarial, AcquireReleaseAndCasOrdersNeverAuthorizePartialPublication) {
  using Event = ConSanMoiInlineReleaseTransactionEvent;
  constexpr std::array acquire_release = {
      Event::PriorSnapshot, Event::Reserve,        Event::GuestAtomic, Event::AcquireImport,
      Event::Metadata,      Event::CausalSnapshot, Event::CommitReady};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      acquire_release, true, /*dynamic_acquire_semantics=*/true,
      /*release_outcome=*/true, /*outcome_dependent_release=*/false));

  constexpr std::array snapshot_before_import = {
      Event::PriorSnapshot, Event::Reserve,  Event::GuestAtomic, Event::CausalSnapshot,
      Event::AcquireImport, Event::Metadata, Event::CommitReady};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(snapshot_before_import, true, true,
                                                              true, false));

  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(acquire_release, true, true, true,
                                                             /*outcome_dependent_release=*/true));
  constexpr std::array failed_acquire_cas = {Event::PriorSnapshot, Event::Reserve,
                                             Event::GuestAtomic, Event::AcquireImport,
                                             Event::RestorePrior};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(failed_acquire_cas, true, true,
                                                             /*release_outcome=*/false,
                                                             /*outcome_dependent_release=*/true));

  constexpr std::array claim_loss = {Event::PriorSnapshot, Event::PoisonCoverage,
                                     Event::GuestAtomic};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      claim_loss, /*claim_succeeded=*/false, /*dynamic_acquire_semantics=*/true,
      /*release_outcome=*/true, /*outcome_dependent_release=*/false));
  constexpr std::array claim_loss_with_commit = {Event::PriorSnapshot, Event::PoisonCoverage,
                                                 Event::GuestAtomic, Event::CommitReady};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(claim_loss_with_commit, false, true,
                                                              true, false));
}

} // namespace
} // namespace rocjitsu
